"""Small local fixtures; no dataset downloads required."""
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
from PIL import Image

from src.dataset import init_dataloader
from src.dataset.bioscan import bioscan1m, bioscan5m, _balance_training, _stratified_split


class BIOSCANTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = Image.new("RGB", (32, 24), (100, 150, 200))

    def make_1m(self):
        root = self.root / "bioscan_1m"
        root.mkdir()
        rows = [("a.jpg", "Diptera", "train"), ("b.jpg", "Coleoptera", "train"),
                ("c.jpg", "Diptera", "validation"), ("d.jpg", "Coleoptera", "test"),
                ("e.jpg", "not_classified", "train"), ("f.jpg", "Novel", "test")]
        frame = pd.DataFrame(rows, columns=["image_file", "order", "large_insect_order"])
        frame["nucraw"] = ["ACGT", "ACG", "NNACGT", "TT", "", "AC"]
        frame["species"] = ["Species A", "Species B", "Species A", "Species B", "  ", "Not_Classified"]
        frame.to_csv(
            root / "BIOSCAN_Insect_Dataset_metadata.tsv", sep="\t", index=False)
        buffer = io.BytesIO()
        self.image.save(buffer, format="JPEG")
        with h5py.File(root / "cropped_256.hdf5", "w") as f:
            group = f.create_group("bioscan_dataset")
            for name, _, _ in rows:
                group[name] = np.frombuffer(buffer.getvalue(), dtype=np.uint8)

    def test_hdf5_local_labels_and_loader_settings(self):
        self.make_1m()
        cfg = {"name": "bioscan_1m", "root": str(self.root), "image_size": 16,
               "train": {"batch_size": 2}, "push": {"batch_size": 1}}
        with patch("src.dataset.bioscan._curl", side_effect=AssertionError("network")):
            with self.assertWarnsRegex(UserWarning, "absent from training"):
                loaders = init_dataloader(cfg)
        ds = loaders["train"].dataset
        self.addCleanup(ds.close)
        self.assertEqual(ds.classes, ["Coleoptera", "Diptera"])
        self.assertEqual(len(ds), 2)
        self.assertEqual(len(loaders["test"].dataset), 1)
        batch = next(iter(loaders["train"]))
        self.assertEqual(tuple(batch["img"].shape), (2, 3, 16, 16))
        self.assertEqual(set(batch["label"].tolist()), {0, 1})
        self.assertEqual(set(batch), {"img", "gen", "label"})
        self.assertEqual(set(batch["gen"]), {"ACGT", "ACG"})
        self.assertEqual(loaders["push"].batch_size, 1)
        self.assertIsNone(ds.__getstate__()["_h5"])

    def test_5m_paths_and_published_splits_without_archives(self):
        root = self.root / "bioscan_5m/bioscan5m"
        rows = []
        for split in ("train", "val", "test", "pretrain"):
            directory = root / "images/cropped_256" / split / "ab"
            directory.mkdir(parents=True)
            self.image.save(directory / f"{split}.jpg")
            rows.append((split, "ab", split, "Diptera", "ACGTN", "Species A"))
        metadata = root / "metadata/csv"
        metadata.mkdir(parents=True)
        pd.DataFrame(rows, columns=["processid", "chunk", "split", "order", "dna_barcode", "species"]).to_csv(
            metadata / "BIOSCAN_5M_Insect_Dataset_metadata.csv", index=False)
        cfg = {"root": str(self.root), "normalize": False, "rank": "species", "min_species_samples": 1, "split_strategy": "published"}
        with patch("src.dataset.bioscan._curl", side_effect=AssertionError("network")):
            datasets = bioscan5m(cfg)
            self.assertEqual(len(datasets["train"]), 1)
            self.assertEqual(tuple(datasets["test"][0]["img"].shape), (3, 224, 224))
            balanced = bioscan5m({**cfg, "oversample": 3.0})
            self.assertEqual(len(balanced["train"]), 3)
            self.assertEqual(len(balanced["push"]), 1)
            self.assertEqual(balanced["train"][2]["gen"], "ACGTN")
            stratified = bioscan5m({**cfg, "split_strategy": "stratified"})
            self.assertEqual(sum(len(stratified[s]) for s in ("train", "val", "test")), 4)
            for split in ("train", "val", "test"):
                for sample in stratified[split]:
                    self.assertEqual(sample["gen"], "ACGTN")
                    self.assertEqual(tuple(sample["img"].shape), (3, 224, 224))
            cfg["split_names"] = {"train": ["train", "pretrain"], "val": "val", "test": "test"}
            self.assertEqual(len(bioscan5m(cfg)["train"]), 2)
        cfg["split_names"]["val"] = "train"
        with self.assertRaisesRegex(ValueError, "overlap"):
            bioscan5m(cfg)

    def test_species_rank_filters_missing_values_before_mapping(self):
        self.make_1m()
        datasets = bioscan1m({"root": str(self.root), "rank": "species", "min_species_samples": 1, "split_strategy": "published"})
        self.assertEqual(set(datasets), {"train", "push", "val", "test"})
        for split, ds in datasets.items():
            self.addCleanup(ds.close)
            self.assertEqual(ds.classes, ["Species A", "Species B"])
            self.assertEqual(set(ds[0]), {"img", "gen", "label"})
        self.assertEqual(len(datasets["train"]), 2)
        self.assertEqual(len(datasets["test"]), 1)
        self.assertEqual(datasets["train"][0]["gen"], "ACGT")

    def test_missing_data_invokes_downloader(self):
        with patch("src.dataset.bioscan._download", side_effect=RuntimeError("download requested")):
            with self.assertRaisesRegex(RuntimeError, "download requested"):
                bioscan1m({"root": str(self.root)})
        with patch("src.dataset.bioscan._download", side_effect=AssertionError("network")):
            with self.assertRaises(FileNotFoundError):
                bioscan1m({"root": str(self.root), "download": False})

    def test_balancing_size_counts_randomness_and_pairing(self):
        frame = pd.DataFrame({"label": ["rare"] * 2 + ["common"] * 18,
                              "image_file": list(range(20)),
                              "gen": [f"dna-{i}" for i in range(20)]})
        for factor in (0.5, 1.0, 1.25, 2.0):
            sampled = _balance_training(frame, factor, seed=17)
            self.assertEqual(len(sampled), round(factor * len(frame)))
            counts = sampled["label"].value_counts()
            self.assertLessEqual(counts.max() - counts.min(), 1)
            self.assertEqual(set(sampled.loc[sampled.label == "rare", "image_file"]), {0, 1})
            if factor <= 1.25:
                common = sampled.loc[sampled.label == "common", "image_file"]
                self.assertEqual(len(common), common.nunique())
            for row in sampled.itertuples():
                self.assertEqual(row.gen, f"dna-{row.image_file}")
            pd.testing.assert_frame_equal(sampled, _balance_training(frame, factor, 17))
        self.assertFalse(_balance_training(frame, 1.0, 17).equals(_balance_training(frame, 1.0, 18)))
        for invalid in (0, -1, float("nan"), float("inf"), True, "2", 0.01):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _balance_training(frame, invalid, 0)

    def test_balancing_only_changes_training(self):
        self.make_1m()
        cfg = {"root": str(self.root), "rank": "species", "min_species_samples": 1, "split_strategy": "published"}
        original = bioscan1m(cfg)
        balanced = bioscan1m({**cfg, "oversample": 3.0, "seed": 42})
        self.assertEqual(len(balanced["train"]), 3 * len(original["train"]))
        self.assertEqual(np.bincount(balanced["train"].targets).tolist(), [3, 3])
        for split in ("push", "val", "test"):
            pd.testing.assert_frame_equal(original[split].frame, balanced[split].frame)
        self.assertEqual(original["train"].class_to_idx, balanced["train"].class_to_idx)

    def test_stratified_coverage_disjointness_and_rare_species(self):
        frame = pd.DataFrame({"label": ["singleton"] + ["pair"] * 2 + ["triple"] * 3 + ["common"] * 100,
                              "sampleid": range(106), "split": ["original"] * 106})
        result = _stratified_split(frame, [0.6, 0.2, 0.2], 7)
        self.assertEqual(sorted(result.sampleid), list(range(106)))
        self.assertEqual(result.sampleid.nunique(), 106)
        train = result[result._partition == "train"]
        self.assertEqual(set(train.label), set(frame.label))
        self.assertEqual(result.loc[result.label == "singleton", "_partition"].tolist(), ["train"])
        self.assertEqual(set(result.loc[result.label == "triple", "_partition"]), {"train", "val", "test"})
        self.assertEqual(result.loc[result.label == "common", "_partition"].value_counts().to_dict(),
                         {"train": 60, "val": 20, "test": 20})
        self.assertTrue((result["split"] == "original").all())
        pd.testing.assert_frame_equal(result, _stratified_split(frame, [0.6, 0.2, 0.2], 7))
        self.assertFalse(result.equals(_stratified_split(frame, [0.6, 0.2, 0.2], 8)))
        for ratios in ([0, 1, 1], [-1, 1, 1], [1, 1], [float("nan"), 1, 1]):
            with self.assertRaises(ValueError):
                _stratified_split(frame, ratios, 0)

    def test_species_default_uses_all_labeled_samples_before_balancing(self):
        self.make_1m()
        path = self.root / "bioscan_1m/BIOSCAN_Insect_Dataset_metadata.tsv"
        frame = pd.read_csv(path, sep="\t", keep_default_na=False)
        frame.loc[5, "species"] = "Singleton"
        frame.loc[5, "large_insect_order"] = "no_split"
        frame.to_csv(path, sep="\t", index=False)
        cfg = {"root": str(self.root), "rank": "species", "min_species_samples": 1, "seed": 5}
        ds = bioscan1m(cfg)
        self.assertEqual(sum(len(ds[s]) for s in ("train", "val", "test")), 5)
        self.assertIn("Singleton", ds["train"].classes)
        self.assertIn("f.jpg", set(ds["train"].frame.image_file))
        balanced = bioscan1m({**cfg, "oversample": 2.0})
        train_ids = set(balanced["train"].frame.image_file)
        for split in ("val", "test"):
            self.assertFalse(train_ids & set(balanced[split].frame.image_file))
            pd.testing.assert_frame_equal(ds[split].frame, balanced[split].frame)

    def test_species_minimum_applies_before_split_and_balancing(self):
        self.make_1m()
        path = self.root / "bioscan_1m/BIOSCAN_Insect_Dataset_metadata.tsv"
        example = pd.read_csv(path, sep="\t", keep_default_na=False).iloc[0].to_dict()
        rows = []
        for species, count in (("Rare", 9), ("Boundary", 10), ("Common", 11), ("not_classified", 20)):
            for i in range(count):
                rows.append({**example, "species": species, "image_file": f"{species}-{i}.jpg"})
        pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
        cfg = {"root": str(self.root), "rank": "species", "split": [0.6, 0.2, 0.2]}
        datasets = bioscan1m(cfg)
        self.assertEqual(sum(len(datasets[s]) for s in ("train", "val", "test")), 21)
        for split in ("train", "val", "test"):
            self.assertEqual(set(datasets[split].frame.label), {"Boundary", "Common"})
        balanced = bioscan1m({**cfg, "oversample": 3.0})
        self.assertEqual(len(balanced["train"]), 3 * len(datasets["train"]))
        self.assertEqual(balanced["train"].classes, ["Boundary", "Common"])
        with self.assertRaisesRegex(ValueError, "No species"):
            bioscan1m({**cfg, "min_species_samples": 12})


if __name__ == "__main__":
    unittest.main()
