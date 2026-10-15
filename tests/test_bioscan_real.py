"""Invariants checked against the real archives in data/; skipped when absent."""
import unittest
import warnings
from pathlib import Path

import numpy as np

from src.dataset.bioscan import bioscan1m, bioscan5m

ROOT = Path("data")
HAS_1M = (ROOT / "bioscan_1m/BIOSCAN_Insect_Dataset_metadata.tsv").is_file() and \
         (ROOT / "bioscan_1m/cropped_256.hdf5").is_file()
HAS_5M = (ROOT / "bioscan_5m/bioscan5m/metadata/csv/BIOSCAN_5M_Insect_Dataset_metadata.csv").is_file() and \
         (ROOT / "bioscan_5m/bioscan5m/images/cropped_256/train").is_dir()

BASE = {"root": str(ROOT), "download": False, "image_size": 32}


def sample(ds, n=4):
    for i in np.linspace(0, len(ds) - 1, min(n, len(ds)), dtype=int):
        yield ds[int(i)]


@unittest.skipUnless(HAS_1M, "BIOSCAN-1M archives not present")
class BIOSCAN1MReal(unittest.TestCase):
    def test_published_order_matches_benchmark(self):
        ds = bioscan1m(BASE)
        self.addCleanup(lambda: [d.close() for d in ds.values()])
        self.assertEqual(len(ds["train"].classes), 16)
        # Benchmark train size is 789813; one row (BIOUG65943-E04.jpg) was deleted
        # from the local TSV because cropped_256.hdf5 has no image for it. A
        # re-download restores that row and makes this assertion fail again.
        self.assertEqual([len(ds[s]) for s in ("train", "val", "test")],
                         [789812, 112835, 225660])
        self.assertEqual(len(ds["push"]), len(ds["train"]))
        for split in ("train", "val", "test"):
            self.assertTrue((ds[split].targets < 16).all())

    def test_species_stratified_matches_config_num_classes(self):
        ds = bioscan1m({**BASE, "rank": "species", "min_species_samples": 10,
                        "split": [0.6, 0.2, 0.2], "seed": 0, "oversample": 1.0})
        self.addCleanup(lambda: [d.close() for d in ds.values()])
        # setup/cfg/resnet50_bioscan_img.yml hard-codes model.num_classes.
        self.assertEqual(len(ds["train"].classes), 1146)
        counts = np.bincount(ds["train"].targets, minlength=1146)
        self.assertLessEqual(counts.max() - counts.min(), 1)
        ids = {s: set(ds[s].frame.image_file) for s in ("train", "push", "val", "test")}
        self.assertFalse(ids["train"] & ids["val"])
        self.assertFalse(ids["train"] & ids["test"])
        self.assertFalse(ids["val"] & ids["test"])
        # push keeps the unbalanced training pool, so it is a superset of the
        # balanced train split but must still never leak into val/test.
        self.assertTrue(ids["train"] <= ids["push"])
        self.assertFalse(ids["push"] & (ids["val"] | ids["test"]))

    def test_transforms_and_payload(self):
        ds = bioscan1m(BASE)
        self.addCleanup(lambda: [d.close() for d in ds.values()])
        for item in sample(ds["push"]):
            self.assertEqual(tuple(item["img"].shape), (3, 32, 32))
            self.assertGreaterEqual(float(item["img"].min()), 0.0)
            self.assertLessEqual(float(item["img"].max()), 1.0)
            self.assertIsInstance(item["gen"], str)
            self.assertIsInstance(item["label"], int)
        self.assertLess(float(next(sample(ds["train"], 1))["img"].min()), 0.0)

    def test_every_training_image_exists_in_hdf5(self):
        """Any row without a stored image raises KeyError mid-epoch in __getitem__.

        The upstream TSV ships one such row (BIOUG65943-E04.jpg, in the default
        published ``rank: order`` train split); it was deleted from the local
        copy. This fails again if the TSV is re-downloaded.
        """
        import h5py
        ds = bioscan1m(BASE)
        self.addCleanup(lambda: [d.close() for d in ds.values()])
        with h5py.File(ROOT / "bioscan_1m/cropped_256.hdf5", "r") as f:
            keys = set(f["bioscan_dataset"].keys())
        for split in ("train", "push", "val", "test"):
            absent = set(ds[split].frame.image_file) - keys
            self.assertEqual(absent, set(), f"{split} references missing images: {sorted(absent)[:5]}")


@unittest.skipUnless(HAS_5M, "BIOSCAN-5M archives not present")
class BIOSCAN5MReal(unittest.TestCase):
    def test_published_order_matches_benchmark(self):
        ds = bioscan5m(BASE)
        self.assertEqual(len(ds["train"].classes), 42)
        self.assertEqual([len(ds[s]) for s in ("train", "val", "test")],
                         [289203, 14757, 39373])

    def test_image_paths_resolve_for_every_partition_layout(self):
        """val/test/*_unseen are flat, train/other_heldout use 1-char chunks, pretrain 2-char."""
        ds = bioscan5m({**BASE, "rank": "species", "split_strategy": "stratified", "seed": 0})
        frame = ds["train"].frame
        seen = set()
        for position, origin in enumerate(frame["split"]):
            if origin in seen:
                continue
            seen.add(origin)
            self.assertEqual(tuple(ds["train"][position]["img"].shape), (3, 32, 32))
        self.assertEqual(seen, {"train", "val", "test", "other_heldout",
                                "key_unseen", "val_unseen", "test_unseen"})

    def test_published_species_excludes_unseen_classes_with_warning(self):
        cfg = {**BASE, "rank": "species", "split_strategy": "published",
               "split_names": {"train": ["train"], "val": ["val", "val_unseen"],
                               "test": ["test", "test_unseen"]}}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds = bioscan5m(cfg)
        messages = [str(w.message) for w in caught]
        self.assertEqual(len(messages), 2, messages)
        self.assertTrue(all("absent from training" in m for m in messages))
        for split in ("val", "test"):
            self.assertTrue(ds[split].frame["label"].isin(ds["train"].classes).all())

    def test_unseen_only_evaluation_split_is_rejected(self):
        cfg = {**BASE, "rank": "species", "split_strategy": "published",
               "split_names": {"train": ["train"], "val": ["val_unseen"], "test": ["test_unseen"]}}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaisesRegex(ValueError, "No labeled BIOSCAN samples in val"):
                bioscan5m(cfg)


if __name__ == "__main__":
    unittest.main()
