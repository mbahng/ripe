"""Loader behaviour for canadian_invertebrates; real data, skipped when absent."""
import unittest
from pathlib import Path

import numpy as np

from src.dataset import init_dataset
from src.dataset.canadian import canadian_invertebrates

ROOT = Path("data/canadian_invertebrates")
HAS_DATA = (ROOT / "canadian_paired_first_image.tsv").is_file() and (ROOT / "canadian_images").is_dir()
BASE = {"root": "data", "image_size": 32, "seed": 0}

# 205500 rows, less 1 absent image, 420 '#NAME?' sequences and 77825 sequences
# carrying a gap or other non-ACGTN character, then species with >=10 samples.
N_SPECIES_ROWS = 58_340
N_SPECIES_CLASSES = 1_959


def pairs(frame, columns):
    return set(map(tuple, frame[columns].itertuples(index=False, name=None)))


@unittest.skipUnless(HAS_DATA, "data/canadian_invertebrates not present")
class Loader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = canadian_invertebrates({**BASE, "rank": "species", "min_samples": 10})

    def test_integrity_exclusions_applied(self):
        total = sum(len(self.ds[s]) for s in ("train", "val", "test"))
        self.assertEqual(total, N_SPECIES_ROWS)
        self.assertEqual(len(self.ds["train"].classes), N_SPECIES_CLASSES)
        for split in ("train", "push", "val", "test"):
            gen = self.ds[split].frame["gen"]
            self.assertFalse(gen.str.contains("#NAME", regex=False).any())
            self.assertFalse(gen.str.contains(r"[^ACGTN]", regex=True).any())
            # Gap-only records previously stripped down to an empty string.
            self.assertGreater(gen.str.len().min(), 0)

    def test_sample_contract_matches_bioscan(self):
        item = self.ds["train"][0]
        self.assertEqual(set(item), {"img", "gen", "label"})
        self.assertEqual(tuple(item["img"].shape), (3, 32, 32))
        self.assertIsInstance(item["gen"], str)
        self.assertIsInstance(item["label"], int)
        self.assertLess(float(self.ds["train"][0]["img"].min()), 0.0)   # normalised
        self.assertGreaterEqual(float(self.ds["push"][0]["img"].min()), 0.0)  # raw

    def test_no_specimen_or_label_group_crosses_a_split(self):
        frames = {s: self.ds[s].frame for s in ("train", "val", "test")}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            self.assertEqual(set(frames[a].processid) & set(frames[b].processid), set())
            # The unit is (component, label): a barcode or photo never recurs
            # across partitions carrying the same label.
            self.assertEqual(pairs(frames[a], ["gen", "label"]) & pairs(frames[b], ["gen", "label"]), set())
            self.assertEqual(pairs(frames[a], ["_digest", "label"]) & pairs(frames[b], ["_digest", "label"]), set())

    def test_every_class_reaches_training_and_most_reach_eval(self):
        frames = {s: self.ds[s].frame for s in ("train", "val", "test")}
        self.assertEqual(set(frames["train"].label), set(self.ds["train"].classes))
        for split in ("val", "test"):
            self.assertGreater(frames[split].label.nunique(), 0.95 * N_SPECIES_CLASSES)

    def test_split_ratios_are_approximately_honoured(self):
        total = sum(len(self.ds[s]) for s in ("train", "val", "test"))
        got = [len(self.ds[s]) / total for s in ("train", "val", "test")]
        for actual, want in zip(got, (0.6, 0.2, 0.2)):
            self.assertLess(abs(actual - want), 0.03)  # whole-group packing

    def test_ungrouped_mode_leaks_and_proves_grouping_matters(self):
        loose = canadian_invertebrates({**BASE, "rank": "species", "group_by": []})
        leaked = pairs(loose["train"].frame, ["gen", "label"]) & pairs(loose["test"].frame, ["gen", "label"])
        self.assertGreater(len(leaked), 1000)

    def test_determinism_and_seed_sensitivity(self):
        again = canadian_invertebrates({**BASE, "rank": "species", "min_samples": 10})
        self.assertTrue(self.ds["train"].frame.equals(again["train"].frame))
        other = canadian_invertebrates({**BASE, "rank": "species", "min_samples": 10, "seed": 1})
        self.assertFalse(self.ds["train"].frame.equals(other["train"].frame))

    def test_oversample_balances_train_only(self):
        balanced = canadian_invertebrates({**BASE, "rank": "species", "oversample": 1.0})
        counts = np.bincount(balanced["train"].targets, minlength=N_SPECIES_CLASSES)
        self.assertLessEqual(counts.max() - counts.min(), 1)
        self.assertEqual(len(balanced["push"]), len(self.ds["train"]))
        for split in ("val", "test"):
            self.assertEqual(len(balanced[split]), len(self.ds[split]))

    def test_rejects_bad_options(self):
        for bad in ({"rank": "tribe"}, {"min_samples": 0}, {"min_samples": True},
                    {"group_by": ["bogus"]}, {"split": [0, 1, 1]}, {"min_sequence_length": 0},
                    {"allowed_bases": "ACGTZ"}, {"allowed_bases": 5}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                canadian_invertebrates({**BASE, "rank": "species", **bad})

    def test_allowed_bases_can_be_widened_to_keep_gapped_reads(self):
        kept = canadian_invertebrates({**BASE, "rank": "species", "allowed_bases": "ACGTN-"})
        total = sum(len(kept[s]) for s in ("train", "val", "test"))
        self.assertGreater(total, N_SPECIES_ROWS)
        self.assertTrue(kept["train"].frame["gen"].str.contains("-", regex=False).any())

    def test_min_sequence_length_filters(self):
        strict = canadian_invertebrates({**BASE, "rank": "species", "min_sequence_length": 600})
        self.assertGreaterEqual(strict["train"].frame["gen"].str.len().min(), 600)
        self.assertLess(sum(len(strict[s]) for s in ("train", "val", "test")), N_SPECIES_ROWS)

    def test_registered_in_init_dataset(self):
        got = init_dataset({**BASE, "name": "canadian_invertebrates", "rank": "order",
                            "train": {"subsample": 0.001}})
        self.assertEqual(set(got), {"train", "push", "val", "test"})
        self.assertGreater(len(got["train"]), 0)


if __name__ == "__main__":
    unittest.main()
