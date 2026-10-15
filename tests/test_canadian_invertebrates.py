"""Integrity checks for the locally-built Canadian Invertebrates image+DNA set.

Skipped when data/canadian_invertebrates is absent. The dataset is assembled by
filtering a BOLD TSV dump to rows having both an image URL and nucleotides
(filter_canadian.py), then fetching one image per specimen with aria2.

Set CANINV_FULL=1 to decode all ~205k images instead of a 3k sample.
"""
import hashlib
import os
import re
import unittest
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path("data/canadian_invertebrates")
TSV = ROOT / "canadian_paired_first_image.tsv"
IMAGES = ROOT / "canadian_images"
ARIA = ROOT / "canadian_images.aria2"
HAS_DATA = TSV.is_file() and IMAGES.is_dir() and ARIA.is_file()

N_ROWS = 205_500
# BIOUG07054-E02 (processid HPPPA525-13) 404s upstream on BOLD; everything else
# is retrievable once '#' in the filename is percent-encoded as %23.
KNOWN_ABSENT = {"HPPPA525-13"}
# 420 rows carry the literal Excel error string '#NAME?' instead of a sequence.
# The corruption is upstream: canadian_invertebrates.tsv has 10,734 such lines.
N_EXCEL_CORRUPT = 420
IUPAC = set("ACGTURYSWKMBDHVN-")


def load():
    return pd.read_csv(TSV, sep="\t", dtype=str, keep_default_na=False, encoding="latin1")


def aria_pairs():
    """Return {out_name: url} from the aria2 input list."""
    pairs, url = {}, None
    with ARIA.open() as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("http"):
                url = s
            elif s.startswith("out="):
                pairs[s[4:]] = url
    return pairs


@unittest.skipUnless(HAS_DATA, "data/canadian_invertebrates not present")
class Metadata(unittest.TestCase):
    def test_schema_and_row_count(self):
        f = load()
        self.assertEqual(len(f), N_ROWS)
        self.assertEqual(list(f.columns)[:3], ["processid", "sampleid", "bin_uri"])
        self.assertIn("nucleotides", f.columns)
        self.assertIn("image_urls", f.columns)

    def test_keys_unique_and_populated(self):
        f = load()
        self.assertEqual(f.processid.duplicated().sum(), 0)
        self.assertEqual(f.sampleid.duplicated().sum(), 0)
        for col in ("processid", "sampleid", "image_urls", "nucleotides"):
            self.assertEqual((f[col].str.strip() == "").sum(), 0, col)

    def test_single_marker_licence_and_one_image_per_row(self):
        f = load()
        self.assertEqual(set(f.markercode), {"COI-5P"})
        self.assertEqual(set(f.copyright_licenses), {"CC-0"})
        self.assertTrue((f.image_urls.str.count(r"\|") == 0).all())

    def test_excel_corruption_is_bounded_and_confined_to_nucleotides(self):
        f = load()
        self.assertEqual((f.nucleotides.str.strip() == "#NAME?").sum(), N_EXCEL_CORRUPT)
        for col in f.columns:
            if col != "nucleotides":
                self.assertEqual(f[col].str.contains("#NAME", regex=False).sum(), 0, col)

    def test_sequences_are_iupac_apart_from_known_corruption(self):
        f = load()
        seq = f.nucleotides.str.upper().str.strip()
        clean = seq[seq != "#NAME?"]
        alphabet = set()
        for s in clean:
            alphabet |= set(s)
        self.assertEqual(alphabet - IUPAC, set())
        self.assertGreaterEqual(clean.str.len().min(), 200)


@unittest.skipUnless(HAS_DATA, "data/canadian_invertebrates not present")
class Images(unittest.TestCase):
    def test_download_list_agrees_with_metadata(self):
        f = load()
        pairs = aria_pairs()
        self.assertEqual({n[:-4] for n in pairs}, set(f.processid))
        by_pid = dict(zip(f.processid, f.image_urls))
        self.assertEqual([n for n in pairs if pairs[n] != by_pid[n[:-4]].strip()], [])

    def test_every_specimen_has_an_image(self):
        f = load()
        present = {os.path.splitext(n)[0] for n in os.listdir(IMAGES)}
        absent = set(f.processid) - present
        self.assertEqual(absent, KNOWN_ABSENT,
                         f"unexpected gaps: {sorted(absent - KNOWN_ABSENT)[:10]}")

    def test_no_empty_truncated_or_non_jpeg_files(self):
        bad_magic, truncated, tiny = [], [], []
        for entry in os.scandir(IMAGES):
            size = entry.stat().st_size
            if size < 2048:
                tiny.append(entry.name)
                continue
            with open(entry.path, "rb") as fh:
                if fh.read(3) != b"\xff\xd8\xff":
                    bad_magic.append(entry.name)
                    continue
                fh.seek(size - 2)
                if fh.read(2) != b"\xff\xd9":
                    truncated.append(entry.name)
        self.assertEqual(tiny, [])
        self.assertEqual(bad_magic, [])
        self.assertEqual(truncated, [])

    def test_images_decode(self):
        files = [e.path for e in os.scandir(IMAGES)]
        if not os.environ.get("CANINV_FULL"):
            import random
            random.seed(0)
            files = random.sample(files, 3000)
        fails = []
        for p in files:
            try:
                with Image.open(p) as im:
                    im.load()
                    if im.mode != "RGB":
                        fails.append((os.path.basename(p), im.mode))
            except Exception as exc:  # noqa: BLE001 - reporting, not handling
                fails.append((os.path.basename(p), repr(exc)[:80]))
        self.assertEqual(fails, [])

    def test_url_filename_matches_sampleid_where_checkable(self):
        """BOLD image filenames often embed the sampleid; catch gross misassignment."""
        f = load()
        stem_re = re.compile(r"/pics/[^/]+/([A-Za-z0-9\-]+)\+\d+\.(?:jpg|JPG|jpeg)$")

        plate_well = re.compile(r"^BIOUG0*(\d+)-([A-H])0*(\d+)$")

        def norm(s):
            m = plate_well.match(s.upper())
            return f"BIOUG{int(m.group(1))}-{m.group(2)}{int(m.group(3))}" if m else None

        checked = agree = 0
        for sid, url in zip(f.sampleid, f.image_urls):
            m = stem_re.search(url.strip())
            if not m:
                continue
            # Compare only when both sides parse as plate-well; URL stems carrying
            # view suffixes (-E02A) or odd punctuation are not misassignments.
            a, b = norm(m.group(1)), norm(sid)
            if a is None or b is None:
                continue
            checked += 1
            agree += a == b
        self.assertGreater(checked, 50_000)
        # 739 of 82,476 disagree by one plate or well - imaging-plate position
        # artefacts, not wrong specimens. BOLD filenames are not authoritative.
        self.assertGreater(agree / checked, 0.99)


@unittest.skipUnless(HAS_DATA, "data/canadian_invertebrates not present")
class Leakage(unittest.TestCase):
    def test_no_specimen_overlap_with_bioscan(self):
        caninv = set(load().processid)
        for path, sep in ((Path("data/bioscan_1m/BIOSCAN_Insect_Dataset_metadata.tsv"), "\t"),
                          (Path("data/bioscan_5m/bioscan5m/metadata/csv/"
                                "BIOSCAN_5M_Insect_Dataset_metadata.csv"), ",")):
            if not path.is_file():
                continue
            other = set()
            for ch in pd.read_csv(path, sep=sep, usecols=["processid"], dtype=str,
                                  keep_default_na=False, chunksize=500_000):
                other.update(ch.processid)
            self.assertEqual(caninv & other, set(), path.name)

    def test_duplicate_image_content_is_bounded(self):
        """193 photos are reused across 686 specimens - group them when splitting."""
        seen = {}
        for entry in os.scandir(IMAGES):
            with open(entry.path, "rb") as fh:
                digest = hashlib.blake2b(fh.read(), digest_size=16).hexdigest()
            seen.setdefault(digest, []).append(entry.name)
        shared = {k: v for k, v in seen.items() if len(v) > 1}
        involved = sum(len(v) for v in shared.values())
        self.assertLess(involved / len(seen), 0.01,
                        f"{involved} files share {len(shared)} hashes")


if __name__ == "__main__":
    unittest.main()
