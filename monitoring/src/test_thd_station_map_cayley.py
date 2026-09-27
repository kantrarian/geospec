"""Tests for monitoring/src/thd_station_map.py (cayley 2026-09-27).

    python -m unittest monitoring/src/test_thd_station_map_cayley.py      (from the repo root)
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thd_station_map as M  # noqa: E402

INPUTS = [M.REGIONS_SRC, M.BOUNDS_SRC, M.REPORT, M.META_DIR + "/sources.json"] + \
         [M.META_DIR + "/" + n for n in M.FDSN_QUERIES]


def copy_inputs(dst):
    for rel in INPUTS:
        (Path(dst) / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(M.REPO / rel, Path(dst) / rel)


class Geometry(unittest.TestCase):
    def test_one_degree_of_latitude(self):
        self.assertAlmostEqual(M.haversine_km(0, 0, 1, 0), 111.195, places=2)

    def test_inside_the_box_is_zero_and_outside_is_positive(self):
        box = {"lat": (35.0, 36.0), "lon": (-118.0, -117.0)}
        self.assertEqual(M.region_distances(35.5, -117.5, box)["box_km"], 0.0)
        out = M.region_distances(37.0, -117.5, box)
        self.assertAlmostEqual(out["box_km"], 111.2, places=1)
        self.assertAlmostEqual(out["centre_km"], 166.8, places=1)

    def test_antimeridian_refuses(self):
        with self.assertRaises(M.MapError):
            M.region_distances(0, 179.5, {"lat": (0, 1), "lon": (-179.5, -178.5)})


class Parsing(unittest.TestCase):
    TEXT = ("#Network | Station | Latitude | Longitude | Elevation | SiteName | StartTime | EndTime\n"
            "XX|OLD|10.0|20.0|5|Old site|1990-01-01T00:00:00|2000-01-01T00:00:00\n"
            "XX|OLD|11.0|21.0|5|New site|2000-01-02T00:00:00|\n")

    def test_the_epoch_in_force_on_the_day_is_chosen(self):
        epochs = M.parse_fdsn_text(self.TEXT, "t")["XX.OLD"]
        self.assertEqual(M._epoch_for(epochs, "2026-09-25")["site"], "New site")
        self.assertEqual(M._epoch_for(epochs, "1995-06-01")["site"], "Old site")
        self.assertIsNone(M._epoch_for(epochs, "1980-01-01"))

    def test_open_epochs_that_disagree_refuse(self):
        text = self.TEXT + "XX|OLD|12.0|22.0|5|Other|2001-01-01T00:00:00|\n"
        with self.assertRaises(M.MapError):
            M._epoch_for(M.parse_fdsn_text(text, "t")["XX.OLD"], "2026-09-25")

    def test_malformed_row_refuses(self):
        with self.assertRaises(M.MapError):
            M.parse_fdsn_text("XX|A|1|2\n", "t")

    def test_recorded_station_from_notes(self):
        rep = lambda notes: {"components": {"seismic_thd": {"notes": notes}}}
        self.assertEqual(M.recorded_station(rep("sta=IU.TUC, THD=0.40, z=0.57, n=91, rate=40Hz")),
                         ("IU.TUC", "LATEST_REPORT_NOTES_STA", 91))
        self.assertEqual(M.recorded_station(rep("Insufficient data from IV.CAFE")),
                         ("IV.CAFE", "LATEST_REPORT_NOTES_ATTEMPTED_NO_VALUE", None))
        self.assertEqual(M.recorded_station(rep("")), (None, "NOT_RECORDED", None))
        self.assertEqual(M.recorded_station(None), (None, "NOT_RECORDED", None))


class Build(unittest.TestCase):
    def test_the_committed_map_is_exactly_what_the_builder_derives(self):
        committed = json.loads((M.REPO / M.OUTPUT).read_text(encoding="utf-8"))
        self.assertEqual(committed, M.build())

    def test_shared_stations_are_derived_not_declared(self):
        rows = {r["region"]: r for r in M.build()["regions"]}
        tuc = sorted([r for r, row in rows.items() if row["recorded_station"] == "IU.TUC"])
        for region in tuc:
            self.assertEqual(sorted(rows[region]["shares_recorded_station_with"] + [region]), tuc)
        # a region with no measured station shares with nobody
        for row in rows.values():
            if row["recorded_basis"] != "LATEST_REPORT_NOTES_STA":
                self.assertEqual(row["shares_recorded_station_with"], [])

    def test_every_region_names_an_official_source(self):
        for row in M.build()["regions"]:
            self.assertIn(row["official_source"]["name"], M.AGENCIES, row["region"])

    def test_tampered_retained_metadata_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            copy_inputs(d)
            path = Path(d) / M.META_DIR / "ncedc.txt"
            path.write_bytes(path.read_bytes().replace(b"37.876221", b"37.876222"))
            with self.assertRaises(M.MapError):
                M.build(d)

    def test_a_region_without_bounds_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            copy_inputs(d)
            path = Path(d) / M.BOUNDS_SRC
            path.write_text(path.read_text(encoding="utf-8").replace("'kaikoura':", "'kaikoura_x':"), encoding="utf-8")
            with self.assertRaises(M.MapError):
                M.build(d)


if __name__ == "__main__":
    unittest.main()
