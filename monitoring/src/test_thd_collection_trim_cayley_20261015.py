"""Weekly THD recal collection trim (cayley 2026-10-15; codex 72777c8b: "a small explicit collection-list change from
configured stations plus approved calibration additions, retaining historical baselines and any declared research
consumers. Treat that as collection-cost maintenance, not qualification repair"; owner 2026-10-15: "start the
collection-list trim while waiting"):

  * BK.BKS, GE.ARPR and GE.CSS are COLLECTION_RETIRED: each is configured by no region, and none is fetched;
  * every station still collected is configured by a region or is an owner-approved CALIBRATION_ADDITION;
  * the collected list over the built-in records is exactly the expected literal (no module answer on both sides);
  * a retired station's newest recalibrated record is RETAINED in the new file as the loader parsed it, so the
    loader still selects it (not the 2026-01 built-in), and its attempt is recorded as NOT_COLLECTED; with no prior
    record nothing is fabricated;
  * a station both retired and added refuses; an explicit --stations entry is still fetched.

Offline; no provider I/O. Run: <py with obspy> -B -m unittest test_thd_collection_trim_cayley_20261015
"""
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibrate_thd_baselines as C      # noqa: E402
import run_thd_recal as R                # noqa: E402
import station_baselines as SB           # noqa: E402

RETIRED = ["BK.BKS", "GE.ARPR", "GE.CSS"]
# the built-in records with a real calibration window, minus the retired ones, then the additions in declared order
EXPECTED_OVER_BUILTINS = ["IU.TUC", "IU.COR", "IU.MAJO", "IU.ANTO", "IU.TATO", "IU.COLA", "MX.TLIG",
                          "IV.CAFE", "BK.BRK", "IU.SNZO"]


def _prior_file(directory, stamp, entries):
    path = Path(directory) / ("thd_baselines_%s.json" % stamp)
    path.write_text(json.dumps({e["station"]: e for e in entries}, indent=2), encoding="utf-8")
    return path


PRIOR = [
    dict(station="IU.TUC", mean_thd=0.35, std_thd=0.04, n_samples=88, calibration_period="2026-06-05 to 2026-09-03",
         calibration_date="2026-10-03", notes="Rolling recal 90d/30d (incident 2026-07-31)"),
    dict(station="BK.BKS", mean_thd=0.3305, std_thd=0.051, n_samples=84, calibration_period="2026-06-05 to 2026-09-03",
         calibration_date="2026-10-03", notes="Rolling recal 90d/30d (incident 2026-07-31)"),
    dict(station="GE.ARPR", mean_thd=0.97, std_thd=0.21, n_samples=80, calibration_period="2026-06-05 to 2026-09-03",
         notes="Rolling recal 90d/30d (incident 2026-07-31)"),          # pre-v3 row: date comes from the file name
]


class CollectionList(unittest.TestCase):
    def test_the_retired_stations_are_exactly_the_three_and_attributed(self):
        self.assertEqual(list(R.COLLECTION_RETIRED), RETIRED)
        self.assertIn("(643dd746)", R.COLLECTION_RETIRED["BK.BKS"])
        for key, why in R.COLLECTION_RETIRED.items():
            self.assertTrue(why.strip(), key)

    def test_no_region_configures_a_retired_station(self):
        import run_ensemble_daily as RD
        for key in RETIRED:
            self.assertNotIn(key, RD.THD_STATION_REGIONS, key + " is configured; it must not be retired")

    def test_over_the_builtin_records_the_list_is_exactly_the_expected_literal(self):
        with mock.patch.dict(R.STATION_BASELINES, SB._BUILTIN_BASELINES, clear=True):
            stations = R._calibratable_stations()
        self.assertEqual(stations, EXPECTED_OVER_BUILTINS)

    def test_every_collected_station_is_configured_or_an_owner_addition(self):
        import run_ensemble_daily as RD
        with mock.patch.dict(R.STATION_BASELINES, SB._BUILTIN_BASELINES, clear=True):
            stations = R._calibratable_stations()
        for key in stations:
            self.assertTrue(key in RD.THD_STATION_REGIONS or key in R.CALIBRATION_ADDITIONS, key)
        for key in RETIRED:
            self.assertNotIn(key, stations)

    def test_a_newest_file_carrying_a_retired_station_does_not_bring_it_back(self):
        recal = SB.StationBaseline(station="GE.CSS", mean_thd=0.21, std_thd=0.05, n_samples=80,
                                   calibration_period="2026-06-05 to 2026-09-03", calibration_date="2026-10-03")
        with mock.patch.dict(R.STATION_BASELINES, dict(SB._BUILTIN_BASELINES, **{"GE.CSS": recal}), clear=True):
            self.assertNotIn("GE.CSS", R._calibratable_stations())

    def test_a_station_both_retired_and_added_refuses(self):
        with mock.patch.dict(R.CALIBRATION_ADDITIONS, {"BK.BKS": "contradiction"}):
            with self.assertRaises(ValueError):
                R._calibratable_stations()


class Retention(unittest.TestCase):
    def _run(self, stations, prior_entries):
        fetched = []

        def calibrate(network, station, **kw):
            fetched.append("%s.%s" % (network, station))
            return {"mean_thd": 0.3, "std_thd": 0.05, "n_samples": 70, "calibration_period": "2026-06-19 to 2026-09-17"}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(R, "BASELINE_DIR", Path(tmp)), \
                mock.patch.object(C, "calibrate_station", side_effect=calibrate):
            if prior_entries:
                _prior_file(tmp, "20261003", prior_entries)
            path = Path(R.run_recal(stations, end_date=datetime(2026, 10, 17)))
            data = json.loads(path.read_text(encoding="utf-8"))
            loaded = {}
            with mock.patch.dict(SB.STATION_BASELINES, {}, clear=True):
                name = SB._load_newest_baseline_file(Path(tmp))
                loaded = dict(SB.STATION_BASELINES)
        return fetched, data, name, loaded

    def test_retired_records_are_retained_not_fetched_and_still_selected(self):
        fetched, data, name, loaded = self._run(["IU.TUC"], PRIOR)
        self.assertEqual(fetched, ["IU.TUC"], "no retired station is fetched")
        self.assertEqual(name, "thd_baselines_20261017.json")
        bks = data["BK.BKS"]
        for field in ("mean_thd", "std_thd", "n_samples", "calibration_period", "calibration_date", "notes"):
            self.assertEqual(bks[field], PRIOR[1][field], field)
        self.assertEqual(data["GE.ARPR"]["calibration_date"], "2026-10-03", "a pre-v3 row keeps its file-name date")
        self.assertEqual(data["GE.ARPR"]["mean_thd"], 0.97)
        self.assertNotIn("GE.CSS", data, "no prior record: nothing is fabricated")
        # the loader keeps the retained recal, not the 2026-01 built-in
        self.assertEqual((loaded["BK.BKS"].mean_thd, loaded["BK.BKS"].calibration_date), (0.3305, "2026-10-03"))
        self.assertNotEqual(loaded["BK.BKS"].calibration_period, SB._BUILTIN_BASELINES["BK.BKS"].calibration_period)
        attempts = {a["station"]: a for a in data["_recal_attempts"]}
        self.assertEqual((attempts["IU.TUC"]["outcome"], attempts["IU.TUC"]["disposition"]), ("VALUE", "RECALIBRATED"))
        for key in ("BK.BKS", "GE.ARPR"):
            a = attempts[key]
            self.assertEqual((a["outcome"], a["disposition"], a["retained_calibration_date"]),
                             ("NOT_COLLECTED", "PRIOR_RECORD_RETAINED", "2026-10-03"), key)
            self.assertEqual(a["reason"], R.COLLECTION_RETIRED[key])
        self.assertNotIn("GE.CSS", attempts)

    def test_an_explicit_stations_entry_is_still_fetched(self):
        fetched, data, _name, _loaded = self._run(["IU.TUC", "BK.BKS"], PRIOR)
        self.assertEqual(fetched, ["IU.TUC", "BK.BKS"])
        self.assertEqual(data["BK.BKS"]["calibration_date"], "2026-10-17")
        self.assertEqual([a["disposition"] for a in data["_recal_attempts"] if a["station"] == "BK.BKS"], ["RECALIBRATED"])

    def test_the_dry_run_names_what_is_not_collected(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertIsNone(R.run_recal(["IU.TUC"], end_date=datetime(2026, 10, 17), dry_run=True))
        text = buf.getvalue()
        self.assertIn("would recal IU.TUC", text)
        for key in RETIRED:
            self.assertIn("not collected %s" % key, text)


if __name__ == "__main__":
    unittest.main()
