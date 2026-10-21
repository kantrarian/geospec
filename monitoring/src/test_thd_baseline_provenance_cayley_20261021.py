"""thd-baseline-provenance-v1 + thd-baseline-station-check-v1 + thd-robust-estimator-v1 (cayley 2026-10-21; codex
b4f6c848 items 1-2, 05e0904b estimator acceptance). Needs obspy (1.5.1 pinned env).

The COMPOSED producer path, not the classifier alone: ensemble.compute_thd_risk measures a synthetic BK.BRK waveform
through the real fetch (obspy's FDSN Client replaced by the trimming client of test_thd_daily_measurement), looks the
baseline up through station_baselines from a temporary directory of dated files, runs the producer-boundary check,
eligibility and the score basis, and serializes the record component. Acceptance:

  * a normal BRK baseline qualifies; the issued block names the file and the sha256 of its bytes, the calibration
    date, window, n, center/scale and the DECLARED estimator, and the as-of rule never names a later recal;
  * a BKS baseline substituted for the measured BRK (the lookup itself replaced) is a typed STATION_MISMATCH: the raw
    value is retained, the baseline block is the no-baseline block, the score is withheld -- never a numeric zero
    score and never a z against the substitute;
  * a convention both sides state and that differs is a typed CONVENTION_MISMATCH; a baseline stamped with no
    convention is UNKNOWN, neither a mismatch nor a match, and is still used;
  * a declared-channel station's selector rule is unchanged (SELECTOR_MISMATCH);
  * with the eligibility rule inactive the record carries no block (byte-identical legacy output);
  * calibrate_station stores round(median, 6) / round(MAD_TO_SIGMA x MAD, 6) and declares exactly that, on values where
    median/scaled MAD and mean/SD differ, and run_thd_recal carries the declaration into the entry the loader reads.
Waveforms and moments are SYNTHETIC. Nothing touches a network, the runner tree or a credential.
"""
import hashlib
import json
import os
import statistics
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

from obspy import Stream

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import calibrate_thd_baselines as C  # noqa: E402
import ensemble as E  # noqa: E402
import run_thd_recal as R  # noqa: E402
import station_baselines as SB  # noqa: E402
import thd_daily_measurement as TDM  # noqa: E402
from test_thd_daily_measurement_cayley_20261008 import DAY, TrimmingClient, waveform  # noqa: E402

BRK = dict(station="BK.BRK", mean_thd=0.392402, std_thd=0.176116, n_samples=91,
           calibration_period="2026-03-28 to 2026-06-26", calibration_date="2026-07-26",
           calibration_convention=TDM.MEASUREMENT_VERSION, estimator=dict(C.ESTIMATOR),
           measurement=dict(identity="a" * 64))
BKS = dict(station="BK.BKS", mean_thd=0.364152, std_thd=0.135428, n_samples=91,
           calibration_period="2026-03-28 to 2026-06-26", calibration_date="2026-07-26")


class ComposedProducerPath(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="thd-provenance-")
        self.addCleanup(self.tmp.cleanup)
        self.bdir = Path(self.tmp.name)
        TrimmingClient.STREAM, TrimmingClient.asked = Stream([waveform("BK", "BRK", "00")]), []
        for patcher in (mock.patch("obspy.clients.fdsn.Client", TrimmingClient),
                        mock.patch.object(SB, "BASELINE_DIR", self.bdir), mock.patch.dict(SB.STATION_BASELINES)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def write(self, stamp, entries):
        path = self.bdir / ("thd_baselines_%s.json" % stamp)
        path.write_text(json.dumps({e["station"]: e for e in entries}, indent=2))
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def compute(self, active=True):
        SB._load_newest_baseline_file()
        result = E.GeoSpecEnsemble(region="norcal_hayward", eligibility_rule_active=active).compute_thd_risk(
            DAY, station_network="BK", station_code="BRK")
        return result, result.to_dict()

    def test_a_normal_brk_baseline_qualifies_and_names_its_bound_file(self):
        path, digest = self.write("20260726", [BRK, BKS])
        later, _ = self.write("20260803", [dict(BRK, mean_thd=0.9, calibration_date="2026-08-03")])
        result, d = self.compute()
        block = d["baseline_provenance"]
        self.assertEqual(block["check"]["code"], "MATCH")
        self.assertEqual((block["check"]["station"], block["check"]["selector"], block["check"]["convention"]),
                         ("EQUAL", "NOT_DECLARED", "EQUAL"))
        self.assertEqual(block["disposition"], "SCORED_AGAINST")
        self.assertEqual(block["source"], dict(kind="DATED_RECAL_FILE", file=path.name, sha256=digest))
        self.assertEqual((block["calibration_date"], block["window_label"], block["n_samples"]),
                         ("2026-07-26", "2026-03-28 to 2026-06-26", 91))
        self.assertEqual((block["center"], block["scale"]), (0.392402, 0.176116))
        self.assertEqual(block["estimator"], C.ESTIMATOR)
        self.assertEqual((block["baseline_station"], block["measured_station"]), ("BK.BRK", "BK.BRK"))
        self.assertEqual(block["measurement_identity"], "a" * 64)
        self.assertEqual((d["baseline"]["mean"], d["baseline"]["std"]), (0.392402, 0.176116))
        self.assertIsNotNone(d["z_score"])
        self.assertTrue(d["calibration"]["eligible_for_tiering"], d["calibration"]["reason"])
        self.assertNotEqual(block["source"]["file"], later.name, "the recal dated after the scored day is never named")

    def test_a_bks_baseline_substituted_for_brk_is_a_typed_mismatch(self):
        self.write("20260726", [BRK, BKS])
        real = SB.get_baseline
        substitute = lambda station_code, network, as_of=None: real("BKS", "BK", as_of=as_of)  # noqa: E731
        with mock.patch.object(SB, "get_baseline", side_effect=substitute):
            result, d = self.compute()
        block = d["baseline_provenance"]
        self.assertEqual((block["check"]["code"], block["check"]["station"]), ("STATION_MISMATCH", "DIFFERENT"))
        self.assertEqual((block["disposition"], block["baseline_station"]), ("REJECTED_BY_CHECK", "BK.BKS"))
        self.assertEqual(d["baseline"], dict(mean=0.0, std=0.0, n_samples=0, window="", quality="missing"))
        self.assertTrue(d["available"])
        self.assertGreater(d["raw_value"], 0.0, "the measured raw value is retained, never zeroed")
        self.assertIsNone(d["risk_score"])
        self.assertIsNone(d["z_score"])
        self.assertEqual(d["score_basis"], E.SCORE_BASIS_NOT_COMPUTED)
        self.assertFalse(d["calibration"]["eligible_for_tiering"])
        self.assertNotIn("BK.BKS", d["notes"])

    def test_a_stated_convention_must_match_and_an_unstated_one_stays_unknown(self):
        self.write("20260726", [dict(BRK, calibration_convention="thd-bound-station-daily-operator-v1")])
        result, d = self.compute()
        self.assertEqual((d["baseline_provenance"]["check"]["code"], d["baseline_provenance"]["check"]["convention"]),
                         ("CONVENTION_MISMATCH", "DIFFERENT"))
        self.assertIsNone(d["z_score"])
        legacy = {k: v for k, v in BRK.items() if k != "calibration_convention"}
        self.write("20260726", [legacy])
        SB.STATION_BASELINES.pop("BK.BRK", None)
        result, d = self.compute()
        check = d["baseline_provenance"]["check"]
        self.assertEqual((check["code"], check["convention"]), ("MATCH", "UNKNOWN"))
        self.assertEqual(d["baseline_provenance"]["disposition"], "SCORED_AGAINST")
        self.assertIsNotNone(d["z_score"])

    def test_inactive_rule_output_carries_no_block(self):
        self.write("20260726", [BRK])
        result, d = self.compute(active=False)
        self.assertNotIn("baseline_provenance", d)
        self.assertIsNone(result.baseline_provenance)


class TheCheckItself(unittest.TestCase):
    def baseline(self, **kw):
        return SB.StationBaseline(**dict(dict(station="IV.CAFE", mean_thd=0.12, std_thd=0.07, n_samples=91,
                                              calibration_period="x", selector="IV.CAFE..HHZ@100Hz",
                                              calibration_convention=TDM.MEASUREMENT_VERSION), **kw))

    def test_selector_rule_is_unchanged_and_no_baseline_is_typed(self):
        declared = dict(selector="IV.CAFE..HHZ@100Hz")
        conv = TDM.MEASUREMENT_VERSION
        self.assertEqual(E.thd_baseline_check("IV.CAFE", self.baseline(), declared, conv)["code"], "MATCH")
        self.assertEqual(E.thd_baseline_check("IV.CAFE", self.baseline(selector=None), declared, conv)["code"],
                         "SELECTOR_MISMATCH")
        self.assertEqual(E.thd_baseline_check("IV.CAFE", None, declared, conv)["code"], "NO_BASELINE")
        self.assertEqual(E.thd_baseline_check("IV.CAFE", self.baseline(), declared, None)["convention"], "UNKNOWN")

    def test_a_built_in_default_is_named_as_such(self):
        block = E.thd_baseline_provenance(SB._BUILTIN_BASELINES["BK.BKS"], E.thd_baseline_check(
            "BK.BKS", SB._BUILTIN_BASELINES["BK.BKS"], None, None), "SCORED_AGAINST", None)
        self.assertEqual(block["source"], dict(kind="BUILT_IN_DEFAULT", file=None, sha256=None))
        self.assertIsNone(block["estimator"])


class Estimator(unittest.TestCase):
    VALUES = [0.21, 0.25, 0.26, 0.28, 0.30, 0.31, 0.33, 0.95, 1.40, 0.27, 0.29, 0.32]

    def test_calibrate_station_stores_and_declares_median_and_scaled_mad(self):
        days = iter(self.VALUES)

        def record(network, station, date, channel="BHZ"):
            return dict(day=date.strftime("%Y-%m-%d"), thd=next(days, None), native_rate_hz=40.0, reason=None)

        with mock.patch.object(C, "compute_daily_record", side_effect=record):
            r = C.calibrate_station("BK", "BRK", days_back=len(self.VALUES) - 1, exclude_recent_days=30,
                                    end_date=datetime(2026, 6, 30))
        median = statistics.median(self.VALUES)
        mad = statistics.median(abs(v - median) for v in self.VALUES)
        self.assertEqual(r["n_samples"], len(self.VALUES))
        self.assertEqual(r["mean_thd"], round(median, 6))
        self.assertEqual(r["std_thd"], round(C.MAD_TO_SIGMA * mad, 6))
        self.assertEqual(r["estimator"], C.ESTIMATOR)
        self.assertEqual(C.ESTIMATOR["scale_multiplier"], C.MAD_TO_SIGMA)
        self.assertNotEqual(r["mean_thd"], r["mean_thd_classic"], "the fixture separates median from mean")
        self.assertNotEqual(r["std_thd"], r["std_thd_classic"], "the fixture separates scaled MAD from SD")

    def test_the_recal_entry_carries_the_declaration_the_loader_reads(self):
        def calibrate(network, station, **kw):
            return {"mean_thd": 0.39, "std_thd": 0.17, "n_samples": 91, "calibration_period": "2026-06-19 to 2026-09-17",
                    "estimator": dict(C.ESTIMATOR)}
        with tempfile.TemporaryDirectory(prefix="thd-recal-") as tmp:
            bdir = Path(tmp)
            with mock.patch.object(R, "BASELINE_DIR", bdir), mock.patch.object(C, "calibrate_station", side_effect=calibrate), \
                    mock.patch.object(R, "_prior_effective_entries", return_value={}):
                path = Path(R.run_recal(["BK.BRK"], end_date=datetime(2026, 10, 17)))
            entry = json.loads(path.read_text())["BK.BRK"]
            self.assertEqual(entry["estimator"], C.ESTIMATOR)
            held = SB._file_baselines(path)["BK.BRK"]
            self.assertEqual((held.estimator, held.source_file, held.source_sha256),
                             (C.ESTIMATOR, path.name, hashlib.sha256(path.read_bytes()).hexdigest()))


if __name__ == "__main__":
    unittest.main()
