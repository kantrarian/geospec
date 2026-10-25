"""thd-undefined-unavailable-v1 tests (cayley 2026-10-25, F1; owner "draft F1"). Needs numpy/scipy; the before/after
checks also need git (they read the base tree d251072e) and run the REAL run_ensemble_daily.run_region_assessment with
the REAL GeoSpecEnsemble, thd_daily_measurement.measure and SeismicTHDAnalyzer on SYNTHETIC waveforms (the fetch is
replaced, nothing touches a network).

F1: when the fundamental band power P1 is not positive, THD = sqrt(sum P_n / P1) is undefined; analyze_window returns
0.0 as a placeholder. The weekly recal already refused such a day (NONPOSITIVE_THD_OR_P1); the daily scorer published
and scored the 0.0 as a measured, very low THD. A dead channel delivering zeros, or a flat line, is exactly that state.
"""
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import thd_daily_measurement as TDM  # noqa: E402

REPO = HERE.parent.parent
BASE = "d251072e"   # public master before this change


def _base_tree(tmp):
    # the whole monitoring tree: the region path also reads tracked data beside src (e.g. data/calibration's registry)
    archive = subprocess.run(["git", "-C", str(REPO), "archive", BASE, "monitoring"], capture_output=True,
                             check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(tmp)], input=archive, check=True)
    return Path(tmp) / "monitoring/src"


class TheRecalRuleIsSharedAndUnchanged(unittest.TestCase):
    """calibrate_thd_baselines.compute_daily_record decides exactly as it did at d251072e, for every value shape."""

    def test_every_record_shape_gets_the_same_receipt_from_base_and_patched_recal(self):
        import calibrate_thd_baselines as patched
        with tempfile.TemporaryDirectory() as tmp:
            src = _base_tree(tmp)
            spec = importlib.util.spec_from_file_location("calibrate_thd_baselines_base_d251072e",
                                                          src / "calibrate_thd_baselines.py")
            base = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(base)
        thds = (None, 0.0, -0.0, 1e-300, 0.42, -1.0, float("nan"), float("inf"))
        p1s = ("absent", None, 0.0, -1.0, 1e-300, 3.0, float("nan"), float("inf"))
        compared = 0
        for thd in thds:
            for p1 in p1s:
                rec = {"thd": thd, "native_rate_hz": 1.0, "measurement_identity": "m", "reason": None}
                if p1 != "absent":
                    rec["p1"] = p1
                with self.subTest(thd=thd, p1=p1), mock.patch.object(TDM, "measure", lambda *a, **k: dict(rec)):
                    day = datetime(2026, 10, 22)
                    got, want = patched.compute_daily_record("IU", "COLA", day), base.compute_daily_record("IU", "COLA", day)
                    self.assertEqual(json.dumps(got, sort_keys=True, default=str), json.dumps(want, sort_keys=True, default=str))
                    unusable = TDM.unusable_value_reason(rec)
                    self.assertEqual(unusable is not None, want["reason"] == "NONPOSITIVE_THD_OR_P1")
                    compared += 1
        self.assertEqual(compared, 64)

    def test_the_rule_names_the_undefined_state(self):
        self.assertEqual(TDM.unusable_value_reason({"thd": 0.0, "p1": 0.0}), "NONPOSITIVE_THD_OR_P1")
        self.assertEqual(TDM.unusable_value_reason({"thd": float("nan"), "p1": 1.0}), "NONPOSITIVE_THD_OR_P1")
        self.assertEqual(TDM.unusable_value_reason({"thd": 0.4, "p1": float("nan")}), "NONPOSITIVE_THD_OR_P1")
        self.assertIsNone(TDM.unusable_value_reason({"thd": 0.4, "p1": 3.0}))
        self.assertIsNone(TDM.unusable_value_reason({"thd": None, "reason": "INSUFFICIENT_DATA"}), "no value is not judged")


DRIVER = r'''
import json, sys
from datetime import datetime
import numpy as np
sys.path.insert(0, sys.argv[1])
import calibration_eligibility_fixtures as FX
FX.install_stubs()
import calibration_eligibility_runner_parity as P
P._install_heavy_stubs()
import ensemble as E
import run_ensemble_daily as RD
import thd_daily_measurement as TDM
from seismic_thd import SeismicTHDAnalyzer
DAY = datetime(2026, 10, 2)
F0 = 1.0 / (12.42 * 3600)
t = np.arange(25 * 3600 + 1, dtype=float)
def tidal(seed):
    x = 1000.0 * np.sin(2 * np.pi * F0 * t)
    for k in range(2, 7):
        x = x + 40.0 * np.sin(2 * np.pi * F0 * k * t + 0.3 * k)
    return x + np.random.default_rng(seed).normal(scale=5.0, size=t.size)
WAVES = {"tidal1": lambda: tidal(1), "tidal2": lambda: tidal(2), "zeros": lambda: np.zeros(t.size),
         "flat": lambda: np.full(t.size, 5.0)}
plan = json.loads(sys.argv[2])
def fake_fetch(station_network, station_code, start, end, channel="BHZ", attempts=None):
    what = plan.get("%s.%s" % (station_network, station_code), "none")
    if attempts is not None:
        attempts.append({"provider": "SYNTHETIC", "outcome": "NO_TRACES" if what == "none" else "DATA_RETURNED"})
    return (None, 0.0) if what == "none" else (WAVES[what](), 1.0)
E.fetch_continuous_data_for_thd = fake_fetch
RD.RECORD_THD_ATTEMPTS = True
out = RD.run_region_assessment("anchorage", DAY).to_dict()
analyzer = SeismicTHDAnalyzer(n_harmonics=5, freq_tolerance=0.1, window_hours=24)
ens = E.GeoSpecEnsemble(region="anchorage")
out["_identities"] = {sid: {"measurement": TDM.measurement_record(*sid.split("."), analyzer)["identity"],
                            "estimator": ens._thd_estimator_identity(sid)} for sid in ("IU.COLA", "AK.BMR")}
print(json.dumps(out, sort_keys=True, default=str))
'''


class BeforeAfter(unittest.TestCase):
    """The REAL runner region path at d251072e and with this change, on the same synthetic inputs."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.base_src = _base_tree(cls.tmp.name)
        cls.driver = Path(cls.tmp.name) / "driver.py"
        cls.driver.write_text(DRIVER, encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_both(self, plan):
        outputs = {}
        for label, src in (("base", self.base_src), ("patched", HERE)):
            proc = subprocess.run([sys.executable, "-B", str(self.driver), str(src), json.dumps(plan)],
                                  capture_output=True, text=True, cwd=str(src))
            self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
            outputs[label] = json.loads(proc.stdout.strip().splitlines()[-1])
        return outputs["base"], outputs["patched"]

    @staticmethod
    def rows(out):
        return [(s["station"], s["outcome"], s["selected"]) for s in out["thd_attempts"]["stations"]]

    @staticmethod
    def without_provenance(out):
        out = json.loads(json.dumps(out))
        source = (out.get("thd_station_configuration") or {}).pop("source", None)
        return out, source

    def test_a_defined_value_and_every_identity_are_unchanged(self):
        base, patched = self.run_both({"IU.COLA": "tidal1"})
        self.assertEqual(self.rows(patched), [("IU.COLA", "VALUE", True), ("AK.BMR", "NOT_ATTEMPTED", False)])
        thd = patched["components"]["seismic_thd"]
        self.assertTrue(thd["available"] and thd["raw_value"] > 0)
        (b, b_source), (p, p_source) = self.without_provenance(base), self.without_provenance(patched)
        self.assertEqual(b, p, "outside the code-provenance stamp the region record is identical, identities included")
        if b_source is not None:
            changed = sorted(f for f in b_source["files"] if b_source["files"][f] != p_source["files"].get(f))
            self.assertTrue(set(changed) <= {"monitoring/src/run_ensemble_daily.py", "monitoring/src/ensemble.py",
                                             "monitoring/src/thd_daily_measurement.py",
                                             "monitoring/src/calibrate_thd_baselines.py"}, changed)

    def test_a_dead_channel_is_unavailable_and_the_fallback_supplies_the_value(self):
        base, patched = self.run_both({"IU.COLA": "zeros", "AK.BMR": "tidal2"})
        # the defect, as published at d251072e: the all-zero trace was SCORED as THD 0.0 and stopped the search
        self.assertEqual(self.rows(base), [("IU.COLA", "VALUE", True), ("AK.BMR", "NOT_ATTEMPTED", False)])
        self.assertEqual(base["components"]["seismic_thd"]["raw_value"], 0.0)
        self.assertTrue(base["components"]["seismic_thd"]["available"])
        # repaired: the undefined ratio is its own outcome, and the configured fallback is asked
        self.assertEqual(self.rows(patched), [("IU.COLA", "UNDEFINED_RATIO", False), ("AK.BMR", "VALUE", True)])
        self.assertIn("NONPOSITIVE_THD_OR_P1", patched["thd_attempts"]["stations"][0]["reason"])
        thd = patched["components"]["seismic_thd"]
        self.assertTrue(thd["available"])
        alone, _ = self.run_both({"IU.COLA": "tidal2"})       # BMR's value is the value of that waveform, nothing else
        self.assertEqual(thd["raw_value"], alone["components"]["seismic_thd"]["raw_value"])

    def test_a_flat_line_with_no_fallback_data_leaves_the_component_unavailable(self):
        base, patched = self.run_both({"IU.COLA": "flat"})
        self.assertEqual(self.rows(base)[0], ("IU.COLA", "VALUE", True))
        self.assertEqual(base["components"]["seismic_thd"]["raw_value"], 0.0)
        self.assertEqual(self.rows(patched), [("IU.COLA", "UNDEFINED_RATIO", False), ("AK.BMR", "NO_DATA", False)])
        thd = patched["components"]["seismic_thd"]
        self.assertFalse(thd["available"], "never a scored placeholder")
        self.assertTrue(thd["notes"].startswith("Insufficient data from AK.BMR"), thd["notes"])  # the last station asked
        self.assertIn("NONPOSITIVE_THD_OR_P1", patched["thd_attempts"]["stations"][0]["reason"])
        self.assertTrue(base["components"]["seismic_thd"]["available"], "at d251072e the flat line was scored")


if __name__ == "__main__":
    unittest.main()
