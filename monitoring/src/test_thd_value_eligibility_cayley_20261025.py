"""thd-value-eligibility-v1 tests (cayley 2026-10-25, F1; owner "draft F1"; codex c1612936 WORKS-WITH-FIX). Needs
numpy/scipy; the before/after checks also need git (they read the base tree d251072e) and run the REAL
run_ensemble_daily.run_region_assessment with the REAL GeoSpecEnsemble, thd_daily_measurement.measure and
SeismicTHDAnalyzer on SYNTHETIC waveforms (the fetch is replaced, nothing touches a network).

F1: the daily scorer published and scored values the weekly recal refuses. When the fundamental band power P1 is not
positive, THD = sqrt(sum P_n / P1) is undefined and analyze_window returns 0.0 as a placeholder; a dead channel
delivering zeros, or a flat line, is exactly that state. The shared rule keeps the recal's positive-only domain and adds
one intentional repair: a non-finite value or power is refused with its own reason (the recal's inline rule admitted
+inf). A refused value is unavailable and the runner tries the next configured station; the notes say which case it
is, and only a non-positive P1 is called an undefined ratio.
"""
import importlib.util
import hashlib
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
INF, NAN = float("inf"), float("nan")


def _base_tree(tmp):
    # the whole monitoring tree: the region path also reads tracked data beside src (e.g. data/calibration's registry)
    archive = subprocess.run(["git", "-C", str(REPO), "archive", BASE, "monitoring"], capture_output=True,
                             check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(tmp)], input=archive, check=True)
    return Path(tmp) / "monitoring/src"


class TheSharedRule(unittest.TestCase):
    """One rule, each refusal with its reason code and the case in words."""

    CASES = [
        # (thd, p1) -> (reason, case); "absent" = no p1 key
        ((0.0, 0.0), ("NONPOSITIVE_THD_OR_P1", TDM.RATIO_UNDEFINED)),
        ((0.0, -1.0), ("NONPOSITIVE_THD_OR_P1", TDM.RATIO_UNDEFINED)),
        ((0.42, 0.0), ("NONPOSITIVE_THD_OR_P1", TDM.RATIO_UNDEFINED)),
        ((0.0, 2.5), ("NONPOSITIVE_THD_OR_P1", TDM.OUTSIDE_POSITIVE_DOMAIN)),
        ((-0.0, 2.5), ("NONPOSITIVE_THD_OR_P1", TDM.OUTSIDE_POSITIVE_DOMAIN)),
        ((-1.0, 2.5), ("NONPOSITIVE_THD_OR_P1", TDM.OUTSIDE_POSITIVE_DOMAIN)),
        ((0.42, None), ("NONPOSITIVE_THD_OR_P1", TDM.P1_NOT_RECORDED)),
        ((0.42, "absent"), ("NONPOSITIVE_THD_OR_P1", TDM.P1_NOT_RECORDED)),
        ((INF, 2.5), ("NONFINITE_THD_OR_P1", TDM.NONFINITE_VALUE)),
        ((-INF, 2.5), ("NONFINITE_THD_OR_P1", TDM.NONFINITE_VALUE)),
        ((NAN, 2.5), ("NONFINITE_THD_OR_P1", TDM.NONFINITE_VALUE)),
        ((0.42, INF), ("NONFINITE_THD_OR_P1", TDM.NONFINITE_VALUE)),
        ((0.42, -INF), ("NONFINITE_THD_OR_P1", TDM.NONFINITE_VALUE)),
        ((0.42, NAN), ("NONFINITE_THD_OR_P1", TDM.NONFINITE_VALUE)),
        ((0.0, NAN), ("NONFINITE_THD_OR_P1", TDM.NONFINITE_VALUE)),     # finiteness is judged before sign
        ((0.42, 2.5), None),
        ((1e-300, 1e-300), None),
    ]

    def test_each_case_has_its_reason_and_its_words(self):
        for (thd, p1), want in self.CASES:
            rec = {"thd": thd} if p1 == "absent" else {"thd": thd, "p1": p1}
            with self.subTest(thd=thd, p1=p1):
                self.assertEqual(TDM.unusable_value(rec), want)
                self.assertEqual(TDM.unusable_value_reason(rec), None if want is None else want[0])

    def test_only_a_nonpositive_fundamental_is_called_an_undefined_ratio(self):
        for (thd, p1), want in self.CASES:
            if want is None or p1 == "absent" or p1 is None:
                continue
            with self.subTest(thd=thd, p1=p1):
                undefined = math.isfinite(thd) and math.isfinite(p1) and p1 <= 0
                self.assertEqual(want[1] == TDM.RATIO_UNDEFINED, undefined)

    def test_no_value_is_not_judged(self):
        self.assertIsNone(TDM.unusable_value({"thd": None, "reason": "INSUFFICIENT_DATA"}))
        self.assertIsNone(TDM.unusable_value({"thd": None, "p1": NAN}))


class TheRecalUsesTheSharedRule(unittest.TestCase):
    """calibrate_thd_baselines.compute_daily_record against d251072e on every record shape: identical receipts wherever
    neither recorded value is non-finite; the intentional repair everywhere else."""

    def test_finite_shapes_are_unchanged_and_nonfinite_shapes_are_refused(self):
        import calibrate_thd_baselines as patched
        with tempfile.TemporaryDirectory() as tmp:
            src = _base_tree(tmp)
            spec = importlib.util.spec_from_file_location("calibrate_thd_baselines_base_d251072e",
                                                          src / "calibrate_thd_baselines.py")
            base = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(base)
        thds = (None, 0.0, -0.0, 1e-300, 0.42, -1.0, NAN, INF, -INF)
        p1s = ("absent", None, 0.0, -1.0, 1e-300, 3.0, NAN, INF, -INF)
        same, repaired, base_admitted_nonfinite = [], [], []
        for thd in thds:
            for p1 in p1s:
                rec = {"thd": thd, "native_rate_hz": 1.0, "measurement_identity": "m", "reason": None}
                if p1 != "absent":
                    rec["p1"] = p1
                with self.subTest(thd=thd, p1=p1), mock.patch.object(TDM, "measure", lambda *a, **k: dict(rec)):
                    day = datetime(2026, 10, 22)
                    got, want = patched.compute_daily_record("IU", "COLA", day), base.compute_daily_record("IU", "COLA", day)
                    p1_value = rec.get("p1")
                    nonfinite = (thd is not None and p1_value is not None
                                 and not (math.isfinite(thd) and math.isfinite(p1_value)))
                    if not nonfinite:
                        self.assertEqual(json.dumps(got, sort_keys=True, default=str),
                                         json.dumps(want, sort_keys=True, default=str))
                        same.append((thd, p1))
                        continue
                    self.assertEqual(got["reason"], "NONFINITE_THD_OR_P1")
                    self.assertEqual((got["thd"], got["native_rate_hz"]), (None, None),
                                     "a non-finite value never enters a baseline")
                    moved = ("reason", "thd", "native_rate_hz")
                    self.assertEqual({k: v for k, v in got.items() if k not in moved},
                                     {k: v for k, v in want.items() if k not in moved}, "every other receipt field as before")
                    if want["thd"] is not None:
                        self.assertIsNone(want["reason"])
                        base_admitted_nonfinite.append((thd, p1))
                    else:
                        self.assertEqual(want["reason"], "NONPOSITIVE_THD_OR_P1")
                    repaired.append((thd, p1))
        self.assertEqual((len(same), len(repaired)), (45, 36))
        # the hole the repair closes: at d251072e these entered a baseline as values
        self.assertEqual(sorted(base_admitted_nonfinite, key=repr), sorted(
            [(t, p) for t in (1e-300, 0.42, INF) for p in (1e-300, 3.0, INF) if INF in (t, p)], key=repr))


class TheRuleIsRegistered(unittest.TestCase):
    """The policy version is recorded with the run's other rule versions and on every recalibrated baseline."""

    def test_the_run_configuration_names_the_rule(self):
        import thd_station_configuration as TSC
        self.assertEqual(TSC.versions()["value_eligibility"], "thd-value-eligibility-v1")
        self.assertEqual(TDM.VALUE_ELIGIBILITY_VERSION, "thd-value-eligibility-v1")

    def test_every_recalibrated_baseline_names_the_rule_and_still_loads(self):
        import calibrate_thd_baselines as C
        import run_thd_recal as R
        import station_baselines as SB

        def calibrate(network, station, **kw):
            return {"mean_thd": 0.39, "std_thd": 0.17, "n_samples": 91, "calibration_period": "2026-06-19 to 2026-09-17",
                    "estimator": dict(C.ESTIMATOR)}
        with tempfile.TemporaryDirectory(prefix="thd-recal-") as tmp:
            with mock.patch.object(R, "BASELINE_DIR", Path(tmp)), \
                    mock.patch.object(C, "calibrate_station", side_effect=calibrate), \
                    mock.patch.object(R, "_prior_effective_entries", return_value={}):
                path = Path(R.run_recal(["BK.BRK"], end_date=datetime(2026, 10, 17)))
            entry = json.loads(path.read_text())["BK.BRK"]
            self.assertEqual(entry["value_eligibility"], TDM.VALUE_ELIGIBILITY_VERSION)
            held = SB._file_baselines(path)["BK.BRK"]
            self.assertEqual((held.mean_thd, held.source_sha256),
                             (0.39, hashlib.sha256(path.read_bytes()).hexdigest()))


DRIVER = r'''
import dataclasses, json, sys
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
# identities FIRST, from the tree's own code, before anything below is replaced (the measurement identity hashes the
# module global `measure` at call time)
analyzer = SeismicTHDAnalyzer(n_harmonics=5, freq_tolerance=0.1, window_hours=24)
ens = E.GeoSpecEnsemble(region="anchorage")
ids = {sid: {"measurement": TDM.measurement_record(*sid.split("."), analyzer)["identity"],
             "estimator": ens._thd_estimator_identity(sid)} for sid in ("IU.COLA", "AK.BMR")}
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
CURRENT = {}
def fake_fetch(station_network, station_code, start, end, channel="BHZ", attempts=None):
    what = CURRENT["waves"].get("%s.%s" % (station_network, station_code), "none")
    if attempts is not None:
        attempts.append({"provider": "SYNTHETIC", "outcome": "NO_TRACES" if what == "none" else "DATA_RETURNED"})
    return (None, 0.0) if what == "none" else (WAVES[what](), 1.0)
_real_measure = TDM.measure
def measure(network, station, *a, **k):
    # a RECORD fixture: the real measurement of the station's waveform, with the analyzer's value and fundamental power
    # replaced in BOTH the record and its result (the scorer reads the result), so either tree scores what it is given.
    # The real `measure` is restored while it runs, so the identity it records is the tree's own.
    TDM.measure = _real_measure
    try:
        rec = _real_measure(network, station, *a, **k)
    finally:
        TDM.measure = measure
    o = CURRENT.get("override", {}).get("%s.%s" % (network, station))
    if o is not None and rec.get("thd") is not None:
        rec = dict(rec, thd=o["thd"], p1=o["p1"],
                   result=dataclasses.replace(rec["result"], thd_value=o["thd"], fundamental_power=o["p1"]))
    return rec
TDM.measure = measure
E.fetch_continuous_data_for_thd = fake_fetch
RD.RECORD_THD_ATTEMPTS = True
outs = []
for plan in json.loads(sys.argv[2]):
    CURRENT.clear(); CURRENT.update(plan)
    outs.append(RD.run_region_assessment("anchorage", DAY).to_dict())
print(json.dumps({"outs": outs, "identities": ids}, sort_keys=True, default=str))
'''

# Record fixtures (codex c1612936 acceptance): each on the primary IU.COLA, with AK.BMR serving a defined waveform.
RECORD_CASES = {
    "zero_thd_positive_p1": ({"thd": 0.0, "p1": 2.5}, "NONPOSITIVE_THD_OR_P1", TDM.OUTSIDE_POSITIVE_DOMAIN),
    "zero_thd_zero_p1": ({"thd": 0.0, "p1": 0.0}, "NONPOSITIVE_THD_OR_P1", TDM.RATIO_UNDEFINED),
    "inf_thd": ({"thd": INF, "p1": 2.5}, "NONFINITE_THD_OR_P1", TDM.NONFINITE_VALUE),
    "minus_inf_thd": ({"thd": -INF, "p1": 2.5}, "NONFINITE_THD_OR_P1", TDM.NONFINITE_VALUE),
    "nan_thd": ({"thd": NAN, "p1": 2.5}, "NONFINITE_THD_OR_P1", TDM.NONFINITE_VALUE),
    "inf_p1": ({"thd": 0.42, "p1": INF}, "NONFINITE_THD_OR_P1", TDM.NONFINITE_VALUE),
    "nan_p1": ({"thd": 0.42, "p1": NAN}, "NONFINITE_THD_OR_P1", TDM.NONFINITE_VALUE),
}
PLANS = dict(
    defined={"waves": {"IU.COLA": "tidal1"}},
    zeros_with_fallback={"waves": {"IU.COLA": "zeros", "AK.BMR": "tidal2"}},
    flat_no_fallback={"waves": {"IU.COLA": "flat"}},
    unusable_last_station={"waves": {"AK.BMR": "zeros"}},
    fallback_alone={"waves": {"IU.COLA": "tidal2"}},
    **{name: {"waves": {"IU.COLA": "tidal1", "AK.BMR": "tidal2"}, "override": {"IU.COLA": rec}}
       for name, (rec, _, _) in RECORD_CASES.items()})


class BeforeAfter(unittest.TestCase):
    """The REAL runner region path at d251072e and with this change, on the same synthetic inputs (one interpreter per
    tree for every plan)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.base_src = _base_tree(cls.tmp.name)
        driver = Path(cls.tmp.name) / "driver.py"
        driver.write_text(DRIVER, encoding="utf-8")
        names = sorted(PLANS)
        cls.out = {}
        for label, src in (("base", cls.base_src), ("patched", HERE)):
            proc = subprocess.run([sys.executable, "-B", str(driver), str(src), json.dumps([PLANS[n] for n in names])],
                                  capture_output=True, text=True, cwd=str(src))
            if proc.returncode != 0:
                raise AssertionError(proc.stderr[-3000:])
            got = json.loads(proc.stdout.strip().splitlines()[-1])
            cls.out[label] = dict(zip(names, got["outs"]), _identities=got["identities"])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @staticmethod
    def rows(out):
        return [(s["station"], s["outcome"], s["selected"]) for s in out["thd_attempts"]["stations"]]

    def both(self, name):
        return self.out["base"][name], self.out["patched"][name]

    def test_a_defined_value_and_every_identity_are_unchanged(self):
        base, patched = self.both("defined")
        self.assertEqual(self.rows(patched), [("IU.COLA", "VALUE", True), ("AK.BMR", "NOT_ATTEMPTED", False)])
        thd = patched["components"]["seismic_thd"]
        self.assertTrue(thd["available"] and thd["raw_value"] > 0)
        self.assertEqual(self.out["base"]["_identities"], self.out["patched"]["_identities"])
        for label, out in (("base", base), ("patched", patched)):
            ids = self.out[label]["_identities"]
            self.assertNotIn("UNIDENTIFIED", json.dumps(ids))
            # the identity the run recorded is the one the tree's own code forms (not a test double's)
            self.assertEqual(out["thd_attempts"]["stations"][0]["measurement_identity"], ids["IU.COLA"]["measurement"])
        b, p = json.loads(json.dumps(base)), json.loads(json.dumps(patched))
        b_conf, p_conf = b.pop("thd_station_configuration"), p.pop("thd_station_configuration")
        self.assertEqual(b, p, "outside the configuration block the region record is identical, identities included")
        # the configuration block differs by exactly the registration: the new version entry, the two content hashes
        # that cover the versions, and the code-provenance stamp of the changed source files
        self.assertEqual(p_conf["versions"], dict(b_conf["versions"], value_eligibility="thd-value-eligibility-v1"))
        self.assertEqual(p_conf["stations"], b_conf["stations"])
        for key in ("configuration_sha256", "region_configuration_sha256"):
            self.assertNotEqual(p_conf[key], b_conf[key])
        changed = sorted(f for f in b_conf["source"]["files"] if b_conf["source"]["files"][f] != p_conf["source"]["files"].get(f))
        self.assertTrue(set(changed) <= {"monitoring/src/run_ensemble_daily.py", "monitoring/src/thd_daily_measurement.py",
                                         "monitoring/src/thd_station_configuration.py"}, changed)
        rest = {k for k in set(b_conf) | set(p_conf) if b_conf.get(k) != p_conf.get(k)}
        self.assertEqual(rest, {"versions", "configuration_sha256", "region_configuration_sha256", "source"})

    def test_a_dead_channel_is_unavailable_and_the_fallback_supplies_the_value(self):
        base, patched = self.both("zeros_with_fallback")
        # the defect, as published at d251072e: the all-zero trace was SCORED as THD 0.0 and stopped the search
        self.assertEqual(self.rows(base), [("IU.COLA", "VALUE", True), ("AK.BMR", "NOT_ATTEMPTED", False)])
        self.assertEqual(base["components"]["seismic_thd"]["raw_value"], 0.0)
        self.assertTrue(base["components"]["seismic_thd"]["available"])
        self.assertEqual(self.rows(patched), [("IU.COLA", "THD_NOT_SCORABLE", False), ("AK.BMR", "VALUE", True)])
        reason = patched["thd_attempts"]["stations"][0]["reason"]
        self.assertTrue(reason.startswith("Unusable THD from IU.COLA (NONPOSITIVE_THD_OR_P1, thd-value-eligibility-v1: "
                                          + TDM.RATIO_UNDEFINED), reason)
        thd = patched["components"]["seismic_thd"]
        self.assertTrue(thd["available"])
        alone = self.out["patched"]["fallback_alone"]["components"]["seismic_thd"]["raw_value"]
        self.assertEqual(thd["raw_value"], alone, "BMR's value is the value of that waveform, nothing else")

    def test_a_flat_line_with_no_fallback_data_leaves_the_component_unavailable(self):
        base, patched = self.both("flat_no_fallback")
        self.assertEqual(self.rows(base)[0], ("IU.COLA", "VALUE", True))
        self.assertEqual(base["components"]["seismic_thd"]["raw_value"], 0.0)
        self.assertTrue(base["components"]["seismic_thd"]["available"], "at d251072e the flat line was scored")
        self.assertEqual(self.rows(patched), [("IU.COLA", "THD_NOT_SCORABLE", False), ("AK.BMR", "NO_DATA", False)])
        thd = patched["components"]["seismic_thd"]
        self.assertFalse(thd["available"], "never a scored placeholder")
        self.assertTrue(thd["notes"].startswith("Insufficient data from AK.BMR"), thd["notes"])  # the last station asked
        self.assertIn(TDM.RATIO_UNDEFINED, patched["thd_attempts"]["stations"][0]["reason"])

    def test_when_the_last_station_asked_is_unusable_the_component_says_so(self):
        base, patched = self.both("unusable_last_station")
        self.assertEqual(self.rows(base), [("IU.COLA", "NO_DATA", False), ("AK.BMR", "VALUE", True)])
        self.assertEqual(base["components"]["seismic_thd"]["raw_value"], 0.0)
        self.assertEqual(self.rows(patched), [("IU.COLA", "NO_DATA", False), ("AK.BMR", "THD_NOT_SCORABLE", False)])
        thd = patched["components"]["seismic_thd"]
        self.assertEqual((thd["available"], thd["raw_value"]), (False, 0.0))
        self.assertTrue(thd["notes"].startswith("Unusable THD from AK.BMR (NONPOSITIVE_THD_OR_P1, "
                                                "thd-value-eligibility-v1: " + TDM.RATIO_UNDEFINED + "; "), thd["notes"])
        self.assertEqual(patched["thd_attempts"]["stations"][1]["reason"], thd["notes"])

    def test_each_refused_record_is_unavailable_with_its_own_words_and_the_fallback_is_asked(self):
        alone = self.out["patched"]["fallback_alone"]["components"]["seismic_thd"]["raw_value"]
        words = {}
        for name, (rec, code, case) in RECORD_CASES.items():
            base, patched = self.both(name)
            with self.subTest(case=name):
                # at d251072e the primary's value was scored, whatever it was
                self.assertEqual(self.rows(base)[0], ("IU.COLA", "VALUE", True))
                self.assertEqual(self.rows(patched), [("IU.COLA", "THD_NOT_SCORABLE", False), ("AK.BMR", "VALUE", True)])
                reason = patched["thd_attempts"]["stations"][0]["reason"]
                self.assertTrue(reason.startswith("Unusable THD from IU.COLA (%s, thd-value-eligibility-v1: %s; "
                                                  % (code, case)), reason)
                self.assertTrue(reason.endswith("); not scored"), reason)
                self.assertEqual(patched["components"]["seismic_thd"]["raw_value"], alone)
                words[name] = case
        self.assertEqual(len(set(words.values())), 3, "undefined, outside the domain and non-finite read differently")


if __name__ == "__main__":
    unittest.main()
