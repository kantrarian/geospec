"""
calibration-eligibility-v3 tests (cayley 2026-10-01): the R3 lag re-check, the structured calibration date, and the
runner wiring (Lambda_geo provenance, configured station->regions map) -- all with the rule OFF in production.

    python -m unittest test_calibration_eligibility_v3_wiring_cayley_20261001

The runner-vs-base comparison needs the base commit's monitoring/src (e9b10680) in
CALIBRATION_ELIGIBILITY_BASE_SRC; without it that one class is SKIPPED by name. It runs the real runner of each tree
in a separate process through calibration_eligibility_runner_parity.py.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import calibration_eligibility_fixtures as FX  # noqa: E402

FX.install_stubs()
import calibration_eligibility as CE  # noqa: E402
import ensemble  # noqa: E402
import station_baselines as SB  # noqa: E402

BASE_SRC = os.environ.get("CALIBRATION_ELIGIBILITY_BASE_SRC")
PARITY = os.path.join(HERE, "calibration_eligibility_runner_parity.py")


def _classify(b, day=FX.SCORED_DAY, lag=FX.MIN_LAG):
    return CE.classify_thd_baseline(b, day, max_age_days=FX.MAX_AGE, min_lag_days=lag)


class LagRecheck(unittest.TestCase):
    """The registered lag, re-checked from the structured calibration date."""

    END = "2026-08-27"

    def _b(self, cal, n=88, period=None):
        return FX.baseline("IU.X", 0.30, 0.05, n, period or "2026-05-29 to %s" % self.END, calibration_date=cal)

    def test_boundaries(self):
        end = datetime(2026, 8, 27)
        cases = (((end + timedelta(days=30)).strftime("%Y-%m-%d"), "calibrated", "CALIBRATED"),
                 ((end + timedelta(days=29)).strftime("%Y-%m-%d"), "missing", "LAG_NOT_HONORED"),
                 (None, "missing", "CALIBRATION_DATE_UNKNOWN"),
                 ("2026-13-02", "missing", "CALIBRATION_DATE_UNKNOWN"),
                 ("2026-09-29", "missing", "CALIBRATED_AFTER_SCORED_DAY"),
                 ("2026-09-28", "calibrated", "CALIBRATED"))
        for cal, status, code in cases:
            with self.subTest(calibration_date=cal):
                e = _classify(self._b(cal))
                self.assertEqual((e.status, e.code, e.eligible_for_tiering), (status, code, status == "calibrated"))
                self.assertTrue(e.reason.startswith(code))

    def test_every_earlier_outcome_is_decided_before_the_lag(self):
        """v2 outcomes other than calibrated/shared do not depend on the calibration date at all."""
        honouring, absent = "2026-09-26", None
        for name, fx in FX.fixture_baselines().items():
            b = fx["baseline"]
            if b is None or fx["expected"] in ("calibrated", "shared_station") or name.startswith("lag_"):
                continue
            with self.subTest(fixture=name):
                with_date = FX.baseline(b.station, b.mean_thd, b.std_thd, b.n_samples, b.calibration_period,
                                        b.notes, calibration_date=honouring)
                without = FX.baseline(b.station, b.mean_thd, b.std_thd, b.n_samples, b.calibration_period,
                                      b.notes, calibration_date=absent)
                self.assertEqual(_classify(with_date).to_dict(), _classify(without).to_dict())

    def test_shared_station_also_requires_the_lag(self):
        b = self._b("2026-09-20")
        e = CE.classify_thd_baseline(b, FX.SCORED_DAY, max_age_days=FX.MAX_AGE, min_lag_days=FX.MIN_LAG,
                                     shared_regions=["ridgecrest", "socal_saf_mojave"])
        self.assertEqual((e.status, e.code), ("missing", "LAG_NOT_HONORED"))

    def test_lambda_geo_lag_is_checked_only_after_both_policies_are_registered(self):
        prov = {"source": "s", "n_days": 90, "window_start": "2026-05-26", "window_end": "2026-08-24",
                "calibrated_on": "2026-09-23"}
        self.assertEqual(CE.classify_lambda_geo(prov, FX.SCORED_DAY, max_age_days=None, min_lag_days=None).code,
                         "NO_REGISTERED_FRESHNESS_POLICY")
        self.assertEqual(CE.classify_lambda_geo(prov, FX.SCORED_DAY, max_age_days=50, min_lag_days=None).code,
                         "NO_REGISTERED_LAG_POLICY")
        self.assertEqual(CE.classify_lambda_geo(prov, FX.SCORED_DAY, max_age_days=50, min_lag_days=30).code,
                         "CALIBRATED")
        self.assertEqual(CE.classify_lambda_geo(dict(prov, calibrated_on="2026-09-22"), FX.SCORED_DAY,
                                                max_age_days=50, min_lag_days=30).code, "LAG_NOT_HONORED")
        # The runner's own registrations: neither exists, so the ratio stays not eligible.
        self.assertIsNone(ensemble.LAMBDA_GEO_BASELINE_MAX_AGE_DAYS)
        self.assertIsNone(ensemble.LAMBDA_GEO_BASELINE_MIN_LAG_DAYS)

    def test_rule_version(self):
        self.assertEqual(CE.ELIGIBILITY_RULE_VERSION, "calibration-eligibility-v3")
        self.assertFalse(CE.ELIGIBILITY_RULE_ACTIVE)


class StructuredCalibrationDate(unittest.TestCase):
    """station_baselines carries the calibration day as a field, from the dated recal file name."""

    def test_name_parsing_never_guesses(self):
        cases = {"thd_baselines_20260927.json": "2026-09-27", "thd_baselines_20260927_rerun.json": "2026-09-27",
                 "thd_baselines_2026.json": None, "thd_baselines_20261340.json": None,
                 "thd_baselines_2026092x.json": None, "other_20260927.json": None}
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(SB.calibration_date_from_name(name), expected)

    def test_loader_sets_the_field_and_defaults_stay_unknown(self):
        snapshot = dict(SB.STATION_BASELINES)
        tmp = tempfile.mkdtemp(prefix="thd_baselines_test_")
        try:
            self.assertTrue(all(getattr(b, "calibration_date", "absent") is None for b in snapshot.values()
                                if "Rolling recal" not in (b.notes or "")))
            with open(os.path.join(tmp, "thd_baselines_20260926.json"), "x", encoding="utf-8") as fh:
                json.dump({"IU.X": {"station": "IU.X", "mean_thd": 0.3, "std_thd": 0.05, "n_samples": 88,
                                    "calibration_period": "2026-05-29 to 2026-08-27"}}, fh)
            used = SB._load_newest_baseline_file(Path(tmp))
            self.assertEqual(used, "thd_baselines_20260926.json")
            b = SB.STATION_BASELINES["IU.X"]
            self.assertEqual(b.calibration_date, "2026-09-26")
            self.assertEqual(_classify(b).code, "CALIBRATED")
        finally:
            SB.STATION_BASELINES.clear()
            SB.STATION_BASELINES.update(snapshot)
            shutil.rmtree(tmp)


class EnsembleUsesTheRegisteredLag(unittest.TestCase):
    """With the rule active, compute_risk re-checks the lag run_thd_recal registers -- read, not restated."""

    def _thd(self, cal):
        b = FX.baseline("BK.BKS", 0.305512, 0.049728, 91, "2026-05-29 to 2026-08-27", calibration_date=cal)
        r = FX.build(ensemble, region="norcal_hayward", network="BK", station="BKS", thd_baseline=b,
                     thd_value=0.4497223, active=True)
        return r.components["seismic_thd"]

    def test_boundary_through_the_real_path(self):
        self.assertEqual(self._thd("2026-09-26").calibration_status, "calibrated")
        thd = self._thd("2026-09-25")
        self.assertEqual(thd.calibration_status, "missing")
        self.assertTrue(thd.eligibility_reason.startswith("LAG_NOT_HONORED"))

    def test_moving_the_registration_moves_the_decision(self):
        import run_thd_recal
        saved = run_thd_recal.EXCLUDE_RECENT_DAYS
        try:
            run_thd_recal.EXCLUDE_RECENT_DAYS = saved + 1
            self.assertTrue(self._thd("2026-09-26").eligibility_reason.startswith("LAG_NOT_HONORED"))
        finally:
            run_thd_recal.EXCLUDE_RECENT_DAYS = saved
        self.assertEqual(self._thd("2026-09-26").calibration_status, "calibrated")

    def test_rule_off_ignores_the_date(self):
        b = FX.baseline("BK.BKS", 0.305512, 0.049728, 91, "2026-05-29 to 2026-08-27", calibration_date=None)
        r = FX.build(ensemble, region="norcal_hayward", network="BK", station="BKS", thd_baseline=b,
                     thd_value=0.4497223, active=False)
        self.assertIsNone(r.components["seismic_thd"].calibration_status)
        self.assertNotIn("calibration", json.dumps(r.components["seismic_thd"].to_dict()))


def _runner():
    import calibration_eligibility_runner_parity as P
    P._install_heavy_stubs()
    import run_ensemble_daily as RD
    return RD


class RunnerWiring(unittest.TestCase):
    """run_ensemble_daily passes provenance and the configured map; the values themselves are unchanged."""

    @classmethod
    def setUpClass(cls):
        cls.RD = _runner()

    def test_configured_map_from_regions(self):
        RD = self.RD
        self.assertEqual(RD.THD_STATION_REGIONS, RD.configured_thd_station_regions(RD.REGIONS))
        self.assertEqual(RD.THD_STATION_REGIONS["IU.TUC"], ["ridgecrest", "socal_saf_coachella", "socal_saf_mojave"])
        self.assertEqual(RD.THD_STATION_REGIONS["IU.ANTO"], ["istanbul_marmara", "turkey_kahramanmaras"])
        self.assertEqual(RD.THD_STATION_REGIONS["IU.MAJO"], ["kumamoto", "tokyo_kanto"])   # tokyo_kanto via fallback
        self.assertEqual(RD.THD_STATION_REGIONS["IU.COLA"], ["anchorage"])
        for region, config in RD.REGIONS.items():
            if config.get("thd_station"):
                key = "%s.%s" % (config.get("thd_network", "CI"), config["thd_station"])
                self.assertIn(region, RD.THD_STATION_REGIONS[key])

    def test_single_read_derives_regions_and_provenance_from_one_object(self):
        RD = self.RD
        tmp = tempfile.mkdtemp(prefix="lg_baselines_test_")
        try:
            good = os.path.join(tmp, "good.json")
            with open(good, "x", encoding="utf-8") as fh:
                json.dump({"calibration_timestamp": "2026-09-26T06:10:00", "regions": {
                    "a": {"available": True, "n_samples": 85, "quality": "good",
                          "calibration_period": "2026-05-29 to 2026-08-27"},
                    "b": {"available": False},
                    "c": {"available": True, "n_samples": 12, "calibration_period": "garbled"}}}, fh)
            read = RD.read_lambda_geo_calibration(baseline_file=good)
            with open(good, "rb") as fh:
                digest = hashlib.sha256(fh.read()).hexdigest()
            self.assertEqual((read["status"], read["sha256"]), ("READ", digest))
            self.assertEqual(sorted(read["regions"]), ["a", "b", "c"])     # exactly as load_lambda_geo_baselines
            got = read["provenance"]
            self.assertEqual(sorted(got), ["a", "c"])
            self.assertEqual(got["a"], {"source": "lambda_geo_baselines.json:a", "n_days": 85,
                                        "window_start": "2026-05-29", "window_end": "2026-08-27",
                                        "calibrated_on": "2026-09-26", "quality": "good",
                                        "calibration_sha256": digest})
            self.assertIsNone(got["c"]["window_end"])
            e = CE.classify_lambda_geo(got["c"], FX.SCORED_DAY, max_age_days=50, min_lag_days=30)
            self.assertEqual(e.code, "WINDOW_UNREADABLE")
            bad = os.path.join(tmp, "bad.json")
            with open(bad, "x", encoding="utf-8") as fh:
                fh.write("{not json")
            bad_read = RD.read_lambda_geo_calibration(baseline_file=bad)
            self.assertEqual((bad_read["regions"], bad_read["provenance"], bad_read["status"]), ({}, {}, "UNREADABLE"))
            absent = RD.read_lambda_geo_calibration(baseline_file=os.path.join(tmp, "absent.json"))
            self.assertEqual((absent["regions"], absent["provenance"], absent["status"]), ({}, {}, "ABSENT"))
        finally:
            shutil.rmtree(tmp)

    def test_a_file_swap_between_reads_cannot_combine_versions(self):
        """codex a2a5cfe0 repair 1: the real fetch reads the calibration ONCE; a second version offered on any
        later read is never used, so the ratio and the attached provenance always share one version."""
        import calibration_eligibility_runner_parity as P
        import copy
        import io
        from unittest.mock import patch
        RD = self.RD
        old = copy.deepcopy(P.LG_BASELINE_FILE)
        new = copy.deepcopy(old)
        old["regions"]["ridgecrest"]["mean_lambda_geo"] = 0.01
        new["regions"]["ridgecrest"]["mean_lambda_geo"] = 0.02
        new["regions"]["ridgecrest"]["n_samples"] = 777
        new["calibration_timestamp"] = "2026-09-27T06:10:00"
        versions = [json.dumps(old).encode("utf-8"), json.dumps(new).encode("utf-8")]
        reads = []
        real_open = open

        def swapped_open(path, mode="r", *a, **k):
            if Path(str(path)).name != "lambda_geo_baselines.json":
                return real_open(path, mode, *a, **k)
            blob = versions[min(len(reads), 1)]
            reads.append(blob)
            return io.BytesIO(blob) if "b" in mode else io.StringIO(blob.decode("utf-8"))

        planted = {"NGL_LAMBDA_GEO_AVAILABLE": True, "NGLLiveAcquisition": P._StubNGL,
                   "GeoNetLiveAcquisition": P._StubNGL, "acquire_region_data": P._stub_acquire,
                   "FAULT_POLYGONS": {"ridgecrest": True}}
        absent = object()
        saved = {name: getattr(RD, name, absent) for name in planted}
        provenance = {}
        try:
            for name, value in planted.items():
                setattr(RD, name, value)
            with patch("builtins.open", swapped_open):
                ratios = RD.fetch_ngl_lambda_geo(["ridgecrest"], P.TARGET, provenance_out=provenance)
        finally:
            for name, value in saved.items():
                if value is absent:
                    delattr(RD, name)
                else:
                    setattr(RD, name, value)
        self.assertEqual(len(reads), 1)
        self.assertAlmostEqual(ratios["ridgecrest"], P.LG_MAX["ridgecrest"] / 0.01)
        rec = provenance["ridgecrest"]
        self.assertEqual((rec["n_days"], rec["calibrated_on"]), (85, "2026-09-26"))
        self.assertEqual(rec["calibration_sha256"], hashlib.sha256(versions[0]).hexdigest())

    def test_run_all_regions_passes_the_provenance_of_the_ratio_it_uses(self):
        RD = self.RD
        seen = {}
        saved = (RD.fetch_ngl_lambda_geo, RD.run_region_assessment, RD.NGL_LAMBDA_GEO_AVAILABLE,
                 RD.LAMBDA_GEO_PILOT_AVAILABLE)

        def fake_fetch(regions, target_date, days_back=120, provenance_out=None):
            if provenance_out is not None:
                provenance_out.update({"ridgecrest": {"source": "own"}, "cascadia": None, "hualien": {"source": "x"}})
            return {"ridgecrest": 1.5, "cascadia": 1.2, "hualien": 9.9}

        def fake_assess(**kwargs):
            seen[kwargs["region"]] = kwargs
            return None

        try:
            RD.fetch_ngl_lambda_geo, RD.run_region_assessment = fake_fetch, fake_assess
            RD.NGL_LAMBDA_GEO_AVAILABLE, RD.LAMBDA_GEO_PILOT_AVAILABLE = True, False
            RD.run_all_regions(FX.SCORED_DAY, regions=["ridgecrest", "cascadia", "hualien", "kaikoura"],
                               lambda_geo_data={"hualien": 3.0}, fetch_events=False)
        finally:
            (RD.fetch_ngl_lambda_geo, RD.run_region_assessment, RD.NGL_LAMBDA_GEO_AVAILABLE,
             RD.LAMBDA_GEO_PILOT_AVAILABLE) = saved
        self.assertEqual(seen["ridgecrest"]["lambda_geo_provenance"], {"source": "own"})
        self.assertIsNone(seen["cascadia"]["lambda_geo_provenance"])
        # The caller-supplied hualien ratio wins, so the NGL provenance must NOT be attached to it.
        self.assertEqual(seen["hualien"]["lambda_geo_ratio"], 3.0)
        self.assertIsNone(seen["hualien"]["lambda_geo_provenance"])
        self.assertIsNone(seen["kaikoura"]["lambda_geo_ratio"])
        self.assertTrue(all(kw["station_regions"] is RD.THD_STATION_REGIONS for kw in seen.values()))


@unittest.skipIf(not BASE_SRC, "base-commit monitoring/src not supplied (CALIBRATION_ELIGIBILITY_BASE_SRC)")
class RunnerIsByteIdenticalWithTheRuleOff(unittest.TestCase):
    """The REAL runner of the base tree and of this tree, same synthetic inputs, separate processes."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="runner_parity_")
        cls.out = {}
        for name, src, extra in (("base", BASE_SRC, []), ("off", HERE, []), ("on", HERE, ["--rule-on"])):
            path = os.path.join(cls.tmp, name + ".json")
            subprocess.run([sys.executable, "-B", PARITY, "--src", src, "--out", path] + extra, check=True)
            with open(path, "rb") as fh:
                cls.out[name] = fh.read()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_rule_off_equals_base(self):
        self.assertEqual(self.out["off"], self.out["base"])

    def test_the_comparison_can_fail(self):
        self.assertNotEqual(self.out["on"], self.out["base"])

    def test_rule_on_carries_the_wiring(self):
        on = json.loads(self.out["on"])
        lg = {r: on[r]["components"]["lambda_geo"]["calibration"]["reason"] for r in on
              if "lambda_geo" in on[r]["components"] and "calibration" in on[r]["components"]["lambda_geo"]}
        self.assertTrue(lg["ridgecrest"].startswith("NO_REGISTERED_FRESHNESS_POLICY: lambda_geo_baselines.json:ridgecrest"))
        self.assertTrue(lg["cascadia"].startswith("NO_PROVENANCE"))      # fallback-median baseline
        self.assertTrue(lg["hualien"].startswith("NO_PROVENANCE"))       # caller-supplied ratio
        thd = {r: on[r]["components"]["seismic_thd"]["calibration"]["status"] for r in on}
        self.assertEqual(thd["ridgecrest"], "shared_station")
        self.assertEqual(thd["kaikoura"], "n0_default")
        self.assertEqual(thd["anchorage"], "stale")
        self.assertEqual(thd["norcal_hayward"], "calibrated")
        self.assertEqual(thd["cascadia"], "missing")                     # fixture carries no calibration date
        base = json.loads(self.out["base"])
        for r in on:
            for name, comp in on[r]["components"].items():
                stripped = {k: v for k, v in comp.items() if k != "calibration"}
                self.assertEqual(stripped, base[r]["components"][name], (r, name))


if __name__ == "__main__":
    unittest.main()
