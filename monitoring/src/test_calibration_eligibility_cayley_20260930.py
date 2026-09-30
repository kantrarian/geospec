"""
Tests for calibration-eligibility-v1 (codex a7d0533d item 1). Pure python; the obspy-backed modules are stubbed by
calibration_eligibility_fixtures when absent. Run: python -m unittest test_calibration_eligibility_cayley_20260930

The base-commit comparison needs BASE_ENSEMBLE_PY (a copy of monitoring/src/ensemble.py at the base commit); the
report script and the packaging step write it to monitoring/data/reports/ensemble_base_<sha>.py, or set the
CALIBRATION_ELIGIBILITY_BASE_ENSEMBLE environment variable. Without it those two comparisons are SKIPPED by name.
"""
import glob
import json
import os
import sys
import unittest
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import calibration_eligibility_fixtures as FX  # noqa: E402  (installs stubs on demand)

FX.install_stubs()
import calibration_eligibility as CE  # noqa: E402
import ensemble  # noqa: E402

BASE_ENSEMBLE_PY = os.environ.get("CALIBRATION_ELIGIBILITY_BASE_ENSEMBLE") or next(
    iter(sorted(glob.glob(os.path.join(HERE, "..", "data", "reports", "ensemble_base_*.py")))), None)


def _thd(result):
    return result.components["seismic_thd"]


class StatusClasses(unittest.TestCase):
    """Every status the rule can assign, from structured fields only."""

    def test_every_thd_fixture_gets_its_expected_status(self):
        for name, fx in FX.fixture_baselines().items():
            with self.subTest(fixture=name):
                e = CE.classify_thd_baseline(fx["baseline"], FX.SCORED_DAY, max_age_days=FX.MAX_AGE,
                                             shared_regions=fx["regions"])
                self.assertEqual(e.status, fx["expected"])
                self.assertEqual(e.eligible_for_tiering, fx["expected"] in ("calibrated", "shared_station"))
                self.assertEqual(e.rule_version, CE.ELIGIBILITY_RULE_VERSION)

    def test_fc_and_lg_fixtures(self):
        for name, fx in FX.FC_FIXTURES.items():
            with self.subTest(fixture=name):
                e = CE.classify_fc_calibration(fx["state"], fx["reasons"])
                self.assertEqual(e.status, fx["expected"])
                self.assertEqual(e.eligible_for_tiering, fx["expected"] == "calibrated")
        for name, fx in FX.LG_FIXTURES.items():
            with self.subTest(fixture=name):
                e = CE.classify_lambda_geo(fx["provenance"], FX.SCORED_DAY, max_age_days=FX.MAX_AGE)
                self.assertEqual(e.status, fx["expected"])
                self.assertEqual(e.eligible_for_tiering, fx["expected"] == "calibrated")

    def test_status_set_is_closed(self):
        self.assertEqual(set(CE.STATUSES), {"missing", "zero", "n0_default", "stale", "expired", "calibrated", "shared_station"})
        with self.assertRaises(ValueError):
            CE._make("unknown", "never a status")
        with self.assertRaises(ValueError):
            CE.classify_fc_calibration("maybe")

    def test_unknown_qualification_never_qualifies(self):
        # Unreadable window with n > 0: age unknown -> missing, not calibrated.
        b = FX.baseline("IU.X", 0.3, 0.05, 40, "unknown")
        e = CE.classify_thd_baseline(b, FX.SCORED_DAY, max_age_days=FX.MAX_AGE)
        self.assertEqual((e.status, e.eligible_for_tiering), ("missing", False))
        # No target date -> age unknown -> missing.
        e = CE.classify_thd_baseline(FX.baseline("IU.X", 0.3, 0.05, 40, "2026-06-30 to 2026-09-27"), None,
                                     max_age_days=FX.MAX_AGE)
        self.assertEqual((e.status, e.eligible_for_tiering), ("missing", False))
        # An unclassified result (None) never counts while the rule is active.
        r = ensemble.MethodResult(name="seismic_thd", available=True, raw_value=0.4)
        self.assertTrue(CE.counts_for_tier(r, active=False))
        self.assertFalse(CE.counts_for_tier(r, active=True))

    def test_rule_reads_n_not_notes(self):
        # Mutation 1: the same default relabelled n>0 with a dated window becomes eligible on its DATA...
        relabelled = FX.baseline("IU.SNZO", 0.30, 0.07, 31, "2026-06-30 to 2026-09-27",
                                 "UNCALIBRATED. estimate based on similar IU broadband")
        e = CE.classify_thd_baseline(relabelled, FX.SCORED_DAY, max_age_days=FX.MAX_AGE)
        self.assertEqual((e.status, e.eligible_for_tiering), ("calibrated", True))
        # ...and a calibrated baseline whose NOTES say 'Auto-calibrated' but whose n is 0 is still n0_default.
        noted = FX.baseline("IU.SNZO", 0.30, 0.07, 0, "2026-06-30 to 2026-09-27", "Auto-calibrated. QA=acceptable.")
        e = CE.classify_thd_baseline(noted, FX.SCORED_DAY, max_age_days=FX.MAX_AGE)
        self.assertEqual((e.status, e.eligible_for_tiering), ("n0_default", False))

    def test_age_limit_boundary(self):
        # Mutation 2: window end exactly MAX_AGE days before -> calibrated; one day older -> stale.
        end = FX.SCORED_DAY - timedelta(days=FX.MAX_AGE)
        ok = FX.baseline("IU.X", 0.3, 0.05, 40, "2026-01-01 to %s" % end.strftime("%Y-%m-%d"))
        old = FX.baseline("IU.X", 0.3, 0.05, 40, "2026-01-01 to %s" % (end - timedelta(days=1)).strftime("%Y-%m-%d"))
        self.assertEqual(CE.classify_thd_baseline(ok, FX.SCORED_DAY, max_age_days=FX.MAX_AGE).status, "calibrated")
        self.assertEqual(CE.classify_thd_baseline(old, FX.SCORED_DAY, max_age_days=FX.MAX_AGE).status, "stale")
        self.assertEqual(CE.baseline_age_days("2026-06-30 to 2026-09-27", FX.SCORED_DAY), 1)
        self.assertIsNone(CE.baseline_age_days("UNCALIBRATED", FX.SCORED_DAY))


class KaikouraReproduction(unittest.TestCase):
    def test_measured_example_under_the_current_mapping(self):
        risk, z = ensemble.thd_to_risk_with_baseline(FX.KAIKOURA_THD, 0.30, 0.07)
        self.assertAlmostEqual(risk, 0.4097225, places=6)
        self.assertAlmostEqual(z, 2.1389, places=3)

    def test_rule_off_scores_the_default_baseline_as_today(self):
        fx = FX.fixture_baselines()["n0_default_kaikoura"]
        r = FX.build(ensemble, region="kaikoura", network="IU", station="SNZO", thd_baseline=fx["baseline"],
                     thd_value=fx["thd"], active=False)
        thd = _thd(r)
        self.assertTrue(thd.available)
        self.assertAlmostEqual(thd.risk_score, 0.4097225, places=6)
        self.assertEqual(thd.baseline_n, 0)
        self.assertIsNone(thd.calibration_status)
        self.assertNotIn("calibration", thd.to_dict())
        self.assertEqual(r.methods_available, 1)
        self.assertEqual((r.tier, r.tier_name), (1, "WATCH"))     # 0.4097 alone -> WATCH today

    def test_rule_on_keeps_the_raw_value_but_removes_it_from_the_tier(self):
        fx = FX.fixture_baselines()["n0_default_kaikoura"]
        r = FX.build(ensemble, region="kaikoura", network="IU", station="SNZO", thd_baseline=fx["baseline"],
                     thd_value=fx["thd"], active=True)
        thd = _thd(r)
        self.assertTrue(thd.available)                                    # numeric availability unchanged
        self.assertAlmostEqual(thd.raw_value, FX.KAIKOURA_THD, places=9)  # raw value visible
        self.assertAlmostEqual(thd.risk_score, 0.4097225, places=6)       # saved score unchanged
        self.assertEqual(thd.calibration_status, "n0_default")
        self.assertIs(thd.eligible_for_tiering, False)
        self.assertEqual(thd.to_dict()["calibration"]["status"], "n0_default")
        self.assertEqual(r.methods_available, 0)
        self.assertEqual((r.tier, r.tier_name), (-1, "DEGRADED"))
        self.assertIn("excluded from tier seismic_thd=n0_default", r.notes)


class RuleOnTierEffects(unittest.TestCase):
    """With the rule ON, only calibrated inputs move the tier; raw values stay in the components."""

    def _run(self, name, **kw):
        fx = FX.fixture_baselines()[name]
        return FX.build(ensemble, region=fx["regions"][0], network=fx["network"], station=fx["station"],
                        thd_baseline=fx["baseline"], thd_value=fx["thd"], shared=fx["regions"], active=True, **kw)

    def test_default_stale_zero_missing_do_not_tier(self):
        for name in ("n0_default_kaikoura", "stale_window", "zero_baseline", "missing_no_baseline"):
            with self.subTest(fixture=name):
                r = self._run(name)
                thd = _thd(r)
                self.assertTrue(thd.available)
                self.assertFalse(thd.eligible_for_tiering)
                self.assertEqual(r.methods_available, 0)
                self.assertEqual(r.tier, -1)

    def test_calibrated_and_shared_tier_as_before(self):
        for name in ("calibrated_fresh", "calibrated_at_age_limit", "shared_station_tuc"):
            with self.subTest(fixture=name):
                on = self._run(name)
                fx = FX.fixture_baselines()[name]
                off = FX.build(ensemble, region=fx["regions"][0], network=fx["network"], station=fx["station"],
                               thd_baseline=fx["baseline"], thd_value=fx["thd"], shared=fx["regions"], active=False)
                self.assertTrue(_thd(on).eligible_for_tiering)
                self.assertEqual(_thd(on).calibration_status, fx["expected"])
                self.assertEqual((on.tier, on.combined_risk, on.methods_available), (off.tier, off.combined_risk, off.methods_available))
                self.assertEqual(on.notes, "")

    def test_fc_expired_and_lg_provenance(self):
        fx = FX.fixture_baselines()["calibrated_fresh"]
        r = FX.build(ensemble, region="norcal_hayward", network="BK", station="BKS", thd_baseline=fx["baseline"],
                     thd_value=fx["thd"], active=True,
                     fc=dict(state="unavailable", reasons=["capsule for norcal_hayward expired 2026-08-01"]),
                     lg_ratio=3.0, lg_provenance=None)
        fcr, lg = r.components["fault_correlation"], r.components["lambda_geo"]
        self.assertEqual((fcr.available, fcr.calibration_status), (False, "expired"))
        self.assertEqual((lg.available, lg.calibration_status, lg.eligible_for_tiering), (True, "missing", False))
        self.assertEqual(r.methods_available, 1)                 # only the calibrated THD counts
        self.assertIn("lambda_geo=missing", r.notes)
        r2 = FX.build(ensemble, region="norcal_hayward", network="BK", station="BKS", thd_baseline=fx["baseline"],
                      thd_value=fx["thd"], active=True, fc=dict(state="admitted"),
                      lg_ratio=3.0, lg_provenance={"source": "ngl-baseline", "n_days": 90, "window_end": "2026-08-29"})
        self.assertEqual(r2.components["fault_correlation"].calibration_status, "calibrated")
        self.assertEqual(r2.components["lambda_geo"].calibration_status, "calibrated")
        self.assertEqual(r2.methods_available, 3)

    def test_rule_off_by_default(self):
        self.assertFalse(CE.ELIGIBILITY_RULE_ACTIVE)
        self.assertFalse(ensemble.GeoSpecEnsemble("kaikoura").eligibility_rule_active)


@unittest.skipIf(BASE_ENSEMBLE_PY is None, "base-commit ensemble.py copy not present (CALIBRATION_ELIGIBILITY_BASE_ENSEMBLE)")
class FlagOffIsByteIdentical(unittest.TestCase):
    """The candidate with the flag OFF serializes exactly what the base commit's ensemble serializes."""

    @classmethod
    def setUpClass(cls):
        cls.base = FX.load_module_from_file(BASE_ENSEMBLE_PY, "ensemble_base")

    def _pair(self, name, **kw):
        fx = FX.fixture_baselines()[name]
        args = dict(region=fx["regions"][0], network=fx["network"], station=fx["station"],
                    thd_baseline=fx["baseline"], thd_value=fx["thd"], shared=fx["regions"])
        args.update(kw)
        base = FX.build(self.base, **args)
        cand = FX.build(ensemble, active=False, **args)
        return json.dumps(FX.result_dict(base), sort_keys=True), json.dumps(FX.result_dict(cand), sort_keys=True)

    def test_every_fixture_serializes_identically_with_the_flag_off(self):
        for name in FX.fixture_baselines():
            with self.subTest(fixture=name):
                a, b = self._pair(name)
                self.assertEqual(a, b)
        a, b = self._pair("calibrated_fresh", fc=dict(state="admitted"), lg_ratio=2.5)
        self.assertEqual(a, b)
        a, b = self._pair("calibrated_fresh", thd_unavailable=True)
        self.assertEqual(a, b)

    def test_flag_on_changes_only_the_named_fields(self):
        fx = FX.fixture_baselines()["n0_default_kaikoura"]
        args = dict(region="kaikoura", network="IU", station="SNZO", thd_baseline=fx["baseline"], thd_value=fx["thd"])
        base, cand = FX.result_dict(FX.build(self.base, **args)), FX.result_dict(FX.build(ensemble, active=True, **args))
        # The ONLY additions with the rule on are the per-component `calibration` blocks (every judged component
        # carries one) and the tier-level fields the rule is allowed to move.
        judged = sorted(name for name, comp in cand["components"].items() if "calibration" in comp)
        self.assertEqual(judged, ["fault_correlation", "seismic_thd"])   # lambda_geo was not supplied -> not judged
        for comp in cand["components"].values():
            comp.pop("calibration", None)
        for key in ("tier", "tier_name", "methods_available", "combined_risk", "confidence", "agreement", "notes", "effective_weights"):
            cand.pop(key), base.pop(key)
        self.maxDiff = None
        self.assertEqual(json.dumps(base, sort_keys=True), json.dumps(cand, sort_keys=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
