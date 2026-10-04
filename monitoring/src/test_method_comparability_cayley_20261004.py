"""method-comparability-v1 contract tests (METHOD_QUALIFICATION_DELIVERY_PLAN M1, 2026-10-04).

Every probe goes through the REAL combiner (ensemble.GeoSpecEnsemble.compute_risk) and, where persistence or the
summary is involved, the REAL runner functions (run_ensemble_daily.check_persistence / save_results). Components are
synthetic MethodResults labelled as such; nothing here is measured data, and nothing activates the rule.
"""
import copy
import json
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import calibration_eligibility_fixtures as FX  # noqa: E402

FX.install_stubs()   # offline: the same stubs the existing eligibility tests use (no obspy, no network)
import calibration_eligibility_runner_parity as P  # noqa: E402

P._install_heavy_stubs()
import calibration_eligibility as CE  # noqa: E402
import ensemble as E  # noqa: E402
import method_comparability as MC  # noqa: E402
import run_ensemble_daily as RD  # noqa: E402

DAY = datetime(2026, 10, 2)
SYNTHETIC = "SYNTHETIC_COMPONENT_CONTRACT_NOT_MEASURED_DATA"
THD_CLASS = "THD_STATION_BASELINE:max_age=%s,min_lag=%s" % (
    E.MAX_BASELINE_AGE_DAYS, CE.registered_constant("run_thd_recal", "EXCLUDE_RECENT_DAYS"))


def comp(name, score=0.0, *, available=True, status=CE.STATUS_CALIBRATED, frozen=False, judged=True, code=None,
         station=None, shared=()):
    """A synthetic MethodResult; `judged` attaches a verdict exactly as CE.attach does."""
    r = E.MethodResult(name=name, available=available, raw_value=score, risk_score=score,
                       is_elevated=score >= 0.5, is_critical=score >= 0.75, notes=SYNTHETIC)
    if judged:
        reason = "%s: synthetic" % (code or ("CALIBRATED" if status in CE.ELIGIBLE_STATUSES else "SYNTHETIC_REFUSAL"))
        CE.attach(r, CE.Eligibility(status=status, eligible_for_tiering=status in CE.ELIGIBLE_STATUSES,
                                    reason=reason, code=reason.split(":")[0]))
    if name == "seismic_thd" and station:
        r.support = {"identity": station, "calibration_class": THD_CLASS, "shared_with": list(shared)}
    elif name == "lambda_geo" and judged and status in CE.ELIGIBLE_STATUSES:
        r.support = {"identity": "synthetic provenance", "calibration_class": "LG_BASELINE_RATIO:synthetic",
                     "shared_with": []}
    elif name == "fault_correlation" and judged and status in CE.ELIGIBLE_STATUSES:
        r.support = {"identity": "synthetic capsule", "calibration_class": "FC_CAPSULE:synthetic",
                     "shared_with": []}
    return r


def unavailable(name, *, status=None, code=None):
    r = E.MethodResult(name=name, available=False, raw_value=0.0, notes=SYNTHETIC)
    if status is not None:
        reason = "%s: synthetic" % code
        CE.attach(r, CE.Eligibility(status=status, eligible_for_tiering=False, reason=reason, code=code))
    return r


def run(region, lg, fc, thd, *, active=True):
    ens = E.GeoSpecEnsemble(region=region, eligibility_rule_active=active)
    ens.compute_lambda_geo_risk = lambda day: lg
    ens.compute_fault_correlation_risk = lambda day: (fc, 0, 0, [])
    ens.compute_thd_risk = lambda day, **kw: thd
    return ens.compute_risk(DAY)


def thd_only(region, score, station="IU.ANTO", shared=("istanbul_marmara", "turkey_kahramanmaras")):
    return run(region, unavailable("lambda_geo"),
               unavailable("fault_correlation", status=CE.STATUS_EXPIRED, code=CE.CAPSULE_PAST_VALID_THROUGH),
               comp("seismic_thd", score, station=station, shared=shared))


class QualificationIsNotTheScore(unittest.TestCase):
    def test_valid_zero_keeps_its_weight_and_is_labelled(self):
        res = run("r", comp("lambda_geo", 0.0), unavailable("fault_correlation"), comp("seismic_thd", 0.8,
                                                                                       station="IU.X"))
        ms = res.method_set
        self.assertEqual((ms["label"], ms["included"]), ("LG+THD", ["lambda_geo", "seismic_thd"]))
        self.assertEqual(ms["states"]["lambda_geo"], MC.VALID_ZERO)
        self.assertEqual(ms["valid_zero_methods"], ["lambda_geo"])
        self.assertAlmostEqual(ms["effective_weights"]["lambda_geo"], 4 / 7)
        self.assertAlmostEqual(res.combined_risk, 0.8 * 3 / 7)
        self.assertAlmostEqual(ms["weighted_coverage"], 0.7)

    def test_unqualified_zero_and_unqualified_high_are_both_excluded(self):
        for score in (0.0, 1.0):
            res = run("r", comp("lambda_geo", score, status=CE.STATUS_MISSING, code=CE.NO_PROVENANCE),
                      unavailable("fault_correlation"), comp("seismic_thd", 0.8, station="IU.X"))
            ms = res.method_set
            self.assertEqual(ms["included"], ["seismic_thd"])
            self.assertEqual(ms["excluded"]["lambda_geo"]["state"], MC.NOT_QUALIFIED)
            self.assertEqual(ms["excluded"]["lambda_geo"]["code"], CE.NO_PROVENANCE)
            self.assertAlmostEqual(res.combined_risk, 0.8)
            self.assertEqual(res.components["lambda_geo"].risk_score, score)   # raw score stays visible

    def test_every_refusal_state_is_distinct_and_excluded(self):
        cases = {
            MC.UNAVAILABLE: unavailable("seismic_thd"),
            MC.FROZEN: comp("seismic_thd", 0.6, frozen=False),
            MC.EXPIRED: unavailable("seismic_thd", status=CE.STATUS_EXPIRED, code=CE.CAPSULE_PAST_VALID_THROUGH),
            MC.STALE: comp("seismic_thd", 0.6, status=CE.STATUS_STALE, code=CE.WINDOW_STALE),
            MC.INVALID_DEFAULT: comp("seismic_thd", 0.6, status=CE.STATUS_N0_DEFAULT, code=CE.ZERO_SAMPLE_DEFAULT),
            MC.NOT_QUALIFIED: comp("seismic_thd", 0.6, status=CE.STATUS_MISSING, code=CE.LAG_NOT_HONORED),
            MC.UNKNOWN: comp("seismic_thd", 0.6, judged=False),
        }
        cases[MC.FROZEN].frozen = True
        for expected, thd in cases.items():
            q = MC.qualification(thd, True)
            self.assertEqual(q["state"], expected)
            self.assertFalse(q["included"], expected)

    def test_unknown_counts_with_the_rule_off_only(self):
        thd = comp("seismic_thd", 0.6, judged=False)
        self.assertTrue(CE.counts_for_tier(thd, False))
        self.assertEqual(MC.qualification(thd, False)["state"], MC.VALID)
        self.assertEqual(MC.qualification(thd, True)["state"], MC.UNKNOWN)

    def test_a_disagreeing_tier_predicate_refuses(self):
        thd = comp("seismic_thd", 0.6)
        thd.frozen = True
        original = MC.CE.counts_for_tier
        MC.CE.counts_for_tier = lambda c, active: True
        try:
            with self.assertRaises(MC.ContractViolation):
                MC.qualification(thd, True)
        finally:
            MC.CE.counts_for_tier = original

    def test_compute_risk_refuses_a_block_that_disagrees(self):
        original = MC.method_set

        def wrong(*a, **k):
            ms = original(*a, **k)
            ms["included"] = []
            return ms
        MC.method_set = wrong
        try:
            with self.assertRaises(MC.ContractViolation):
                run("r", comp("lambda_geo", 0.2), unavailable("fault_correlation"), comp("seismic_thd", 0.3))
        finally:
            MC.method_set = original


class TierRulesAreKept(unittest.TestCase):
    def test_all_invalid_is_degraded_with_no_group(self):
        res = run("kaikoura", unavailable("lambda_geo"), unavailable("fault_correlation"),
                  comp("seismic_thd", 0.41, status=CE.STATUS_N0_DEFAULT, code=CE.ZERO_SAMPLE_DEFAULT))
        self.assertEqual((res.tier_name, res.method_set["label"]), ("DEGRADED", MC.NO_METHODS_LABEL))
        self.assertEqual(MC.comparison_groups({"kaikoura": res.to_dict()}), {})

    def test_single_method_cap_is_kept(self):
        res = thd_only("turkey_kahramanmaras", 0.8114640273239097)
        self.assertEqual((res.tier_name, res.method_set["label"]), ("WATCH", "THD"))
        self.assertIn("Tier capped at WATCH", res.notes)

    def test_shared_station_support_is_not_independent(self):
        ms = thd_only("istanbul_marmara", 0.8114640273239097).method_set
        row = ms["support"]["seismic_thd"]
        self.assertEqual(row["shared_with"], ["istanbul_marmara", "turkey_kahramanmaras"])
        self.assertFalse(row["independent_of_other_regions"])
        self.assertTrue(ms["comparability_complete"])

    def test_rule_off_emits_nothing_new(self):
        res = run("r", comp("lambda_geo", 0.0, judged=False), unavailable("fault_correlation"),
                  comp("seismic_thd", 0.8, judged=False), active=False)
        self.assertIsNone(res.method_set)
        self.assertNotIn("method_set", res.to_dict())


class ComparisonAcrossRegions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mc-cayley-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def save(self, results, persistence=None):
        path = RD.save_results(results, __import__("pathlib").Path(self.tmp), DAY, persistence)
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    def test_different_method_sets_do_not_share_a_maximum(self):
        two = run("istanbul_marmara", comp("lambda_geo", 0.0), unavailable("fault_correlation"),
                  comp("seismic_thd", 0.8114640273239097, station="IU.ANTO",
                       shared=("istanbul_marmara", "turkey_kahramanmaras")))
        one = thd_only("turkey_kahramanmaras", 0.8114640273239097)
        out = self.save({"istanbul_marmara": two, "turkey_kahramanmaras": one})
        s = out["summary"]
        self.assertIsNone(s["max_risk_region"])
        self.assertIsNone(s["max_risk"])
        self.assertIn("2 comparability groups", s["max_risk_withheld"])
        labels = sorted(g["label"] for g in s["comparison"]["groups"].values())
        self.assertEqual(labels, ["LG+THD", "THD"])

    def test_one_group_keeps_the_legacy_maximum(self):
        a = thd_only("istanbul_marmara", 0.30)
        b = thd_only("turkey_kahramanmaras", 0.81)
        s = self.save({"istanbul_marmara": a, "turkey_kahramanmaras": b})["summary"]
        self.assertEqual((s["max_risk_region"], s["max_risk"]), ("turkey_kahramanmaras", 0.81))
        self.assertNotIn("max_risk_withheld", s)
        self.assertEqual(len(s["comparison"]["groups"]), 1)

    def test_an_exact_tie_on_a_shared_station_is_reported_not_broken(self):
        a = thd_only("istanbul_marmara", 0.8114640273239097)
        b = thd_only("turkey_kahramanmaras", 0.8114640273239097)
        s = self.save({"istanbul_marmara": a, "turkey_kahramanmaras": b})["summary"]
        self.assertIsNone(s["max_risk_region"])
        self.assertEqual(s["max_risk_tied_regions"], ["istanbul_marmara", "turkey_kahramanmaras"])
        (group,) = s["comparison"]["groups"].values()
        self.assertEqual(group["max_risk_regions"], ["istanbul_marmara", "turkey_kahramanmaras"])
        self.assertIsNone(group["max_risk_region"])

    def test_rule_off_summary_is_unchanged(self):
        res = run("r", comp("lambda_geo", 0.1, judged=False), unavailable("fault_correlation"),
                  comp("seismic_thd", 0.3, judged=False), active=False)
        s = self.save({"r": res})["summary"]
        self.assertNotIn("comparison", s)
        self.assertEqual(s["max_risk_region"], "r")


class PersistenceUnderRegimes(unittest.TestCase):
    def persistence(self, result, priors):
        """priors: issued rows nearest first (None = hole), fed through the REAL check_persistence loader."""
        def loader(days_back):
            row = priors[days_back - 1] if days_back - 1 < len(priors) else None
            return None if row is None else {"regions": {"turkey_kahramanmaras": row}}
        return RD.check_persistence({"turkey_kahramanmaras": result}, None, DAY, loader=loader)[
            "turkey_kahramanmaras"]

    def issued(self, tier, method_set=None):
        row = {"tier": tier}
        if method_set is not None:
            row["method_set"] = copy.deepcopy(method_set)
        return row

    def test_a_new_regime_does_not_inherit_old_tiers(self):
        cur = thd_only("turkey_kahramanmaras", 0.8114640273239097)
        p = self.persistence(cur, [self.issued(1), self.issued(1), self.issued(1)])
        self.assertEqual((p["consecutive_days"], p["is_confirmed"]), (1, False))
        self.assertEqual(p["regime"]["regime_transition"]["from"], MC.PRE_CONTRACT_REGIME)
        self.assertEqual(p["tier_history"], [1, 1, 1, 1])   # issued tiers are shown as issued

    def test_same_regime_days_count(self):
        cur = thd_only("turkey_kahramanmaras", 0.8114640273239097)
        same = cur.method_set
        p = self.persistence(cur, [self.issued(1, same), self.issued(1, same), self.issued(0, same)])
        self.assertEqual((p["consecutive_days"], p["is_confirmed"]), (3, True))
        self.assertIsNone(p["regime"]["regime_transition"])
        self.assertFalse(p["regime"]["method_set_changed"])

    def test_method_set_change_is_recorded_and_resets_confirmation(self):
        cur = thd_only("turkey_kahramanmaras", 0.8114640273239097)
        other = copy.deepcopy(cur.method_set)
        other["label"], other["comparability_key"] = "LG+THD", "LG+THD|" + other["regime"] + "|x"
        p = self.persistence(cur, [self.issued(1, other), self.issued(0, other)])
        self.assertEqual((p["consecutive_days"], p["is_confirmed"]), (1, False))
        self.assertTrue(p["regime"]["method_set_changed"])
        self.assertEqual(p["regime"]["method_set_history"][-2:], ["LG+THD", "THD"])
        self.assertTrue(p["regime"]["resets_on_method_set_change"])

    def test_a_hole_breaks_the_count(self):
        cur = thd_only("turkey_kahramanmaras", 0.8114640273239097)
        p = self.persistence(cur, [None, self.issued(1, cur.method_set)])
        self.assertEqual(p["consecutive_days"], 1)

    def test_rule_off_persistence_has_no_regime(self):
        res = run("turkey_kahramanmaras", unavailable("lambda_geo"), unavailable("fault_correlation"),
                  comp("seismic_thd", 0.81, judged=False), active=False)
        p = self.persistence(res, [self.issued(1), self.issued(1)])
        self.assertNotIn("regime", p)
        self.assertEqual((p["consecutive_days"], p["is_confirmed"]), (3, True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
