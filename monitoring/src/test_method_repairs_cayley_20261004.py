"""Repair-round tests (codex db9a28ff findings 2 and 4, method side), through the REAL combiner and runner.

  * the comparison identity carries the estimator/normalization code identity and the effective weights: identical
    method names under different weights or estimators never share a group; an unidentified estimator is incomplete
  * a maximum is labelled a descriptive method-score statistic, not calibrated regional risk
  * the dormant Lambda_geo-only path goes through the qualified combiner with the rule on (legacy with it off)
  * credential-shaped exception text is redacted before it becomes an attempt record
Synthetic components are labelled; nothing here is measured data or activates anything.
"""
import json
import os
import pathlib
import shutil
import sys
import tempfile
import unittest
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import test_method_comparability_cayley_20261004 as T  # noqa: E402  (installs the offline stubs)

CE, E, MC, RD = T.CE, T.E, T.MC, T.RD
import evidence_redaction as ER  # noqa: E402
import test_thd_attempts_cayley_20261004 as A  # noqa: E402

DAY = T.DAY


def thd_region(region, score, *, estimator=T.SYNTHETIC_ESTIMATOR, weights=None, lg_score=None):
    lg = T.comp("lambda_geo", lg_score) if lg_score is not None else T.unavailable("lambda_geo")
    ens = E.GeoSpecEnsemble(region=region, eligibility_rule_active=True, weights=weights)
    ens.compute_lambda_geo_risk = lambda day: lg
    ens.compute_fault_correlation_risk = lambda day: (T.unavailable("fault_correlation"), 0, 0, [])
    thd = T.comp("seismic_thd", score, station="IU.X" + region[:2].upper(), shared=(region,), estimator=estimator)
    ens.compute_thd_risk = lambda day, **kw: thd
    return ens.compute_risk(DAY)


class ComparisonIdentity(unittest.TestCase):
    def test_different_effective_weights_never_share_a_group(self):
        a = thd_region("aa", 0.3, lg_score=0.0)
        b = thd_region("bb", 0.3, lg_score=0.0, weights={"lambda_geo": 0.6, "fault_correlation": 0.2,
                                                           "seismic_thd": 0.2})
        self.assertEqual((a.method_set["label"], b.method_set["label"]), ("LG+THD", "LG+THD"))
        self.assertNotEqual(a.method_set["comparability_key"], b.method_set["comparability_key"])
        groups = MC.comparison_groups({"aa": a.to_dict(), "bb": b.to_dict()})
        self.assertEqual(len(groups), 2)

    def test_different_estimators_never_share_a_group(self):
        a = thd_region("aa", 0.3)
        b = thd_region("bb", 0.3, estimator="SYNTHETIC_ESTIMATOR_v2")
        self.assertNotEqual(a.method_set["comparability_key"], b.method_set["comparability_key"])
        self.assertEqual(len(MC.comparison_groups({"aa": a.to_dict(), "bb": b.to_dict()})), 2)

    def test_same_operator_on_different_stations_compares_descriptively(self):
        a, b = thd_region("aa", 0.3), thd_region("bb", 0.4)
        self.assertEqual(a.method_set["comparability_key"], b.method_set["comparability_key"])
        (group,) = MC.comparison_groups({"aa": a.to_dict(), "bb": b.to_dict()}).values()
        self.assertEqual(group["maximum_basis"], MC.MAXIMUM_BASIS)
        self.assertIn("NOT_CALIBRATED_REGIONAL_RISK", group["maximum_basis"])

    def test_an_unidentified_estimator_is_incomplete_and_withholds_the_maximum(self):
        a = thd_region("aa", 0.3, estimator=None)
        self.assertEqual(a.method_set["support"]["seismic_thd"]["estimator"], MC.UNIDENTIFIED)
        self.assertFalse(a.method_set["comparability_complete"])
        self.assertEqual(MC.comparison_groups({"aa": a.to_dict()}), {})
        tmp = tempfile.mkdtemp(prefix="mc-rep-")
        self.addCleanup(shutil.rmtree, tmp, True)
        path = RD.save_results({"aa": a, "bb": thd_region("bb", 0.4)}, pathlib.Path(tmp), DAY)
        summary = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))["summary"]
        self.assertIsNone(summary["max_risk_region"])
        self.assertEqual(summary["comparison"]["regions_with_incomplete_support"], ["aa"])

    def test_the_real_thd_support_carries_a_code_identity(self):
        ens = E.GeoSpecEnsemble(region="anchorage", eligibility_rule_active=True,
                                station_regions={"IU.COLA": ["anchorage"]})
        sup = ens.thd_support("IU.COLA")
        # IU.COLA is pinned to location 00 (3227c6a6), a selector change, so its operator class joins (codex cfb195ff)
        self.assertRegex(sup["estimator"], r"analyze_window@[0-9a-f]{12}\+thd_to_risk_with_baseline@[0-9a-f]{12}"
                                           r"\|operator_class=thd-operator-class-v1:[0-9a-f]{16}$")
        self.assertEqual(sup["shared_with"], ["anchorage"])
        self.assertEqual(MC.code_identity(len), MC.UNIDENTIFIED, "a builtin has no readable source")


class LambdaGeoOnlyPath(unittest.TestCase):
    def setUp(self):
        self.saved = dict(RD.REGIONS["anchorage"])
        self.addCleanup(RD.REGIONS.__setitem__, "anchorage", self.saved)
        RD.REGIONS["anchorage"] = dict(self.saved, seismic_available=False)
        flag, boundary = CE.ELIGIBILITY_RULE_ACTIVE, CE.EFFECTIVE_SCORED_DAY
        self.addCleanup(setattr, CE, "ELIGIBILITY_RULE_ACTIVE", flag)
        self.addCleanup(setattr, CE, "EFFECTIVE_SCORED_DAY", boundary)

    def test_rule_on_goes_through_the_qualified_combiner(self):
        CE.ELIGIBILITY_RULE_ACTIVE, CE.EFFECTIVE_SCORED_DAY = True, DAY.date().isoformat()   # an activation declares both
        res = RD.run_region_assessment("anchorage", DAY, lambda_geo_ratio=2.0, lambda_geo_provenance=None)
        self.assertIsNotNone(res.method_set, "the bypass would carry no method set")
        self.assertEqual(list(res.components), ["lambda_geo"])
        self.assertEqual(res.method_set["states"], {"lambda_geo": MC.NOT_QUALIFIED})
        self.assertEqual((res.tier_name, res.methods_available), ("DEGRADED", 0))

    def test_rule_off_keeps_the_legacy_construction(self):
        CE.ELIGIBILITY_RULE_ACTIVE = False
        res = RD.run_region_assessment("anchorage", DAY, lambda_geo_ratio=2.0)
        self.assertIsNone(res.method_set)
        self.assertEqual((res.methods_available, res.agreement, res.confidence), (1, "single_method", 0.5))
        self.assertAlmostEqual(res.combined_risk, E.lambda_geo_to_risk(2.0))


class Redaction(unittest.TestCase):
    SECRET = "https://alice:s3cr3tPW@service.example/x token=AbCdEf123456 HINET_PASSWORD=hunter2"

    def test_vectors(self):
        out = ER.redact(self.SECRET)
        for leaked in ("alice", "s3cr3tPW", "AbCdEf123456", "hunter2"):
            self.assertNotIn(leaked, out)
        self.assertEqual(ER.redact("HTTP Error 404: Not Found"), "HTTP Error 404: Not Found")
        digest = "19ff5853fb3a7c3ff8447f82578d25c9c800bd205f1254a6da658b689ffab6ed"
        self.assertIn(digest, ER.redact("digest " + digest))
        self.assertTrue(ER.redact("x" * 500).endswith("..."))

    def test_provider_exception_is_redacted_in_the_attempt_record(self):
        saved = A._install_fake_obspy()
        self.addCleanup(A._restore, saved)
        st = A._real_seismic_thd()
        original = A._Client.get_waveforms
        secret = self.SECRET

        def raising(self_, **kw):
            raise RuntimeError(secret)
        A._Client.get_waveforms = raising
        self.addCleanup(setattr, A._Client, "get_waveforms", original)
        attempts = []
        st.fetch_continuous_data_for_thd("IU", "STA", datetime(2026, 10, 1), datetime(2026, 10, 2), attempts=attempts)
        self.assertEqual(attempts[0]["outcome"], "PROVIDER_ERROR")
        for leaked in ("alice", "s3cr3tPW", "AbCdEf123456", "hunter2"):
            self.assertNotIn(leaked, attempts[0]["reason"])

    def test_an_error_note_is_redacted_in_the_station_record(self):
        comp = E.MethodResult(name="seismic_thd", available=False, raw_value=0.0, notes="Error: " + self.SECRET)
        outcome, reason = RD.thd_station_outcome(comp, [])
        self.assertEqual(outcome, "ERROR")
        self.assertNotIn("hunter2", reason)
        self.assertNotIn("s3cr3tPW", reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
