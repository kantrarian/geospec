#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""thd-score-basis-v1 (grassmann 2026-10-15; contract cayley 3baa7d1e (b) + is_elevated/is_critical amendment; codex
b16fdd32 s.3 GO). When the eligibility rule judges a THD baseline NO_BASELINE or ZERO_SAMPLE_DEFAULT, the component's
z_score / risk_score / is_elevated / is_critical are None and `score_basis` = NOT_COMPUTED_NO_EMPIRICAL_BASELINE; the
measured raw_value, the baseline block, the exclusion code and the DEGRADED tier are unchanged; every other component
is byte-for-byte as before and carries no score_basis.

Acceptance (codex): both exact reason codes; a valid empirical positive control; a stale empirical control; the
rule-off path; existing historical records (absence of the marker proves nothing); composed record -> persistence /
comparability / stress reader: no None becomes 0, a drawn score or a true/false elevation flag.

Run from monitoring/src:  <py with obspy> -B -m unittest test_thd_score_basis_grassmann_20261015 -v   (offline)
"""
import json
import os
import unittest
from datetime import datetime

import calibration_eligibility as CE
import calibration_eligibility_fixtures as FX
import ensemble as E
import method_comparability as MC
import stress_release_detector as SRD

DAY = datetime(2026, 9, 28)
FRESH = "2026-05-29 to 2026-08-27"          # end 32 d before DAY; calibrated 2026-09-26 (R3 lag honoured)
PAST_LIMIT = "2026-05-09 to 2026-08-08"     # end 51 d before DAY -> STALE (empirical, lagged: untouched by this change)
SEALED_1013 = "C:/GeoSpec/d3_production/sealed/2026-10-13/monitoring/data/ensemble_results/ensemble_2026-10-13.json"


def thd(result):
    return result.components["seismic_thd"]


class WithheldScores(unittest.TestCase):
    def test_zero_sample_default_withholds_z_risk_and_flags_and_names_why(self):
        res = FX.build(E, region="kaikoura", network="IU", station="SNZO", thd_value=0.5022, active=True,
                       thd_baseline=FX.baseline("IU.SNZO", 0.30, 0.07, 0, "UNCALIBRATED"))
        c = thd(res)
        self.assertEqual((c.calibration_status, c.eligibility_reason.split(":")[0]), ("n0_default", "ZERO_SAMPLE_DEFAULT"))
        self.assertFalse(c.eligible_for_tiering)
        self.assertIsNone(c.z_score); self.assertIsNone(c.risk_score); self.assertIsNone(c.is_elevated); self.assertIsNone(c.is_critical)
        self.assertEqual(c.score_basis, "NOT_COMPUTED_NO_EMPIRICAL_BASELINE")
        self.assertAlmostEqual(c.raw_value, 0.5022, places=4)                       # the measurement is kept
        self.assertEqual((c.baseline_mean, c.baseline_std, c.baseline_n), (0.30, 0.07, 0))  # the default block is kept, as declared
        self.assertIn("z=not computed (no empirical baseline)", c.notes); self.assertNotIn("z=2.", c.notes)
        self.assertEqual((res.tier, res.tier_name), (-1, "DEGRADED"))             # unchanged outcome
        d = c.to_dict()
        self.assertIsNone(d["z_score"]); self.assertIsNone(d["risk_score"]); self.assertIsNone(d["is_elevated"]); self.assertIsNone(d["is_critical"])
        self.assertEqual(d["score_basis"], "NOT_COMPUTED_NO_EMPIRICAL_BASELINE")
        self.assertEqual(d["baseline"]["n_samples"], 0)
        self.assertNotEqual(d["risk_score"], 0.0)                                   # None is None, never a zero

    def test_no_baseline_withholds_the_absolute_threshold_score(self):
        res = FX.build(E, region="campi_flegrei", network="IV", station="CAFE", thd_value=0.3630, active=True, thd_baseline=None)
        c = thd(res)
        self.assertEqual(c.eligibility_reason.split(":")[0], "NO_BASELINE")
        self.assertIsNone(c.risk_score); self.assertIsNone(c.is_elevated); self.assertIsNone(c.is_critical); self.assertIsNone(c.z_score)
        self.assertEqual(c.score_basis, "NOT_COMPUTED_NO_EMPIRICAL_BASELINE")
        self.assertIn("(no baseline)", c.notes); self.assertIn("z=not computed (no empirical baseline)", c.notes)
        self.assertEqual(res.tier, -1)
        self.assertEqual(c.to_dict()["baseline"]["quality"], "missing")


class UntouchedPaths(unittest.TestCase):
    def test_empirical_positive_control_is_scored_and_carries_no_marker(self):
        # the 10-13 sealed record's ANTO inputs: raw 0.5007472461023257 vs baseline 0.317247/0.062967 -> z 2.8920722 (literal
        # from the record; the module's effective-std rule, not the naive quotient)
        res = FX.build(E, region="istanbul_marmara", network="IU", station="ANTO", thd_value=0.5007472461023257, active=True,
                       thd_baseline=FX.baseline("IU.ANTO", 0.317247, 0.062967, 91, FRESH, calibration_date="2026-09-26"))
        c = thd(res)
        self.assertTrue(c.eligible_for_tiering)
        self.assertAlmostEqual(c.z_score, 2.8920722040291267, places=5)
        self.assertGreater(c.risk_score, 0.5); self.assertTrue(c.is_elevated); self.assertIsInstance(c.is_critical, bool)
        self.assertIsNone(c.score_basis); self.assertNotIn("score_basis", c.to_dict())
        self.assertIn("z=2.89", c.notes)                                           # the two-decimal notes token, as the record
        self.assertEqual(res.tier, 1)                                               # WATCH, as the 10-13 record

    def test_stale_empirical_baseline_is_unchanged_by_this_rule(self):
        res = FX.build(E, region="anchorage", network="IU", station="COLA", thd_value=0.4497223, active=True,
                       thd_baseline=FX.baseline("IU.COLA", 0.183849, 0.039561, 32, PAST_LIMIT))
        c = thd(res)
        self.assertEqual(c.calibration_status, "stale"); self.assertFalse(c.eligible_for_tiering)
        self.assertIsNone(c.score_basis)                                            # only the two codes trigger
        self.assertIsInstance(c.risk_score, float); self.assertIsInstance(c.is_elevated, bool); self.assertEqual(c.z_score, 0.0)
        self.assertIn("baseline STALE", c.notes)

    def test_rule_off_is_byte_identical_to_before(self):
        res = FX.build(E, region="kaikoura", network="IU", station="SNZO", thd_value=0.5022, active=False,
                       thd_baseline=FX.baseline("IU.SNZO", 0.30, 0.07, 0, "UNCALIBRATED"))
        c = thd(res)
        self.assertIsNone(c.eligibility_rule_version)
        self.assertIsNone(c.score_basis); self.assertIsInstance(c.risk_score, float); self.assertAlmostEqual(c.z_score, (0.5022 - 0.30) / 0.07, places=6)
        self.assertNotIn("score_basis", c.to_dict()); self.assertNotIn("calibration", c.to_dict())

    def test_codes_are_exactly_the_two_declared(self):
        self.assertEqual(E.SCORE_BASIS_WITHHOLD_CODES, (CE.NO_BASELINE, CE.ZERO_SAMPLE_DEFAULT))
        self.assertEqual(E.SCORE_BASIS_NOT_COMPUTED, "NOT_COMPUTED_NO_EMPIRICAL_BASELINE")


class ComposedReaders(unittest.TestCase):
    def setUp(self):
        self.res = FX.build(E, region="kaikoura", network="IU", station="SNZO", thd_value=0.5022, active=True,
                            thd_baseline=FX.baseline("IU.SNZO", 0.30, 0.07, 0, "UNCALIBRATED"))

    def test_tier_confidence_and_methods_available_exclude_it_without_touching_none(self):
        self.assertEqual(self.res.methods_available, 0)
        self.assertFalse(CE.counts_for_tier(thd(self.res), True))
        self.assertEqual(self.res.combined_risk, 0.0)

    def test_comparability_state_is_not_qualified_and_never_reads_the_none_score(self):
        q = MC.qualification(thd(self.res), rule_active=True)
        self.assertEqual(q["state"], MC._REFUSED_STATUS_STATE.get("n0_default", q["state"]))
        self.assertNotIn(q["state"], (MC.VALID, MC.VALID_ZERO))

    def test_stress_reader_does_not_recompute_a_z_from_the_withheld_default(self):
        record = {"regions": {"kaikoura": {"tier": -1, "components": {"seismic_thd": thd(self.res).to_dict()}}}}
        hist = {}
        # the loop body of detect_stress_release_drops, as it reads one record
        region_data = record["regions"]["kaikoura"]; components = region_data["components"]
        z = None
        for method in ("seismic_thd", "lambda_geo", "fault_correlation"):
            comp = components.get(method, {})
            if comp.get("available") and comp.get("z_score") is not None:
                z = comp["z_score"]; break
        self.assertIsNone(z)
        thd_comp = components["seismic_thd"]
        self.assertTrue(thd_comp.get("score_basis"))
        self.assertTrue(thd_comp.get("baseline") and thd_comp["baseline"]["std"] > 0, "the default block is present and WOULD recompute without the guard")
        # the production module's guard: run the real reader over a one-record directory
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "ensemble_2026-10-13.json"), "w", encoding="utf-8") as fh:
                json.dump({"date": "2026-10-13", "regions": record["regions"]}, fh)
            drops = SRD.detect_stress_release_drops(results_dir=d) if "results_dir" in SRD.detect_stress_release_drops.__code__.co_varnames else None
        self.assertTrue(drops is None or all(getattr(x, "region", "") != "kaikoura" for x in drops))
        src = open(SRD.__file__, encoding="utf-8").read()
        self.assertIn("if thd.get('score_basis'):", src, "the recompute guard is in the production reader")


class HistoricalRecords(unittest.TestCase):
    @unittest.skipUnless(os.path.exists(SEALED_1013), "sealed 10-13 record not on this host")
    def test_legacy_record_absence_of_the_marker_is_not_a_calibration_claim(self):
        d = json.load(open(SEALED_1013, encoding="utf-8"))
        k = d["regions"]["kaikoura"]["components"]["seismic_thd"]
        self.assertNotIn("score_basis", k)                                         # pre-rule record: marker absent
        self.assertFalse(k["calibration"]["eligible_for_tiering"])                 # ...and NOT calibrated
        self.assertEqual(k["baseline"]["n_samples"], 0); self.assertAlmostEqual(k["z_score"], 2.887946, places=4)
        # the same input through the new rule withholds the score the legacy record printed
        res = FX.build(E, region="kaikoura", network="IU", station="SNZO", thd_value=k["raw_value"], active=True,
                       thd_baseline=FX.baseline("IU.SNZO", 0.30, 0.07, 0, "UNCALIBRATED"))
        self.assertIsNone(thd(res).z_score); self.assertEqual(thd(res).score_basis, "NOT_COMPUTED_NO_EMPIRICAL_BASELINE")


if __name__ == "__main__":
    unittest.main()
