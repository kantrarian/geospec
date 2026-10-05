"""The amendment's effective scored-day boundary (calibration_eligibility.EFFECTIVE_SCORED_DAY).

Offline (the eligibility stubs). The rule applies to scored day D on the production daily path only when
ELIGIBILITY_RULE_ACTIVE and D >= EFFECTIVE_SCORED_DAY; an active rule with an UNSET boundary refuses; the shipped
constants are OFF / UNSET. The production construction site (run_ensemble_daily.run_region_assessment) is checked to
pass the per-day value, not the bare module flag.
"""
import os
import sys
import unittest
from datetime import date, datetime
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import test_method_comparability_cayley_20261004 as T  # noqa: E402,F401  (installs the offline stubs)
import calibration_eligibility as CE  # noqa: E402
import run_ensemble_daily as RD  # noqa: E402


class Boundary(unittest.TestCase):
    def test_shipped_constants_activate_on_2026_10_07(self):
        # The activation commit (owner go, effective scored day 2026-10-07 UTC; AMENDMENT_2026-10-05_method_qualification).
        self.assertIs(CE.ELIGIBILITY_RULE_ACTIVE, True)
        self.assertEqual(CE.EFFECTIVE_SCORED_DAY, "2026-10-07")
        self.assertIs(RD.RECORD_THD_ATTEMPTS, True)
        for day in (datetime(2026, 10, 2), date(2026, 10, 6), "2026-10-06"):
            self.assertIs(CE.rule_active_for_scored_day(day), False, day)
        for day in (datetime(2026, 10, 7), date(2026, 10, 8), "2030-01-01"):
            self.assertIs(CE.rule_active_for_scored_day(day), True, day)

    def test_an_active_rule_without_a_boundary_refuses(self):
        with mock.patch.object(CE, "ELIGIBILITY_RULE_ACTIVE", True), mock.patch.object(CE, "EFFECTIVE_SCORED_DAY", None):
            with self.assertRaises(ValueError) as raised:
                CE.rule_active_for_scored_day(datetime(2026, 10, 12))
            # UNSET is named as such, distinct from a malformed boundary
            self.assertIn("must declare its effective scored-day boundary", str(raised.exception))

    def test_the_boundary_is_inclusive_and_earlier_days_keep_the_legacy_behaviour(self):
        with mock.patch.object(CE, "ELIGIBILITY_RULE_ACTIVE", True), \
                mock.patch.object(CE, "EFFECTIVE_SCORED_DAY", "2026-10-12"):
            self.assertIs(CE.rule_active_for_scored_day(datetime(2026, 10, 11, 23, 59)), False)
            self.assertIs(CE.rule_active_for_scored_day(datetime(2026, 10, 12)), True)
            self.assertIs(CE.rule_active_for_scored_day(date(2026, 10, 13)), True)
            self.assertIs(CE.rule_active_for_scored_day("2026-10-12"), True)
            self.assertIs(CE.rule_active_for_scored_day("2026-09-30"), False)

    def test_malformed_days_and_boundaries_refuse(self):
        with mock.patch.object(CE, "ELIGIBILITY_RULE_ACTIVE", True):
            for bad in ("2026-13-01", "2026-10-1", "10/12/2026", 20261012, ""):
                with self.subTest(boundary=bad), mock.patch.object(CE, "EFFECTIVE_SCORED_DAY", bad):
                    with self.assertRaises(ValueError):
                        CE.rule_active_for_scored_day("2026-10-12")
            with mock.patch.object(CE, "EFFECTIVE_SCORED_DAY", "2026-10-12"):
                for bad in (None, "2026/10/12", 3):
                    with self.subTest(day=bad), self.assertRaises(ValueError):
                        CE.rule_active_for_scored_day(bad)
        with self.assertRaises(ValueError):          # the day is validated even while the rule is off
            CE.rule_active_for_scored_day("not-a-day")


class _Captured(Exception):
    pass


class ProductionPathPassesTheDaysRule(unittest.TestCase):
    def capture(self, day):
        seen = {}

        class Recorder:
            def __init__(self, **kwargs):
                seen.update(kwargs)
                raise _Captured()
        with mock.patch.object(RD, "GeoSpecEnsemble", Recorder):
            self.assertIsNone(RD.run_region_assessment("ridgecrest", day))
        return seen

    def test_off_and_unset_passes_false(self):
        self.assertIs(self.capture(datetime(2026, 10, 2))["eligibility_rule_active"], False)

    def test_a_misconfigured_rule_stops_the_run_instead_of_failing_each_region_quietly(self):
        with mock.patch.object(CE, "ELIGIBILITY_RULE_ACTIVE", True), mock.patch.object(CE, "EFFECTIVE_SCORED_DAY", None):
            with self.assertRaises(ValueError):
                RD.run_region_assessment("ridgecrest", datetime(2026, 10, 12))

    def test_active_passes_the_per_day_value(self):
        with mock.patch.object(CE, "ELIGIBILITY_RULE_ACTIVE", True), \
                mock.patch.object(CE, "EFFECTIVE_SCORED_DAY", "2026-10-12"):
            self.assertIs(self.capture(datetime(2026, 10, 11))["eligibility_rule_active"], False)
            self.assertIs(self.capture(datetime(2026, 10, 12))["eligibility_rule_active"], True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
