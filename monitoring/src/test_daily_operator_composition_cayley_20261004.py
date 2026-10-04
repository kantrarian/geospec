"""Integration of grassmann 809e8302 (thd-bound-station-daily-operator-v1) with cayley's M1 contract and attempts record.

Offline (the eligibility stubs; no obspy, no network). The provider fetch is replaced at the ensemble's own boundary
(ensemble.fetch_continuous_data_for_thd) with a SYNTHETIC fake that records what it was given; nothing is measured.
  1. the attempts sink reaches the SHARED daily operator's fetch for a bound station (IU.SNZO);
  2. without recording, the operator receives the unwrapped fetch (unchanged behaviour);
  3. an unbound station never reaches the operator and still records through the legacy fetch;
  4. a bound station's comparison key carries the daily operator (descriptor identity + producing code), and a change
     of the operator changes the key; an unreadable part makes the whole key UNIDENTIFIED.
"""
import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import test_method_comparability_cayley_20261004 as T  # noqa: E402,F401  (installs the offline stubs)
import ensemble as E  # noqa: E402
import method_comparability as MC  # noqa: E402
import thd_bound_station_operator as OP  # noqa: E402

DAY = datetime(2026, 10, 2)
SYNTHETIC_RECORD = {"provider": "SYNTHETIC", "outcome": "NO_TRACES", "note": "SYNTHETIC_NOT_MEASURED"}


class FakeFetch:
    def __init__(self):
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)
        if "attempts" in kw:
            kw["attempts"].append(dict(SYNTHETIC_RECORD, station="%s.%s" % (kw["station_network"], kw["station_code"])))
        return None, 0.0


class SinkReachesTheSharedOperator(unittest.TestCase):
    def run_one(self, station, record):
        fake = FakeFetch()
        seen = []
        original = OP.daily_measurement

        def spy(*a, **k):
            seen.append(k.get("fetch"))
            return original(*a, **k)
        with mock.patch.object(E, "fetch_continuous_data_for_thd", fake), \
                mock.patch.object(OP, "daily_measurement", side_effect=spy):
            ens = E.GeoSpecEnsemble(region="kaikoura", eligibility_rule_active=False, record_thd_attempts=record)
            net, sta = station.split(".")
            result = ens.compute_thd_risk(DAY, station_network=net, station_code=sta)
        return ens, result, fake, seen

    def test_a_bound_station_records_through_the_daily_operator(self):
        ens, result, fake, seen = self.run_one("IU.SNZO", True)
        self.assertFalse(result.available)
        self.assertEqual(len(seen), 1, "the bound station goes through the shared daily operator")
        (call,) = fake.calls
        self.assertIs(call["attempts"], ens.last_thd_fetch_attempts)
        self.assertEqual(call["start"], DAY - timedelta(hours=ens.thd_analyzer.window_hours + 1))
        self.assertEqual(call["end"], DAY)
        self.assertEqual(ens.last_thd_fetch_attempts, [dict(SYNTHETIC_RECORD, station="IU.SNZO")])

    def test_without_recording_the_operator_gets_the_unwrapped_fetch(self):
        ens, result, fake, seen = self.run_one("IU.SNZO", False)
        self.assertIs(seen[0], fake)
        self.assertNotIn("attempts", fake.calls[0])
        self.assertIsNone(ens.last_thd_fetch_attempts)

    def test_an_unbound_station_never_reaches_the_operator_and_still_records(self):
        ens, result, fake, seen = self.run_one("IU.COLA", True)
        self.assertEqual(seen, [])
        self.assertIs(fake.calls[0]["attempts"], ens.last_thd_fetch_attempts)
        self.assertEqual(ens.last_thd_fetch_attempts, [dict(SYNTHETIC_RECORD, station="IU.COLA")])


class BoundStationComparisonKey(unittest.TestCase):
    def ens(self):
        return E.GeoSpecEnsemble(region="kaikoura", eligibility_rule_active=True,
                                 station_regions={"IU.SNZO": ["kaikoura"], "IU.COLA": ["anchorage"]})

    def test_the_key_carries_the_daily_operator(self):
        ens = self.ens()
        bound, unbound = ens.thd_support("IU.SNZO")["estimator"], ens.thd_support("IU.COLA")["estimator"]
        self.assertTrue(bound.startswith(unbound + "|daily_operator=" + OP.daily_operator_identity("IU.SNZO") + "|"))
        self.assertIn("daily_measurement@", bound)
        self.assertIn("fetch_bound@", bound)
        self.assertIn("stitch_window@", bound)
        self.assertNotIn("daily_operator=", unbound, "an unbound station's key is unchanged")

    def test_a_change_of_the_operator_changes_the_key(self):
        before = self.ens().thd_support("IU.SNZO")["estimator"]
        with mock.patch.dict(OP.DAILY_OPERATOR, estimator_rate_hz=2.0):
            self.assertNotEqual(self.ens().thd_support("IU.SNZO")["estimator"], before)
        with mock.patch.dict(OP.BOUND_STATIONS["IU.SNZO"], location="10"):
            self.assertNotEqual(self.ens().thd_support("IU.SNZO")["estimator"], before)
        self.assertEqual(self.ens().thd_support("IU.SNZO")["estimator"], before)

    def test_an_unreadable_operator_makes_the_key_unidentified(self):
        with mock.patch.object(OP, "daily_measurement", len):
            self.assertEqual(self.ens().thd_support("IU.SNZO")["estimator"], MC.UNIDENTIFIED)
        self.assertNotEqual(self.ens().thd_support("IU.COLA")["estimator"], MC.UNIDENTIFIED)

    def test_compose_identity(self):
        self.assertEqual(MC.compose_identity("a", "b"), "a|b")
        for parts in ((), ("a", MC.UNIDENTIFIED), ("a", ""), ("a", None)):
            self.assertEqual(MC.compose_identity(*parts), MC.UNIDENTIFIED, parts)


class FaultCorrelationKeySibling(unittest.TestCase):
    """The FC estimator is a composite (code | processing | topology): an unreadable code part must make the whole
    key UNIDENTIFIED rather than hide inside the string. Driven through the REAL compute_risk with the eligibility
    fixtures' admitted capsule (synthetic, labelled)."""
    def fc_support(self):
        import calibration_eligibility_fixtures as FX
        fx = FX.fixture_baselines()["calibrated_fresh"]
        r = FX.build(E, region="norcal_hayward", network="BK", station="BKS", thd_baseline=fx["baseline"],
                     thd_value=fx["thd"], active=True,
                     fc=dict(state="admitted", capsule=FX.fc_fixtures()["fc_admitted"]["capsule"]))
        return r.components["fault_correlation"].support

    def test_the_fc_key_is_code_processing_topology(self):
        est = self.fc_support()["estimator"]
        self.assertTrue(est.startswith(MC.code_identity(E.fault_correlation_to_risk) + "|processing="), est)
        self.assertIn("|topology=", est)

    def test_an_unreadable_fc_code_part_makes_the_key_unidentified(self):
        original = MC.code_identity

        def unreadable(*objs):
            return MC.UNIDENTIFIED if E.fault_correlation_to_risk in objs else original(*objs)
        with mock.patch.object(MC, "code_identity", side_effect=unreadable):
            self.assertEqual(self.fc_support()["estimator"], MC.UNIDENTIFIED)


if __name__ == "__main__":
    unittest.main(verbosity=2)
