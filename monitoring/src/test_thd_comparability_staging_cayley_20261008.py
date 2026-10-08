"""Staged THD comparability binding (codex cfb195ff decision; cayley 2026-10-08). Needs obspy (1.5.1 pinned env).

The comparability key is driven through the REAL path: ensemble.thd_support -> method_comparability.method_set. The
cases codex named: the unchanged legacy path keeps its key byte-identical; a pure extraction keeps the legacy operator
class; the FIRST baseline computed under the new calibration convention changes the key and routine recals within it
do not; real operator changes (selector, coverage admission, response handling, estimator) cannot keep the old class;
provenance (basis text, route, source hashes, dates, receipts) never enters the key; the bound SNZO identity is kept.
Nothing here touches a network, disables the reset flag, rewrites history or activates a baseline.
"""
import json
import os
import sys
import tempfile
import types
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import calibration_eligibility as CE  # noqa: E402
import ensemble as E  # noqa: E402
import method_comparability as MC  # noqa: E402
import seismic_thd as ST  # noqa: E402
import station_baselines as SB  # noqa: E402
import thd_bound_station_operator as OP  # noqa: E402
import thd_daily_measurement as TDM  # noqa: E402
import thd_provider_routing as TPR  # noqa: E402
from test_thd_provider_routing_cayley_20261005 import configured_thd_networks  # noqa: E402

V1 = TDM.MEASUREMENT_VERSION


def baseline(station="IU.TUC", convention=None, period="2026-05-01 to 2026-07-01", mean=0.5, date="2026-07-31"):
    return SB.StationBaseline(station=station, mean_thd=mean, std_thd=0.1, n_samples=60, calibration_period=period,
                              calibration_date=date, calibration_convention=convention)


class Staging(unittest.TestCase):
    def setUp(self):
        stations, _ = configured_thd_networks()
        self.unbound = sorted(s for s in stations if s not in OP.BOUND_STATIONS)
        self.assertIn("AK.BMR", self.unbound)

    def ens(self):
        return E.GeoSpecEnsemble(region="anchorage", eligibility_rule_active=True,
                                 station_regions={s: ["r"] for s in self.unbound + ["IU.SNZO"]})

    def key(self, station, base=None, ens=None):
        """The real comparability key of a one-method (THD) region supported by `station` scored against `base`."""
        component = types.SimpleNamespace(name="seismic_thd", available=True, frozen=False, risk_score=0.4)
        support = (ens or self.ens()).thd_support(station, baseline=base)
        return MC.method_set({"seismic_thd": component}, {"seismic_thd": 1.0}, False,
                             support={"seismic_thd": support})["comparability_key"]

    def legacy_support(self, ens, station):
        """The THD support exactly as ensemble.thd_support built it at 823a84bc."""
        return {"identity": station,
                "calibration_class": "THD_STATION_BASELINE:max_age=%s,min_lag=%s" % (
                    E.MAX_BASELINE_AGE_DAYS, CE.registered_constant("run_thd_recal", "EXCLUDE_RECENT_DAYS")),
                "estimator": MC.compose_identity(MC.code_identity(type(ens.thd_analyzer).analyze_window,
                                                                  E.thd_to_risk_with_baseline)),
                "shared_with": ["r"]}

    def test_the_unchanged_legacy_path_keeps_its_823a84bc_support(self):
        ens = self.ens()
        with mock.patch.dict(TPR.STATION_LOCATIONS, {}, clear=True):
            for station in self.unbound:
                with self.subTest(station=station):
                    for base in (None, baseline(station)):
                        self.assertEqual(ens.thd_support(station, baseline=base), self.legacy_support(ens, station))

    def test_a_pure_extraction_keeps_the_legacy_operator_class(self):
        analyzer = ST.SeismicTHDAnalyzer()
        self.assertEqual(TDM.LEGACY_OPERATOR_CLASS["analyzer"]["fundamental_freq"], ST.PRIMARY_TIDAL_FREQ)
        with mock.patch.dict(TPR.STATION_LOCATIONS, {}, clear=True):
            for station in self.unbound:
                with self.subTest(station=station):
                    self.assertEqual(TDM.operator_class(*station.split(".", 1), analyzer), TDM.LEGACY_OPERATOR_CLASS)
                    self.assertIsNone(TDM.operator_class_part(*station.split(".", 1), analyzer))

    def test_the_installed_pins_change_the_class_of_exactly_the_pinned_stations(self):
        ens = self.ens()
        moved = {s for s in self.unbound if "|operator_class=" in ens.thd_support(s)["estimator"]}
        self.assertEqual(moved, set(TPR.STATION_LOCATIONS) & set(self.unbound))
        self.assertEqual(moved, {"BK.BKS", "IU.ANTO", "IU.COLA", "IU.COR", "IU.MAJO", "IU.TATO", "IU.TUC", "MX.TLIG"})

    def test_the_first_new_convention_changes_the_key_and_routine_recals_do_not(self):
        legacy = self.key("AK.BMR", baseline("AK.BMR"))
        first = self.key("AK.BMR", baseline("AK.BMR", V1))
        self.assertNotEqual(first, legacy)
        self.assertIn(",convention=" + V1, first)
        for routine in (baseline("AK.BMR", V1, period="2026-06-01 to 2026-08-01", mean=0.9, date="2026-08-31"),
                        baseline("AK.BMR", V1, period="2026-07-01 to 2026-09-01", mean=0.2, date="2026-09-30")):
            with self.subTest(period=routine.calibration_period):
                self.assertEqual(self.key("AK.BMR", routine), first)
        self.assertEqual(self.key("AK.BMR", None), legacy, "a missing or stale baseline adds no convention")

    def test_a_declared_convention_is_never_read_as_legacy(self):
        legacy = self.key("AK.BMR", baseline("AK.BMR"))
        self.assertNotEqual(self.key("AK.BMR", baseline("AK.BMR", "thd-daily-measurement-v2")), legacy)
        self.assertNotEqual(self.key("AK.BMR", baseline("AK.BMR", OP.DAILY_OPERATOR["operator_version"])), legacy)

    def test_real_operator_changes_cannot_keep_the_old_class(self):
        with mock.patch.dict(TPR.STATION_LOCATIONS, {}, clear=True):
            ens = self.ens()
            before, before_key = ens.thd_support("AK.BMR")["estimator"], self.key("AK.BMR", ens=ens)
            changes = {
                "selector": lambda: mock.patch.dict(TPR.STATION_LOCATIONS, {"AK.BMR": {
                    "location": "00", "basis": "RETAINED: synthetic"}}),
                "coverage_admission": lambda: mock.patch.object(TDM, "COVERAGE_ADMISSION", "REFUSE_PARTIAL_DAY"),
                "response_handling": lambda: mock.patch.object(TDM, "RESPONSE_HANDLING", "REMOVED_TO_VELOCITY"),
                "window": lambda: mock.patch.dict(TDM.MEASUREMENT, {"window": "[target, target + 25 h]"}),
                "resampling": lambda: mock.patch.dict(TDM.MEASUREMENT, {"resampling": "decimate"}),
                "estimator": lambda: mock.patch.dict(TDM.MEASUREMENT, {"estimator": "SeismicTHDAnalyzer.compute_thd"}),
                "analyzer": lambda: mock.patch.object(ens.thd_analyzer, "n_harmonics", 4),
            }
            for name, change in changes.items():
                with self.subTest(change=name):
                    with change():
                        self.assertNotEqual(ens.thd_support("AK.BMR")["estimator"], before)
                        self.assertNotEqual(self.key("AK.BMR", ens=ens), before_key)
                    self.assertEqual(ens.thd_support("AK.BMR")["estimator"], before)

    def test_provenance_never_enters_the_key(self):
        ens = self.ens()
        before = self.key("IU.TUC", baseline("IU.TUC", V1), ens=ens)
        pin = dict(TPR.STATION_LOCATIONS["IU.TUC"], basis="RETAINED: another retained basis, same location")
        provenance = {
            "basis_text": lambda: mock.patch.dict(TPR.STATION_LOCATIONS, {"IU.TUC": pin}),
            "route": lambda: mock.patch.dict(TPR.NETWORK_ROUTES, {"IU": {"adapters": ((TPR.FDSN, "GEOFON"),),
                                                                         "basis": "synthetic"}}),
            "source_hash": lambda: mock.patch.object(TDM, "_code_identity", return_value="x@000000000000"),
            "coverage_facts_version": lambda: mock.patch.object(TPR, "COVERAGE_VERSION", "thd-coverage-facts-v9"),
        }
        for name, change in provenance.items():
            with self.subTest(change=name), change():
                self.assertEqual(self.key("IU.TUC", baseline("IU.TUC", V1), ens=ens), before)
        moved = baseline("IU.TUC", V1, period="2026-04-01 to 2026-06-30", mean=1.7, date="2026-07-30")
        self.assertEqual(self.key("IU.TUC", moved, ens=ens), before, "dates and contents are not the key")

    def test_the_bound_snzo_identity_is_kept(self):
        ens = self.ens()
        expected = MC.compose_identity(MC.code_identity(type(ens.thd_analyzer).analyze_window,
                                                        E.thd_to_risk_with_baseline),
                                       "daily_operator=" + OP.daily_operator_identity("IU.SNZO"),
                                       MC.code_identity(OP.daily_measurement, OP.fetch_bound, OP.stitch_window))
        self.assertEqual(ens.thd_support("IU.SNZO")["estimator"], expected)
        own = baseline("IU.SNZO", OP.DAILY_OPERATOR["operator_version"])
        self.assertEqual(ens.thd_support("IU.SNZO", baseline=own)["calibration_class"],
                         ens.thd_support("IU.SNZO")["calibration_class"])

    def test_an_unknown_operator_is_unidentified_never_legacy(self):
        ens = self.ens()
        with mock.patch.object(TDM, "operator_class_part", side_effect=RuntimeError("synthetic")):
            self.assertEqual(ens.thd_support("AK.BMR")["estimator"], MC.UNIDENTIFIED)
        with mock.patch.dict(TPR.STATION_LOCATIONS, {"AK.BMR": {"location": "00", "basis": "a guess"}}):
            self.assertEqual(ens.thd_support("AK.BMR")["estimator"], MC.UNIDENTIFIED)


class ThePathsThatFeedTheKey(unittest.TestCase):
    def test_the_recal_stamps_the_convention_and_the_loader_reads_it(self):
        import calibrate_thd_baselines as C
        import run_thd_recal as R
        moments = {"mean_thd": 0.5, "std_thd": 0.1, "n_samples": 60, "calibration_period": "2026-06-01 to 2026-08-30"}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(R, "BASELINE_DIR", Path(tmp)), \
                mock.patch.object(C, "calibrate_station", return_value=moments), \
                mock.patch.object(R, "_prior_effective_entries", return_value={}), \
                mock.patch.object(TDM, "measurement_record", side_effect=RuntimeError("identity failed")):
            written = json.loads(Path(R.run_recal(["IU.TUC", "IU.SNZO"], end_date=datetime(2026, 10, 1))).read_text())
        self.assertEqual(written["IU.TUC"]["calibration_convention"], V1, "stamped even when the identity fails")
        self.assertEqual(written["IU.TUC"]["measurement"]["identity"], "UNIDENTIFIED")
        self.assertEqual(written["IU.SNZO"]["calibration_convention"], OP.DAILY_OPERATOR["operator_version"])
        self.assertEqual(SB._baseline_from_entry(written["IU.TUC"], "thd_baselines_20261001.json").calibration_convention, V1)
        legacy = {k: v for k, v in written["IU.TUC"].items() if k != "calibration_convention"}
        self.assertIsNone(SB._baseline_from_entry(legacy, "thd_baselines_20261001.json").calibration_convention)
        self.assertIsNone(SB._baseline_from_entry(dict(legacy, calibration_convention=7), "x").calibration_convention)

    def test_the_compute_path_scores_with_the_selected_baselines_convention(self):
        import numpy as np
        from obspy import Stream
        from test_thd_daily_measurement_cayley_20261008 import DAY, TrimmingClient, waveform
        TrimmingClient.STREAM, TrimmingClient.asked = Stream([waveform("IU", "TUC", "00")]), []
        fresh = baseline("IU.TUC", V1, period="2026-05-01 to 2026-07-01", date="2026-07-01")
        stale = baseline("IU.TUC", V1, period="2025-01-01 to 2025-03-01", date="2025-03-01")
        classes = {}
        for name, base in (("fresh", fresh), ("stale", stale)):
            with mock.patch("obspy.clients.fdsn.Client", TrimmingClient), \
                    mock.patch.dict(SB.STATION_BASELINES, {"IU.TUC": base}):
                ens = E.GeoSpecEnsemble(region="ridgecrest", eligibility_rule_active=True,
                                        station_regions={"IU.TUC": ["ridgecrest"]})
                result = ens.compute_thd_risk(DAY, station_network="IU", station_code="TUC")
            self.assertTrue(result.available and np.isfinite(result.raw_value))
            classes[name] = result.support["calibration_class"]
        self.assertTrue(classes["fresh"].endswith(",convention=" + V1))
        self.assertNotIn("convention=", classes["stale"], "a stale baseline is not scored against, so adds nothing")


if __name__ == "__main__":
    unittest.main()
