"""THD provider candidate (cayley 2026-10-08; codex 1515 s4, 1523 item 2). Needs obspy (1.5.1 pinned env).

Extends the thd-provider-routing-v1 harness (the REAL fetch, obspy's FDSN Client replaced by a recording fake, refusals
written by obspy's own raise_on_error) with the acceptance cases codex named: exact selector, honest partial coverage
with the value path unchanged, wrong/crossed response epoch, timeout and both providers of a two-adapter route down.
Waveforms are SYNTHETIC. Nothing here touches a network, reads a credential or activates anything.
"""
import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest import mock

import numpy as np
from obspy import Stream, Trace, UTCDateTime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import seismic_thd as ST  # noqa: E402
import thd_provider_routing as TPR  # noqa: E402
from test_thd_provider_routing_cayley_20261005 import (END, START, RecordingClient, RoutedFetch,  # noqa: E402
                                                       obspy_refusal, synthetic_stream)


def gapped_stream(net, sta, loc, cha, rate=20.0, gap_hours=(10, 11), head_hours=0.0):
    """Two traces covering [START+head, END) except the [gap) hours; deterministic, non-constant samples."""
    def trace(t0_hours, t1_hours):
        n = int(round((t1_hours - t0_hours) * 3600 * rate))
        tr = Trace(data=np.sin(np.arange(n) / 50.0).astype(np.float64))
        tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel = net, sta, loc, cha
        tr.stats.sampling_rate, tr.stats.starttime = rate, UTCDateTime(START) + t0_hours * 3600
        return tr
    return Stream([trace(head_hours, gap_hours[0]), trace(gap_hours[1], 25.0)])


class ExactSelector(RoutedFetch):
    def test_unpinned_station_keeps_the_wildcard_and_says_so(self):
        _, _, attempts = self.fetch("IV", "CAFE", {"INGV": synthetic_stream("IV", "CAFE", "", "BHZ")})
        self.assertEqual(RecordingClient.asked, [("INGV", "IV", "CAFE", "*", "BHZ")])
        self.assertEqual((attempts[0]["nslc_requested"], attempts[0]["selector_basis"]),
                         ("IV.CAFE.*.BHZ", "WILDCARD_LOCATION_NOT_PINNED"))

    def test_a_retained_pin_requests_exactly_that_location(self):
        pin = {"location": "00", "basis": "RETAINED: synthetic trace id IU.MAJO.00.BHZ"}
        with mock.patch.dict(TPR.STATION_LOCATIONS, {"IU.MAJO": pin}):
            _, _, attempts = self.fetch("IU", "MAJO", {"IRIS": synthetic_stream("IU", "MAJO", "00", "BHZ")})
        self.assertEqual(RecordingClient.asked, [("IRIS", "IU", "MAJO", "00", "BHZ")])
        self.assertEqual(attempts[0]["nslc_requested"], "IU.MAJO.00.BHZ")
        self.assertTrue(attempts[0]["selector_basis"].startswith("PINNED RETAINED: "))

    def test_a_pin_without_retained_evidence_or_with_a_bad_location_is_refused(self):
        for pin in ({"location": "00", "basis": "guess"}, {"location": "000", "basis": "RETAINED: x"},
                    {"location": "0a", "basis": "RETAINED: x"}, {"basis": "RETAINED: x"}):
            with self.subTest(pin=pin), mock.patch.dict(TPR.STATION_LOCATIONS, {"IU.MAJO": pin}):
                with self.assertRaises(ValueError):
                    TPR.selector_for("IU", "MAJO")


class HonestCoverage(RoutedFetch):
    def test_a_complete_day_is_full_with_no_fill(self):
        _, _, attempts = self.fetch("IV", "CAFE", {"INGV": synthetic_stream("IV", "CAFE", "", "BHZ")})
        cov = attempts[0]["coverage"]
        self.assertEqual((cov["status"], cov["fill"], cov["gaps"], cov["coverage_fraction"]), ("FULL", "NONE", 0, 1.0))

    def test_a_gap_is_partial_counted_and_the_value_path_is_unchanged(self):
        stream = gapped_stream("IV", "CAFE", "", "BHZ")
        with_sink, rate, attempts = self.fetch("IV", "CAFE", {"INGV": stream.copy()})
        RecordingClient.BEHAVIOUR = {"INGV": stream.copy()}
        without, rate_without = ST.fetch_continuous_data_for_thd("IV", "CAFE", START, END)
        self.assertTrue(np.array_equal(with_sink, without))
        self.assertEqual(rate, rate_without)
        cov = attempts[0]["coverage"]
        self.assertEqual((cov["status"], cov["gaps"], cov["fill"]), ("PARTIAL", 1, "MERGE_INTERPOLATE_UNCHANGED"))
        self.assertAlmostEqual(cov["gap_seconds"], 3600.0, delta=0.1)
        self.assertAlmostEqual(cov["coverage_fraction"], 24 / 25, places=4)

    def test_a_late_start_is_partial_with_the_missing_head_stated(self):
        _, _, attempts = self.fetch("IV", "CAFE", {"INGV": gapped_stream("IV", "CAFE", "", "BHZ", gap_hours=(12, 12),
                                                                          head_hours=2.0)})
        cov = attempts[0]["coverage"]
        self.assertEqual((cov["status"], cov["gaps"]), ("PARTIAL", 0))
        self.assertAlmostEqual(cov["missing_head_seconds"], 7200.0, delta=0.1)

    def test_a_measurement_failure_is_unmeasured_and_never_drops_the_value(self):
        class NoGaps(Stream):
            def get_gaps(self, *args, **kwargs):
                raise RuntimeError("synthetic measurement failure")
        stream = NoGaps(synthetic_stream("IV", "CAFE", "", "BHZ").traces)
        data, rate, attempts = self.fetch("IV", "CAFE", {"INGV": stream})
        self.assertEqual((len(data), rate, attempts[0]["outcome"]), (100 * 3600 * 25, 100.0, "DATA_RETURNED"))
        self.assertEqual(attempts[0]["coverage"], {"status": "UNMEASURED", "error_class": "RuntimeError"})


class ResponseEpoch(unittest.TestCase):
    WINDOW = ("2026-07-30T23:00:00", "2026-08-01T00:00:00")

    def test_one_epoch_spanning_the_window(self):
        epochs = [("2024-01-01T00:00:00", None, "IU.COLA.00.BHZ@2024")]
        self.assertEqual(TPR.response_epoch_status(epochs, *self.WINDOW), ("SINGLE_EPOCH", "IU.COLA.00.BHZ@2024"))

    def test_a_response_change_inside_the_window_is_crossed(self):
        epochs = [("2024-01-01T00:00:00", "2026-07-31T12:00:00", "old"), ("2026-07-31T12:00:00", None, "new")]
        self.assertEqual(TPR.response_epoch_status(epochs, *self.WINDOW), ("CROSSES_EPOCH_BOUNDARY", ["old", "new"]))

    def test_no_retained_epoch_overlapping_the_window_is_named(self):
        epochs = [("2020-01-01T00:00:00", "2025-01-01T00:00:00", "retired")]
        self.assertEqual(TPR.response_epoch_status(epochs, *self.WINDOW), ("NO_COVERING_EPOCH", None))

    def test_an_epoch_that_only_partly_overlaps_is_not_a_single_epoch(self):
        epochs = [("2026-07-31T06:00:00", None, "late-start")]
        self.assertEqual(TPR.response_epoch_status(epochs, *self.WINDOW), ("EPOCH_PARTIALLY_COVERS_WINDOW", "late-start"))


class BothProvidersDown(RoutedFetch):
    def test_bk_with_ncedc_timed_out_and_iris_unavailable_is_two_typed_transport_errors_and_no_value(self):
        data, rate, attempts = self.fetch("BK", "BKS", {"NCEDC": obspy_refusal(None, TimeoutError("timed out")),
                                                        "IRIS": obspy_refusal(503)})
        self.assertEqual((data, rate), (None, 0.0))
        self.assertEqual([(a["provider"], a["typed_outcome"], a["http_status"]) for a in attempts],
                         [("NCEDC", "TRANSPORT_ERROR", None), ("IRIS", "TRANSPORT_ERROR", 503)])
        self.assertTrue(all("coverage" not in a for a in attempts), "no coverage is invented for a failed attempt")

    def test_the_first_provider_down_and_the_second_answering_uses_the_second(self):
        data, rate, attempts = self.fetch("BK", "BKS", {"NCEDC": obspy_refusal(None, TimeoutError("timed out")),
                                                        "IRIS": synthetic_stream("BK", "BKS", "00", "BHZ")})
        self.assertEqual(rate, 100.0)
        self.assertEqual([(a["provider"], a["typed_outcome"]) for a in attempts],
                         [("NCEDC", "TRANSPORT_ERROR"), ("IRIS", "DATA_RETURNED")])


class FallbackWithoutItsOwnBaseline(unittest.TestCase):
    """A fallback that answers but has no baseline of its own is scored on the absolute mapping with baseline
    'missing' -- never on the primary's baseline (station_baselines.get_baseline is keyed by NET.STA and nothing in
    the runner substitutes) -- and, with the eligibility rule on, is not eligible for tiering. Uses the candidate's
    anchorage chain (IU.COLA primary, AK.BMR fallback); the fetch is a synthetic stand-in."""
    DAY = datetime(2026, 10, 8)   # eligibility rule ON (effective scored day 2026-10-07)

    def setUp(self):
        import ensemble as E
        import run_ensemble_daily as RD
        self.E, self.RD = E, RD
        original = E.fetch_continuous_data_for_thd
        self.addCleanup(setattr, E, "fetch_continuous_data_for_thd", original)
        self.answering = set()

        def fake(station_network, station_code, start, end, channel="BHZ", attempts=None):
            if "%s.%s" % (station_network, station_code) not in self.answering:
                return None, 0.0
            t = np.arange(25 * 3600, dtype=np.float64)
            return np.sin(2 * np.pi * t / (12.42 * 3600)) + 0.05 * np.sin(4 * np.pi * t / (12.42 * 3600)), 1.0
        E.fetch_continuous_data_for_thd = fake

    def thd(self, *answering):
        self.answering = set(answering)
        return self.RD.run_region_assessment("anchorage", self.DAY).components["seismic_thd"]

    def test_bmr_value_uses_no_baseline_and_is_not_eligible(self):
        import calibration_eligibility as CE
        thd = self.thd("AK.BMR")
        self.assertTrue(thd.available)
        self.assertIn("sta=AK.BMR", thd.notes)
        self.assertIn("(no baseline)", thd.notes)
        self.assertEqual((thd.baseline_quality, thd.baseline_mean, thd.baseline_n), ("missing", 0.0, 0))
        self.assertIs(thd.eligible_for_tiering, False)
        self.assertFalse(CE.counts_for_tier(thd, True))

    def test_cola_is_judged_against_its_own_baseline_never_missing(self):
        thd = self.thd("IU.COLA")
        self.assertIn("sta=IU.COLA", thd.notes)
        self.assertNotEqual(thd.baseline_quality, "missing")


if __name__ == "__main__":
    unittest.main()
