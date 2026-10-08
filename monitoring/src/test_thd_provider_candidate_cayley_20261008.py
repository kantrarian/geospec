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


class InstalledPins(RoutedFetch):
    """The installed table is grassmann's retained evidence (3227c6a6): one pin per station with a served VALUE trace id
    in scored 10-03..10-06, the pin's location equal to that trace id's, and no pin where nothing was served."""
    PINNED = {"BK.BKS": "00", "IU.ANTO": "00", "IU.COLA": "00", "IU.COR": "00", "IU.MAJO": "00", "IU.TATO": "00",
              "IU.TUC": "00", "MX.TLIG": ""}
    NOT_SERVED = ("AK.BMR", "AK.SSL", "G.UNM", "HINET.N.KI2H", "IU.SNZO", "IV.CAFE")

    def test_exactly_the_retained_stations_are_pinned(self):
        self.assertEqual({k: v["location"] for k, v in TPR.STATION_LOCATIONS.items()}, self.PINNED)
        for key in self.NOT_SERVED:
            with self.subTest(key=key):
                self.assertNotIn(key, TPR.STATION_LOCATIONS)

    def test_each_pin_is_the_location_of_the_trace_id_its_basis_quotes(self):
        for key, pin in TPR.STATION_LOCATIONS.items():
            with self.subTest(key=key):
                net, sta, loc, cha = pin["basis"].split("served trace id ", 1)[1].split(" ", 1)[0].split(".")
                self.assertEqual((net + "." + sta, loc, cha), (key, pin["location"], "BHZ"))
                self.assertEqual(TPR.selector_for(net, sta), (pin["location"], "PINNED " + pin["basis"]))

    def test_the_anchorage_primary_is_requested_at_its_pinned_location(self):
        _, _, attempts = self.fetch("IU", "COLA", {"IRIS": synthetic_stream("IU", "COLA", "00", "BHZ")})
        self.assertEqual(RecordingClient.asked, [("IRIS", "IU", "COLA", "00", "BHZ")])
        self.assertEqual(attempts[0]["nslc_requested"], "IU.COLA.00.BHZ")

    def test_a_blank_location_pin_asks_for_the_blank_location_not_the_wildcard(self):
        self.assertEqual(TPR.selector_for("MX", "TLIG")[0], "")


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
        with mock.patch.object(TPR, "coverage_facts", side_effect=RuntimeError("synthetic measurement failure")):
            data, rate, attempts = self.fetch("IV", "CAFE", {"INGV": synthetic_stream("IV", "CAFE", "", "BHZ")})
        self.assertEqual((len(data), rate, attempts[0]["outcome"]), (100 * 3600 * 25, 100.0, "DATA_RETURNED"))
        self.assertEqual(attempts[0]["coverage"], {"status": "UNMEASURED", "error_class": "RuntimeError"})

    def test_overlapping_input_leaves_the_value_path_unchanged_and_says_how_it_was_resolved(self):
        stream = gapped_stream("IV", "CAFE", "", "BHZ", gap_hours=(10, 9))   # [0, 10 h) and [9 h, 25 h): 1 h overlap
        with_sink, rate, attempts = self.fetch("IV", "CAFE", {"INGV": stream.copy()})
        RecordingClient.BEHAVIOUR = {"INGV": stream.copy()}
        without, rate_without = ST.fetch_continuous_data_for_thd("IV", "CAFE", START, END)
        self.assertTrue(np.array_equal(with_sink, without))
        self.assertEqual(rate, rate_without)
        cov = attempts[0]["coverage"]
        self.assertEqual((cov["overlaps"], cov["overlap_resolution"], cov["fill"], cov["gaps"], cov["status"]),
                         (1, "MERGE_METHOD_1_LATER_TRACE_KEPT_UNCHANGED", "NONE", 0, "FULL"))
        self.assertAlmostEqual(cov["overlap_seconds"], 3600.0, delta=0.1)

    def test_returned_data_the_local_detrend_refuses_is_local_not_a_provider_error(self):
        stream = synthetic_stream("IV", "CAFE", "", "BHZ")
        stream[0].data = stream[0].data.astype(np.float64)
        stream[0].data[:200] = np.nan    # obspy's linear detrend raises on nonfinite samples
        data, rate, attempts = self.fetch("IV", "CAFE", {"INGV": stream.copy()})
        RecordingClient.BEHAVIOUR = {"INGV": stream.copy()}
        self.assertEqual((data, rate), ST.fetch_continuous_data_for_thd("IV", "CAFE", START, END))
        self.assertEqual((data, rate), (None, 0.0))
        a = attempts[0]
        self.assertEqual((a["outcome"], a["typed_outcome"], a["exception_class"], a["http_status"], a["traces_before_merge"]),
                         ("LOCAL_PROCESSING_ERROR", "LOCAL_PROCESSING_ERROR", "ValueError", None, 1))
        self.assertIn("LOCAL_PROCESSING_ERROR", TPR.TYPED_OUTCOMES)
        cov = a["coverage_by_trace_id"]["IV.CAFE..BHZ"]
        self.assertEqual((cov["nonfinite_or_masked_handling"], cov["status"]), ("UNCHANGED_IN_VALUE_PATH", "PARTIAL"))
        self.assertAlmostEqual(cov["nonfinite_or_masked_seconds"], 2.0, places=6)

    def test_a_provider_that_raises_before_returning_stays_a_provider_outcome(self):
        _, _, attempts = self.fetch("IV", "CAFE", {"INGV": obspy_refusal(503)})
        self.assertEqual((attempts[0]["outcome"], attempts[0]["typed_outcome"]), ("PROVIDER_ERROR", "TRANSPORT_ERROR"))
        self.assertNotIn("coverage_by_trace_id", attempts[0])

def one_hz(offset, count, data=None):
    """1 Hz XX.TEST.00.BHZ trace starting `offset` seconds after T0; the request is [T0, T0 + 100 s)."""
    tr = Trace(np.ones(count) if data is None else data)
    tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel = "XX", "TEST", "00", "BHZ"
    tr.stats.sampling_rate, tr.stats.starttime = 1.0, CoverageEdges.T0 + offset
    return tr


class CoverageEdges(unittest.TestCase):
    """codex 1614 finding 4: support is the union of finite, unmasked sample intervals clipped to the request."""
    T0 = UTCDateTime("2026-01-01T00:00:00Z")

    def facts(self, *traces):
        return TPR.coverage_facts(Stream(list(traces)), self.T0, self.T0 + 100)["XX.TEST.00.BHZ"]

    def head(self, cov, *keys):
        return tuple(cov[k] for k in keys)

    def test_data_beyond_both_ends_is_clipped_to_the_request(self):
        cov = self.facts(one_hz(-10, 120))
        self.assertEqual(self.head(cov, "covered_seconds", "extent_seconds", "missing_head_seconds",
                                   "missing_tail_seconds", "status"), (100.0, 100.0, 0.0, 0.0, "FULL"))

    def test_a_trace_outside_the_request_neither_adds_nor_removes_coverage(self):
        cov = self.facts(one_hz(-200, 100), one_hz(0, 100), one_hz(150, 20))
        self.assertEqual(self.head(cov, "covered_seconds", "coverage_fraction", "gaps", "overlaps", "status"),
                         (100.0, 1.0, 0, 0, "FULL"))
        self.assertEqual(self.facts(one_hz(-200, 100))["status"], "NO_SAMPLES_IN_REQUEST")

    def test_a_duplicate_is_an_overlap_not_extra_coverage(self):
        cov = self.facts(one_hz(0, 100), one_hz(0, 100))
        self.assertEqual(self.head(cov, "covered_seconds", "overlaps", "overlap_seconds", "overlap_resolution", "fill"),
                         (100.0, 1, 100.0, "MERGE_METHOD_1_LATER_TRACE_KEPT_UNCHANGED", "NONE"))

    def test_overlap_resolution_is_disclosed_apart_from_gap_fill(self):
        cov = self.facts(one_hz(0, 70), one_hz(50, 50))
        self.assertEqual(self.head(cov, "overlaps", "overlap_seconds", "fill", "status"), (1, 20.0, "NONE", "FULL"))
        both = self.facts(one_hz(0, 30), one_hz(20, 20), one_hz(60, 40))   # overlap [20, 30), gap [40, 60)
        self.assertEqual(self.head(both, "overlaps", "overlap_seconds", "gaps", "gap_seconds", "fill", "overlap_resolution"),
                         (1, 10.0, 1, 20.0, "MERGE_INTERPOLATE_UNCHANGED", "MERGE_METHOD_1_LATER_TRACE_KEPT_UNCHANGED"))

    def test_nonfinite_and_masked_runs_are_not_numeric_coverage(self):
        data = np.ones(100)
        data[25:75] = np.nan
        data[80] = np.inf
        cov = self.facts(one_hz(0, 100, data))
        self.assertEqual(self.head(cov, "covered_seconds", "extent_seconds", "nonfinite_or_masked_seconds", "gaps",
                                   "fill", "nonfinite_or_masked_handling", "status"),
                         (49.0, 100.0, 51.0, 0, "NONE", "UNCHANGED_IN_VALUE_PATH", "PARTIAL"))
        masked = np.ma.masked_array(np.ones(100), mask=[10 <= i < 30 for i in range(100)])
        self.assertEqual(self.head(self.facts(one_hz(0, 100, masked)), "covered_seconds", "status"), (80.0, "PARTIAL"))

    def test_an_empty_stream_and_an_empty_trace(self):
        self.assertEqual(TPR.coverage_facts(Stream(), self.T0, self.T0 + 100), {})
        cov = self.facts(one_hz(0, 0))
        self.assertEqual(self.head(cov, "covered_seconds", "missing_head_seconds", "missing_tail_seconds", "status"),
                         (0.0, None, None, "NO_SAMPLES_IN_REQUEST"))

    def test_each_end_is_allowed_one_sample_period_and_no_more(self):
        self.assertEqual(self.facts(one_hz(1, 99))["status"], "FULL")
        self.assertEqual(self.facts(one_hz(0, 99))["status"], "FULL")
        late, short = self.facts(one_hz(2, 98)), self.facts(one_hz(0, 98))
        self.assertEqual(self.head(late, "missing_head_seconds", "edge_fill", "status"), (2.0, "NOT_FILLED", "PARTIAL"))
        self.assertEqual(self.head(short, "missing_tail_seconds", "edge_fill", "status"), (2.0, "NOT_FILLED", "PARTIAL"))

    def test_sub_half_sample_jitter_is_not_a_gap_but_a_larger_hole_is(self):
        jitter = self.facts(one_hz(0, 50), one_hz(50.3, 49))
        self.assertEqual(self.head(jitter, "gaps", "status"), (0, "FULL"))
        self.assertAlmostEqual(jitter["covered_seconds"], 99.0, places=6)
        hole = self.facts(one_hz(0, 50), one_hz(50.6, 49))
        self.assertEqual(self.head(hole, "gaps", "status"), (1, "PARTIAL"))
        self.assertAlmostEqual(hole["gap_seconds"], 0.6, places=6)

    def test_unordered_request_bounds_are_refused(self):
        for end in (self.T0, self.T0 - 1):
            with self.subTest(end=str(end)), self.assertRaises(ValueError):
                TPR.coverage_facts(Stream([one_hz(0, 100)]), self.T0, end)


class EpochInstants(unittest.TestCase):
    """codex 1614 finding 3: exact UTC instants, offsets applied, half-open edges, ordered bounds."""
    WINDOW = ("2026-01-01T00:00:00.000Z", "2026-01-01T00:01:40Z")

    def status(self, *epochs):
        return TPR.response_epoch_status(list(epochs), *self.WINDOW)

    def test_a_fractional_second_late_start_is_partial(self):
        self.assertEqual(self.status(("2026-01-01T00:00:00.500Z", None, "late")),
                         ("EPOCH_PARTIALLY_COVERS_WINDOW", "late"))
        self.assertEqual(self.status(("2026-01-01T00:00:00.000000001Z", None, "1ns")),
                         ("EPOCH_PARTIALLY_COVERS_WINDOW", "1ns"))

    def test_offsets_are_applied_and_an_offset_free_instant_is_utc(self):
        self.assertEqual(self.status(("2026-01-01T00:00:00-05:00", None, "future")), ("NO_COVERING_EPOCH", None))
        self.assertEqual(self.status(("2026-01-01T05:00:00+05:00", None, "same")), ("SINGLE_EPOCH", "same"))
        self.assertEqual(TPR.utc_instant_ns("2026-01-01T00:00:00"), TPR.utc_instant_ns("2026-01-01T00:00:00Z"))
        self.assertEqual(TPR.utc_instant_ns("1970-01-01T00:00:01.000000001Z"), 1_000_000_001)

    def test_edges_are_half_open(self):
        self.assertEqual(self.status(("2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "ended")), ("NO_COVERING_EPOCH", None))
        self.assertEqual(self.status(("2026-01-01T00:01:40Z", None, "next")), ("NO_COVERING_EPOCH", None))
        self.assertEqual(self.status(("2025-01-01T00:00:00Z", "2026-01-01T00:01:40Z", "exact")), ("SINGLE_EPOCH", "exact"))
        self.assertEqual(self.status((None, None, "unbounded")), ("SINGLE_EPOCH", "unbounded"))

    def test_sequential_epochs_cross_and_are_ordered_by_start(self):
        self.assertEqual(self.status(("2026-01-01T00:00:50Z", None, "new"),
                                     ("2025-01-01T00:00:00Z", "2026-01-01T00:00:50Z", "old")),
                         ("CROSSES_EPOCH_BOUNDARY", ["old", "new"]))

    def test_overlapping_metadata_is_not_called_a_response_change(self):
        for epochs in ((("2025-01-01T00:00:00Z", None, "a"), ("2025-06-01T00:00:00Z", None, "b")),
                       (("2025-01-01T00:00:00Z", None, "a"), ("2025-01-01T00:00:00Z", None, "a")),
                       (("2025-01-01T00:00:00Z", None, "a"), ("2026-01-01T00:00:10Z", "2026-01-01T00:00:20Z", "b"))):
            with self.subTest(epochs=epochs):
                self.assertEqual(self.status(*epochs)[0], "OVERLAPPING_EPOCH_METADATA")

    def test_malformed_or_unordered_instants_are_refused(self):
        for bad in ("2026-13-01T00:00:00Z", "2026-01-01", "2026-01-01T24:00:00Z", "2026-01-01T00:00:00.1234567890Z",
                    "2026-01-01T00:00:00+15:00", "2026-01-01 00:00:00Z", None, 0):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                TPR.utc_instant_ns(bad)
        with self.assertRaises(ValueError):
            self.status(("2026-01-01T00:00:50Z", "2026-01-01T00:00:10Z", "inverted"))
        with self.assertRaises(ValueError):
            self.status(("2026-01-01T00:00:50Z", "2026-01-01T00:00:50Z", "empty"))
        for window in (("2026-01-01T00:01:40Z", "2026-01-01T00:00:00Z"), ("2026-01-01T00:00:00Z",) * 2):
            with self.subTest(window=window), self.assertRaises(ValueError):
                TPR.response_epoch_status([("2025-01-01T00:00:00Z", None, "a")], *window)


class RetainedResponseEpochs(RoutedFetch):
    """The retained StationXML (grassmann 3ebc83ba) read as diagnostic epoch status, unknown outside its query window."""
    SERVED_WITH_XML = ("BK.BKS.00.BHZ", "IU.ANTO.00.BHZ", "IU.COLA.00.BHZ", "IU.COR.00.BHZ", "IU.MAJO.00.BHZ",
                       "IU.SNZO.00.BHZ", "IU.TATO.00.BHZ", "IU.TUC.00.BHZ", "MX.TLIG..BHZ")

    def at(self, trace_id, start, end):
        return TPR.response_epoch_for(trace_id, start, end)

    def test_every_served_channel_with_retained_xml_is_in_the_table_and_none_without(self):
        for nslc in self.SERVED_WITH_XML:
            with self.subTest(nslc=nslc):
                entry = TPR.RESPONSE_METADATA[nslc]
                self.assertTrue(entry["basis"].startswith("RETAINED: stationxml_response_20261006/"))
                self.assertTrue(all(label == "%s@%s" % (nslc, b) for b, _, label in entry["epochs"]))
        for nslc in ("AK.BMR..BHZ", "AK.SSL..BHZ", "G.UNM.00.BHZ", "IV.CAFE..BHZ"):
            with self.subTest(nslc=nslc):
                self.assertNotIn(nslc, TPR.RESPONSE_METADATA)
                self.assertEqual(self.at(nslc, "2026-09-01T00:00:00Z", "2026-09-02T00:00:00Z"),
                                 {"status": "NO_RETAINED_RESPONSE_METADATA", "labels": None})

    def test_cola_crosses_its_2026_07_31_change_and_is_single_after_it(self):
        crossed = self.at("IU.COLA.00.BHZ", "2026-07-30T23:00:00Z", "2026-08-01T00:00:00Z")
        self.assertEqual(crossed["status"], "CROSSES_EPOCH_BOUNDARY")
        self.assertEqual(crossed["labels"], ["IU.COLA.00.BHZ@2023-07-12T00:00:00.000000Z",
                                             "IU.COLA.00.BHZ@2026-07-31T02:00:00.000000Z"])
        self.assertEqual(self.at("IU.COLA.00.BHZ", "2026-09-01T23:00:00Z", "2026-09-03T00:00:00Z")["status"], "SINGLE_EPOCH")

    def test_outside_the_retained_query_window_the_response_is_unknown(self):
        after = self.at("IU.COLA.00.BHZ", "2026-10-07T23:00:00Z", "2026-10-09T00:00:00Z")
        self.assertEqual((after["status"], after["known"][1]), ("NO_COVERING_EPOCH", "2026-10-06T05:22:27.794893"))
        straddle = self.at("IU.COLA.00.BHZ", "2026-10-05T23:00:00Z", "2026-10-07T00:00:00Z")
        self.assertEqual(straddle["status"], "EPOCH_PARTIALLY_COVERS_WINDOW")
        before = self.at("IU.COLA.00.BHZ", "2025-10-01T00:00:00Z", "2025-10-02T00:00:00Z")
        self.assertEqual(before["status"], "NO_COVERING_EPOCH", "an epoch starting in 2023 is known only from the query")

    def test_ncedc_echoed_offset_and_far_future_end_are_read_exactly(self):
        lo, hi = TPR.RESPONSE_METADATA["BK.BKS.00.BHZ"]["known"]
        self.assertEqual(TPR.utc_instant_ns(lo), TPR.utc_instant_ns("2025-10-17T07:00:00Z"))
        self.assertLessEqual(TPR.utc_instant_ns(hi), TPR.utc_instant_ns("2026-10-06T05:22:32Z"))
        self.assertEqual(self.at("BK.BKS.00.BHZ", "2025-10-17T00:00:00Z", "2025-10-17T06:00:00Z")["status"],
                         "NO_COVERING_EPOCH")
        self.assertEqual(self.at("BK.BKS.00.BHZ", "2026-09-01T00:00:00Z", "2026-09-02T00:00:00Z")["status"],
                         "SINGLE_EPOCH")

    def test_the_fetch_records_the_epoch_of_the_trace_used_without_touching_the_value(self):
        with_sink, rate, attempts = self.fetch("IU", "COLA", {"IRIS": synthetic_stream("IU", "COLA", "00", "BHZ")})
        epoch = attempts[0]["response_epoch"]
        self.assertEqual((epoch["status"], epoch["labels"]), ("SINGLE_EPOCH", "IU.COLA.00.BHZ@2026-07-31T02:00:00.000000Z"))
        RecordingClient.BEHAVIOUR = {"IRIS": synthetic_stream("IU", "COLA", "00", "BHZ")}
        without, rate_without = ST.fetch_continuous_data_for_thd("IU", "COLA", START, END)
        self.assertTrue(np.array_equal(with_sink, without))
        self.assertEqual(rate, rate_without)

    def test_an_epoch_evaluation_failure_is_unmeasured_and_never_drops_the_value(self):
        with mock.patch.object(TPR, "response_epoch_for", side_effect=RuntimeError("synthetic")):
            data, rate, attempts = self.fetch("IU", "COLA", {"IRIS": synthetic_stream("IU", "COLA", "00", "BHZ")})
        self.assertEqual((attempts[0]["outcome"], rate), ("DATA_RETURNED", 100.0))
        self.assertIsNotNone(data)
        self.assertEqual(attempts[0]["response_epoch"], {"status": "UNMEASURED", "error_class": "RuntimeError"})


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
