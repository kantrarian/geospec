"""thd-bound-station-operator-v2 controls (cayley 2026-10-10; codex b7a29458 correction 2).

Offline, synthetic records only (no provider I/O, no waveform acquisition). The v2 stitch accepts an overlap of any length
only when every overlapping sample is identical to the assembled support, on the native grid and inside one retained
response epoch; everything else that v1 refused still refuses, by name, with the measured facts. Inputs v1 accepted give
byte-identical output: v1 is loaded from the reviewed public commit (23aa8658, git object) and compared directly.

Run: <py with obspy> -B -m unittest test_thd_bound_operator_v2_cayley_20261010   (cwd monitoring/src)
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
from obspy import Trace, UTCDateTime

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import thd_bound_station_operator as OP  # noqa: E402

RATE = 40.0
DT = 1.0 / RATE
T0 = UTCDateTime("2026-10-06T23:00:00")
W = dict(network="IU", station="SNZO", location="00", channel="BHZ", expected_rate=RATE)
V1_COMMIT = "23aa86586b6bc321ec6330cc7cfa5e3982cf942b"


def trace(start, data, *, net="IU", sta="SNZO", loc="00", chan="BHZ", rate=RATE):
    tr = Trace(data if np.ma.isMaskedArray(data) else np.asarray(data))   # keep a mask: asarray would strip it
    tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel = net, sta, loc, chan
    tr.stats.sampling_rate, tr.stats.starttime = rate, start
    return tr


def series(n, seed=0):
    return (np.random.default_rng(seed).standard_normal(n) * 1000).astype(np.int32)


def window(n_samples):
    return dict(start=T0.datetime, end=(T0 + n_samples * DT).datetime)


def load_v1():
    """The v1 operator as reviewed and served (public commit 23aa8658), from the git object; None when not reachable."""
    try:
        source = subprocess.run(["git", "-C", str(HERE), "show", "%s:monitoring/src/thd_bound_station_operator.py" % V1_COMMIT],
                                capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    tmp = tempfile.NamedTemporaryFile("wb", suffix="_v1_operator.py", delete=False)
    tmp.write(source); tmp.close()
    spec = importlib.util.spec_from_file_location("thd_bound_station_operator_v1", tmp.name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    os.unlink(tmp.name)
    return module


class Joins(unittest.TestCase):
    def test_317_sample_identical_overlap_joins_once(self):
        whole = series(10000, seed=3)
        a = trace(T0, whole[:6000])
        b = trace(T0 + (6000 - 317) * DT, whole[6000 - 317:])       # the 10-08 shape: 317 identical samples, then new
        data, rate, rec = OP.stitch_window([b, a], **W, **window(10000))
        self.assertIsNone(rec["refusal"], rec)
        self.assertEqual(data.size, 10000)
        np.testing.assert_array_equal(data, whole.astype(np.float64))
        self.assertEqual((rec["identical_overlap_samples"], rec["identical_overlaps"]), (317, 1))

    def test_fully_nested_identical_record_never_shrinks_the_end(self):
        whole = series(8000, seed=4)
        outer = trace(T0, whole)
        inner = trace(T0 + 2000 * DT, whole[2000:3000])
        data, _, rec = OP.stitch_window([inner, outer], **W, **window(8000))
        self.assertIsNone(rec["refusal"], rec)
        np.testing.assert_array_equal(data, whole.astype(np.float64))
        self.assertEqual(rec["identical_overlap_samples"], 1000)

    def test_fully_duplicate_record_contributes_nothing(self):
        whole = series(5000, seed=5)
        data, _, rec = OP.stitch_window([trace(T0, whole), trace(T0, whole.copy())], **W, **window(5000))
        self.assertIsNone(rec["refusal"], rec)
        self.assertEqual(data.size, 5000)
        self.assertEqual(rec["identical_overlap_samples"], 5000)

    def test_shorter_record_at_the_same_start_is_nested(self):
        whole = series(5000, seed=6)
        data, _, rec = OP.stitch_window([trace(T0, whole[:100]), trace(T0, whole)], **W, **window(5000))
        self.assertIsNone(rec["refusal"], rec)
        self.assertEqual(data.size, 5000)

    def test_three_records_with_an_identical_overlap_and_an_abut(self):
        whole = series(9000, seed=7)
        recs = [trace(T0, whole[:3000]), trace(T0 + 2900 * DT, whole[2900:6000]), trace(T0 + 6000 * DT, whole[6000:])]
        data, _, rec = OP.stitch_window(recs, **W, **window(9000))
        self.assertIsNone(rec["refusal"], rec)
        np.testing.assert_array_equal(data, whole.astype(np.float64))


class Refusals(unittest.TestCase):
    def refuses(self, records, n, kind, refusal="GAP_OR_OVERLAP", **extra):
        data, rate, rec = OP.stitch_window(records, **W, **window(n), **extra)
        self.assertIsNone(data)
        self.assertEqual((rate, rec["refusal"]), (0.0, refusal), rec)
        if kind is not None:
            self.assertEqual(rec["discontinuity"]["kind"], kind, rec)
        return rec

    def test_one_sample_disagreement_in_a_317_sample_overlap(self):
        whole = series(10000, seed=8)
        later = whole[6000 - 317:].copy()
        later[100] += 1                                            # one differing sample inside the overlap
        rec = self.refuses([trace(T0, whole[:6000]), trace(T0 + (6000 - 317) * DT, later)], 10000, "UNEQUAL_OVERLAP")
        self.assertEqual((rec["discontinuity"]["differing_samples"], rec["discontinuity"]["first_difference_index"]), (1, 100))
        self.assertEqual(rec["discontinuity"]["overlap_samples"], 317)

    def test_nested_record_with_one_differing_sample(self):
        whole = series(8000, seed=9)
        inner = whole[2000:3000].copy(); inner[-1] -= 7
        self.refuses([trace(T0, whole), trace(T0 + 2000 * DT, inner)], 8000, "UNEQUAL_OVERLAP")

    def test_true_gap_is_never_filled(self):
        whole = series(9000, seed=10)
        rec = self.refuses([trace(T0, whole[:3000]), trace(T0 + 3005 * DT, whole[3005:])], 9000, "GAP")
        self.assertEqual(rec["discontinuity"]["gap_samples"], 5)
        self.assertAlmostEqual(rec["discontinuity"]["gap_seconds"], 5 * DT)

    def test_off_grid_record_refuses(self):
        whole = series(9000, seed=11)
        rec = self.refuses([trace(T0, whole[:3000]), trace(T0 + (2900 + 0.4) * DT, whole[2900:])], 9000, "OFF_GRID")
        self.assertAlmostEqual(rec["discontinuity"]["offset_samples"], 0.4, places=4)

    def test_quarter_sample_jitter_is_still_on_grid(self):
        whole = series(6000, seed=12)
        data, _, rec = OP.stitch_window([trace(T0, whole[:3000]), trace(T0 + (3000 + 0.2) * DT, whole[3000:])], **W, **window(6000))
        self.assertIsNone(rec["refusal"], rec)                     # v1's abut tolerance (dt/4), unchanged

    def test_wrong_nslc_and_rate_refuse(self):
        whole = series(4000, seed=13)
        self.refuses([trace(T0, whole, sta="SNZX")], 4000, None, refusal="NSLC_MISMATCH")
        self.refuses([trace(T0, whole, loc="10")], 4000, None, refusal="LOCATION_MISMATCH")
        self.refuses([trace(T0, whole, chan="HHZ")], 4000, None, refusal="CHANNEL_MISMATCH")
        self.refuses([trace(T0, whole, rate=20.0)], 4000, None, refusal="RATE_MISMATCH")

    def test_masked_and_nonfinite_refuse(self):
        masked = np.ma.masked_array(series(4000, seed=14).astype(np.float64), mask=np.zeros(4000, bool)); masked.mask[7] = True
        self.refuses([trace(T0, masked)], 4000, "MASKED", refusal="MASKED_SAMPLES")
        values = series(4000, seed=15).astype(np.float64); values[9] = np.nan
        self.refuses([trace(T0, values)], 4000, "NONFINITE", refusal="MASKED_SAMPLES")

    def test_identical_overlap_across_a_retained_epoch_boundary_refuses(self):
        whole = series(10000, seed=16)
        boundary = (T0 + 5900 * DT)
        epochs = [("2018-03-21T00:00:00Z", str(boundary), "old"), (str(boundary), None, "new")]
        rec = self.refuses([trace(T0, whole[:6000]), trace(T0 + (6000 - 317) * DT, whole[6000 - 317:])], 10000,
                           "CROSSES_RESPONSE_EPOCH", response_epochs=epochs)
        self.assertEqual(rec["discontinuity"]["overlap_samples"], 317)
        data, _, ok = OP.stitch_window([trace(T0, whole[:6000]), trace(T0 + (6000 - 317) * DT, whole[6000 - 317:])],
                                       **W, **window(10000), response_epochs=[("2018-03-21T00:00:00Z", None, "only")])
        self.assertIsNone(ok["refusal"], ok)

    def test_uncovered_window_states_its_coverage(self):
        whole = series(4000, seed=17)
        rec = self.refuses([trace(T0, whole)], 10000, None, refusal="WINDOW_NOT_COVERED")
        self.assertEqual(rec["coverage"]["fraction"], 0.4)
        self.assertAlmostEqual(rec["coverage"]["missing_tail_seconds"], 6000 * DT)


class V1Identity(unittest.TestCase):
    """Everything v1 accepted joins byte-identically under v2 (the existing SNZO baselines stay the same quantity)."""

    @classmethod
    def setUpClass(cls):
        cls.v1 = load_v1()

    def setUp(self):
        if self.v1 is None:
            self.skipTest("public commit %s not reachable from this checkout" % V1_COMMIT[:8])

    def same(self, records, n):
        a = self.v1.stitch_window([tr.copy() for tr in records], **W, **window(n))
        b = OP.stitch_window([tr.copy() for tr in records], **W, **window(n))
        self.assertIsNone(a[2]["refusal"], a[2])
        self.assertIsNone(b[2]["refusal"], b[2])
        np.testing.assert_array_equal(a[0], b[0])
        self.assertEqual((a[1], a[2]["support_sha256"], a[2]["npts"], a[2]["start"], a[2]["end"]),
                         (b[1], b[2]["support_sha256"], b[2]["npts"], b[2]["start"], b[2]["end"]))

    def test_single_record(self):
        self.same([trace(T0 - 10 * DT, series(5000, seed=20))], 4000)

    def test_exact_abut(self):
        whole = series(8000, seed=21)
        self.same([trace(T0, whole[:4000]), trace(T0 + 4000 * DT, whole[4000:])], 8000)

    def test_one_identical_duplicate_sample(self):
        whole = series(8000, seed=22)
        self.same([trace(T0, whole[:4000]), trace(T0 + 3999 * DT, whole[3999:])], 7999)

    def test_v1_refused_the_317_sample_overlap_that_v2_admits(self):
        whole = series(10000, seed=23)
        records = [trace(T0, whole[:6000]), trace(T0 + (6000 - 317) * DT, whole[6000 - 317:])]
        _, _, rec = self.v1.stitch_window(records, **W, **window(10000))
        self.assertEqual(rec["refusal"], "GAP_OR_OVERLAP")


class Identity(unittest.TestCase):
    def test_the_operator_is_versioned(self):
        spec = OP.BOUND_STATIONS["IU.SNZO"]
        self.assertEqual(spec["operator_version"], "thd-bound-station-operator-v2")
        self.assertIn("IDENTICAL_OVERLAP", spec["gap_policy"])
        self.assertIn("NO_FILL", spec["gap_policy"])

    def test_the_identity_differs_from_v1(self):
        v1 = load_v1()
        if v1 is None:
            self.skipTest("public commit not reachable")
        self.assertNotEqual(OP.operator_identity("IU.SNZO"), v1.operator_identity("IU.SNZO"))
        self.assertNotEqual(OP.daily_operator_identity("IU.SNZO"), v1.daily_operator_identity("IU.SNZO"))


class Replay(unittest.TestCase):
    """The retained-bytes replay tool, on synthetic miniSEED written to a temporary directory (no provider I/O)."""

    def test_counts_recovered_refused_accepted_unknown(self):
        import json
        from obspy import Stream
        import thd_bound_operator_replay_cayley_20261010 as REPLAY
        if load_v1() is None:
            self.skipTest("public commit not reachable")
        n_day = int(RATE * 3600 * 25)
        streams = Stream()
        # scenarios ten days apart, so no 25 h window sees another scenario's records
        d1 = UTCDateTime("2026-07-02") - 25 * 3600                      # one clean record
        streams += trace(d1, series(n_day, seed=31))
        d2 = UTCDateTime("2026-07-12") - 25 * 3600                      # a 317-sample IDENTICAL overlap
        whole = series(n_day, seed=32)
        streams += trace(d2, whole[:n_day // 2]); streams += trace(d2 + (n_day // 2 - 317) * DT, whole[n_day // 2 - 317:])
        d3 = UTCDateTime("2026-07-22") - 25 * 3600                      # a true gap of 5 samples
        gap = series(n_day, seed=33)
        streams += trace(d3, gap[:n_day // 2]); streams += trace(d3 + (n_day // 2 + 5) * DT, gap[n_day // 2 + 5:])
        totals, rows = {}, {}
        with tempfile.TemporaryDirectory() as tmp:
            for i, tr in enumerate(streams):
                tr.data = tr.data.astype(np.int32)
                Stream([tr]).write(str(Path(tmp) / ("rec%02d.mseed" % i)), format="MSEED")
            for day in ("2026-07-02", "2026-07-12", "2026-07-22", "2026-08-01"):   # 08-01: nothing retained
                out = Path(tmp) / ("replay_%s.json" % day)
                self.assertEqual(REPLAY.main(["--retained", tmp, "--days", "%s:%s" % (day, day), "--out", str(out)]), 0)
                receipt = json.loads(out.read_text())
                for k, v in receipt["counts"].items():
                    totals[k] = totals.get(k, 0) + v
                rows[day] = receipt["rows"][0]
        self.assertEqual(totals, dict(RECOVERED=1, STILL_REFUSED=1, BOTH_ACCEPT=1, V2_ONLY_REFUSES=0, UNKNOWN=1,
                                      BOTH_ACCEPT_BYTES_DIFFER=0))
        self.assertEqual(rows["2026-07-12"]["v2_identical_overlap_samples"], 317)
        self.assertEqual(rows["2026-07-22"]["v2_discontinuity"]["kind"], "GAP")
        self.assertIs(rows["2026-07-02"]["support_identical"], True)
        self.assertEqual(rows["2026-08-01"]["outcome"], "UNKNOWN")

    def test_runner_cache_layout_two_fetch_seams_labelled(self):
        """grassmann e5a34e44: the runner's seismic_cache pickles, cache days D-2 and D-1 per scored day D. Midnight-start
        runs that abut exactly -> BOTH_ACCEPT; ~07:00Z runs whose seam repeats 317 identical samples -> RECOVERED and
        labelled not-the-issued-input; a missing cache day -> UNKNOWN (CACHE_DAY_MISSING)."""
        import json
        import pickle
        from obspy import Stream
        import thd_bound_operator_replay_cayley_20261010 as REPLAY
        if load_v1() is None:
            self.skipTest("public commit not reachable")
        day_n = int(RATE * 86400)

        def put(root, cache_day, start, data):
            d = Path(root) / "kaikoura" / cache_day.replace("-", "")
            d.mkdir(parents=True, exist_ok=True)
            (d / "hope_fault_waveforms.pkl").write_bytes(pickle.dumps({"IU.SNZO": Stream([trace(start, data)]),
                                                                        "IU.OTHER": Stream()}))
        with tempfile.TemporaryDirectory() as tmp:
            a = series(2 * day_n, seed=41)                                   # midnight runs, exact abut
            put(tmp, "2026-07-01", UTCDateTime("2026-07-01"), a[:day_n]); put(tmp, "2026-07-02", UTCDateTime("2026-07-02"), a[day_n:])
            b = series(2 * day_n, seed=42)                                   # 07:00Z runs, 317 identical samples at the seam
            s = UTCDateTime("2026-07-11T07:00:00")
            put(tmp, "2026-07-11", s, b[:day_n])
            put(tmp, "2026-07-12", s + (day_n - 317) * DT, b[day_n - 317:2 * day_n - 317])
            put(tmp, "2026-07-21", UTCDateTime("2026-07-21"), series(day_n, seed=43))   # 07-22 never retained
            out = Path(tmp) / "replay.json"
            rc = REPLAY.main(["--runner-cache", tmp, "--days", "2026-07-03:2026-07-23", "--out", str(out)])
            receipt = json.loads(out.read_text())
        self.assertEqual(rc, 0)
        rows = {r["day"]: r for r in receipt["rows"]}
        self.assertEqual(rows["2026-07-03"]["outcome"], "BOTH_ACCEPT")
        self.assertIs(rows["2026-07-03"]["support_identical"], True)
        self.assertEqual(rows["2026-07-03"]["issued_input"], "CACHE_RUNS_MIDNIGHT_START")
        self.assertEqual(rows["2026-07-13"]["outcome"], "RECOVERED")
        self.assertEqual(rows["2026-07-13"]["v2_identical_overlap_samples"], 317)
        self.assertEqual(rows["2026-07-13"]["issued_input"], "CACHE_RUN_NON_MIDNIGHT_START_NOT_THE_ISSUED_INPUT")
        self.assertEqual((rows["2026-07-23"]["outcome"], rows["2026-07-23"]["unknown_reason"]), ("UNKNOWN", "CACHE_DAY_MISSING"))
        for r in receipt["rows"]:
            self.assertEqual(r["input_kind"], "RUNNER_CACHE_TWO_INDEPENDENT_FETCHES")
        self.assertEqual(receipt["counts"], dict(RECOVERED=1, STILL_REFUSED=0, BOTH_ACCEPT=1, V2_ONLY_REFUSES=0, UNKNOWN=19,
                                                 BOTH_ACCEPT_BYTES_DIFFER=0))
        self.assertEqual(receipt["counts_by_issued_input"], {"CACHE_RUNS_MIDNIGHT_START": {"BOTH_ACCEPT": 1},
                                                             "CACHE_RUN_NON_MIDNIGHT_START_NOT_THE_ISSUED_INPUT": {"RECOVERED": 1}})


class ChannelPassThrough(unittest.TestCase):
    """grassmann e5a34e44 finding: the bound station's channel is bound too. A non-BHZ request refuses at the shared
    measurement (as for an unbound station) and the channel reaches the operator's fetch, where fetch_bound refuses it."""

    class _Analyzer:
        window_hours = 24

        def analyze_window(self, data, sample_rate, station, window_time):
            from unittest import mock
            return mock.Mock(thd_value=0.3, fundamental_power=1.0, dominant_frequency=2.2e-5, snr=5.0)

    def test_measure_refuses_another_channel_for_the_bound_station(self):
        import thd_daily_measurement as TDM
        calls = []
        with self.assertRaises(ValueError) as cm:
            TDM.measure("IU", "SNZO", datetime(2026, 10, 8), analyzer=self._Analyzer(),
                        fetch=lambda **kw: calls.append(kw) or (None, 0.0), channel="HHZ")
        self.assertIn("THD_MEASUREMENT_CHANNEL_NOT_BOUND", str(cm.exception))
        self.assertEqual(calls, [], "nothing is fetched for a refused channel")

    def test_the_bound_channel_is_passed_through(self):
        import thd_daily_measurement as TDM
        calls = []

        def fetch(**kw):
            calls.append(kw)
            return np.zeros(40 * 3600 * 13), 40.0
        rec = TDM.measure("IU", "SNZO", datetime(2026, 10, 8), analyzer=self._Analyzer(), fetch=fetch)
        self.assertEqual((calls[0]["channel"], rec["thd"]), ("BHZ", 0.3))

    def test_the_operator_forwards_a_named_channel_and_keeps_the_old_shape_otherwise(self):
        calls = []

        def fetch(**kw):
            calls.append(kw)
            return None, 0.0
        OP.daily_measurement("IU", "SNZO", datetime(2026, 10, 8), analyzer=self._Analyzer(), fetch=fetch, channel="HHZ")
        OP.daily_measurement("IU", "SNZO", datetime(2026, 10, 8), analyzer=self._Analyzer(), fetch=fetch)
        self.assertEqual(calls[0]["channel"], "HHZ")
        self.assertNotIn("channel", calls[1])

    def test_fetch_bound_refuses_a_forwarded_other_channel(self):
        data, rate, rec = OP.fetch_bound("IU", "SNZO", datetime(2026, 10, 6, 23), datetime(2026, 10, 8), "HHZ",
                                         client_factory=lambda name: self.fail("no client for a refused channel"))
        self.assertEqual((data, rate, rec["refusal"]), (None, 0.0, "CHANNEL_MISMATCH"))

if __name__ == "__main__":
    unittest.main()
