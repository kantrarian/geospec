"""thd-daily-measurement-v1 (cayley 2026-10-08; codex 1614 finding 1). Needs obspy (1.5.1 pinned env).

The daily scorer (ensemble.compute_thd_risk) and the weekly calibration (calibrate_thd_baselines.compute_daily_thd) are
fed the SAME synthetic waveform through the REAL fetch (obspy's FDSN Client replaced by a client that trims a retained-
style stream to the requested bounds and records every request). Acceptance: identical request bounds and selector,
the same selected trace, the same estimator rate and the same raw THD -- with missing and partial data, across a
response-epoch change and for a station without its own baseline -- and the daily value equal to the pre-change inline
computation. Waveforms are SYNTHETIC. Nothing here touches a network, reads a credential or activates anything.
"""
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import numpy as np
from obspy import Stream, Trace, UTCDateTime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import calibrate_thd_baselines as C  # noqa: E402
import ensemble as E  # noqa: E402
import thd_bound_station_operator as OP  # noqa: E402
import thd_daily_measurement as TDM  # noqa: E402
import thd_provider_routing as TPR  # noqa: E402
from seismic_thd import SeismicTHDAnalyzer  # noqa: E402

DAY = datetime(2026, 8, 1)
RATE = 40.0


def waveform(net, sta, loc, hours=76, start="2026-07-30T00:00:00", seed=1614, rate=RATE):
    """Deterministic int32 counts: a semidiurnal tide with harmonics plus noise, so THD is finite and non-trivial."""
    n = int(rate * 3600 * hours)
    t = np.arange(n) / rate
    f0 = 2.2365360529611736e-05
    signal = 4e5 * np.sin(2 * np.pi * f0 * t) + 4e4 * np.sin(4 * np.pi * f0 * t) + 1e4 * np.sin(6 * np.pi * f0 * t)
    noise = np.random.default_rng(seed).normal(size=n) * 2e3
    tr = Trace((signal + noise).astype(np.int32))
    tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel = net, sta, loc, "BHZ"
    tr.stats.sampling_rate, tr.stats.starttime = rate, UTCDateTime(start)
    return tr


class TrimmingClient:
    """Stands in for obspy.clients.fdsn.Client: returns STREAM trimmed to the requested bounds (as a provider would)
    and records each request's provider, selector and bounds."""
    STREAM = Stream()
    asked = []

    def __init__(self, name, timeout=None):
        self.name = name

    def get_waveforms(self, network, station, location, channel, starttime, endtime):
        TrimmingClient.asked.append((self.name, network, station, location, channel, str(starttime), str(endtime)))
        st = Stream([tr.copy() for tr in TrimmingClient.STREAM
                     if (tr.stats.network, tr.stats.station, tr.stats.channel) == (network, station, channel)
                     and location in ("*", tr.stats.location)])
        return st.trim(starttime, endtime, nearest_sample=False)


class BothCallersHarness(unittest.TestCase):
    """The trimming client and a spy on the shared measurement; holds no tests itself."""

    def setUp(self):
        TrimmingClient.STREAM, TrimmingClient.asked = Stream(), []
        patcher = mock.patch("obspy.clients.fdsn.Client", TrimmingClient)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.records = []
        real = TDM.measure

        def spy(*args, **kwargs):
            rec = real(*args, **kwargs)
            self.records.append(rec)
            return rec
        patcher = mock.patch.object(TDM, "measure", side_effect=spy)
        patcher.start()
        self.addCleanup(patcher.stop)

    def both(self, net, sta, day=DAY):
        """(daily MethodResult, its station attempts, weekly (value, rate), the two measurement records, the requests)"""
        runner = E.GeoSpecEnsemble(region="ridgecrest", eligibility_rule_active=False, record_thd_attempts=True)
        daily = runner.compute_thd_risk(day, station_network=net, station_code=sta)
        daily_asked, attempts = list(TrimmingClient.asked), runner.last_thd_fetch_attempts
        TrimmingClient.asked = []
        weekly = C.compute_daily_thd(net, sta, day)
        self.assertEqual(len(self.records), 2, "both callers went through the shared measurement")
        return daily, attempts, weekly, self.records, (daily_asked, list(TrimmingClient.asked))

    SAME = ("requested_window", "native_rate_hz", "estimator_rate_hz", "n_native_samples", "n_processed_samples", "thd",
            "measurement_identity")


class SameWaveformBothCallers(BothCallersHarness):
    def test_identical_bounds_selector_trace_rate_and_raw_thd(self):
        TrimmingClient.STREAM = Stream([waveform("IU", "TUC", "00"), waveform("IU", "TUC", "10", seed=7)])
        daily, attempts, (weekly, weekly_rate), (a, b), (asked_daily, asked_weekly) = self.both("IU", "TUC")
        self.assertEqual(asked_daily, asked_weekly)
        self.assertEqual(asked_daily, [("IRIS", "IU", "TUC", "00", "BHZ", "2026-07-30T23:00:00.000000Z",
                                        "2026-08-01T00:00:00.000000Z")])
        self.assertEqual({k: a[k] for k in self.SAME}, {k: b[k] for k in self.SAME})
        self.assertEqual((a["estimator_rate_hz"], a["native_rate_hz"]), (1.0, RATE))
        self.assertEqual(attempts[0]["trace_id"], "IU.TUC.00.BHZ")
        self.assertTrue(daily.available)
        self.assertEqual((daily.raw_value, weekly, weekly_rate), (a["thd"], a["thd"], RATE))

    def test_the_daily_value_equals_the_pre_change_inline_computation(self):
        TrimmingClient.STREAM = Stream([waveform("IU", "TUC", "00")])
        daily, _, _, (a, _), _ = self.both("IU", "TUC")
        # the unbound daily path as it stood before thd-daily-measurement-v1 (ensemble.compute_thd_risk, 823a84bc)
        import seismic_thd as ST
        from math import gcd
        from scipy.signal import resample_poly
        data, rate = ST.fetch_continuous_data_for_thd("IU", "TUC", DAY - timedelta(hours=25), DAY)
        up, down = int(1.0 * 100), int(rate * 100)
        common = gcd(up, down)
        data = resample_poly(data, up // common, down // common)
        legacy = SeismicTHDAnalyzer(window_hours=24).analyze_window(data=data, sample_rate=1.0, station="IU.TUC",
                                                                     window_time=DAY)
        self.assertEqual(daily.raw_value, legacy.thd_value)
        self.assertEqual(a["thd"], legacy.thd_value)

    def test_missing_data_is_unavailable_in_both(self):
        TrimmingClient.STREAM = Stream([waveform("IU", "TUC", "00", hours=6, start="2026-07-31T18:00:00")])
        daily, _, (weekly, _), (a, b), _ = self.both("IU", "TUC")
        self.assertFalse(daily.available)
        self.assertIsNone(weekly)
        self.assertEqual((a["reason"], b["reason"]), ("INSUFFICIENT_DATA", "INSUFFICIENT_DATA"))

    def test_partial_data_above_the_floor_is_the_same_value_in_both(self):
        TrimmingClient.STREAM = Stream([waveform("IU", "TUC", "00", hours=13, start="2026-07-31T11:00:00")])
        daily, attempts, (weekly, _), (a, b), _ = self.both("IU", "TUC")
        self.assertEqual({k: a[k] for k in self.SAME}, {k: b[k] for k in self.SAME})
        self.assertEqual((daily.raw_value, weekly), (a["thd"], a["thd"]))
        self.assertEqual(attempts[0]["coverage"]["status"], "PARTIAL")

    def test_a_response_epoch_change_inside_the_window_is_measured_identically_and_named(self):
        TrimmingClient.STREAM = Stream([waveform("IU", "COLA", "00")])   # COLA.00 changed epoch 2026-07-31T02:00Z
        daily, attempts, (weekly, _), (a, b), _ = self.both("IU", "COLA")
        self.assertEqual({k: a[k] for k in self.SAME}, {k: b[k] for k in self.SAME})
        self.assertEqual((daily.raw_value, weekly), (a["thd"], a["thd"]))
        self.assertEqual(attempts[0]["response_epoch"]["status"], "CROSSES_EPOCH_BOUNDARY")

    def test_a_station_without_its_own_baseline_is_measured_identically(self):
        TrimmingClient.STREAM = Stream([waveform("AK", "BMR", "")])
        daily, _, (weekly, _), (a, b), (asked_daily, asked_weekly) = self.both("AK", "BMR")
        self.assertIsNone(__import__("station_baselines").get_baseline("BMR", "AK"))
        self.assertEqual(asked_daily, asked_weekly)
        self.assertEqual(asked_daily[0][3], "*", "unpinned: the wildcard, recorded as such")
        self.assertEqual({k: a[k] for k in self.SAME}, {k: b[k] for k in self.SAME})
        self.assertEqual((daily.raw_value, weekly), (a["thd"], a["thd"]))
        self.assertEqual(daily.baseline_quality, "missing")


class Identity(unittest.TestCase):
    def analyzer(self, **kw):
        return SeismicTHDAnalyzer(**dict(dict(n_harmonics=5, freq_tolerance=0.1, window_hours=24), **kw))

    def test_the_identity_is_stable_and_binds_selector_route_and_estimator(self):
        base = TDM.measurement_record("IU", "TUC", self.analyzer())
        self.assertEqual(base["identity"], TDM.measurement_record("IU", "TUC", self.analyzer())["identity"])
        self.assertEqual((base["selector"]["location"], base["measurement"]["version"]), ("00", "thd-daily-measurement-v1"))
        variants = {
            "pin": lambda: mock.patch.dict(TPR.STATION_LOCATIONS, {"IU.TUC": {"location": "10", "basis": "RETAINED: x"}}),
            "route": lambda: mock.patch.dict(TPR.NETWORK_ROUTES, {"IU": {"adapters": ((TPR.FDSN, "GEOFON"),),
                                                                         "basis": "x"}}),
            "resampling": lambda: mock.patch.dict(TDM.MEASUREMENT, {"resampling": "x"}),
        }
        for name, patch in variants.items():
            with self.subTest(name=name), patch():
                self.assertNotEqual(TDM.measurement_record("IU", "TUC", self.analyzer())["identity"], base["identity"])
        self.assertNotEqual(TDM.measurement_record("IU", "TUC", self.analyzer(n_harmonics=4))["identity"], base["identity"])
        self.assertNotEqual(TDM.measurement_record("IU", "COR", self.analyzer())["identity"], base["identity"])

    def test_a_bound_station_keeps_its_reviewed_operator_identity(self):
        rec = TDM.measurement_record("IU", "SNZO", self.analyzer())
        self.assertEqual((rec["identity"], rec["delegated_to"]),
                         (OP.daily_operator_identity("IU.SNZO"), "thd_bound_station_operator.daily_measurement"))
        with mock.patch.object(OP, "daily_measurement", return_value={"thd": None, "operator_identity": "op"}) as dm:
            out = TDM.measure("IU", "SNZO", DAY, analyzer=self.analyzer(), fetch=None)
        self.assertEqual((dm.call_count, out["measurement_identity"]), (1, "op"))

    def test_another_channel_is_refused_not_measured_under_this_identity(self):
        with self.assertRaises(ValueError):
            TDM.measure("IU", "TUC", DAY, analyzer=self.analyzer(), fetch=None, channel="LHZ")
        self.assertEqual(C.compute_daily_thd("IU", "TUC", DAY, "LHZ"), (None, None))

    def test_an_identity_failure_never_blocks_the_measurement(self):
        calls = []

        def fetch(**kw):
            calls.append(kw)
            return None, 0.0
        with mock.patch.object(TDM, "measurement_record", side_effect=RuntimeError("synthetic")):
            out = TDM.measure("IU", "TUC", DAY, analyzer=self.analyzer(), fetch=fetch)
        self.assertEqual((out["measurement_identity"], out["reason"], len(calls)), ("UNIDENTIFIED", "INSUFFICIENT_DATA", 1))


class RecalStampsTheMeasurement(unittest.TestCase):
    def test_each_recalibrated_entry_carries_the_measurement_record(self):
        import run_thd_recal as R
        moments = {"mean_thd": 0.5, "std_thd": 0.1, "n_samples": 60, "calibration_period": "2026-06-01 to 2026-08-30"}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(R, "BASELINE_DIR", Path(tmp)), \
                mock.patch.object(C, "calibrate_station", return_value=moments), \
                mock.patch.object(R, "_prior_effective_entries", return_value={}):
            path = R.run_recal(["IU.TUC", "IU.SNZO"], end_date=datetime(2026, 10, 1))
            written = json.loads(Path(path).read_text())
        analyzer = SeismicTHDAnalyzer(n_harmonics=5, freq_tolerance=0.1, window_hours=24)
        self.assertEqual(written["IU.TUC"]["measurement"]["identity"],
                         TDM.measurement_record("IU", "TUC", analyzer)["identity"])
        self.assertEqual(written["IU.SNZO"]["measurement"]["identity"], OP.daily_operator_identity("IU.SNZO"))
        self.assertEqual(written["IU.SNZO"]["operator"]["identity"], OP.operator_identity("IU.SNZO"))


class CalibrationReceipts(BothCallersHarness):
    """grassmann 641c01b7: every calibration day is written with the input it was computed from, so a recal can be
    re-run and checked; the digest is of the samples AS RETURNED, before the merge changes them."""

    def test_a_calibration_day_names_its_input_and_matches_the_daily_attempt(self):
        TrimmingClient.STREAM = Stream([waveform("IU", "TUC", "00"), waveform("IU", "TUC", "10", seed=7)])
        daily, attempts, (weekly, _), _, _ = self.both("IU", "TUC")
        receipt = C.compute_daily_record("IU", "TUC", DAY)
        self.assertEqual((receipt["day"], receipt["thd"], receipt["reason"]), ("2026-08-01", weekly, None))
        self.assertEqual((receipt["provider"], receipt["trace_id"], receipt["coverage_status"], receipt["response_epoch"]),
                         ("IRIS", "IU.TUC.00.BHZ", "FULL", "SINGLE_EPOCH"))
        self.assertEqual(receipt["raw_samples_sha256"], attempts[0]["raw_samples_sha256"])
        returned = TrimmingClient.STREAM.select(location="00").copy().trim(
            UTCDateTime(DAY - timedelta(hours=25)), UTCDateTime(DAY), nearest_sample=False)
        self.assertEqual(receipt["raw_samples_sha256"], TPR.raw_samples_digest(returned))
        self.assertEqual(receipt["measurement_identity"], self.records[0]["measurement_identity"])

    def test_a_day_without_a_value_keeps_its_reason_and_its_input(self):
        TrimmingClient.STREAM = Stream([waveform("IU", "TUC", "00", hours=6, start="2026-07-31T18:00:00")])
        receipt = C.compute_daily_record("IU", "TUC", DAY)
        self.assertEqual((receipt["thd"], receipt["reason"], receipt["coverage_status"]),
                         (None, "INSUFFICIENT_DATA", "PARTIAL"))
        self.assertIsNotNone(receipt["raw_samples_sha256"])
        TrimmingClient.STREAM = Stream()
        empty = C.compute_daily_record("IU", "TUC", DAY)
        self.assertEqual((empty["thd"], empty["reason"], empty["providers_tried"]),
                         (None, "INSUFFICIENT_DATA", ["IRIS:NO_TRACES"]))


class RawSamplesDigest(unittest.TestCase):
    def test_deterministic_order_free_and_sensitive_to_one_sample_and_a_mask(self):
        a, b = waveform("IU", "TUC", "00", hours=1), waveform("IU", "TUC", "10", hours=1, seed=7)
        base = TPR.raw_samples_digest(Stream([a, b]))
        self.assertEqual(base, TPR.raw_samples_digest(Stream([b.copy(), a.copy()])))
        changed = a.copy()
        changed.data[100] += 1
        self.assertNotEqual(TPR.raw_samples_digest(Stream([changed, b])), base)
        masked = a.copy()
        masked.data = np.ma.masked_array(masked.data, mask=np.zeros(masked.data.size, dtype=bool))
        masked.data.mask[5] = True
        self.assertNotEqual(TPR.raw_samples_digest(Stream([masked, b])), base)


class ReceiptsReachTheBaselineFile(unittest.TestCase):
    def test_calibrate_station_returns_one_receipt_per_attempted_day(self):
        days = []

        def record(network, station, date, channel="BHZ"):
            day = date.strftime("%Y-%m-%d")
            days.append(day)
            ok = date.day % 4 != 0
            return {"day": day, "thd": 0.5 + date.day / 100 if ok else None, "native_rate_hz": 40.0 if ok else None,
                    "reason": None if ok else "INSUFFICIENT_DATA"}
        with mock.patch.object(C, "compute_daily_record", side_effect=record):
            r = C.calibrate_station("IU", "TUC", days_back=20, exclude_recent_days=0, end_date=datetime(2026, 8, 30))
        self.assertEqual([x["day"] for x in r["daily_receipts"]], days)
        self.assertEqual(len(r["daily_receipts"]), r["n_attempted"])
        self.assertEqual(r["n_samples"], sum(1 for x in r["daily_receipts"] if x["thd"] is not None))
        self.assertEqual(r["daily_values"], [(x["day"], x["thd"]) for x in r["daily_receipts"] if x["thd"] is not None])

    def test_the_recal_writes_every_calibration_day_into_the_entry(self):
        import run_thd_recal as R
        receipts = [{"day": "2026-08-01", "thd": 0.5, "native_rate_hz": 40.0, "reason": None,
                     "raw_samples_sha256": "a" * 64}, {"day": "2026-08-02", "thd": None, "reason": "INSUFFICIENT_DATA"}]
        moments = {"mean_thd": 0.5, "std_thd": 0.1, "n_samples": 60, "calibration_period": "2026-06-01 to 2026-08-30",
                   "daily_receipts": receipts}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(R, "BASELINE_DIR", Path(tmp)), \
                mock.patch.object(C, "calibrate_station", return_value=moments), \
                mock.patch.object(R, "_prior_effective_entries", return_value={}):
            written = json.loads(Path(R.run_recal(["IU.TUC"], end_date=datetime(2026, 10, 1))).read_text())
        self.assertEqual(written["IU.TUC"]["calibration_days"], receipts)


if __name__ == "__main__":
    unittest.main()
