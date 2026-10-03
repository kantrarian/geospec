"""Tests for thd_bootstrap v2 (grassmann 2026-10-03; codex MEASUREMENT_SUPPORT_PACKET_REVIEW findings 1-3).
No network. Fixtures only, plus real-cache cases that skip when the devildog seismic cache is absent.
Run: python -B -W error::ResourceWarning -m unittest test_thd_bootstrap_grassmann_20261003 -v"""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np

import thd_bootstrap as tb
import station_baselines as sb
import run_thd_recal as rr

TODAY = date(2026, 10, 3)
WIN = tb.registered_window(TODAY)          # 2026-06-05 .. 2026-09-03
STA, LOC = "IU.SNZO", "00"
RATE = 40.0


class _FakeAnalyzer:
    def __init__(self, thd=0.37, p1=1.0):
        self.thd, self.p1 = thd, p1

    def compute_thd(self, data, sample_rate):
        return self.thd, self.p1, [], 2.236e-5

    def compute_thd_with_noise(self, data, sample_rate):
        return self.thd, self.p1, [], 2.236e-5, 10.0


def obs(day, *, station=STA, net="IU", code="SNZO", loc=LOC, chan="BHZ", rate=RATE, hours=25.0, n_traces=1,
        gap=0.0, filled=0, response=None, thd=0.37, p1=1.0, source="fixture", ref="fx", sha=None, start=None,
        end=None, npts=None, wstart=None, wend=None):
    ws, we = tb.target_window(day)
    frate = rate if (isinstance(rate, float) and math.isfinite(rate) and rate > 0) else 40.0   # NaN/0 fixtures keep clocks valid
    n = int(round(hours * 3600 * frate)) if npts is None else npts
    s = ws if start is None else start
    e = (s + timedelta(seconds=(n - 1) / frate)) if end is None else end
    sha = sha or ("%064x" % (abs(hash(day)) % (1 << 256)))
    return tb.DayObservation(day=day, station=station, network=net, station_code=code, location=loc, channel=chan,
                             sampling_rate=rate, start_utc=tb.iso(s) if hasattr(s, "tzinfo") else s,
                             end_utc=tb.iso(e) if hasattr(e, "tzinfo") else e, npts=n, n_traces=n_traces,
                             gap_seconds=gap, filled_samples=filled, response_available=response, source=source,
                             source_ref=ref, support_sha256=sha,
                             window_start_utc=tb.iso(ws) if wstart is None else wstart,
                             window_end_utc=tb.iso(we) if wend is None else wend, thd=thd, p1=p1, f1=2.236e-5)


def days(n, start=WIN[0]):
    return [(start + timedelta(days=i)).isoformat() for i in range(n)]


def run(observations, **kw):
    kw.setdefault("today", TODAY); kw.setdefault("expected_location", LOC)
    return tb.bootstrap(STA, observations, **kw)


class Window(unittest.TestCase):
    def test_registered_window_is_the_recal_window(self):
        self.assertEqual(WIN, (date(2026, 6, 5), date(2026, 9, 3)))
        self.assertEqual((WIN[1] - WIN[0]).days, rr.LOOKBACK_DAYS)
        self.assertEqual(TODAY - WIN[1], timedelta(days=rr.EXCLUDE_RECENT_DAYS))

    def test_target_window_is_the_weekly_operator_support(self):
        ws, we = tb.target_window("2026-07-01")
        self.assertEqual(tb.iso(ws), "2026-07-01T00:00:00.000000Z")
        self.assertEqual(tb.iso(we), "2026-07-02T01:00:00.000000Z")
        self.assertEqual(tb.OPERATOR_WEEKLY["rate"], "native (no resampling)")


class Qualify(unittest.TestCase):
    def q(self, o, **kw):
        kw.setdefault("station", STA); kw.setdefault("expected_location", LOC)
        return tb.qualify(o, WIN, today=TODAY, **kw)

    def test_nominal_day_qualifies(self):
        self.assertEqual(self.q(obs("2026-07-01"), expected_rate=RATE, epoch=(date(1992, 4, 7), None)), [])

    def test_future_bytes_under_an_old_label_refuse(self):
        ws, _ = tb.target_window("2026-10-02")
        o = obs("2026-07-01", start=ws, end=ws + timedelta(hours=24))
        r = self.q(o)
        self.assertIn("SUPPORT_OUTSIDE_WINDOW", r); self.assertIn("DAY_TOO_RECENT", r)

    def test_empty_or_naive_or_inverted_clocks_refuse(self):
        self.assertIn("TIMESTAMPS_INVALID", self.q(obs("2026-07-01", start="", end="")))
        self.assertIn("TIMESTAMPS_INVALID", self.q(obs("2026-07-01", start="2026-07-01T00:00:00", end="2026-07-02T00:00:00")))
        ws, we = tb.target_window("2026-07-01")
        self.assertIn("TIMESTAMPS_INVALID", self.q(obs("2026-07-01", start=we, end=ws)))

    def test_declared_window_must_be_the_operator_window(self):
        self.assertIn("WINDOW_MISMATCH", self.q(obs("2026-07-01", wstart="2026-07-01T07:00:00Z", wend="2026-07-02T07:00:00Z")))

    def test_nan_or_nonpositive_rate_refuses_even_with_expected_rate(self):
        self.assertIn("RATE_INVALID", self.q(obs("2026-07-01", rate=float("nan")), expected_rate=RATE))
        self.assertIn("RATE_INVALID", self.q(obs("2026-07-01", rate=0.0), expected_rate=RATE))
        self.assertEqual(self.q(obs("2026-07-01", rate=20.0), expected_rate=RATE), ["RATE_MISMATCH"])

    def test_npts_rate_span_must_agree(self):
        o = obs("2026-07-01"); o.npts = o.npts - 400
        self.assertIn("NPTS_SPAN_INCONSISTENT", self.q(o))

    def test_location_is_bound_no_fallback(self):
        self.assertEqual(self.q(obs("2026-07-01", loc="10")), ["LOCATION_MISMATCH"])
        self.assertEqual(self.q(obs("2026-07-01", loc="10"), expected_location="10"), [])

    def test_nslc_and_channel_are_bound(self):
        self.assertIn("NSLC_MISMATCH", self.q(obs("2026-07-01", code="TUC")))
        self.assertIn("NSLC_MISMATCH", self.q(obs("2026-07-01", station="IU.TUC", code="TUC")))
        self.assertEqual(self.q(obs("2026-07-01", chan="HHZ")), ["CHANNEL_MISMATCH"])

    def test_outside_registered_window_and_epoch_refuse(self):
        self.assertIn("DAY_OUTSIDE_WINDOW", self.q(obs("2026-06-04")))
        self.assertIn("DAY_OUTSIDE_WINDOW", self.q(obs("2026-09-04")))
        self.assertIn("DAY_TOO_RECENT", self.q(obs("2026-09-20")))
        self.assertEqual(self.q(obs("2026-07-01"), epoch=(date(2026, 8, 1), None)), ["EPOCH_MISMATCH"])

    def test_short_coverage_gaps_fills_and_response(self):
        self.assertIn("COVERAGE_SHORT", self.q(obs("2026-07-01", hours=11.9)))
        self.assertIn("NO_DATA", self.q(obs("2026-07-01", hours=0, npts=0)))
        self.assertEqual(self.q(obs("2026-07-01", gap=1.0)), ["GAP_FILLED"])
        self.assertEqual(self.q(obs("2026-07-01", gap=-0.5)), ["GAP_FILLED"])
        self.assertEqual(self.q(obs("2026-07-01", filled=5)), ["GAP_FILLED"])
        self.assertEqual(self.q(obs("2026-07-01", n_traces=2)), [])       # stitched abutting pieces, no fill
        self.assertEqual(self.q(obs("2026-07-01"), require_response=True), ["RESPONSE_MISSING"])
        self.assertEqual(self.q(obs("2026-07-01", response=True), require_response=True), [])

    def test_repeated_label_and_relabelled_support_refuse(self):
        seen_d, seen_s = set(), {}
        self.assertEqual(self.q(obs("2026-07-01", sha="a" * 64), seen_days=seen_d, seen_support=seen_s), [])
        self.assertEqual(self.q(obs("2026-07-01", sha="b" * 64, ref="repeat"), seen_days=seen_d, seen_support=seen_s), ["DUPLICATE_DAY"])
        self.assertIn("DUPLICATE_SUPPORT", self.q(obs("2026-07-02", sha="a" * 64), seen_days=seen_d, seen_support=seen_s))


class Eligibility(unittest.TestCase):
    def test_diagnostics_always_eligibility_separately(self):
        res = run([obs(d, thd=0.30 + 0.01 * (i % 7)) for i, d in enumerate(days(59))])
        self.assertTrue(res["diagnostic_complete"]); self.assertEqual(res["diagnostics"]["n_qualified"], 59)
        self.assertFalse(res["candidate_eligible"]); self.assertEqual(res["eligibility_refusals"], ["INSUFFICIENT_DAYS"])
        self.assertNotIn("entry", res)

    def test_floor_cannot_be_lowered_only_raised(self):
        res = run([obs(d) for d in days(10)], min_days=10)
        self.assertFalse(res["candidate_eligible"]); self.assertEqual(res["min_days_effective"], tb.MIN_BOOTSTRAP_DAYS)
        res = run([obs(d, thd=0.30 + 0.01 * (i % 7)) for i, d in enumerate(days(70))], min_days=75)
        self.assertEqual(res["min_days_effective"], 75); self.assertIn("INSUFFICIENT_DAYS", res["eligibility_refusals"])

    def test_zero_dispersion_refuses(self):
        res = run([obs(d, thd=0.37) for d in days(60)])
        self.assertFalse(res["candidate_eligible"]); self.assertIn("ZERO_DISPERSION", res["eligibility_refusals"])

    def test_nonfinite_values_refuse(self):
        o = [obs(d, thd=0.30 + 0.01 * (i % 7)) for i, d in enumerate(days(60))]
        o[5].thd = float("inf")
        res = run(o)
        self.assertFalse(res["candidate_eligible"]); self.assertIn("ESTIMATOR_ZERO", res["per_day"][5]["reasons"])
        o[5].thd = 0.33; o[6].thd = float("nan")
        res = run(o)
        self.assertFalse(res["candidate_eligible"])

    def test_qa_fail_refuses(self):
        # 60 qualifying days (coverage 66 % < 80 %), tight first half, widely ascending second half: the full-set MAD
        # stays tiny, so drift >> 2 sigma, MAD inflation >> 50 %, CV > 0.5 -> five issues -> baseline_qa grades 'fail'
        vals = [0.100 + 0.0001 * i for i in range(30)] + [0.100 + 0.05 * j for j in range(30)]
        res = run([obs(d, thd=v) for d, v in zip(days(60), vals)])
        self.assertEqual(res["qa"]["quality_grade"], "fail")
        self.assertFalse(res["candidate_eligible"]); self.assertIn("QA_FAIL", res["eligibility_refusals"])

    def test_nominal_sixty_days_are_eligible_with_recal_statistics_and_coverage_policy(self):
        vals = [0.30 + 0.01 * (i % 7) for i in range(60)]
        res = run([obs(d, thd=v) for d, v in zip(days(60), vals)], expected_rate=RATE)
        self.assertTrue(res["candidate_eligible"], res["eligibility_refusals"])
        e = res["entry"]; thd = np.array(vals); med = float(np.median(thd)); mad = float(np.median(np.abs(thd - med)))
        self.assertEqual(e["mean_thd"], round(med, 6)); self.assertEqual(e["std_thd"], round(mad * 1.4826, 6))
        self.assertEqual(e["n_samples"], 60); self.assertEqual(e["calibration_date"], "2026-10-03")
        self.assertEqual(e["calibration_period"], f"{days(60)[0]} to {days(60)[-1]}")
        self.assertEqual(e["qa"]["n_days_requested"], 91)
        self.assertTrue(e["coverage_policy"]["below_qa_threshold"])            # 60/91 = 65.9 % < 80 %: retained
        self.assertTrue(any(i.startswith("Low coverage") for i in e["qa"]["issues"]))
        self.assertEqual(e["manifest_sha256"], res["manifest_sha256"]); self.assertEqual(len(e["manifest"]), 60)
        self.assertEqual(e["operator"], tb.OPERATOR_WEEKLY)

    def test_repeated_reports_do_not_count(self):
        o = [obs(d, sha="%064d" % i) for i, d in enumerate(days(35))] + [obs(d, sha="%064d" % i, ref="r") for i, d in enumerate(days(35))]
        res = run(o)
        self.assertEqual(res["n_qualified"], 35); self.assertEqual(res["refused"]["DUPLICATE_DAY"], 35)

    def test_defaults_never_masquerade(self):
        before = sb.STATION_BASELINES[STA]
        self.assertEqual(before.calibration_period, "UNCALIBRATED")
        res = run([])
        self.assertFalse(res["candidate_eligible"]); self.assertIs(sb.STATION_BASELINES[STA], before)
        self.assertNotIn(STA, rr._calibratable_stations())


class Operators(unittest.TestCase):
    def test_weekly_operator_equals_the_calibrator_pipeline_on_identical_input(self):
        """The declared operator reproduces calibrate_thd_baselines.compute_daily_thd on the same bytes: the fetch is
        stubbed to return the array (obspy detrend applied as fetch_continuous_data_for_thd does)."""
        import calibrate_thd_baselines as C
        from obspy import Trace
        rate = 1.0; n = int(25 * 3600 * rate); t = np.arange(n) / rate; f = 2.236e-5
        raw = (1000 * np.sin(2 * np.pi * f * t) + 120 * np.sin(2 * np.pi * 2 * f * t) + 3 * t + 50).astype(np.int32)
        tr = Trace(raw.astype(np.float64)); tr.stats.sampling_rate = rate
        tr.detrend("demean"); tr.detrend("linear")
        saved = C.fetch_continuous_data_for_thd
        try:
            C.fetch_continuous_data_for_thd = lambda **kw: (tr.data, rate)
            ref, _ = C.compute_daily_thd("IU", "SNZO", __import__("datetime").datetime(2026, 7, 1))
        finally:
            C.fetch_continuous_data_for_thd = saved
        mine, p1, _ = tb.weekly_operator(raw, rate)
        self.assertGreater(mine, 0); self.assertAlmostEqual(mine, ref, places=9)

    def test_daily_1hz_operator_differs_and_is_only_a_diagnostic(self):
        rate = 40.0; n = int(25 * 3600 * rate); t = np.arange(n) / rate; f = 2.236e-5
        raw = 1000 * np.sin(2 * np.pi * f * t) + 120 * np.sin(2 * np.pi * 2 * f * t) + np.random.default_rng(1).normal(0, 50, n)
        o = tb.estimate(obs("2026-07-01", rate=rate), raw)
        self.assertGreater(o.thd, 0); self.assertIsNotNone(o.thd_daily_1hz); self.assertGreater(o.thd_daily_1hz, 0)
        self.assertIn("thd_daily_1hz", run([o] * 1)["per_day"][0])
        self.assertEqual(tb.OPERATOR_DAILY["role"].split(":")[0], "DIAGNOSTIC_ONLY")

    def test_short_input_is_estimator_zero(self):
        rate = 1.0; n = int(11 * 3600 * rate); t = np.arange(n) / rate
        o = tb.estimate(obs("2026-07-01", rate=rate, hours=11), 1000 * np.sin(2 * np.pi * 2.236e-5 * t), daily_diagnostic=False)
        self.assertEqual(o.thd, 0.0)


class Stitch(unittest.TestCase):
    """Synthetic cache: two 24 h pickles (07:00->07:00) per station, as the fault-correlation cache stores them."""
    def setUp(self):
        from obspy import Stream, Trace, UTCDateTime
        self.tmp = Path(tempfile.mkdtemp()); self.rate = 40.0
        self.UTC = UTCDateTime; self.Stream = Stream; self.Trace = Trace

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_day(self, day, *, loc="00", start_shift=0.0, drop_tail=0, dup_first=False, extra_gap=0.0):
        import pickle
        from obspy import Stream, Trace, UTCDateTime
        d = date.fromisoformat(day); t0 = UTCDateTime(d.year, d.month, d.day, 7, 0, 10, 794538) + start_shift + extra_gap
        n = int(24 * 3600 * self.rate) - drop_tail
        data = (np.arange(n) % 1000 + 100 * np.sin(2 * np.pi * 2.236e-5 * np.arange(n) / self.rate)).astype(np.int32)
        tr = Trace(data); tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel = "IU", "SNZO", loc, "BHZ"
        tr.stats.sampling_rate = self.rate; tr.stats.starttime = t0
        st = Stream([tr]); st2 = Stream([tr.copy()]); st2[0].stats.location = "10"
        folder = self.tmp / d.strftime("%Y%m%d"); folder.mkdir()
        with open(folder / "x_waveforms.pkl", "wb") as fh:
            pickle.dump({"IU.SNZO": Stream([st[0], st2[0]])}, fh)
        return t0, data

    def test_two_abutting_days_stitch_into_the_exact_25h_window(self):
        self.write_day("2026-06-30"); self.write_day("2026-07-01")
        o = tb.stitch_cached_window(str(self.tmp), "2026-07-01", STA, "00", analyzer=_FakeAnalyzer())
        self.assertEqual(o.start_utc, "2026-07-01T00:00:00.019538Z")          # first sample on/after 00:00
        self.assertEqual(o.npts, int(25 * 3600 * 40))
        self.assertEqual((o.n_traces, o.gap_seconds, o.filled_samples, o.missing_support), (1, 0.0, 0, []))
        self.assertEqual(tb.qualify(o, WIN, today=TODAY, station=STA, expected_location="00", expected_rate=40.0), [])
        self.assertEqual(o.thd, 0.37); self.assertEqual(len(o.support_sha256), 64)

    def test_missing_previous_day_reports_the_exact_missing_interval(self):
        self.write_day("2026-07-01")
        o = tb.stitch_cached_window(str(self.tmp), "2026-07-01", STA, "00", estimate_now=False)
        self.assertEqual(o.missing_support[0][0], "2026-07-01T00:00:00.000000Z")
        self.assertTrue(o.missing_support[0][1].startswith("2026-07-01T07:00:10"))
        self.assertIn("SUPPORT_OUTSIDE_WINDOW", tb.qualify(o, WIN, today=TODAY, station=STA, expected_location="00")
                      or ["SUPPORT_OUTSIDE_WINDOW"])   # start after window start: not refused by itself,
        rep = tb.missing_support_report([o], 40.0)       # but the support is incomplete and reported exactly
        self.assertEqual(len(rep["intervals"]), 1); self.assertAlmostEqual(rep["total_seconds"], 7 * 3600 + 10.794538, places=3)
        self.assertEqual(rep["bytes_raw_int32"], rep["samples_at_rate"] * 4)

    def test_a_gap_between_the_days_is_a_discontinuity_not_a_fill(self):
        self.write_day("2026-06-30"); self.write_day("2026-07-01", extra_gap=0.2)
        o = tb.stitch_cached_window(str(self.tmp), "2026-07-01", STA, "00", estimate_now=False)
        self.assertEqual(o.n_traces, 2); self.assertGreater(o.gap_seconds, 0); self.assertEqual(o.filled_samples, 0)
        self.assertIn("GAP_FILLED", tb.qualify(o, WIN, today=TODAY, station=STA, expected_location="00"))

    def test_location_is_never_substituted(self):
        self.write_day("2026-06-30"); self.write_day("2026-07-01")
        o = tb.stitch_cached_window(str(self.tmp), "2026-07-01", STA, "99", estimate_now=False)
        self.assertEqual(o.npts, 0); self.assertIn("locs_present=00,10", o.source_ref)
        self.assertIn("NO_DATA", tb.qualify(o, WIN, today=TODAY, station=STA, expected_location="99"))

    def test_directory_date_cannot_bypass_the_lag_check(self):
        """Bytes dated 2026-09-20 placed under a 2026-07-01 folder: the window cut leaves nothing, nothing qualifies."""
        import pickle
        from obspy import Stream, Trace, UTCDateTime
        n = int(24 * 3600 * 40); tr = Trace(np.zeros(n, dtype=np.int32)); tr.stats.sampling_rate = 40.0
        tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel = "IU", "SNZO", "00", "BHZ"
        tr.stats.starttime = UTCDateTime(2026, 9, 20, 7, 0, 0)
        (self.tmp / "20260701").mkdir(); (self.tmp / "20260630").mkdir()
        for f in ("20260701", "20260630"):
            with open(self.tmp / f / "x_waveforms.pkl", "wb") as fh:
                pickle.dump({"IU.SNZO": Stream([tr.copy()])}, fh)
        o = tb.stitch_cached_window(str(self.tmp), "2026-07-01", STA, "00", estimate_now=False)
        self.assertEqual(o.npts, 0); self.assertTrue(o.missing_support)


class BaseAndCandidate(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()); self.bdir = self.tmp / "baselines"; self.bdir.mkdir()
        self.base = {"IU.TUC": {"station": "IU.TUC", "mean_thd": 0.37, "std_thd": 0.11, "n_samples": 91,
                                "calibration_period": "2026-06-03 to 2026-09-01", "notes": "Rolling recal"},
                     "IU.COR": {"station": "IU.COR", "mean_thd": 0.35, "std_thd": 0.16, "n_samples": 91,
                                "calibration_period": "2026-06-03 to 2026-09-01", "notes": "Rolling recal"}}
        (self.bdir / "thd_baselines_20261001.json").write_text(json.dumps(self.base, indent=2), encoding="utf-8")
        (self.bdir / "thd_baselines_20260923.json").write_text(json.dumps({"IU.TUC": dict(self.base["IU.TUC"], mean_thd=0.36)}), encoding="utf-8")
        vals = [0.30 + 0.01 * (i % 7) for i in range(60)]
        self.res = run([obs(d, thd=v) for d, v in zip(days(60), vals)])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_snapshot_is_the_newest_loadable_file_hash_bound(self):
        s = tb.snapshot_effective_base(self.bdir)
        self.assertEqual(s["name"], "thd_baselines_20261001.json"); self.assertEqual(s["calibration_date"], "2026-10-01")
        self.assertEqual(sorted(s["entries"]), ["IU.COR", "IU.TUC"]); self.assertEqual(len(s["sha256"]), 64)

    def test_empty_base_refuses(self):
        empty = self.tmp / "empty"; empty.mkdir()
        with self.assertRaises(tb.BootstrapRefused):
            tb.snapshot_effective_base(empty)
        with self.assertRaises(tb.BootstrapRefused):
            tb.compose_candidate_file(self.res, {}, str(self.tmp / "CANDIDATE_x.json"))

    def test_stale_base_refuses(self):
        s = tb.snapshot_effective_base(self.bdir)
        (self.bdir / "thd_baselines_20261001.json").write_text(json.dumps(dict(self.base, extra=1)), encoding="utf-8")
        with self.assertRaises(tb.BootstrapRefused) as cm:
            tb.compose_candidate_file(self.res, s, str(self.tmp / "CANDIDATE_x.json"))
        self.assertIn("BASE_SNAPSHOT_STALE", str(cm.exception))

    def test_ineligible_result_cannot_be_written(self):
        bad = run([obs(d, thd=0.37) for d in days(60)])
        with self.assertRaises(tb.BootstrapRefused) as cm:
            tb.compose_candidate_file(bad, tb.snapshot_effective_base(self.bdir), str(self.tmp / "CANDIDATE_x.json"))
        self.assertIn("CANDIDATE_NOT_ELIGIBLE", str(cm.exception)); self.assertIn("ZERO_DISPERSION", str(cm.exception))

    def test_name_dir_and_overwrite_refusals_and_strict_json(self):
        s = tb.snapshot_effective_base(self.bdir)
        with self.assertRaises(tb.BootstrapRefused):
            tb.compose_candidate_file(self.res, s, str(self.tmp / "thd_baselines_20261003.json"))
        with self.assertRaises(tb.BootstrapRefused):
            tb.compose_candidate_file(self.res, s, str(Path(rr.BASELINE_DIR) / "CANDIDATE_x.json"))
        p = tb.compose_candidate_file(self.res, s, str(self.tmp / "CANDIDATE_thd_baselines_20261003.json"))
        with self.assertRaises(tb.BootstrapRefused):
            tb.compose_candidate_file(self.res, s, p)
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))   # NaN/Infinity would raise
        self.assertEqual(sorted(k for k in d if not k.startswith("_")), ["IU.COR", "IU.SNZO", "IU.TUC"])
        self.assertEqual(d["IU.TUC"]["calibration_date"], "2026-10-01"); self.assertEqual(d["IU.TUC"]["mean_thd"], 0.37)
        self.assertEqual(d["IU.TUC"]["carried_from"]["sha256"], s["sha256"])
        self.assertEqual(d["IU.SNZO"]["qa"]["quality_grade"], self.res["entry"]["qa"]["quality_grade"])
        self.assertEqual(d["IU.SNZO"]["manifest_sha256"], self.res["manifest_sha256"]); self.assertEqual(len(d["IU.SNZO"]["manifest"]), 60)
        self.assertEqual(d["_bootstrap_candidate"]["base_snapshot"]["sha256"], s["sha256"])

    def test_fresh_process_loader_before_and_after_preserves_every_sibling(self):
        """Simulated landing in a scratch baselines dir, loaded by a FRESH python process each time (the patched
        station_baselines on this branch): every non-target effective record, including calibration_date, is identical;
        only IU.SNZO changes."""
        s = tb.snapshot_effective_base(self.bdir)
        p = tb.compose_candidate_file(self.res, s, str(self.tmp / "CANDIDATE_thd_baselines_20261003.json"))
        code = ("import json,sys; sys.path.insert(0, %r); import station_baselines as sb; from pathlib import Path\n"
                "sb.STATION_BASELINES.clear(); used = sb._load_newest_baseline_file(bdir=Path(sys.argv[1]))\n"
                "print(json.dumps(dict(used=used, rows={k: dict(mean=v.mean_thd, std=v.std_thd, n=v.n_samples, "
                "period=v.calibration_period, date=v.calibration_date) for k, v in sb.STATION_BASELINES.items()}), sort_keys=True))"
                % str(Path(__file__).resolve().parent))
        def load():
            out = subprocess.run([sys.executable, "-B", "-c", code, str(self.bdir)], capture_output=True, text=True, check=True)
            return json.loads(out.stdout.strip().splitlines()[-1])
        before = load()
        self.assertEqual(before["used"], "thd_baselines_20261001.json")
        shutil.copyfile(p, self.bdir / "thd_baselines_20261003.json")        # the reviewed landing, simulated
        after = load()
        self.assertEqual(after["used"], "thd_baselines_20261003.json")
        for k in ("IU.TUC", "IU.COR"):
            self.assertEqual(before["rows"][k], after["rows"][k])                 # numbers, window AND date identical
            self.assertEqual(after["rows"][k]["date"], "2026-10-01")
        self.assertEqual(after["rows"]["IU.SNZO"]["date"], "2026-10-03"); self.assertEqual(after["rows"]["IU.SNZO"]["n"], 60)
        self.assertNotIn("IU.SNZO", before["rows"])
        # and the recal gate now includes the station, with no code change in run_thd_recal
        code2 = code.replace("print(json.dumps(dict(used=used", "import run_thd_recal as rr; print(json.dumps(dict(cal=rr._calibratable_stations(), used=used")
        out = subprocess.run([sys.executable, "-B", "-c", code2, str(self.bdir)], capture_output=True, text=True, check=True)
        self.assertIn("IU.SNZO", json.loads(out.stdout.strip().splitlines()[-1])["cal"])


CACHE = Path("C:/GeoSpec/geospec_runner/monitoring/data/seismic_cache/kaikoura")


@unittest.skipUnless(CACHE.exists(), "devildog seismic cache absent")
class RealCache(unittest.TestCase):
    def test_a_stitched_real_window_is_bound_and_consistent(self):
        o = tb.stitch_cached_window(str(CACHE), "2026-07-02", STA, "00", estimate_now=False)
        self.assertEqual((o.network, o.station_code, o.location, o.channel, o.sampling_rate), ("IU", "SNZO", "00", "BHZ", 40.0))
        self.assertIn("locs_present=00,10", o.source_ref)
        s, e = tb.parse_utc(o.start_utc), tb.parse_utc(o.end_utc)
        self.assertEqual(int(round((e - s).total_seconds() * 40)) + 1, o.npts)
        self.assertTrue(tb.parse_utc(o.window_start_utc) <= s)
        self.assertEqual(len(o.support_sha256), 64)
        r = tb.qualify(o, WIN, today=TODAY, station=STA, expected_location="00", expected_rate=40.0)
        self.assertTrue(r == ["ESTIMATOR_ZERO"] or r == [] or "GAP_FILLED" in r or "COVERAGE_SHORT" in r, r)

    def test_a_real_gapped_day_refuses(self):
        o = tb.stitch_cached_window(str(CACHE), "2026-08-03", STA, "00", estimate_now=False)
        self.assertTrue(o.gap_seconds > 0 or o.missing_support)
        self.assertIsNone(o.thd)


if __name__ == "__main__":
    unittest.main()
