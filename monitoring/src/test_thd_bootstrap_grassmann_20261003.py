"""Tests for thd_bootstrap (grassmann 2026-10-03). No network; fixtures only, plus one real-cache test that skips
when the devildog seismic cache is absent. Run: python -m unittest test_thd_bootstrap_grassmann_20261003 -v"""
import json
import os
import shutil
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np

import thd_bootstrap as tb
import station_baselines as sb
import run_thd_recal as rr

TODAY = date(2026, 10, 3)
WIN = tb.registered_window(TODAY)          # 2026-06-05 .. 2026-09-03
STA = "IU.SNZO"


class _FakeAnalyzer:
    """Returns a fixed THD for any input: tests of qualification must not depend on the estimator."""
    def __init__(self, thd=0.37, p1=1.0):
        self.thd, self.p1 = thd, p1

    def compute_thd(self, data, sample_rate):
        return self.thd, self.p1, [], 2.236e-5


def obs(day, *, station=STA, loc="00", chan="BHZ", rate=40.0, hours=24.0, n_traces=1, gap=0.0, filled=0,
        response=None, thd=0.37, p1=1.0, source="fixture", ref="fx"):
    npts = int(hours * 3600 * rate)
    o = tb.DayObservation(day=day, station=station, location=loc, channel=chan, sampling_rate=rate,
                          start_utc=day + "T00:00:00Z", end_utc=day + "T23:59:59Z", npts=npts, n_traces=n_traces,
                          gap_seconds=gap, filled_samples=filled, response_available=response, source=source,
                          source_ref=ref, thd=thd, p1=p1, f1=2.236e-5)
    return o


def days(n, start=WIN[0]):
    return [(start + timedelta(days=i)).isoformat() for i in range(n)]


class Window(unittest.TestCase):
    def test_registered_window_is_the_recal_window(self):
        self.assertEqual(WIN, (date(2026, 6, 5), date(2026, 9, 3)))
        self.assertEqual((WIN[1] - WIN[0]).days, rr.LOOKBACK_DAYS)
        self.assertEqual(TODAY - WIN[1], timedelta(days=rr.EXCLUDE_RECENT_DAYS))


class Qualify(unittest.TestCase):
    def q(self, o, **kw):
        return tb.qualify(o, WIN, today=TODAY, **kw)

    def test_contiguous_in_window_day_qualifies(self):
        self.assertEqual(self.q(obs("2026-07-01")), [])

    def test_outside_window_refuses(self):
        self.assertIn("DAY_OUTSIDE_WINDOW", self.q(obs("2026-06-04")))
        self.assertIn("DAY_OUTSIDE_WINDOW", self.q(obs("2026-09-04")))

    def test_too_recent_refuses_by_name(self):
        r = self.q(obs("2026-09-20"))
        self.assertIn("DAY_TOO_RECENT", r)
        self.assertIn("DAY_OUTSIDE_WINDOW", r)

    def test_future_day_refuses(self):
        self.assertIn("DAY_TOO_RECENT", self.q(obs("2026-10-10")))

    def test_wrong_channel_refuses(self):
        self.assertEqual(self.q(obs("2026-07-01", chan="HHZ")), ["CHANNEL_MISMATCH"])

    def test_wrong_epoch_refuses(self):
        self.assertEqual(self.q(obs("2026-07-01"), epoch=(date(2026, 8, 1), None)), ["EPOCH_MISMATCH"])
        self.assertEqual(self.q(obs("2026-07-01"), epoch=(date(1992, 4, 7), date(2026, 6, 30))), ["EPOCH_MISMATCH"])
        self.assertEqual(self.q(obs("2026-07-01"), epoch=(date(1992, 4, 7), None)), [])

    def test_wrong_rate_refuses(self):
        self.assertEqual(self.q(obs("2026-07-01", rate=20.0), expected_rate=40.0), ["RATE_MISMATCH"])

    def test_short_coverage_refuses(self):
        self.assertEqual(self.q(obs("2026-07-01", hours=11.9)), ["COVERAGE_SHORT"])
        self.assertEqual(self.q(obs("2026-07-01", hours=0)), ["NO_DATA"])

    def test_gaps_overlaps_and_fills_never_qualify_but_abutting_traces_do(self):
        self.assertEqual(self.q(obs("2026-07-01", gap=1.0)), ["GAP_FILLED"])
        self.assertEqual(self.q(obs("2026-07-01", gap=-0.5)), ["GAP_FILLED"])      # overlap
        self.assertEqual(self.q(obs("2026-07-01", filled=5)), ["GAP_FILLED"])
        self.assertEqual(self.q(obs("2026-07-01", n_traces=2, gap=0.0)), [])      # abutting, no fill

    def test_response_required_refuses_when_unmeasured_or_absent(self):
        self.assertEqual(self.q(obs("2026-07-01", response=None), require_response=True), ["RESPONSE_MISSING"])
        self.assertEqual(self.q(obs("2026-07-01", response=False), require_response=True), ["RESPONSE_MISSING"])
        self.assertEqual(self.q(obs("2026-07-01", response=True), require_response=True), [])

    def test_repeated_day_is_a_duplicate_not_a_new_day(self):
        seen = set()
        self.assertEqual(self.q(obs("2026-07-01"), seen_days=seen), [])
        self.assertEqual(self.q(obs("2026-07-01", ref="second report"), seen_days=seen), ["DUPLICATE_DAY"])


class Bootstrap(unittest.TestCase):
    def test_fifty_nine_days_refuse_insufficient(self):
        res = tb.bootstrap(STA, [obs(d) for d in days(59)], today=TODAY)
        self.assertFalse(res["ok"])
        self.assertEqual(res["refusal"], "INSUFFICIENT_DAYS")
        self.assertEqual(res["n_qualified"], 59)
        self.assertNotIn("entry", res)

    def test_sixty_days_produce_the_recal_statistics(self):
        vals = [0.30 + 0.01 * (i % 7) for i in range(60)]
        res = tb.bootstrap(STA, [obs(d, thd=v) for d, v in zip(days(60), vals)], today=TODAY)
        self.assertTrue(res["ok"], res)
        e = res["entry"]
        thd = np.array(vals)
        med = float(np.median(thd)); mad = float(np.median(np.abs(thd - med)))
        self.assertEqual(e["mean_thd"], round(med, 6))
        self.assertEqual(e["std_thd"], round(mad * 1.4826, 6))
        self.assertEqual(e["n_samples"], 60)
        self.assertEqual(e["calibration_period"], f"{days(60)[0]} to {days(60)[-1]}")   # days used, not requested
        self.assertNotEqual(e["calibration_period"], "UNCALIBRATED")
        self.assertEqual(e["qa"]["n_days_requested"], 91)
        self.assertTrue(any(i.startswith("Low coverage") for i in e["qa"]["issues"]))   # 60/91 < 80%: disclosed
        self.assertEqual(e["bootstrap"]["days_refused"], {})
        self.assertIn("BOOTSTRAP", e["notes"])

    def test_repeated_reports_do_not_count_as_new_days(self):
        o = [obs(d) for d in days(35)] + [obs(d, ref="repeat") for d in days(35)]
        res = tb.bootstrap(STA, o, today=TODAY)
        self.assertFalse(res["ok"])
        self.assertEqual(res["n_qualified"], 35)
        self.assertEqual(res["refused"]["DUPLICATE_DAY"], 35)

    def test_every_refusal_is_counted_by_name(self):
        o = [obs(d) for d in days(60)] + [obs("2026-05-01"), obs("2026-09-20"), obs("2026-07-01", gap=2.0),
                                          obs("2026-07-02", chan="HHZ"), obs("2026-07-03", thd=0.0)]
        res = tb.bootstrap(STA, o, today=TODAY)
        self.assertTrue(res["ok"])
        r = res["refused"]
        self.assertEqual(r["DAY_OUTSIDE_WINDOW"], 2)
        self.assertEqual(r["DAY_TOO_RECENT"], 1)
        self.assertEqual(r["DUPLICATE_DAY"], 3)       # 07-01, 07-02, 07-03 already seen
        self.assertEqual(r["GAP_FILLED"], 1)
        self.assertEqual(r["CHANNEL_MISMATCH"], 1)
        self.assertEqual(res["n_qualified"], 60)

    def test_estimator_zero_refuses(self):
        res = tb.bootstrap(STA, [obs(d, thd=0.0, p1=0.0) for d in days(60)], today=TODAY)
        self.assertFalse(res["ok"])
        self.assertEqual(res["refused"]["ESTIMATOR_ZERO"], 60)

    def test_station_mismatch_raises(self):
        with self.assertRaises(tb.BootstrapRefused):
            tb.bootstrap(STA, [obs("2026-07-01", station="IU.TUC")], today=TODAY)

    def test_defaults_never_masquerade_as_calibrated(self):
        before = sb.STATION_BASELINES[STA]
        self.assertEqual(before.calibration_period, "UNCALIBRATED")
        res = tb.bootstrap(STA, [], today=TODAY)
        self.assertFalse(res["ok"])
        self.assertIs(sb.STATION_BASELINES[STA], before)          # untouched by a refusal
        self.assertNotIn(STA, rr._calibratable_stations())        # still excluded from the weekly recal


class Estimator(unittest.TestCase):
    def test_real_analyzer_sees_harmonics_in_a_synthetic_tide(self):
        rate = 1.0; n = int(24 * 3600 * rate); t = np.arange(n) / rate
        f = 2.236e-5
        x = 1000 * np.sin(2 * np.pi * f * t) + 100 * np.sin(2 * np.pi * 2 * f * t) + 5 * t + 20
        o = tb.estimate(obs("2026-07-01", rate=rate, hours=24), x)
        self.assertGreater(o.thd, 0.0); self.assertGreater(o.p1, 0.0)

    def test_short_input_yields_estimator_zero(self):
        rate = 1.0; n = int(11 * 3600 * rate); t = np.arange(n) / rate
        o = tb.estimate(obs("2026-07-01", rate=rate, hours=11), 1000 * np.sin(2 * np.pi * 2.236e-5 * t))
        self.assertEqual(o.thd, 0.0)
        res = tb.bootstrap(STA, [o], today=TODAY)
        self.assertIn("COVERAGE_SHORT", res["per_day"][0]["reasons"])


class CandidateFile(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        res = tb.bootstrap(STA, [obs(d) for d in days(60)], today=TODAY)
        self.entry = res["entry"]
        self.base = {"IU.TUC": {"station": "IU.TUC", "mean_thd": 0.37, "std_thd": 0.11, "n_samples": 91,
                                "calibration_period": "2026-06-03 to 2026-09-01", "notes": "Rolling recal"}}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_candidate_name_never_matches_the_loader_glob(self):
        with self.assertRaises(tb.BootstrapRefused):
            tb.compose_candidate_file(self.entry, self.base, str(self.tmp / "thd_baselines_20261003.json"))

    def test_candidate_refuses_the_production_baselines_dir(self):
        with self.assertRaises(tb.BootstrapRefused):
            tb.compose_candidate_file(self.entry, self.base, str(Path(rr.BASELINE_DIR) / "CANDIDATE_x.json"))
        self.assertFalse((Path(rr.BASELINE_DIR) / "CANDIDATE_x.json").exists())

    def test_candidate_merges_base_entries_and_refuses_overwrite(self):
        p = tb.compose_candidate_file(self.entry, self.base, str(self.tmp / "CANDIDATE_thd_baselines_20261003.json"))
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        self.assertEqual(sorted(d), ["IU.SNZO", "IU.TUC"])
        self.assertEqual(d["IU.TUC"]["mean_thd"], 0.37)
        self.assertEqual(d["IU.SNZO"]["n_samples"], 60)
        self.assertIn("bootstrap", d["IU.SNZO"])
        with self.assertRaises(tb.BootstrapRefused):
            tb.compose_candidate_file(self.entry, self.base, p)

    def test_landed_candidate_exits_the_deadlock_with_no_code_change(self):
        """Simulate the reviewed landing: the candidate renamed to the loader's name in a scratch baselines dir.
        station_baselines loads it newest-first and _calibratable_stations() then includes IU.SNZO."""
        p = tb.compose_candidate_file(self.entry, self.base, str(self.tmp / "CANDIDATE_thd_baselines_20261003.json"))
        landed = self.tmp / "thd_baselines_20261003.json"
        shutil.copyfile(p, landed)
        saved = dict(sb.STATION_BASELINES)
        try:
            used = sb._load_newest_baseline_file(bdir=self.tmp)
            self.assertEqual(used, landed.name)
            self.assertEqual(sb.STATION_BASELINES[STA].calibration_period, self.entry["calibration_period"])
            self.assertEqual(sb.STATION_BASELINES[STA].n_samples, 60)
            self.assertEqual(sb.STATION_BASELINES[STA].calibration_date, "2026-10-03")
            self.assertIn(STA, rr._calibratable_stations())
        finally:
            sb.STATION_BASELINES.clear(); sb.STATION_BASELINES.update(saved)
        self.assertNotIn(STA, rr._calibratable_stations())


CACHE = Path("C:/GeoSpec/geospec_runner/monitoring/data/seismic_cache/kaikoura/20260701/hope_fault_waveforms.pkl")


@unittest.skipUnless(CACHE.exists(), "devildog seismic cache absent")
class RealCache(unittest.TestCase):
    def test_observe_one_retained_day(self):
        o = tb.observe_cached_day(str(CACHE), STA, "2026-07-01", estimate_now=False)
        self.assertEqual((o.station, o.location, o.channel, o.sampling_rate), (STA, "00", "BHZ", 40.0))
        self.assertEqual((o.n_traces, o.gap_seconds, o.filled_samples), (1, 0.0, 0))
        self.assertEqual(o.npts, 3456000)
        self.assertAlmostEqual(o.coverage_hours(), 24.0, places=3)
        self.assertIn("locs_present=00,10", o.source_ref)
        self.assertEqual(len(o.source_sha256), 64)
        self.assertEqual(tb.qualify(o, WIN, today=TODAY, expected_rate=40.0, epoch=(date(1992, 4, 7), None)), [])

    def test_abutting_cached_day_merges_without_fill_and_is_estimated(self):
        p = CACHE.parent.parent / "20260719" / "hope_fault_waveforms.pkl"
        if not p.exists():
            self.skipTest("20260719 cache day absent")
        o = tb.observe_cached_day(str(p), STA, "2026-07-19")
        self.assertEqual(o.n_traces, 2)
        self.assertEqual((o.gap_seconds, o.filled_samples), (0.0, 0))
        self.assertGreater(o.thd, 0.0)
        self.assertEqual(tb.qualify(o, WIN, today=TODAY, expected_rate=40.0), [])

    def test_gapped_cached_day_refuses_and_is_not_estimated(self):
        p = CACHE.parent.parent / "20260803" / "hope_fault_waveforms.pkl"
        if not p.exists():
            self.skipTest("20260803 cache day absent")
        o = tb.observe_cached_day(str(p), STA, "2026-08-03")
        self.assertGreater(o.gap_seconds, 0.0)
        self.assertIsNone(o.thd)
        self.assertIn("GAP_FILLED", tb.qualify(o, WIN, today=TODAY))


if __name__ == "__main__":
    unittest.main()
