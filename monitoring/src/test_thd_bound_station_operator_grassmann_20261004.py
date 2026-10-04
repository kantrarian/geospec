"""Controls for the bound-station THD operator and the weekly-recal preservation (grassmann 2026-10-04; codex db9a28ff
finding 3). Offline: a fake FDSN client stands in for IRIS; no provider I/O. Covers wrong location / channel / NSLC,
gaps, overlaps, identical-duplicate join, rate mismatch, masked samples, window coverage, normal valid data (value
identical to the production estimator on the same array), the shared-fetch dispatch (bound -> operator; unbound ->
untouched wildcard path), and run_recal: valid recal stamps the operator + date, partial failure PRESERVES the prior
record with its original date and logs the attempt, no prior record -> explicit unavailable attempt, aged preserved
record is judged stale by the existing age rule, 8 siblings preserved, rollback."""
import json
import os
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import numpy as np
from obspy import Stream, Trace, UTCDateTime

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import thd_bound_station_operator as OP  # noqa: E402
import seismic_thd as ST                  # noqa: E402

RATE = 40.0
T0 = UTCDateTime("2026-07-01T00:00:00")


def trace(start, n, *, loc="00", chan="BHZ", net="IU", sta="SNZO", rate=RATE, seed=0, data=None):
    rng = np.random.default_rng(seed)
    tr = Trace((rng.standard_normal(n) * 1000).astype(np.int32) if data is None else data)
    tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel = net, sta, loc, chan
    tr.stats.sampling_rate, tr.stats.starttime = rate, start
    return tr


class FakeClient:
    def __init__(self, stream):
        self.stream = stream
        self.calls = []

    def get_waveforms(self, **kw):
        self.calls.append(kw)
        return self.stream


def factory(stream):
    return lambda name: FakeClient(stream)


class Stitch(unittest.TestCase):
    W = dict(network="IU", station="SNZO", location="00", channel="BHZ", expected_rate=RATE)

    def test_valid_single_trace_covers_window(self):
        n = int(RATE * 3600 * 2)
        data, rate, rec = OP.stitch_window([trace(T0, n)], start=T0.datetime, end=(T0 + 3600).datetime, **self.W)
        self.assertIsNone(rec["refusal"]); self.assertEqual(rate, RATE); self.assertEqual(data.size, int(RATE * 3600))
        self.assertEqual(len(rec["support_sha256"]), 64)

    def test_exact_abut_joins(self):
        n = int(RATE * 3600)
        a = trace(T0, n, seed=1); b = trace(T0 + n / RATE, n, seed=2)
        data, rate, rec = OP.stitch_window([b, a], start=T0.datetime, end=(T0 + 7200).datetime, **self.W)
        self.assertIsNone(rec["refusal"]); self.assertEqual(data.size, 2 * n)

    def test_identical_duplicate_sample_joins_once(self):
        n = int(RATE * 3600)
        a = trace(T0, n, seed=1)
        b = trace(T0 + (n - 1) / RATE, n, seed=2); b.data[0] = a.data[-1]
        data, rate, rec = OP.stitch_window([a, b], start=T0.datetime, end=(T0 + (2 * n - 1) / RATE).datetime, **self.W)
        self.assertIsNone(rec["refusal"]); self.assertEqual(data.size, 2 * n - 1)

    def test_gap_refuses_no_fill(self):
        n = int(RATE * 3600)
        a = trace(T0, n); b = trace(T0 + n / RATE + 0.2, n)
        data, rate, rec = OP.stitch_window([a, b], start=T0.datetime, end=(T0 + 7200).datetime, **self.W)
        self.assertIsNone(data); self.assertEqual(rec["refusal"], "GAP_OR_OVERLAP")

    def test_overlap_refuses(self):
        n = int(RATE * 3600)
        a = trace(T0, n, seed=1); b = trace(T0 + (n - 40) / RATE, n, seed=2)
        data, rate, rec = OP.stitch_window([a, b], start=T0.datetime, end=(T0 + 7000).datetime, **self.W)
        self.assertIsNone(data); self.assertEqual(rec["refusal"], "GAP_OR_OVERLAP")

    def test_wrong_location_channel_nslc_refuse_by_name(self):
        n = int(RATE * 60)
        for kw, code in ((dict(loc="10"), "LOCATION_MISMATCH"), (dict(chan="HHZ"), "CHANNEL_MISMATCH"), (dict(sta="TUC"), "NSLC_MISMATCH")):
            data, rate, rec = OP.stitch_window([trace(T0, n, **kw)], start=T0.datetime, end=(T0 + 30).datetime, **self.W)
            self.assertIsNone(data); self.assertEqual(rec["refusal"], code)

    def test_rate_mismatch_refuses(self):
        n = 6000
        data, rate, rec = OP.stitch_window([trace(T0, n, rate=100.0)], start=T0.datetime, end=(T0 + 30).datetime, **self.W)
        self.assertIsNone(data); self.assertEqual(rec["refusal"], "RATE_MISMATCH")
        data, rate, rec = OP.stitch_window([trace(T0, n), trace(T0 + n / RATE, n, rate=20.0)], start=T0.datetime, end=(T0 + 30).datetime, **self.W)
        self.assertEqual(rec["refusal"], "RATE_MISMATCH")

    def test_masked_samples_refuse(self):
        n = int(RATE * 60)
        tr = trace(T0, n); tr.data = np.ma.masked_array(tr.data, mask=np.zeros(n, bool)); tr.data.mask[5] = True
        data, rate, rec = OP.stitch_window([tr], start=T0.datetime, end=(T0 + 30).datetime, **self.W)
        self.assertEqual(rec["refusal"], "MASKED_SAMPLES")

    def test_window_not_covered_refuses(self):
        n = int(RATE * 1800)
        data, rate, rec = OP.stitch_window([trace(T0, n)], start=T0.datetime, end=(T0 + 3600).datetime, **self.W)
        self.assertEqual(rec["refusal"], "WINDOW_NOT_COVERED")
        data, rate, rec = OP.stitch_window([], start=T0.datetime, end=(T0 + 3600).datetime, **self.W)
        self.assertEqual(rec["refusal"], "NO_TRACES")


class Fetch(unittest.TestCase):
    def test_bound_fetch_requests_bound_location_and_detrends_like_production(self):
        n = int(RATE * 3600 * 25)
        tr = trace(T0, n, seed=3)
        st = Stream([tr])
        fc = FakeClient(st)
        data, rate, rec = OP.fetch_bound("IU", "SNZO", T0.datetime, (T0 + 25 * 3600).datetime, client_factory=lambda name: fc)
        self.assertIsNone(rec["refusal"]); self.assertEqual(fc.calls[0]["location"], "00"); self.assertEqual(fc.calls[0]["channel"], "BHZ")
        # production detrend on the same raw array: obspy demean then linear
        ref = Stream([trace(T0, n, seed=3)]); ref.detrend("demean"); ref.detrend("linear")
        np.testing.assert_allclose(data, ref[0].data, rtol=0, atol=1e-6 * np.abs(ref[0].data).max())
        self.assertEqual(rec["operator_identity"], OP.operator_identity("IU.SNZO"))

    def test_bound_fetch_refuses_wrong_location_from_provider(self):
        n = int(RATE * 60)
        data, rate, rec = OP.fetch_bound("IU", "SNZO", T0.datetime, (T0 + 30).datetime, client_factory=factory(Stream([trace(T0, n, loc="10")])))
        self.assertIsNone(data); self.assertEqual(rate, 0.0); self.assertEqual(rec["refusal"], "LOCATION_MISMATCH")

    def test_provider_error_is_typed(self):
        class Boom:
            def get_waveforms(self, **kw):
                raise RuntimeError("down")
        data, rate, rec = OP.fetch_bound("IU", "SNZO", T0.datetime, (T0 + 30).datetime, client_factory=lambda n: Boom())
        self.assertEqual(rec["refusal"], "PROVIDER_ERROR")

    def test_operator_identity_is_stable_and_bound_to_spec(self):
        a = OP.operator_identity("IU.SNZO")
        self.assertEqual(len(a), 64); self.assertEqual(a, OP.operator_identity("IU.SNZO"))
        rec = OP.operator_record("IU.SNZO")
        self.assertEqual(rec["identity"], a); self.assertEqual(rec["location"], "00"); self.assertEqual(rec["expected_rate_hz"], 40.0)

    def test_shared_fetch_dispatches_bound_station_and_leaves_others_untouched(self):
        n = int(RATE * 3600 * 25)
        good = Stream([trace(T0, n, seed=4)])
        orig = OP.fetch_bound
        with mock.patch.object(OP, "fetch_bound", wraps=lambda *a, **k: orig(*a, client_factory=factory(good), **k)) as fb:
            data, rate = ST.fetch_continuous_data_for_thd("IU", "SNZO", T0.datetime, (T0 + 25 * 3600).datetime)
            self.assertEqual(fb.call_count, 1); self.assertEqual(rate, RATE); self.assertEqual(data.size, n)
        gapped = Stream([trace(T0, int(RATE * 3600)), trace(T0 + 3601, int(RATE * 3600))])
        with mock.patch.object(OP, "fetch_bound", wraps=lambda *a, **k: orig(*a, client_factory=factory(gapped), **k)):
            data, rate = ST.fetch_continuous_data_for_thd("IU", "SNZO", T0.datetime, (T0 + 7200).datetime)
            self.assertIsNone(data); self.assertEqual(rate, 0.0)
        # an unbound station never reaches the operator: the wildcard path is exercised with a fake obspy Client
        fake_mod = types.SimpleNamespace(Client=lambda name, timeout=120: FakeClient(Stream([trace(T0, int(RATE * 60), sta="TUC", loc="10")])))
        with mock.patch.object(OP, "fetch_bound") as fb, mock.patch.dict(sys.modules, {"obspy.clients.fdsn": fake_mod}):
            data, rate = ST.fetch_continuous_data_for_thd("IU", "TUC", T0.datetime, (T0 + 60).datetime)
            self.assertEqual(fb.call_count, 0); self.assertEqual(rate, RATE)   # location '10' accepted by the legacy wildcard path

    def test_bound_value_equals_production_estimator_on_same_array(self):
        from seismic_thd import SeismicTHDAnalyzer
        n = int(RATE * 3600 * 25)
        tr = trace(T0, n, seed=5)
        data, rate, rec = OP.fetch_bound("IU", "SNZO", T0.datetime, (T0 + 25 * 3600).datetime, client_factory=factory(Stream([tr])))
        an = SeismicTHDAnalyzer(n_harmonics=5, freq_tolerance=0.1, window_hours=24)
        thd_a = an.compute_thd(data, rate)[0]
        ref = Stream([trace(T0, n, seed=5)]); ref.detrend("demean"); ref.detrend("linear")
        thd_b = an.compute_thd(ref[0].data.astype(np.float64), RATE)[0]
        self.assertAlmostEqual(thd_a, thd_b, places=9)


class RecalPreservation(unittest.TestCase):
    SIB = ["IU.TUC", "IU.COR", "IU.MAJO", "IU.ANTO", "IU.TATO", "IU.COLA", "BK.BKS", "MX.TLIG"]

    def setUp(self):
        import run_thd_recal as R
        self.R = R
        self.tmp = tempfile.TemporaryDirectory(prefix="recal-preserve-")
        self.bdir = Path(self.tmp.name)
        self._old = R.BASELINE_DIR; R.BASELINE_DIR = self.bdir
        prior = {k: dict(station=k, mean_thd=0.3 + i * 0.01, std_thd=0.05, n_samples=91, calibration_period="2026-06-03 to 2026-09-01",
                         notes="Rolling recal") for i, k in enumerate(self.SIB)}
        prior["IU.SNZO"] = dict(station="IU.SNZO", mean_thd=0.296568, std_thd=0.051472, n_samples=72, calibration_period="2026-06-05 to 2026-09-02",
                                calibration_date="2026-10-03", notes="bootstrap candidate", operator=OP.operator_record("IU.SNZO"))
        (self.bdir / "thd_baselines_20261003.json").write_text(json.dumps(prior), encoding="utf-8")
        self.prior = prior

    def tearDown(self):
        self.R.BASELINE_DIR = self._old; self.tmp.cleanup()

    def run_with(self, results, end=datetime(2026, 10, 10)):
        def fake_calibrate(network, station, days_back, exclude_recent_days, end_date):
            r = results[f"{network}.{station}"]
            if isinstance(r, Exception):
                raise r
            return r
        fake_mod = types.SimpleNamespace(calibrate_station=fake_calibrate)
        with mock.patch.dict(sys.modules, {"calibrate_thd_baselines": fake_mod}):
            return self.R.run_recal(self.SIB + ["IU.SNZO"], end_date=end)

    def ok(self, mean):
        return dict(mean_thd=mean, std_thd=0.05, n_samples=90, calibration_period="2026-06-12 to 2026-09-10")

    def test_valid_recal_stamps_date_and_operator_for_bound_station_only(self):
        path = self.run_with({k: self.ok(0.31) for k in self.SIB + ["IU.SNZO"]})
        out = json.loads(Path(path).read_text())
        self.assertEqual(out["IU.SNZO"]["calibration_date"], "2026-10-10")
        self.assertEqual(out["IU.SNZO"]["operator"]["identity"], OP.operator_identity("IU.SNZO"))
        self.assertNotIn("operator", out["IU.TUC"]); self.assertEqual(out["IU.TUC"]["calibration_date"], "2026-10-10")
        self.assertEqual([a["disposition"] for a in out["_recal_attempts"]].count("RECALIBRATED"), 9)

    def test_partial_failure_preserves_prior_record_with_original_date_and_logs_attempt(self):
        res = {k: self.ok(0.31) for k in self.SIB}; res["IU.SNZO"] = RuntimeError("IRIS timeout")
        out = json.loads(Path(self.run_with(res)).read_text())
        kept = out["IU.SNZO"]
        self.assertEqual(kept["mean_thd"], 0.296568); self.assertEqual(kept["calibration_date"], "2026-10-03")
        self.assertEqual(kept["calibration_period"], "2026-06-05 to 2026-09-02"); self.assertEqual(kept["operator"]["identity"], OP.operator_identity("IU.SNZO"))
        att = [a for a in out["_recal_attempts"] if a["station"] == "IU.SNZO"][0]
        self.assertEqual((att["outcome"], att["disposition"]), ("ERROR", "PRIOR_RECORD_PRESERVED"))
        for k in self.SIB:
            self.assertEqual(out[k]["calibration_date"], "2026-10-10")
        # the preserved record is judged by the existing age rule, not refreshed: window end 09-02 vs a scored day 11-01
        import ensemble as E
        self.assertGreater(E._baseline_age_days(kept["calibration_period"], datetime(2026, 11, 1)), E.MAX_BASELINE_AGE_DAYS)
        self.assertLessEqual(E._baseline_age_days(kept["calibration_period"], datetime(2026, 10, 10)), E.MAX_BASELINE_AGE_DAYS)

    def test_empty_result_preserves_and_no_prior_is_explicitly_unavailable(self):
        res = {k: self.ok(0.31) for k in self.SIB}; res["IU.SNZO"] = dict(mean_thd=None, error="n=4 < 10")
        out = json.loads(Path(self.run_with(res)).read_text())
        self.assertEqual(out["IU.SNZO"]["mean_thd"], 0.296568)
        att = [a for a in out["_recal_attempts"] if a["station"] == "IU.SNZO"][0]; self.assertEqual(att["outcome"], "EMPTY")
        res2 = dict(res); res2["GE.CSS"] = RuntimeError("no data")
        def fake_calibrate(network, station, days_back, exclude_recent_days, end_date):
            r = res2[f"{network}.{station}"]
            if isinstance(r, Exception):
                raise r
            return r
        with mock.patch.dict(sys.modules, {"calibrate_thd_baselines": types.SimpleNamespace(calibrate_station=fake_calibrate)}):
            path = self.R.run_recal(self.SIB + ["IU.SNZO", "GE.CSS"], end_date=datetime(2026, 10, 10))
        out = json.loads(Path(path).read_text())
        self.assertNotIn("GE.CSS", out)
        att = [a for a in out["_recal_attempts"] if a["station"] == "GE.CSS"][0]; self.assertEqual(att["disposition"], "UNAVAILABLE_NO_PRIOR_RECORD")

    def test_all_failed_writes_nothing(self):
        res = {k: RuntimeError("x") for k in self.SIB + ["IU.SNZO"]}
        self.assertIsNone(self.run_with(res))
        self.assertEqual(sorted(p.name for p in self.bdir.glob("thd_baselines_*.json")), ["thd_baselines_20261003.json"])

    def test_loader_reads_preserved_file_and_rollback(self):
        import station_baselines as sb
        res = {k: self.ok(0.31) for k in self.SIB}; res["IU.SNZO"] = RuntimeError("IRIS timeout")
        path = Path(self.run_with(res))
        saved = dict(sb.STATION_BASELINES)
        try:
            used = sb._load_newest_baseline_file(bdir=self.bdir)
            self.assertEqual(used, path.name)
            self.assertEqual(sb.STATION_BASELINES["IU.SNZO"].calibration_date, "2026-10-03")
            self.assertEqual(sb.STATION_BASELINES["IU.TUC"].calibration_date, "2026-10-10")
            self.assertNotIn("_recal_attempts", sb.STATION_BASELINES)
            os.remove(path)
            used = sb._load_newest_baseline_file(bdir=self.bdir)
            self.assertEqual(used, "thd_baselines_20261003.json"); self.assertEqual(sb.STATION_BASELINES["IU.SNZO"].n_samples, 72)
        finally:
            sb.STATION_BASELINES.clear(); sb.STATION_BASELINES.update(saved)


if __name__ == "__main__":
    unittest.main()
