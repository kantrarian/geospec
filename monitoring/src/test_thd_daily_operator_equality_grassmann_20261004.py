"""Acceptance for codex eb83ad47 finding 1: the two PRODUCTION entrypoints -- ensemble.GeoSpecEnsemble.compute_thd_risk and
calibrate_thd_baselines.compute_daily_thd -- measure IU.SNZO through the SAME operator. The same deterministic synthetic
25-hour / 40-Hz waveform replaces provider I/O (as in codex's probe); both paths are called for the same target day and
compared on requested windows, native vs estimator rates, processed sample counts, raw THD (exact) and operator identity.
An unbound station (IU.TUC) is shown to keep its two legacy paths (different windows/estimators), so the change is
scoped to bound stations. Also: a gapped fixture is refused on BOTH paths, and the daily-operator identity binds
native 40 Hz and estimator 1 Hz distinctly."""
import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np
from obspy import Stream, Trace, UTCDateTime

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import thd_bound_station_operator as OP   # noqa: E402
import ensemble as E                      # noqa: E402
import calibrate_thd_baselines as C       # noqa: E402
import seismic_thd as S                   # noqa: E402

RATE = 40.0
DAY = datetime(2026, 8, 1)


def synthetic_stream(seed=238, hours=52, start=UTCDateTime("2026-07-30T00:00:00"), loc="00", sta="SNZO"):
    rng = np.random.default_rng(seed)
    n = int(RATE * 3600 * hours)
    tr = Trace((rng.normal(size=n) * 1000).astype(np.int32))
    tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel = "IU", sta, loc, "BHZ"
    tr.stats.sampling_rate, tr.stats.starttime = RATE, start
    return Stream([tr])


class FakeClient:
    def __init__(self, stream):
        self.stream = stream; self.calls = []

    def get_waveforms(self, **kw):
        self.calls.append({k: str(v) for k, v in kw.items()})
        st = self.stream.copy()
        return st.trim(kw["starttime"], kw["endtime"], nearest_sample=False)


class EntrypointEquality(unittest.TestCase):
    def run_both(self, stream):
        fc = FakeClient(stream)
        orig = OP.fetch_bound
        calls = {}
        with mock.patch.object(OP, "fetch_bound", wraps=lambda *a, **k: orig(*a, client_factory=lambda name: fc, **k)):
            runner = E.GeoSpecEnsemble(region="kaikoura", eligibility_rule_active=False)
            seen = {}
            orig_dm = OP.daily_measurement

            def spy(*a, **k):
                m = orig_dm(*a, **k); seen.setdefault("records", []).append(m); return m
            with mock.patch.object(OP, "daily_measurement", side_effect=spy):
                daily = runner.compute_thd_risk(DAY, station_network="IU", station_code="SNZO")
                weekly_value, weekly_rate = C.compute_daily_thd("IU", "SNZO", DAY)
        calls["fetch"] = fc.calls
        return daily, weekly_value, weekly_rate, seen.get("records", []), calls

    def test_same_fixture_same_window_rates_samples_thd_and_identity(self):
        daily, weekly_value, weekly_rate, recs, calls = self.run_both(synthetic_stream())
        self.assertTrue(daily.available); self.assertIsNotNone(weekly_value)
        self.assertEqual(len(recs), 2)
        a, b = recs
        self.assertEqual(a["requested_window"], b["requested_window"])
        self.assertEqual(a["requested_window"], [datetime(2026, 7, 30, 23).isoformat(), DAY.isoformat()])
        self.assertEqual((a["native_rate_hz"], a["estimator_rate_hz"]), (40.0, 1.0)); self.assertEqual((b["native_rate_hz"], b["estimator_rate_hz"]), (40.0, 1.0))
        self.assertEqual(a["n_processed_samples"], b["n_processed_samples"]); self.assertEqual(a["n_native_samples"], b["n_native_samples"])
        self.assertEqual(a["operator_identity"], b["operator_identity"]); self.assertEqual(a["operator_identity"], OP.daily_operator_identity("IU.SNZO"))
        self.assertEqual(daily.raw_value, weekly_value)                    # exact, same code path
        self.assertEqual(weekly_rate, 40.0)                                # calibrator reports the NATIVE rate
        # both provider requests were location-bound and identical
        self.assertEqual(len(calls["fetch"]), 2); self.assertEqual(calls["fetch"][0], calls["fetch"][1]); self.assertEqual(calls["fetch"][0]["location"], "00")

    def test_gapped_fixture_refused_on_both_paths(self):
        st = synthetic_stream()
        a = st[0].copy(); b = st[0].copy()
        a.trim(endtime=UTCDateTime("2026-07-31T10:00:00")); b.trim(starttime=UTCDateTime("2026-07-31T10:00:00.5"))
        daily, weekly_value, weekly_rate, recs, calls = self.run_both(Stream([a, b]))
        self.assertFalse(daily.available); self.assertIsNone(weekly_value); self.assertEqual(weekly_rate, None)

    def test_unbound_station_is_measured_by_the_shared_daily_measurement(self):
        # was test_unbound_station_keeps_its_legacy_paths (assertNotEqual): codex 1614 finding 1 / thd-daily-measurement-v1
        # (cayley 2026-10-08) calibrates an unbound station with the daily measurement too, so the two values now agree
        st = synthetic_stream(sta="TUC", loc="10", hours=76)   # covers the legacy weekly window [Aug 1 00:00, Aug 2 01:00) too
        import types
        fake_mod = types.SimpleNamespace(Client=lambda name, timeout=120: FakeClient(st))
        with mock.patch.dict(sys.modules, {"obspy.clients.fdsn": fake_mod}), mock.patch.object(OP, "daily_measurement") as dm:
            runner = E.GeoSpecEnsemble(region="ridgecrest", eligibility_rule_active=False)
            daily = runner.compute_thd_risk(DAY, station_network="IU", station_code="TUC")
            weekly_value, weekly_rate = C.compute_daily_thd("IU", "TUC", DAY)
            self.assertEqual(dm.call_count, 0)                             # an unbound station never enters the bound operator
        self.assertTrue(daily.available); self.assertIsNotNone(weekly_value)
        self.assertEqual(daily.raw_value, weekly_value)

    def test_daily_operator_identity_binds_native_and_estimator_rates_distinctly(self):
        rec = OP.daily_operator_record("IU.SNZO")
        self.assertEqual((rec["native_rate_hz"], rec["estimator_rate_hz"]), (40.0, 1.0))
        self.assertEqual(rec["fetch"]["identity"], OP.operator_identity("IU.SNZO"))
        self.assertNotEqual(rec["identity"], OP.operator_identity("IU.SNZO"))


if __name__ == "__main__":
    unittest.main()
