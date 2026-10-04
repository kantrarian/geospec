"""REAL chain (needs obspy): ensemble.compute_thd_risk -> thd_bound_station_operator.daily_measurement ->
seismic_thd.fetch_continuous_data_for_thd (bound dispatch + cayley attempts composition) -> fetch_bound/stitch_window.
Only the FDSN client is replaced, with grassmann's deterministic SYNTHETIC 40 Hz stream (his equality-test fixture);
nothing here is a measurement. Recording attempts must not change the value; the bound attempt names the fetch
operator, the returned epoch and the native sample count the daily operator consumed; a gap is the operator's refusal.
"""
import os
import sys
import unittest
from unittest import mock

from obspy import Stream, UTCDateTime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import test_thd_daily_operator_equality_grassmann_20261004 as G  # noqa: E402  (fixture + fake client)
import ensemble as E  # noqa: E402
import thd_bound_station_operator as OP  # noqa: E402


def run(stream, record):
    fc = G.FakeClient(stream)
    orig, seen = OP.fetch_bound, []
    orig_dm = OP.daily_measurement

    def spy(*a, **k):
        m = orig_dm(*a, **k)
        seen.append(m)
        return m
    with mock.patch.object(OP, "fetch_bound", wraps=lambda *a, **k: orig(*a, client_factory=lambda name: fc, **k)), \
            mock.patch.object(OP, "daily_measurement", side_effect=spy):
        ens = E.GeoSpecEnsemble(region="kaikoura", eligibility_rule_active=False, record_thd_attempts=record)
        result = ens.compute_thd_risk(G.DAY, station_network="IU", station_code="SNZO")
    return ens, result, seen, fc.calls


class RealChainAttempts(unittest.TestCase):
    def test_recording_does_not_change_the_value_and_records_the_bound_attempt(self):
        quiet, r0, _, calls0 = run(G.synthetic_stream(), False)
        ens, r1, seen, calls1 = run(G.synthetic_stream(), True)
        self.assertTrue(r0.available and r1.available)
        self.assertEqual(r1.raw_value, r0.raw_value)                         # exact
        self.assertEqual(calls1, calls0)
        self.assertIsNone(quiet.last_thd_fetch_attempts)
        (a,) = ens.last_thd_fetch_attempts
        (m,) = seen
        self.assertEqual((a["outcome"], a["provider"], a["nslc_requested"]), ("DATA_RETURNED", "IRIS", "IU.SNZO.00.BHZ"))
        self.assertEqual(a["operator_identity"], OP.operator_identity("IU.SNZO"))
        self.assertEqual(a["window_requested"], m["requested_window"])
        self.assertEqual((a["n_samples"], a["sampling_rate"]), (m["n_native_samples"], m["native_rate_hz"]))
        self.assertIsNotNone(a["epoch"][0])
        self.assertNotEqual(a["epoch"][0][:10], G.DAY.date().isoformat(), "the returned epoch is not the scored day")

    def test_a_gap_is_the_operators_refusal(self):
        st = G.synthetic_stream()
        a, b = st[0].copy(), st[0].copy()
        a.trim(endtime=UTCDateTime("2026-07-31T10:00:00"))
        b.trim(starttime=UTCDateTime("2026-07-31T10:00:00.5"))
        ens, r, seen, _ = run(Stream([a, b]), True)
        self.assertFalse(r.available)
        (att,) = ens.last_thd_fetch_attempts
        self.assertEqual(att["outcome"], "OPERATOR_REFUSED")
        self.assertTrue(att["reason"].startswith("GAP_OR_OVERLAP"), att["reason"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
