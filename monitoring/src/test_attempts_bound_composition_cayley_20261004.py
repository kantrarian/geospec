"""Composition of thd-station-attempts-v1 (cayley) with the bound-station operator (grassmann b92e7087).

The REAL seismic_thd.fetch_continuous_data_for_thd dispatches a bound station to the operator before the provider loop
that records attempts; without the composition a bound station (IU.SNZO) would carry no attempt detail at all. The
operator's fetch is replaced at ITS module boundary (thd_bound_station_operator.fetch_bound) with SYNTHETIC results;
obspy is faked as in test_thd_attempts. Nothing here is a measurement.
"""
import os
import sys
import unittest
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import test_thd_attempts_cayley_20261004 as A  # noqa: E402  (stubs + fake obspy helpers)
import thd_bound_station_operator as OP  # noqa: E402

START, END = datetime(2026, 10, 1), datetime(2026, 10, 2)


class BoundPathRecordsItsAttempt(unittest.TestCase):
    def setUp(self):
        saved = A._install_fake_obspy()
        self.addCleanup(A._restore, saved)
        self.ST = A._real_seismic_thd()
        original = OP.fetch_bound
        self.addCleanup(setattr, OP, "fetch_bound", original)
        self.calls = []

    def fake(self, result):
        def fetch_bound(network, station, start, end, channel=None):
            self.calls.append((network, station))
            return result
        OP.fetch_bound = fetch_bound

    def test_a_bound_value_records_the_operator_and_the_returned_epoch(self):
        rec = dict(station="IU.SNZO", operator_identity="ab" * 32, location="00", channel="BHZ", client="IRIS",
                   requested=[START.isoformat(), END.isoformat()], refusal=None, n_traces=2, rate=40.0,
                   npts=40 * 3600 * 24, start="2026-10-01T00:00:00.000000Z", end="2026-10-01T23:59:59.975000Z")
        self.fake((np.zeros(40 * 3600 * 24), 40.0, rec))
        attempts = []
        data, rate = self.ST.fetch_continuous_data_for_thd("IU", "SNZO", START, END, attempts=attempts)
        self.assertEqual((len(data), rate, self.calls), (40 * 3600 * 24, 40.0, [("IU", "SNZO")]))
        (r,) = attempts
        self.assertEqual((r["outcome"], r["provider"], r["nslc_requested"], r["trace_id"]),
                         ("DATA_RETURNED", "IRIS", "IU.SNZO.00.BHZ", "IU.SNZO.00.BHZ"))
        self.assertEqual(r["operator_identity"], "ab" * 32)
        self.assertEqual(r["epoch"], ["2026-10-01T00:00:00.000000Z", "2026-10-01T23:59:59.975000Z"])
        self.assertEqual((r["n_samples"], r["sampling_rate"]), (40 * 3600 * 24, 40.0))

    def test_a_refusal_records_the_operator_code_and_no_secret(self):
        rec = dict(station="IU.SNZO", operator_identity="cd" * 32, location="00", channel="BHZ", client="IRIS",
                   requested=[START.isoformat(), END.isoformat()], refusal="GAP_OR_OVERLAP", n_traces=3)
        self.fake((None, 0.0, rec))
        attempts = []
        self.assertEqual(self.ST.fetch_continuous_data_for_thd("IU", "SNZO", START, END, attempts=attempts), (None, 0.0))
        self.assertEqual((attempts[0]["outcome"], attempts[0]["reason"]), ("OPERATOR_REFUSED", "GAP_OR_OVERLAP"))
        rec2 = dict(rec, refusal="PROVIDER_ERROR", error="token=AbCdEf123456")
        self.fake((None, 0.0, rec2))
        attempts = []
        self.ST.fetch_continuous_data_for_thd("IU", "SNZO", START, END, attempts=attempts)
        self.assertNotIn("AbCdEf123456", attempts[0]["reason"])

    def test_without_a_sink_the_bound_path_is_unchanged(self):
        rec = dict(station="IU.SNZO", operator_identity="ef" * 32, location="00", channel="BHZ", client="IRIS",
                   requested=[START.isoformat(), END.isoformat()], refusal=None, n_traces=1, rate=40.0,
                   npts=10, start="s", end="e")
        self.fake((np.ones(10), 40.0, rec))
        data, rate = self.ST.fetch_continuous_data_for_thd("IU", "SNZO", START, END)
        self.assertEqual((len(data), rate), (10, 40.0))

    def test_an_unbound_station_still_records_its_providers(self):
        A._Client.BEHAVIOUR = {"IRIS": "empty"}
        attempts = []
        self.ST.fetch_continuous_data_for_thd("IU", "ANTO", START, END, attempts=attempts)
        self.assertEqual([(a["provider"], a["outcome"]) for a in attempts], [("IRIS", "NO_TRACES")])
        self.assertEqual(self.calls, [], "an unbound station never reaches the operator")


if __name__ == "__main__":
    unittest.main(verbosity=2)
