"""thd-station-attempts-v1 tests (METHOD_QUALIFICATION_DELIVERY_PLAN M4, 2026-10-04).

The REAL seismic_thd.fetch_continuous_data_for_thd (loaded from this directory's file) runs against an in-memory fake
FDSN client: obspy is not installed here, so the provider behaviour is SYNTHETIC and labelled as such. The runner path
is the REAL run_ensemble_daily.run_region_assessment with the ensemble's fetch replaced by a fake that appends the
same provider records. Nothing here touches a network or activates anything.
"""
import importlib.util
import os
import sys
import types
import unittest
from datetime import datetime
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import calibration_eligibility_fixtures as FX  # noqa: E402

FX.install_stubs()
import calibration_eligibility_runner_parity as P  # noqa: E402

P._install_heavy_stubs()
import ensemble as E  # noqa: E402
import run_ensemble_daily as RD  # noqa: E402

DAY = datetime(2026, 10, 2)


# ------------------------------------------------------------------------------------------- fake FDSN (SYNTHETIC)
class _Stats:
    def __init__(self, net, sta, loc, cha, rate, start, npts):
        self.network, self.station, self.location, self.channel = net, sta, loc, cha
        self.sampling_rate, self.starttime, self.npts = rate, start, npts
        self.endtime = "%s+%ds" % (start, int(npts / rate))


class _Trace:
    def __init__(self, net, sta, loc, cha, rate, start, npts):
        self.data = np.zeros(npts)
        self.stats = _Stats(net, sta, loc, cha, rate, start, npts)

    @property
    def id(self):
        s = self.stats
        return "%s.%s.%s.%s" % (s.network, s.station, s.location, s.channel)


class _Stream(list):
    def merge(self, method=1, fill_value=None):
        first = self[0]
        total = sum(len(t.data) for t in self)
        merged = _Trace(first.stats.network, first.stats.station, first.stats.location, first.stats.channel,
                        first.stats.sampling_rate, first.stats.starttime, total)
        self[:] = [merged]

    def detrend(self, kind):
        return None


class _Client:
    BEHAVIOUR = {}

    def __init__(self, name, timeout=None):
        self.name = name

    def get_waveforms(self, network, station, location, channel, starttime, endtime):
        what = self.BEHAVIOUR.get(self.name, "raise")
        if what == "raise":
            raise RuntimeError("synthetic %s refusal" % self.name)
        if what == "empty":
            return _Stream()
        npts = what
        return _Stream([_Trace(network, station, "00", channel, 20.0, "2026-10-01T00:00:00", npts // 2),
                        _Trace(network, station, "00", channel, 20.0, "2026-10-01T12:00:00", npts - npts // 2)])


def _install_fake_obspy():
    saved = {name: sys.modules.get(name) for name in ("obspy", "obspy.clients", "obspy.clients.fdsn")}
    obspy = types.ModuleType("obspy")
    obspy.UTCDateTime = lambda value: value
    clients = types.ModuleType("obspy.clients")
    fdsn = types.ModuleType("obspy.clients.fdsn")
    fdsn.Client = _Client
    sys.modules.update({"obspy": obspy, "obspy.clients": clients, "obspy.clients.fdsn": fdsn})
    return saved


def _restore(saved):
    for name, mod in saved.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


def _real_seismic_thd():
    spec = importlib.util.spec_from_file_location("seismic_thd_real_under_test", os.path.join(HERE, "seismic_thd.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FetchRecordsProviders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ST = _real_seismic_thd()

    def setUp(self):
        self.saved = _install_fake_obspy()
        self.addCleanup(_restore, self.saved)

    def fetch(self, attempts=None, net="BK"):
        kwargs = {} if attempts is None else {"attempts": attempts}
        return self.ST.fetch_continuous_data_for_thd(net, "STA", datetime(2026, 10, 1), datetime(2026, 10, 2),
                                                    **kwargs)

    def test_each_provider_tried_is_recorded_until_one_returns(self):
        # thd-provider-routing-v1: BK routes NCEDC then IRIS (an unmapped network is refused, see
        # test_thd_provider_routing_cayley_20261005); the recording contract is unchanged.
        _Client.BEHAVIOUR = {"NCEDC": "raise", "IRIS": 20 * 3600 * 24, "GEOFON": "raise", "SCEDC": "raise"}
        attempts = []
        data, rate = self.fetch(attempts)
        self.assertEqual((len(data), rate), (20 * 3600 * 24, 20.0))
        self.assertEqual([a["provider"] for a in attempts], ["NCEDC", "IRIS"])
        self.assertEqual([a["outcome"] for a in attempts], ["PROVIDER_ERROR", "DATA_RETURNED"])
        self.assertIn("synthetic NCEDC refusal", attempts[0]["reason"])
        used = attempts[1]
        self.assertEqual((used["trace_id"], used["location"], used["channel"]), ("BK.STA.00.BHZ", "00", "BHZ"))
        self.assertEqual((used["traces_before_merge"], used["traces_after_merge"]), (2, 1))
        self.assertEqual(used["nslc_requested"], "BK.STA.*.BHZ")
        self.assertTrue(used["response"].startswith("NOT_REMOVED"))
        self.assertEqual(used["epoch"][0], "2026-10-01T00:00:00")

    def test_all_providers_failing_is_recorded_and_returns_nothing(self):
        _Client.BEHAVIOUR = {"IRIS": "empty"}
        attempts = []
        self.assertEqual(self.fetch(attempts, net="IU"), (None, 0.0))
        self.assertEqual([(a["provider"], a["outcome"]) for a in attempts], [("IRIS", "NO_TRACES")])

    def test_without_a_sink_the_result_is_the_same(self):
        _Client.BEHAVIOUR = {"NCEDC": "empty", "IRIS": 20 * 3600 * 24}
        with_sink = self.fetch([])
        without = self.fetch(None)
        self.assertEqual(with_sink[1], 20.0)
        self.assertEqual(with_sink[1], without[1])
        self.assertTrue(np.array_equal(with_sink[0], without[0]))


class RunnerRecordsEveryConfiguredStation(unittest.TestCase):
    def setUp(self):
        self.calls = []
        original_fetch, original_flag = E.fetch_continuous_data_for_thd, RD.RECORD_THD_ATTEMPTS
        self.addCleanup(setattr, E, "fetch_continuous_data_for_thd", original_fetch)
        self.addCleanup(setattr, RD, "RECORD_THD_ATTEMPTS", original_flag)
        self.plan = {}

        def fake_fetch(station_network, station_code, start, end, channel="BHZ", attempts=None):
            sid = "%s.%s" % (station_network, station_code)
            self.calls.append((sid, attempts is not None))
            what = self.plan.get(sid, "none")
            if attempts is not None:
                attempts.append({"provider": "SYNTHETIC", "nslc_requested": sid + ".*.BHZ",
                                 "outcome": "NO_TRACES" if what == "none" else "DATA_RETURNED"})
            if what == "none":
                return None, 0.0
            hours = 13 if what == "full" else 2
            return np.zeros(hours * 3600), 1.0
        E.fetch_continuous_data_for_thd = fake_fetch

    def assess(self, region):
        return RD.run_region_assessment(region, DAY)

    def rows(self, out):
        return [(s["role"], s["station"], s["attempted"], s["outcome"], s["selected"]) for s in out["stations"]]

    def test_primary_silent_fallback_used_fallback2_not_attempted(self):
        # a synthetic three-station region keeps the FALLBACK2 path covered now that no configured region has one
        RD.RECORD_THD_ATTEMPTS = True
        three = dict(RD.REGIONS["anchorage"], name="Synthetic three-station", thd_station="PRI", thd_network="XX",
                     fallback_station="FB1", fallback_network="XX", fallback2_station="FB2", fallback2_network="XX")
        self.plan = {"XX.PRI": "none", "XX.FB1": "full"}
        with mock.patch.dict(RD.REGIONS, {"synthetic_three": three}):
            out = self.assess("synthetic_three").to_dict()["thd_attempts"]
        self.assertEqual(out["schema"], "thd-station-attempts-v3")
        self.assertEqual(out["scored_day"], "2026-10-02")
        self.assertEqual(self.rows(out), [("CONFIGURED_PRIMARY", "XX.PRI", True, "NO_DATA", False),
                                          ("CONFIGURED_FALLBACK", "XX.FB1", True, "VALUE", True),
                                          ("CONFIGURED_FALLBACK2", "XX.FB2", False, "NOT_ATTEMPTED", False)])
        self.assertEqual(out["stations"][0]["providers"][0]["outcome"], "NO_TRACES")

    def test_anchorage_is_cola_then_bmr_and_ssl_is_no_longer_asked(self):
        RD.RECORD_THD_ATTEMPTS = True
        self.plan = {"IU.COLA": "full"}
        out = self.assess("anchorage").to_dict()["thd_attempts"]
        self.assertEqual(self.rows(out), [("CONFIGURED_PRIMARY", "IU.COLA", True, "VALUE", True),
                                          ("CONFIGURED_FALLBACK", "AK.BMR", False, "NOT_ATTEMPTED", False)])
        self.assertNotIn("AK.SSL", [sid for sid, _ in self.calls])

    def test_short_data_is_insufficient_samples_not_no_data(self):
        RD.RECORD_THD_ATTEMPTS = True
        self.plan = {"IU.COLA": "short", "AK.BMR": "short"}
        out = self.assess("anchorage").to_dict()["thd_attempts"]
        self.assertEqual([s["outcome"] for s in out["stations"]], ["INSUFFICIENT_SAMPLES"] * 2)
        self.assertFalse(any(s["selected"] for s in out["stations"]))

    def test_flag_off_records_nothing_and_passes_no_sink(self):
        RD.RECORD_THD_ATTEMPTS = False
        self.plan = {"IU.COLA": "none", "AK.BMR": "full"}
        result = self.assess("anchorage")
        self.assertNotIn("thd_attempts", result.to_dict())
        self.assertIsNone(result.thd_attempts)
        self.assertEqual(self.calls, [("IU.COLA", False), ("AK.BMR", False)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
