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

    def test_an_ncedc_attempt_records_the_named_bundle_and_an_iris_attempt_does_not(self):
        # 2026-11-22 (codex 78548eb68): thd-scoped-tls-trust-v1 covers NCEDC; the attempt that used it says which bundle
        try:
            import certifi
        except ImportError:
            self.skipTest("certifi is not installed")
        import hashlib
        _Client.BEHAVIOUR = {"NCEDC": "raise", "IRIS": 20 * 3600 * 24}
        attempts = []
        self.fetch(attempts)
        ncedc, iris = attempts
        self.assertEqual((ncedc["provider"], iris["provider"]), ("NCEDC", "IRIS"))
        with open(certifi.where(), "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        self.assertEqual(ncedc["tls_trust"], dict(version="thd-scoped-tls-trust-v1", source="CERTIFI_BUNDLE",
                                                  package_version=certifi.__version__, bundle_sha256=digest))
        self.assertNotIn("tls_trust", iris)
        _Client.BEHAVIOUR = {"NCEDC": 20 * 3600 * 24}
        served = []
        data, rate = self.fetch(served)
        self.assertEqual(([a["provider"] for a in served], served[0]["outcome"], rate), (["NCEDC"], "DATA_RETURNED", 20.0))
        self.assertEqual(served[0]["tls_trust"]["bundle_sha256"], digest)

    # ---- codex 99934af01: the PREPARED bundle on an attempt whose client constructor fails -------------------------
    # The failure of 2026-11-18..20 was raised by the client CONSTRUCTOR (obspy's service discovery), before the fetch
    # had the trust facts in hand. The fetch is NOT edited (its source is part of every measurement identity): the
    # caller names the list it records into (thd_provider_routing.recording_attempts) and fdsn_client writes the facts.
    @staticmethod
    def _refusing_ncedc_constructor():
        original = _Client.__init__

        class FDSNNoServiceException(Exception):
            pass

        def refuse_at_discovery(client, name, timeout=None):
            if name == "NCEDC":
                raise FDSNNoServiceException("synthetic constructor refusal")
            return original(client, name, timeout)
        return refuse_at_discovery

    @staticmethod
    def _certifi_facts():
        import certifi
        import hashlib
        with open(certifi.where(), "rb") as fh:
            return dict(version="thd-scoped-tls-trust-v1", source="CERTIFI_BUNDLE", package_version=certifi.__version__,
                        bundle_sha256=hashlib.sha256(fh.read()).hexdigest())

    def test_the_prepared_bundle_survives_a_constructor_that_fails_at_discovery(self):
        try:
            import certifi  # noqa: F401
        except ImportError:
            self.skipTest("certifi is not installed")
        import thd_provider_routing as TPR
        attempts = []
        with mock.patch.object(_Client, "__init__", self._refusing_ncedc_constructor()), \
                mock.patch.object(_Client, "BEHAVIOUR", {"IRIS": "empty"}), TPR.recording_attempts(attempts):
            self.assertEqual(self.fetch(attempts), (None, 0.0))
        ncedc, iris = attempts
        self.assertEqual((ncedc["provider"], ncedc["outcome"], ncedc["typed_outcome"], ncedc["exception_class"]),
                         ("NCEDC", "PROVIDER_ERROR", "TRANSPORT_ERROR", "FDSNNoServiceException"))
        self.assertEqual(ncedc["tls_trust"], self._certifi_facts())
        self.assertEqual((iris["provider"], iris["outcome"]), ("IRIS", "NO_TRACES"))
        self.assertNotIn("tls_trust", iris)

    def test_the_real_ensemble_path_records_the_prepared_bundle(self):
        # the production composition: GeoSpecEnsemble.compute_thd_risk -> thd_daily_measurement.measure -> the REAL
        # fetch, with recording on. This is the caller that names the list; nothing else has to.
        try:
            import certifi  # noqa: F401
        except ImportError:
            self.skipTest("certifi is not installed")
        self.addCleanup(setattr, E, "fetch_continuous_data_for_thd", E.fetch_continuous_data_for_thd)
        E.fetch_continuous_data_for_thd = self.ST.fetch_continuous_data_for_thd
        for recording in (True, False):
            ensemble = E.GeoSpecEnsemble(region="norcal_hayward", record_thd_attempts=recording)
            with mock.patch.object(_Client, "__init__", self._refusing_ncedc_constructor()), \
                    mock.patch.object(_Client, "BEHAVIOUR", {"IRIS": "empty"}):
                result = ensemble.compute_thd_risk(DAY, "BK", "BRK")
            self.assertFalse(result.available)
            if not recording:
                self.assertIsNone(ensemble.last_thd_fetch_attempts)
                continue
            ncedc, iris = ensemble.last_thd_fetch_attempts
            self.assertEqual((ncedc["provider"], ncedc["outcome"], ncedc["exception_class"]),
                             ("NCEDC", "PROVIDER_ERROR", "FDSNNoServiceException"))
            self.assertEqual(ncedc["tls_trust"], self._certifi_facts())
            self.assertEqual(iris["provider"], "IRIS")
            self.assertNotIn("tls_trust", iris)
        import thd_provider_routing as TPR
        self.assertIsNone(getattr(TPR._RECORDING, "attempts", None), "the recording scope is closed after the call")

    def test_a_bundle_that_cannot_be_loaded_reports_no_digest_and_the_fallback_still_runs(self):
        # nothing was hashed, so nothing is reported: no invented digest, no default-trust client for NCEDC
        import thd_provider_routing as TPR
        built = []
        original = _Client.__init__

        def counting(client, name, timeout=None):
            built.append(name)
            return original(client, name, timeout)
        attempts = []
        with mock.patch.object(TPR, "trust_context", side_effect=OSError("bundle unreadable")), \
                mock.patch.object(_Client, "__init__", counting), \
                mock.patch.object(_Client, "BEHAVIOUR", {"IRIS": 20 * 3600 * 24}), TPR.recording_attempts(attempts):
            data, rate = self.fetch(attempts)
        ncedc, iris = attempts
        self.assertEqual((ncedc["provider"], ncedc["outcome"], ncedc["exception_class"]), ("NCEDC", "PROVIDER_ERROR", "OSError"))
        self.assertNotIn("tls_trust", ncedc)
        self.assertEqual(built, ["IRIS"], "no client of any kind was built for NCEDC")
        self.assertEqual((iris["outcome"], rate), ("DATA_RETURNED", 20.0))

    def test_outside_a_recording_scope_the_fetch_behaves_exactly_as_before(self):
        # the stated limit: a bare fetch (no caller scope) still loses the facts on a constructor failure, as at
        # d1ea267c; the value path is identical with and without the scope
        import thd_provider_routing as TPR
        refuse = self._refusing_ncedc_constructor()
        bare, scoped = [], []
        with mock.patch.object(_Client, "__init__", refuse), mock.patch.object(_Client, "BEHAVIOUR", {"IRIS": 20 * 3600 * 24}):
            a = self.fetch(bare)
            with TPR.recording_attempts(scoped):
                b = self.fetch(scoped)
            c = self.fetch(None)
        self.assertNotIn("tls_trust", bare[0])
        self.assertEqual({k: v for k, v in scoped[0].items() if k != "tls_trust"}, bare[0])
        self.assertEqual(scoped[1], bare[1])
        self.assertEqual((a[1], b[1], c[1]), (20.0, 20.0, 20.0))
        self.assertTrue(np.array_equal(a[0], b[0]) and np.array_equal(a[0], c[0]))

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
                                 "outcome": {"none": "NO_TRACES", "local": "LOCAL_PROCESSING_ERROR"}.get(
                                     what, "DATA_RETURNED"), "reason": "ValueError: synthetic" if what == "local" else None})
            if what in ("none", "local"):
                return None, 0.0
            hours = 13 if what == "full" else 2
            # thd-value-eligibility-v1 (F1): an all-zero trace is the UNDEFINED-ratio state (P1 = 0), refused like the
            # recal refuses it; "full" data is a defined M2-period waveform
            n = hours * 3600
            return ((np.sin(2 * np.pi * np.arange(n) / (12.42 * 3600)) + 0.05 * np.sin(4 * np.pi * np.arange(n) / (12.42 * 3600))) if what == "full" else np.zeros(n)), 1.0
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

    def test_each_attempted_station_carries_its_measurement_identity(self):
        import thd_daily_measurement as TDM
        from seismic_thd import SeismicTHDAnalyzer
        RD.RECORD_THD_ATTEMPTS = True
        self.plan = {"IU.COLA": "none", "AK.BMR": "full"}
        out = self.assess("anchorage").to_dict()["thd_attempts"]
        analyzer = SeismicTHDAnalyzer(n_harmonics=5, freq_tolerance=0.1, window_hours=24)
        self.assertEqual([s["measurement_identity"] for s in out["stations"]],
                         [TDM.measurement_record("IU", "COLA", analyzer)["identity"],
                          TDM.measurement_record("AK", "BMR", analyzer)["identity"]])
        self.plan = {"IU.COLA": "full"}
        out = self.assess("anchorage").to_dict()["thd_attempts"]
        self.assertNotIn("measurement_identity", out["stations"][1], "a station not attempted was not measured")

    def test_returned_data_refused_locally_is_its_own_station_outcome(self):
        RD.RECORD_THD_ATTEMPTS = True
        self.plan = {"IU.COLA": "local", "AK.BMR": "none"}
        out = self.assess("anchorage").to_dict()["thd_attempts"]
        self.assertEqual([s["outcome"] for s in out["stations"]], ["LOCAL_PROCESSING_ERROR", "NO_DATA"])
        self.assertIn("local processing refused it: ValueError: synthetic", out["stations"][0]["reason"])

    def test_flag_off_records_nothing_and_passes_no_sink(self):
        RD.RECORD_THD_ATTEMPTS = False
        self.plan = {"IU.COLA": "none", "AK.BMR": "full"}
        result = self.assess("anchorage")
        self.assertNotIn("thd_attempts", result.to_dict())
        self.assertIsNone(result.thd_attempts)
        self.assertEqual(self.calls, [("IU.COLA", False), ("AK.BMR", False)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
