"""thd-provider-routing-v1 tests (codex review 70ec9745 section 3 acceptance). Needs obspy (1.5.1 pinned env).

The REAL seismic_thd.fetch_continuous_data_for_thd runs with only obspy's FDSN Client replaced by a recording fake.
Every provider refusal is produced by obspy's OWN raise_on_error (the writer of the exceptions the daily run retains),
not a hand-typed message; the no-channel fixture is written by obspy's STATIONTXT writer; the station-level fixtures are
the RETAINED station listings in monitoring/config/station_metadata, sha256-checked against sources.json. Waveforms are
SYNTHETIC. Nothing here touches a network, reads a credential or activates anything.
"""
import ast
import hashlib
import io
import json
import os
import sys
import unittest
from datetime import datetime
from typing import Dict, List
from unittest import mock
from urllib.parse import parse_qs, urlparse

import numpy as np
from obspy import Stream, Trace, UTCDateTime
from obspy.clients.fdsn.client import raise_on_error
from obspy.clients.fdsn.header import URL_MAPPINGS, FDSNNoServiceException
from obspy.core.inventory import Channel, Inventory, Network, Station

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import evidence_redaction as ER  # noqa: E402
import seismic_thd as ST  # noqa: E402
import thd_provider_routing as TPR  # noqa: E402

START, END = datetime(2026, 10, 1, 23), datetime(2026, 10, 3)
METADATA = os.path.join(HERE, "..", "config", "station_metadata")


def configured_thd_networks():
    """The configured THD roster, from the PRODUCER: run_ensemble_daily's own REGIONS literal and its own
    configured_thd_station_regions, executed from the source (importing the runner needs its heavy stack)."""
    with open(os.path.join(HERE, "run_ensemble_daily.py"), encoding="utf-8") as stream:
        tree = ast.parse(stream.read())
    nodes = [n for n in tree.body
             if (isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "REGIONS" for t in n.targets))
             or (isinstance(n, ast.FunctionDef) and n.name == "configured_thd_station_regions")]
    assert len(nodes) == 2, "REGIONS / configured_thd_station_regions not found in run_ensemble_daily.py"
    namespace = {"Dict": Dict, "List": List}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "run_ensemble_daily.py", "exec"), namespace)
    stations = namespace["configured_thd_station_regions"](namespace["REGIONS"])
    return stations, {sid.split(".", 1)[0] for sid in stations}


def obspy_refusal(code, body=b""):
    """The exception obspy itself raises for this HTTP outcome (code None = no HTTP response)."""
    try:
        raise_on_error(code, io.BytesIO(body) if isinstance(body, bytes) else body)
    except Exception as exc:  # noqa: BLE001 -- the writer's exception is the fixture
        return exc
    raise AssertionError("raise_on_error did not raise for %r" % (code,))


def synthetic_stream(net, sta, loc, cha, rate=100.0, hours=25):
    trace = Trace(data=np.zeros(int(rate * 3600 * hours), dtype=np.float64))
    trace.stats.network, trace.stats.station, trace.stats.location, trace.stats.channel = net, sta, loc, cha
    trace.stats.sampling_rate, trace.stats.starttime = rate, UTCDateTime(START)
    return Stream([trace])


class RecordingClient:
    """Stands in for obspy.clients.fdsn.Client. BEHAVIOUR[name] is an exception to raise from get_waveforms, a Stream
    to return, or ("init", exc) to fail at construction (service discovery)."""
    BEHAVIOUR = {}
    made = []
    asked = []

    def __init__(self, name, timeout=None):
        RecordingClient.made.append(name)
        what = self.BEHAVIOUR.get(name)
        if isinstance(what, tuple) and what[0] == "init":
            raise what[1]
        self.name = name

    def get_waveforms(self, network, station, location, channel, starttime, endtime):
        RecordingClient.asked.append((self.name, network, station, location, channel))
        what = self.BEHAVIOUR.get(self.name, Stream())
        if isinstance(what, BaseException):
            raise what
        return what


class RoutedFetch(unittest.TestCase):
    def setUp(self):
        RecordingClient.BEHAVIOUR, RecordingClient.made, RecordingClient.asked = {}, [], []
        patcher = mock.patch("obspy.clients.fdsn.Client", RecordingClient)
        patcher.start()
        self.addCleanup(patcher.stop)

    def fetch(self, net, sta, behaviour):
        RecordingClient.BEHAVIOUR = behaviour
        attempts = []
        data, rate = ST.fetch_continuous_data_for_thd(net, sta, START, END, attempts=attempts)
        return data, rate, attempts


class TheMapCoversTheConfiguredRoster(unittest.TestCase):
    def test_every_configured_thd_network_has_an_explicit_route(self):
        stations, networks = configured_thd_networks()
        self.assertIn("IV.CAFE", stations)                         # the roster really was read
        self.assertEqual(sorted(networks - set(TPR.NETWORK_ROUTES)), [])
        self.assertTrue({"IU", "BK", "IV", "AK", "MX", "G", "HINET"} <= networks)

    def test_every_fdsn_adapter_resolves_and_no_route_repeats_a_service(self):
        for network, entry in TPR.NETWORK_ROUTES.items():
            with self.subTest(network=network):
                self.assertTrue(entry["basis"].split(":", 1)[0] in ("RETAINED", "LADDER", "EXPECTATION"))
                urls = [URL_MAPPINGS[key] for kind, key in entry["adapters"] if kind == TPR.FDSN]
                self.assertEqual(len(urls), len(set(urls)), urls)
                self.assertTrue(all(kind in (TPR.FDSN, TPR.NIED_HINET) for kind, _ in entry["adapters"]))

    def test_obspy_still_maps_gfz_and_geofon_to_one_service(self):
        # the basis for dropping the GE ladder's GFZ leg; if obspy ever splits them this fails and the leg is revisited
        self.assertEqual(URL_MAPPINGS["GFZ"], URL_MAPPINGS["GEOFON"])


class DeterministicRouting(RoutedFetch):
    def test_iv_asks_only_ingv_the_authoritative_centre(self):
        _, _, attempts = self.fetch("IV", "CAFE", {"INGV": obspy_refusal(204)})
        self.assertEqual(RecordingClient.made, ["INGV"])
        self.assertEqual([(a["provider"], a["typed_outcome"]) for a in attempts], [("INGV", "NO_DATA")])

    def test_ak_mx_g_ask_only_iris(self):
        for net, sta in (("AK", "SSL"), ("MX", "TLIG"), ("G", "UNM")):
            with self.subTest(station=net + "." + sta):
                RecordingClient.made = []
                self.fetch(net, sta, {"IRIS": obspy_refusal(204)})
                self.assertEqual(RecordingClient.made, ["IRIS"])

    def test_bk_asks_ncedc_then_iris(self):
        _, _, attempts = self.fetch("BK", "BKS", {"NCEDC": obspy_refusal(204), "IRIS": obspy_refusal(204)})
        self.assertEqual([a["provider"] for a in attempts], ["NCEDC", "IRIS"])

    def test_hinet_makes_no_request_and_says_auth_required(self):
        data, rate, attempts = self.fetch("HINET", "N.KI2H", {})
        self.assertEqual((data, rate, RecordingClient.made), (None, 0.0, []))
        self.assertEqual([(a["provider"], a["adapter"], a["outcome"], a["typed_outcome"]) for a in attempts],
                         [("NIED", "NIED_HINET", "NOT_REQUESTED", "AUTH_REQUIRED")])

    def test_an_unmapped_network_is_refused_by_name_and_nothing_is_asked(self):
        data, rate, attempts = self.fetch("XX", "STA", {})
        self.assertEqual((data, rate, RecordingClient.made), (None, 0.0, []))
        self.assertEqual([(a["outcome"], a["typed_outcome"]) for a in attempts], [("NOT_REQUESTED", "NETWORK_NOT_ROUTED")])
        self.assertEqual(attempts[0]["routing"], TPR.ROUTING_VERSION)

    def test_routing_without_a_sink_returns_the_same(self):
        stream = synthetic_stream("IV", "CAFE", "", "BHZ")
        with_sink = self.fetch("IV", "CAFE", {"INGV": stream})
        RecordingClient.BEHAVIOUR = {"INGV": synthetic_stream("IV", "CAFE", "", "BHZ")}
        without = ST.fetch_continuous_data_for_thd("IV", "CAFE", START, END)
        self.assertEqual(with_sink[1], without[1])
        self.assertTrue(np.array_equal(with_sink[0], without[0]))


class ExactSelectorPositive(RoutedFetch):
    def test_iv_cafe_bhz_at_ingv_returns_data_with_its_identity(self):
        data, rate, attempts = self.fetch("IV", "CAFE", {"INGV": synthetic_stream("IV", "CAFE", "", "BHZ")})
        self.assertEqual((len(data), rate), (100 * 3600 * 25, 100.0))
        self.assertEqual(RecordingClient.asked, [("INGV", "IV", "CAFE", "*", "BHZ")])
        used = attempts[0]
        self.assertEqual((used["provider"], used["outcome"], used["typed_outcome"]), ("INGV", "DATA_RETURNED", "DATA_RETURNED"))
        self.assertEqual((used["trace_id"], used["nslc_requested"]), ("IV.CAFE..BHZ", "IV.CAFE.*.BHZ"))


class TypedNegatives(RoutedFetch):
    CASES = (  # (label, how the provider fails, typed outcome, exception class, http status)
        ("auth 401", obspy_refusal(401), "AUTH_REQUIRED", "FDSNUnauthorizedException", 401),
        ("auth 403", obspy_refusal(403), "AUTH_REQUIRED", "FDSNForbiddenException", 403),
        ("selector 422", obspy_refusal(422, b"Error 422: RequestValidationError: station must be 1-8 characters"),
         "INVALID_SELECTOR", "FDSNException", 422),
        ("selector 400", obspy_refusal(400), "INVALID_SELECTOR", "FDSNBadRequestException", 400),
        ("no data 204", obspy_refusal(204), "NO_DATA", "FDSNNoDataException", 204),
        ("empty stream", Stream(), "NO_DATA", None, None),
        ("timeout", obspy_refusal(None, TimeoutError("timed out")), "TRANSPORT_ERROR", "FDSNTimeoutException", None),
        ("unavailable 503", obspy_refusal(503), "TRANSPORT_ERROR", "FDSNServiceUnavailableException", 503),
        ("discovery", ("init", FDSNNoServiceException("No FDSN services could be discovered at 'https://x'.")),
         "TRANSPORT_ERROR", "FDSNNoServiceException", None),
        ("too large 413", obspy_refusal(413), "PROVIDER_ERROR", "FDSNRequestTooLargeException", 413),
        ("other", RuntimeError("synthetic"), "PROVIDER_ERROR", "RuntimeError", None),
    )

    def test_each_refusal_is_typed_and_keeps_class_and_status(self):
        for label, how, typed, cls, status in self.CASES:
            with self.subTest(case=label):
                RecordingClient.made = []
                data, rate, attempts = self.fetch("IV", "CAFE", {"INGV": how})
                self.assertEqual((data, rate), (None, 0.0))
                record = attempts[0]
                self.assertEqual((record["typed_outcome"], record["exception_class"], record["http_status"]),
                                 (typed, cls, status))
                if cls is not None:
                    self.assertEqual(record["outcome"], "PROVIDER_ERROR")
                    # the retained reason is exactly what the fetch always wrote, and history types the same way
                    self.assertTrue(record["reason"].startswith(cls + ": "))
                    self.assertEqual(TPR.classify_retained_reason(record["reason"]), (typed, cls, status))

    def test_a_failed_discovery_falls_through_to_the_next_adapter(self):
        _, _, attempts = self.fetch("GE", "STA", {"GEOFON": ("init", FDSNNoServiceException("no services")),
                                                  "IRIS": obspy_refusal(204)})
        self.assertEqual([(a["provider"], a["typed_outcome"]) for a in attempts],
                         [("GEOFON", "TRANSPORT_ERROR"), ("IRIS", "NO_DATA")])

    def test_a_credential_in_provider_text_is_still_redacted(self):
        exc = obspy_refusal(401, b"token=AbCdEf123456XyZ password=hunter2")
        _, _, attempts = self.fetch("IV", "CAFE", {"INGV": exc})
        for leaked in ("AbCdEf123456XyZ", "hunter2"):
            self.assertNotIn(leaked, attempts[0]["reason"])
        self.assertEqual(attempts[0]["typed_outcome"], "AUTH_REQUIRED")

    def test_retained_text_that_is_not_the_writer_shape_is_not_typed(self):
        for text in (None, "", "no class prefix", "two words: x", "Insufficient data from AK.SSL"):
            with self.subTest(text=text):
                self.assertIsNone(TPR.classify_retained_reason(text))


def retained_listing(name):
    with open(os.path.join(METADATA, "sources.json"), encoding="utf-8") as stream:
        source = json.load(stream)["files"][name]
    with open(os.path.join(METADATA, name), "rb") as stream:
        raw = stream.read()
    assert hashlib.sha256(raw).hexdigest() == source["sha256"], name + " differs from its recorded sha256"
    query = parse_qs(urlparse(source["url"]).query)
    asked = (set(query["net"][0].split(",")), set(query["sta"][0].split(",")))
    return raw.decode("utf-8"), asked


def channel_listing(channel, start="2005-07-10", end=None):
    cha = Channel(code=channel, location_code="", latitude=41.028, longitude=15.2366, elevation=1070.0, depth=0.0,
                  sample_rate=100.0, start_date=UTCDateTime(start), end_date=UTCDateTime(end) if end else None)
    sta = Station(code="CAFE", latitude=41.028, longitude=15.2366, elevation=1070.0, channels=[cha],
                  start_date=UTCDateTime("2005-07-10"))
    out = io.StringIO()
    Inventory(networks=[Network(code="IV", stations=[sta])], source="SYNTHETIC").write(out, format="STATIONTXT",
                                                                                         level="channel")
    return out.getvalue()


class MetadataFirstRefinement(unittest.TestCase):
    WINDOW = ("2026-10-01T23:00:00", "2026-10-03T00:00:00")

    def test_retained_station_listings(self):
        earthscope, asked_es = retained_listing("earthscope.txt")
        ingv, asked_ingv = retained_listing("ingv.txt")
        cases = ((("AK", "SSL"), earthscope, asked_es, ("INVALID_SELECTOR", "NO_MATCHING_STATION_IN_INVENTORY")),
                 (("IU", "TUC"), earthscope, asked_es, ("NO_DATA", "CHANNEL_UNVERIFIED_STATION_LEVEL_ONLY")),
                 (("IV", "CAFE"), ingv, asked_ingv, ("NO_DATA", "CHANNEL_UNVERIFIED_STATION_LEVEL_ONLY")),
                 (("BK", "BKS"), earthscope, asked_es, ("NO_DATA", "INVENTORY_DID_NOT_ASK")))
        for (net, sta), text, asked, expected in cases:
            with self.subTest(station=net + "." + sta):
                self.assertEqual(TPR.refine_no_data(net, sta, "BHZ", self.WINDOW, text, asked), expected)

    def test_channel_level_listing_separates_no_channel_from_no_data(self):
        asked = ({"IV"}, {"CAFE"})
        self.assertEqual(TPR.refine_no_data("IV", "CAFE", "BHZ", self.WINDOW, channel_listing("HHZ"), asked),
                         ("NO_MATCHING_CHANNEL", "NO_CHANNEL_EPOCH_COVERING_WINDOW"))
        self.assertEqual(TPR.refine_no_data("IV", "CAFE", "BHZ", self.WINDOW, channel_listing("BHZ", end="2020-01-01"),
                                            asked), ("NO_MATCHING_CHANNEL", "NO_CHANNEL_EPOCH_COVERING_WINDOW"))
        self.assertEqual(TPR.refine_no_data("IV", "CAFE", "BHZ", self.WINDOW, channel_listing("BHZ"), asked),
                         ("NO_DATA", "CHANNEL_IN_INVENTORY_PROVIDER_RETURNED_NO_DATA"))

    def test_a_listing_without_its_header_is_refused(self):
        with self.assertRaises(ValueError):
            TPR.refine_no_data("IV", "CAFE", "BHZ", self.WINDOW, "IV|CAFE|41|15|1070||2005-07-10|", ({"IV"}, {"CAFE"}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
