"""
Tests for thd_retention.py v2 (prospective THD acquisition retention; codex be9c2a19 section 4, a2a5cfe0 repairs 2-3).

    python -m unittest test_thd_retention_cayley_20261001

The StationXML documents are HAND-BUILT to the FDSN StationXML 1.1 element structure (no provider document is
retained anywhere, obspy is not installed, and no acquisition is authorized). The "waveform" objects are a labelled
SYNTHETIC container decoded by a test adapter; production would decode miniSEED through ObsPy. The first real
level=response document and the first real adapter are expected to exercise paths these fixtures do not.
"""
import copy
import hashlib
import json
import math
import os
import unittest
from datetime import datetime, timedelta, timezone

import thd_retention as R

HERE = os.path.dirname(os.path.abspath(__file__))
START = datetime(2026, 9, 26, 23, 0, tzinfo=timezone.utc)
END = START + timedelta(hours=25)
RATE = 40.0
EXPECTED = int(25 * 3600 * RATE) + 1
MAGIC = b"SYNTHETIC-WAVEFORM-FIXTURE-v1\n"


def stationxml(*channels):
    """channels: dicts with net, sta, loc, cha, start, end (None = open), response (bool), input_units (or None)."""
    parts = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<FDSNStationXML xmlns="http://www.fdsn.org/xml/station/1" schemaVersion="1.1">',
             '<Source>SYNTHETIC_FIXTURE_HAND_BUILT</Source><Created>2026-10-01T00:00:00Z</Created>']
    by_station = {}
    for c in channels:
        by_station.setdefault((c["net"], c["sta"]), []).append(c)
    for (net, sta), chans in by_station.items():
        parts.append('<Network code="%s"><Station code="%s" startDate="1992-01-01T00:00:00Z">' % (net, sta))
        parts.append('<Latitude>32.3</Latitude><Longitude>-110.8</Longitude><Elevation>906</Elevation>'
                     '<Site><Name>fixture</Name></Site>')
        for c in chans:
            end_attr = ' endDate="%s"' % c["end"] if c.get("end") else ""
            parts.append('<Channel code="%s" locationCode="%s" startDate="%s"%s>' % (c["cha"], c["loc"], c["start"], end_attr))
            parts.append('<Latitude>32.3</Latitude><Longitude>-110.8</Longitude><Elevation>906</Elevation>'
                         '<Depth>0</Depth><SampleRate>40</SampleRate>')
            if c.get("response", True):
                units = ('<InputUnits><Name>%s</Name></InputUnits>' % c["input_units"]) if c.get("input_units") else ""
                parts.append('<Response><InstrumentSensitivity><Value>3.3E9</Value><Frequency>0.02</Frequency>%s'
                             '<OutputUnits><Name>count</Name></OutputUnits></InstrumentSensitivity></Response>' % units)
            parts.append('</Channel>')
        parts.append('</Station></Network>')
    parts.append('</FDSNStationXML>')
    return "".join(parts).encode("utf-8")


def chan(loc="00", cha="BHZ", start="2020-01-01T00:00:00.0000", end="2027-01-01T00:00:00", **kw):
    return dict(dict(net="IU", sta="TUC", loc=loc, cha=cha, start=start, end=end, response=True, input_units="m/s"), **kw)


def seg(start=START, end=END, loc="00", cha="BHZ", rate=RATE, npts=None):
    if npts is None:
        npts = int(round((end - start).total_seconds() * rate)) + 1
    return dict(network="IU", station="TUC", location=loc, channel=cha, starttime_utc=start.isoformat(),
                endtime_utc=end.isoformat(), sampling_rate_hz=rate, npts=npts)


def waveform(segments):
    """The synthetic retained 'raw' object: a labelled container holding the segment headers."""
    return MAGIC + json.dumps(segments, sort_keys=True).encode("utf-8")


def decode(blob):
    """Test adapter (production: ObsPy). Refuses anything that is not the synthetic container."""
    if not blob.startswith(MAGIC):
        raise ValueError("not a synthetic waveform container")
    return json.loads(blob[len(MAGIC):].decode("utf-8"))


class Store(dict):
    def put(self, blob):
        self[hashlib.sha256(blob).hexdigest()] = bytes(blob)
        return blob

    def load(self, sha):
        return self.get(sha)


def record(segments=None, response=None, request="IU.TUC.*.BHZ", output_units="count", raw=None, store=None):
    segments = [seg()] if segments is None else segments
    raw = waveform(segments) if raw is None else raw
    response = stationxml(chan()) if response is None else response
    if store is not None:
        store.put(raw)
        store.put(response)
    net, sta, loc, cha = request.split(".")
    return R.build_record(
        request=dict(network=net, station=sta, location=loc, channel=cha,
                     starttime_utc=START.isoformat(), endtime_utc=END.isoformat()),
        provider=dict(name="IRIS", service_url="https://service.iris.edu/fdsnws/dataselect/1/query"),
        segments=segments, raw_bytes=raw, response_xml=response,
        preprocessing_version="seismic_thd@d4d0cf28+ensemble@e9b10680", output_units=output_units)


def check(rec, fraction=0.9, start=START, end=END):
    return R.validate(rec, min_valid_fraction=fraction, measurement_start=start, measurement_end=end)


def admit(rec, value, store, fraction=0.9, start=START, end=END):
    return R.admissible_value(rec, value, min_valid_fraction=fraction, measurement_start=start, measurement_end=end,
                              load_object=store.load, decode_waveform=decode)


class CompleteRecord(unittest.TestCase):
    def test_nominal_record_and_value_are_admitted(self):
        store = Store()
        rec = record(store=store)
        v = check(rec)
        self.assertEqual((v.accepted, v.code), (True, R.ACCEPTED))
        self.assertEqual((v.facts["distinct"], v.facts["expected"], v.facts["duplicate"]), (EXPECTED, EXPECTED, 0))
        ch = rec["response"]["channels"][0]
        self.assertEqual((ch["nslc"], ch["input_units"], ch["output_units"], ch["start"]),
                         ("IU.TUC.00.BHZ", "m/s", "count", "2020-01-01T00:00:00Z"))
        self.assertEqual(rec["sensitivity_at_tidal_frequencies"], "NOT_ASSESSED")
        value, verdict = admit(rec, 0.4497, store)
        self.assertEqual((value, verdict.code), (0.4497, R.ADMITTED))

    def test_genuine_zero_is_admissible(self):
        store = Store()
        value, verdict = admit(record(store=store), 0.0, store)
        self.assertEqual((value, verdict.accepted), (0.0, True))

    def test_digest_is_deterministic_and_binds_content(self):
        a, b = record(), record()
        self.assertEqual(a["record_sha256"], b["record_sha256"])
        tampered = copy.deepcopy(a)
        tampered["returned"][0]["npts"] -= 1
        self.assertEqual(check(tampered).code, R.RECORD_MALFORMED)


class CodexProbeCasesNowRefuse(unittest.TestCase):
    """codex a2a5cfe0 probe.py counterexamples, converted to refusal assertions (the nominal positive is above)."""

    def test_duplicate_half_window_counts_once(self):
        half = START + timedelta(hours=12.5)
        rec = record(segments=[seg(end=half), seg(end=half)])
        v = check(rec, 0.9)
        self.assertEqual(v.code, R.INADEQUATE_VALID_COVERAGE)
        half_n = int(round(12.5 * 3600 * RATE)) + 1
        self.assertEqual((v.facts["distinct"], v.facts["duplicate"]), (half_n, half_n))
        self.assertAlmostEqual(v.facts["distinct"] / v.facts["expected"], 0.5, places=4)
        self.assertTrue(check(rec, half_n / EXPECTED).accepted)        # genuine ~50% coverage at its own threshold

    def test_samples_outside_the_measurement_do_not_count(self):
        rec = record(segments=[seg(start=START + timedelta(days=2), end=END + timedelta(days=2))])
        v = check(rec)
        self.assertEqual(v.code, R.INADEQUATE_VALID_COVERAGE)
        self.assertEqual((v.facts["distinct"], v.facts["outside_window"]), (0, EXPECTED))

    def test_measurement_outside_or_reversed(self):
        rec = record()
        self.assertEqual(check(rec, start=START + timedelta(days=2), end=END + timedelta(days=2)).code,
                         R.MEASUREMENT_WINDOW_INVALID)
        self.assertEqual(check(rec, start=END, end=START).code, R.MEASUREMENT_WINDOW_INVALID)
        self.assertEqual(check(rec, start=START, end=START).code, R.MEASUREMENT_WINDOW_INVALID)

    def test_npts_disagreeing_with_span(self):
        lie = seg(end=START + timedelta(hours=1), npts=EXPECTED)
        self.assertEqual(check(record(segments=[lie])).code, R.SEGMENT_INCONSISTENT)

    def test_wildcard_HH_does_not_match_BHZ(self):
        self.assertEqual(check(record(request="IU.TUC.*.HH?")).code, R.NSLC_MISMATCH)

    def test_nonfinite_value_is_refused_not_returned(self):
        store = Store()
        rec = record(store=store)
        for bad, code in ((float("nan"), R.NONFINITE_VALUE), (float("inf"), R.NONFINITE_VALUE),
                          (-float("inf"), R.NONFINITE_VALUE), (True, R.VALUE_INVALID), ("0.4", R.VALUE_INVALID)):
            with self.subTest(value=bad):
                value, verdict = admit(rec, bad, store)
                self.assertIsNone(value)
                self.assertEqual(verdict.code, code)

    def test_record_refusals_never_admit_a_number(self):
        half = START + timedelta(hours=12.5)
        cases = ((dict(segments=[seg(end=half), seg(end=half)]), {}, R.INADEQUATE_VALID_COVERAGE),
                 (dict(request="IU.TUC.*.HH?"), {}, R.NSLC_MISMATCH),
                 (dict(segments=[seg(end=START + timedelta(hours=1), npts=EXPECTED)]), {}, R.SEGMENT_INCONSISTENT),
                 (dict(), dict(start=END, end=START), R.MEASUREMENT_WINDOW_INVALID))
        for build_kw, window, code in cases:
            with self.subTest(code=code):
                store = Store()
                value, verdict = admit(record(store=store, **build_kw), 0.4497, store, **window)
                self.assertIsNone(value)
                self.assertEqual(verdict.code, code)

    def test_arbitrary_non_waveform_bytes_do_not_pass_admission(self):
        store = Store()
        rec = record(raw=b"not a waveform", store=store)
        self.assertTrue(check(rec).accepted)          # the RECORD is coherent; that alone is not proof
        value, verdict = admit(rec, 0.4497, store)
        self.assertIsNone(value)
        self.assertEqual(verdict.code, R.RAW_UNDECODABLE)


class SegmentAndGrid(unittest.TestCase):
    def test_rate_and_count_types(self):
        for bad_rate in (0.0, -40.0, True):
            with self.subTest(rate=bad_rate):
                self.assertEqual(check(record(segments=[seg(rate=bad_rate, npts=10)])).code, R.SEGMENT_INCONSISTENT)
        with self.assertRaises(R.RetentionError):                 # NaN cannot even enter a canonical record
            record(segments=[seg(rate=float("nan"), npts=10)])
        self.assertEqual(check(record(segments=[seg(npts=float(EXPECTED))])).code, R.SEGMENT_INCONSISTENT)

    def test_grid_alignment(self):
        off = START + timedelta(seconds=0.4 / RATE)
        self.assertEqual(check(record(segments=[seg(start=off, end=off + timedelta(hours=25) - timedelta(seconds=1))])).code,
                         R.GRID_MISALIGNED)

    def test_exact_boundaries(self):
        # complete window passes at 1.0; one sample short fails at 1.0 and passes at its exact fraction
        self.assertTrue(check(record(), 1.0).accepted)
        short = record(segments=[seg(end=END - timedelta(seconds=1 / RATE))])
        self.assertEqual(check(short, 1.0).code, R.INADEQUATE_VALID_COVERAGE)
        self.assertTrue(check(short, (EXPECTED - 1) / EXPECTED).accepted)
        # an interior gap: distinct samples exclude it exactly
        gap_start, gap_end = START + timedelta(hours=7), START + timedelta(hours=13)
        gapped = record(segments=[seg(end=gap_start), seg(start=gap_end)])
        missing = int(round((gap_end - gap_start).total_seconds() * RATE)) - 1
        v = check(gapped, 0.5)
        self.assertEqual(v.facts["distinct"], EXPECTED - missing)
        self.assertEqual(check(gapped, 0.9).code, R.INADEQUATE_VALID_COVERAGE)

    def test_measurement_sub_window_counts_its_own_samples(self):
        m_start, m_end = START + timedelta(hours=1), START + timedelta(hours=24)
        v = check(record(), 1.0, start=m_start, end=m_end)
        self.assertTrue(v.accepted)
        self.assertEqual((v.facts["expected"], v.facts["outside_window"]), (int(23 * 3600 * RATE) + 1, EXPECTED - (int(23 * 3600 * RATE) + 1)))


class IdentityGrammar(unittest.TestCase):
    def test_patterns(self):
        self.assertTrue(check(record(request="IU.TUC.*.BH?")).accepted)
        self.assertTrue(check(record(request="IU.TUC.00.BHZ")).accepted)
        self.assertEqual(check(record(request="iu.TUC.*.BHZ")).code, R.NSLC_MISMATCH)      # case-sensitive
        self.assertEqual(check(record(request="IU.TUC.10.BHZ")).code, R.NSLC_MISMATCH)
        empty_loc = [dict(seg(), location="")]
        self.assertTrue(R.nslc_matches("IU.TUC.--.BHZ", "IU.TUC..BHZ"))
        self.assertEqual(check(record(segments=empty_loc, request="IU.TUC.--.BHZ",
                                      response=stationxml(chan(loc="")))).code, R.ACCEPTED)

    def test_grammar(self):
        with self.assertRaises(R.RetentionError):
            record(request="IU.TUC.*.BH[Z]")
        self.assertEqual(check(record(segments=[dict(seg(), channel="BH*")])).code, R.NSLC_MALFORMED)
        self.assertEqual(check(record(segments=[seg(loc="00"), seg(loc="10")])).code, R.AMBIGUOUS_LOCATION)
        self.assertEqual(check(record(segments=[])).code, R.AMBIGUOUS_LOCATION)
        self.assertEqual(check(record(segments=[seg(cha="BHN")], request="IU.TUC.*.BHZ")).code, R.NSLC_MISMATCH)
        self.assertEqual(check(record(response=stationxml(chan(loc="10")))).code, R.NSLC_MISMATCH)


class ResponseAndUnits(unittest.TestCase):
    def test_response_epoch_must_cover_the_measurement(self):
        self.assertEqual(check(record(response=stationxml(chan(end="2026-09-27T12:00:00")))).code, R.RESPONSE_EPOCH_MISMATCH)
        self.assertEqual(check(record(response=stationxml(chan(start="2026-09-27T01:00:00")))).code, R.RESPONSE_EPOCH_MISMATCH)
        self.assertEqual(check(record(response=stationxml(chan(response=False)))).code, R.RESPONSE_EPOCH_MISMATCH)
        self.assertTrue(check(record(response=stationxml(chan(end=None)))).accepted)
        both = stationxml(chan(end="2026-09-27T12:00:00"), chan(start="2026-09-01T00:00:00", end=None))
        self.assertTrue(check(record(response=both)).accepted)

    def test_units_and_retention_fields(self):
        self.assertEqual(check(record(response=stationxml(chan(input_units=None)))).code, R.UNITS_NOT_STATED)
        self.assertEqual(check(record(output_units="")).code, R.UNITS_NOT_STATED)
        self.assertEqual(check(record(raw=b"")).code, R.RAW_NOT_RETAINED)


class RetainedObjects(unittest.TestCase):
    """The record is re-opened through the trusted adapter before any value is admitted."""

    def _refused(self, rec, store, code):
        value, verdict = admit(rec, 0.4497, store)
        self.assertIsNone(value)
        self.assertEqual(verdict.code, code, verdict.reason)

    def test_missing_and_corrupt_objects(self):
        store = Store()
        rec = record(store=store)
        del store[rec["raw"]["sha256"]]
        self._refused(rec, store, R.RAW_OBJECT_MISSING)
        store = Store()
        rec = record(store=store)
        store[rec["raw"]["sha256"]] = store[rec["raw"]["sha256"]] + b"x"
        self._refused(rec, store, R.RAW_OBJECT_CORRUPT)
        store = Store()
        rec = record(store=store)
        del store[rec["response"]["sha256"]]
        self._refused(rec, store, R.RESPONSE_OBJECT_MISSING)
        store = Store()
        rec = record(store=store)
        store[rec["response"]["sha256"]] = store[rec["response"]["sha256"]].replace(b"3.3E9", b"3.4E9")
        self._refused(rec, store, R.RESPONSE_OBJECT_CORRUPT)

    def test_decoded_identity_must_match_the_record(self):
        store = Store()
        other = waveform([seg(loc="10")])                    # the bytes really hold another location
        rec = record(raw=other, store=store)                  # ...but the record claims location 00
        self._refused(rec, store, R.DECODED_IDENTITY_MISMATCH)
        store = Store()
        shorter = waveform([seg(end=END - timedelta(hours=1))])
        self._refused(record(raw=shorter, store=store), store, R.DECODED_IDENTITY_MISMATCH)


class DeclaredThresholdAndInputs(unittest.TestCase):
    def test_threshold_has_no_default_and_is_validated(self):
        with self.assertRaises(TypeError):
            R.validate(record(), measurement_start=START, measurement_end=END)
        for bad in (0, 0.0, 1.5, True, float("nan"), "0.9", None):
            with self.subTest(fraction=bad):
                with self.assertRaises(R.RetentionError):
                    R.validate(record(), min_valid_fraction=bad, measurement_start=START, measurement_end=END)

    def test_naive_and_reversed_request_times_are_caller_defects(self):
        for st, en in (("2026-09-26T23:00:00", END.isoformat()), (END.isoformat(), START.isoformat())):
            with self.subTest(start=st, end=en):
                with self.assertRaises(R.RetentionError):
                    R.build_record(request=dict(network="IU", station="TUC", location="*", channel="BHZ",
                                                starttime_utc=st, endtime_utc=en),
                                   provider={}, segments=[seg()], raw_bytes=b"x", response_xml=None,
                                   preprocessing_version="v", output_units="count")

    def test_not_stationxml(self):
        with self.assertRaises(R.RetentionError):
            R.response_channels(b"<root/>")
        with self.assertRaises(R.RetentionError):
            R.response_channels(b"not xml")


class NotWired(unittest.TestCase):
    """Prospective only: no runtime module imports it, and the live fetch helper is untouched."""

    def test_no_runtime_import(self):
        for name in ("ensemble.py", "run_ensemble_daily.py", "seismic_thd.py", "station_baselines.py"):
            with open(os.path.join(HERE, name), encoding="utf-8") as fh:
                self.assertNotIn("thd_retention", fh.read(), name)


if __name__ == "__main__":
    unittest.main()
