"""
Tests for thd_retention.py (prospective THD acquisition retention record; codex be9c2a19 ruling section 4).

    python -m unittest test_thd_retention_cayley_20261001

The StationXML documents here are HAND-BUILT to the FDSN StationXML 1.1 element structure (no provider document is
retained anywhere, obspy is not installed, and no acquisition is authorized). The first real level=response
document is expected to exercise parser paths these fixtures do not.
"""
import copy
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


def seg(start=START, end=END, loc="00", cha="BHZ", rate=RATE):
    npts = int(round((end - start).total_seconds() * rate)) + 1
    return dict(network="IU", station="TUC", location=loc, channel=cha, starttime_utc=start.isoformat(),
                endtime_utc=end.isoformat(), sampling_rate_hz=rate, npts=npts)


def record(segments=None, response=None, request_loc="*", output_units="count", raw=b"MSEED-fixture-bytes"):
    return R.build_record(
        request=dict(network="IU", station="TUC", location=request_loc, channel="BHZ",
                     starttime_utc=START.isoformat(), endtime_utc=END.isoformat()),
        provider=dict(name="IRIS", service_url="https://service.iris.edu/fdsnws/dataselect/1/query"),
        segments=[seg()] if segments is None else segments,
        raw_bytes=raw, response_xml=stationxml(chan()) if response is None else response,
        preprocessing_version="seismic_thd@d4d0cf28+ensemble@e9b10680", output_units=output_units)


def check(rec, fraction=0.9):
    return R.validate(rec, min_valid_fraction=fraction, measurement_start=START, measurement_end=END)


class CompleteRecord(unittest.TestCase):
    def test_accepted_and_carries_the_facts(self):
        rec = record()
        v = check(rec)
        self.assertEqual((v.accepted, v.code), (True, R.ACCEPTED))
        self.assertEqual(rec["samples"]["expected"], EXPECTED)
        self.assertEqual(rec["samples"]["returned"], EXPECTED)
        self.assertEqual(rec["samples"]["gaps"], [])
        ch = rec["response"]["channels"][0]
        self.assertEqual((ch["nslc"], ch["input_units"], ch["output_units"]), ("IU.TUC.00.BHZ", "m/s", "count"))
        self.assertEqual(ch["start"], "2020-01-01T00:00:00Z")
        self.assertEqual(rec["sensitivity_at_tidal_frequencies"], "NOT_ASSESSED")
        self.assertIn("NOT_ASSESSED", v.detail)

    def test_digest_is_deterministic_and_binds_content(self):
        a, b = record(), record()
        self.assertEqual(a["record_sha256"], b["record_sha256"])
        self.assertNotEqual(record(raw=b"MSEED-fixture-bytez")["raw"]["sha256"], a["raw"]["sha256"])
        tampered = copy.deepcopy(a)
        tampered["samples"]["returned"] -= 1
        self.assertEqual(check(tampered).code, R.RECORD_MALFORMED)


class TypedRefusals(unittest.TestCase):
    def assertRefused(self, rec, code, fraction=0.9):
        v = check(rec, fraction)
        self.assertEqual((v.accepted, v.code), (False, code), v.reason)
        value, verdict = R.admissible_value(rec, 0.4497, min_valid_fraction=fraction, measurement_start=START,
                                            measurement_end=END)
        self.assertIsNone(value)                    # never converted to 0.0
        self.assertEqual(verdict.code, code)

    def test_ambiguous_location(self):
        self.assertRefused(record(segments=[seg(loc="00"), seg(loc="10")]), R.AMBIGUOUS_LOCATION)
        self.assertRefused(record(segments=[]), R.AMBIGUOUS_LOCATION)

    def test_channel_and_response_identity(self):
        self.assertRefused(record(segments=[seg(cha="BHN")]), R.NSLC_MISMATCH)
        self.assertRefused(record(request_loc="10"), R.NSLC_MISMATCH)
        self.assertRefused(record(response=stationxml(chan(loc="10"))), R.NSLC_MISMATCH)

    def test_rate(self):
        half = START + timedelta(hours=12)
        self.assertRefused(record(segments=[seg(end=half), seg(start=half + timedelta(seconds=1), rate=20.0)]),
                           R.RATE_INCONSISTENT)

    def test_raw_and_response_retention(self):
        self.assertRefused(record(raw=b""), R.RAW_NOT_RETAINED)
        rec = R.build_record(
            request=dict(network="IU", station="TUC", location="*", channel="BHZ", starttime_utc=START.isoformat(),
                         endtime_utc=END.isoformat()),
            provider=dict(name="IRIS", service_url="x"), segments=[seg()], raw_bytes=b"x", response_xml=None,
            preprocessing_version="v", output_units="count")
        self.assertRefused(rec, R.RESPONSE_NOT_RETAINED)

    def test_response_epoch_must_cover_the_measurement(self):
        self.assertRefused(record(response=stationxml(chan(end="2026-09-27T12:00:00"))), R.RESPONSE_EPOCH_MISMATCH)
        self.assertRefused(record(response=stationxml(chan(start="2026-09-27T01:00:00"))), R.RESPONSE_EPOCH_MISMATCH)
        self.assertRefused(record(response=stationxml(chan(response=False))), R.RESPONSE_EPOCH_MISMATCH)
        # An open epoch, or a second epoch that covers, is accepted.
        self.assertTrue(check(record(response=stationxml(chan(end=None)))).accepted)
        both = stationxml(chan(end="2026-09-27T12:00:00"), chan(start="2026-09-01T00:00:00", end=None))
        self.assertTrue(check(record(response=both)).accepted)

    def test_units(self):
        self.assertRefused(record(response=stationxml(chan(input_units=None))), R.UNITS_NOT_STATED)
        self.assertRefused(record(output_units=""), R.UNITS_NOT_STATED)

    def test_inadequate_valid_coverage_with_gap_accounting(self):
        gap_start, gap_end = START + timedelta(hours=7), START + timedelta(hours=13)
        rec = record(segments=[seg(end=gap_start), seg(start=gap_end)])
        missing = int(round((gap_end - gap_start).total_seconds() * RATE)) - 1
        self.assertEqual(rec["samples"]["merge_would_fill"], missing)
        self.assertEqual(rec["samples"]["gaps"][0]["position"], "interior")
        self.assertEqual(rec["samples"]["returned"] + missing, EXPECTED)
        self.assertRefused(rec, R.INADEQUATE_VALID_COVERAGE, fraction=0.9)
        self.assertTrue(check(rec, 0.7).accepted)
        late = record(segments=[seg(start=START + timedelta(hours=1))])
        self.assertEqual(late["samples"]["missing_at_edges"], int(3600 * RATE))
        self.assertEqual(late["samples"]["gaps"][0]["position"], "leading")


class DeclaredThresholdAndInputs(unittest.TestCase):
    def test_threshold_has_no_default_and_is_validated(self):
        with self.assertRaises(TypeError):
            R.validate(record(), measurement_start=START, measurement_end=END)
        for bad in (0, 0.0, 1.5, True, float("nan"), "0.9", None):
            with self.subTest(fraction=bad):
                with self.assertRaises(R.RetentionError):
                    R.validate(record(), min_valid_fraction=bad, measurement_start=START, measurement_end=END)

    def test_naive_and_reversed_times_are_caller_defects(self):
        with self.assertRaises(R.RetentionError):
            R.build_record(request=dict(network="IU", station="TUC", location="*", channel="BHZ",
                                        starttime_utc="2026-09-26T23:00:00", endtime_utc=END.isoformat()),
                           provider={}, segments=[seg()], raw_bytes=b"x", response_xml=None,
                           preprocessing_version="v", output_units="count")
        with self.assertRaises(R.RetentionError):
            R.build_record(request=dict(network="IU", station="TUC", location="*", channel="BHZ",
                                        starttime_utc=END.isoformat(), endtime_utc=START.isoformat()),
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
