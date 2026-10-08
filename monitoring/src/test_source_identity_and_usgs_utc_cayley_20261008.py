"""Regression tests for codex cfb195ff source_identity_and_utc.patch (cayley 2026-10-08), so the candidate's own suite
holds both fixes. Synthetic only: a mocked USGS response, an emulated non-UTC host, no real HTTP.
"""
import json
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import event_scorer as ES  # noqa: E402
import thd_daily_measurement as TDM  # noqa: E402
from seismic_thd import SeismicTHDAnalyzer  # noqa: E402


class UnreadableSourceStaysUnidentified(unittest.TestCase):
    def test_an_unreadable_source_never_yields_an_apparently_identified_measurement(self):
        with mock.patch.object(TDM, "_code_identity", return_value="UNIDENTIFIED"):
            self.assertEqual(TDM.measurement_record("IU", "TUC", SeismicTHDAnalyzer())["identity"], "UNIDENTIFIED")
        self.assertRegex(TDM.measurement_record("IU", "TUC", SeismicTHDAnalyzer())["identity"], r"^[0-9a-f]{64}$")


class UsgsEpochMillisecondsAreUtc(unittest.TestCase):
    def test_the_loader_asks_for_utc_itself_on_a_non_utc_host(self):
        event_time = datetime(2026, 10, 4, 12, tzinfo=timezone.utc)

        def fromtimestamp(seconds, tz=None):   # a UTC-4 host: naive conversion is local wall time
            if tz is None:
                return datetime.fromtimestamp(seconds, timezone(timedelta(hours=-4))).replace(tzinfo=None)
            return datetime.fromtimestamp(seconds, tz)
        body = {"features": [{"id": "synthetic", "properties": {"time": int(event_time.timestamp() * 1000), "mag": 5.6},
                              "geometry": {"coordinates": [36.2, 37.8, 10]}}]}
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(body).encode()
        with mock.patch("urllib.request.urlopen", return_value=response), \
                mock.patch.object(ES, "datetime", wraps=datetime) as clock:
            clock.fromtimestamp.side_effect = fromtimestamp
            loaded = ES.load_events_from_usgs(datetime(2026, 10, 1), datetime(2026, 10, 6))
        self.assertEqual(loaded[0].time, event_time)
        self.assertEqual(loaded[0].time.utcoffset(), timedelta(0))


if __name__ == "__main__":
    unittest.main()
