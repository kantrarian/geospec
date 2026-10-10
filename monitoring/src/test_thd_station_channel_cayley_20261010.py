"""thd-station-channel-v1 controls (cayley 2026-10-10; codex b7a29458 correction 2, CAFE).

IV.CAFE's BH* epochs closed 2021-09-24; the station records HHZ at 100 Hz. The declaration makes that a DIFFERENT quantity
with its own operator class, checks the native rate on every fetch, stamps the exact selector into the recal's baseline,
and scores the region ONLY against a baseline carrying that selector: an unstamped, BHZ or other-station baseline is
treated as absent, so the region stays unqualified until its own selector has an eligible baseline ("a returned waveform
is not calibration"). Every BHZ station is unchanged: class, fetch call shape and key.

Offline: fake fetches and the calibration-eligibility harness; no provider I/O.
Run: <py with obspy> -B -m unittest test_thd_station_channel_cayley_20261010   (cwd monitoring/src)
"""
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibration_eligibility_fixtures as FX  # noqa: E402
import thd_daily_measurement as TDM            # noqa: E402
import thd_provider_routing as TPR             # noqa: E402

SELECTOR = "IV.CAFE..HHZ@100Hz"


class Declaration(unittest.TestCase):
    def test_cafe_is_declared_hhz_100hz_with_a_retained_basis(self):
        d = TPR.declared_channel("IV", "CAFE")
        self.assertEqual((d["selector"], d["location"], d["channel"], d["native_rate_hz"]), (SELECTOR, "", "HHZ", 100.0))
        self.assertTrue(d["basis"].startswith("RETAINED: "))
        # updated deliberately 2026-10-10 (owner "request metadata", 643dd746): the response is now RETAINED (55995dfa)
        self.assertTrue(d["response"].startswith("RETAINED: "), "the retained response is named, not assumed")
        self.assertEqual(TPR.selector_for("IV", "CAFE")[0], "")

    def test_only_cafe_is_declared(self):
        self.assertEqual(sorted(TPR.STATION_CHANNELS), ["IV.CAFE"])
        for net, sta in (("IU", "COLA"), ("BK", "BKS"), ("IU", "SNZO"), ("G", "UNM")):
            self.assertIsNone(TPR.declared_channel(net, sta))

    def test_an_invalid_declaration_refuses(self):
        for bad in (dict(channel="HH"), dict(native_rate_hz=100), dict(native_rate_hz=-1.0), dict(location="XYZ"),
                    dict(basis="EXPECTATION: maybe")):
            with self.subTest(bad=bad), mock.patch.dict(TPR.STATION_CHANNELS, {"IV.CAFE": dict(TPR.STATION_CHANNELS["IV.CAFE"], **bad)}):
                with self.assertRaises(ValueError):
                    TPR.declared_channel("IV", "CAFE")


class OperatorClass(unittest.TestCase):
    def setUp(self):
        import seismic_thd as ST
        self.analyzer = ST.SeismicTHDAnalyzer()

    def test_cafe_class_carries_channel_and_rate(self):
        cls = TDM.operator_class("IV", "CAFE", self.analyzer)
        self.assertEqual((cls["channel"], cls["native_rate_hz"], cls["selector"]), ("HHZ", 100.0, ""))
        self.assertIsNotNone(TDM.operator_class_part("IV", "CAFE", self.analyzer))

    def test_bhz_stations_have_no_channel_field(self):
        for net, sta in (("IU", "COLA"), ("BK", "BKS"), ("MX", "TLIG"), ("G", "UNM")):
            with self.subTest(station=sta):
                cls = TDM.operator_class(net, sta, self.analyzer)
                self.assertNotIn("channel", cls)
                self.assertNotIn("native_rate_hz", cls)

    def test_cafe_measurement_record_names_its_channel(self):
        rec = TDM.measurement_record("IV", "CAFE", self.analyzer)
        self.assertEqual((rec["measurement"]["channel"], rec["measurement"]["native_rate_hz"]), ("HHZ", 100.0))
        self.assertEqual(TDM.measurement_record("IU", "COLA", self.analyzer)["measurement"]["channel"], "BHZ")


class _Analyzer:
    window_hours = 24

    def analyze_window(self, data, sample_rate, station, window_time):
        return mock.Mock(thd_value=0.3, fundamental_power=1.0, dominant_frequency=2.2e-5, snr=5.0)


class Measure(unittest.TestCase):
    def test_cafe_is_fetched_on_hhz_and_measured_at_100hz(self):
        calls = []

        def fetch(**kw):
            calls.append(kw)
            return np.zeros(100 * 3600 * 13), 100.0
        rec = TDM.measure("IV", "CAFE", datetime(2026, 10, 8), analyzer=_Analyzer(), fetch=fetch)
        self.assertEqual(calls[0]["channel"], "HHZ")
        self.assertEqual((rec["thd"], rec["native_rate_hz"], rec["estimator_rate_hz"]), (0.3, 100.0, 1.0))

    def test_a_different_native_rate_is_refused(self):
        rec = TDM.measure("IV", "CAFE", datetime(2026, 10, 8), analyzer=_Analyzer(),
                          fetch=lambda **kw: (np.zeros(40 * 3600 * 13), 40.0))
        self.assertEqual((rec["thd"], rec["reason"], rec["native_rate_hz"]), (None, "NATIVE_RATE_MISMATCH", 40.0))

    def test_a_bhz_station_keeps_its_call_shape(self):
        calls = []

        def fetch(**kw):
            calls.append(kw)
            return np.zeros(40 * 3600 * 13), 40.0
        TDM.measure("IU", "COLA", datetime(2026, 10, 8), analyzer=_Analyzer(), fetch=fetch)
        self.assertNotIn("channel", calls[0])

    def test_another_channel_for_cafe_is_refused(self):
        with self.assertRaises(ValueError):
            TDM.measure("IV", "CAFE", datetime(2026, 10, 8), analyzer=_Analyzer(), fetch=lambda **kw: (None, 0.0), channel="HNZ")


class RecalStamp(unittest.TestCase):
    def test_the_recal_stamps_the_selector_for_cafe_only(self):
        import calibrate_thd_baselines as C
        import run_thd_recal as R

        def calibrate(network, station, **kw):
            return {"mean_thd": 0.3, "std_thd": 0.05, "n_samples": 60, "calibration_period": "2026-06-11 to 2026-09-09"}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(R, "BASELINE_DIR", Path(tmp)), \
                mock.patch.object(C, "calibrate_station", side_effect=calibrate), \
                mock.patch.object(R, "_prior_effective_entries", return_value={}):
            path = Path(R.run_recal(["IV.CAFE", "IU.COLA"], end_date=datetime(2026, 10, 9)))
            data = json.loads(path.read_text())
        self.assertEqual(data["IV.CAFE"]["selector"], SELECTOR)
        self.assertNotIn("selector", data["IU.COLA"])


class ScoringUsesOnlyTheStampedBaseline(unittest.TestCase):
    """compute_thd_risk for campi_flegrei, rule ON, through the shared eligibility harness."""

    def run_with(self, selector):
        from station_baselines import StationBaseline
        fresh = FX.fixture_baselines()["calibrated_fresh"]["baseline"]
        base = StationBaseline(station="IV.CAFE", mean_thd=0.3, std_thd=0.05, n_samples=60,
                               calibration_period=fresh.calibration_period, notes="fixture",
                               calibration_date=fresh.calibration_date, selector=selector)
        import ensemble
        return FX.build(ensemble, region="campi_flegrei", network="IV", station="CAFE", thd_baseline=base,
                        thd_value=0.3, shared=["campi_flegrei"], active=True)

    def test_an_unstamped_baseline_is_treated_as_absent(self):
        r = self.run_with(None)
        thd = r.components["seismic_thd"]
        self.assertEqual((thd.calibration_status, thd.eligible_for_tiering, r.methods_available, r.tier), ("missing", False, 0, -1))

    def test_a_bhz_stamped_baseline_is_treated_as_absent(self):
        r = self.run_with("IV.CAFE..BHZ@40Hz")
        self.assertEqual((r.components["seismic_thd"].calibration_status, r.methods_available), ("missing", 0))

    def test_the_hhz_stamped_baseline_is_used(self):
        r = self.run_with(SELECTOR)
        thd = r.components["seismic_thd"]
        self.assertEqual((thd.calibration_status, thd.eligible_for_tiering, r.methods_available), ("calibrated", True, 1))


if __name__ == "__main__":
    unittest.main()
