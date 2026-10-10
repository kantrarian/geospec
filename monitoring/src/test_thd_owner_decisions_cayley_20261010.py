"""Owner decisions routed in agent-framework 643dd746 (asylum 2026-10-10 ~16:27Z: "brk replaces bks. release the cafe
channel. request metadata."), as the method candidate carries them:

  * norcal_hayward's THD station is BK.BRK (BHZ, location pinned from the retained StationXML), routed NCEDC -> IRIS;
    BK.BKS is no longer configured anywhere;
  * the CAFE declaration's response is RETAINED (the owner-authorized 10-10 StationXML), not NO_RETAINED_*;
  * RESPONSE_METADATA carries the two retrieved channels with their epochs, and a 10-08 window resolves to one epoch;
  * the weekly recal calibrates the owner-approved CALIBRATION_ADDITIONS (IV.CAFE, BK.BRK) and NOTHING else new --
    no configured fallback (AK.BMR, G.UNM), no Hi-net, no UNCALIBRATED default (IU.SNZO) is pulled in silently;
  * BK.BRK without a baseline is unqualified (no early tiering, no BKS baseline reuse).

Offline; no provider I/O. Run: <py with obspy> -B -m unittest test_thd_owner_decisions_cayley_20261010
"""
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import seismic_thd as ST                     # noqa: E402
import thd_provider_routing as TPR           # noqa: E402
import thd_station_configuration as TSC      # noqa: E402


class Configuration(unittest.TestCase):
    def test_norcal_hayward_reads_brk_pinned_and_routed(self):
        import run_ensemble_daily as RD
        c = TSC.run_configuration(RD.REGIONS, ST.SeismicTHDAnalyzer(window_hours=24),
                                  provenance=dict(commit=None, commit_status="TEST", files={}))
        (st,) = c["norcal_hayward"]["stations"]
        self.assertEqual((st["station"], st["selector"], st["selector_kind"], st["providers"]),
                         ("BK.BRK", "BK.BRK.00.BHZ", "PINNED_LOCATION", ["FDSN:NCEDC", "FDSN:IRIS"]))

    def test_bks_is_no_longer_configured(self):
        import run_ensemble_daily as RD
        configured = {"%s.%s" % (n, s) for cfg in RD.REGIONS.values() for _, n, s in TSC.configured_stations(cfg)}
        self.assertNotIn("BK.BKS", configured)
        self.assertIn("BK.BRK", configured)

    def test_the_brk_pin_names_its_retained_source(self):
        pin = TPR.METADATA_LOCATION_PINS["BK.BRK"]
        self.assertEqual(pin["location"], "00")
        self.assertTrue(pin["basis"].startswith("RETAINED: level=response StationXML"))
        self.assertIn("55995dfa", pin["basis"])
        self.assertNotIn("BK.BRK", TPR.STATION_LOCATIONS, "served-trace pins quote a served trace; BRK has none yet")
        self.assertTrue(TPR.selector_for("BK", "BRK")[1].startswith("PINNED_FROM_STATIONXML RETAINED: "))


class ResponseMetadata(unittest.TestCase):
    def test_cafe_declaration_response_is_retained(self):
        d = TPR.declared_channel("IV", "CAFE")
        self.assertTrue(d["response"].startswith("RETAINED: "), d["response"])
        self.assertIn("NOT removed", d["response"])

    def test_both_channels_resolve_one_epoch_for_a_10_08_window(self):
        for nslc, label in (("BK.BRK.00.BHZ", "BK.BRK.00.BHZ@2011-09-12T19:15:00.000000Z"),
                            ("IV.CAFE..HHZ", "IV.CAFE..HHZ@2021-09-24T09:45:00.000000Z")):
            with self.subTest(nslc=nslc):
                r = TPR.response_epoch_for(nslc, "2026-10-06T23:00:00Z", "2026-10-08T00:00:00Z")
                self.assertEqual((r["status"], r["labels"]), ("SINGLE_EPOCH", label))
                self.assertIn("sha256", r["basis"])

    def test_a_window_after_retrieval_is_not_claimed(self):
        r = TPR.response_epoch_for("IV.CAFE..HHZ", "2026-10-20T23:00:00Z", "2026-10-22T00:00:00Z")
        self.assertNotEqual(r["status"], "SINGLE_EPOCH", "nothing after the retrieval instant is known")


class Recal(unittest.TestCase):
    def test_the_recal_adds_exactly_the_owner_approved_stations(self):
        import run_thd_recal as R
        self.assertEqual(list(R.CALIBRATION_ADDITIONS), ["IV.CAFE", "BK.BRK"])
        stations = R._calibratable_stations()
        self.assertEqual(stations[-2:], ["IV.CAFE", "BK.BRK"])
        for never in ("AK.BMR", "G.UNM", "HINET.N.KI2H", "IU.SNZO"):
            self.assertNotIn(never, stations, never + " is not owner-approved for calibration")
        self.assertEqual(len(stations), len(set(stations)))

    def test_additions_are_attributed(self):
        import run_thd_recal as R
        for key, why in R.CALIBRATION_ADDITIONS.items():
            self.assertIn("643dd746", why, key)

    def test_a_recal_of_brk_writes_its_own_entry_and_cafe_its_selector(self):
        import json
        import calibrate_thd_baselines as C
        import run_thd_recal as R

        def calibrate(network, station, **kw):
            return {"mean_thd": 0.3, "std_thd": 0.05, "n_samples": 70, "calibration_period": "2026-06-11 to 2026-09-09"}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(R, "BASELINE_DIR", Path(tmp)), \
                mock.patch.object(C, "calibrate_station", side_effect=calibrate), \
                mock.patch.object(R, "_prior_effective_entries", return_value={}):
            data = json.loads(Path(R.run_recal(["IV.CAFE", "BK.BRK"], end_date=datetime(2026, 10, 12))).read_text())
        self.assertEqual(data["IV.CAFE"]["selector"], "IV.CAFE..HHZ@100Hz")
        self.assertNotIn("selector", data["BK.BRK"])
        self.assertEqual(data["BK.BRK"]["calibration_convention"], "thd-daily-measurement-v1")


class NoEarlyTiering(unittest.TestCase):
    def test_brk_without_its_own_baseline_is_unqualified(self):
        import calibration_eligibility_fixtures as FX
        import ensemble
        r = FX.build(ensemble, region="norcal_hayward", network="BK", station="BRK", thd_baseline=None,
                     thd_value=0.3, shared=["norcal_hayward"], active=True)
        thd = r.components["seismic_thd"]
        self.assertFalse(thd.eligible_for_tiering)
        self.assertEqual(r.methods_available, 0)


if __name__ == "__main__":
    unittest.main()
