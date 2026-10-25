"""thd-station-configuration-v1 controls (cayley 2026-10-10; codex b7a29458 correction 3: "Retain an immutable effective
station configuration identity per scored report: configuration hash/version, source commit, exact primary/fallback
selectors, provider routing and operator identity. Distinguish configured station from the station that actually
supplied the value").

  * the configured stations of every region, in role order, with the exact selector the run would request, its basis
    kind (pinned / wildcard / declared channel / bound operator), routing, declared trust and operator identity;
  * a configured station that was never attempted is still listed (configuration, not outcome);
  * hashes cover content only: provenance never moves them; a changed selector moves its region's hash and the run's,
    and no other region's;
  * the source commit is reported only for a clean checkout of the configuration files; otherwise typed, never guessed;
  * the runner stamps the block on THD and LG-only results; the flag off leaves the record without it.

Offline: fake fetches; no provider I/O. Run: <py with obspy> -B -m unittest test_thd_station_configuration_cayley_20261010
"""
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import seismic_thd as ST                     # noqa: E402
import thd_provider_routing as TPR           # noqa: E402
import thd_station_configuration as TSC      # noqa: E402

PROVENANCE = dict(commit="0" * 40, commit_status="HEAD_CLEAN", files={}, files_hash_rule=TSC.FILES_HASH_RULE)


def analyzer():
    return ST.SeismicTHDAnalyzer(window_hours=24)


def configuration(provenance=PROVENANCE):
    import run_ensemble_daily as RD
    return TSC.run_configuration(RD.REGIONS, analyzer(), provenance=provenance)


def rows(block):
    return [(s["role"], s["station"], s["selector"], s["selector_kind"]) for s in block["stations"]]


class ConfiguredStations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = configuration()

    def test_every_region_is_covered_with_one_run_hash(self):
        import run_ensemble_daily as RD
        self.assertEqual(sorted(self.c), sorted(RD.REGIONS))
        self.assertEqual(len({b["configuration_sha256"] for b in self.c.values()}), 1)
        for b in self.c.values():
            self.assertEqual(b["schema"], "thd-station-configuration-v1")
            self.assertRegex(b["configuration_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(b["region_configuration_sha256"], r"^[0-9a-f]{64}$")

    def test_anchorage_lists_cola_primary_and_bmr_fallback_never_ssl(self):
        self.assertEqual(rows(self.c["anchorage"]), [
            ("CONFIGURED_PRIMARY", "IU.COLA", "IU.COLA.00.BHZ", "PINNED_LOCATION"),
            ("CONFIGURED_FALLBACK", "AK.BMR", "AK.BMR.*.BHZ", "WILDCARD_LOCATION")])
        self.assertNotIn("AK.SSL", str(self.c["anchorage"]), "the historical AK.SSL is not today's configuration")

    def test_cafe_is_the_declared_hhz_selector_with_its_scoped_trust(self):
        (st,) = self.c["campi_flegrei"]["stations"]
        self.assertEqual((st["selector"], st["selector_kind"], st["native_rate_hz"]), ("IV.CAFE..HHZ", "DECLARED_CHANNEL", 100.0))
        self.assertEqual((st["providers"], st["tls_trust"]), (["FDSN:INGV"], {"INGV": "CERTIFI_BUNDLE"}))

    def test_snzo_is_the_bound_operator_v2(self):
        (st,) = self.c["kaikoura"]["stations"]
        import thd_bound_station_operator as OP
        self.assertEqual((st["selector"], st["selector_kind"], st["native_rate_hz"]), ("IU.SNZO.00.BHZ", "BOUND_OPERATOR", 40.0))
        self.assertEqual(st["operator"], dict(kind="BOUND_STATION_OPERATOR", version="thd-bound-station-operator-v2",
                                              identity=OP.daily_operator_identity("IU.SNZO")))

    def test_kanto_is_nied_then_majo_and_bks_is_ncedc_then_iris(self):
        self.assertEqual([(s["station"], s["providers"]) for s in self.c["tokyo_kanto"]["stations"]],
                         [("HINET.N.KI2H", ["NIED_HINET:NIED"]), ("IU.MAJO", ["FDSN:IRIS"])])
        self.assertEqual(self.c["norcal_hayward"]["stations"][0]["providers"], ["FDSN:NCEDC", "FDSN:IRIS"])

    def test_unbound_operator_identity_is_the_measurement_identity(self):
        import thd_daily_measurement as TDM
        st = self.c["anchorage"]["stations"][0]
        m = TDM.measurement_record("IU", "COLA", analyzer())
        self.assertEqual((st["operator"]["kind"], st["operator"]["version"], st["operator"]["identity"]),
                         ("DAILY_MEASUREMENT", "thd-daily-measurement-v1", m["identity"]))
        self.assertTrue(st["operator"]["operator_class"].startswith("operator_class=thd-operator-class-v1:"))

    def test_an_unrouted_network_is_named_not_sprayed(self):
        out = TSC.run_configuration({"x": dict(seismic_available=True, thd_station="ABC", thd_network="ZZ")}, analyzer(),
                                    provenance=PROVENANCE)
        (st,) = out["x"]["stations"]
        self.assertEqual((st["providers"], st["routing"], st["tls_trust"]), (None, "NETWORK_NOT_ROUTED", {}))

    def test_a_region_without_seismic_thd_has_no_stations(self):
        out = TSC.run_configuration({"lg_only": dict(seismic_available=False, thd_station="TUC", thd_network="IU")},
                                    analyzer(), provenance=PROVENANCE)
        self.assertEqual(out["lg_only"]["stations"], [])

    def test_fallback2_and_network_defaults(self):
        out = TSC.run_configuration({"x": dict(seismic_available=True, thd_station="AAA", fallback_station="BBB",
                                               fallback2_station="CCC")}, analyzer(), provenance=PROVENANCE)
        self.assertEqual([(s["role"], s["station"]) for s in out["x"]["stations"]],
                         [("CONFIGURED_PRIMARY", "CI.AAA"), ("CONFIGURED_FALLBACK", "IU.BBB"), ("CONFIGURED_FALLBACK2", "IU.CCC")])


class Identity(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(configuration(), configuration())

    def test_provenance_never_moves_a_hash(self):
        a = configuration()
        b = configuration(dict(PROVENANCE, commit="f" * 40, files={"x": "y"}))
        for region in a:
            self.assertEqual(a[region]["region_configuration_sha256"], b[region]["region_configuration_sha256"])
            self.assertEqual(a[region]["configuration_sha256"], b[region]["configuration_sha256"])
            self.assertNotEqual(a[region]["source"], b[region]["source"])

    def test_a_changed_selector_moves_only_its_region_and_the_run(self):
        a = configuration()
        with mock.patch.dict(TPR.STATION_LOCATIONS, {"IU.COR": dict(TPR.STATION_LOCATIONS["IU.COR"], location="10")}):
            b = configuration()
        self.assertEqual(b["cascadia"]["stations"][0]["selector"], "IU.COR.10.BHZ")
        changed = sorted(r for r in a if a[r]["region_configuration_sha256"] != b[r]["region_configuration_sha256"])
        self.assertEqual(changed, ["cascadia"])
        self.assertNotEqual(a["cascadia"]["configuration_sha256"], b["cascadia"]["configuration_sha256"])

    def test_a_changed_route_or_trust_moves_the_hash(self):
        a = configuration()
        with mock.patch.dict(TPR.NETWORK_ROUTES, {"BK": dict(TPR.NETWORK_ROUTES["BK"], adapters=(("FDSN", "IRIS"),))}):
            b = configuration()
        with mock.patch.dict(TPR.TLS_TRUST, {}, clear=True):
            c = configuration()
        self.assertNotEqual(a["norcal_hayward"]["region_configuration_sha256"], b["norcal_hayward"]["region_configuration_sha256"])
        self.assertNotEqual(a["campi_flegrei"]["region_configuration_sha256"], c["campi_flegrei"]["region_configuration_sha256"])
        self.assertEqual(a["anchorage"]["region_configuration_sha256"], c["anchorage"]["region_configuration_sha256"])

    def test_region_hash_is_the_documented_canonical_digest(self):
        b = configuration()["anchorage"]
        expect = TSC._sha(dict(versions=b["versions"], stations=b["stations"]))
        self.assertEqual(b["region_configuration_sha256"], expect)


class SourceProvenance(unittest.TestCase):
    def runner(self, head_rc=0, status_rc=0, status_out=""):
        def run(cmd, **kw):
            if "rev-parse" in cmd:
                return subprocess.CompletedProcess(cmd, head_rc, stdout="a" * 40 + "\n", stderr="")
            return subprocess.CompletedProcess(cmd, status_rc, stdout=status_out, stderr="")
        return run

    def test_statuses(self):
        cases = [(dict(), ("a" * 40, "HEAD_CLEAN")),
                 (dict(status_out=" M monitoring/src/thd_provider_routing.py\n"), (None, "WORKTREE_DIFFERS")),
                 (dict(head_rc=128), (None, "NOT_A_GIT_CHECKOUT")),
                 (dict(status_rc=1), (None, "GIT_UNAVAILABLE"))]
        for kw, expect in cases:
            with self.subTest(kw=kw):
                out = TSC.source_provenance(HERE.parents[1], runner=self.runner(**kw))
                self.assertEqual((out["commit"], out["commit_status"]), expect)

    def test_no_git_binary_is_typed(self):
        def boom(cmd, **kw):
            raise FileNotFoundError("git")
        out = TSC.source_provenance(HERE.parents[1], runner=boom)
        self.assertEqual((out["commit"], out["commit_status"]), (None, "GIT_UNAVAILABLE"))

    def test_files_hash_crlf_as_lf(self):
        with tempfile.TemporaryDirectory() as tmp:
            for rel in TSC.SOURCE_FILES:
                p = Path(tmp) / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"a\r\nb\r\n")
            out = TSC.source_provenance(tmp, runner=self.runner(head_rc=128))
        self.assertEqual(set(out["files"].values()), {hashlib.sha256(b"a\nb\n").hexdigest()})
        self.assertEqual(out["files_hash_rule"], "SHA256_OF_BYTES_CRLF_TO_LF")

    def test_a_missing_file_is_null_not_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = TSC.source_provenance(tmp, runner=self.runner(head_rc=128))
        self.assertEqual(sorted(out["files"]), sorted(TSC.SOURCE_FILES))
        self.assertEqual(set(out["files"].values()), {None})


class RunnerStamp(unittest.TestCase):
    """The REAL run_region_assessment with the ensemble fetch replaced (same harness shape as test_thd_attempts)."""

    def setUp(self):
        import ensemble as E
        import run_ensemble_daily as RD
        self.E, self.RD = E, RD
        original_fetch = E.fetch_continuous_data_for_thd
        self.addCleanup(setattr, E, "fetch_continuous_data_for_thd", original_fetch)
        self.addCleanup(setattr, RD, "RECORD_THD_STATION_CONFIGURATION", RD.RECORD_THD_STATION_CONFIGURATION)
        TSC._PROVENANCE["source"] = PROVENANCE
        self.addCleanup(TSC._PROVENANCE.clear)

        def fake_fetch(station_network, station_code, start, end, channel="BHZ", attempts=None):
            if attempts is not None:
                attempts.append({"provider": "SYNTHETIC", "nslc_requested": "%s.%s.*.%s" % (station_network, station_code, channel),
                                 "outcome": "DATA_RETURNED"})
            # thd-value-eligibility-v1 (F1): an all-zero trace is the UNDEFINED-ratio state (P1 = 0), refused like the
            # recal refuses it; "full" data is a defined M2-period waveform
            return np.sin(2 * np.pi * np.arange(13 * 3600) / (12.42 * 3600)) + 0.05 * np.sin(4 * np.pi * np.arange(13 * 3600) / (12.42 * 3600)), 1.0
        E.fetch_continuous_data_for_thd = fake_fetch

    def test_thd_path_carries_the_block_and_lists_the_unattempted_fallback(self):
        out = self.RD.run_region_assessment("anchorage", datetime(2026, 10, 8)).to_dict()
        block = out["thd_station_configuration"]
        self.assertEqual(block, configuration()["anchorage"])
        attempted = [s["station"] for s in out["thd_attempts"]["stations"] if s["attempted"]]
        self.assertEqual(attempted, ["IU.COLA"], "the fallback was configured, not attempted")
        self.assertEqual([s["station"] for s in block["stations"]], ["IU.COLA", "AK.BMR"])

    def test_lg_only_path_carries_the_block(self):
        lg_only = dict(self.RD.REGIONS["anchorage"], seismic_available=False)
        with mock.patch.dict(self.RD.REGIONS, {"anchorage": lg_only}):
            out = self.RD.run_region_assessment("anchorage", datetime(2026, 10, 8), use_seismic=False).to_dict()
            expect = TSC.run_configuration(self.RD.REGIONS, analyzer(), provenance=PROVENANCE)["anchorage"]
        self.assertEqual(out["thd_station_configuration"], expect)
        self.assertEqual(out["thd_station_configuration"]["stations"], [])

    def test_flag_off_leaves_the_record_without_it(self):
        self.RD.RECORD_THD_STATION_CONFIGURATION = False
        out = self.RD.run_region_assessment("anchorage", datetime(2026, 10, 8)).to_dict()
        self.assertNotIn("thd_station_configuration", out)


if __name__ == "__main__":
    unittest.main()
