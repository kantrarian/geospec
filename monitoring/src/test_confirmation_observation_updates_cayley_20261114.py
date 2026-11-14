"""observation-update-confirmation-v1 regression tests (cayley 2026-11-14; codex 191be4566 / 896778785 / 603fefee0).

Report-day persistence stays as counted; confirmation also needs each included method's RETAINED observation to have
advanced. Acceptance cases (codex 191be4566): (a) identical LG input on two report dates -> two report days, one LG
epoch, not confirmed; (b) changed retrieval/report timestamps only -> same; (c) changed baseline only -> not a new
observed epoch; (d) a genuine next retained epoch -> a new epoch, subject to the existing support checks; (e) missing
identity, gaps and support changes -> an explicit non-confirmation/reset reason.

Two fixture sources, both from the writer, never from this reader:
  * ISSUED: the public report issued for scored day 2026-11-12 (Tokyo Kanto ELEVATED on LG observation 2026-10-19),
    read from the committed file and pinned by sha256;
  * WRITER: EnsembleResults from the REAL combiner (ensemble.GeoSpecEnsemble.compute_risk) serialised by the REAL
    EnsembleResult.to_dict, fed through the REAL run_ensemble_daily.check_persistence / save_results.
Component values in WRITER cases are synthetic and labelled; nothing here is measured data or a forecast of a run.
"""
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import calibration_eligibility_fixtures as FX  # noqa: E402

FX.install_stubs()
import calibration_eligibility_runner_parity as P  # noqa: E402

P._install_heavy_stubs()
import calibration_eligibility as CE  # noqa: E402
import ensemble as E  # noqa: E402
import method_comparability as MC  # noqa: E402
import run_ensemble_daily as RD  # noqa: E402
import test_method_comparability_cayley_20261004 as T  # noqa: E402  (the existing synthetic component helpers)

ISSUED_PATH = Path(HERE).parents[1] / "docs" / "ensemble" / "2026-11-12" / "20261114T121025240487Z-b7d6f9d5.json"
ISSUED_SHA256 = "65283f26cb64d4f6e39f7aaa6d18afda304cd79321b2a77b8e4173de77acacb3"
REQUIRED = 2
SYNTHETIC_DIGEST = "SYNTHETIC_DIGEST_NOT_MEASURED_DATA_"


def issued_report():
    raw = ISSUED_PATH.read_bytes()
    assert hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() == ISSUED_SHA256, "issued fixture bytes changed"
    return json.loads(raw)


def as_result(row):
    """The current-day object form of an issued row: the attributes check_persistence and the identity reader use."""
    comps = {k: SimpleNamespace(**v) for k, v in row["components"].items()}
    return SimpleNamespace(tier=row["tier"], tier_name=row["tier_name"], method_set=copy.deepcopy(row["method_set"]),
                           components=comps, thd_attempts=copy.deepcopy(row.get("thd_attempts")))


def next_day(row, days=1):
    """The same row as issued `days` later: report date, THD scored day and THD requested window advance (the THD
    operator measures each report day's own window); NOTHING of the LG observation changes."""
    out = copy.deepcopy(row)
    d = datetime.fromisoformat(out["date"]) + timedelta(days=days)
    out["date"] = d.isoformat()
    block = out.get("thd_attempts")
    if block:
        block["scored_day"] = d.strftime("%Y-%m-%d")
        for st in block["stations"]:
            for p in st.get("providers") or []:
                w = p.get("window_requested")
                if w:
                    p["window_requested"] = [(datetime.fromisoformat(x) + timedelta(days=days)).isoformat() for x in w]
    return out


def lg_measurement(row):
    return row["components"]["lambda_geo"]["baseline_provenance"]["measurement"]


def persistence(current_row, priors, region):
    """REAL check_persistence with a loader serving issued prior rows nearest first (None = hole)."""
    def loader(days_back):
        r = priors[days_back - 1] if days_back - 1 < len(priors) else None
        return None if r is None else {"regions": {region: r}}
    return RD.check_persistence({region: as_result(current_row)}, None, datetime(2026, 11, 13), loader=loader)[region]


class IssuedTokyoRepeatedInput(unittest.TestCase):
    """The actual issued 2026-11-12 Tokyo row as the prior day; the current day built from it."""

    def setUp(self):
        self.rep = issued_report()
        self.tokyo = self.rep["regions"]["tokyo_kanto"]
        self.assertEqual((self.tokyo["tier"], self.tokyo["persistence"]["is_confirmed"]), (2, False))
        self.assertEqual(lg_measurement(self.tokyo)["observation_day"], "2026-10-19")

    def test_a_identical_lg_input_is_two_report_days_one_observation(self):
        p = persistence(next_day(self.tokyo), [self.tokyo, None], "tokyo_kanto")
        lg = p["regime"]["observation_updates"]["lambda_geo"]
        self.assertEqual(p["consecutive_days"], 2)               # report-day persistence kept as counted
        self.assertEqual(p["regime"]["report_days"], 2)
        self.assertTrue(p["regime"]["report_days_met"])
        self.assertEqual((lg["epochs"], lg["steps"], lg["observation_updates"]),
                         (["2026-10-19", "2026-10-19"], [MC.STEP_REPEATED], 1))
        self.assertFalse(p["is_confirmed"])
        self.assertEqual(p["regime"]["not_confirmed_reasons"], ["REPEATED_INPUT:lambda_geo"])
        thd = p["regime"]["observation_updates"]["seismic_thd"]
        self.assertEqual((thd["steps"], thd["observation_updates"]), ([MC.STEP_ADVANCED], 2))
        self.assertEqual(thd["epochs"], ["2026-11-13T00:00:00", "2026-11-12T00:00:00"])

    def test_b_new_retrieval_and_report_timestamps_do_not_advance_the_observation(self):
        cur = next_day(self.tokyo)
        m = lg_measurement(cur)
        m["as_of"] = "2026-11-13T23:59:59"
        m["retrieval"]["files_mtime_utc"] = {k: "2026-11-15T12:01:27Z" for k in m["retrieval"]["files_mtime_utc"]}
        m["retrieval"]["newest_mtime_utc"] = m["retrieval"]["oldest_mtime_utc"] = "2026-11-15T12:01:27Z"
        m["station_freshness"]["scored_day"] = "2026-11-13"
        m["observation_age_days"] = 25
        p = persistence(cur, [self.tokyo], "tokyo_kanto")
        self.assertEqual(p["regime"]["observation_updates"]["lambda_geo"]["steps"], [MC.STEP_REPEATED])
        self.assertEqual((p["consecutive_days"], p["is_confirmed"]), (2, False))
        self.assertEqual(p["regime"]["not_confirmed_reasons"], ["REPEATED_INPUT:lambda_geo"])

    def test_c_changed_baseline_is_not_a_new_observed_epoch(self):
        cur = next_day(self.tokyo)
        prov = cur["components"]["lambda_geo"]["baseline_provenance"]
        prov["baseline_value"], prov["calibrated_on"], prov["calibration_sha256"] = 0.031, "2026-11-18", "0" * 64
        # baseline fields alone: still the same observation
        p = persistence(cur, [self.tokyo], "tokyo_kanto")
        self.assertEqual(p["regime"]["observation_updates"]["lambda_geo"]["observation_updates"], 1)
        self.assertFalse(p["is_confirmed"])
        # a real recalibration also changes the support identity: the existing reset applies, and still one epoch
        cur["method_set"]["support"]["lambda_geo"]["identity"] = (
            "lambda_geo provenance lambda_geo_baselines_rule_20261118.json:tokyo_kanto window_end 2026-10-14 "
            "calibrated_on 2026-11-18")
        p = persistence(cur, [self.tokyo], "tokyo_kanto")
        self.assertEqual(p["regime"]["regime_transition"]["kind"], "MEASUREMENT_SUPPORT")
        self.assertEqual((p["consecutive_days"], p["is_confirmed"]), (1, False))
        self.assertEqual(p["regime"]["observation_updates"]["lambda_geo"]["observation_updates"], 1)
        self.assertEqual(p["regime"]["not_confirmed_reasons"], ["REPORT_DAYS_BELOW_REQUIRED"])

    def test_d_a_genuine_next_retained_epoch_is_a_new_observation(self):
        cur = next_day(self.tokyo)
        m = lg_measurement(cur)
        m["observation_day"] = "2026-10-20"
        m["input_sha256"] = SYNTHETIC_DIGEST + "input_next_epoch"
        m["input_selection_sha256"] = SYNTHETIC_DIGEST + "selection_next_epoch"
        p = persistence(cur, [self.tokyo], "tokyo_kanto")
        lg = p["regime"]["observation_updates"]["lambda_geo"]
        self.assertEqual((lg["steps"], lg["observation_updates"]), ([MC.STEP_ADVANCED], 2))
        self.assertEqual((p["consecutive_days"], p["is_confirmed"]), (2, True))
        self.assertEqual(p["regime"]["not_confirmed_reasons"], [])

    def test_d_a_new_epoch_still_obeys_the_support_check(self):
        cur = next_day(self.tokyo)
        lg_measurement(cur)["observation_day"] = "2026-10-20"
        cur["method_set"]["support"]["lambda_geo"]["identity"] = "UNIDENTIFIED"
        p = persistence(cur, [self.tokyo], "tokyo_kanto")
        self.assertEqual(p["regime"]["regime_transition"]["kind"], "MEASUREMENT_SUPPORT")
        self.assertFalse(p["is_confirmed"])

    def test_e_missing_current_identity_blocks_with_its_reason(self):
        cur = next_day(self.tokyo)
        del lg_measurement(cur)["input_sha256"]
        p = persistence(cur, [self.tokyo], "tokyo_kanto")
        self.assertEqual(p["regime"]["observation_updates"]["lambda_geo"]["observation_updates"], None)
        self.assertFalse(p["is_confirmed"])
        self.assertEqual(p["regime"]["not_confirmed_reasons"], ["UNKNOWN_OBSERVATION_IDENTITY:lambda_geo"])

    def test_e_missing_prior_identity_blocks_with_its_reason(self):
        prior = copy.deepcopy(self.tokyo)
        del prior["components"]["lambda_geo"]["baseline_provenance"]["measurement"]
        p = persistence(next_day(self.tokyo), [prior], "tokyo_kanto")
        self.assertEqual(p["regime"]["observation_updates"]["lambda_geo"]["steps"], [MC.STEP_UNKNOWN])
        self.assertEqual(p["regime"]["not_confirmed_reasons"], ["UNKNOWN_OBSERVATION_IDENTITY:lambda_geo"])
        self.assertFalse(p["is_confirmed"])

    def test_e_a_gap_resets_with_its_reason(self):
        p = persistence(next_day(self.tokyo), [None, self.tokyo], "tokyo_kanto")
        self.assertEqual((p["consecutive_days"], p["is_confirmed"]), (1, False))
        self.assertEqual(p["regime"]["not_confirmed_reasons"], ["REPORT_DAYS_BELOW_REQUIRED"])

    def test_a_revised_same_epoch_is_a_revision_not_an_update(self):
        cur = next_day(self.tokyo)
        lg_measurement(cur)["input_sha256"] = SYNTHETIC_DIGEST + "revised_same_epoch"
        p = persistence(cur, [self.tokyo], "tokyo_kanto")
        lg = p["regime"]["observation_updates"]["lambda_geo"]
        self.assertEqual((lg["steps"], lg["observation_updates"], lg["revisions"]), ([MC.STEP_REVISED], 1, 1))
        self.assertEqual(p["regime"]["not_confirmed_reasons"], ["REVISED_SAME_EPOCH:lambda_geo"])
        self.assertFalse(p["is_confirmed"])

    def test_an_earlier_epoch_after_a_later_one_blocks(self):
        cur = next_day(self.tokyo)
        lg_measurement(cur)["observation_day"] = "2026-10-18"
        p = persistence(cur, [self.tokyo], "tokyo_kanto")
        self.assertEqual(p["regime"]["not_confirmed_reasons"], ["EPOCH_REGRESSED:lambda_geo"])
        self.assertFalse(p["is_confirmed"])

    def test_three_days_two_epochs_confirm_and_say_so(self):
        older = copy.deepcopy(self.tokyo)
        m = lg_measurement(older)
        m["observation_day"], m["input_sha256"] = "2026-10-18", SYNTHETIC_DIGEST + "older_epoch"
        older["date"] = "2026-11-11T00:00:00"
        p = persistence(next_day(self.tokyo), [self.tokyo, older], "tokyo_kanto")
        lg = p["regime"]["observation_updates"]["lambda_geo"]
        self.assertEqual(lg["steps"], [MC.STEP_REPEATED, MC.STEP_ADVANCED])
        self.assertEqual((p["consecutive_days"], lg["observation_updates"], p["is_confirmed"]), (3, 2, True))

    def test_codex_repetition_fixture_no_longer_confirms(self):
        """codex 191be4566: regime_persistence(row tier, row method_set, [row, None], 2) returned is_confirmed true."""
        row = self.tokyo
        without_current = MC.regime_persistence(row["tier"], row["method_set"], [row, None], REQUIRED)
        self.assertFalse(without_current["is_confirmed"])
        self.assertEqual(without_current["not_confirmed_reasons"], ["UNKNOWN_OBSERVATION_IDENTITY:lambda_geo"])
        with_current = MC.regime_persistence(row["tier"], row["method_set"], [row, None], REQUIRED,
                                             current_row=next_day(row))
        self.assertEqual((with_current["consecutive_days"], with_current["is_confirmed"]), (2, False))
        self.assertEqual(with_current["not_confirmed_reasons"], ["REPEATED_INPUT:lambda_geo"])

    def test_kumamoto_repeated_zero_lg_blocks_a_thd_driven_watch_under_rule_a(self):
        """Rule (A): every included method must advance, whatever its score (no value-dependent confirmation)."""
        kum = copy.deepcopy(self.rep["regions"]["kumamoto"])
        kum["tier"] = 1                       # SYNTHETIC tier for the rule probe; issued Kumamoto was tier 0
        p = persistence(next_day(kum), [kum], "kumamoto")
        self.assertEqual(p["regime"]["observation_updates"]["lambda_geo"]["observation_updates"], 1)
        self.assertEqual(p["regime"]["not_confirmed_reasons"], ["REPEATED_INPUT:lambda_geo"])
        self.assertFalse(p["is_confirmed"])


class IssuedThdOnlyControls(unittest.TestCase):
    """THD-only regions keep confirming when each report measures its own window, and only then."""

    def setUp(self):
        self.rep = issued_report()

    def test_issued_confirmed_thd_regions_still_confirm(self):
        for region in ("campi_flegrei", "kaikoura"):
            row = self.rep["regions"][region]
            self.assertTrue(row["persistence"]["is_confirmed"])
            p = persistence(next_day(row), [row], region)
            thd = p["regime"]["observation_updates"]["seismic_thd"]
            self.assertEqual((thd["steps"], p["is_confirmed"]), ([MC.STEP_ADVANCED], True), region)
            self.assertEqual(p["regime"]["not_confirmed_reasons"], [], region)

    def test_a_repeated_retained_thd_window_is_a_repeated_input(self):
        row = self.rep["regions"]["campi_flegrei"]
        cur = copy.deepcopy(row)
        cur["date"] = "2026-11-13T00:00:00"     # a new report date, the SAME retained window and scored day
        p = persistence(cur, [row], "campi_flegrei")
        self.assertEqual(p["regime"]["not_confirmed_reasons"], ["REPEATED_INPUT:seismic_thd"])
        self.assertFalse(p["is_confirmed"])

    def test_an_inconsistent_retained_thd_window_is_unknown(self):
        row = self.rep["regions"]["campi_flegrei"]
        cur = next_day(row)
        cur["thd_attempts"]["scored_day"] = "2026-11-20"
        p = persistence(cur, [row], "campi_flegrei")
        self.assertEqual(p["regime"]["not_confirmed_reasons"], ["UNKNOWN_OBSERVATION_IDENTITY:seismic_thd"])
        self.assertFalse(p["is_confirmed"])

    def test_tier_zero_reports_no_reasons_and_is_not_confirmed(self):
        row = self.rep["regions"]["kumamoto"]
        p = persistence(next_day(row), [row], "kumamoto")
        self.assertEqual((p["consecutive_days"], p["is_confirmed"], p["regime"]["not_confirmed_reasons"]),
                         (0, False, []))


class WriterPathAndConsumers(unittest.TestCase):
    """Rows written by the REAL EnsembleResult.to_dict; persistence and summary through the REAL runner functions."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="confirm-cayley-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def lg_thd(self, epoch, digest):
        res = T.run("tokyo_kanto", T.comp("lambda_geo", 1.0), T.unavailable("fault_correlation"),
                    T.comp("seismic_thd", 0.0, station="IU.MAJO", shared=("kumamoto", "tokyo_kanto")))
        res.components["lambda_geo"].baseline_provenance = {"measurement": {
            "observation_day": epoch, "input_sha256": SYNTHETIC_DIGEST + digest,
            "input_selection_sha256": SYNTHETIC_DIGEST + "selection", "operator_identity": SYNTHETIC_DIGEST + "op"}}
        return res

    def issued_row(self, res):
        return json.loads(json.dumps(res.to_dict()))

    def check(self, cur, prior_rows):
        def loader(days_back):
            r = prior_rows[days_back - 1] if days_back - 1 < len(prior_rows) else None
            return None if r is None else {"regions": {"tokyo_kanto": r}}
        return RD.check_persistence({"tokyo_kanto": cur}, None, T.DAY, loader=loader)

    def test_writer_rows_repeated_and_advanced(self):
        prior = self.issued_row(self.lg_thd("2026-10-19", "a"))
        self.assertEqual(prior["tier"], 2)
        same = self.check(self.lg_thd("2026-10-19", "a"), [prior])["tokyo_kanto"]
        self.assertEqual((same["consecutive_days"], same["is_confirmed"]), (2, False))
        self.assertEqual(same["regime"]["not_confirmed_reasons"], ["REPEATED_INPUT:lambda_geo"])
        new = self.check(self.lg_thd("2026-10-20", "b"), [prior])["tokyo_kanto"]
        self.assertEqual((new["consecutive_days"], new["is_confirmed"]), (2, True))
        # writer rows carry no thd_attempts: THD advances by its operator window and says so
        self.assertEqual(new["regime"]["observation_updates"]["seismic_thd"]["steps"], [MC.STEP_ADVANCED_BY_WINDOW])

    def test_every_consumer_reads_the_repaired_flag(self):
        prior = self.issued_row(self.lg_thd("2026-10-19", "a"))
        cur = self.lg_thd("2026-10-19", "a")
        pers = self.check(cur, [prior])
        path = RD.save_results({"tokyo_kanto": cur}, Path(self.tmp), T.DAY, pers)
        with open(path, encoding="utf-8") as handle:
            saved = json.load(handle)
        row = saved["regions"]["tokyo_kanto"]
        self.assertFalse(row["persistence"]["is_confirmed"])
        self.assertEqual(row["persistence"]["consecutive_days"], 2)
        self.assertEqual(row["persistence"]["regime"]["not_confirmed_reasons"], ["REPEATED_INPUT:lambda_geo"])
        self.assertEqual((saved["summary"]["confirmed_watch_count"], saved["summary"]["preliminary_watch_count"]),
                         (0, 1))
        # the printed alert and the return code read the same flag (run_ensemble_daily print_summary / main)
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            RD.print_summary({"tokyo_kanto": cur}, pers)
        self.assertNotIn("ALERT: CONFIRMED", buf.getvalue())
        self.assertIn("PRELIM(2d)", buf.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
