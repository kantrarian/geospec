"""Contract v3 transition and the LG identity regression sequence (cayley 2026-11-16; codex 5851b0649 answer 3,
8c8d84db9).

Rows are written by the REAL combiner and EnsembleResult.to_dict and judged by the REAL check_persistence /
regime_persistence. LEGACY rows are those same rows as a v2 report carried them: the v2 contract's regime and key, and NO
selected-observation content (never manufactured for history). Required sequence (codex answer 3):
  legacy v2 WATCH+ streak (long) -> first v3 WATCH+ row: one report day, REGIME_TRANSITION, not confirmed, no legacy
  UNKNOWN leaking through the boundary -> second v3 row, every included method genuinely advanced: may confirm -> a
  same-epoch repeat, a same-epoch revision, an epoch regression, or missing/malformed identity: never a second update.
The LG content digests and THD sample digests below are SYNTHETIC and labelled. The writer-built LG identity (real
producer output) is tested in test_lg_observation_identity_cayley_20261116.
"""
import copy
import hashlib
import json
import unittest
from datetime import datetime, timedelta

import test_confirmation_observation_updates_cayley_20261114 as C   # stubs, the issued-report helpers, the writer rows

RD, MC, T = C.RD, C.MC, C.T
V2 = "method-comparability-v2"
DAY = T.DAY


def content(epoch, label, **override):
    oc = {"schema": MC.LG_OBSERVATION_CONTENT_SCHEMA, "encoding": MC.LG_OBSERVATION_CONTENT_ENCODING,
          "status": "KNOWN", "epoch": epoch,
          "sha256": hashlib.sha256(("SYNTHETIC selected observation " + label).encode()).hexdigest()}
    oc.update(override)
    return oc


def thd_block(station, day, label):
    end, start = day.isoformat(), (day - timedelta(hours=25)).isoformat()
    return {"schema": "thd-station-attempts-v3", "scored_day": day.date().isoformat(),
            "stations": [{"selected": True, "station": station, "outcome": "VALUE",
                          "measurement_identity": hashlib.sha256(b"SYNTHETIC THD operator").hexdigest(),
                          "providers": [{"outcome": "DATA_RETURNED", "trace_id": station + "..BHZ",
                                         "window_requested": [start, end], "epoch": [start + "Z", end + "Z"],
                                         "raw_samples_sha256": hashlib.sha256(
                                             ("SYNTHETIC samples " + label + end).encode()).hexdigest()}]}]}


def lg_thd(epoch, oc, day):
    """LG + THD (IU.MAJO) Tokyo row from the real combiner; `oc` is the measurement's observation_content or None."""
    res = T.run("tokyo_kanto", T.comp("lambda_geo", 1.0), T.unavailable("fault_correlation"),
                T.comp("seismic_thd", 0.0, station="IU.MAJO", shared=("kumamoto", "tokyo_kanto")))
    m = {"observation_day": epoch, "input_sha256": C.SYNTHETIC_DIGEST + "input" + day.isoformat(),
         "input_selection_sha256": C.SYNTHETIC_DIGEST + "selection" + day.isoformat(),
         "operator_identity": C.SYNTHETIC_DIGEST + "op"}
    if oc is not None:
        m["observation_content"] = oc
    res.components["lambda_geo"].baseline_provenance = {"measurement": m}
    res.date = day
    res.thd_attempts = thd_block("IU.MAJO", day, "majo")
    return res


def issued(res):
    return json.loads(json.dumps(res.to_dict()))


def as_legacy_v2(row):
    """The row as an issued v2 report carried it: v2 regime/key, no selected-observation content."""
    out = copy.deepcopy(row)
    ms, cur = out["method_set"], MC.COMPARISON_CONTRACT_VERSION
    ms["contract_version"] = V2
    ms["regime"] = ms["regime"].replace(cur, V2)
    ms["comparability_key"] = ms["comparability_key"].replace(cur, V2)
    lg = (out.get("components") or {}).get("lambda_geo") or {}
    ((lg.get("baseline_provenance") or {}).get("measurement") or {}).pop("observation_content", None)
    return out


def legacy_streak(n=10):
    """n issued v2 WATCH+ Tokyo rows, nearest first, each on its own LG epoch and THD trace."""
    rows = []
    for k in range(n):
        day = DAY - timedelta(days=k + 2)
        rows.append(as_legacy_v2(issued(lg_thd((datetime(2026, 10, 17) - timedelta(days=k)).date().isoformat(),
                                               content("2026-10-17", "unused"), day))))
    return rows


def judged(cur, priors):
    p = MC.regime_persistence(cur.tier, cur.method_set, priors, 2, current_row=cur)

    def loader(days_back):                      # the runner's own loader shape (it reads three prior days)
        r = priors[days_back - 1] if days_back - 1 < len(priors) else None
        return None if r is None else {"regions": {"tokyo_kanto": r}}
    runner = RD.check_persistence({"tokyo_kanto": cur}, None, DAY, loader=loader)["tokyo_kanto"]
    assert runner["regime"]["not_confirmed_reasons"] == p["not_confirmed_reasons"], (runner, p)
    assert runner["is_confirmed"] == p["is_confirmed"]
    return p


class Transition(unittest.TestCase):
    def setUp(self):
        self.legacy = legacy_streak()
        self.assertTrue(all(r["tier"] >= 1 for r in self.legacy))
        self.assertTrue(all(r["method_set"]["contract_version"] == V2 for r in self.legacy))
        self.day1 = issued(lg_thd("2026-10-19", content("2026-10-19", "a"), DAY - timedelta(days=1)))

    def test_the_contract_is_v3_with_its_own_rule_and_basis(self):
        self.assertEqual((MC.COMPARISON_CONTRACT_VERSION, MC.CONFIRMATION_RULE_VERSION),
                         ("method-comparability-v3", "observation-update-confirmation-v2"))
        self.assertEqual(self.day1["method_set"]["contract_version"], "method-comparability-v3")
        self.assertFalse(hasattr(MC, "LG_OBSERVATION_CONTENT"), "the as-of-bearing content tuple is gone")

    def test_a_long_legacy_streak_resets_once_and_leaks_no_unknown(self):
        cur = lg_thd("2026-10-19", content("2026-10-19", "a"), DAY)
        p = judged(cur, self.legacy)
        self.assertEqual((p["consecutive_days"], p["is_confirmed"]), (1, False))
        self.assertEqual(p["not_confirmed_reasons"], ["REPORT_DAYS_BELOW_REQUIRED:REGIME_TRANSITION"])
        self.assertEqual(p["regime_transition"]["from"], self.legacy[0]["method_set"]["regime"])
        self.assertEqual(p["regime_transition"]["to"], MC.regime_id())
        for method in ("lambda_geo", "seismic_thd"):
            u = p["observation_updates"][method]
            self.assertEqual((u["report_days"], u["steps"], u["blocking"], u["observation_updates"]), (1, [], [], 1),
                             method)

    def test_legacy_rows_are_never_given_content_and_direct_comparison_is_unknown(self):
        old = MC.observation_identity("lambda_geo", self.legacy[0])
        self.assertEqual(old["status"], "UNKNOWN")
        self.assertIn("pre-v3", old["reason"])
        new = MC.observation_identity("lambda_geo", self.day1)
        self.assertEqual(new["status"], "KNOWN")
        self.assertEqual(MC.observation_step(new, old), MC.STEP_UNKNOWN)

    def test_the_second_v3_day_with_every_method_advanced_confirms(self):
        cur = lg_thd("2026-10-20", content("2026-10-20", "b"), DAY)
        p = judged(cur, [self.day1] + self.legacy)
        self.assertEqual((p["consecutive_days"], p["is_confirmed"], p["not_confirmed_reasons"]), (2, True, []))
        self.assertEqual(p["observation_updates"]["lambda_geo"]["steps"], [MC.STEP_ADVANCED])
        self.assertEqual(p["observation_updates"]["seismic_thd"]["steps"], [MC.STEP_ADVANCED])
        self.assertEqual(p["observation_updates"]["lambda_geo"]["observation_updates"], 2)

    def test_nothing_but_an_advance_is_a_second_update(self):
        cases = {
            "repeat": ("2026-10-19", content("2026-10-19", "a"), "REPEATED_INPUT:lambda_geo"),
            "revision": ("2026-10-19", content("2026-10-19", "b"), "REVISED_SAME_EPOCH:lambda_geo"),
            "regression": ("2026-10-18", content("2026-10-18", "c"), "EPOCH_REGRESSED:lambda_geo"),
            "missing": ("2026-10-20", None, "UNKNOWN_OBSERVATION_IDENTITY:lambda_geo"),
            "schema": ("2026-10-20", content("2026-10-20", "d", schema="lg-observation-content-v0"),
                       "UNKNOWN_OBSERVATION_IDENTITY:lambda_geo"),
            "encoding": ("2026-10-20", content("2026-10-20", "d", encoding="json-v0"),
                         "UNKNOWN_OBSERVATION_IDENTITY:lambda_geo"),
            "status": ("2026-10-20", content("2026-10-20", "d", status="UNKNOWN", reason="INPUT_DIGEST_MISMATCH"),
                       "UNKNOWN_OBSERVATION_IDENTITY:lambda_geo"),
            "epoch": ("2026-10-20", content("2026-10-19", "d"), "UNKNOWN_OBSERVATION_IDENTITY:lambda_geo"),
            "short": ("2026-10-20", content("2026-10-20", "d", sha256="a" * 63), "UNKNOWN_OBSERVATION_IDENTITY:lambda_geo"),
            "upper": ("2026-10-20", content("2026-10-20", "d", sha256="A" * 64), "UNKNOWN_OBSERVATION_IDENTITY:lambda_geo"),
            "type": ("2026-10-20", content("2026-10-20", "d", sha256=7), "UNKNOWN_OBSERVATION_IDENTITY:lambda_geo"),
            "not_a_mapping": ("2026-10-20", "lg-observation-content-v1:" + "a" * 64,
                              "UNKNOWN_OBSERVATION_IDENTITY:lambda_geo"),
        }
        for name, (epoch, oc, reason) in cases.items():
            with self.subTest(case=name):
                p = judged(lg_thd(epoch, oc, DAY), [self.day1] + self.legacy)
                self.assertEqual((p["consecutive_days"], p["is_confirmed"]), (2, False))
                self.assertEqual(p["not_confirmed_reasons"], [reason])
                self.assertNotEqual(p["observation_updates"]["lambda_geo"]["observation_updates"], 2)
                self.assertEqual(p["observation_updates"]["seismic_thd"]["steps"], [MC.STEP_ADVANCED])

    def test_an_unknown_prior_identity_blocks_an_otherwise_advanced_pair(self):
        prior = issued(lg_thd("2026-10-19", None, DAY - timedelta(days=1)))   # a v3 run that did not record it
        p = judged(lg_thd("2026-10-20", content("2026-10-20", "b"), DAY), [prior] + self.legacy)
        self.assertEqual((p["consecutive_days"], p["is_confirmed"]), (2, False))
        self.assertEqual(p["not_confirmed_reasons"], ["UNKNOWN_OBSERVATION_IDENTITY:lambda_geo"])

    def test_the_first_v3_day_discloses_its_own_unknown_identity(self):
        p = judged(lg_thd("2026-10-19", None, DAY), self.legacy)
        self.assertEqual(p["not_confirmed_reasons"], ["REPORT_DAYS_BELOW_REQUIRED:REGIME_TRANSITION",
                                                      "UNKNOWN_OBSERVATION_IDENTITY:lambda_geo"])


class ThdOnlyTransitionCost(unittest.TestCase):
    """The global contract restarts THD-only WATCH+ sequences once too (codex answer 2: disclosed, no score moves)."""

    def row(self, day, label):
        res = T.thd_only("turkey_kahramanmaras", 0.9)
        res.date = day
        res.thd_attempts = thd_block("IU.ANTO", day, label)
        return res

    def test_thd_only_pays_the_reset_once_then_confirms(self):
        first = self.row(DAY - timedelta(days=1), "d1")
        self.assertGreaterEqual(first.tier, 1)
        legacy = [as_legacy_v2(issued(self.row(DAY - timedelta(days=k + 2), "L%d" % k))) for k in range(5)]
        p1 = MC.regime_persistence(first.tier, first.method_set, legacy, 2, current_row=first)
        self.assertEqual((p1["consecutive_days"], p1["is_confirmed"], p1["not_confirmed_reasons"]),
                         (1, False, ["REPORT_DAYS_BELOW_REQUIRED:REGIME_TRANSITION"]))
        cur = self.row(DAY, "d2")
        p2 = MC.regime_persistence(cur.tier, cur.method_set, [issued(first)] + legacy, 2, current_row=cur)
        self.assertEqual((p2["consecutive_days"], p2["is_confirmed"], p2["not_confirmed_reasons"]), (2, True, []))
        self.assertEqual(first.combined_risk, issued(self.row(DAY - timedelta(days=1), "d1"))["combined_risk"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
