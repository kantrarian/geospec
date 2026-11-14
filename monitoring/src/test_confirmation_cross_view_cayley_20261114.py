"""Confirmation cross-view fixture and checks (cayley 2026-11-14; codex 93df4f774 item 3).

ONE saved fixture, produced here from the issued 2026-11-12 report through the REAL run_ensemble_daily.check_persistence
and print_summary and the REAL raw_history_cayley.day_from_report, must give the same flag, report days, observation
counts, dates and reasons in the report JSON, the printed summary/alert, the raw-history day the Pages download reads, and
(test_confirmation_card.js) the Pages card line and download columns. Three regions, all contract v2:
  tokyo_kanto    v2 pair, same LG observation 2026-10-19  -> NOT confirmed, REPEATED_INPUT:lambda_geo
  campi_flegrei  v2 pair, genuinely later THD trace        -> confirmed
  kaikoura       v2 pair, issued THD record has no sample digest -> NOT confirmed, UNKNOWN_OBSERVATION_IDENTITY:seismic_thd
Synthetic parts (the later reports, their THD trace digests) are labelled; nothing here is a forecast of a run.
Usage: py -3.11 -B test_confirmation_cross_view_cayley_20261114.py            (check)
       py -3.11 -B test_confirmation_cross_view_cayley_20261114.py --write    (regenerate the fixture once, reviewed)
"""
import contextlib
import copy
import io
import json
import os
import sys
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import test_confirmation_observation_updates_cayley_20261114 as C  # noqa: E402  (stubs + issued fixture + helpers)
import raw_history_cayley as RH  # noqa: E402

RD, MC = C.RD, C.MC
FIXTURE = Path(HERE).parent / "test_fixtures" / "confirmation_cross_view_fixture_cayley_20261114.json"
REGIONS = ("tokyo_kanto", "campi_flegrei", "kaikoura")


def build():
    rep = C.issued_report()
    regions = {}
    for region in REGIONS:
        issued = rep["regions"][region]
        prior = C.as_v2(C.next_day(issued))           # 2026-11-13, contract v2 (SYNTHETIC later report)
        cur = C.next_day(prior)                        # 2026-11-14, contract v2 (SYNTHETIC later report)
        if region == "kaikoura":                       # keep the issued record's missing digest, never invent one
            for row in (prior, cur):
                for st in row["thd_attempts"]["stations"]:
                    for p in st.get("providers") or []:
                        p.pop("raw_samples_sha256", None)
        pers = C.persistence(cur, [prior, issued], region)
        cur["persistence"] = json.loads(json.dumps(pers))
        regions[region] = cur
    report = {"date": "2026-11-14", "schema_note": "SYNTHETIC v2 report built from the issued 2026-11-12 rows",
              "regions": regions}
    results = {r: C.SimpleNamespace(tier=row["tier"], tier_name=row["tier_name"], combined_risk=row["combined_risk"],
                                    confidence=row["confidence"], methods_available=row["methods_available"])
               for r, row in regions.items()}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        RD.print_summary(results, {r: row["persistence"] for r, row in regions.items()})
    raw_days = {r: RH.day_from_report(report, r, {"kind": "SYNTHETIC_FIXTURE"}) for r in REGIONS}
    return {"report": report, "printed_summary": buf.getvalue(), "raw_history_days": raw_days,
            "expected": {
                "tokyo_kanto": {"is_confirmed": False, "report_days": 2, "lg_observation_updates": 1, "lg_latest": "2026-10-19",
                                "thd_observation_updates": 2, "reasons": ["REPEATED_INPUT:lambda_geo"],
                                "card_status": "NOT_CONFIRMED",
                                "card_text": "Not confirmed: 2 report days; LG 1 observation (latest 2026-10-19); THD 2 observations"
                                             " — LG repeated input (the same observation as the previous report)"},
                "campi_flegrei": {"is_confirmed": True, "report_days": 2, "thd_observation_updates": 2, "reasons": [],
                                  "card_status": "CONFIRMED", "card_text": "Confirmed: 2 report days; THD 2 observations"},
                "kaikoura": {"is_confirmed": False, "report_days": 2, "thd_observation_updates": None,
                             "reasons": ["UNKNOWN_OBSERVATION_IDENTITY:seismic_thd"], "card_status": "NOT_CONFIRMED",
                             "card_text": "Not confirmed: 2 report days; THD observation unknown — THD observation identity "
                                          "missing from the retained evidence (a value exists; a new observation cannot be "
                                          "established)"}}}


def dumps(obj):
    return json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


class CrossView(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.built = build()
        cls.saved = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_the_saved_fixture_is_exactly_what_the_writers_produce(self):
        self.assertEqual(dumps(self.saved), dumps(self.built))

    def test_report_json_carries_the_expected_flag_counts_dates_and_reasons(self):
        for region, e in self.saved["expected"].items():
            p = self.saved["report"]["regions"][region]["persistence"]
            u = p["regime"]["observation_updates"]
            self.assertEqual(p["is_confirmed"], e["is_confirmed"], region)
            self.assertEqual((p["consecutive_days"], p["regime"]["report_days"]), (e["report_days"],) * 2, region)
            self.assertEqual(p["regime"]["not_confirmed_reasons"], e["reasons"], region)
            self.assertEqual(p["regime"]["contract_version"] if "contract_version" in p["regime"] else
                             self.saved["report"]["regions"][region]["method_set"]["contract_version"], "method-comparability-v2")
            self.assertEqual(u["seismic_thd"]["observation_updates"], e["thd_observation_updates"], region)
            if "lg_observation_updates" in e:
                self.assertEqual(u["lambda_geo"]["observation_updates"], e["lg_observation_updates"])
                self.assertEqual(u["lambda_geo"]["epochs"][0], e["lg_latest"])

    def test_printed_summary_and_alert_agree(self):
        s = self.saved["printed_summary"]
        self.assertIn("CONFIRMED(2d)", s)                         # campi_flegrei
        self.assertEqual(s.count("PRELIM(2d)"), 2)                # tokyo_kanto, kaikoura
        alert = s.split("*** ALERT: CONFIRMED elevated regions (2+ consecutive days) ***")
        self.assertEqual(len(alert), 2, s)
        self.assertIn("campi_flegrei", alert[1].split("Note:")[0])
        self.assertNotIn("tokyo_kanto", alert[1].split("Note:")[0])

    def test_raw_history_day_copies_the_issued_persistence(self):
        for region, e in self.saved["expected"].items():
            issued = self.saved["report"]["regions"][region]["persistence"]
            got = self.saved["raw_history_days"][region]["combined"]["persistence"]
            self.assertEqual((got["is_confirmed"], got["consecutive_days"], got["report_days"]),
                             (issued["is_confirmed"], issued["consecutive_days"], issued["regime"]["report_days"]))
            self.assertEqual(got["not_confirmed_reasons"], e["reasons"])
            self.assertEqual(got["confirmation_rule"], MC.CONFIRMATION_RULE_VERSION)
            for m, u in issued["regime"]["observation_updates"].items():
                self.assertEqual(got["observation_updates"][m]["observation_updates"], u["observation_updates"])
                self.assertEqual(got["observation_updates"][m]["epochs"], u["epochs"])

    def test_a_report_issued_before_the_rule_is_copied_without_invented_fields(self):
        rep = C.issued_report()
        got = RH.persistence_entry(rep["regions"]["kaikoura"])
        self.assertEqual((got["is_confirmed"], got["consecutive_days"]), (True, 2))     # as issued 2026-11-12
        self.assertIsNone(got["confirmation_rule"])
        self.assertIsNone(got["not_confirmed_reasons"])
        self.assertIsNone(got["observation_updates"])


if __name__ == "__main__":
    if "--write" in sys.argv:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        with open(FIXTURE, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(dumps(build()))
        print("wrote", FIXTURE)
    else:
        unittest.main(verbosity=2)
