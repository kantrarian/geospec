#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M-F4 MATURITY RECORD generator (cayley) -- codex 0542Z shape-review
condition 4.

WHY THIS IS SEPARATE FROM THE MAG CUTOFF
----------------------------------------
The successor caps the MAG lanes at 2026-07-31 because NASA publishes
high-resolution SYM/H only through then. **M-F4 does not use that
interval at all.** Its frozen calibration runs

    [CAL_START, calibration_issue_end]
    calibration_issue_end = min(freeze_day - H, snapshot_end - H)

with `CAL_START = 2025-10-18` and `H = 7` days, straight from the
frozen `w2_mf4` constants. Re-using 07-31 as an M-F4 cutoff would be a
category error, so this record derives the bound from the frozen
equation instead of inheriting a neighbouring lane's number.

THE TWO GATES codex REQUIRES KEPT APART
---------------------------------------
(a) **Calibration-ledger sufficiency** -- does the historical daily-
    risk archive already reach `calibration_issue_end`? This module
    answers that arithmetically from the frozen equation.
(b) **Prospective producer validity** -- is there a technically valid
    daily-risk producer for every ACCRUAL day through the maturity
    tail? Historical sufficiency does NOT prove this, and the 08-22
    owner continuity authorization is NOT a bypass for an expired
    producer capsule. This module records the requirement; it cannot
    discharge it.

Opens no window-2 value; no network; admits nothing.
"""
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

OUT_REL = os.path.join("docs", "f2g_window2_execution",
                       "mf4_maturity_record_v4.json")
SCHEMA = "f2g-w2-mf4-maturity-record-v4"
# the successor schedule (prestart_schedule_2026-08-28.md)
PRESTART_DAY = "2026-08-28"
SELECTION_CUTOFF = "2026-08-27"      # the catalog snapshot cutoff
# the D2 daily-monitor capsule's registered coverage (renewal lift
# ab14a58): scored days are admitted through this date
D2_SCORED_THROUGH = "2026-08-23"


def build():
    import w2_mf4 as MF4

    freeze_day = PRESTART_DAY
    snapshot_end = SELECTION_CUTOFF
    issue_end = MF4.calibration_issue_end(freeze_day,
                                          snapshot_end).isoformat()
    # (a) does the historical archive already reach the matured bound?
    sufficient = issue_end <= D2_SCORED_THROUGH
    margin = ((MF4._d(D2_SCORED_THROUGH) - MF4._d(issue_end)).days
              if sufficient else
              -(MF4._d(issue_end) - MF4._d(D2_SCORED_THROUGH)).days)
    return {
        "schema": SCHEMA,
        "why_separate": "M-F4 does not use the MAG 01-01..07-31 "
                        "interval; its bound derives from the frozen "
                        "maturity equation, never from a neighbouring "
                        "lane's cutoff (codex 0542Z condition 4)",
        "frozen_constants": {
            "cal_start": MF4.CAL_START,
            "h_days": MF4.H_DAYS,
            "equation": "calibration_issue_end = min(freeze_day - H, "
                        "snapshot_end - H)",
            "source": "monitoring/src/w2_mf4.py (frozen)"},
        "gate_a_calibration_ledger": {
            "freeze_day": freeze_day,
            "snapshot_end": snapshot_end,
            "calibration_issue_end": issue_end,
            "calibration_interval": [MF4.CAL_START, issue_end],
            "d2_archive_scored_through": D2_SCORED_THROUGH,
            "historical_rows_sufficient": bool(sufficient),
            "margin_days": margin,
            "finding": (
                f"the matured bound {issue_end} is "
                f"{'within' if sufficient else 'BEYOND'} the "
                f"historical daily-risk archive "
                f"({D2_SCORED_THROUGH}), margin {margin} d -- so the "
                "D2 capsule expiry does NOT threaten the M-F4 "
                "CALIBRATION LEDGER"
                if sufficient else
                "the matured bound lies BEYOND the archive; the "
                "ledger cannot close on historical rows alone"),
            "still_required_from_the_archive": {
                "risk_series_support": "OPEN -- per-region support "
                                       "census over [cal_start, "
                                       "issue_end], from the "
                                       "daily-risk archive",
                "training_digest": "OPEN -- canonical digest of the "
                                   "exact rows fitted",
                "catalog_snapshot": "OPEN -- pinned snapshot + its "
                                    "digest",
                "note": "these live on the monitor host; this record "
                        "fixes the BOUND they must satisfy so they "
                        "cannot be chosen after the fact"}},
        "gate_b_prospective_producer": {
            "requirement": "a technically valid daily-risk producer "
                           "for EVERY accrual day through the "
                           "maturity tail",
            "accrual_span": "evaluation_start .. evaluation_end + H "
                            "(2026-08-29 .. 2027-01-14 under the "
                            "08-28 successor schedule)",
            "status": "OPEN -- the D2 producer capsule expires after "
                      "scored day 2026-08-23; gate (a) sufficiency "
                      "does NOT discharge this",
            "authorization_note": "the 2026-08-22 owner artifact "
                                  "already authorizes daily-monitor "
                                  "CONTINUITY through window-2 "
                                  "close; that authorization is NOT "
                                  "a bypass for an expired producer "
                                  "capsule, and re-asking the owner "
                                  "for continuity already granted "
                                  "would be wrong. This routes as a "
                                  "TECHNICAL renewal/proof before "
                                  "PRESTART (codex 0542Z cond. 4)"},
        "claim_ceiling": {
            "scientific_admission": "NONE -- this records bounds and "
                                    "requirements only",
            "lambda_geo": "INCONCLUSIVE"},
        "producer": "monitoring/src/w2_mf4_maturity_record_gen_"
                    "cayley.py"}


def main():
    repo = os.path.abspath(os.path.join(_HERE, "..", ".."))
    out = json.dumps(build(), indent=1, sort_keys=True) + "\n"
    with open(os.path.join(repo, OUT_REL), "w", encoding="utf-8",
              newline="\n") as f:
        f.write(out)
    print(f"wrote {OUT_REL.replace(os.sep, '/')}")
    print("artifact sha256:",
          hashlib.sha256(out.encode()).hexdigest())


def _selftest():
    import w2_mf4 as MF4
    import w2_expected_contracts_gen_cayley as GEN

    a, b = build(), build()
    assert a == b, "the maturity record must be deterministic"
    ga = a["gate_a_calibration_ledger"]
    # the bound must come from the FROZEN equation, recomputed here
    want = MF4.calibration_issue_end(ga["freeze_day"],
                                     ga["snapshot_end"]).isoformat()
    assert ga["calibration_issue_end"] == want
    # M-F4 must NOT inherit the MAG cutoff -- the category error the
    # condition exists to prevent
    assert ga["calibration_issue_end"] != GEN.MAG_CUTOFF, \
        "M-F4 must not reuse the MAG 07-31 cutoff"
    assert a["frozen_constants"]["cal_start"] == "2025-10-18"
    assert a["frozen_constants"]["h_days"] == 7
    # gate (b) must stay OPEN: sufficiency of history never closes it
    assert a["gate_b_prospective_producer"]["status"].startswith(
        "OPEN"), "gate (b) cannot be discharged by gate (a)"
    print("w2_mf4_maturity_record selftest: ALL PASS "
          f"(issue_end {ga['calibration_issue_end']}, historical "
          f"sufficient={ga['historical_rows_sufficient']} margin "
          f"{ga['margin_days']}d, gate (b) OPEN)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
