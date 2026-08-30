#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 CALENDAR AUTHORITY v4 generator (cayley) -- codex
Gate-2 ruling 2026-08-30T18:44Z repair 2 (REFUSE_CURRENT_REVISION at
public 8cc564e7): the registered 08-28 PRESTART was missed (composed
prestart completed 2026-08-30); the v3 calendar's own rule says a
missed PRESTART "supersedes the schedule, never slides these dates".
This is the APPEND-ONLY SUCCESSOR: v3 stands as history; nothing is
backdated to 08-29 and no v3 date is silently carried forward.

This is the PRODUCER of docs/f2g_window2_execution/
calendar_authority_w2_v4.json. Like the v3 producer it derives the
date arrays with its OWN datetime logic (independently of
w2_power_harness_cayley's w2_calendar_frame) so the harness
selftest's equality check crosses two derivations, not one.
Deterministic: same bytes every run.

The single moving constant is PRESTART_DAY. Everything derives:
  cutoff          = PRESTART_DAY - 1 day  (last COMPLETE UTC day at
                                           execution)
  baseline_days   = 60 exact UTC dates ending at cutoff
  excluded_days   = [PRESTART_DAY]        (never an engine position)
  evaluation_days = 132 exact UTC dates starting PRESTART_DAY + 1
                    (evaluation_start = first UTC day after the
                     completed PASS prestart, per the Gate-2 repair)
  engine_days     = baseline || evaluation (192 positions)
  B1B authority: n_blocks=16, block_len=12, baseline_positions=60
  (five baseline blocks, eleven evaluation blocks -- structure
   identical to v3; only the dates move)

PRESTART_DAY = 2026-09-03 is OWNER-WORDED: asylum in-session
2026-08-30, verbatim "proceed eval starting on 9/04" (quote
sha256 56a08a16...), which fixes evaluation_start 2026-09-04
and hence PRESTART 2026-09-03 -- the proposed constant,
unchanged.

Claim ceiling: calendar registration only; no power value, no
scientific claim, no evaluation open; Lambda_geo INCONCLUSIVE.
"""
import datetime
import hashlib
import json
import os

OUT_REL = os.path.join("docs", "f2g_window2_execution",
                       "calendar_authority_w2_v4.json")

PRESTART_DAY = "2026-09-03"
BASELINE_LEN = 60
EVAL_LEN = 132


def _span_len(first, n):
    d0 = datetime.date.fromisoformat(first)
    return [(d0 + datetime.timedelta(days=i)).isoformat()
            for i in range(n)]


def _digest(days):
    return hashlib.sha256(json.dumps(
        list(days), separators=(",", ":")).encode()).hexdigest()


def build():
    p = datetime.date.fromisoformat(PRESTART_DAY)
    cutoff = (p - datetime.timedelta(days=1)).isoformat()
    baseline_start = (p - datetime.timedelta(
        days=BASELINE_LEN)).isoformat()
    eval_start = (p + datetime.timedelta(days=1)).isoformat()
    baseline = _span_len(baseline_start, BASELINE_LEN)
    evaluation = _span_len(eval_start, EVAL_LEN)
    engine = baseline + evaluation
    assert baseline[-1] == cutoff
    assert len(baseline) == BASELINE_LEN and \
        len(evaluation) == EVAL_LEN
    assert len(engine) == 192 and PRESTART_DAY not in engine
    assert engine == sorted(engine) and len(set(engine)) == 192
    assert cutoff < eval_start and cutoff < PRESTART_DAY
    b1b = {"n_blocks": 16, "block_len": 12, "baseline_positions": 60}
    assert b1b["n_blocks"] * b1b["block_len"] == len(engine)
    assert b1b["baseline_positions"] == len(baseline)
    assert b1b["baseline_positions"] // b1b["block_len"] == 5
    assert (len(engine) - b1b["baseline_positions"]) \
        // b1b["block_len"] == 11
    frame = {"frame_id": "w2-calendar-v4-noncal",
             "baseline_days": baseline,
             "excluded_days": [PRESTART_DAY],
             "evaluation_days": evaluation,
             "engine_days": engine,
             "b1b": b1b}
    return {
        "schema": "f2g-w2-calendar-authority-v4",
        "frame": frame,
        "digests": {
            "baseline_days_sha256": _digest(baseline),
            "evaluation_days_sha256": _digest(evaluation),
            "engine_days_sha256": _digest(engine)},
        "cutoff_binding": {
            "cutoff": cutoff,
            "rule": "the greatest UTC date whose complete-day bytes "
                    "exist before PRESTART execution begins; "
                    "strictly earlier than evaluation_start; the "
                    "PRESTART day is excluded; a missed " +
                    PRESTART_DAY + " PRESTART supersedes the "
                    "schedule, never slides these dates"},
        "supersedes": {
            "artifact": "docs/f2g_window2_execution/"
                        "calendar_authority_w2_v3.json",
            "reason": "registered 08-28 PRESTART missed (composed "
                      "prestart completed 2026-08-30); append-only "
                      "successor per the v3 rule and codex Gate-2 "
                      "2026-08-30T18:44Z repair 2; v3 stands as "
                      "history; no backdating to 2026-08-29"},
        "provenance": {
            "ruling": "codex Gate-2 REFUSE_CURRENT_REVISION "
                      "2026-08-30T18:44Z repair 2 (append-only "
                      "successor calendar; evaluation_start = first "
                      "UTC day after a future completed PASS "
                      "prestart; no silent v3 carry-forward)",
            "owner_decision": "asylum in-session 2026-08-30 "
                              "'proceed eval starting on 9/04' "
                              "(quote sha 56a08a16, SUCCESSOR "
                              "schedule artifact prestart_schedule_"
                              "2026-09-03.md; supersedes the 08-28 "
                              "schedule per its no-silent-absorption "
                              "clause and codex Gate-2 repair 2)",
            "producer": "monitoring/src/"
                        "w2_calendar_authority_gen_v4_cayley.py",
            "claim_ceiling": "calendar registration only; no power "
                             "value, no scientific claim; Lambda_geo "
                             "INCONCLUSIVE"}}


def main():
    repo = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    out_path = os.path.join(repo, OUT_REL)
    body = json.dumps(build(), indent=1, sort_keys=True) + "\n"
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
    print(f"wrote {OUT_REL}")
    print("artifact sha256:",
          hashlib.sha256(body.encode()).hexdigest())


if __name__ == "__main__":
    main()
