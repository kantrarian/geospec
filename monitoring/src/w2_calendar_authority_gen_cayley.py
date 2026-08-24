#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 CALENDAR AUTHORITY v3 generator (cayley) -- codex
2026-08-23T1400Z ruling 1 under the owner PRESTART decision
("prestart 8/26 it is", sha 5b44934a).

This is the PRODUCER of docs/f2g_window2_execution/
calendar_authority_w2_v3.json. It derives the date arrays with its
OWN datetime logic (independently of w2_power_harness_cayley's
w2_calendar_frame) so the harness selftest's equality check crosses
two derivations, not one. Deterministic: same bytes every run.

Registered objects (codex 1400Z, verbatim contract):
  baseline_days   = 2026-06-29 .. 2026-08-27  (60 exact UTC dates)
  excluded_days   = [2026-08-28]              (PRESTART; never an
                                               engine position)
  evaluation_days = 2026-08-29 .. 2027-01-07  (132 exact UTC dates)
  engine_days     = baseline || evaluation    (192 positions)
  B1B authority: n_blocks=16, block_len=12, baseline_positions=60
  (five baseline blocks, eleven evaluation blocks)

Cutoff binding (codex ratification, rule carried to the successor):
cutoff = 2026-08-27 is the
greatest UTC date whose complete-day bytes exist before PRESTART
execution begins, strictly earlier than evaluation_start; the
PRESTART day is excluded. If PRESTART does not complete on 08-28 the
schedule is SUPERSEDED, never silently slid.
"""
import datetime
import hashlib
import json
import os

OUT_REL = os.path.join("docs", "f2g_window2_execution",
                       "calendar_authority_w2_v3.json")


def _span(a, b):
    d0 = datetime.date.fromisoformat(a)
    d1 = datetime.date.fromisoformat(b)
    assert d0 <= d1
    return [(d0 + datetime.timedelta(days=i)).isoformat()
            for i in range((d1 - d0).days + 1)]


def _digest(days):
    return hashlib.sha256(json.dumps(
        list(days), separators=(",", ":")).encode()).hexdigest()


def build():
    baseline = _span("2026-06-29", "2026-08-27")
    evaluation = _span("2026-08-29", "2027-01-07")
    engine = baseline + evaluation
    assert len(baseline) == 60 and len(evaluation) == 132
    assert len(engine) == 192 and "2026-08-28" not in engine
    assert engine == sorted(engine) and len(set(engine)) == 192
    b1b = {"n_blocks": 16, "block_len": 12, "baseline_positions": 60}
    assert b1b["n_blocks"] * b1b["block_len"] == len(engine)
    assert b1b["baseline_positions"] == len(baseline)
    assert b1b["baseline_positions"] // b1b["block_len"] == 5
    assert (len(engine) - b1b["baseline_positions"]) \
        // b1b["block_len"] == 11
    frame = {"frame_id": "w2-calendar-v3-noncal",
             "baseline_days": baseline,
             "excluded_days": ["2026-08-28"],
             "evaluation_days": evaluation,
             "engine_days": engine,
             "b1b": b1b}
    return {
        "schema": "f2g-w2-calendar-authority-v3",
        "frame": frame,
        "digests": {
            "baseline_days_sha256": _digest(baseline),
            "evaluation_days_sha256": _digest(evaluation),
            "engine_days_sha256": _digest(engine)},
        "cutoff_binding": {
            "cutoff": "2026-08-27",
            "rule": "the greatest UTC date whose complete-day bytes "
                    "exist before PRESTART execution begins; "
                    "strictly earlier than evaluation_start; the "
                    "PRESTART day is excluded; a missed 08-28 "
                    "PRESTART supersedes the schedule, never slides "
                    "these dates"},
        "provenance": {
            "ruling": "codex 2026-08-23T1400Z ruling 1 (calendar "
                      "option (a) with non-compression adapter "
                      "contract)",
            "owner_decision": "asylum in-session 'restart 8-28 is "
                              "fine' (quote sha c2fdcf76, SUCCESSOR "
                              "schedule artifact prestart_schedule_"
                              "2026-08-28.md; supersedes the 08-26 "
                              "schedule per its no-silent-absorption "
                              "clause)",
            "producer": "monitoring/src/"
                        "w2_calendar_authority_gen_cayley.py",
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
