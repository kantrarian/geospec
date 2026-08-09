#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D2 STEP-4B CAMPAIGN red-KATs (cayley, 2026-08-09) — contract
`codex-d2-step4b-2026-08-09-v1` (codex 0129, `ecd5bff`); pinned segmented implementation
GeoSpec `3950a2c`. These bars gate grassmann's campaign PRODUCER (red-first); codex's step-4b
acceptance entry point separately verifies the produced BATCH. Nothing here fetches, lifts,
tunes, or claims; the campaign's first archive request additionally requires the DIRECT verifiable
owner launch go in grassmann's session (SB-8) — this bar cannot and does not substitute for it.

CONTRACT SEAMS (grassmann implements `monitoring/src/d2_step4b_producer.py` to THIS, unedited)
==============================================================================================
* CAMPAIGN = {"contract_id": "codex-d2-step4b-2026-08-09-v1",
  "incident_reference": "2026-07-29", "min_admitted_days": 60, "window_days": 90,
  "lag_days": 30, "providers": {"istanbul_marmara": "eida.koeri.boun.edu.tr",
  "turkey_kahramanmaras": "eida.koeri.boun.edu.tr", "socal_coachella": "s3://scedc-pds"}}.
* schedule_days(reference_day: str) -> list[str] — the exact half-open [ref-120d, ref-30d),
  90 calendar days, ISO dates ascending.
* build_campaign_plan(carriers: dict, activation_reference: str) -> dict
    - carriers: {carrier: {segment: [ordered NSLC candidate lists per station]}}; validates
      >= 2 stations per segment and >= 2 segments per carrier (fail-closed ValueError);
      providers only from CAMPAIGN["providers"] (unknown carrier -> ValueError);
    - output binds both arm schedules, per-station ORDERED NSLC candidate lists, and provider
      endpoints; NO outcome-bearing field is accepted or emitted (ratios/thresholds/labels);
    - DETERMINISTIC: identical inputs -> byte-identical canonical JSON (sort_keys, ',:', LF).
* plan_digest(plan) -> 64-hex sha256 over the canonical plan bytes.
* session_from_record(record_bytes: bytes) -> (start_utc, end_utc)
    - parses the published daily-monitoring record's EXACT half-open request interval
      (aware UTC); the registered session MUST be exactly 86,400.000000 s (else ValueError);
      malformed/missing interval -> ValueError. NO inferred/midnight/nearest fallback exists
      anywhere in the producer: a scheduled day with no record is
      UNAVAILABLE_NO_PUBLISHED_RECORD, gets NO archive request and NO attempt row.
* select_channel(candidates: list[str], available: set[str]) -> str | None
    - the FIRST candidate (frozen order) present in `available`; None if none; there is no
      post-QC swap seam — QC failure of fetched data never re-enters selection.
* threshold_from_admitted(ratios: list[float]) -> float | None
    - None when n < 60; else sort ascending and return ratios_sorted[ceil(0.05*n)-1]
      (nearest-rank lower 5% quantile, zero-based).
* derive_replay_ratios(prior_evidence_bytes: bytes, expected_sha256: str) -> dict
    - verifies sha256(prior_evidence_bytes) == expected pin, then extracts the six sealed
      control/incident ratios FROM THE BYTES; producer-entered ratio values have no path in.
* admit_candidate(carrier, incident_summary, activation_summary, replay: dict) -> (status, info)
    - ADMITTED_CANDIDATE iff BOTH arms have >= 60 admitted days AND reproducible thresholds AND
      replay has both ratios AND incident_ratio >= incident_threshold AND
      control_ratio >= incident_threshold; else deterministically one of
      BLOCKED_INSUFFICIENT_CALIBRATION / BLOCKED_REPLAY_UNAVAILABLE /
      BLOCKED_ARTIFACT_PERSISTS / BLOCKED_NEGATIVE_CONTROL.
* verify_launch_authorization(receipt: dict) -> bool
    - True iff receipt == {"status": "VERIFIED_DIRECT", "in_session_timestamp_utc": <parseable
      aware-UTC>, "owner_quote_sha256": <64-hex>}; RELAYED/missing/malformed -> False; and the
      producer's fetch entry point (run_campaign / fetch phase) REFUSES to issue any archive
      request when this is False (SystemExit/exception before any provider I/O).
* attempt-row semantics (locked at plan level here; batch-level closure is codex's bar):
  exactly ONE summary row per REGISTERED station/scheduled-day with a published record; statuses
  in {FETCHED, UNAVAILABLE, ERROR}; overlap days between the two arms are single acquisition
  entries reused byte-for-byte (same object hashes in both arm rows).

RED AS AUTHORED (`d2_step4b_producer.py` absent / seams absent).
"""
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

FAILS = []


def check(desc, ok, detail=""):
    print(f"    [{'PASS' if ok else 'FAIL'}] {desc}" + (f" - {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(desc)


def raises(fn, exc=Exception):
    try:
        fn()
        return False
    except exc:
        return True


CARRIERS_OK = {
    "istanbul_marmara": {
        "izmit": [["KO.SAUV..HHZ", "KO.SAUV..BHZ"], ["KO.GAZK..HHZ"]],
        "marmara_west": [["KO.NMR6..HHZ"], ["KO.NMR3..HHZ"], ["KO.BOTS..HHZ"]],
    },
    "socal_coachella": {
        "coachella_south": [["CI.BOR..BHZ", "CI.BOR..HHZ"], ["CI.TRO..BHZ"]],
        "brawley_seismic_zone": [["CI.WMC..BHZ"], ["CI.RXH..BHZ"]],
    },
}


def main():
    try:
        import d2_step4b_producer as P
    except ImportError:
        check("SB-0 producer module import (d2_step4b_producer.py)", False,
              "AWAITING grassmann's producer -- red-first as authored")
        return
    need = ("CAMPAIGN", "schedule_days", "build_campaign_plan", "plan_digest",
            "session_from_record", "select_channel", "threshold_from_admitted",
            "derive_replay_ratios", "admit_candidate", "verify_launch_authorization")
    if not all(hasattr(P, n) for n in need):
        check("SB-0b step-4b producer seams present", False,
              "AWAITING grassmann's producer -- red-first as authored")
        return

    # ---- SB-1: frozen campaign constants -----------------------------------
    C = P.CAMPAIGN
    check("SB-1 frozen campaign constants (contract id, references, 60/90/30, provider cap)",
          C.get("contract_id") == "codex-d2-step4b-2026-08-09-v1"
          and C.get("incident_reference") == "2026-07-29"
          and C.get("min_admitted_days") == 60 and C.get("window_days") == 90
          and C.get("lag_days") == 30
          and C.get("providers", {}).get("socal_coachella") == "s3://scedc-pds"
          and C.get("providers", {}).get("istanbul_marmara") == "eida.koeri.boun.edu.tr"
          and C.get("providers", {}).get("turkey_kahramanmaras") == "eida.koeri.boun.edu.tr"
          and set(C.get("providers", {})) == {"istanbul_marmara", "turkey_kahramanmaras",
                                              "socal_coachella"},
          f"CAMPAIGN={C}")

    # ---- SB-2: exact arm schedules -----------------------------------------
    inc = P.schedule_days("2026-07-29")
    check("SB-2 incident arm = exactly [2026-03-31, 2026-06-29), 90 ascending days",
          len(inc) == 90 and inc[0] == "2026-03-31" and inc[-1] == "2026-06-28"
          and inc == sorted(inc) and len(set(inc)) == 90,
          f"n={len(inc)} first={inc[0] if inc else None} last={inc[-1] if inc else None}")
    act = P.schedule_days("2026-08-09")
    check("SB-2b activation arm from reference A follows [A-120d, A-30d)",
          len(act) == 90 and act[0] == "2026-04-11" and act[-1] == "2026-07-09",
          f"first={act[0] if act else None} last={act[-1] if act else None}")

    # ---- SB-3: outcome-blind, deterministic plan ---------------------------
    p1 = P.build_campaign_plan(CARRIERS_OK, "2026-08-09")
    p2 = P.build_campaign_plan(CARRIERS_OK, "2026-08-09")
    b1 = json.dumps(p1, sort_keys=True, separators=(",", ":"))
    b2 = json.dumps(p2, sort_keys=True, separators=(",", ":"))
    d1 = P.plan_digest(p1)
    check("SB-3 plan is deterministic (byte-identical on identical inputs) with a 64-hex digest",
          b1 == b2 and d1 == P.plan_digest(p2) and len(d1) == 64)
    plan_text = b1.lower()
    check("SB-3b plan is OUTCOME-BLIND: no ratio/threshold/lambda/admitted fields anywhere",
          all(tok not in plan_text for tok in ('"ratio', '"threshold', "lambda2", "admitted",
                                               "artifact_removed", "control_clear")))
    bad1 = {"istanbul_marmara": {"izmit": [["KO.SAUV..HHZ"]],
                                 "marmara_west": [["KO.NMR6..HHZ"], ["KO.NMR3..HHZ"]]}}
    check("SB-3c a segment with ONE station refuses to plan (>=2 stations/segment)",
          raises(lambda: P.build_campaign_plan(bad1, "2026-08-09")))
    bad2 = {"istanbul_marmara": {"izmit": [["KO.SAUV..HHZ"], ["KO.GAZK..HHZ"]]}}
    check("SB-3d a carrier with ONE segment refuses to plan (>=2 segments/carrier)",
          raises(lambda: P.build_campaign_plan(bad2, "2026-08-09")))
    bad3 = {"tokyo_kanto": {"a": [["JP.JYT..HHZ"], ["PS.TSK..HHZ"]],
                            "b": [["XX.A..HHZ"], ["XX.B..HHZ"]]}}
    check("SB-3e a carrier outside the provider cap refuses to plan (no tokyo_kanto, no "
          "expansion)", raises(lambda: P.build_campaign_plan(bad3, "2026-08-09")))

    # ---- SB-4: published-phase session binding, no inferred fallback -------
    rec = {"schema": "daily-monitoring-record", "day": "2026-05-01",
           "request_interval": {"start": "2026-04-30T07:00:13.094647Z",
                                "end": "2026-05-01T07:00:13.094647Z"}}
    s, e = P.session_from_record(json.dumps(rec).encode())
    check("SB-4 the published record's EXACT half-open interval is the session "
          "(microfractional phase preserved; duration exactly 86,400 s)",
          s.tzinfo is not None and (e - s) == timedelta(seconds=86400)
          and s.microsecond == 94647,
          f"start={s} end={e}")
    rec_short = dict(rec)
    rec_short["request_interval"] = {"start": "2026-04-30T07:00:13.094647Z",
                                     "end": "2026-05-01T04:00:13.094647Z"}
    check("SB-4b a non-86,400 s registered session refuses (never padded/extended)",
          raises(lambda: P.session_from_record(json.dumps(rec_short).encode())))
    check("SB-4c a record without an interval refuses (no midnight/nearest/inferred fallback)",
          raises(lambda: P.session_from_record(json.dumps({"day": "2026-05-01"}).encode()))
          and raises(lambda: P.session_from_record(b"{not json")))

    # ---- SB-5: frozen NSLC selection order ---------------------------------
    cands = ["CI.BOR..BHZ", "CI.BOR..HHZ", "CI.BOR..HNZ"]
    check("SB-5 channel selection takes the FIRST available candidate in frozen order; "
          "None when none; order never re-sorted",
          P.select_channel(cands, {"CI.BOR..HHZ", "CI.BOR..HNZ"}) == "CI.BOR..HHZ"
          and P.select_channel(cands, {"CI.BOR..BHZ", "CI.BOR..HHZ"}) == "CI.BOR..BHZ"
          and P.select_channel(cands, set()) is None
          and P.select_channel(cands, {"CI.XXX..BHZ"}) is None)

    # ---- SB-6: nearest-rank lower-5% threshold + 60-day floor --------------
    r60 = [float(i) for i in range(1, 61)]                 # ceil(3)-1=2 -> value 3.0
    r90 = [float(i) for i in range(1, 91)]                 # ceil(4.5)-1=4 -> value 5.0
    r61 = [float(i) for i in range(1, 62)]                 # ceil(3.05)-1=3 -> value 4.0
    import random
    shuffled = r90[:]
    random.Random(7).shuffle(shuffled)
    check("SB-6 nearest-rank lower-5%: n=60 -> 3.0, n=90 -> 5.0, n=61 -> 4.0; input order "
          "irrelevant; n=59 -> None (BLOCKED_INSUFFICIENT_CALIBRATION floor)",
          P.threshold_from_admitted(r60) == 3.0 and P.threshold_from_admitted(r90) == 5.0
          and P.threshold_from_admitted(r61) == 4.0
          and P.threshold_from_admitted(shuffled) == 5.0
          and P.threshold_from_admitted(r60[:-1]) is None)

    # ---- SB-7: replay ratios derived from bound evidence bytes -------------
    prior = {"carriers": {"socal_coachella": {"control_ratio": 0.9985, "incident_ratio": 0.9987},
                          "turkey_kahramanmaras": {"control_ratio": 0.8779,
                                                   "incident_ratio": 0.9340},
             "istanbul_marmara": {"control_ratio": 0.3710, "incident_ratio": 0.3522}}}
    prior_bytes = json.dumps(prior, sort_keys=True, separators=(",", ":")).encode()
    pin = hashlib.sha256(prior_bytes).hexdigest()
    got = P.derive_replay_ratios(prior_bytes, pin)
    check("SB-7 replay ratios come FROM the pinned evidence bytes",
          abs(got["socal_coachella"]["incident_ratio"] - 0.9987) < 1e-12
          and abs(got["istanbul_marmara"]["control_ratio"] - 0.3710) < 1e-12)
    check("SB-7b tampered evidence bytes fail the pin (no producer-entered ratio path)",
          raises(lambda: P.derive_replay_ratios(prior_bytes + b" ", pin)))

    def arm(n_admitted, threshold):
        return {"admitted_days": n_admitted, "threshold": threshold}

    st, _ = P.admit_candidate("socal_coachella", arm(60, 0.30), arm(62, 0.31),
                              {"control_ratio": 0.9985, "incident_ratio": 0.9987})
    st2, _ = P.admit_candidate("socal_coachella", arm(59, 0.30), arm(62, 0.31),
                               {"control_ratio": 0.9985, "incident_ratio": 0.9987})
    st3, _ = P.admit_candidate("socal_coachella", arm(60, 0.30), arm(62, 0.31), None)
    st4, _ = P.admit_candidate("socal_coachella", arm(60, 0.9990), arm(62, 0.31),
                               {"control_ratio": 0.9992, "incident_ratio": 0.9987})
    st5, _ = P.admit_candidate("socal_coachella", arm(60, 0.9990), arm(62, 0.31),
                               {"control_ratio": 0.9985, "incident_ratio": 0.9992})
    check("SB-7c candidate rule: all-4-conditions -> ADMITTED_CANDIDATE; <60 days -> "
          "BLOCKED_INSUFFICIENT_CALIBRATION; no replay -> BLOCKED_REPLAY_UNAVAILABLE; "
          "incident<threshold -> BLOCKED_ARTIFACT_PERSISTS; control<threshold -> "
          "BLOCKED_NEGATIVE_CONTROL",
          st == "ADMITTED_CANDIDATE" and st2 == "BLOCKED_INSUFFICIENT_CALIBRATION"
          and st3 == "BLOCKED_REPLAY_UNAVAILABLE" and st4 == "BLOCKED_ARTIFACT_PERSISTS"
          and st5 == "BLOCKED_NEGATIVE_CONTROL",
          f"{st}/{st2}/{st3}/{st4}/{st5}")

    # ---- SB-8: the launch gate (grassmann's consent classifier, formalized) -
    good_receipt = {"status": "VERIFIED_DIRECT",
                    "in_session_timestamp_utc": "2026-08-09T02:00:00Z",
                    "owner_quote_sha256": "a" * 64}
    check("SB-8 verify_launch_authorization: VERIFIED_DIRECT + parseable UTC + 64-hex quote "
          "sha ACCEPTS; RELAYED / missing fields / bad hex REJECT",
          P.verify_launch_authorization(good_receipt) is True
          and P.verify_launch_authorization({**good_receipt, "status": "RELAYED"}) is False
          and P.verify_launch_authorization({**good_receipt, "owner_quote_sha256": "zz"}) is False
          and P.verify_launch_authorization({k: v for k, v in good_receipt.items()
                                             if k != "in_session_timestamp_utc"}) is False
          and P.verify_launch_authorization({}) is False
          and P.verify_launch_authorization(None) is False)
    # the fetch entry point must refuse BEFORE any provider I/O without the verified receipt
    if hasattr(P, "run_campaign"):
        refused = raises(lambda: P.run_campaign(plan=p1, launch_authorization=None,
                                                dry_run=True), exc=BaseException)
        check("SB-8b run_campaign refuses to start (before any provider I/O) without a "
              "VERIFIED_DIRECT launch receipt", refused)
    else:
        check("SB-8b run_campaign entry point present (refuses without verified launch receipt)",
              False, "AWAITING grassmann's producer -- red-first as authored")


main()
print()
if FAILS:
    print(f"D2 STEP-4B CAMPAIGN RED-KAT FAILURES: {FAILS}")
    sys.exit(1)
print("ALL D2 STEP-4B CAMPAIGN RED-KATs PASS (outcome-blind plan + published-phase-only sessions "
      "+ frozen selection + pinned-evidence admission + direct-launch gate)")
