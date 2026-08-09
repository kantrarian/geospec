#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D2 STEP-4B CAMPAIGN red-KATs (cayley) — contract `codex-d2-step4b-2026-08-09-v1`
(codex 0129, `ecd5bff`); pinned segmented implementation GeoSpec `3950a2c`. These bars gate
grassmann's campaign PRODUCER (red-first); codex's step-4b acceptance entry point separately
verifies the produced BATCH. Nothing here fetches, lifts, tunes, or claims; the campaign's
first archive request additionally required the DIRECT verifiable owner launch go in
grassmann's session (SB-8) — this bar cannot and does not substitute for it.

REV 2 (2026-08-09, authorized by codex 0300 F2 + codex 0313 SB-4 directive):
  * SB-4 REVISED to `published-end-anchored-segmented-v2` (codex phasebar 0313): real public
    records carry NO request_interval; the session anchors on the published per-region naive
    `date` interpreted as UTC = request END, START = END − 86,400 s exactly. New RED cases:
    shifted-end, wrong-region, unavailable-component, non-UTC-offset reinterpretation,
    top-level day mismatch, decoy-request_interval anti-fallback, record-byte mutation.
  * SB-7 REVISED to the ACCEPTED evidence schema (codex 0300 F2): the producer consumes the
    exact sealed `d2_diagnostic_result.json` bytes pinned at `ee75e449…` DIRECTLY and derives
    every replay value from `results[carrier][incident|control]` (status==OK both phases,
    ratio recomputed from ordered_eigenvalues, matrix digest + common support from bytes).
    Fixture = the REAL sealed artifact (`fixtures/d2_diagnostic_result.json`, byte-identical
    to the mirrored diagnostic capsule), not a synthetic wrapper.
  * All other checks semantically identical to rev-1 (geospec 65755de).
  RED-FIRST DELTA vs producer `c1576a9`: exactly ['SB-4-GATE(v2)', 'SB-7-GATE(v2)'].

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
* REGION_KEY = {"istanbul_marmara": "istanbul_marmara",
  "socal_coachella": "socal_saf_coachella", "turkey_kahramanmaras": "turkey_kahramanmaras"}
  — the exact carrier -> published-region mapping (codex phasebar 0313).
* session_from_record(record_bytes: bytes, *, carrier: str, scored_day: str)
    -> (start_utc, end_utc)   [published-end-anchored-segmented-v2]
    - parse the PORTABLE public daily-monitoring record bytes; require top-level
      record["date"] == scored_day;
    - region = record["regions"][REGION_KEY[carrier]] must exist (unknown carrier or missing
      region -> ValueError);
    - region["components"]["fault_correlation"]["available"] must be exactly True;
    - request END = region["date"], a naive-or-Z serialized instant (microseconds preserved,
      up to 6 digits) interpreted as UTC; ANY other form (explicit non-Z offset, epoch,
      prose) refuses;
    - END must lie on scored_day (end.date().isoformat() == scored_day);
    - request START = END - exactly 86,400 s; return aware-UTC (start, end);
    - NO other timestamp path exists: a legacy/decoy `request_interval` object in the record
      is IGNORED; no midnight/nearest-day/cache/inferred fallback. A scheduled day with no
      record remains UNAVAILABLE_NO_PUBLISHED_RECORD: no archive request, no metric.
* select_channel(candidates: list[str], available: set[str]) -> str | None
    - the FIRST candidate (frozen order) present in `available`; None if none; there is no
      post-QC swap seam — QC failure of fetched data never re-enters selection.
* threshold_from_admitted(ratios: list[float]) -> float | None
    - None when n < 60; else sort ascending and return ratios_sorted[ceil(0.05*n)-1]
      (nearest-rank lower 5% quantile, zero-based).
* DIAGNOSTIC_RESULT_SHA256 =
  "ee75e449aa0b1003a3cf047432a91a9adc1db4c7497b1e1d9d47f01d552a4b35" (the accepted sealed
  diagnostic result — codex retention PASS 0043; pinned DIRECTLY, no caller-supplied pin).
* parse_diagnostic_results(doc: dict) -> dict
    - doc is the parsed diagnostic-result JSON; requires results for EXACTLY the three
      campaign carriers; for each carrier and both phases ("incident", "control"):
      status == "OK"; ratio RECOMPUTED as ordered_eigenvalues[1]/ordered_eigenvalues[0]
      (descending) and required to match stored lambda2_lambda1 within rel/abs 1e-6
      (mismatch = tamper -> ValueError); returns per carrier:
      {"incident_ratio", "control_ratio", "incident_common_support",
       "control_common_support", "incident_matrix_digest", "control_matrix_digest"}
      where *_matrix_digest = sha256 hex of the canonical JSON (sort_keys, ',:') of
      correlation_matrix and *_common_support = common_support_count from the bytes;
    - missing carrier, missing phase, non-OK status, or ratio/eigenvalue mismatch ->
      ValueError.
* derive_replay_ratios(diagnostic_result_bytes: bytes) -> dict
    - sha256(bytes) MUST equal DIAGNOSTIC_RESULT_SHA256 (else ValueError; NO override or
      expected-sha parameter exists) -> json parse -> parse_diagnostic_results(doc).
      Producer-entered ratio values have no path in; `prior_evidence.json` stays a receipt
      wrapper only (codex 0300 F2).
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
"""
import hashlib
import inspect
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

DIAG_FIXTURE = os.path.join(HERE, "fixtures", "d2_diagnostic_result.json")
DIAG_SHA256 = "ee75e449aa0b1003a3cf047432a91a9adc1db4c7497b1e1d9d47f01d552a4b35"
REGION_KEY = {
    "istanbul_marmara": "istanbul_marmara",
    "socal_coachella": "socal_saf_coachella",
    "turkey_kahramanmaras": "turkey_kahramanmaras",
}

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


def _pub_record(scored_day, region_dates, fc_available=None, extra=None):
    """Portable public daily-monitoring record bytes in the REAL shape (top-level date +
    regions[key].date + components.fault_correlation.available). fc_available overrides
    availability per region key; extra merges extra top-level fields (e.g. a decoy
    request_interval)."""
    fc_available = fc_available or {}
    regions = {}
    for key, d in region_dates.items():
        regions[key] = {
            "date": d,
            "components": {"fault_correlation":
                           {"available": fc_available.get(key, True)}},
        }
    rec = {"date": scored_day, "regions": regions}
    if extra:
        rec.update(extra)
    return json.dumps(rec, sort_keys=True).encode("utf-8")


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

    # ---- SB-4 v2: published-end-anchored session binding (codex 0313) ------
    sig = None
    try:
        sig = inspect.signature(P.session_from_record)
    except (TypeError, ValueError):
        pass
    v2_session = (sig is not None and "carrier" in sig.parameters
                  and "scored_day" in sig.parameters
                  and getattr(P, "REGION_KEY", None) == REGION_KEY)
    if not v2_session:
        check("SB-4-GATE(v2) session seam is published-end-anchored-segmented-v2 "
              "(carrier/scored_day-aware + exact REGION_KEY)", False,
              "AWAITING producer v2 session seam -- red-first as revised")
    else:
        day = "2026-05-01"
        good = _pub_record(day, {
            "istanbul_marmara": "2026-05-01T07:00:13.094647",
            "socal_saf_coachella": "2026-05-01T07:00:13.094647",
            "turkey_kahramanmaras": "2026-05-01T07:00:13.094647"})
        s, e = P.session_from_record(good, carrier="socal_coachella", scored_day=day)
        check("SB-4 published naive region.date is the UTC request END (microseconds "
              "preserved); START is exactly END - 86,400 s; both aware-UTC",
              s.tzinfo is not None and e.tzinfo is not None
              and e == datetime(2026, 5, 1, 7, 0, 13, 94647, tzinfo=timezone.utc)
              and (e - s) == timedelta(seconds=86400),
              f"start={s} end={e}")
        goodz = _pub_record(day, {"istanbul_marmara": "2026-05-01T07:00:13.094647Z",
                                  "socal_saf_coachella": "2026-05-01T07:00:13.094647Z",
                                  "turkey_kahramanmaras": "2026-05-01T07:00:13.094647Z"})
        s2, e2 = P.session_from_record(goodz, carrier="istanbul_marmara", scored_day=day)
        check("SB-4a Z-suffixed form accepted identically; carrier maps through REGION_KEY",
              e2 == e and (e2 - s2) == timedelta(seconds=86400))
        shifted = _pub_record(day, {"socal_saf_coachella": "2026-05-02T07:00:13.094647",
                                    "istanbul_marmara": "2026-05-01T07:00:13.094647",
                                    "turkey_kahramanmaras": "2026-05-01T07:00:13.094647"})
        check("SB-4b SHIFTED-END refuses (published end not on the scored day)",
              raises(lambda: P.session_from_record(shifted, carrier="socal_coachella",
                                                   scored_day=day)))
        check("SB-4c WRONG-REGION refuses (carrier's mapped region absent; unknown carrier "
              "refuses)",
              raises(lambda: P.session_from_record(
                  _pub_record(day, {"istanbul_marmara": "2026-05-01T07:00:13.094647"}),
                  carrier="socal_coachella", scored_day=day))
              and raises(lambda: P.session_from_record(good, carrier="ridgecrest",
                                                       scored_day=day)))
        unavail = _pub_record(day, {"socal_saf_coachella": "2026-05-01T07:00:13.094647",
                                    "istanbul_marmara": "2026-05-01T07:00:13.094647",
                                    "turkey_kahramanmaras": "2026-05-01T07:00:13.094647"},
                              fc_available={"socal_saf_coachella": False})
        check("SB-4d UNAVAILABLE-COMPONENT refuses (fault_correlation.available is not True)",
              raises(lambda: P.session_from_record(unavail, carrier="socal_coachella",
                                                   scored_day=day)))
        offset = _pub_record(day, {"socal_saf_coachella": "2026-05-01T07:00:13.094647+03:00",
                                   "istanbul_marmara": "2026-05-01T07:00:13.094647",
                                   "turkey_kahramanmaras": "2026-05-01T07:00:13.094647"})
        check("SB-4e NON-UTC OFFSET refuses (naive-or-Z only; no timezone reinterpretation)",
              raises(lambda: P.session_from_record(offset, carrier="socal_coachella",
                                                   scored_day=day)))
        check("SB-4f TOP-LEVEL DAY MISMATCH refuses (record.date != scored_day)",
              raises(lambda: P.session_from_record(good, carrier="socal_coachella",
                                                   scored_day="2026-05-02")))
        decoy = _pub_record(day, {"socal_saf_coachella": "2026-05-01T07:00:13.094647",
                                  "istanbul_marmara": "2026-05-01T07:00:13.094647",
                                  "turkey_kahramanmaras": "2026-05-01T07:00:13.094647"},
                            extra={"request_interval":
                                   {"start": "2026-04-30T00:00:00Z",
                                    "end": "2026-05-01T00:00:00Z"}})
        sd, ed = P.session_from_record(decoy, carrier="socal_coachella", scored_day=day)
        check("SB-4g ANTI-FALLBACK: a decoy request_interval in the record is IGNORED; the "
              "session anchors on region.date only",
              ed == e and (ed - sd) == timedelta(seconds=86400), f"end={ed}")
        mutated = good.replace(b"07:00:13.094647", b"07:00:13.094648")
        sm, em = P.session_from_record(mutated, carrier="socal_coachella", scored_day=day)
        check("SB-4h RECORD-BYTE MUTATION is faithfully reflected (bytes are authoritative: "
              "one-microsecond change moves the anchor exactly); malformed bytes refuse",
              em == e + timedelta(microseconds=1)
              and raises(lambda: P.session_from_record(b"{not json",
                                                       carrier="socal_coachella",
                                                       scored_day=day)))

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

    # ---- SB-7 v2: replay derivation from the ACCEPTED sealed artifact ------
    with open(DIAG_FIXTURE, "rb") as f:
        diag_bytes = f.read()
    check("SB-7-FIXTURE the committed fixture IS the accepted sealed artifact "
          "(sha ee75e449...)",
          hashlib.sha256(diag_bytes).hexdigest() == DIAG_SHA256)
    v2_replay = (getattr(P, "DIAGNOSTIC_RESULT_SHA256", None) is not None
                 and hasattr(P, "parse_diagnostic_results"))
    if not v2_replay:
        check("SB-7-GATE(v2) replay seam consumes the accepted diagnostic-result schema "
              "(DIAGNOSTIC_RESULT_SHA256 + parse_diagnostic_results present)", False,
              "AWAITING producer v2 replay seam -- red-first as revised")
    else:
        check("SB-7 producer pins the accepted artifact DIRECTLY",
              P.DIAGNOSTIC_RESULT_SHA256 == DIAG_SHA256)
        got = P.derive_replay_ratios(diag_bytes)
        doc = json.loads(diag_bytes.decode("utf-8"))
        res = doc["results"]

        def md(carrier, phase):
            return hashlib.sha256(json.dumps(
                res[carrier][phase]["correlation_matrix"],
                sort_keys=True, separators=(",", ":")).encode()).hexdigest()

        ok_vals = True
        for carrier in REGION_KEY:
            for phase in ("incident", "control"):
                entry = res[carrier][phase]
                ok_vals = (ok_vals
                           and abs(got[carrier][f"{phase}_ratio"]
                                   - entry["lambda2_lambda1"]) <= 1e-6
                           and got[carrier][f"{phase}_common_support"]
                           == entry["common_support_count"]
                           and got[carrier][f"{phase}_matrix_digest"] == md(carrier, phase))
        check("SB-7a real bytes -> every ratio/common-support/matrix-digest derived from the "
              "artifact for all three carriers x both phases (socal incident 0.9986667, "
              "istanbul control 0.37098175, turkey control 0.87788606, ...)",
              set(got) == set(REGION_KEY) and ok_vals)
        check("SB-7b BYTE-TAMPER refuses (any byte change fails the direct pin; no override "
              "parameter exists)",
              raises(lambda: P.derive_replay_ratios(diag_bytes + b" "))
              and raises(lambda: P.derive_replay_ratios(
                  diag_bytes.replace(b"0.9986667", b"0.9986668", 1)))
              and "expected_sha256" not in
                  inspect.signature(P.derive_replay_ratios).parameters)
        missing = json.loads(diag_bytes.decode("utf-8"))
        del missing["results"]["turkey_kahramanmaras"]
        check("SB-7c MISSING-CARRIER refuses (exactly the three campaign carriers required)",
              raises(lambda: P.parse_diagnostic_results(missing)))
        nonok = json.loads(diag_bytes.decode("utf-8"))
        nonok["results"]["socal_coachella"]["incident"]["status"] = "DEGRADED"
        check("SB-7d NON-OK STATUS refuses (both phases must be status OK)",
              raises(lambda: P.parse_diagnostic_results(nonok)))
        tampered = json.loads(diag_bytes.decode("utf-8"))
        tampered["results"]["socal_coachella"]["incident"]["ordered_eigenvalues"][1] *= 0.5
        check("SB-7e EIGENVALUE/RATIO TAMPER refuses (ratio is RECOMPUTED from "
              "ordered_eigenvalues and must match the stored scalar)",
              raises(lambda: P.parse_diagnostic_results(tampered)))

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
    check("SB-7f candidate rule: all-4-conditions -> ADMITTED_CANDIDATE; <60 days -> "
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
print("ALL D2 STEP-4B CAMPAIGN RED-KATs PASS (outcome-blind plan + published-end-anchored "
      "sessions + frozen selection + accepted-artifact replay derivation + direct-launch gate)")
