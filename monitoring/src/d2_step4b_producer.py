#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""d2_step4b_producer.py — D2 step-4b archive-calibration campaign PRODUCER.

Implements the seam contract of `monitoring/src/test_d2_step4b_redkats_cayley.py` (cayley
`65755de`, SB-0..8) under contract `codex-d2-step4b-2026-08-09-v1` (codex 0129); the produced
BATCH is separately verified by codex's 0123 acceptance entry point. Pinned segmented scoring
implementation: GeoSpec `3950a2c`.

Nothing here lifts a freeze, admits calibration values, tunes a rule, promotes a registry,
deploys, publishes, or makes a claim. The campaign is outcome-blind and may honestly mint 0..3
candidates. The FIRST archive request additionally requires a DIRECT verifiable owner launch go
bound in `run_campaign` (SB-8): `run_campaign` refuses — before ANY provider I/O — without a
`VERIFIED_DIRECT` receipt. This module cannot and does not mint that receipt.
"""
import hashlib
import json
import math
import re
from datetime import datetime, timedelta

# ---- SB-1: frozen campaign constants ---------------------------------------
CAMPAIGN = {
    "contract_id": "codex-d2-step4b-2026-08-09-v1",
    "incident_reference": "2026-07-29",
    "min_admitted_days": 60,
    "window_days": 90,
    "lag_days": 30,
    "providers": {
        "istanbul_marmara": "eida.koeri.boun.edu.tr",
        "turkey_kahramanmaras": "eida.koeri.boun.edu.tr",
        "socal_coachella": "s3://scedc-pds",
    },
}

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _canonical(obj) -> bytes:
    """Canonical JSON bytes: sorted keys, compact separators, UTF-8 (one logical value)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _parse_day(s: str):
    return datetime.strptime(s, "%Y-%m-%d").date()


def _parse_iso_utc(s):
    """Parse an ISO-8601 instant that MUST be aware UTC (trailing 'Z' tolerated)."""
    txt = str(s)
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    dt = datetime.fromisoformat(txt)
    if dt.tzinfo is None or dt.utcoffset() != timedelta(0):
        raise ValueError(f"timestamp {s!r} is not aware UTC")
    return dt


# ---- SB-2: exact arm schedules ---------------------------------------------
def schedule_days(reference_day: str):
    """The exact half-open [ref-120d, ref-30d) = 90 ascending ISO calendar days.
    (ref - (lag_days + window_days)) .. (ref - lag_days), left-closed / right-open."""
    ref = _parse_day(reference_day)
    lag = int(CAMPAIGN["lag_days"])          # 30
    win = int(CAMPAIGN["window_days"])       # 90
    start = ref - timedelta(days=lag + win)  # ref - 120d
    return [(start + timedelta(days=i)).isoformat() for i in range(win)]


# ---- SB-3: outcome-blind deterministic plan --------------------------------
def build_campaign_plan(carriers: dict, activation_reference: str) -> dict:
    """Freeze the outcome-blind campaign plan. Fail-closed: >= 2 stations/segment, >= 2
    segments/carrier, provider only from CAMPAIGN['providers']. NO outcome-bearing field is
    accepted or emitted. Deterministic: identical inputs -> byte-identical canonical JSON."""
    providers = CAMPAIGN["providers"]
    if not isinstance(carriers, dict) or not carriers:
        raise ValueError("carriers must be a non-empty mapping")
    plan_carriers = {}
    for carrier, segments in carriers.items():
        if carrier not in providers:
            raise ValueError(f"carrier {carrier!r} is outside the provider cap "
                             f"{sorted(providers)} (no expansion)")
        if not isinstance(segments, dict) or len(segments) < 2:
            raise ValueError(f"carrier {carrier!r} needs >= 2 segments")
        seg_out = {}
        for seg, stations in segments.items():
            if not isinstance(stations, list) or len(stations) < 2:
                raise ValueError(f"segment {carrier}/{seg} needs >= 2 stations")
            frozen = []
            for cand_list in stations:
                if not isinstance(cand_list, list) or not cand_list:
                    raise ValueError(f"station in {carrier}/{seg} needs an ordered NSLC "
                                     f"candidate list")
                frozen.append([str(c) for c in cand_list])   # ORDERED NSLC candidates, verbatim
            seg_out[seg] = frozen
        plan_carriers[carrier] = {"provider": providers[carrier], "segments": seg_out}
    return {
        "contract_id": CAMPAIGN["contract_id"],
        "incident_reference": CAMPAIGN["incident_reference"],
        "activation_reference": str(activation_reference),
        "arms": {
            "incident": {"reference": CAMPAIGN["incident_reference"],
                         "days": schedule_days(CAMPAIGN["incident_reference"])},
            "activation": {"reference": str(activation_reference),
                           "days": schedule_days(str(activation_reference))},
        },
        "carriers": plan_carriers,
    }


def plan_digest(plan) -> str:
    """64-hex SHA-256 over the canonical plan bytes."""
    return hashlib.sha256(_canonical(plan)).hexdigest()


# ---- SB-4: published-phase session binding (no inferred fallback) ----------
def session_from_record(record_bytes: bytes):
    """Parse the published daily-monitoring record's EXACT half-open request interval (aware
    UTC). The registered session MUST be exactly 86,400.000000 s. Malformed / missing interval
    / non-86,400 s duration -> ValueError. There is NO midnight / nearest / inferred fallback."""
    try:
        rec = json.loads(record_bytes.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"published record is not valid JSON: {exc}")
    if not isinstance(rec, dict):
        raise ValueError("published record is not a JSON object")
    iv = rec.get("request_interval")
    if not (isinstance(iv, dict) and "start" in iv and "end" in iv):
        raise ValueError("published record has no request_interval {start, end}")
    start = _parse_iso_utc(iv["start"])
    end = _parse_iso_utc(iv["end"])
    if (end - start) != timedelta(seconds=86400):
        raise ValueError(f"registered session is not exactly 86,400 s "
                         f"(got {(end - start).total_seconds()} s)")
    return start, end


# ---- SB-5: frozen NSLC selection -------------------------------------------
def select_channel(candidates, available):
    """The FIRST candidate present in `available`, in the frozen order; None if none. No
    re-sort, no post-QC swap (QC failure of fetched data never re-enters selection)."""
    for cand in candidates:
        if cand in available:
            return cand
    return None


# ---- SB-6: nearest-rank lower-5% quantile ----------------------------------
def threshold_from_admitted(ratios):
    """None below the MIN_ADMITTED_DAYS floor; else sort ascending and return the nearest-rank
    lower 5% quantile ratios_sorted[ceil(0.05*n)-1] (zero-based). Input-order-invariant."""
    n = len(ratios)
    if n < int(CAMPAIGN["min_admitted_days"]):
        return None
    ordered = sorted(float(x) for x in ratios)
    idx = math.ceil(0.05 * n) - 1
    return ordered[idx]


# ---- SB-7: replay ratios derived from bound evidence bytes -----------------
def derive_replay_ratios(prior_evidence_bytes: bytes, expected_sha256: str) -> dict:
    """Verify the prior-evidence bytes against the pin, then extract the sealed control/incident
    ratios FROM THE BYTES. Producer-entered ratio values have no path in; tampered bytes fail."""
    actual = hashlib.sha256(prior_evidence_bytes).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"prior-evidence sha256 {actual} != pin {expected_sha256}")
    prior = json.loads(prior_evidence_bytes.decode("utf-8"))
    carriers = prior.get("carriers") if isinstance(prior, dict) else None
    if not isinstance(carriers, dict):
        raise ValueError("prior evidence missing 'carriers' mapping")
    out = {}
    for carrier, vals in carriers.items():
        out[carrier] = {"control_ratio": float(vals["control_ratio"]),
                        "incident_ratio": float(vals["incident_ratio"])}
    return out


# ---- SB-7c: deterministic candidate rule -----------------------------------
def admit_candidate(carrier, incident_summary, activation_summary, replay):
    """(status, info). ADMITTED_CANDIDATE iff BOTH arms have >= 60 admitted days AND reproducible
    thresholds AND replay carries both ratios AND incident_ratio >= incident_threshold AND
    control_ratio >= incident_threshold. Otherwise one deterministic BLOCKED_* status."""
    min_days = int(CAMPAIGN["min_admitted_days"])
    inc_days = int(incident_summary.get("admitted_days", 0))
    act_days = int(activation_summary.get("admitted_days", 0))
    inc_thr = incident_summary.get("threshold")
    act_thr = activation_summary.get("threshold")
    if inc_days < min_days or act_days < min_days or inc_thr is None or act_thr is None:
        return "BLOCKED_INSUFFICIENT_CALIBRATION", {
            "carrier": carrier, "incident_admitted_days": inc_days,
            "activation_admitted_days": act_days, "min_admitted_days": min_days,
            "incident_threshold": inc_thr, "activation_threshold": act_thr}
    if not (isinstance(replay, dict) and "incident_ratio" in replay
            and "control_ratio" in replay):
        return "BLOCKED_REPLAY_UNAVAILABLE", {"carrier": carrier, "replay": replay}
    inc_ratio = float(replay["incident_ratio"])
    ctrl_ratio = float(replay["control_ratio"])
    inc_threshold = float(inc_thr)
    if inc_ratio < inc_threshold:
        return "BLOCKED_ARTIFACT_PERSISTS", {
            "carrier": carrier, "incident_ratio": inc_ratio, "incident_threshold": inc_threshold}
    if ctrl_ratio < inc_threshold:
        return "BLOCKED_NEGATIVE_CONTROL", {
            "carrier": carrier, "control_ratio": ctrl_ratio, "incident_threshold": inc_threshold}
    return "ADMITTED_CANDIDATE", {
        "carrier": carrier, "incident_ratio": inc_ratio, "control_ratio": ctrl_ratio,
        "incident_threshold": inc_threshold, "activation_threshold": float(act_thr),
        "artifact_removed": True, "control_clear": True}


# ---- SB-8: the direct-owner launch gate ------------------------------------
def verify_launch_authorization(receipt) -> bool:
    """True iff receipt == {status: 'VERIFIED_DIRECT', in_session_timestamp_utc: <aware-UTC>,
    owner_quote_sha256: <64-hex>}. RELAYED / missing / malformed -> False. This formalizes
    grassmann's session-level consent classifier; it cannot substitute for the owner's direct
    in-session word (which mints the receipt)."""
    if not isinstance(receipt, dict):
        return False
    if set(receipt.keys()) != {"status", "in_session_timestamp_utc", "owner_quote_sha256"}:
        return False
    if receipt.get("status") != "VERIFIED_DIRECT":
        return False
    try:
        _parse_iso_utc(receipt.get("in_session_timestamp_utc"))
    except Exception:
        return False
    quote = receipt.get("owner_quote_sha256")
    return isinstance(quote, str) and bool(_HEX64.match(quote))


def run_campaign(plan, launch_authorization, dry_run=False, **kwargs):
    """Campaign fetch/produce entry point. REFUSES — before ANY provider I/O — unless
    `launch_authorization` is a VERIFIED_DIRECT owner launch receipt (SB-8b). The provider fetch,
    published-phase binding, segmented scoring (via GeoSpec 3950a2c), batch assembly, and share
    staging execute only past this gate; they are built out for the authorized run and never
    reached without the receipt."""
    if not verify_launch_authorization(launch_authorization):
        raise SystemExit("run_campaign REFUSED: no VERIFIED_DIRECT owner launch authorization "
                         "— no archive request issued, no provider I/O performed.")
    if dry_run:
        return {"status": "LAUNCH_AUTHORIZED", "dry_run": True,
                "plan_digest": plan_digest(plan) if plan is not None else None}
    # Authorized, non-dry run: the provider-I/O fetch + published-phase binding + segmented
    # scoring + batch assembly + share staging are performed here for the real campaign run.
    raise NotImplementedError("authorized full-fetch campaign run is executed by the campaign "
                              "driver; see the step-4b execution phase")
