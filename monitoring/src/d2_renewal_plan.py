#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D2 calibration renewal plan/validators (contract codex-d2-campaign-v2-renewal-
2026-08-16-v1, frozen by codex 80bb3c6; acceptance bar REV 2 = cayley 13b8d34).

Reissues the frozen v2 method as a new campaign instance: same estimator,
selection rule, scientific gates, capsule schema, and claim limits — new campaign
identity, time anchor/windows, source attestation, acquired bytes, thresholds,
capsule digests, and validity interval. Everything here is bytes-pinned and
outcome-blind; nothing performs provider I/O. The fresh fire-time owner go naming
the contract id and A remains required before any external request; the registry
lift is separately owner-gated.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import date, datetime, timedelta, timezone
from typing import Optional, Tuple

RENEWAL_CONTRACT_ID = "codex-d2-campaign-v2-renewal-2026-08-16-v1"
RENEWAL_ANCHOR = "2026-08-16"
V2_POOL_SHA256 = "15d0e32c51c027dc144c5c6d57ec5f100a59374f6248abfc5c56ee38628ddc67"
# Set when the renewal Phase-0.5-equivalent evidence bundle freezes (bar checks the
# pin exists + refusal semantics; codex verifies the eventual binding at its lane).
RENEWAL_BUNDLE_SHA256 = "4519bbd1eae0c517a2f91822c853bc42272a6c85966f0e7436cd3e74e3c79b54"

_INCIDENT_START, _INCIDENT_END = "2026-03-01", "2026-06-29"      # end exclusive
_ACTIVATION_START, _ACTIVATION_END = "2026-03-19", "2026-07-17"  # [A-150d, A-30d)
_INCIDENT_REFERENCE = "2026-07-29"
_EMBARGO_DAYS = 30
_VALID_THROUGH = "2026-08-23"                                    # A+7d inclusive
_EXPIRY_UTC = "2026-08-24T00:00:00Z"
_TARGET_ORDER = ("istanbul_marmara", "socal_coachella", "turkey_kahramanmaras")
_TERMINAL_STATES = {"ADMITTED_CANDIDATE", "COVERAGE_INFEASIBLE",
                    "BLOCKED_INSUFFICIENT_CALIBRATION", "BLOCKED_TOPOLOGY",
                    "BLOCKED_NO_TRUE_CARRIER", "HOLD"}
_CORE_FILES = ("monitoring/src/seismic_data.py", "monitoring/src/fault_correlation.py",
               "monitoring/src/ensemble.py", "monitoring/src/d2_step4b_campaign_run.py",
               "monitoring/src/d2_renewal_plan.py")
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", ".."))


def _canon(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                       allow_nan=False) + "\n").encode()


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _days_between(a: str, b: str):
    out = []
    d = date.fromisoformat(a)
    end = date.fromisoformat(b)
    while d < end:
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def renewal_arms() -> dict:
    """The contract's §1 frame, exactly; every consumer derives dates from here."""
    return {
        "contract_id": RENEWAL_CONTRACT_ID,
        "activation_reference_day": RENEWAL_ANCHOR,
        "incident_reference": _INCIDENT_REFERENCE,
        "incident": {"start": _INCIDENT_START, "end": _INCIDENT_END,
                     "days": _days_between(_INCIDENT_START, _INCIDENT_END)},
        "activation": {"start": _ACTIVATION_START, "end": _ACTIVATION_END,
                       "days": _days_between(_ACTIVATION_START, _ACTIVATION_END)},
        "embargo_days": _EMBARGO_DAYS,
        "valid_through": _VALID_THROUGH,
        "expiry_utc": _EXPIRY_UTC,
    }


def _union_days():
    a = renewal_arms()
    return sorted(set(a["incident"]["days"]) | set(a["activation"]["days"]))


def build_renewal_candidate_pool(v2_pool_bytes) -> Tuple[dict, bytes]:
    """§2: a COPY-ONLY re-envelope of the frozen v2 candidate pool. Input must be
    the exact pinned bytes; a parsed dict or any byte mutation refuses. No
    candidate may be added, removed, reassigned, or reordered."""
    if not isinstance(v2_pool_bytes, (bytes, bytearray)):
        raise TypeError("v2 pool must be the exact BYTES, never a parsed dict")
    if _sha(bytes(v2_pool_bytes)) != V2_POOL_SHA256:
        raise ValueError("v2 pool bytes do not match the pinned V2_POOL_SHA256")
    pool = json.loads(bytes(v2_pool_bytes))
    pool["schema"] = "geospec-d2-campaign-v2-renewal-pool-v1"
    pool["contract_id"] = RENEWAL_CONTRACT_ID
    pool["source_pool_sha256"] = V2_POOL_SHA256
    pool["created_utc"] = _now_iso()
    body = {k: v for k, v in pool.items() if k != "pool_digest"}
    pool["pool_digest"] = _sha(_canon(body))
    return pool, _canon(pool)


def _load_bundle(bundle_bytes) -> dict:
    if not isinstance(bundle_bytes, (bytes, bytearray)):
        raise TypeError("bundle must be the exact BYTES, never a parsed dict")
    if _sha(bytes(bundle_bytes)) != RENEWAL_BUNDLE_SHA256:
        raise ValueError("bundle bytes do not match the pinned RENEWAL_BUNDLE_SHA256")
    bundle = json.loads(bytes(bundle_bytes))
    if bundle.get("contract_id") != RENEWAL_CONTRACT_ID:
        raise ValueError("bundle carries the wrong contract id")
    if bundle.get("activation_reference_day") not in (None, RENEWAL_ANCHOR):
        raise ValueError("bundle anchor differs from the contract anchor")
    return bundle


def build_renewal_plan(bundle_bytes) -> Tuple[dict, bytes]:
    """§2 (REV 3 #2): the campaign plan from the FROZEN renewal evidence bundle
    bytes only — LEDGER-FIRST: the bundle must already bind the published-phase
    ledger digest, and the plan CARRIES that binding. An unbound bundle refuses.
    Canonical plan bytes carry the exact core blob vector at HEAD (§3)."""
    bundle = _load_bundle(bundle_bytes)
    pls = bundle.get("phase_ledger_sha256")
    if not (isinstance(pls, str) and len(pls) == 64
            and all(c in "0123456789abcdef" for c in pls)):
        raise ValueError("bundle does not bind the published-phase ledger digest "
                         "(unbound bundle — build the ledger FIRST)")
    arms = renewal_arms()
    cp = bundle.get("campaign_plan", {})
    carriers = [c for c in cp.get("eligible_carriers", []) if c in _TARGET_ORDER]
    if not carriers:
        raise ValueError("bundle names no eligible renewal carriers")
    import d2_step4b_campaign_run as CR
    plan = {
        "schema": "geospec-d2-step4b-campaign-plan-v1",
        "contract_id": RENEWAL_CONTRACT_ID,
        "activation_reference_day": RENEWAL_ANCHOR,
        "incident_reference_day": arms["incident_reference"],
        "carriers": carriers,
        "providers": {c: dict(CR.PROVIDERS[c]) for c in carriers},
        "station_registry": cp.get("selected_registry", {}),
        "incident_days": arms["incident"]["days"],
        "activation_days": arms["activation"]["days"],
        "scheduled_days": _union_days(),
        "acquisition_order": ["KOERI", "SCEDC"],
        "free_sources_only": True,
        "outcomes_inspected_before_schedule": False,
        "coverage_infeasible": cp.get("coverage_infeasible", {}),
        "core_blobs": core_blob_map(),
        "phase_ledger_sha256": pls,
        "created_utc": _now_iso(),
    }
    return plan, _canon(plan)


def build_renewal_phase_ledger(bundle_bytes) -> Tuple[dict, bytes]:
    """§2: the renewal published-phase ledger for the exact new arms. Rows come
    from the bundle's binder output when present; otherwise a REGISTERED skeleton
    covering the exact scheduled union (to be bound by the binder before fire)."""
    bundle = _load_bundle(bundle_bytes)
    cp = bundle.get("campaign_plan", {})
    carriers = [c for c in cp.get("eligible_carriers", []) if c in _TARGET_ORDER] \
        or list(_TARGET_ORDER)
    rows = cp.get("phase_ledger_rows")
    if not rows:
        rows = [{"carrier_key": c, "scored_day": day, "status": "REGISTERED"}
                for c in carriers for day in _union_days()]
    ledger = {"schema": "geospec-d2-published-phase-ledger-v1",
              "contract_id": RENEWAL_CONTRACT_ID,
              "rows": rows, "created_utc": _now_iso()}
    return ledger, _canon(ledger)


def validate_renewal_phase_ledger(ledger, bundle) -> bool:
    """True iff the ledger carries the renewal contract, covers the EXACT 138-day
    scheduled union, and the bundle binds these exact ledger bytes by sha."""
    try:
        if not isinstance(ledger, dict) or not isinstance(bundle, dict):
            return False
        if ledger.get("contract_id") != RENEWAL_CONTRACT_ID:
            return False
        days = sorted({r["scored_day"] for r in ledger.get("rows", [])})
        if days != _union_days():
            return False
        if bundle.get("phase_ledger_sha256") != _sha(_canon(ledger)):
            return False
        return True
    except Exception:
        return False


def core_blob_map() -> dict:
    """§3: the exact git blob sha of every core file at the CURRENT HEAD — the
    source attestation each capsule and plan must carry."""
    out = {}
    for f in _CORE_FILES:
        parts = subprocess.run(["git", "ls-tree", "HEAD", f], capture_output=True,
                               text=True, cwd=_REPO).stdout.split()
        out[f] = parts[2] if len(parts) >= 3 else None
    return out


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=_REPO).stdout.strip()


def classify_station_refusal(frs) -> str:
    """§3: a genuine POST-SORT overlap holds as TRUE_OVERLAP_UNRULED — never
    silently deduplicated, merged, or consumed. frs = (data, rate_hz, aware_start)
    tuples. Returns 'TRUE_OVERLAP_UNRULED' or 'DISJOINT'."""
    spans = []
    for data, rate, start in frs:
        n = len(data)
        if n <= 0 or not rate or rate <= 0:
            continue
        spans.append((start, start + timedelta(seconds=n / float(rate))))
    spans.sort(key=lambda s: s[0])
    for (s1, e1), (s2, _e2) in zip(spans, spans[1:]):
        if s2 < e1:
            return "TRUE_OVERLAP_UNRULED"
    return "DISJOINT"


def mark_coverage_infeasible(potentials) -> bool:
    """§2: a carrier below the 60-day potential floor in EITHER arm is
    COVERAGE_INFEASIBLE and must never be fetched."""
    return (int(potentials.get("incident_potential", 0)) < 60
            or int(potentials.get("activation_potential", 0)) < 60)


def validate_reuse_entry(entry, *, campaign_start_utc, contract_id, expected) -> bool:
    """§4 (REV 3 #3): existing physical bytes may be deduplicated ONLY on a fresh
    in-run re-attestation binding provider identity, session, digest, and a
    timestamp inside THIS campaign under THIS contract — validated against the
    EXPECTED immutable-object record, so a well-formed-but-WRONG identity,
    session, or digest refuses. Shape alone is never standing."""
    try:
        if not isinstance(entry, dict) or entry.get("reuse") is not True:
            return False
        if entry.get("contract_id") != contract_id:
            return False
        att = entry.get("attested_utc")
        if not isinstance(att, str) or att < campaign_start_utc:
            return False
        pid = entry.get("provider_identity")
        if not isinstance(pid, str) or not pid:
            return False
        h = entry.get("sha256")
        if not (isinstance(h, str) and len(h) == 64
                and all(c in "0123456789abcdef" for c in h)):
            return False
        sess = entry.get("session")
        if not (isinstance(sess, dict) and sess.get("start") and sess.get("end")):
            return False
        if not isinstance(expected, dict):
            return False
        if pid != expected.get("provider_identity"):
            return False
        if sess != expected.get("session"):
            return False
        if h != expected.get("sha256"):
            return False
        return True
    except Exception:
        return False


def validate_batch_form(batch) -> bool:
    """§4: outcome-blind honest batch — EXACTLY the three frozen targets in the
    contract order, each in a terminal state. Duplicates, reorders, substitutes,
    favorable subsets, wrong contract, and nonterminal states refuse; the honest
    0-candidate batch accepts."""
    try:
        if not isinstance(batch, dict) or batch.get("contract_id") != RENEWAL_CONTRACT_ID:
            return False
        rows = batch.get("carriers")
        if not isinstance(rows, list):
            return False
        keys = [r.get("carrier_key") for r in rows]
        if keys != list(_TARGET_ORDER):
            return False
        return all(r.get("state") in _TERMINAL_STATES for r in rows)
    except Exception:
        return False


def validate_renewal_capsule(capsule, *, expected_source_commit) -> bool:
    """§5: every required scalar/binding, independently. The source attestation
    must equal the exact expected renewal implementation commit — any pre-repair
    or mismatched commit refuses."""
    try:
        import math
        import seismic_data as SD
        c = capsule
        if not isinstance(c, dict) or c.get("schema") != "geospec-d2-calibration-v1":
            return False
        if c.get("region") not in _TARGET_ORDER:
            return False
        if c.get("band_tag") != "1-10Hz":
            return False
        if c.get("processing_version") != SD.PROCESSING_VERSION:
            return False
        if c.get("topology_version") != "t1":
            return False
        thr = c.get("threshold")
        if (isinstance(thr, bool) or not isinstance(thr, (int, float))
                or not math.isfinite(thr) or not (0.0 < float(thr) < 1.0)):
            return False
        win = c.get("calibration_window")
        if win != {"start": _ACTIVATION_START, "end": _ACTIVATION_END}:
            return False
        for k in ("input_manifest_sha256", "replay_output_sha256"):
            h = c.get(k)
            if not (isinstance(h, str) and len(h) == 64
                    and all(ch in "0123456789abcdef" for ch in h)):
                return False
        iu = c.get("issued_utc")
        if not isinstance(iu, str):
            return False
        issued = datetime.fromisoformat(iu.replace("Z", "+00:00"))
        if issued > datetime.now(timezone.utc):
            return False
        if c.get("valid_through") != _VALID_THROUGH:
            return False
        src = c.get("source_commit")
        if not (isinstance(src, str) and len(src) == 40
                and src == expected_source_commit):
            return False
        return True
    except Exception:
        return False


def compute_batch_root(entries) -> str:
    """§5 (REV 3 #1): the batch root derives ONLY from reopened-and-matched
    bytes — BOTH the capsule and the manifest are reopened from their entry
    paths, both digests are recomputed, and ANY declared-digest mismatch or
    missing artifact REFUSES. Never from an entry's claimed digest alone."""
    leaves = []
    for e in sorted(entries, key=lambda x: x["carrier_key"]):
        with open(e["capsule_path"], "rb") as fh:
            cap_sha = _sha(fh.read())
        if cap_sha != e["capsule_sha256"]:
            raise ValueError("declared capsule_sha256 does not match reopened "
                             "capsule bytes for " + str(e.get("carrier_key")))
        with open(e["manifest_path"], "rb") as fh:
            man_sha = _sha(fh.read())
        if man_sha != e["manifest_sha256"]:
            raise ValueError("declared manifest_sha256 does not match reopened "
                             "manifest bytes for " + str(e.get("carrier_key")))
        leaves.append({"carrier_key": e["carrier_key"],
                       "capsule_sha256": cap_sha,
                       "manifest_sha256": man_sha})
    return _sha(_canon(leaves))


def validate_registry_candidate(record, *, expected) -> bool:
    """§5: a registry candidate binds capsule path, exact capsule sha, region,
    topology, contract id, batch root, and NONEMPTY verification receipts —
    per-field flips refuse. A copied filename or matching threshold is not
    standing."""
    try:
        if not isinstance(record, dict) or not isinstance(expected, dict):
            return False
        for f in ("capsule_path", "capsule_sha256", "region", "topology_version",
                  "contract_id", "batch_root_sha256"):
            if record.get(f) != expected.get(f):
                return False
        rx = record.get("verification_receipts")
        if not (isinstance(rx, list) and rx):
            return False
        return True
    except Exception:
        return False


def renewal_admits(day, capsule, lift_effective_utc) -> bool:
    """§5 no-backfill boundary, executable: an unlifted capsule admits nothing; a
    gap day already refused stale (scored before the lift became effective) is
    NEVER retroactively admitted; past valid_through stays stale."""
    try:
        if not lift_effective_utc:
            return False
        if day > capsule["valid_through"]:
            return False
        if day <= str(lift_effective_utc)[:10]:
            return False
        return True
    except Exception:
        return False
