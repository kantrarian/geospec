#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Campaign-v2 plan builder (Phase 0.75) -- contract `codex-d2-campaign-v2-2026-08-10-v1`.

Bytes-only, pinned, deterministic (t2 REV 2 pattern): the ONLY input that can mint a v2
campaign plan is the exact Phase-0.5 evidence bundle bytes (GeoSpec `7d7d4cb`,
`campaign_v2_phase05/phase0_bundle.json`); a dict -- however valid-looking -- and ANY
re-encoded mutation refuse via the SHA-256 pin, so no unattested path exists. The plan is
derived verbatim from the bundle's owner-approved, codex-verified content (selected
registries, arms, providers); this module adds NOTHING that is not already pinned there.

plan_bytes are canonical AT CREATION (sort_keys, ',:' separators, allow_nan=False, ASCII,
single trailing LF) per cayley's 1735 inputs-vs-artifacts ruling applied forward: v2 plan
bytes are born canonical so downstream identity (campaign_id = sha256(plan_bytes || ...))
never depends on a later rewrite. Phase-1 consumes ONLY this builder's plan; the fetch
itself remains on asylum's fresh SB-8 go.
"""
import hashlib
import json

# The exact Phase-0.5 bundle (GeoSpec 7d7d4cb, codex 0600 R1.2 bar = PASS).
PHASE0_BUNDLE_SHA256 = "ad1ec8b2435dcda9f596a1f72d4fe01621007629123e17a3598ee1df66a708d6"

_CONTRACT_ID = "codex-d2-campaign-v2-2026-08-10-v1"
_SCHEMA = "geospec-d2-campaign-v2-plan-v1"


def _canonical(doc):
    return (json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                       allow_nan=False) + "\n").encode()


def plan_digest(plan_bytes):
    """SHA-256 hexdigest of the canonical plan bytes."""
    if not isinstance(plan_bytes, (bytes, bytearray)):
        raise ValueError("plan_digest requires the plan BYTES")
    return hashlib.sha256(bytes(plan_bytes)).hexdigest()


def build_v2_campaign_plan(bundle_bytes):
    """(plan: dict, plan_bytes: bytes) from the EXACT pinned Phase-0.5 bundle bytes.

    Refuses anything that is not byte-identical to the pinned bundle: non-bytes input
    (a parsed dict has no attestation), and any mutation/re-encoding (pin mismatch).
    Deterministic: identical input bytes -> byte-identical plan_bytes.
    """
    if not isinstance(bundle_bytes, (bytes, bytearray)):
        raise ValueError("bundle must be the exact attested BYTES, not a parsed object")
    raw = bytes(bundle_bytes)
    if hashlib.sha256(raw).hexdigest() != PHASE0_BUNDLE_SHA256:
        raise ValueError("bundle bytes do not match PHASE0_BUNDLE_SHA256 -- refusing")
    bundle = json.loads(raw.decode("utf-8"))
    if bundle.get("contract_id") != _CONTRACT_ID:
        raise ValueError("bundle contract_id drift")

    incident_days = list(bundle["arms"]["incident"]["days"])
    activation_days = list(bundle["arms"]["activation"]["days"])
    eligible = list(bundle["campaign_plan"]["eligible_carriers"])

    station_registry = {}
    for carrier in eligible:
        cinfo = bundle["carriers"][carrier]
        # nslc_candidates in the bundle's segment rows are the frozen pool's ordered_nslc
        # verbatim (bar-verified); index them for the selected stations.
        nslc_of = {row["station_id"]: list(row["nslc_candidates"])
                   for rows in cinfo["segments"].values() for row in rows}
        rows_out = []
        for segment in sorted(cinfo["selected_registry"]):
            for station_id in cinfo["selected_registry"][segment]:
                rows_out.append({"segment_name": segment,
                                 "station_id": station_id,
                                 "ordered_nslc_candidates": nslc_of[station_id]})
        station_registry[carrier] = rows_out

    plan = {
        "schema": _SCHEMA,
        "contract_id": _CONTRACT_ID,
        "activation_reference_day": bundle["activation_reference"],
        "incident_reference_day": bundle["arms"]["incident"]["reference"],
        "incident_days": incident_days,
        "activation_days": activation_days,
        "scheduled_days": sorted(set(incident_days) | set(activation_days)),
        "carriers": sorted(eligible),
        "providers": bundle["campaign_plan"]["providers"],
        "station_registry": station_registry,
        "free_sources_only": True,
        "outcomes_inspected_before_schedule": False,
        "phase0_bundle_sha256": PHASE0_BUNDLE_SHA256,
    }
    return plan, _canonical(plan)
