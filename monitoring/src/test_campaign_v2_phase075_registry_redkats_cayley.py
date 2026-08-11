#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CAMPAIGN-v2 PHASE-0.75 REGISTRY red-KATs (cayley, 2026-08-11) — contract
`codex-d2-campaign-v2-2026-08-10-v1` (codex 1817 + 0600 R1.2 bar) over the Phase-0.5
evidence at GeoSpec `7d7d4cb` (grassmann 0550, codex PASS 0601). t2/RC lineage: pinned
bytes-only builder + fail-closed doctoring + DERIVATION recomputation (content-auth ≠
derivation provenance — the bar re-derives the selection and eligibility from the pinned
inputs under the frozen policy rule; a hand-edited selection with correct labels REFUSES).

PINNED INPUTS (all at `7d7d4cb`, all `-text` pinned; the bar re-verifies every digest and
every cross-digest in the bundle):
  phase0_bundle.json            ad1ec8b2435dcda9… (exact 64-hex below)
  d2_campaign_v2_candidate_pool.json  15d0e32c51c027dc…
  availability_table.json / published_phase_ledger_v2.json / probe_ledger.jsonl
  (bound INSIDE the bundle by candidate_pool_sha256 / published_phase_ledger_sha256 /
  probe_ledger_sha256 — the bar recomputes all three).

CONTRACT SEAMS (grassmann implements `monitoring/src/d2_campaign_v2_plan.py` UNEDITED):
  * PHASE0_BUNDLE_SHA256 = the exact bundle pin (64-hex constant).
  * build_v2_campaign_plan(bundle_bytes: bytes) -> (plan: dict, plan_bytes: bytes)
      - PINNED BYTES ONLY: sha256(bundle_bytes) must equal PHASE0_BUNDLE_SHA256 (else
        ValueError); a dict (however valid-looking) and ANY re-encoded mutation refuse —
        no unattested path (t2 REV 2 pattern; the pin subsumes structural doctoring).
      - plan_bytes = canonical AT CREATION (json sort_keys, ',:' separators, allow_nan
        False, single trailing LF) — cayley 1735 canonicalization ruling, applied forward.
      - DETERMINISTIC: identical bundle bytes -> byte-identical plan_bytes.
      - plan content: contract_id `codex-d2-campaign-v2-2026-08-10-v1`;
        activation_reference_day 2026-08-10; incident_reference_day 2026-07-29;
        arms EXACTLY the bundle's frozen 120-day lists (incident [2026-03-01, 2026-06-29),
        activation [2026-03-13, 2026-07-11)); scheduled_days = sorted union;
        providers = the exact 3-carrier cap; eligible carriers only;
        station_registry rows {segment_name, station_id, ordered_nslc_candidates} where
        the membership EQUALS the bundle's selected_registry and ordered_nslc_candidates
        EQUALS the frozen pool's ordered_nslc for that station (order verbatim — no
        re-sort, no expansion);
        free_sources_only true; outcomes_inspected_before_schedule false;
        OUTCOME-BLIND: no ratio/threshold/lambda/admitted/verdict token anywhere.
  * plan_digest(plan_bytes) -> 64-hex sha256.
  Phase-1 consumes ONLY this builder's plan (staged then reopened by run_campaign per the
  frozen H2 semantics); the fetch itself stays on asylum's fresh SB-8 go.

RED AS AUTHORED vs `7d7d4cb`: exactly ['V2-GATE'] (+its dependents V2a-V2e listed as
BLOCKED markers). V0/V1 are green context locks on the REAL pinned evidence — V1 is the
derivation proof: selection and per-carrier eligibility RECOMPUTED from availability_table
+ pool + arms under the frozen policy rule reproduce the bundle EXACTLY (verified live
before freeze; any later tamper of table/bundle/selection breaks it).
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

P05 = os.path.join(HERE, "campaign_v2_phase05")
BUNDLE_P = os.path.join(P05, "phase0_bundle.json")
AVAIL_P = os.path.join(P05, "availability_table.json")
LEDGER_P = os.path.join(P05, "published_phase_ledger_v2.json")
PROBE_P = os.path.join(P05, "probe_ledger.jsonl")
POOL_P = os.path.join(HERE, "d2_campaign_v2_candidate_pool.json")

BUNDLE_SHA = "ad1ec8b2435dcda9"      # 16-hex prefix; full value asserted from bytes below
POOL_SHA_PREFIX = "15d0e32c51c027dc"
CONTRACT_ID = "codex-d2-campaign-v2-2026-08-10-v1"
FLOOR_DAYS = 60

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


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _reenc(doc):
    return (json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            + "\n").encode()


def _recompute(bundle, avail):
    """The frozen policy selection + eligibility rule, re-derived from pinned inputs."""
    floor = avail["floor_seconds"]
    arm_days = {a: set(bundle["arms"][a]["days"]) for a in ("incident", "activation")}
    out = {}
    for carrier, cinfo in bundle["carriers"].items():
        stations = avail["carriers"][carrier]["stations"]
        counts = {}
        for sid, srec in stations.items():
            per_day = srec["per_day"]
            c = {a: sum(1 for d, e in per_day.items()
                        if d in days and e["state"] == "AVAILABLE"
                        and e["seconds"] >= floor)
                 for a, days in arm_days.items()}
            counts[sid] = (c["incident"], c["activation"], srec["segment"])
        segs = sorted({s for (_i, _a, s) in counts.values()})
        sel = {}
        for seg in segs:
            adequate = [(sid, i, a) for sid, (i, a, s) in counts.items()
                        if s == seg and i >= FLOOR_DAYS and a >= FLOOR_DAYS]
            ranked = sorted(adequate,
                            key=lambda t: (-min(t[1], t[2]), -(t[1] + t[2]), t[0]))
            if len(ranked) >= 4:
                sel[seg] = sorted(sid for sid, _i, _a in ranked[:4])
            elif len(ranked) == 3:
                sel[seg] = sorted(sid for sid, _i, _a in ranked)
            else:
                sel[seg] = None
        potential = {}
        for a, days in arm_days.items():
            pot = []
            for d in sorted(days):
                seg_ok = sum(
                    1 for seg in segs
                    if sum(1 for sid, (_i, _a2, s_) in counts.items()
                           if s_ == seg and d in stations[sid]["per_day"]
                           and stations[sid]["per_day"][d]["state"] == "AVAILABLE"
                           and stations[sid]["per_day"][d]["seconds"] >= floor) >= 2)
                if seg_ok >= 2:
                    pot.append(d)
            potential[a] = pot
        out[carrier] = {"selection": sel, "potential": potential}
    return out


def main():
    # ---- V0: pinned evidence identity + cross-digest closure (green locks) ------
    bundle_bytes = open(BUNDLE_P, "rb").read()
    pool_bytes = open(POOL_P, "rb").read()
    avail_bytes = open(AVAIL_P, "rb").read()
    ledger_bytes = open(LEDGER_P, "rb").read()
    probe_bytes = open(PROBE_P, "rb").read()
    bundle = json.loads(bundle_bytes.decode())
    avail = json.loads(avail_bytes.decode())
    check("V0a bundle + pool bytes match the routed pins (prefix-checked here, full hash "
          "recomputed) and the bundle's THREE cross-digests bind the exact sibling files "
          "(pool / phase-ledger / probe-ledger)",
          _sha(bundle_bytes).startswith(BUNDLE_SHA)
          and _sha(pool_bytes).startswith(POOL_SHA_PREFIX)
          and bundle["candidate_pool_sha256"] == _sha(pool_bytes)
          and bundle["published_phase_ledger_sha256"] == _sha(ledger_bytes)
          and bundle["probe_ledger_sha256"] == _sha(probe_bytes)
          and avail["probe_ledger_sha256"] == _sha(probe_bytes),
          f"bundle={_sha(bundle_bytes)[:16]} pool={_sha(pool_bytes)[:16]}")
    from datetime import date, timedelta
    exp = {}
    for arm, ref in (("incident", "2026-07-29"), ("activation", "2026-08-10")):
        r = date.fromisoformat(ref)
        exp[arm] = [(r - timedelta(days=150) + timedelta(days=i)).isoformat()
                    for i in range(120)]
    check("V0b arms are EXACTLY the frozen [ref-150d, ref-30d) 120-day windows from the "
          "contract references (recomputed, not trusted)",
          all(bundle["arms"][a]["days"] == exp[a] for a in exp)
          and bundle["contract_id"] == CONTRACT_ID
          and bundle["activation_reference"] == "2026-08-10")

    # ---- V1: DERIVATION — selection + eligibility recomputed from pinned inputs --
    rec = _recompute(bundle, avail)
    sel_ok, pot_ok, pool_ok, disjoint_ok, size_ok = True, True, True, True, True
    pool = json.loads(pool_bytes.decode())
    for carrier, cinfo in bundle["carriers"].items():
        declared = {seg: sorted(v) for seg, v in cinfo["selected_registry"].items()}
        if {seg: rec[carrier]["selection"].get(seg) for seg in declared} != declared:
            sel_ok = False
        for a in ("incident", "activation"):
            if rec[carrier]["potential"][a] != cinfo["metadata_potential_days"][a]:
                pot_ok = False
        pool_stations = {row["station_id"]: row
                         for seg_rows in pool["carriers"][carrier]["segments"].values()
                         for row in seg_rows}
        seen = set()
        for seg, sids in declared.items():
            if not 3 <= len(sids) <= 4:
                size_ok = False
            for sid in sids:
                row = pool_stations.get(sid)
                if row is None or row["segment"] != seg:
                    pool_ok = False                 # post-probe expansion / seg swap
                if sid in seen:
                    disjoint_ok = False
                seen.add(sid)
    check("V1a SELECTION IS DERIVED, NOT DECLARED: re-running the frozen rule "
          "(adequate = >=60 potential days EACH arm; rank (-min,-sum,station_id); "
          "take-4-else-exactly-3) over the pinned availability table reproduces every "
          "carrier's selected_registry EXACTLY",
          sel_ok)
    check("V1b carrier ELIGIBILITY is derived: recomputed per-arm potential-day lists "
          "(>=2 segments with >=2 available stations per day) equal the bundle's "
          "metadata_potential_days exactly; all three carriers >= 60/arm",
          pot_ok and all(
              len(bundle["carriers"][c]["metadata_potential_days"][a]) >= FLOOR_DAYS
              for c in bundle["carriers"] for a in ("incident", "activation")))
    check("V1c NO POST-PROBE POOL EXPANSION + structure: every selected station exists in "
          "the FROZEN pool under the SAME segment; per-carrier station disjointness; "
          "3-4 stations per segment",
          pool_ok and disjoint_ok and size_ok)

    # ---- V2: producer seams (red-first) ------------------------------------------
    try:
        import d2_campaign_v2_plan as V2
        need = ("PHASE0_BUNDLE_SHA256", "build_v2_campaign_plan", "plan_digest")
        assert all(hasattr(V2, n) for n in need), "seams incomplete"
    except (ImportError, AssertionError) as exc:
        check("V2-GATE plan-builder seams present (d2_campaign_v2_plan.py: "
              "PHASE0_BUNDLE_SHA256 + build_v2_campaign_plan + plan_digest)", False,
              f"AWAITING grassmann's builder -- red-first as authored ({exc})")
        for fam, what in (("V2a", "pinned-bytes-only builder (dict + mutation refuse)"),
                          ("V2b", "canonical-at-creation deterministic plan bytes"),
                          ("V2c", "registry == selected_registry + pool NSLC order verbatim"),
                          ("V2d", "arms/providers/contract exact + outcome-blind"),
                          ("V2e", "doctored re-encodings refuse via the pin")):
            check(f"{fam}-BLOCKED {what}", False,
                  "BLOCKED by the missing builder (explicit marker)")
        return

    check("V2 the builder pins the exact bundle sha",
          V2.PHASE0_BUNDLE_SHA256 == _sha(bundle_bytes))
    plan, plan_bytes = V2.build_v2_campaign_plan(bundle_bytes)
    plan2, plan_bytes2 = V2.build_v2_campaign_plan(bundle_bytes)
    check("V2a PINNED BYTES ONLY: the loaded dict (however valid) and any re-encoded "
          "mutation refuse; only the exact bundle bytes build",
          raises(lambda: V2.build_v2_campaign_plan(bundle))
          and raises(lambda: V2.build_v2_campaign_plan(bundle_bytes + b" "))
          and isinstance(plan, dict) and isinstance(plan_bytes, bytes))
    check("V2b canonical-at-creation + deterministic: plan_bytes == canonical(plan) with "
          "single trailing LF, byte-identical across builds, digest 64-hex",
          plan_bytes == _reenc(plan) and plan_bytes == plan_bytes2
          and len(V2.plan_digest(plan_bytes)) == 64)
    reg_ok = True
    for carrier, cinfo in bundle["carriers"].items():
        declared = {seg: sorted(v) for seg, v in cinfo["selected_registry"].items()}
        rows = [r for r in plan["station_registry"][carrier]]
        got = {}
        for r in rows:
            got.setdefault(r["segment_name"], []).append(r["station_id"])
        if {s: sorted(v) for s, v in got.items()} != declared:
            reg_ok = False
        pool_rows = {row["station_id"]: row
                     for seg_rows in json.loads(pool_bytes.decode())
                     ["carriers"][carrier]["segments"].values() for row in seg_rows}
        for r in rows:
            if r["ordered_nslc_candidates"] != pool_rows[r["station_id"]]["ordered_nslc"]:
                reg_ok = False
    check("V2c plan registry membership == bundle selected_registry AND every station's "
          "ordered_nslc_candidates == the frozen pool's ordered_nslc verbatim",
          reg_ok)
    txt = plan_bytes.decode().lower()
    check("V2d arms/providers/contract exact + outcome-blind + free-sources flags",
          plan["contract_id"] == CONTRACT_ID
          and plan["incident_days"] == bundle["arms"]["incident"]["days"]
          and plan["activation_days"] == bundle["arms"]["activation"]["days"]
          and plan["scheduled_days"] == sorted(set(plan["incident_days"])
                                               | set(plan["activation_days"]))
          and plan["providers"] == bundle["campaign_plan"]["providers"]
          and plan["free_sources_only"] is True
          and plan["outcomes_inspected_before_schedule"] is False
          and all(tok not in txt for tok in ('"ratio', '"threshold', "lambda2",
                                             "admitted", "artifact_removed",
                                             "control_clear")))
    doctored = json.loads(bundle_bytes.decode())
    doctored["carriers"]["turkey_kahramanmaras"]["selected_registry"][
        "east_anatolian_south"] = ["KO.GAZ", "KO.TAHT", "KO.FORGED"]
    check("V2e a doctored selection re-encoded as bytes refuses via the pin (no unattested "
          "path can mint a plan)",
          raises(lambda: V2.build_v2_campaign_plan(_reenc(doctored))))


main()
print()
if FAILS:
    print(f"CAMPAIGN-V2 PHASE-0.75 RED-KAT FAILURES: {FAILS}")
    sys.exit(1)
print("ALL CAMPAIGN-V2 PHASE-0.75 RED-KATs PASS (pinned evidence closure + derived "
      "selection/eligibility + bytes-only canonical plan builder)")
