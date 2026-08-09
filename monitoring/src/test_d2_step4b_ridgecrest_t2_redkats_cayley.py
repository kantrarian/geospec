#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RIDGECREST T2 EXTENSION red-KAT supplement (cayley, 2026-08-09) — companion to
`test_d2_step4b_redkats_cayley.py`, per contract `codex-d2-step4b-2026-08-09-v1`
("ridgecrest is not admitted by an unregistered redraw") and
`docs/AMENDMENT_2026-08-09_ridgecrest_t2_topology.md` (outcome-blind little_lake redraw:
shared CI.LRL/CI.WBS replaced by unshared CI.WBM/CI.DTP; selection metadata-only, nearest two
live unshared stations to the unchanged polygon; no WBM/DTP waveform ever processed in this
program as of the amendment date).

ADDITIONAL PRODUCER SEAMS (extend `monitoring/src/d2_step4b_producer.py`; base seams unchanged)
===============================================================================================
* RIDGECREST_T2_REGISTRY_SHA256 =
  "449273b866f682d1363806daef5509cac40f1480d003a7fd0731b71a365f2657" (the amendment pin).
* load_ridgecrest_t2(registry_bytes: bytes) -> dict — verifies
  sha256(registry_bytes) == RIDGECREST_T2_REGISTRY_SHA256 then parses; ANY byte difference
  refuses (ValueError). There is no unpinned load path.
* REV 2 (codex 0520 F1): build_ridgecrest_extension_plan(activation_reference: str,
  registry_bytes: bytes) -> dict — the PUBLIC builder consumes PINNED BYTES ONLY and calls
  load_ridgecrest_t2 itself; a plain dict (however valid-looking) and any re-encoded
  mutation of the registry REFUSE. No second public raw-dict builder exists; structural
  validation (shared NET.STA / thin segments / wrong identity) may live in a private
  validator behind the pin. Plan semantics unchanged:
    - carrier "ridgecrest", topology_version "t2", provider "s3://scedc-pds";
    - incident arm = schedule_days(CAMPAIGN["incident_reference"]), activation arm =
      schedule_days(activation_reference), same [ref-120d, ref-30d) rule as the base plan;
    - station/NSLC structure comes ONLY from the registry (frozen candidate order preserved);
    - fail-closed ValueError on: segments sharing any NET.STA, fewer than 2 stations in any
      segment, fewer than 2 segments, wrong region/topology_version/provider in the registry;
    - OUTCOME-BLIND and deterministic exactly like build_campaign_plan (no
      ratio/threshold/lambda/admitted/verdict fields; byte-identical canonical JSON on
      identical inputs; digestable via plan_digest).
* The BASE campaign is untouched: build_campaign_plan still refuses carrier "ridgecrest"
  (three-carrier policy), and admit_candidate("ridgecrest", ...) with no replay evidence
  remains BLOCKED_REPLAY_UNAVAILABLE — this supplement registers a redraw; it does not mint
  admission, lift a freeze, or commission production topology.

RED AS AUTHORED at RT-0b (producer module / extension seams absent). RT-0a and RT-5 are
context checks on committed artifacts (registry file, production t1 state) and are green.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

REGISTRY_PATH = os.path.join(HERE, "ridgecrest_t2_registry.json")
REGISTRY_SHA256 = "449273b866f682d1363806daef5509cac40f1480d003a7fd0731b71a365f2657"

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


def netsta(nslc):
    parts = nslc.split(".")
    return f"{parts[0]}.{parts[1]}"


def main():
    # ---- RT-0a: the registered redraw artifact itself (green context check) ----
    with open(REGISTRY_PATH, "rb") as f:
        reg_bytes = f.read()
    ok_sha = hashlib.sha256(reg_bytes).hexdigest() == REGISTRY_SHA256
    reg = json.loads(reg_bytes.decode("utf-8")) if ok_sha else {}
    segs = reg.get("segments", {})
    all_stations = [(seg, st) for seg in segs for st in segs[seg]["stations"]]
    counts = {}
    for _, st in all_stations:
        counts[st] = counts.get(st, 0) + 1
    ll = segs.get("little_lake", {}).get("stations", {})
    check("RT-0a registry artifact: byte-sha matches the amendment pin; region/t2/provider "
          "exact; three segments; NET.STA disjoint; little_lake = {CI.JRC2, CI.WBM, CI.DTP}; "
          "every NSLC list is [BHZ, HHZ] for its own station",
          ok_sha and reg.get("region") == "ridgecrest"
          and reg.get("topology_version") == "t2"
          and reg.get("provider") == "s3://scedc-pds"
          and set(segs) == {"ridgecrest_mainshock", "airport_lake", "little_lake"}
          and all(c == 1 for c in counts.values())
          and set(ll) == {"CI.JRC2", "CI.WBM", "CI.DTP"}
          and all(segs[seg]["stations"][st]["nslc_candidates"]
                  == [f"{st}..BHZ", f"{st}..HHZ"] for seg, st in all_stations)
          and all(len(segs[seg]["stations"]) >= 2 for seg in segs),
          f"sha_ok={ok_sha} little_lake={sorted(ll)}")

    # ---- RT-5: production stays t1 / blocked (green context check) ------------
    # Prefer the production gate itself; where fault_correlation's obspy dependency is
    # absent (geomen system python), apply the identical shared-NET.STA counting rule to
    # the SAME authoritative source (fault_segments.get_segments_for_region). E-host and
    # codex runs exercise validate_topology directly.
    try:
        try:
            from fault_correlation import validate_topology
            ok_t, reasons = validate_topology("ridgecrest")
        except ImportError:
            from fault_segments import get_segments_for_region
            cnt = {}
            for s in get_segments_for_region("ridgecrest"):
                for ns in {f"{st.network}.{st.code}" for st in s.stations}:
                    cnt[ns] = cnt.get(ns, 0) + 1
            dup = sorted(k for k, c in cnt.items() if c >= 2)
            ok_t, reasons = (not dup, [f"shared NET.STA across correlated segments: {dup}"]
                             if dup else [])
        joined = " ".join(reasons)
        check("RT-5 production ridgecrest REMAINS t1-blocked (shared CI.LRL + CI.WBS named) "
              "-- no silent commissioning of t2",
              ok_t is False and "CI.LRL" in joined and "CI.WBS" in joined,
              f"ok={ok_t} reasons={reasons}")
    except Exception as exc:
        check("RT-5 production topology state readable", False, str(exc))

    # ---- RT-0b: extension seams (the RED gate as authored) ---------------------
    try:
        import d2_step4b_producer as P
    except ImportError:
        check("RT-0b producer module + ridgecrest-t2 extension seams", False,
              "AWAITING grassmann's producer -- red-first as authored")
        return
    need = ("RIDGECREST_T2_REGISTRY_SHA256", "load_ridgecrest_t2",
            "build_ridgecrest_extension_plan", "build_campaign_plan", "schedule_days",
            "plan_digest", "admit_candidate", "CAMPAIGN")
    if not all(hasattr(P, n) for n in need):
        check("RT-0b producer module + ridgecrest-t2 extension seams", False,
              "AWAITING extension seams -- red-first as authored")
        return

    # ---- RT-1: the pin is the amendment's pin ----------------------------------
    check("RT-1 producer pins the amendment registry sha",
          P.RIDGECREST_T2_REGISTRY_SHA256 == REGISTRY_SHA256)

    # ---- RT-2: pinned load path only -------------------------------------------
    loaded = P.load_ridgecrest_t2(reg_bytes)
    check("RT-2 load_ridgecrest_t2 verifies bytes then parses; ANY byte change refuses",
          loaded == reg
          and raises(lambda: P.load_ridgecrest_t2(reg_bytes + b" "))
          and raises(lambda: P.load_ridgecrest_t2(
              reg_bytes.replace(b"CI.DTP", b"CI.RRC"))))

    # ---- RT-2b (REV 2, codex 0520 F1): the builder consumes PINNED BYTES only --
    def _reenc(mutated):
        return (json.dumps(mutated, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True) + "\n").encode()

    forged = json.loads(reg_bytes.decode("utf-8"))
    forged["segments"]["little_lake"]["stations"]["CI.WBM"]["nslc_candidates"] = \
        ["CI.FORGED..BHZ"]
    check("RT-2b PUBLIC builder refuses an unattested dict (however valid-looking) AND "
          "re-encoded bytes of any candidate mutation (codex 0520 forge reproduced)",
          raises(lambda: P.build_ridgecrest_extension_plan("2026-08-09", loaded))
          and raises(lambda: P.build_ridgecrest_extension_plan("2026-08-09", dict(forged)))
          and raises(lambda: P.build_ridgecrest_extension_plan("2026-08-09",
                                                               _reenc(forged))))

    # ---- RT-3: outcome-blind deterministic extension plan (from pinned bytes) --
    try:
        p1 = P.build_ridgecrest_extension_plan("2026-08-09", reg_bytes)
        p2 = P.build_ridgecrest_extension_plan("2026-08-09", reg_bytes)
    except Exception as exc:
        check("RT-3-GATE builder consumes the pinned registry bytes", False,
              f"AWAITING bytes-only builder (codex 0520 F1) -- {exc!r}")
        return
    b1 = json.dumps(p1, sort_keys=True, separators=(",", ":"))
    check("RT-3 extension plan: carrier/t2/provider exact; both arms match schedule_days; "
          "deterministic; 64-hex digest",
          b1 == json.dumps(p2, sort_keys=True, separators=(",", ":"))
          and p1.get("carrier") == "ridgecrest"
          and p1.get("topology_version") == "t2"
          and p1.get("provider") == "s3://scedc-pds"
          and p1.get("incident_arm") == P.schedule_days(P.CAMPAIGN["incident_reference"])
          and p1.get("activation_arm") == P.schedule_days("2026-08-09")
          and len(P.plan_digest(p1)) == 64)
    plan_text = b1.lower()
    check("RT-3b extension plan is OUTCOME-BLIND (no ratio/threshold/lambda/admitted/verdict "
          "fields)",
          all(tok not in plan_text for tok in ('"ratio', '"threshold', "lambda2", "admitted",
                                               "artifact_removed", "control_clear")))
    frozen_order_ok = all(
        p1_seg["stations"][st]["nslc_candidates"] == segs[seg]["stations"][st]["nslc_candidates"]
        for seg, p1_seg in p1.get("segments", {}).items()
        for st in p1_seg.get("stations", {}))
    check("RT-3c extension plan preserves the registry's frozen NSLC candidate order verbatim",
          set(p1.get("segments", {})) == set(segs) and frozen_order_ok)

    # ---- RT-4 (REV 2): doctored registries refuse as RE-ENCODED BYTES ----------
    # With the bytes-only public builder the pin subsumes structural doctoring: every
    # mutation, canonically re-encoded, must refuse (no path exists for an unattested dict).
    shared = json.loads(reg_bytes.decode("utf-8"))
    shared["segments"]["little_lake"]["stations"]["CI.LRL"] = \
        shared["segments"]["airport_lake"]["stations"]["CI.LRL"]
    thin = json.loads(reg_bytes.decode("utf-8"))
    thin["segments"]["little_lake"]["stations"] = \
        {"CI.JRC2": thin["segments"]["little_lake"]["stations"]["CI.JRC2"]}
    oneseg = json.loads(reg_bytes.decode("utf-8"))
    oneseg["segments"] = {"little_lake": oneseg["segments"]["little_lake"]}
    wrongprov = json.loads(reg_bytes.decode("utf-8"))
    wrongprov["provider"] = "eida.koeri.boun.edu.tr"
    check("RT-4 doctored registries refuse as re-encoded bytes: shared NET.STA / "
          "single-station segment / single segment / wrong provider",
          raises(lambda: P.build_ridgecrest_extension_plan("2026-08-09", _reenc(shared)))
          and raises(lambda: P.build_ridgecrest_extension_plan("2026-08-09", _reenc(thin)))
          and raises(lambda: P.build_ridgecrest_extension_plan("2026-08-09", _reenc(oneseg)))
          and raises(lambda: P.build_ridgecrest_extension_plan("2026-08-09",
                                                               _reenc(wrongprov))))

    # ---- RT-6: base three-carrier policy intact ---------------------------------
    base_with_rc = {"ridgecrest": {seg: [segs[seg]["stations"][st]["nslc_candidates"]
                                         for st in sorted(segs[seg]["stations"])]
                                   for seg in segs}}
    check("RT-6 build_campaign_plan STILL refuses carrier 'ridgecrest' (base three-carrier "
          "policy unchanged by the supplement)",
          raises(lambda: P.build_campaign_plan(base_with_rc, "2026-08-09")))

    # ---- RT-7: admission honesty -------------------------------------------------
    arm = {"admitted_days": 90, "threshold": 0.05}
    st7, _ = P.admit_candidate("ridgecrest", arm, arm, None)
    check("RT-7 ridgecrest with full calibration but NO replay evidence is deterministically "
          "BLOCKED_REPLAY_UNAVAILABLE (extension mints thresholds, never admission)",
          st7 == "BLOCKED_REPLAY_UNAVAILABLE", f"status={st7}")


main()
print()
if FAILS:
    print(f"RIDGECREST T2 SUPPLEMENT RED-KAT FAILURES: {FAILS}")
    sys.exit(1)
print("ALL RIDGECREST T2 SUPPLEMENT RED-KATs PASS (pinned registry + outcome-blind extension "
      "plan + fail-closed doctoring + base policy intact + no admission path)")
