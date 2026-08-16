#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D2 CALIBRATION RENEWAL red-KATs (cayley, 2026-08-16) — the acceptance bars for codex's
frozen renewal contract `codex-d2-campaign-v2-renewal-2026-08-16-v1` (inbox `80bb3c6`),
§6 areas 1-5. Owner authority: asylum "renew koeri" + all-three-carriers scope (relay
`95d7dc2`, inbox-relay rule).

ARCHITECTURE: the renewal is a V2 RE-ISSUANCE — estimator, selection, gates, capsule
schema, claim limits UNCHANGED. The frozen v2 bar suites therefore REMAIN the enforcement
for the unchanged semantics, and this bar COMPOSES them (subprocess green required):
  - test_d2_koeri_order_canon_redkats_cayley.py   (repaired-shell order KATs, §3)
  - test_d2_step4b_executor_redkats_cayley.py     (executor H-gates, receipts, §4)
  - test_d2_step4b_provider_redkats_cayley.py     (provider I/O discipline, §4)
  - test_campaign_v2_phase075_registry_redkats_cayley.py (selection semantics, §2)
This file pins ONLY the renewal deltas, red-first.

SEAMS PINNED BY THIS BAR (naming decisions implementing the contract):
  module `monitoring/src/d2_renewal_plan.py` exposing:
    RENEWAL_CONTRACT_ID = "codex-d2-campaign-v2-renewal-2026-08-16-v1"
    RENEWAL_ANCHOR = "2026-08-16"
    V2_POOL_SHA256 = "15d0e32c51c027dc144c5c6d57ec5f100a59374f6248abfc5c56ee38628ddc67"
    renewal_arms() -> dict  (incident/activation day lists + windows + union + embargo +
                             valid_through + expiry_utc + incident_reference)
    build_renewal_candidate_pool(v2_pool_bytes) -> (pool, pool_bytes)
        bytes-pinned on V2_POOL_SHA256; copy-only re-envelope (new contract id +
        creation stamp + digest); REFUSES dicts, mutated bytes, and any
        availability/outcome-bearing field.
    build_renewal_plan(bundle_bytes) -> (plan, plan_bytes)
        bytes-pinned on the renewal Phase-0.5-equivalent bundle (module constant
        RENEWAL_BUNDLE_SHA256, set at that bundle's freeze); refuses non-bytes and
        contract-id drift; plan carries activation_reference_day == RENEWAL_ANCHOR,
        core_blobs (the §3 blob map), and NO stale source attestation.
  executor delta (d2_step4b_campaign_run):
    a TRUE-overlap refusal in acquisition must surface reason code
    "TRUE_OVERLAP_UNRULED" on the station attempt (hold, never dedup/merge/consume) —
    contract §3; enumeration-order repair is in scope, O-1..O-4 policy is NOT.

EXPECTED ON THE CURRENT TREE (red-first): RN-0 RED (renewal module absent) which
short-circuits RN-1/RN-2 sub-checks to their awaiting-reds; RN-3a/RN-4a compose-greens
PASS (the frozen suites are green on the landed tree); RN-3c TRUE_OVERLAP_UNRULED RED
(reason code absent); RN-5 loader stale-boundary lock GREEN (existing loader semantics).
GREEN = all checks PASS after grassmann's bar-unedited implementation. Provider I/O
remains gated on the FRESH fire-time owner go naming the contract id and A; nothing in
this bar or its fixtures performs external I/O (hermetic stubs only).
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

FAILS = []

CONTRACT_ID = "codex-d2-campaign-v2-renewal-2026-08-16-v1"
ANCHOR = "2026-08-16"
V2_POOL_SHA = "15d0e32c51c027dc144c5c6d57ec5f100a59374f6248abfc5c56ee38628ddc67"
STALE_SOURCE_COMMIT_PREFIX = "292b1069"
TARGET_ORDER = ["istanbul_marmara", "socal_coachella", "turkey_kahramanmaras"]


def check(desc, ok, detail=""):
    print(f"    [{'PASS' if ok else 'FAIL'}] {desc}" + (f" - {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(desc)


def days_between(a, b):
    """ISO day list for [a, b) — the bar's independent date arithmetic."""
    out = []
    d = date.fromisoformat(a)
    end = date.fromisoformat(b)
    while d < end:
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def run_suite(name, timeout=1800):
    r = subprocess.run([sys.executable, "-B", os.path.join(HERE, name)],
                       capture_output=True, text=True, timeout=timeout, cwd=HERE)
    return r.returncode == 0, (r.stdout + r.stderr)[-200:]


def main():
    # =====================================================================
    # RN-0 — pinned renewal seams present
    # =====================================================================
    try:
        import d2_renewal_plan as RP
    except ImportError as e:
        check("RN-0 pinned module d2_renewal_plan (RENEWAL_CONTRACT_ID/RENEWAL_ANCHOR/"
              "V2_POOL_SHA256/renewal_arms/build_renewal_candidate_pool/"
              "build_renewal_plan)", False, f"import failed: {e}")
        RP = None
    if RP is not None:
        need = (getattr(RP, "RENEWAL_CONTRACT_ID", None) == CONTRACT_ID
                and getattr(RP, "RENEWAL_ANCHOR", None) == ANCHOR
                and getattr(RP, "V2_POOL_SHA256", None) == V2_POOL_SHA
                and all(callable(getattr(RP, f, None)) for f in
                        ("renewal_arms", "build_renewal_candidate_pool",
                         "build_renewal_plan")))
        check("RN-0 pinned module d2_renewal_plan (RENEWAL_CONTRACT_ID/RENEWAL_ANCHOR/"
              "V2_POOL_SHA256/renewal_arms/build_renewal_candidate_pool/"
              "build_renewal_plan)", need)
        if not need:
            RP = None

    # =====================================================================
    # RN-1 — §6.1 exact arithmetic + identity (independent recompute)
    # =====================================================================
    exp_incident = days_between("2026-03-01", "2026-06-29")
    exp_activation = days_between("2026-03-19", "2026-07-17")
    exp_union = sorted(set(exp_incident) | set(exp_activation))
    A = date.fromisoformat(ANCHOR)
    arith_ok = (len(exp_incident) == 120 and len(exp_activation) == 120
                and len(exp_union) == 138
                and exp_activation[0] == (A - timedelta(days=150)).isoformat()
                and (date.fromisoformat(exp_activation[-1])
                     + timedelta(days=1)).isoformat() == (A - timedelta(days=30)).isoformat()
                and (A + timedelta(days=7)).isoformat() == "2026-08-23")
    check("RN-1a bar-side arithmetic self-check: incident [2026-03-01,06-29)=120d, "
          "activation [A-150,A-30)=[2026-03-19,07-17)=120d, union 138d, A+7d=2026-08-23",
          arith_ok)

    if RP is not None:
        try:
            arms = RP.renewal_arms()
            ok1 = (list(arms["incident"]["days"]) == exp_incident
                   and list(arms["activation"]["days"]) == exp_activation
                   and arms["contract_id"] == CONTRACT_ID
                   and arms["activation_reference_day"] == ANCHOR
                   and arms["incident_reference"] == "2026-07-29"
                   and int(arms["embargo_days"]) == 30
                   and arms["valid_through"] == "2026-08-23"
                   and arms["expiry_utc"] == "2026-08-24T00:00:00Z"
                   and sorted(set(arms["incident"]["days"])
                              | set(arms["activation"]["days"])) == exp_union)
            check("RN-1b renewal_arms() == the contract exactly (days lists, references, "
                  "embargo 30, valid_through 2026-08-23, expiry 2026-08-24T00:00:00Z)",
                  ok1, f"keys={sorted(arms.keys()) if isinstance(arms, dict) else type(arms)}")
        except Exception as e:
            check("RN-1b renewal_arms() == the contract exactly (days lists, references, "
                  "embargo 30, valid_through 2026-08-23, expiry 2026-08-24T00:00:00Z)",
                  False, f"raised {e}")
    else:
        check("RN-1b renewal_arms() == the contract exactly (days lists, references, "
              "embargo 30, valid_through 2026-08-23, expiry 2026-08-24T00:00:00Z)",
              False, "AWAITING implementation (RN-0 red)")

    # frozen executor constants the renewal reuses verbatim (lock, green both sides)
    try:
        import d2_step4b_campaign_run as CR
        pol = CR.POLICY
        import math
        vals = sorted([10.0, 3.0, 7.0, 1.0, 9.0, 5.0] * 10)   # n=60 synthetic
        k = max(0, math.ceil(pol["lower_quantile"] * len(vals)) - 1)
        ok1c = (pol["min_admitted_days"] == 60 and pol["lower_quantile"] == 0.05
                and pol["candidate_valid_days"] == 7
                and pol["quantile_rule"] == "nearest-rank-lower-tail-ceil"
                and k == 2 and vals[k] == 1.0)
        check("RN-1c frozen executor POLICY unchanged (floor 60, lower 5% nearest-rank "
              "ceil -> index 2 of 60, valid days 7) — the renewal reuses it verbatim",
              ok1c, f"policy={ {k2: pol.get(k2) for k2 in ('min_admitted_days','lower_quantile','candidate_valid_days')} }")
    except Exception as e:
        check("RN-1c frozen executor POLICY unchanged (floor 60, lower 5% nearest-rank "
              "ceil -> index 2 of 60, valid days 7) — the renewal reuses it verbatim",
              False, f"raised {e}")

    # =====================================================================
    # RN-2 — §6.2 candidate-pool-before-outcome chronology + copy-only pool
    # =====================================================================
    v2_pool_path = os.path.join(HERE, "d2_campaign_v2_candidate_pool.json")
    v2_bytes = open(v2_pool_path, "rb").read()
    check("RN-2a the v2 candidate pool bytes still hash to the pinned V2_POOL_SHA256 "
          "(static-pool source integrity)",
          hashlib.sha256(v2_bytes).hexdigest() == V2_POOL_SHA)

    if RP is not None:
        try:
            pool, pool_bytes = RP.build_renewal_candidate_pool(v2_bytes)
            v2_pool = json.loads(v2_bytes)
            # copy-only: station identities/coords/polygons/NSLC orders byte-equal
            def strip_env(p):
                return {k: v for k, v in p.items()
                        if k not in ("contract_id", "created_utc", "pool_digest",
                                     "schema", "source_pool_sha256")}
            copy_ok = (strip_env(pool) == strip_env(v2_pool)
                       and pool.get("contract_id") == CONTRACT_ID
                       and pool.get("source_pool_sha256") == V2_POOL_SHA)
            # refusals: dict input, mutated bytes, outcome-bearing field
            r1 = r2 = r3 = False
            try:
                RP.build_renewal_candidate_pool(json.loads(v2_bytes))
            except Exception:
                r1 = True
            try:
                RP.build_renewal_candidate_pool(v2_bytes + b" ")
            except Exception:
                r2 = True
            poisoned = json.loads(v2_bytes)
            poisoned["availability"] = {"istanbul_marmara": 96}
            try:
                RP.build_renewal_candidate_pool(json.dumps(poisoned).encode())
            except Exception:
                r3 = True   # (also refused by the sha pin — the outcome-field rule must
                            #  ALSO hold for any future pool rev; codex verifies wording)
            check("RN-2b renewal pool = copy-only re-envelope of the v2 pool (identities/"
                  "coords/polygons/NSLC orders equal; new contract id + source sha; "
                  "refuses dict input, mutated bytes, outcome-bearing fields)",
                  copy_ok and r1 and r2 and r3,
                  f"copy_ok={copy_ok} refusals=({r1},{r2},{r3})")
        except Exception as e:
            check("RN-2b renewal pool = copy-only re-envelope of the v2 pool (identities/"
                  "coords/polygons/NSLC orders equal; new contract id + source sha; "
                  "refuses dict input, mutated bytes, outcome-bearing fields)",
                  False, f"raised {e}")
        # plan builder refusals + chronology + target order + core blobs + stale commit
        try:
            bad = json.dumps({"contract_id": "codex-d2-campaign-v2-2026-08-10-v1"}).encode()
            refused = False
            try:
                RP.build_renewal_plan(bad)
            except Exception:
                refused = True
            has_bundle_pin = isinstance(getattr(RP, "RENEWAL_BUNDLE_SHA256", None), str) \
                and len(getattr(RP, "RENEWAL_BUNDLE_SHA256")) == 64
            check("RN-2c build_renewal_plan refuses wrong-contract/unpinned bundles and "
                  "the module carries RENEWAL_BUNDLE_SHA256 (set at the renewal "
                  "Phase-0.5-equivalent freeze)", refused and has_bundle_pin,
                  f"refused={refused} bundle_pin={has_bundle_pin}")
        except Exception as e:
            check("RN-2c build_renewal_plan refuses wrong-contract/unpinned bundles and "
                  "the module carries RENEWAL_BUNDLE_SHA256 (set at the renewal "
                  "Phase-0.5-equivalent freeze)", False, f"raised {e}")
    else:
        for nm in ("RN-2b renewal pool = copy-only re-envelope of the v2 pool (identities/"
                   "coords/polygons/NSLC orders equal; new contract id + source sha; "
                   "refuses dict input, mutated bytes, outcome-bearing fields)",
                   "RN-2c build_renewal_plan refuses wrong-contract/unpinned bundles and "
                   "the module carries RENEWAL_BUNDLE_SHA256 (set at the renewal "
                   "Phase-0.5-equivalent freeze)"):
            check(nm, False, "AWAITING implementation (RN-0 red)")

    ok2d, tail = run_suite("test_campaign_v2_phase075_registry_redkats_cayley.py")
    check("RN-2d COMPOSE: frozen v2 selection-semantics bar green (renewal reuses the "
          "derivation verbatim)", ok2d, tail)

    # =====================================================================
    # RN-3 — §6.3 repaired-shell attestation + true-overlap refusal
    # =====================================================================
    ok3a, tail = run_suite("test_d2_koeri_order_canon_redkats_cayley.py")
    check("RN-3a COMPOSE: order-canonicalization bar green on this tree (repaired shell "
          "present; enumeration-order repair in scope, O-1..O-4 NOT)", ok3a, tail)

    if RP is not None:
        try:
            core = ["monitoring/src/seismic_data.py", "monitoring/src/fault_correlation.py",
                    "monitoring/src/ensemble.py", "monitoring/src/d2_step4b_campaign_run.py",
                    "monitoring/src/d2_renewal_plan.py"]
            head = {}
            for f in core:
                out = subprocess.run(["git", "ls-tree", "HEAD", f], capture_output=True,
                                     text=True, cwd=os.path.join(HERE, "..", "..")).stdout.split()
                head[f] = out[2] if len(out) >= 3 else None
            blobs = getattr(RP, "core_blob_map", lambda: None)()
            ok3b = isinstance(blobs, dict) and all(
                blobs.get(f) == head[f] and head[f] for f in core)
            check("RN-3b core_blob_map() binds the exact D2 core blob vector to git HEAD "
                  "(seismic_data/fault_correlation/ensemble/executor/renewal-plan) — the "
                  "plan and every capsule manifest must carry these blobs",
                  ok3b, f"blobs={blobs if not isinstance(blobs, dict) else 'dict'} head_ok={all(head.values())}")
        except Exception as e:
            check("RN-3b core_blob_map() binds the exact D2 core blob vector to git HEAD "
                  "(seismic_data/fault_correlation/ensemble/executor/renewal-plan) — the "
                  "plan and every capsule manifest must carry these blobs",
                  False, f"raised {e}")
    else:
        check("RN-3b core_blob_map() binds the exact D2 core blob vector to git HEAD "
              "(seismic_data/fault_correlation/ensemble/executor/renewal-plan) — the "
              "plan and every capsule manifest must carry these blobs",
              False, "AWAITING implementation (RN-0 red)")

    # TRUE_OVERLAP_UNRULED: a genuine post-sort overlap in acquisition must HOLD with
    # this exact reason code (never dedup/merge/consume). Pinned executor seam.
    try:
        import d2_step4b_campaign_run as CR
        has_state = "TRUE_OVERLAP_UNRULED" in open(
            os.path.join(HERE, "d2_step4b_campaign_run.py"), encoding="utf-8").read()
        clf = getattr(CR, "classify_station_refusal", None)
        ok3c = False
        det3c = f"state_in_source={has_state} classify={callable(clf)}"
        if callable(clf):
            import numpy as np
            t0 = datetime(2026, 3, 2, tzinfo=timezone.utc)
            frs = [(np.zeros(30000), 100.0, t0),
                   (np.zeros(30000), 100.0, t0 + timedelta(seconds=150))]  # true overlap
            ok3c = clf(frs) == "TRUE_OVERLAP_UNRULED" and clf(
                [(np.zeros(30000), 100.0, t0),
                 (np.zeros(30000), 100.0, t0 + timedelta(seconds=400))]) != "TRUE_OVERLAP_UNRULED"
        check("RN-3c TRUE-overlap acquisition refusal carries reason code "
              "TRUE_OVERLAP_UNRULED (hold — never dedup/merge/consume) and a disjoint "
              "pair does NOT (pinned seam classify_station_refusal)", ok3c, det3c)
    except Exception as e:
        check("RN-3c TRUE-overlap acquisition refusal carries reason code "
              "TRUE_OVERLAP_UNRULED (hold — never dedup/merge/consume) and a disjoint "
              "pair does NOT (pinned seam classify_station_refusal)", False, f"raised {e}")

    check("RN-3d the stale v1-era source attestation is BANNED from renewal surfaces: "
          "the literal prefix 292b1069 may appear in NO renewal plan or capsule "
          "source_commit (enforced concretely at RN-5 mint time; asserted here on the "
          "plan builder module source as a tripwire)",
          RP is not None and STALE_SOURCE_COMMIT_PREFIX not in open(
              os.path.join(HERE, "d2_renewal_plan.py"), encoding="utf-8").read()
          if RP is not None else False,
          "AWAITING implementation (RN-0 red)" if RP is None else "")

    # =====================================================================
    # RN-4 — §6.4 fresh-root receipts + gates + honest batch (compose + deltas)
    # =====================================================================
    ok4a, tail = run_suite("test_d2_step4b_executor_redkats_cayley.py")
    check("RN-4a COMPOSE: frozen v2 executor bar green (receipts/H-gates/threshold "
          "arithmetic — the renewal runs this executor verbatim)", ok4a, tail)
    ok4b, tail = run_suite("test_d2_step4b_provider_redkats_cayley.py")
    check("RN-4b COMPOSE: frozen v2 provider bar green (provider I/O discipline)",
          ok4b, tail)

    if RP is not None:
        try:
            # COVERAGE_INFEASIBLE: a bundle carrying a carrier below the 60-day potential
            # floor in either arm must be marked infeasible and carry NO fetch entry.
            synth = getattr(RP, "mark_coverage_infeasible", None)
            ok4c = callable(synth) and synth({"incident_potential": 59,
                                             "activation_potential": 96}) is True \
                and synth({"incident_potential": 96, "activation_potential": 96}) is False
            check("RN-4c below-floor carrier (either arm < 60 potential) is "
                  "COVERAGE_INFEASIBLE — no fetch (pinned seam mark_coverage_infeasible)",
                  ok4c)
        except Exception as e:
            check("RN-4c below-floor carrier (either arm < 60 potential) is "
                  "COVERAGE_INFEASIBLE — no fetch (pinned seam mark_coverage_infeasible)",
                  False, f"raised {e}")
        try:
            # dedup re-attestation: manifest reuse entries must carry a FRESH attestation
            val = getattr(RP, "validate_reuse_entry", None)
            good = {"reuse": True, "attested_utc": "2026-08-16T02:00:00Z",
                    "provider_identity": "obj://x", "sha256": "a" * 64}
            bad = {"reuse": True, "sha256": "a" * 64}
            ok4d = callable(val) and val(good) is True and val(bad) is False
            check("RN-4d byte-dedup requires FRESH per-run re-attestation (pinned seam "
                  "validate_reuse_entry: reuse entries without live attestation refuse)",
                  ok4d)
        except Exception as e:
            check("RN-4d byte-dedup requires FRESH per-run re-attestation (pinned seam "
                  "validate_reuse_entry: reuse entries without live attestation refuse)",
                  False, f"raised {e}")
        try:
            # honest 0..3 batch form: the batch document must enumerate ALL THREE targets
            # in the outcome-blind order with terminal states; a favorable subset refuses.
            bval = getattr(RP, "validate_batch_form", None)
            full = {"contract_id": CONTRACT_ID,
                    "carriers": [{"carrier_key": k, "state": "ADMITTED_CANDIDATE"}
                                 for k in TARGET_ORDER]}
            subset = {"contract_id": CONTRACT_ID,
                      "carriers": [{"carrier_key": "istanbul_marmara",
                                    "state": "ADMITTED_CANDIDATE"}]}
            ok4e = callable(bval) and bval(full) is True and bval(subset) is False
            check("RN-4e batch form is honest 0..3: ALL THREE carriers enumerated in the "
                  "outcome-blind order with terminal states; favorable-subset batches "
                  "refuse (pinned seam validate_batch_form)", ok4e)
        except Exception as e:
            check("RN-4e batch form is honest 0..3: ALL THREE carriers enumerated in the "
                  "outcome-blind order with terminal states; favorable-subset batches "
                  "refuse (pinned seam validate_batch_form)", False, f"raised {e}")
    else:
        for nm in ("RN-4c below-floor carrier (either arm < 60 potential) is "
                   "COVERAGE_INFEASIBLE — no fetch (pinned seam mark_coverage_infeasible)",
                   "RN-4d byte-dedup requires FRESH per-run re-attestation (pinned seam "
                   "validate_reuse_entry: reuse entries without live attestation refuse)",
                   "RN-4e batch form is honest 0..3: ALL THREE carriers enumerated in the "
                   "outcome-blind order with terminal states; favorable-subset batches "
                   "refuse (pinned seam validate_batch_form)"):
            check(nm, False, "AWAITING implementation (RN-0 red)")

    # =====================================================================
    # RN-5 — §6.5 capsule/registry/stale boundary (loader lock runs NOW)
    # =====================================================================
    try:
        if "obspy" not in sys.modules:                      # hermetic obspy stub
            import types
            import numpy as _np
            from scipy import signal as _sig
            from scipy.signal import hilbert as _hilb
            _o = types.ModuleType("obspy")
            _o.Stream, _o.Trace, _o.UTCDateTime = type("S", (), {}), type("T", (), {}), object
            _f = types.ModuleType("obspy.clients.fdsn")
            _f.Client = type("C", (), {"__init__": lambda s, *a, **k: None})
            _c = types.ModuleType("obspy.clients")
            _c.fdsn = _f
            _o.clients = _c
            _sg = types.ModuleType("obspy.signal")
            _sf = types.ModuleType("obspy.signal.filter")
            _sf.envelope = lambda a: _np.abs(_hilb(a))
            _sf.bandpass = lambda d, fmin, fmax, df, corners=4, zerophase=False: d
            _sg.filter = _sf
            for nm, md in (("obspy", _o), ("obspy.clients", _c), ("obspy.clients.fdsn", _f),
                           ("obspy.signal", _sg), ("obspy.signal.filter", _sf)):
                sys.modules[nm] = md
        import fault_correlation as FC
        import seismic_data as SD
        cap = {"schema": "geospec-d2-calibration-v1", "region": "istanbul_marmara",
               "band_tag": "1-10Hz", "processing_version": SD.PROCESSING_VERSION,
               "topology_version": "t1", "threshold": 0.21,
               "calibration_window": {"start": "2026-03-19", "end": "2026-07-17"},
               "source_commit": "f" * 40,
               "input_manifest_sha256": "a" * 64, "replay_output_sha256": "b" * 64,
               "issued_utc": "2026-08-16T02:00:00Z", "valid_through": "2026-08-23"}
        capdir = tempfile.mkdtemp()
        body = json.dumps(cap, sort_keys=True).encode()
        cappath = os.path.join(capdir, "istanbul_marmara.json")
        open(cappath, "wb").write(body)
        pin = hashlib.sha256(body).hexdigest()

        def load(day):
            try:
                FC.load_calibration_capsule(
                    "istanbul_marmara", day, band_tag="1-10Hz",
                    processing_version=SD.PROCESSING_VERSION, topology_version="t1",
                    capsule_dir=capdir, expected_sha256=pin)
                return "ADMIT"
            except FC.CalibrationUnavailable as e:
                return "REFUSE:" + ";".join(getattr(e, "reasons", [str(e)]))[:60]

        r23 = load("2026-08-23")
        r24 = load("2026-08-24")
        ok5a = r23 == "ADMIT" and r24.startswith("REFUSE")
        check("RN-5a stale boundary LOCK on the existing loader: a renewal-shaped capsule "
              "(valid_through 2026-08-23) ADMITS scored day 08-23 and REFUSES 08-24 "
              "(expiry 2026-08-24T00:00:00Z)", ok5a, f"d23={r23} d24={r24}")
        mut = dict(cap)
        mut["threshold"] = 0.5
        open(cappath, "wb").write(json.dumps(mut, sort_keys=True).encode())
        ok5b = load("2026-08-20").startswith("REFUSE")
        check("RN-5b registered-digest binding LOCK: a capsule mutated under the "
              "unchanged registry pin refuses (no self-attestation)", ok5b)
    except Exception as e:
        check("RN-5a stale boundary LOCK on the existing loader: a renewal-shaped capsule "
              "(valid_through 2026-08-23) ADMITS scored day 08-23 and REFUSES 08-24 "
              "(expiry 2026-08-24T00:00:00Z)", False, f"raised {e}")
        check("RN-5b registered-digest binding LOCK: a capsule mutated under the "
              "unchanged registry pin refuses (no self-attestation)", False, f"raised {e}")

    if RP is not None:
        try:
            cval = getattr(RP, "validate_renewal_capsule", None)
            good = dict(cap)
            ok = callable(cval) and cval(good) is True
            stale = dict(cap)
            stale["source_commit"] = STALE_SOURCE_COMMIT_PREFIX + "e" * 32
            ok_stale = callable(cval) and cval(stale) is False
            badwin = dict(cap)
            badwin["calibration_window"] = {"start": "2026-03-13", "end": "2026-07-11"}
            ok_win = callable(cval) and cval(badwin) is False
            check("RN-5c validate_renewal_capsule: renewal window {2026-03-19, 2026-07-17} "
                  "end-exclusive + valid_through 2026-08-23 required; the stale 292b1069 "
                  "source attestation REFUSES; v2's old window refuses",
                  ok and ok_stale and ok_win,
                  f"good={ok} stale_refused={ok_stale} oldwin_refused={ok_win}")
        except Exception as e:
            check("RN-5c validate_renewal_capsule: renewal window {2026-03-19, 2026-07-17} "
                  "end-exclusive + valid_through 2026-08-23 required; the stale 292b1069 "
                  "source attestation REFUSES; v2's old window refuses", False, f"raised {e}")
    else:
        check("RN-5c validate_renewal_capsule: renewal window {2026-03-19, 2026-07-17} "
              "end-exclusive + valid_through 2026-08-23 required; the stale 292b1069 "
              "source attestation REFUSES; v2's old window refuses",
              False, "AWAITING implementation (RN-0 red)")


main()
print()
if FAILS:
    print(f"D2 RENEWAL RED-KAT FAILURES: {FAILS}")
    sys.exit(1)
print("ALL D2 RENEWAL RED-KATs PASS (contract arithmetic + copy-only pool + repaired-shell "
      "attestation + fresh-root gates + capsule/stale discipline; provider I/O still gated "
      "on the fresh fire-time owner go naming the contract id and A)")
