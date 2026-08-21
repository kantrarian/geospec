#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SEALED-RUN RESULT VERIFIER v2.1 (cayley) -- frozen BEFORE the fire.

v2.1 = v2 + codex second-pass residuals 2/3/5:
  R2  the result claim capsule is SCHEMA-CLOSED: families exactly
      B1A/B2A/B3A each with exactly full/loco_folds/loco_gate; result
      seal == the committed manifest's seal; codex_instrument_pass ==
      the authorization's note object; canonical manifest path; exact
      sources/checkpoint keysets; header codex-note sha == the
      authorization note sha; reservation run UUID + authorities +
      lease == the result and header; REGISTERED typing note and
      non-claims strings exact.
  R3  executed-bytes attestation: the verifier SELF-ATTESTS (its own
      disk LF bytes must equal sources.verifier_sha256) and confirms the
      IMPORTED instrument and engine modules' __file__ bytes against the
      recorded shas; in bind_git mode every source must also equal its
      HEAD blob and the engine the frozen 24b0d8f blob.
  R5  a PLANTED-POSITIVE fixture (power-lane B2A m=3 generator remapped
      onto the real registry) executes one full positive, all 35 folds,
      and the conjunctive gate through the production verify path;
      missing-fold / extra-fold / mutated-fold / wrong-fold-station /
      mutated-gate are pinned refusals.
Everything from v2 stands: verifier-local schema-closed checkpoint
loading, panel rebuild + digest match, selection closure, EVERY full and
fold row RECOMPUTED at the recorded draws, gate recomputation, and
independent typing enforcement (TYPING_PROMOTION refusal).
"""
import hashlib
import json
import os
import subprocess
import sys

import f2g_sealed_run_instrument_cayley as I

RESULT_SCHEMA = "f2g-sealed-run-result-v2.1"
EVIDENCE_SCHEMA = "f2g-sealed-run-evidence-v2.1"
CANONICAL_CKPT = "docs/f2g_sealed_run_evidence.jsonl"
AUTH_PATH = "docs/f2g_sealed_run_fire_authorization.json"
DRIVER_REL = "monitoring/src/f2g_sealed_run_driver_cayley.py"
INSTRUMENT_REL = "monitoring/src/f2g_sealed_run_instrument_cayley.py"
FAMS = ("B1A", "B2A", "B3A")
REGISTERED_TYPING_NOTE = (
    "B2A verdict-bearing under the certified power contract; B1A/B3A "
    "nonpositives are typed CANNOT_DETERMINE_NO_POWER non-answers, never "
    "'no signal'")
REGISTERED_NON_CLAIMS = (
    "no earthquake forecast, precursor, or displacement claims; "
    "Lambda_geo remains INCONCLUSIVE; this result reports the registered "
    "Phase-B family statistics on the sealed Phase-A graph series and "
    "nothing else")
RESULT_KEYS = {"schema", "generated_utc", "seal", "run_uuid",
               "reservation", "fire_authorization_sha256",
               "codex_instrument_pass", "input_manifest", "sources",
               "checkpoint", "env", "panel_sha256", "n_draws", "alpha",
               "families", "typing_note", "non_claims"}
# codex 1521 lease-receipt closure (convergence option A): the packaged
# reservation must BE the winning remote receipt, independently verifiable
RESERVATION_KEYS = {"schema", "run_uuid", "auth_sha256",
                    "manifest_sha256", "driver_sha256",
                    "instrument_sha256", "verifier_sha256",
                    "engine_sha256", "lease_ref", "lease_commit",
                    "ckpt_path", "created_utc"}
RESERVATION_SCHEMA = "f2g-sealed-run-reservation-v2"
LEASE_REF = "refs/f2g/fire-lease"
LEASE_PAYLOAD_KEYS = {"schema", "run_uuid", "auth_sha256", "host",
                      "created_utc"}
LEASE_PAYLOAD_SCHEMA = "f2g-fire-lease-v1"


def _valid_utc(value):
    """codex 1554 attached repair (verbatim): canonical driver
    %Y-%m-%dT%H:%M:%SZ round-trip validation."""
    import time as _time
    if not isinstance(value, str):
        return False
    try:
        parsed = _time.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return False
    return _time.strftime("%Y-%m-%dT%H:%M:%SZ", parsed) == value


def _verify_lease_receipt(resv, result, remote_sha, payload):
    """Pure receipt check (unit-KATable): the remote ref must equal the
    reservation's lease commit and the lease payload must bind this run.
    Returns None on pass, else a typed reason."""
    if remote_sha is None or remote_sha != resv.get("lease_commit"):
        return ("LEASE_RECEIPT: remote fire-lease ref is not the "
                "reservation's lease commit")
    if not isinstance(payload, dict) or \
            set(payload) != LEASE_PAYLOAD_KEYS:
        return "LEASE_RECEIPT: lease payload keyset"
    if payload["schema"] != LEASE_PAYLOAD_SCHEMA:
        return "LEASE_RECEIPT: lease payload schema"
    if payload["run_uuid"] != result.get("run_uuid"):
        return "LEASE_RECEIPT: lease payload run_uuid mismatch"
    if payload["auth_sha256"] != \
            result.get("fire_authorization_sha256"):
        return "LEASE_RECEIPT: lease payload auth mismatch"
    if not (isinstance(payload["host"], str)
            and payload["host"].strip()) or \
            not _valid_utc(payload["created_utc"]):
        return "LEASE_RECEIPT: lease payload host/UTC malformed"
    return None
SOURCES_KEYS = {"driver_sha256", "instrument_sha256", "verifier_sha256",
                "engine_sha256"}
STAGE_FIELDS = {
    "header": {"key", "stage", "schema", "purpose", "run_uuid",
               "codex_note_sha256", "auth_sha256", "manifest_sha256",
               "driver_sha256", "instrument_sha256", "verifier_sha256",
               "engine_sha256", "lease_commit"},
    "panel": {"key", "stage", "panel_sha256"},
    "full": {"key", "stage", "family", "result", "panel_sha256", "dt"},
    "loco": {"key", "stage", "family", "station", "result",
             "panel_sha256", "dt"},
}


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _lf(path):
    return open(path, "rb").read().replace(b"\r\n", b"\n")


def _blob(repo, ref):
    try:
        return subprocess.check_output(["git", "cat-file", "blob", ref],
                                       cwd=repo,
                                       stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None


def _load_ckpt(path, reasons):
    done = {}
    rows = [json.loads(l) for l in open(path, encoding="utf-8")
            if l.strip()]
    if not rows or rows[0].get("stage") != "header":
        reasons.append("CHECKPOINT_HEADER_MISSING")
        return None
    for i, r in enumerate(rows):
        st = r.get("stage")
        if st not in STAGE_FIELDS:
            reasons.append(f"CHECKPOINT_UNKNOWN_STAGE: row {i} {st!r}")
            return None
        if set(r) != STAGE_FIELDS[st]:
            reasons.append(f"CHECKPOINT_ROW_SCHEMA: row {i} ({st})")
            return None
        if r["key"] in done:
            reasons.append(f"CHECKPOINT_DUPLICATE_KEY: {r['key']}")
            return None
        if st == "header" and (r["key"] != "header" or i != 0):
            reasons.append("CHECKPOINT_HEADER_MISMATCH: header placement")
            return None
        if st == "panel" and r["key"] != "panel":
            reasons.append("CHECKPOINT_ROW_SCHEMA: panel key")
            return None
        if st == "full" and (r["family"] not in FAMS or
                             r["key"] != f"full|{r['family']}"):
            reasons.append(f"CHECKPOINT_ROW_SCHEMA: {r['key']}")
            return None
        if st == "loco" and (r["family"] not in FAMS or
                             r["key"] !=
                             f"loco|{r['family']}|{r['station']}"):
            reasons.append(f"CHECKPOINT_ROW_SCHEMA: {r['key']}")
            return None
        done[r["key"]] = r
    return done


def verify(repo, result, ckpt_path, expected_draws=9999, bind_git=True,
           snapshots_dir=None, manifest=None, auth=None):
    """manifest/auth: parsed committed documents. In bind_git mode they
    are loaded from HEAD (arguments ignored); fixture mode must supply
    both so the capsule-closure equalities always execute."""
    reasons = []
    if not isinstance(result, dict) or \
            result.get("schema") != RESULT_SCHEMA:
        return False, ["RESULT_SCHEMA: wrong schema"]
    missing = RESULT_KEYS - set(result)
    extra = set(result) - RESULT_KEYS
    if missing or extra:
        return False, [f"RESULT_SCHEMA: missing {sorted(missing)} "
                       f"extra {sorted(extra)}"]
    if bind_git:
        mb = _blob(repo, "HEAD:" + I.MANIFEST_OUT)
        ab = _blob(repo, "HEAD:" + AUTH_PATH)
        if mb is None or ab is None:
            return False, ["BINDING: manifest/authorization not in HEAD"]
        manifest = json.loads(mb)
        auth = json.loads(ab)
        if _sha(mb) != result["input_manifest"].get("sha256"):
            reasons.append("BINDING: input manifest")
        if _sha(ab) != result["fire_authorization_sha256"]:
            reasons.append("BINDING: fire authorization")
    if manifest is None or auth is None:
        return False, ["VERIFIER_USAGE: fixture mode requires manifest "
                       "and auth documents"]
    # ---- R2: capsule closure ----
    if result["input_manifest"].get("path") != I.MANIFEST_OUT:
        reasons.append("CAPSULE: manifest path not canonical")
    if json.dumps(result["seal"], sort_keys=True) != \
            json.dumps(manifest.get("seal"), sort_keys=True):
        reasons.append("CAPSULE: result seal != manifest seal")
    if json.dumps(result["codex_instrument_pass"], sort_keys=True) != \
            json.dumps(auth.get("codex_pass_note"), sort_keys=True):
        reasons.append("CAPSULE: codex_instrument_pass != authorization "
                       "note")
    if result["typing_note"] != REGISTERED_TYPING_NOTE:
        reasons.append("CAPSULE: typing_note not the registered text")
    if result["non_claims"] != REGISTERED_NON_CLAIMS:
        reasons.append("CAPSULE: non_claims not the registered text")
    if not isinstance(result["sources"], dict) or \
            set(result["sources"]) != SOURCES_KEYS:
        reasons.append("CAPSULE: sources keyset")
        return False, reasons
    if not isinstance(result["checkpoint"], dict) or \
            set(result["checkpoint"]) != {"path", "sha256"}:
        reasons.append("CAPSULE: checkpoint keyset")
        return False, reasons
    fam_obj = result["families"]
    if not isinstance(fam_obj, dict) or set(fam_obj) != set(FAMS):
        reasons.append("CAPSULE: families must be exactly B1A/B2A/B3A")
        return False, reasons
    for fam in FAMS:
        if not isinstance(fam_obj[fam], dict) or \
                set(fam_obj[fam]) != {"full", "loco_folds", "loco_gate"}:
            reasons.append(f"CAPSULE: family {fam} keyset")
            return False, reasons
    resv = result["reservation"]
    if not isinstance(resv, dict) or set(resv) != RESERVATION_KEYS:
        reasons.append("CAPSULE: reservation keyset not exactly the "
                       "driver's v2 keyset")
        return False, reasons
    if resv["schema"] != RESERVATION_SCHEMA or \
            resv["lease_ref"] != LEASE_REF or \
            not _valid_utc(resv["created_utc"]) or \
            resv["run_uuid"] != result["run_uuid"] or \
            resv["auth_sha256"] != \
            result["fire_authorization_sha256"] or \
            resv["manifest_sha256"] != \
            result["input_manifest"].get("sha256") or \
            resv["driver_sha256"] != \
            result["sources"].get("driver_sha256") or \
            resv["instrument_sha256"] != \
            result["sources"].get("instrument_sha256") or \
            resv["verifier_sha256"] != \
            result["sources"].get("verifier_sha256") or \
            resv["engine_sha256"] != \
            result["sources"].get("engine_sha256") or \
            resv["ckpt_path"] != result["checkpoint"].get("path"):
        reasons.append("CAPSULE: reservation does not bind the result's "
                       "authorities (schema/ref/engine/UTC closed)")
    if bind_git:
        # the packaged reservation must BE the winning remote receipt
        try:
            out = subprocess.check_output(
                ["git", "ls-remote", "origin", LEASE_REF],
                cwd=repo).decode().strip()
            remote_sha = out.split()[0] if out else None
            payload = None
            if remote_sha:
                subprocess.run(["git", "fetch", "-q", "origin",
                                LEASE_REF], cwd=repo,
                               capture_output=True)
                pb = _blob(repo, f"{remote_sha}:lease.json")
                payload = json.loads(pb) if pb else None
            bad = _verify_lease_receipt(resv, result, remote_sha, payload)
            if bad:
                reasons.append(bad)
        except Exception as exc:
            reasons.append(f"LEASE_RECEIPT: remote unverifiable ({exc})")
    if result["n_draws"] != expected_draws:
        reasons.append(f"WRONG_DRAWS: {result['n_draws']} != "
                       f"{expected_draws}")
    envd = result["env"]
    if not isinstance(envd, dict) or \
            set(envd) != {"python", "numpy", "platform"} or \
            not all(isinstance(envd[k], str) and envd[k] for k in envd):
        reasons.append("RESULT_SCHEMA: env")
    # ---- R3: executed-bytes attestation ----
    if _sha(_lf(os.path.abspath(__file__))) != \
            result["sources"].get("verifier_sha256"):
        reasons.append("SOURCE_UNATTESTED: verifier self-attestation")
    if _sha(_lf(I.__file__)) != \
            result["sources"].get("instrument_sha256"):
        reasons.append("SOURCE_UNATTESTED: imported instrument bytes")
    if bind_git:
        for rel, key in ((DRIVER_REL, "driver_sha256"),
                         (INSTRUMENT_REL, "instrument_sha256"),
                         (I.ENGINE_PATH, "engine_sha256")):
            b = _blob(repo, "HEAD:" + rel)
            if b is None or _sha(b) != result["sources"].get(key):
                reasons.append(f"BINDING: {key}")
        vb = _blob(repo, "HEAD:" +
                   "monitoring/src/f2g_sealed_run_result_verifier_"
                   "cayley.py")
        if vb is None or _sha(vb) != \
                result["sources"].get("verifier_sha256"):
            reasons.append("BINDING: verifier_sha256")
        fz = _blob(repo, f"{I.ENGINE_COMMIT}:{I.ENGINE_PATH}")
        if fz is None or _sha(fz) != \
                result["sources"].get("engine_sha256"):
            reasons.append("BINDING: engine != frozen 24b0d8f blob")
        if result["checkpoint"].get("path") != CANONICAL_CKPT:
            reasons.append("BINDING: checkpoint path not canonical")
    if not os.path.exists(ckpt_path) or \
            _sha(_lf(ckpt_path)) != result["checkpoint"].get("sha256"):
        reasons.append("BINDING: checkpoint sha")
        return False, reasons
    done = _load_ckpt(ckpt_path, reasons)
    if done is None:
        return False, reasons
    hdr = done["header"]
    if hdr.get("schema") != EVIDENCE_SCHEMA or \
            hdr.get("purpose") != "sealed-run-production" or \
            hdr.get("run_uuid") != result["run_uuid"] or \
            hdr.get("codex_note_sha256") != \
            (auth.get("codex_pass_note") or {}).get("blob_sha256") or \
            hdr.get("auth_sha256") != \
            result["fire_authorization_sha256"] or \
            hdr.get("manifest_sha256") != \
            result["input_manifest"].get("sha256") or \
            hdr.get("driver_sha256") != \
            result["sources"].get("driver_sha256") or \
            hdr.get("instrument_sha256") != \
            result["sources"].get("instrument_sha256") or \
            hdr.get("verifier_sha256") != \
            result["sources"].get("verifier_sha256") or \
            hdr.get("engine_sha256") != \
            result["sources"].get("engine_sha256") or \
            hdr.get("lease_commit") != resv.get("lease_commit"):
        reasons.append("CHECKPOINT_HEADER_MISMATCH: header does not bind "
                       "the result's authorities")
        return False, reasons
    # ---- panel rebuild ----
    sys.path.insert(0, os.path.join(repo, "monitoring", "src"))
    os.chdir(repo)
    import d2_f2g_phase_b_stats as E
    if _sha(_lf(E.__file__)) != result["sources"].get("engine_sha256"):
        reasons.append("SOURCE_UNATTESTED: imported engine bytes")
        return False, reasons
    if result["alpha"] != E.ALPHA_FAMILY:
        reasons.append("WRONG_ALPHA")
    sdir = snapshots_dir or os.path.join(repo, I.ARTIFACT_ROOT,
                                         "snapshots")
    panel = I.build_panel(repo, sdir,
                          allow_real=snapshots_dir is None)
    pdig = I.panel_digest(panel)
    if pdig != result["panel_sha256"] or \
            done.get("panel", {}).get("panel_sha256") != pdig:
        reasons.append("PANEL_MISMATCH: rebuilt panel digest differs")
        return False, reasons
    for k, r in done.items():
        if r["stage"] in ("full", "loco") and r["panel_sha256"] != pdig:
            reasons.append(f"PANEL_MISMATCH: row {k}")
            return False, reasons
    # ---- selection closure + recomputation ----
    fulls = {r["family"]: r for r in done.values()
             if r["stage"] == "full"}
    if set(fulls) != set(FAMS):
        reasons.append("CLOSURE: full rows must be exactly one per family")
        return False, reasons
    stations = I.all_stations(repo)
    alpha = E.ALPHA_FAMILY
    fams_fn = {"B1A": E.b1a_family_cal, "B2A": E.b2a_family_cal,
               "B3A": E.b3a_family_cal}
    b2a_blob = _blob(repo, f"{I.RESULTS_COMMIT}:{I.B2A_RESULTS_PATH}")
    b2a_res = json.loads(b2a_blob)
    if b2a_res.get("terminal_type") != "CERTIFIED":
        reasons.append("CONTRACT: pinned B2A results not CERTIFIED")
    contracts = {"B1A": None, "B3A": None,
                 "B2A": {"certified": True,
                         "results_commit": I.RESULTS_COMMIT,
                         "results_path": I.B2A_RESULTS_PATH,
                         "results_blob_sha256": _sha(b2a_blob),
                         "certified_points":
                             [c["point"] for c in
                              b2a_res["certified_points"]],
                         "lb95": b2a_res["certified_points"][0]["lb95"]}}
    for fam in FAMS:
        locos = {r["station"]: r for r in done.values()
                 if r["stage"] == "loco" and r["family"] == fam}
        full = fulls[fam]["result"]
        p = full.get("p_value")
        positive = p is not None and p <= alpha
        if positive and sorted(locos) != sorted(stations):
            reasons.append(f"CLOSURE: {fam} positive but LOCO folds != "
                           "the 35-station registry")
            return False, reasons
        if not positive and locos:
            reasons.append(f"CLOSURE: {fam} nonpositive but LOCO rows "
                           "present")
            return False, reasons
        re_full = fams_fn[fam](panel, doc_sha256=I.AMENDMENT2_SHA,
                               n_draws=result["n_draws"],
                               power_contract=contracts[fam])
        if json.dumps(re_full, sort_keys=True) != \
                json.dumps(full, sort_keys=True):
            reasons.append(f"RESULT_NOT_RECOMPUTED: {fam} full row")
        rf = result["families"][fam]
        if json.dumps(rf.get("full"), sort_keys=True) != \
                json.dumps(re_full, sort_keys=True):
            reasons.append(f"RESULT_NOT_RECOMPUTED: {fam} result.full")
        v = (rf.get("full") or {}).get("verdict")
        if fam in ("B1A", "B3A") and not positive and \
                v != "CANNOT_DETERMINE_NO_POWER":
            reasons.append(f"TYPING_PROMOTION: {fam} nonpositive verdict "
                           f"{v!r}")
        if fam == "B2A" and not positive and v != "NEGATIVE":
            reasons.append(f"TYPING: B2A nonpositive must be NEGATIVE "
                           f"(certified contract), got {v!r}")
        if positive:
            fold_list = []
            for st in stations:
                re_fold = fams_fn[fam](I.drop_station(panel, st),
                                       doc_sha256=I.AMENDMENT2_SHA,
                                       n_draws=result["n_draws"],
                                       power_contract=contracts[fam],
                                       fold=f"loco:{st}")
                if json.dumps(re_fold, sort_keys=True) != \
                        json.dumps(locos[st]["result"], sort_keys=True):
                    reasons.append(f"RESULT_NOT_RECOMPUTED: {fam} "
                                   f"loco:{st}")
                fold_list.append({"station": st,
                                  "p_value": re_fold.get("p_value")})
            gate = E.loco_gate({"p_value": p}, fold_list, alpha)
            if json.dumps(rf.get("loco_gate"), sort_keys=True) != \
                    json.dumps(gate, sort_keys=True):
                reasons.append(f"RESULT_NOT_RECOMPUTED: {fam} loco_gate")
            if json.dumps(rf.get("loco_folds"), sort_keys=True) != \
                    json.dumps(fold_list, sort_keys=True):
                reasons.append(f"RESULT_NOT_RECOMPUTED: {fam} loco_folds")
        else:
            if rf.get("loco_folds") is not None or \
                    rf.get("loco_gate") is not None:
                reasons.append(f"CLOSURE: {fam} result carries folds/gate "
                               "for a nonpositive")
    return (not reasons), reasons


# ---------------------- fixture self-test ----------------------

def _fixture_docs(repo, ver_sha, ins_sha, eng_sha):
    """Fixture manifest+auth documents carrying REAL attestable source
    shas (so self/imported attestation always executes) and placeholder
    binding shas for the git-only surfaces."""
    manifest = {"seal": {"kat-seal": True}}
    auth = {"schema": "f2g-sealed-run-fire-authorization-v2",
            "codex_pass_note": {"path": "kat-note.md",
                                "blob_sha256": "n" * 64,
                                "ref": "kat-ref"},
            "driver_blob_sha256": "d" * 64,
            "instrument_blob_sha256": ins_sha,
            "verifier_blob_sha256": ver_sha,
            "manifest_sha256": "m" * 64,
            "seal_quote_sha256": "s" * 64}
    return manifest, auth


def _fixture_run(repo, scratch, tree_maker, tag, draws):
    """Complete synthetic sealed-run fixture via the driver's run core."""
    import f2g_sealed_run_driver_cayley as D
    sys.path.insert(0, os.path.join(repo, "monitoring", "src"))
    os.chdir(repo)
    import d2_f2g_phase_b_stats as E
    ver_sha = _sha(_lf(os.path.abspath(__file__)))
    ins_sha = _sha(_lf(I.__file__))
    eng_sha = _sha(_lf(E.__file__))
    manifest, auth = _fixture_docs(repo, ver_sha, ins_sha, eng_sha)
    tree = tree_maker(repo, os.path.join(scratch, f"srv_{tag}_tree"))
    panel = I.build_panel(repo, tree)
    pdig = I.panel_digest(panel)
    ck = os.path.join(scratch, f"srv_{tag}_ckpt.jsonl")
    if os.path.exists(ck):
        os.unlink(ck)
    hdr = {"key": "header", "stage": "header", "schema": EVIDENCE_SCHEMA,
           "purpose": "sealed-run-production", "run_uuid": "f" * 32,
           "codex_note_sha256": "n" * 64, "auth_sha256": "a" * 64,
           "manifest_sha256": "m" * 64, "driver_sha256": "d" * 64,
           "instrument_sha256": ins_sha, "verifier_sha256": ver_sha,
           "engine_sha256": eng_sha, "lease_commit": "l" * 40}
    D.emit(ck, hdr)
    D.emit(ck, {"key": "panel", "stage": "panel", "panel_sha256": pdig})
    done = {"header": hdr, "panel": {"panel_sha256": pdig}}
    b2a_blob = _blob(repo, f"{I.RESULTS_COMMIT}:{I.B2A_RESULTS_PATH}")
    b2a_res = json.loads(b2a_blob)
    contracts = {"B1A": None, "B3A": None,
                 "B2A": {"certified": True,
                         "results_commit": I.RESULTS_COMMIT,
                         "results_path": I.B2A_RESULTS_PATH,
                         "results_blob_sha256": _sha(b2a_blob),
                         "certified_points":
                             [c["point"] for c in
                              b2a_res["certified_points"]],
                         "lb95": b2a_res["certified_points"][0]["lb95"]}}
    results = D.run_families(E, panel, pdig, contracts, draws, ck,
                             I.all_stations(repo), done)
    res = {"schema": RESULT_SCHEMA, "generated_utc": "KAT",
           "seal": manifest["seal"], "run_uuid": "f" * 32,
           "reservation": {"schema": "f2g-sealed-run-reservation-v2",
                           "run_uuid": "f" * 32,
                           "auth_sha256": "a" * 64,
                           "manifest_sha256": "m" * 64,
                           "driver_sha256": "d" * 64,
                           "instrument_sha256": ins_sha,
                           "verifier_sha256": ver_sha,
                           "engine_sha256": eng_sha,
                           "lease_ref": LEASE_REF,
                           "lease_commit": "l" * 40,
                           "ckpt_path": CANONICAL_CKPT,
                           "created_utc": "2026-01-01T00:00:00Z"},
           "fire_authorization_sha256": "a" * 64,
           "codex_instrument_pass": auth["codex_pass_note"],
           "input_manifest": {"path": I.MANIFEST_OUT,
                              "sha256": "m" * 64},
           "sources": {"driver_sha256": "d" * 64,
                       "instrument_sha256": ins_sha,
                       "verifier_sha256": ver_sha,
                       "engine_sha256": eng_sha},
           "checkpoint": {"path": CANONICAL_CKPT,
                          "sha256": _sha(_lf(ck))},
           "env": {"python": "kat", "numpy": "kat", "platform": "kat"},
           "panel_sha256": pdig, "n_draws": draws,
           "alpha": E.ALPHA_FAMILY, "families": results,
           "typing_note": REGISTERED_TYPING_NOTE,
           "non_claims": REGISTERED_NON_CLAIMS}
    return tree, ck, res, manifest, auth


def self_test(repo, scratch):
    results = {}
    ok_all = True

    def rec(name, ok, reasons, expect):
        refused = (not ok) and any(expect in str(r) for r in reasons)
        results[name] = (f"REFUSED ({expect})" if refused else
                         ("DEFECT -- ACCEPTED" if ok else
                          "DEFECT -- wrong reason: "
                          + "; ".join(map(str, reasons[:3]))))
        return refused

    import copy
    # ---------- nonpositive fixture ----------
    tree, ck, res, man, auth = _fixture_run(
        repo, scratch, I.make_synthetic_tree, "null", 49)
    ok, r = verify(repo, res, ck, expected_draws=49, bind_git=False,
                   snapshots_dir=tree, manifest=man, auth=auth)
    results["positive-fixture-null"] = ("PASS" if ok else "DEFECT: "
                                        + "; ".join(map(str, r[:4])))
    ok_all &= ok

    def dv(mut_res=None, mut_ck=None, expected_draws=49, base=None):
        b = base or (tree, ck, res)
        t_, c_, r_ = b
        r2 = copy.deepcopy(r_)
        ck2 = c_
        if mut_ck is not None:
            ck2 = c_ + ".mut"
            rows = [json.loads(l) for l in open(c_, encoding="utf-8")
                    if l.strip()]
            rows = mut_ck(rows)
            with open(ck2, "w", encoding="utf-8", newline="\n") as f:
                for row in rows:
                    f.write(json.dumps(row, sort_keys=True) + "\n")
            r2["checkpoint"]["sha256"] = _sha(_lf(ck2))
        if mut_res is not None:
            mut_res(r2)
        return verify(repo, r2, ck2, expected_draws=expected_draws,
                      bind_git=False, snapshots_dir=t_, manifest=man,
                      auth=auth)

    # R2 combined capsule mutation (codex's exact reproduction)
    def combo(x):
        x["seal"] = {"forged": True}
        x["codex_instrument_pass"] = {"forged": True}
        x["typing_note"] = "whatever"
        x["non_claims"] = "whatever"
        x["families"]["EXTRA"] = {"full": {"p_value": 0.0001,
                                           "verdict": "POSITIVE"},
                                  "loco_folds": None, "loco_gate": None}
    ok, r = dv(mut_res=combo)
    ok_all &= rec("combined-capsule-mutation", ok, r, "CAPSULE")
    ok, r = dv(mut_res=lambda x: x.update(seal={"forged": True}))
    ok_all &= rec("seal-mutation", ok, r, "CAPSULE: result seal")
    ok, r = dv(mut_res=lambda x: x.update(
        codex_instrument_pass={"forged": True}))
    ok_all &= rec("codex-pass-mutation", ok, r,
                  "CAPSULE: codex_instrument_pass")
    ok, r = dv(mut_res=lambda x: x.update(typing_note="promoted"))
    ok_all &= rec("typing-note-mutation", ok, r, "CAPSULE: typing_note")
    ok, r = dv(mut_res=lambda x: x.update(non_claims="claims!"))
    ok_all &= rec("non-claims-mutation", ok, r, "CAPSULE: non_claims")
    ok, r = dv(mut_res=lambda x: x["families"].update(
        EXTRA={"full": {}, "loco_folds": None, "loco_gate": None}))
    ok_all &= rec("extra-family", ok, r, "CAPSULE: families")
    ok, r = dv(mut_res=lambda x: x["reservation"].update(
        run_uuid="0" * 32))
    ok_all &= rec("reservation-mutation", ok, r, "CAPSULE: reservation")
    # codex 1521 exact reproduction: forged lease_ref + extra field
    ok, r = dv(mut_res=lambda x: x["reservation"].update(
        lease_ref="refs/f2g/forged", unexpected="accepted-extra-field"))
    ok_all &= rec("lease-ref-and-extra-reservation-field", ok, r,
                  "CAPSULE: reservation keyset")
    ok, r = dv(mut_res=lambda x: x["reservation"].update(
        lease_ref="refs/f2g/forged"))
    ok_all &= rec("wrong-lease-ref", ok, r, "CAPSULE: reservation")
    ok, r = dv(mut_res=lambda x: x["reservation"].update(
        schema="f2g-sealed-run-reservation-v1"))
    ok_all &= rec("reservation-schema-downgrade", ok, r,
                  "CAPSULE: reservation")
    ok, r = dv(mut_res=lambda x: x["reservation"].update(
        created_utc="not-a-utc-timestamp"))
    ok_all &= rec("reservation-malformed-utc", ok, r,
                  "CAPSULE: reservation")
    # lease-receipt UNIT refusals (the bind_git remote surface, pure fn)
    good_payload = {"schema": LEASE_PAYLOAD_SCHEMA,
                    "run_uuid": res["run_uuid"],
                    "auth_sha256": res["fire_authorization_sha256"],
                    "host": "kat-host", "created_utc": "2026-01-01T00:00:00Z"}
    unit = [
        ("lease-remote-missing", None, dict(good_payload)),
        ("lease-remote-wrong-commit", "0" * 40, dict(good_payload)),
        ("lease-payload-extra-key",
         res["reservation"]["lease_commit"],
         dict(good_payload, extra=1)),
        ("lease-payload-wrong-run",
         res["reservation"]["lease_commit"],
         dict(good_payload, run_uuid="0" * 32)),
        ("lease-payload-wrong-auth",
         res["reservation"]["lease_commit"],
         dict(good_payload, auth_sha256="0" * 64)),
        # codex 1554 fix: canonical UTC + nonblank host, not truthiness
        ("lease-payload-blank-host",
         res["reservation"]["lease_commit"],
         dict(good_payload, host="   ")),
        ("lease-payload-malformed-utc",
         res["reservation"]["lease_commit"],
         dict(good_payload, created_utc="not-a-utc-timestamp")),
    ]
    for name, rsha, pay in unit:
        bad = _verify_lease_receipt(res["reservation"], res, rsha, pay)
        refused = bad is not None and "LEASE_RECEIPT" in bad
        results[name] = (f"REFUSED ({bad})" if refused
                         else "DEFECT -- ACCEPTED")
        ok_all &= refused
    assert _verify_lease_receipt(res["reservation"], res,
                                 res["reservation"]["lease_commit"],
                                 good_payload) is None
    results["lease-receipt-positive"] = "PASS (unit)"
    # R3 attestation
    ok, r = dv(mut_res=lambda x: x["sources"].update(
        verifier_sha256="0" * 64))
    ok_all &= rec("wrong-verifier-sha", ok, r, "SOURCE_UNATTESTED")
    ok, r = dv(mut_res=lambda x: (x["sources"].update(
        engine_sha256="0" * 64), x["reservation"].update(
        engine_sha256="0" * 64)))
    ok_all &= rec("wrong-engine-sha", ok, r,
                  "CHECKPOINT_HEADER_MISMATCH")
    # v2 negatives retained
    ok, r = dv(mut_res=lambda x: x["families"]["B1A"]["full"].update(
        verdict="NEGATIVE"))
    ok_all &= rec("typing-promotion", ok, r, "RESULT_NOT_RECOMPUTED")
    ok, r = dv(expected_draws=9999)
    ok_all &= rec("wrong-draws", ok, r, "WRONG_DRAWS")
    ok, r = dv(mut_ck=lambda rows: rows + [dict(rows[-1],
                                                key="full|B2A")])
    ok_all &= rec("duplicate-key", ok, r, "CHECKPOINT_DUPLICATE_KEY")
    ok, r = dv(mut_ck=lambda rows: [dict(rows[0], run_uuid="0" * 32)]
               + rows[1:])
    ok_all &= rec("header-mismatch", ok, r, "CHECKPOINT_HEADER_MISMATCH")
    r2 = copy.deepcopy(res)
    r2["checkpoint"]["sha256"] = "0" * 64
    ok, r = verify(repo, r2, ck, expected_draws=49, bind_git=False,
                   snapshots_dir=tree, manifest=man, auth=auth)
    ok_all &= rec("ckpt-bytes-changed", ok, r, "BINDING: checkpoint sha")
    # ---------- R5: planted-positive fixture (full LOCO branch) ----------
    ptree, pck, pres, man, auth = _fixture_run(
        repo, scratch, I.make_positive_tree, "positive", 99)
    b2a_p = pres["families"]["B2A"]["full"].get("p_value")
    gate = pres["families"]["B2A"].get("loco_gate")
    assert b2a_p is not None and b2a_p <= pres["alpha"] and gate, \
        ("planted positive did not go positive", b2a_p)
    ok, r = verify(repo, pres, pck, expected_draws=99, bind_git=False,
                   snapshots_dir=ptree, manifest=man, auth=auth)
    results["positive-fixture-loco"] = (
        f"PASS (B2A full p={b2a_p}, 35 folds + gate recomputed through "
        f"the production path; gate pass={gate.get('pass')})" if ok else
        "DEFECT: " + "; ".join(map(str, r[:4])))
    ok_all &= ok
    pbase = (ptree, pck, pres)

    def pv(mut_res=None, mut_ck=None):
        return dv(mut_res=mut_res, mut_ck=mut_ck, expected_draws=99,
                  base=pbase)

    ok, r = pv(mut_ck=lambda rows: [x for x in rows
                                    if x.get("key") !=
                                    f"loco|B2A|{sorted(I.all_stations(repo))[0]}"])
    ok_all &= rec("missing-fold", ok, r, "CLOSURE")
    ok, r = pv(mut_ck=lambda rows: rows + [{
        "key": "loco|B2A|XX.FAKE", "stage": "loco", "family": "B2A",
        "station": "XX.FAKE", "result": {"p_value": 0.01},
        "panel_sha256": pres["panel_sha256"], "dt": 0.0}])
    ok_all &= rec("extra-fold", ok, r, "CLOSURE")

    def mut_fold(rows):
        for row in rows:
            if row.get("stage") == "loco" and row["family"] == "B2A":
                row["result"] = dict(row["result"], p_value=0.9)
                break
        return rows
    ok, r = pv(mut_ck=mut_fold)
    ok_all &= rec("mutated-fold", ok, r, "RESULT_NOT_RECOMPUTED")

    def wrong_station(rows):
        sts = [row for row in rows if row.get("stage") == "loco"]
        a, b = sts[0], sts[1]
        a_st, b_st = a["station"], b["station"]
        a["station"], a["key"] = b_st, f"loco|B2A|{b_st}"
        b["station"], b["key"] = a_st, f"loco|B2A|{a_st}"
        return rows
    ok, r = pv(mut_ck=wrong_station)
    ok_all &= rec("wrong-fold-station", ok, r, "RESULT_NOT_RECOMPUTED")
    ok, r = pv(mut_res=lambda x: x["families"]["B2A"]["loco_gate"]
               .update(pass_=True) or x["families"]["B2A"]["loco_gate"]
               .update(n_pass=99))
    ok_all &= rec("mutated-gate", ok, r, "RESULT_NOT_RECOMPUTED")
    return ok_all, results


if __name__ == "__main__":
    repo = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
    if len(sys.argv) > 2 and sys.argv[2] == "--self-test":
        ok, results = self_test(repo, sys.argv[3])
        print(json.dumps(results, indent=1))
        sys.exit(0 if ok else 2)
    res = json.loads(open(os.path.join(repo, "docs",
                                       "f2g_sealed_run_result.json"),
                          encoding="utf-8").read())
    ok, reasons = verify(repo, res, os.path.join(repo, CANONICAL_CKPT),
                         expected_draws=9999, bind_git=True)
    print("PASS" if ok else "REFUSE: " + "; ".join(map(str, reasons[:8])))
    sys.exit(0 if ok else 2)
