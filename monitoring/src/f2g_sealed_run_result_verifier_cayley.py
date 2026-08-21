#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SEALED-RUN RESULT VERIFIER (cayley) -- frozen BEFORE the fire (codex
pre-fire fix 5).

verify(repo, result, ckpt_path, expected_draws, bind_git, snapshots_dir):
independently validates the sealed result artifact against its evidence
checkpoint: exact result schema; git bindings (manifest / authorization /
driver / instrument / engine blob shas, canonical checkpoint path + LF
sha, reservation); VERIFIER-LOCAL schema-closed checkpoint loading (exact
per-stage field sets, no duplicates, no unknown stages, header == the
result's binding tuple); panel REBUILT from the bound inputs and digest-
matched; evidence selection closure (exactly one full row per family;
LOCO rows present iff the full-data p passes alpha, station set exactly
the 35-fold registry); EVERY full and fold row RECOMPUTED through the
attested engine at the recorded draw count and compared field-by-field;
the LOCO gate recomputed; and the frozen typing rule enforced
independently (a B1A/B3A nonpositive promoted past
CANNOT_DETERMINE_NO_POWER refuses TYPING_PROMOTION -- certified-contract
verdicts only where the contract derives from the pinned results blob).
self_test runs the negative matrix on synthetic-tree fixtures (49-draw
fixture runs; production verification always expected_draws=9999 with
bind_git=True). Positive fixture is fold-free (synthetic nulls are
nonpositive); fold-path code is exercised by the closure negatives --
DISCLOSED.
"""
import hashlib
import json
import os
import subprocess
import sys

import f2g_sealed_run_instrument_cayley as I

RESULT_SCHEMA = "f2g-sealed-run-result-v2"
EVIDENCE_SCHEMA = "f2g-sealed-run-evidence-v2"
CANONICAL_CKPT = "docs/f2g_sealed_run_evidence.jsonl"
AUTH_PATH = "docs/f2g_sealed_run_fire_authorization.json"
RESERVATION_PATH = "docs/f2g_sealed_run_reservation.json"
DRIVER_REL = "monitoring/src/f2g_sealed_run_driver_cayley.py"
INSTRUMENT_REL = "monitoring/src/f2g_sealed_run_instrument_cayley.py"
FAMS = ("B1A", "B2A", "B3A")
RESULT_KEYS = {"schema", "generated_utc", "seal", "run_uuid",
               "reservation", "fire_authorization_sha256",
               "codex_instrument_pass", "input_manifest", "sources",
               "checkpoint", "env", "panel_sha256", "n_draws", "alpha",
               "families", "typing_note", "non_claims"}
STAGE_FIELDS = {
    "header": {"key", "stage", "schema", "purpose", "run_uuid",
               "codex_note_sha256", "auth_sha256", "manifest_sha256",
               "driver_sha256", "instrument_sha256", "engine_sha256"},
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
           snapshots_dir=None):
    reasons = []
    if not isinstance(result, dict) or \
            result.get("schema") != RESULT_SCHEMA:
        return False, ["RESULT_SCHEMA: wrong schema"]
    missing = RESULT_KEYS - set(result)
    extra = set(result) - RESULT_KEYS
    if missing or extra:
        return False, [f"RESULT_SCHEMA: missing {sorted(missing)} "
                       f"extra {sorted(extra)}"]
    if result["n_draws"] != expected_draws:
        reasons.append(f"WRONG_DRAWS: {result['n_draws']} != "
                       f"{expected_draws}")
    envd = result["env"]
    if not isinstance(envd, dict) or \
            set(envd) != {"python", "numpy", "platform"} or \
            not all(isinstance(envd[k], str) and envd[k] for k in envd):
        reasons.append("RESULT_SCHEMA: env")
    # ---- git bindings (production) ----
    if bind_git:
        man = _blob(repo, "HEAD:" + I.MANIFEST_OUT)
        if man is None or _sha(man) != \
                result["input_manifest"].get("sha256"):
            reasons.append("BINDING: input manifest")
        auth = _blob(repo, "HEAD:" + AUTH_PATH)
        if auth is None or _sha(auth) != \
                result["fire_authorization_sha256"]:
            reasons.append("BINDING: fire authorization")
        for rel, key in ((DRIVER_REL, "driver_sha256"),
                         (INSTRUMENT_REL, "instrument_sha256"),
                         (I.ENGINE_PATH, "engine_sha256")):
            b = _blob(repo, "HEAD:" + rel)
            if b is None or _sha(b) != result["sources"].get(key):
                reasons.append(f"BINDING: {key}")
        if result["checkpoint"].get("path") != CANONICAL_CKPT:
            reasons.append("BINDING: checkpoint path not canonical")
        rp = os.path.join(repo, RESERVATION_PATH)
        if not os.path.exists(rp) or json.loads(
                open(rp, encoding="utf-8").read()) != \
                result["reservation"]:
            reasons.append("BINDING: reservation")
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
            hdr.get("auth_sha256") != \
            result["fire_authorization_sha256"] or \
            hdr.get("manifest_sha256") != \
            result["input_manifest"].get("sha256") or \
            hdr.get("driver_sha256") != \
            result["sources"].get("driver_sha256") or \
            hdr.get("instrument_sha256") != \
            result["sources"].get("instrument_sha256") or \
            hdr.get("engine_sha256") != \
            result["sources"].get("engine_sha256"):
        reasons.append("CHECKPOINT_HEADER_MISMATCH: header does not bind "
                       "the result's authorities")
        return False, reasons
    # ---- panel rebuild ----
    sys.path.insert(0, os.path.join(repo, "monitoring", "src"))
    os.chdir(repo)
    import d2_f2g_phase_b_stats as E
    if result["alpha"] != E.ALPHA_FAMILY:
        reasons.append("WRONG_ALPHA")
    sdir = snapshots_dir or os.path.join(repo, I.ARTIFACT_ROOT,
                                         "snapshots")
    panel = I.build_panel(repo, sdir, allow_real=True)
    pdig = I.panel_digest(panel)
    if pdig != result["panel_sha256"] or \
            done.get("panel", {}).get("panel_sha256") != pdig:
        reasons.append("PANEL_MISMATCH: rebuilt panel digest differs")
        return False, reasons
    for k, r in done.items():
        if r["stage"] in ("full", "loco") and r["panel_sha256"] != pdig:
            reasons.append(f"PANEL_MISMATCH: row {k}")
            return False, reasons
    # ---- selection closure ----
    fulls = {r["family"]: r for r in done.values()
             if r["stage"] == "full"}
    if set(fulls) != set(FAMS):
        reasons.append("CLOSURE: full rows must be exactly one per family")
        return False, reasons
    stations = I.all_stations(repo)
    alpha = E.ALPHA_FAMILY
    fams_fn = {"B1A": E.b1a_family_cal, "B2A": E.b2a_family_cal,
               "B3A": E.b3a_family_cal}
    # contracts derive from the pinned results blob, never the result echo
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
        full_row = fulls[fam]
        full = full_row["result"]
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
        # ---- recompute the full row ----
        re_full = fams_fn[fam](panel, doc_sha256=I.AMENDMENT2_SHA,
                               n_draws=result["n_draws"],
                               power_contract=contracts[fam])
        if json.dumps(re_full, sort_keys=True) != \
                json.dumps(full, sort_keys=True):
            reasons.append(f"RESULT_NOT_RECOMPUTED: {fam} full row")
        rf = result["families"].get(fam, {})
        if json.dumps(rf.get("full"), sort_keys=True) != \
                json.dumps(re_full, sort_keys=True):
            reasons.append(f"RESULT_NOT_RECOMPUTED: {fam} result.full")
        # ---- typing enforcement (independent of recompute) ----
        v = (rf.get("full") or {}).get("verdict")
        if fam in ("B1A", "B3A") and not positive and \
                v != "CANNOT_DETERMINE_NO_POWER":
            reasons.append(f"TYPING_PROMOTION: {fam} nonpositive verdict "
                           f"{v!r}")
        if fam == "B2A" and not positive and v != "NEGATIVE":
            reasons.append(f"TYPING: B2A nonpositive must be NEGATIVE "
                           f"(certified contract), got {v!r}")
        # ---- folds + gate ----
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

def _fixture_run(repo, scratch, draws=49):
    """Build a complete synthetic sealed-run fixture (tree -> panel ->
    49-draw run -> ckpt + result) using the driver's shared run core."""
    import f2g_sealed_run_driver_cayley as D
    sys.path.insert(0, os.path.join(repo, "monitoring", "src"))
    os.chdir(repo)
    import d2_f2g_phase_b_stats as E
    tree = I.make_synthetic_tree(repo, os.path.join(scratch,
                                                    "srv_fixture_tree"))
    panel = I.build_panel(repo, tree)
    pdig = I.panel_digest(panel)
    ck = os.path.join(scratch, "srv_fixture_ckpt.jsonl")
    if os.path.exists(ck):
        os.unlink(ck)
    hdr = {"key": "header", "stage": "header", "schema": EVIDENCE_SCHEMA,
           "purpose": "sealed-run-production", "run_uuid": "f" * 32,
           "codex_note_sha256": "n" * 64, "auth_sha256": "a" * 64,
           "manifest_sha256": "m" * 64, "driver_sha256": "d" * 64,
           "instrument_sha256": "i" * 64, "engine_sha256": "e" * 64}
    D.emit(ck, hdr)
    D.emit(ck, {"key": "panel", "stage": "panel", "panel_sha256": pdig})
    done = {"header": hdr, "panel": {"panel_sha256": pdig}}
    contracts = {"B1A": None, "B3A": None,
                 "B2A": json.loads(_blob(
                     repo, f"{I.RESULTS_COMMIT}:{I.B2A_RESULTS_PATH}"))}
    # rebuild the derived contract exactly as verify() does
    b2a_res = contracts["B2A"]
    contracts["B2A"] = {"certified": True,
                        "results_commit": I.RESULTS_COMMIT,
                        "results_path": I.B2A_RESULTS_PATH,
                        "results_blob_sha256": _sha(_blob(
                            repo,
                            f"{I.RESULTS_COMMIT}:{I.B2A_RESULTS_PATH}")),
                        "certified_points":
                            [c["point"] for c in
                             b2a_res["certified_points"]],
                        "lb95": b2a_res["certified_points"][0]["lb95"]}
    results = D.run_families(E, panel, pdig, contracts, draws, ck,
                             I.all_stations(repo), done)
    res = {"schema": RESULT_SCHEMA, "generated_utc": "KAT",
           "seal": {"kat": True}, "run_uuid": "f" * 32,
           "reservation": {"kat": True},
           "fire_authorization_sha256": "a" * 64,
           "codex_instrument_pass": {"kat": True},
           "input_manifest": {"path": I.MANIFEST_OUT,
                              "sha256": "m" * 64},
           "sources": {"driver_sha256": "d" * 64,
                       "instrument_sha256": "i" * 64,
                       "engine_sha256": "e" * 64},
           "checkpoint": {"path": CANONICAL_CKPT,
                          "sha256": _sha(_lf(ck))},
           "env": {"python": "kat", "numpy": "kat", "platform": "kat"},
           "panel_sha256": pdig, "n_draws": draws,
           "alpha": E.ALPHA_FAMILY, "families": results,
           "typing_note": "kat", "non_claims": "kat"}
    return tree, ck, res


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

    tree, ck, res = _fixture_run(repo, scratch)
    ok, r = verify(repo, res, ck, expected_draws=49, bind_git=False,
                   snapshots_dir=tree)
    results["positive-fixture"] = ("PASS" if ok else
                                   "DEFECT: " + "; ".join(map(str,
                                                              r[:4])))
    ok_all &= ok
    import copy

    def dv(mut_res=None, mut_ck=None, expected_draws=49):
        r2 = copy.deepcopy(res)
        ck2 = ck
        if mut_ck is not None:
            ck2 = ck + ".mut"
            rows = [json.loads(l) for l in open(ck, encoding="utf-8")
                    if l.strip()]
            rows = mut_ck(rows)
            with open(ck2, "w", encoding="utf-8", newline="\n") as f:
                for row in rows:
                    f.write(json.dumps(row, sort_keys=True) + "\n")
            r2["checkpoint"]["sha256"] = _sha(_lf(ck2))
        if mut_res is not None:
            mut_res(r2)
        return verify(repo, r2, ck2, expected_draws=expected_draws,
                      bind_git=False, snapshots_dir=tree)

    ok, r = dv(mut_res=lambda x: x["families"]["B1A"]["full"].update(
        verdict="NEGATIVE"))
    ok_all &= rec("typing-promotion", ok, r, "RESULT_NOT_RECOMPUTED")
    ok, r = dv(mut_res=lambda x: x["families"]["B2A"]["full"].update(
        p_value=0.0001))
    ok_all &= rec("result-field-mutation", ok, r,
                  "RESULT_NOT_RECOMPUTED")
    ok, r = dv(expected_draws=9999)
    ok_all &= rec("wrong-draws", ok, r, "WRONG_DRAWS")
    ok, r = dv(mut_ck=lambda rows: rows + [dict(
        rows[-1], key="full|B2A")])
    ok_all &= rec("duplicate-key", ok, r, "CHECKPOINT_DUPLICATE_KEY")
    ok, r = dv(mut_ck=lambda rows: rows + [{"key": "x", "stage": "weird"}])
    ok_all &= rec("unknown-stage", ok, r, "CHECKPOINT_UNKNOWN_STAGE")

    def fake_full(rows):
        # an injected positive without its 35 folds trips selection
        # closure before recompute even runs -- the registered expectation
        for row in rows:
            if row.get("key") == "full|B2A":
                row["result"] = dict(row["result"], p_value=0.0001,
                                     verdict="POSITIVE_PRE_LOCO")
        return rows
    ok, r = dv(mut_ck=fake_full)
    ok_all &= rec("fake-full-row", ok, r, "CLOSURE")

    def fake_fold(rows):
        rows.append({"key": "loco|B1A|XX.FAKE", "stage": "loco",
                     "family": "B1A", "station": "XX.FAKE",
                     "result": {"p_value": 0.5}, "panel_sha256":
                     res["panel_sha256"], "dt": 0.0})
        return rows
    ok, r = dv(mut_ck=fake_fold)
    ok_all &= rec("fake-fold-on-nonpositive", ok, r, "CLOSURE")

    def hdr_mut(rows):
        rows[0] = dict(rows[0], run_uuid="0" * 32)
        return rows
    ok, r = dv(mut_ck=hdr_mut)
    ok_all &= rec("header-mismatch", ok, r, "CHECKPOINT_HEADER_MISMATCH")

    def panel_mut(rows):
        for row in rows:
            if row.get("stage") == "panel":
                row["panel_sha256"] = "0" * 64
        return rows
    ok, r = dv(mut_ck=panel_mut)
    ok_all &= rec("panel-mismatch", ok, r, "PANEL_MISMATCH")
    r2 = copy.deepcopy(res)
    r2["checkpoint"]["sha256"] = "0" * 64
    ok, r = verify(repo, r2, ck, expected_draws=49, bind_git=False,
                   snapshots_dir=tree)
    ok_all &= rec("ckpt-bytes-changed", ok, r, "BINDING: checkpoint sha")
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
