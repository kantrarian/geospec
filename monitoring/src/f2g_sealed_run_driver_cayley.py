#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SEALED-RUN FIRE DRIVER v2 (cayley) -- the ONE real-data run.

Authorized: fresh owner seal option (a) (owner_seal_cal_fresh @ 931ec65).
v2 implements the codex pre-fire WORKS-WITH-FIX findings 1/2/3/5:

GATE CHAIN (every gate BEFORE any real snapshot parse):
  A. FIRE AUTHORIZATION (finding 1): a committed record at AUTH_PATH must
     bind the codex PASS note (geospec path + blob sha), THIS driver's and
     the instrument's exact blob shas, the input-manifest sha, and the
     seal quote sha. --codex-pass must equal the recorded note ref; the
     note blob, both source blobs, and the working-tree bytes must all
     match. An arbitrary string can no longer open the gate.
  B. ONE-SHOT (finding 2): the result path must be absent on disk, in
     HEAD, and in `git log --all` history; a process LOCK (exclusive
     create) admits exactly one process; a durable RESERVATION (exclusive
     create, run UUID + all binding shas + the canonical checkpoint path)
     is created before allow_real=True. A restart may ONLY resume the
     same reservation and the canonical checkpoint (--resume); stale
     locks are never auto-stolen (operator removes after confirming no
     live process -- disclosed manual step).
  C. CHECKPOINT (finding 3): one canonical checkpoint path; on first fire
     it must be ABSENT; on resume it is loaded schema-closed (exact
     per-stage field sets, no duplicate keys, no unknown stages, header
     must equal the full binding tuple, every row bound to the rebuilt
     panel digest). The frozen sealed-result verifier independently
     recomputes every cached row after the run.
  D. CLOSURE (finding 5): the input manifest is REBUILT at fire time and
     compared to the committed manifest (generated_utc normalized); the
     B2A power contract is re-derived from the pinned results blob and
     exact-checked; the result artifact binds run UUID, reservation,
     authorization, source/engine shas, checkpoint path+sha, and the
     runtime environment.
Usage: driver.py <repo> --codex-pass <ref> [--resume]
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import f2g_sealed_run_instrument_cayley as I

RESULT_PATH = "docs/f2g_sealed_run_result.json"
CANONICAL_CKPT = "docs/f2g_sealed_run_evidence.jsonl"
RESERVATION_PATH = "docs/f2g_sealed_run_reservation.json"
AUTH_PATH = "docs/f2g_sealed_run_fire_authorization.json"
LOCK_PATH = "docs/.f2g_sealed_run.lock"
AUTH_SCHEMA = "f2g-sealed-run-fire-authorization-v1"
EVIDENCE_SCHEMA = "f2g-sealed-run-evidence-v2"
SEAL_QUOTE_SHA = ("bb94a28bec0060d7d45b799f17536c499539f324da0f58ac3a1edcf"
                  "df594a7e4")
DRIVER_REL = "monitoring/src/f2g_sealed_run_driver_cayley.py"
INSTRUMENT_REL = "monitoring/src/f2g_sealed_run_instrument_cayley.py"
FAMS = ("B1A", "B2A", "B3A")
STAGE_FIELDS = {
    "header": {"key", "stage", "schema", "purpose", "run_uuid",
               "codex_note_sha256", "auth_sha256", "manifest_sha256",
               "driver_sha256", "instrument_sha256", "engine_sha256"},
    "panel": {"key", "stage", "panel_sha256"},
    "full": {"key", "stage", "family", "result", "panel_sha256", "dt"},
    "loco": {"key", "stage", "family", "station", "result",
             "panel_sha256", "dt"},
}


def sha_b(b):
    return hashlib.sha256(b).hexdigest()


def lf(path):
    return open(path, "rb").read().replace(b"\r\n", b"\n")


def blob(repo, ref):
    try:
        return subprocess.check_output(["git", "cat-file", "blob", ref],
                                       cwd=repo,
                                       stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None


def refuse(code, detail=""):
    print(f"{code}{': ' + detail if detail else ''}")
    sys.exit(2)


def check_auth_record(auth, codex_pass_arg, resolve):
    """Finding-1 attestation. resolve(rel) -> (disk_lf_sha, head_blob_sha)
    is injected so KATs can exercise every refusal. Returns None on pass,
    else a typed reason string."""
    if not isinstance(auth, dict) or auth.get("schema") != AUTH_SCHEMA:
        return "AUTH_SCHEMA"
    need = {"schema", "codex_pass_note", "driver_blob_sha256",
            "instrument_blob_sha256", "manifest_sha256",
            "seal_quote_sha256"}
    if set(auth) != need:
        return "AUTH_SCHEMA"
    note = auth["codex_pass_note"]
    if not isinstance(note, dict) or \
            set(note) != {"path", "blob_sha256", "ref"}:
        return "AUTH_SCHEMA"
    if codex_pass_arg != note["ref"]:
        return "AUTH_REF_MISMATCH"
    if auth["seal_quote_sha256"] != SEAL_QUOTE_SHA:
        return "AUTH_SEAL_MISMATCH"
    d, h = resolve(note["path"])
    if h is None or h != note["blob_sha256"] or d != h:
        return "AUTH_NOTE_UNATTESTED"
    d, h = resolve(DRIVER_REL)
    if h is None or h != auth["driver_blob_sha256"] or d != h:
        return "SOURCE_UNATTESTED: driver"
    d, h = resolve(INSTRUMENT_REL)
    if h is None or h != auth["instrument_blob_sha256"] or d != h:
        return "SOURCE_UNATTESTED: instrument"
    d, h = resolve(I.MANIFEST_OUT)
    if h is None or h != auth["manifest_sha256"] or d != h:
        return "MANIFEST_MISMATCH"
    return None


def load_ckpt_strict(path, expect_header):
    """Finding-3 schema-closed loader. Refuses duplicates, unknown
    stages, malformed rows, and any header not equal to the full binding
    tuple. Returns {key: row}."""
    done = {}
    rows = [json.loads(l) for l in open(path, encoding="utf-8")
            if l.strip()]
    if not rows or rows[0].get("stage") != "header":
        raise RuntimeError("CHECKPOINT_HEADER_MISSING")
    for i, r in enumerate(rows):
        st = r.get("stage")
        if st not in STAGE_FIELDS:
            raise RuntimeError(f"CHECKPOINT_UNKNOWN_STAGE: row {i} {st!r}")
        if set(r) != STAGE_FIELDS[st]:
            raise RuntimeError(f"CHECKPOINT_ROW_SCHEMA: row {i} ({st})")
        if r["key"] in done:
            raise RuntimeError(f"CHECKPOINT_DUPLICATE_KEY: {r['key']}")
        if st == "header":
            if i != 0 or r != expect_header:
                raise RuntimeError("CHECKPOINT_HEADER_MISMATCH")
        elif st == "panel":
            if r["key"] != "panel":
                raise RuntimeError("CHECKPOINT_ROW_SCHEMA: panel key")
        elif st == "full":
            if r["family"] not in FAMS or \
                    r["key"] != f"full|{r['family']}":
                raise RuntimeError(f"CHECKPOINT_ROW_SCHEMA: {r['key']}")
        elif st == "loco":
            if r["family"] not in FAMS or \
                    r["key"] != f"loco|{r['family']}|{r['station']}":
                raise RuntimeError(f"CHECKPOINT_ROW_SCHEMA: {r['key']}")
        done[r["key"]] = r
    return done


def emit(path, row):
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def derive_b2a_contract(repo, manifest):
    """Finding-5: the B2A contract must re-derive from the pinned results
    blob, never be trusted from the manifest echo."""
    raw = blob(repo, f"{I.RESULTS_COMMIT}:{I.B2A_RESULTS_PATH}")
    if raw is None:
        return "CONTRACT_DERIVATION_MISMATCH: results blob unreadable"
    res = json.loads(raw)
    mc = manifest["power_contracts"]["B2A"]
    pts = [c["point"] for c in res.get("certified_points", [])]
    lbs = {c["lb95"] for c in res.get("certified_points", [])}
    if res.get("terminal_type") != "CERTIFIED" or \
            pts != mc.get("certified_points") or \
            len(lbs) != 1 or list(lbs)[0] != mc.get("lb95") or \
            sha_b(raw) != mc.get("results_blob_sha256"):
        return "CONTRACT_DERIVATION_MISMATCH"
    return None


def run_families(E, panel, pdig, contracts, n_draws, ckpt, stations,
                 done):
    """Shared run core (also used by the result verifier's fixture
    harness). Emits checkpoint rows; returns per-family results."""
    alpha = E.ALPHA_FAMILY
    fams = {"B1A": E.b1a_family_cal, "B2A": E.b2a_family_cal,
            "B3A": E.b3a_family_cal}
    results = {}
    for fam, fn in fams.items():
        key = f"full|{fam}"
        if key in done:
            if done[key]["panel_sha256"] != pdig:
                raise RuntimeError(f"CHECKPOINT_PANEL_MISMATCH: {key}")
            full = done[key]["result"]
            print(f"[{fam}] full from checkpoint (binding-validated)",
                  flush=True)
        else:
            ts = time.time()
            full = fn(panel, doc_sha256=I.AMENDMENT2_SHA, n_draws=n_draws,
                      power_contract=contracts[fam])
            emit(ckpt, {"key": key, "stage": "full", "family": fam,
                        "result": full, "panel_sha256": pdig,
                        "dt": round(time.time() - ts, 1)})
            done[key] = {"panel_sha256": pdig, "result": full}
        p = full.get("p_value")
        print(f"[{fam}] full p={p} verdict={full.get('verdict')!r}",
              flush=True)
        folds = None
        gate = None
        if p is not None and p <= alpha:
            folds = []
            for st in stations:
                fk = f"loco|{fam}|{st}"
                if fk in done:
                    if done[fk]["panel_sha256"] != pdig:
                        raise RuntimeError(
                            f"CHECKPOINT_PANEL_MISMATCH: {fk}")
                    fr = done[fk]["result"]
                else:
                    ts = time.time()
                    fr = fn(I.drop_station(panel, st),
                            doc_sha256=I.AMENDMENT2_SHA, n_draws=n_draws,
                            power_contract=contracts[fam],
                            fold=f"loco:{st}")
                    emit(ckpt, {"key": fk, "stage": "loco", "family": fam,
                                "station": st, "result": fr,
                                "panel_sha256": pdig,
                                "dt": round(time.time() - ts, 1)})
                    done[fk] = {"panel_sha256": pdig, "result": fr}
                folds.append({"station": st,
                              "p_value": fr.get("p_value")})
            gate = E.loco_gate({"p_value": p}, folds, alpha)
            print(f"[{fam}] LOCO gate: {gate}", flush=True)
        else:
            print(f"[{fam}] nonpositive full -> folds not applicable "
                  "(conjunctive gate cannot promote)", flush=True)
        results[fam] = {"full": full, "loco_folds": folds,
                        "loco_gate": gate}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--codex-pass", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)
    os.chdir(repo)

    def resolve(rel):
        p = os.path.join(repo, rel)
        d = sha_b(lf(p)) if os.path.exists(p) else None
        h = blob(repo, f"HEAD:{rel}")
        return d, (sha_b(h) if h is not None else None)

    # gate A: fire authorization
    auth_blob = blob(repo, f"HEAD:{AUTH_PATH}")
    if auth_blob is None:
        refuse("AUTH_MISSING", "no committed fire-authorization record")
    if not os.path.exists(os.path.join(repo, AUTH_PATH)) or \
            sha_b(lf(os.path.join(repo, AUTH_PATH))) != sha_b(auth_blob):
        refuse("AUTH_UNCOMMITTED", "record differs from HEAD")
    auth = json.loads(auth_blob)
    bad = check_auth_record(auth, args.codex_pass, resolve)
    if bad:
        refuse(bad)
    auth_sha = sha_b(auth_blob)
    # gate D1: manifest rebuild comparison (generated_utc normalized)
    man_blob = blob(repo, f"HEAD:{I.MANIFEST_OUT}")
    committed = json.loads(man_blob)
    rebuilt = I.build_input_manifest(repo, write=False)
    a, b2 = dict(committed), dict(rebuilt)
    a.pop("generated_utc", None)
    b2.pop("generated_utc", None)
    if json.dumps(a, sort_keys=True) != json.dumps(b2, sort_keys=True):
        refuse("MANIFEST_DRIFT", "rebuilt manifest differs from committed")
    bad = derive_b2a_contract(repo, committed)
    if bad:
        refuse(bad)
    # gate B: one-shot across disk, HEAD, and ALL history
    if os.path.exists(os.path.join(repo, RESULT_PATH)):
        refuse("ONE_SHOT_REFUSAL", "result exists on disk")
    if blob(repo, f"HEAD:{RESULT_PATH}") is not None:
        refuse("ONE_SHOT_REFUSAL", "result exists in HEAD")
    hist = subprocess.check_output(
        ["git", "log", "--all", "--oneline", "--", RESULT_PATH],
        cwd=repo).decode().strip()
    if hist:
        refuse("HISTORY_ONE_SHOT", "result path appears in git history")
    # process lock (exclusive; stale locks are an operator decision)
    try:
        lock_fd = os.open(os.path.join(repo, LOCK_PATH),
                          os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        refuse("LOCK_HELD", "another process holds the run lock (remove "
               "manually ONLY after confirming no live process)")
    try:
        engine_sha = sha_b(blob(repo, f"HEAD:{I.ENGINE_PATH}"))
        res_path = os.path.join(repo, RESERVATION_PATH)
        ck_path = os.path.join(repo, CANONICAL_CKPT)
        if not args.resume:
            if os.path.exists(res_path):
                refuse("RESERVATION_EXISTS", "use --resume for the same "
                       "reservation")
            if os.path.exists(ck_path) or \
                    blob(repo, f"HEAD:{CANONICAL_CKPT}") is not None:
                refuse("CHECKPOINT_PREEXISTS", "canonical checkpoint must "
                       "be absent on first fire")
            run_uuid = os.urandom(16).hex()
            reservation = {"schema": "f2g-sealed-run-reservation-v1",
                           "run_uuid": run_uuid,
                           "auth_sha256": auth_sha,
                           "manifest_sha256": sha_b(man_blob),
                           "driver_sha256": auth["driver_blob_sha256"],
                           "instrument_sha256":
                               auth["instrument_blob_sha256"],
                           "engine_sha256": engine_sha,
                           "ckpt_path": CANONICAL_CKPT,
                           "created_utc": time.strftime(
                               "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            fd = os.open(res_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                json.dump(reservation, f, indent=1, sort_keys=True)
        else:
            if not os.path.exists(res_path):
                refuse("RESERVATION_MISSING", "--resume without a "
                       "reservation")
            reservation = json.loads(open(res_path,
                                          encoding="utf-8").read())
            if reservation.get("auth_sha256") != auth_sha or \
                    reservation.get("manifest_sha256") != sha_b(man_blob) \
                    or reservation.get("driver_sha256") != \
                    auth["driver_blob_sha256"] or \
                    reservation.get("instrument_sha256") != \
                    auth["instrument_blob_sha256"] or \
                    reservation.get("ckpt_path") != CANONICAL_CKPT:
                refuse("RESERVATION_MISMATCH")
            run_uuid = reservation["run_uuid"]
        header = {"key": "header", "stage": "header",
                  "schema": EVIDENCE_SCHEMA,
                  "purpose": "sealed-run-production",
                  "run_uuid": run_uuid,
                  "codex_note_sha256":
                      auth["codex_pass_note"]["blob_sha256"],
                  "auth_sha256": auth_sha,
                  "manifest_sha256": sha_b(man_blob),
                  "driver_sha256": auth["driver_blob_sha256"],
                  "instrument_sha256": auth["instrument_blob_sha256"],
                  "engine_sha256": engine_sha}
        if os.path.exists(ck_path):
            done = load_ckpt_strict(ck_path, header)
        else:
            emit(ck_path, header)
            done = {"header": header}
        # ---- all gates green: build the REAL panel ----
        t0 = time.time()
        sys.path.insert(0, os.path.join(repo, "monitoring", "src"))
        import d2_f2g_phase_b_stats as E
        panel = I.build_panel(repo, os.path.join(repo, I.ARTIFACT_ROOT,
                                                 "snapshots"),
                              allow_real=True)
        pdig = I.panel_digest(panel)
        print(f"[panel] REAL panel built + digested {pdig[:12]}... "
              f"({time.time()-t0:.0f}s)", flush=True)
        if "panel" in done:
            if done["panel"]["panel_sha256"] != pdig:
                refuse("CHECKPOINT_PANEL_MISMATCH", "resumed checkpoint "
                       "was built from different inputs")
        else:
            emit(ck_path, {"key": "panel", "stage": "panel",
                           "panel_sha256": pdig})
            done["panel"] = {"panel_sha256": pdig}
        contracts = {"B1A": None, "B3A": None,
                     "B2A": committed["power_contracts"]["B2A"]}
        results = run_families(E, panel, pdig, contracts, 9999, ck_path,
                               I.all_stations(repo), done)
        import numpy as np
        out = {
            "schema": "f2g-sealed-run-result-v2",
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                           time.gmtime()),
            "seal": committed["seal"],
            "run_uuid": run_uuid,
            "reservation": reservation,
            "fire_authorization_sha256": auth_sha,
            "codex_instrument_pass": auth["codex_pass_note"],
            "input_manifest": {"path": I.MANIFEST_OUT,
                               "sha256": sha_b(man_blob)},
            "sources": {"driver_sha256": auth["driver_blob_sha256"],
                        "instrument_sha256":
                            auth["instrument_blob_sha256"],
                        "engine_sha256": engine_sha},
            "checkpoint": {"path": CANONICAL_CKPT,
                           "sha256": sha_b(lf(ck_path))},
            "env": {"python": platform.python_version(),
                    "numpy": np.__version__,
                    "platform": platform.platform()},
            "panel_sha256": pdig,
            "n_draws": 9999,
            "alpha": E.ALPHA_FAMILY,
            "families": results,
            "typing_note": "B2A verdict-bearing under the certified "
                           "power contract; B1A/B3A nonpositives are "
                           "typed CANNOT_DETERMINE_NO_POWER non-answers, "
                           "never 'no signal'",
            "non_claims": "no earthquake forecast, precursor, or "
                          "displacement claims; Lambda_geo remains "
                          "INCONCLUSIVE; this result reports the "
                          "registered Phase-B family statistics on the "
                          "sealed Phase-A graph series and nothing else",
        }
        with open(os.path.join(repo, RESULT_PATH), "w", encoding="utf-8",
                  newline="\n") as f:
            json.dump(out, f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"SEALED RUN COMPLETE -> {RESULT_PATH} "
              f"({time.time()-t0:.0f}s total)", flush=True)
    finally:
        os.close(lock_fd)
        try:
            os.unlink(os.path.join(repo, LOCK_PATH))
        except OSError:
            pass


if __name__ == "__main__":
    main()
