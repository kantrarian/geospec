#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SEALED-RUN FIRE DRIVER v2.1 (cayley) -- the ONE real-data run.

Authorized: fresh owner seal option (a) (owner_seal_cal_fresh @ 931ec65).
v2.1 = v2 + the codex second-pass residuals:

  R1  ONE-SHOT PER AUTHORIZED EXPERIMENT: the local lock and reservation
      live under `git rev-parse --git-common-dir` (shared by every
      worktree of the clone), and a REMOTE ATOMIC FIRE LEASE is acquired
      before allow_real=True -- an orphan commit carrying the run UUID is
      pushed expect-absent to refs/f2g/fire-lease and then read back; the
      remote ref-update serialization guarantees exactly one winner
      across clones and hosts. The lease commit is bound into the
      reservation, checkpoint header, and result.
  R3  EXECUTED-BYTES ATTESTATION: driver, instrument, RESULT VERIFIER,
      and engine must each satisfy disk == HEAD == the recorded
      registered blob before any real parse (engine additionally ==
      the frozen 24b0d8f blob). The authorization schema (v2) binds the
      verifier blob too; one canonical sha per source rides the
      reservation, checkpoint header, and result.
  (R2/R4/R5 live in the instrument and result verifier.)

Gates from v2 unchanged: committed fire-authorization record binding the
codex PASS note; --codex-pass must equal the recorded ref; one-shot
across disk + HEAD + git log --all; canonical checkpoint absent on first
fire; schema-closed resume loader; fire-time manifest rebuild comparison;
B2A contract re-derivation; typed refusals everywhere; checkpointed
idempotent run core; hash-sealed result artifact.
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
AUTH_PATH = "docs/f2g_sealed_run_fire_authorization.json"
AUTH_SCHEMA = "f2g-sealed-run-fire-authorization-v2"
EVIDENCE_SCHEMA = "f2g-sealed-run-evidence-v2.1"
LEASE_REF = "refs/f2g/fire-lease"
SEAL_QUOTE_SHA = ("bb94a28bec0060d7d45b799f17536c499539f324da0f58ac3a1edcf"
                  "df594a7e4")
DRIVER_REL = "monitoring/src/f2g_sealed_run_driver_cayley.py"
INSTRUMENT_REL = "monitoring/src/f2g_sealed_run_instrument_cayley.py"
VERIFIER_REL = "monitoring/src/f2g_sealed_run_result_verifier_cayley.py"
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


def git_out(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo).decode()


def common_dir(repo):
    d = git_out(repo, "rev-parse", "--git-common-dir").strip()
    return d if os.path.isabs(d) else os.path.join(repo, d)


def refuse(code, detail=""):
    print(f"{code}{': ' + detail if detail else ''}")
    sys.exit(2)


def check_auth_record(auth, codex_pass_arg, resolve):
    """Finding-1 attestation (v2.1: verifier bound too). resolve(rel) ->
    (disk_lf_sha, head_blob_sha). Returns None on pass, else a typed
    reason."""
    if not isinstance(auth, dict) or auth.get("schema") != AUTH_SCHEMA:
        return "AUTH_SCHEMA"
    need = {"schema", "codex_pass_note", "driver_blob_sha256",
            "instrument_blob_sha256", "verifier_blob_sha256",
            "manifest_sha256", "seal_quote_sha256"}
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
    for rel, key in ((DRIVER_REL, "driver_blob_sha256"),
                     (INSTRUMENT_REL, "instrument_blob_sha256"),
                     (VERIFIER_REL, "verifier_blob_sha256")):
        d, h = resolve(rel)
        if h is None or h != auth[key] or d != h:
            return f"SOURCE_UNATTESTED: {rel.rsplit('/', 1)[-1]}"
    d, h = resolve(I.MANIFEST_OUT)
    if h is None or h != auth["manifest_sha256"] or d != h:
        return "MANIFEST_MISMATCH"
    return None


def acquire_lease(repo, ref, payload):
    """Remote atomic fire lease (residual 1): expect-absent create of an
    orphan commit at `ref` on origin, then read-back verification -- the
    remote serializes ref updates, so exactly one contender's commit can
    be the ref value. Returns (won, lease_commit_sha, remote_sha)."""
    existing = git_out(repo, "ls-remote", "origin", ref).strip()
    if existing:
        return False, None, existing.split()[0]
    p = subprocess.run(["git", "hash-object", "-w", "--stdin"],
                       input=json.dumps(payload, sort_keys=True)
                       .encode() + b"\n", cwd=repo,
                       capture_output=True)
    bsha = p.stdout.decode().strip()
    p = subprocess.run(["git", "mktree"],
                       input=f"100644 blob {bsha}\tlease.json\n".encode(),
                       cwd=repo, capture_output=True)
    tsha = p.stdout.decode().strip()
    csha = subprocess.run(
        ["git", "commit-tree", tsha, "-m",
         f"f2g fire lease {payload.get('run_uuid', '')}"],
        cwd=repo, capture_output=True,
        env=dict(os.environ,
                 GIT_AUTHOR_NAME="cayley",
                 GIT_AUTHOR_EMAIL="mail.rjmathews@gmail.com",
                 GIT_COMMITTER_NAME="cayley",
                 GIT_COMMITTER_EMAIL="mail.rjmathews@gmail.com")) \
        .stdout.decode().strip()
    subprocess.run(["git", "push", "origin", f"{csha}:{ref}"], cwd=repo,
                   capture_output=True)
    after = git_out(repo, "ls-remote", "origin", ref).strip()
    remote_sha = after.split()[0] if after else None
    return (remote_sha == csha), csha, remote_sha


def load_ckpt_strict(path, expect_header):
    """Schema-closed loader (finding 3; v2.1 adds exact-one header/panel
    key enforcement)."""
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
            if r["key"] != "header" or i != 0 or r != expect_header:
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
    """Shared run core (also the result verifier's fixture harness)."""
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

    # gate A: fire authorization (attested, v2 schema incl verifier)
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
    # gate A2 (residual 3): engine executed bytes == HEAD == frozen blob
    eng_disk = sha_b(lf(os.path.join(repo, I.ENGINE_PATH)))
    eng_head = blob(repo, f"HEAD:{I.ENGINE_PATH}")
    eng_frozen = blob(repo, f"{I.ENGINE_COMMIT}:{I.ENGINE_PATH}")
    if eng_head is None or eng_frozen is None or \
            eng_disk != sha_b(eng_head) or eng_disk != sha_b(eng_frozen):
        refuse("SOURCE_UNATTESTED: engine (disk/HEAD/frozen 24b0d8f "
               "disagree)")
    engine_sha = eng_disk
    # gate D1: manifest rebuild comparison
    man_blob = blob(repo, f"HEAD:{I.MANIFEST_OUT}")
    committed = json.loads(man_blob)
    rebuilt = I.build_input_manifest(repo, write=False)
    a, b2 = dict(committed), dict(rebuilt)
    a.pop("generated_utc", None)
    b2.pop("generated_utc", None)
    if json.dumps(a, sort_keys=True) != json.dumps(b2, sort_keys=True):
        refuse("MANIFEST_DRIFT")
    bad = derive_b2a_contract(repo, committed)
    if bad:
        refuse(bad)
    # gate B: one-shot across disk, HEAD, history
    if os.path.exists(os.path.join(repo, RESULT_PATH)):
        refuse("ONE_SHOT_REFUSAL", "result exists on disk")
    if blob(repo, f"HEAD:{RESULT_PATH}") is not None:
        refuse("ONE_SHOT_REFUSAL", "result exists in HEAD")
    if git_out(repo, "log", "--all", "--oneline", "--",
               RESULT_PATH).strip():
        refuse("HISTORY_ONE_SHOT", "result path appears in git history")
    # residual 1: lock + reservation in the COMMON git dir (shared by
    # every worktree of this clone)
    cdir = common_dir(repo)
    lock_path = os.path.join(cdir, "f2g_sealed_run.lock")
    res_path = os.path.join(cdir, "f2g_sealed_run_reservation.json")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        refuse("LOCK_HELD", "another process holds the run lock (remove "
               "manually ONLY after confirming no live process)")
    try:
        ck_path = os.path.join(repo, CANONICAL_CKPT)
        if not args.resume:
            if os.path.exists(res_path):
                refuse("RESERVATION_EXISTS", "use --resume")
            if os.path.exists(ck_path) or \
                    blob(repo, f"HEAD:{CANONICAL_CKPT}") is not None:
                refuse("CHECKPOINT_PREEXISTS")
            run_uuid = os.urandom(16).hex()
            # residual 1: remote atomic fire lease BEFORE allow_real
            won, lease_commit, remote_sha = acquire_lease(
                repo, LEASE_REF,
                {"schema": "f2g-fire-lease-v1", "run_uuid": run_uuid,
                 "auth_sha256": auth_sha,
                 "host": platform.node(),
                 "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                              time.gmtime())})
            if not won:
                refuse("LEASE_LOST", f"remote fire lease already held "
                       f"({remote_sha})")
            reservation = {"schema": "f2g-sealed-run-reservation-v2",
                           "run_uuid": run_uuid,
                           "auth_sha256": auth_sha,
                           "manifest_sha256": sha_b(man_blob),
                           "driver_sha256": auth["driver_blob_sha256"],
                           "instrument_sha256":
                               auth["instrument_blob_sha256"],
                           "verifier_sha256":
                               auth["verifier_blob_sha256"],
                           "engine_sha256": engine_sha,
                           "lease_ref": LEASE_REF,
                           "lease_commit": lease_commit,
                           "ckpt_path": CANONICAL_CKPT,
                           "created_utc": time.strftime(
                               "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            fd = os.open(res_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                json.dump(reservation, f, indent=1, sort_keys=True)
        else:
            if not os.path.exists(res_path):
                refuse("RESERVATION_MISSING")
            reservation = json.loads(open(res_path,
                                          encoding="utf-8").read())
            if reservation.get("auth_sha256") != auth_sha or \
                    reservation.get("manifest_sha256") != \
                    sha_b(man_blob) or \
                    reservation.get("driver_sha256") != \
                    auth["driver_blob_sha256"] or \
                    reservation.get("instrument_sha256") != \
                    auth["instrument_blob_sha256"] or \
                    reservation.get("verifier_sha256") != \
                    auth["verifier_blob_sha256"] or \
                    reservation.get("ckpt_path") != CANONICAL_CKPT:
                refuse("RESERVATION_MISMATCH")
            run_uuid = reservation["run_uuid"]
            held = git_out(repo, "ls-remote", "origin",
                           LEASE_REF).strip()
            if not held or held.split()[0] != \
                    reservation.get("lease_commit"):
                refuse("LEASE_MISMATCH", "remote lease is not this "
                       "reservation's lease commit")
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
                  "verifier_sha256": auth["verifier_blob_sha256"],
                  "engine_sha256": engine_sha,
                  "lease_commit": reservation["lease_commit"]}
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
                refuse("CHECKPOINT_PANEL_MISMATCH")
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
            "schema": "f2g-sealed-run-result-v2.1",
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
                        "verifier_sha256": auth["verifier_blob_sha256"],
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
            "typing_note": REGISTERED_TYPING_NOTE,
            "non_claims": REGISTERED_NON_CLAIMS,
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
            os.unlink(lock_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
