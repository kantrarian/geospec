#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SEALED-RUN FIRE DRIVER (cayley) -- the ONE real-data run.

Authorized: fresh owner seal option (a) (owner_seal_cal_fresh @ 931ec65).
GATES ENFORCED HERE:
  1. --codex-pass <inbox ref> is REQUIRED (the codex instrument-check
     pass); firing without it refuses.
  2. ONE-SHOT: refuses if the sealed result artifact already exists on
     disk or in HEAD.
  3. The committed input manifest must byte-match the working tree.
  4. Engine checkout must attest against the pinned blob.
Semantics: per family FULL run at the registered 9,999 draws under the
frozen Amendment-2 seed root; the 35 LOCO folds run ONLY if the full-data
p passes alpha (the conjunctive gate cannot promote, so folds are
irrelevant to a nonpositive -- same early-exit as the registered power
driver); B2A carries the certified power contract (verdict-bearing),
B1A/B3A carry none (nonpositives type CANNOT_DETERMINE_NO_POWER).
Checkpointed and idempotent by key; the checkpoint is the run's evidence
capsule and is committed with the result artifact.
Usage: driver.py <repo> --codex-pass <ref> --ckpt <path>
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import f2g_sealed_run_instrument_cayley as I

RESULT_PATH = "docs/f2g_sealed_run_result.json"
EVIDENCE_SCHEMA = "f2g-sealed-run-evidence-v1"


def sha_b(b):
    return hashlib.sha256(b).hexdigest()


def emit(path, row):
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def load_done(path):
    done = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done[r["key"]] = r
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--codex-pass", required=True,
                    help="inbox ref of the codex instrument-check PASS")
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)
    os.chdir(repo)
    # gate 2: one-shot
    if os.path.exists(os.path.join(repo, RESULT_PATH)):
        print("ONE_SHOT_REFUSAL: sealed result artifact already exists")
        sys.exit(2)
    try:
        subprocess.check_output(["git", "cat-file", "-e",
                                 f"HEAD:{RESULT_PATH}"], cwd=repo,
                                stderr=subprocess.DEVNULL)
        print("ONE_SHOT_REFUSAL: sealed result artifact already committed")
        sys.exit(2)
    except subprocess.CalledProcessError:
        pass
    # gate 3: committed manifest == working tree
    man_disk = open(os.path.join(repo, I.MANIFEST_OUT), "rb").read() \
        .replace(b"\r\n", b"\n")
    try:
        man_blob = subprocess.check_output(
            ["git", "cat-file", "blob", f"HEAD:{I.MANIFEST_OUT}"],
            cwd=repo, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("MANIFEST_UNCOMMITTED: input manifest is not in HEAD")
        sys.exit(2)
    if sha_b(man_disk) != sha_b(man_blob):
        print("MANIFEST_UNCOMMITTED: input manifest differs from HEAD")
        sys.exit(2)
    manifest = json.loads(man_blob)
    # gate 4 + full input re-verification at fire time (byte hashes)
    I.build_input_manifest(repo, write=False)
    sys.path.insert(0, os.path.join(repo, "monitoring", "src"))
    import d2_f2g_phase_b_stats as E
    fams = {"B1A": E.b1a_family_cal, "B2A": E.b2a_family_cal,
            "B3A": E.b3a_family_cal}
    contracts = {"B1A": None, "B3A": None,
                 "B2A": manifest["power_contracts"]["B2A"]}
    done = load_done(args.ckpt)
    if not done:
        emit(args.ckpt, {"key": "header", "stage": "header",
                         "purpose": "sealed-run-production",
                         "schema": EVIDENCE_SCHEMA,
                         "codex_instrument_pass": args.codex_pass,
                         "input_manifest_sha256": sha_b(man_blob)})
        done = load_done(args.ckpt)
    hdr = done.get("header", {})
    if hdr.get("purpose") != "sealed-run-production":
        print("CHECKPOINT_PURPOSE_MISMATCH")
        sys.exit(2)
    t0 = time.time()
    panel = I.build_panel(repo, os.path.join(repo, I.ARTIFACT_ROOT,
                                             "snapshots"), allow_real=True)
    pdig = I.panel_digest(panel)
    print(f"[panel] REAL panel built + digested {pdig[:12]}... "
          f"({time.time()-t0:.0f}s)", flush=True)
    stations = I.all_stations(repo)
    alpha = E.ALPHA_FAMILY
    results = {}
    for fam, fn in fams.items():
        key = f"full|{fam}"
        if key in done:
            full = done[key]["result"]
            print(f"[{fam}] full result from checkpoint (idempotent)",
                  flush=True)
        else:
            ts = time.time()
            full = fn(panel, doc_sha256=I.AMENDMENT2_SHA, n_draws=9999,
                      power_contract=contracts[fam])
            emit(args.ckpt, {"key": key, "stage": "full", "family": fam,
                             "result": full, "panel_sha256": pdig,
                             "dt": round(time.time() - ts, 1)})
            done = load_done(args.ckpt)
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
                    fr = done[fk]["result"]
                else:
                    ts = time.time()
                    fr = fn(I.drop_station(panel, st),
                            doc_sha256=I.AMENDMENT2_SHA, n_draws=9999,
                            power_contract=contracts[fam],
                            fold=f"loco:{st}")
                    emit(args.ckpt, {"key": fk, "stage": "loco",
                                     "family": fam, "station": st,
                                     "result": fr,
                                     "dt": round(time.time() - ts, 1)})
                    done = load_done(args.ckpt)
                folds.append({"station": st, "p_value": fr.get("p_value")})
            gate = E.loco_gate({"p_value": p}, folds, alpha)
            print(f"[{fam}] LOCO gate: {gate}", flush=True)
        else:
            print(f"[{fam}] nonpositive full -> folds not applicable "
                  "(conjunctive gate cannot promote)", flush=True)
        results[fam] = {"full": full, "loco_folds": folds,
                        "loco_gate": gate}
    out = {
        "schema": "f2g-sealed-run-result-v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                       time.gmtime()),
        "seal": manifest["seal"],
        "codex_instrument_pass": args.codex_pass,
        "input_manifest": {"path": I.MANIFEST_OUT,
                           "sha256": sha_b(man_blob)},
        "panel_sha256": pdig,
        "n_draws": 9999,
        "alpha": alpha,
        "families": results,
        "typing_note": "B2A verdict-bearing under the certified power "
                       "contract; B1A/B3A nonpositives are typed "
                       "CANNOT_DETERMINE_NO_POWER non-answers, never "
                       "'no signal'",
        "non_claims": "no earthquake forecast, precursor, or displacement "
                      "claims; Lambda_geo remains INCONCLUSIVE; this "
                      "result reports the registered Phase-B family "
                      "statistics on the sealed Phase-A graph series and "
                      "nothing else",
    }
    with open(os.path.join(repo, RESULT_PATH), "w", encoding="utf-8",
              newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"SEALED RUN COMPLETE -> {RESULT_PATH} "
          f"({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
