#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FULL equivalence gate for the amended-lane estimation (codex repair 2).

Common rev-1.3 sec 4 requires exact full-data AND EVERY-LOCO-fold p/verdict
equality before any table is admissible. The v1 driver gate checked full +
ONE fold; this script re-proves the B1A memoized reduction against the
SOURCE-ATTESTED admitted engine over full + ALL 35 station folds on the
registered fixture, and attests B2A/B3A as direct-engine (no reduction, so
equality is definitional; their binding is the engine attestation itself).
Emits docs/f2g_phase_b_power_a_fullgate_receipt.json.
"""
import hashlib
import json
import subprocess
import sys

import d2_f2g_phase_b_stats as E
import f2g_phase_b_power_estimation_cayley as D0
import f2g_phase_b_power_estimation_a_cayley as DA


def sha256_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def git_short(path):
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%h", "--", path],
        cwd=r"C:\geospec").decode().strip()


def main(out_path):
    engine_file = E.__file__
    driver_file = DA.__file__
    receipt = {
        "schema": "f2g-phase-b-amended-fullgate-receipt-v1",
        "engine_commit": git_short("monitoring/src/d2_f2g_phase_b_stats.py"),
        "engine_disk_sha256": sha256_file(engine_file),
        "driver_commit": git_short(
            "monitoring/src/f2g_phase_b_power_estimation_a_cayley.py"),
        "driver_disk_sha256": sha256_file(driver_file),
        "fixture": {"family": "B1A",
                    "point": {"delta_lat": 1.2, "k": 25, "n_e": 10},
                    "rep": 0, "n_draws": 199},
    }
    panel = DA.make_panel("B1A", {"delta_lat": 1.2, "k": 25, "n_e": 10}, 0)
    memo = DA.B1AMemo(panel)
    checks = []
    d_p, _nv, d_T = memo.family_p(199, "full")
    eng = E.b1a_family(panel, doc_sha256=DA.AMENDMENT_SHA, n_draws=199,
                       power_contract={"certified": True})
    checks.append({"fold": "full", "driver_p": d_p, "engine_p": eng["p_value"],
                   "p_equal": d_p == eng["p_value"],
                   "T_equal": d_T == eng["T_obs"],
                   "verdict_equal": (d_p is not None and
                                     (d_p <= E.ALPHA_FAMILY) ==
                                     ("POSITIVE" in str(eng["verdict"])))})
    for st in D0.all_stations(panel):
        fp, _n, fT = memo.family_p(199, f"loco:{st}", excluded_station=st)
        ef = E.b1a_family(D0.drop_station(panel, st),
                          doc_sha256=DA.AMENDMENT_SHA, n_draws=199,
                          power_contract={"certified": True},
                          fold=f"loco:{st}")
        checks.append({"fold": f"loco:{st}", "driver_p": fp,
                       "engine_p": ef["p_value"],
                       "p_equal": fp == ef["p_value"],
                       "T_equal": fT == ef["T_obs"],
                       "verdict_equal": (fp is None) ==
                                        (ef["p_value"] is None)})
    all_ok = all(c["p_equal"] and c["T_equal"] for c in checks)
    receipt["folds_checked"] = sum(1 for c in checks
                                   if c["fold"].startswith("loco:"))
    receipt["full_equal"] = bool(checks[0]["p_equal"] and checks[0]["T_equal"])
    receipt["fold_equal_all"] = bool(all(
        c["p_equal"] and c["T_equal"] for c in checks[1:]))
    receipt["all_equal"] = bool(all_ok)
    receipt["per_fold_p_vector_sha256"] = hashlib.sha256(json.dumps(
        [c["driver_p"] for c in checks], sort_keys=True).encode()).hexdigest()
    receipt["checks"] = checks
    receipt["b2a_b3a_binding"] = (
        "direct-engine families: the estimation driver calls E.b2a_family/"
        "E.b3a_family with no reduction, so p/verdict equality is "
        "definitional; their binding is the engine attestation above")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(receipt, f, indent=1, sort_keys=True)
        f.write("\n")
    print(json.dumps({k: receipt[k] for k in
                      ("engine_commit", "folds_checked", "full_equal",
                       "fold_equal_all", "all_equal")}))
    if not all_ok:
        sys.exit(2)


if __name__ == "__main__":
    main(sys.argv[1])
