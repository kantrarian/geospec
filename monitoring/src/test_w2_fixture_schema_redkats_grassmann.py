#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FIXTURE/PRODUCTION SCHEMA SEPARATION -- red-KAT lock (grassmann).

codex 0404Z item 3: the schema repair at `ef9fe97f` was accepted on
code review, but its four-way green claim had NO committed regression
lock -- so the separation could be weakened or deleted without any
test noticing. That is the same evidence gap this program keeps
finding, in miniature and in my own surface.

This lock is PORTABLE BY CONSTRUCTION: it builds its own body, store
and archive in a temporary directory. It never reads the evidence
host's `E:` store and it has NO skip path -- a missing input is a
failure, not a green exit.

The four directions:
  1. the PRODUCTION verifier refuses the FIXTURE schema
  2. the FIXTURE verifier refuses the PRODUCTION schema
  3. a non-real fixture with one genuinely reopened+recomputed
     lineage body is ACCEPTED, reporting bodies_recomputed == 1 and
     the fixture-only anti-admission stamp
  4. a zero-reuse fixture refuses as VACUOUS, while the all-HTTP
     fixture may still exercise ceiling membership only
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import w2_disposition_capsule_grassmann as DISP
import w2_acquisition_capture_grassmann as CAP

FAILS = []


def check(name, ok, detail=""):
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}"
          + (f" - {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def _refuses(fn, needle):
    try:
        fn()
        return False
    except DISP.DispositionRefusal as e:
        return needle in str(e)


def main():
    repo = os.path.abspath(os.path.join(_HERE, "..", ".."))
    tmp = tempfile.mkdtemp(prefix="w2_fixture_schema_")
    store = os.path.join(tmp, "store")
    os.makedirs(store, exist_ok=True)
    try:
        # the REAL authority supplies static_layer so contracts can
        # derive; the key set is REDUCED, so this is provably not the
        # production census and the mint guard stays satisfied
        real, _sha = DISP._authority("HEAD")
        pk = real["prestart_expected_keys"]
        lane, ck = "MAG_FEED", "frn"
        day = pk[lane][ck][0]
        keys = {lane: {ck: [day]}}
        fauth = dict(real)
        fauth["prestart_expected_keys"] = keys
        fauth["prestart_expected_keys_sha256"] = DISP._canon(keys)
        v4key = f"{lane}/{ck}/{day}"

        # (4) an ALL-HTTP fixture: ceiling membership only; its
        # lineage verification is VACUOUS and must refuse
        allhttp = DISP.build_fixture_capsule(
            fauth, [v4key], store, os.path.join(tmp, "a1.json"))
        check("fixture carries the FIXTURE schema",
              allhttp["schema"] == DISP.FIXTURE_CAPSULE_SCHEMA,
              allhttp["schema"])
        check("zero-reuse fixture lineage refuses VACUOUS",
              _refuses(lambda: DISP.verify_fixture_lineage_registry(
                  allhttp, fauth, store), "VACUOUS"))

        # (3) a fixture WITH one real reopened + recomputed body.
        # The body is MINTED here, not borrowed from the evidence
        # host: a minimal USGS-shaped all-null day, which the
        # registered transform admits as ADMITTED_ABSENCE.
        from datetime import datetime, timedelta
        d0 = datetime.fromisoformat(day + "T00:00:00")
        times = [(d0 + timedelta(minutes=i)).strftime(
            "%Y-%m-%dT%H:%M:%S.000Z") for i in range(1440)]
        body = json.dumps({
            "type": "Timeseries",
            "metadata": {"intermagnet": {
                "imo": {"iaga_code": "FRN"},
                "reported_orientation": "XYZF"}},
            "times": times,
            "values": [{"id": c, "values": [None] * 1440}
                       for c in ("X", "Y", "Z", "F")]}).encode()
        bsha = hashlib.sha256(body).hexdigest()
        with open(os.path.join(store, bsha + ".body"), "wb") as f:
            f.write(body)
        import w2_accrual_instrument_cayley as ACC
        s4 = ACC.authoritative_static_contract(fauth, lane, ck, day)
        art = CAP.admission_transform(lane, body, s4)
        old_auth = {"commit": subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            capture_output=True).stdout.decode().strip(),
            "path": DISP.AUTHORITY_PATH,
            "blob_sha256": _sha,
            "keys_sha256": real["prestart_expected_keys_sha256"]}
        lineage = {v4key: {
            "v3_key": v4key, "raw_body_sha256": bsha,
            "raw_body_bytes": len(body),
            "outcome": art.get("outcome"),
            "s_v3_sha256": DISP._canon(
                ACC.authoritative_static_contract(real, lane, ck,
                                                  day)),
            "t_v3_sha256": "a" * 64,
            "s_v4_sha256": DISP._canon(s4)}}
        fx = DISP.build_fixture_capsule(
            fauth, [], store, os.path.join(tmp, "a2.json"),
            old_authority=old_auth, reuse=lineage)
        try:
            out = DISP.verify_fixture_lineage_registry(fx, fauth,
                                                       store)
            ok3 = (out["bodies_recomputed"] == 1
                   and out["claim_scope"] == "FIXTURE_ONLY"
                   and out["admission_eligible"] is False
                   and out["authorizes"] == "NOTHING")
            check("fixture WITH one recomputed lineage body is "
                  "ACCEPTED and stamped FIXTURE_ONLY", ok3, str(out))
        except DISP.DispositionRefusal as e:
            check("fixture WITH one recomputed lineage body is "
                  "ACCEPTED and stamped FIXTURE_ONLY", False,
                  str(e)[:150])

        # (1) production verifiers refuse the FIXTURE schema
        check("PRODUCTION ceiling refuses the fixture schema",
              _refuses(lambda: DISP.verify_ceiling(fx, fauth),
                       "not the registered schema"))
        check("PRODUCTION lineage registry refuses the fixture "
              "schema",
              _refuses(lambda: DISP.verify_lineage_registry(
                  fx, fauth, store_root=store),
                  "not the registered schema"))

        # (2) the FIXTURE verifier refuses the PRODUCTION schema
        prod_shaped = dict(fx, schema=DISP.CAPSULE_SCHEMA)
        check("FIXTURE verifier refuses the production schema",
              _refuses(lambda: DISP.verify_fixture_lineage_registry(
                  prod_shaped, fauth, store),
                  "not the registered schema"))

        # the mint guard: a fixture may never be built over the REAL
        # registered authority
        check("build_fixture_capsule refuses the REAL authority",
              _refuses(lambda: DISP.build_fixture_capsule(
                  real, [v4key], store,
                  os.path.join(tmp, "a3.json")),
                  "REFUSES the REAL registered authority"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


main()
print()
if FAILS:
    print(f"W2 FIXTURE-SCHEMA FAILURES ({len(FAILS)}): {FAILS}")
    sys.exit(1)
print("ALL W2 FIXTURE-SCHEMA RED-KATs PASS")
