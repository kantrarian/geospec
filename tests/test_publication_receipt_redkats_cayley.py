#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R6 §1 PUBLICATION-RECEIPT red-KATs (cayley, 2026-08-07) — GeoSpec Forward Plan P2 item 1 (receipt-first).

THE CONTRACT UNDER TEST (the registered R5-prerequisites correction to R6 §1)
-----------------------------------------------------------------------------
The hit-clock must be SERVER-stamped: a durable receipt binding {alarm artifact hashes, commit SHA, deployment id,
server created_at}. A git commit timestamp is CLIENT-controlled and insufficient. Days without a receipt are
INELIGIBLE for hit credit, and an earlier availability must NEVER be synthesized (absence degrades conservatively
to 23:59:59Z of day D, hit-ineligible — never to any earlier time).

INTERFACE (grassmann implements `src/publication_receipt.py` to THIS; the decouple — do not edit this bar)
----------------------------------------------------------------------------------------------------------
* SCHEMA = "geospec-publication-receipt-v1"
* build_publication_receipt(artifact_paths: dict[str, str], commit_sha: str, deployment: dict) -> dict
    - artifact_paths: {repo_relpath: abs_path}; every file hashed sha256 into receipt["artifact_hashes"].
    - deployment MUST be a SERVER-side record: requires non-empty `id`, parseable UTC `created_at`, and a
      `source` naming the server API (e.g. "github-pages-build", "github-actions-run"). A deployment whose
      source is client-side ("git-commit-timestamp", "local-clock", missing) => raise ValueError (refuse to
      build — a client-stamped receipt is worse than none because it LOOKS durable).
    - receipt fields: schema, artifact_hashes, commit_sha, deployment{id, created_at, source}, built_utc.
* verify_publication_receipt(receipt: dict, artifact_bytes: dict[str, bytes]) -> True | raise ValueError
    - recomputes every artifact hash from bytes; any mismatch/missing artifact/extra hash => raise;
      schema + required fields + parseable server created_at enforced.
* alarm_available_at_utc(day_iso: str, receipt: dict | None) -> str
    - receipt present+valid => EXACTLY deployment.created_at (the server stamp; never adjusted earlier).
    - receipt None => f"{day_iso}T23:59:59Z" (conservative ceiling), NEVER any earlier value.
* day_eligible_for_hit(day_record: dict) -> bool
    - True IFF day_record["publication_receipt"] is a schema-valid receipt with a server-side source.
      (The R4 scorer consumes this: receipt-less days may still be scored for FALSE-ALARM accounting but can
      never earn HIT credit.)

RED AS AUTHORED: src/publication_receipt.py does not exist. Goes green on grassmann's implementation, unedited.
Wiring (his lane, separate from this bar): run_and_publish queries the GitHub Pages build API post-push
(`gh api repos/{owner}/{repo}/pages/builds/latest`) and writes the receipt to monitoring/receipts/<day>.json;
the R4 prospective scorer consults day_eligible_for_hit.
"""
import hashlib
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
for p in (SRC, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

FAILS = []


def check(desc, ok, detail=""):
    print(f"    [{'PASS' if ok else 'FAIL'}] {desc}" + (f" - {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(desc)


def raises(fn, exc=ValueError):
    try:
        fn()
        return False
    except exc:
        return True


COMMIT = "a" * 40
SERVER_DEP = {"id": "build-20260807-01", "created_at": "2026-08-07T11:09:03Z", "source": "github-pages-build"}


def main():
    try:
        import publication_receipt as PR
    except ImportError:
        check("R6R-0 interface present: src/publication_receipt.py implements the contract",
              False, "AWAITING grassmann's implementation -- red-first as authored")
        return

    with tempfile.TemporaryDirectory() as td:
        art = os.path.join(td, "ensemble_latest.json")
        payload = json.dumps({"regions": {"kumamoto": {"risk": 0.31, "tier": 1}},
                              "generated": "2026-08-07"}).encode()
        with open(art, "wb") as fh:
            fh.write(payload)
        paths = {"docs/ensemble_latest.json": art}

        # R6R-1 build + verify round-trip on real bytes
        rc = PR.build_publication_receipt(paths, COMMIT, dict(SERVER_DEP))
        check("R6R-1a receipt schema + fields",
              rc.get("schema") == "geospec-publication-receipt-v1" and rc.get("commit_sha") == COMMIT
              and rc["deployment"]["created_at"] == SERVER_DEP["created_at"]
              and rc["artifact_hashes"]["docs/ensemble_latest.json"] == hashlib.sha256(payload).hexdigest())
        check("R6R-1b verify passes on the exact bytes",
              PR.verify_publication_receipt(rc, {"docs/ensemble_latest.json": payload}) is True)

        # R6R-2 any byte mutation breaks verification loudly
        check("R6R-2 mutated artifact bytes fail verification",
              raises(lambda: PR.verify_publication_receipt(rc, {"docs/ensemble_latest.json": payload + b" "})))

        # R6R-3 server timestamp required to BUILD
        for bad in ({**SERVER_DEP, "created_at": ""}, {**SERVER_DEP, "created_at": "not-a-time"},
                    {k: v for k, v in SERVER_DEP.items() if k != "created_at"}):
            pass
        check("R6R-3 missing/unparseable server created_at refuses to build",
              raises(lambda: PR.build_publication_receipt(paths, COMMIT, {**SERVER_DEP, "created_at": ""}))
              and raises(lambda: PR.build_publication_receipt(
                  paths, COMMIT, {k: v for k, v in SERVER_DEP.items() if k != "created_at"})))

        # R6R-4 client-side sources are REFUSED (a client-stamped receipt is worse than none)
        for src_name in ("git-commit-timestamp", "local-clock", ""):
            check(f"R6R-4 client-side deployment source {src_name!r} refuses to build",
                  raises(lambda s=src_name: PR.build_publication_receipt(
                      paths, COMMIT, {**SERVER_DEP, "source": s})))
        check("R6R-4d missing source refuses to build",
              raises(lambda: PR.build_publication_receipt(
                  paths, COMMIT, {k: v for k, v in SERVER_DEP.items() if k != "source"})))

        # R6R-5 availability semantics: server stamp EXACTLY, or the conservative ceiling
        check("R6R-5a with receipt: availability == the server stamp exactly",
              PR.alarm_available_at_utc("2026-08-07", rc) == SERVER_DEP["created_at"])
        check("R6R-5b without receipt: availability == 23:59:59Z ceiling (never earlier)",
              PR.alarm_available_at_utc("2026-08-07", None) == "2026-08-07T23:59:59Z")

        # R6R-6 hit eligibility: receipt-bearing day True; receipt-less/tampered day False
        check("R6R-6a receipt-bearing day is hit-eligible",
              PR.day_eligible_for_hit({"publication_receipt": rc}) is True)
        check("R6R-6b receipt-less day is NOT hit-eligible",
              PR.day_eligible_for_hit({}) is False and PR.day_eligible_for_hit({"publication_receipt": None}) is False)
        forged = json.loads(json.dumps(rc))
        forged["deployment"]["source"] = "git-commit-timestamp"
        check("R6R-6c client-source receipt is NOT hit-eligible",
              PR.day_eligible_for_hit({"publication_receipt": forged}) is False)

        # R6R-7 tampered receipt hash fails verify
        forged2 = json.loads(json.dumps(rc))
        forged2["artifact_hashes"]["docs/ensemble_latest.json"] = "0" * 64
        check("R6R-7 tampered artifact hash in receipt fails verification",
              raises(lambda: PR.verify_publication_receipt(forged2, {"docs/ensemble_latest.json": payload})))


main()
print()
if FAILS:
    print(f"R6-S1 PUBLICATION-RECEIPT RED-KAT FAILURES: {FAILS}")
    sys.exit(1)
print("ALL R6-S1 PUBLICATION-RECEIPT RED-KATs PASS (server-stamped hit-clock enforced)")
