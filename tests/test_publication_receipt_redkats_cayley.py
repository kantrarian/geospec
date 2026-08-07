#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R6 §1 PUBLICATION-RECEIPT red-KATs — REV 2 (cayley, 2026-08-07) under codex NO-GO `f5296dc`.

REV 2 (all three codex findings):
  B1  standing-requires-verification: `day_eligible_for_hit` confers eligibility ONLY on a typed
      `VerifiedReceipt` minted by `admit_receipt` after BYTE verification against independently loaded
      artifacts AND server-record reopening. Any bare dict — including one that fails byte verification —
      is False. The exact composed bypass codex reproduced is a named negative here.
  B2  a `source` string is not an attestation: `admit_receipt` reopens the named server record via an injected
      `server_record_loader(api_url)` and matches id / status==built / no error / commit / timestamp.
      Relabelled client dicts fail admission. `commit_sha` must be 40-hex at BUILD.
  HIGH availability = the server COMPLETION stamp (`updated_at` of a built, error-free build), never
      `created_at` — per the dated prospective correction
      docs/CORRECTION_2026-08-07_receipt_availability_completion_stamp.md. Receipts BIND their day
      (transplant-detectable).

INTERFACE (grassmann implements src/publication_receipt.py REV 2 to THIS, unedited — the decouple)
---------------------------------------------------------------------------------------------------
* SCHEMA = "geospec-publication-receipt-v2"
* build_publication_receipt(day, artifact_paths, commit_sha, deployment) -> dict
    - day: "YYYY-MM-DD", bound INSIDE the receipt; commit_sha: full 40-hex (else ValueError);
    - deployment REQUIRES: id, api_url, status=="built", error in (None, ""), created_at, updated_at
      (parseable UTC, updated_at >= created_at), source in the server allowlist; anything else => ValueError
      (fail closed — no synthetic deployments, no commit[:12] fallbacks);
    - receipt: {schema, day, artifact_hashes, commit_sha, deployment{...}, availability_utc == deployment
      updated_at, built_utc}.
* verify_publication_receipt(receipt, artifact_bytes) -> True | raise      (byte binding, as rev 1)
* class VerifiedReceipt  — typed result carrying (day, availability_utc, receipt); constructable in practice
    only via admit_receipt (soft convention; eligibility type-checks it).
* admit_receipt(receipt, day, artifact_loader, server_record_loader) -> VerifiedReceipt | raise
    - artifact_loader(commit_sha, relpath) -> bytes    (production: git cat-file blob <commit>:<relpath>)
    - server_record_loader(api_url) -> dict            (production: gh api <url>)
    - checks: schema v2; receipt.day == day; EVERY recorded artifact re-hashed from loader bytes (missing or
      extra => raise); server record matches (id, status built, error empty, commit == receipt.commit_sha,
      updated_at == receipt.availability_utc); availability parseable.
* day_eligible_for_hit(x) -> bool   — True IFF isinstance(x, VerifiedReceipt). EVERY dict is False.
* alarm_available_at_utc(day, verified: VerifiedReceipt | None) -> str
    - VerifiedReceipt => its availability_utc EXACTLY; None => f"{day}T23:59:59Z"; never anything else.

RED AS AUTHORED (rev-2 interface absent from the landed rev-1 module).
"""
import copy
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


DAY = "2026-08-05"
COMMIT = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
API_URL = "https://api.github.com/repos/kantrarian/geospec/pages/builds/1137391428"
DEP = {"id": "1137391428", "api_url": API_URL, "status": "built", "error": "",
       "created_at": "2026-08-07T11:08:05Z", "updated_at": "2026-08-07T11:08:33Z",
       "source": "github-pages-build"}
REL = "docs/ensemble_latest.json"
PAYLOAD = json.dumps({"day": DAY, "regions": {}}).encode()


def _loaders(payload=PAYLOAD, record=None):
    """Deterministic injected evidence: git-object bytes + the reopened server record."""
    rec = record if record is not None else {"id": DEP["id"], "status": "built", "error": "",
                                             "commit": COMMIT, "created_at": DEP["created_at"],
                                             "updated_at": DEP["updated_at"]}

    def artifact_loader(commit_sha, relpath):
        if commit_sha == COMMIT and relpath == REL and payload is not None:
            return payload
        raise ValueError(f"no blob {commit_sha[:8]}:{relpath}")

    def server_record_loader(api_url):
        if api_url == API_URL and rec is not None:
            return dict(rec)
        raise ValueError(f"no server record at {api_url}")

    return artifact_loader, server_record_loader


def main():
    try:
        import publication_receipt as PR
    except ImportError:
        check("P2R-0 module import", False, "src/publication_receipt.py missing")
        return
    needed = ("build_publication_receipt", "verify_publication_receipt", "admit_receipt",
              "VerifiedReceipt", "day_eligible_for_hit", "alarm_available_at_utc")
    if not all(hasattr(PR, n) for n in needed) or getattr(PR, "SCHEMA", "") != "geospec-publication-receipt-v2":
        check("P2R-0 rev-2 interface present (admit_receipt/VerifiedReceipt/schema v2)",
              False, "AWAITING grassmann's rev-2 -- red-first as authored")
        return

    with tempfile.TemporaryDirectory() as td:
        art = os.path.join(td, "ensemble_latest.json")
        with open(art, "wb") as fh:
            fh.write(PAYLOAD)
        paths = {REL: art}
        rc = PR.build_publication_receipt(DAY, paths, COMMIT, dict(DEP))

        # -- build-time contract --
        check("P2R-1a schema v2 + day bound + availability == COMPLETION stamp (updated_at, NOT created_at)",
              rc["schema"] == "geospec-publication-receipt-v2" and rc["day"] == DAY
              and rc["availability_utc"] == DEP["updated_at"]
              and rc["artifact_hashes"][REL] == hashlib.sha256(PAYLOAD).hexdigest())
        check("P2R-1b non-40hex commit refuses to build",
              raises(lambda: PR.build_publication_receipt(DAY, paths, "abc123", dict(DEP))))
        check("P2R-1c non-built / errored / missing-updated_at deployments refuse to build",
              raises(lambda: PR.build_publication_receipt(DAY, paths, COMMIT, {**DEP, "status": "building"}))
              and raises(lambda: PR.build_publication_receipt(DAY, paths, COMMIT, {**DEP, "error": "boom"}))
              and raises(lambda: PR.build_publication_receipt(
                  DAY, paths, COMMIT, {k: v for k, v in DEP.items() if k != "updated_at"})))
        check("P2R-1d client-side/missing source refuses to build",
              raises(lambda: PR.build_publication_receipt(DAY, paths, COMMIT, {**DEP, "source": "local-clock"}))
              and raises(lambda: PR.build_publication_receipt(
                  DAY, paths, COMMIT, {k: v for k, v in DEP.items() if k != "source"})))
        check("P2R-1e missing api_url refuses to build (fail closed, no synthetic deployment)",
              raises(lambda: PR.build_publication_receipt(
                  DAY, paths, COMMIT, {k: v for k, v in DEP.items() if k != "api_url"})))

        # -- byte verifier (rev-1 role intact) --
        check("P2R-2 verify passes on exact bytes; mutation fails",
              PR.verify_publication_receipt(rc, {REL: PAYLOAD}) is True
              and raises(lambda: PR.verify_publication_receipt(rc, {REL: PAYLOAD + b" "})))

        # -- ADMISSION (the standing-bearing step) --
        al, sl = _loaders()
        vr = PR.admit_receipt(rc, DAY, al, sl)
        check("P2R-3a valid receipt + loaders admit to a VerifiedReceipt with the completion stamp",
              isinstance(vr, PR.VerifiedReceipt) and vr.availability_utc == DEP["updated_at"]
              and vr.day == DAY)

        # codex B1 composed bypass, verbatim: hash-invalid receipt must fail admission AND eligibility
        forged = copy.deepcopy(rc)
        forged["artifact_hashes"][REL] = "0" * 64
        check("P2R-3b codex-B1 composed bypass CLOSED: hash-invalid receipt fails admission",
              raises(lambda: PR.admit_receipt(forged, DAY, al, sl)))
        check("P2R-3c ...and day_eligible_for_hit(dict) is False for ANY dict (valid or forged)",
              PR.day_eligible_for_hit(forged) is False and PR.day_eligible_for_hit(rc) is False
              and PR.day_eligible_for_hit(vr) is True)

        rand = copy.deepcopy(rc)
        rand["artifact_hashes"][REL] = hashlib.sha256(b"other").hexdigest()
        check("P2R-3d random (nonzero) wrong hash fails admission",
              raises(lambda: PR.admit_receipt(rand, DAY, al, sl)))
        missing = copy.deepcopy(rc)
        missing["artifact_hashes"]["docs/absent.json"] = hashlib.sha256(b"x").hexdigest()
        check("P2R-3e recorded artifact with no loadable bytes fails admission",
              raises(lambda: PR.admit_receipt(missing, DAY, al, sl)))
        empty = copy.deepcopy(rc)
        empty["artifact_hashes"] = {}
        check("P2R-3f empty artifact set fails admission (a receipt must attest something)",
              raises(lambda: PR.admit_receipt(empty, DAY, al, sl)))

        # codex B2: server-record reopening — relabelled client dicts + mismatches fail
        check("P2R-4a codex-B2 relabel attack CLOSED: no server record at the named URL -> no admission",
              raises(lambda: PR.admit_receipt(rc, DAY, al, _loaders(record=None)[1])))
        for field, val in (("status", "building"), ("error", "failed"), ("commit", "f" * 40),
                           ("updated_at", "2026-08-07T99:99:99Z"), ("id", "999")):
            bad_rec = {"id": DEP["id"], "status": "built", "error": "", "commit": COMMIT,
                       "created_at": DEP["created_at"], "updated_at": DEP["updated_at"]}
            bad_rec[field] = val
            check(f"P2R-4b server-record mismatch on {field} fails admission",
                  raises(lambda r=bad_rec: PR.admit_receipt(rc, DAY, al, _loaders(record=r)[1])))

        # day binding: transplant detection
        check("P2R-5 day-transplant CLOSED: admitting a day-D receipt for day E fails",
              raises(lambda: PR.admit_receipt(rc, "2026-08-06", al, sl)))

        # availability semantics
        check("P2R-6a with VerifiedReceipt: availability == the completion stamp exactly",
              PR.alarm_available_at_utc(DAY, vr) == DEP["updated_at"])
        check("P2R-6b without: the ceiling, never earlier",
              PR.alarm_available_at_utc(DAY, None) == f"{DAY}T23:59:59Z")


main()
print()
if FAILS:
    print(f"R6-S1 PUBLICATION-RECEIPT REV-2 RED-KAT FAILURES: {FAILS}")
    sys.exit(1)
print("ALL R6-S1 PUBLICATION-RECEIPT REV-2 RED-KATs PASS (verified-standing hit-clock enforced)")
