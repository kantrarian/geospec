#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R6 §1 PUBLICATION-RECEIPT red-KATs — REV 3 (cayley, 2026-08-07) under codex WORKS-WITH-FIX `d76842a`.

REV 3 repairs (codex rev-2 contract review, five findings):
  #1 BLOCKER  _loaders sentinel: `record=None` now means EXPLICITLY ABSENT (raises), the omitted arg
      means the valid default — P2R-3a and P2R-4a are jointly satisfiable (codex's attached diff, verbatim).
  #3 HIGH     day binding by the CARRIER, not the mutable field: the v2 artifact policy names MANDATORY
      alarm carriers (docs/ensemble_latest.json + docs/data.csv) inside an exact ALLOWLIST; admission
      re-parses the REOPENED ensemble bytes and requires ensemble["date"] == receipt.day == requested day.
      New negatives: edited receipt.day over day-D bytes (P2R-7a), omitted mandatory carrier (P2R-7b/3g),
      unknown-but-loadable extra path (P2R-7c/3h).
  #5 HIGH     sealed proof capsule: direct VerifiedReceipt construction is rejected (P2R-8a); exact
      receipt/deployment keysets (P2R-8b/1h); receipt-side availability_utc mutated earlier fails against
      the intact record (P2R-8c); shape + timestamp-chain contract created_at <= updated_at ==
      availability_utc == reopened.updated_at, lowercase-hex, pinned repo URL (P2R-8d/1h).

INTERFACE (grassmann implements src/publication_receipt.py REV 3 to THIS, unedited — the decouple)
---------------------------------------------------------------------------------------------------
* SCHEMA = "geospec-publication-receipt-v2"                     (schema id unchanged; policy tightened)
* MANDATORY_ARTIFACTS = ("docs/ensemble_latest.json", "docs/data.csv")   # alarm carrier + scoring carrier
* ARTIFACT_ALLOWLIST  = MANDATORY_ARTIFACTS + ("docs/validated_events.json",
                        "docs/r4_prospective_record.json", "docs/r5_daily.json")
* build_publication_receipt(day, artifact_paths, commit_sha, deployment) -> dict
    - day "YYYY-MM-DD" (parseable, else ValueError); commit_sha lowercase 40-hex (else ValueError);
    - artifact_paths: MANDATORY_ARTIFACTS all present; every key in ARTIFACT_ALLOWLIST; the ensemble
      payload must parse as JSON with ["date"] == day (the carrier binds the day at build);
    - deployment EXACT keyset {id, api_url, status, error, created_at, updated_at, source}: status=="built",
      error in (None, ""), created_at <= updated_at (parseable UTC), source in the server allowlist,
      api_url == f"https://api.github.com/repos/kantrarian/geospec/pages/builds/{id}" (pinned repo/shape);
      anything missing/extra/mismatched => ValueError (fail closed — no synthetic deployments, no fallbacks);
    - receipt EXACT keyset {schema, day, artifact_hashes, commit_sha, deployment, availability_utc,
      built_utc}; availability_utc == deployment.updated_at (the COMPLETION stamp).
* verify_publication_receipt(receipt, artifact_bytes) -> True | raise      (byte binding, as rev 1)
* class VerifiedReceipt — typed result (day, availability_utc, receipt) SEALED behind admission: direct
    construction raises (module-private minting token/factory); only admit_receipt mints instances.
* admit_receipt(receipt, day, artifact_loader, server_record_loader) -> VerifiedReceipt | raise
    - artifact_loader(commit_sha, relpath) -> bytes    (production: git cat-file blob <commit>:<relpath>)
    - server_record_loader(api_url) -> dict            (production: gh api <url>)
    - checks, ALL fail-closed: schema v2; EXACT receipt keyset (no extra/missing top-level fields);
      receipt.day == requested day; artifact_hashes: mandatory carriers present, no key outside the
      allowlist, every value lowercase 64-hex, EVERY recorded artifact re-hashed from loader bytes
      (missing/extra/unloadable => raise); the reopened ensemble bytes parse with ["date"] ==
      receipt.day == requested day (the CARRIER binds the day — a mutable receipt field never does);
      server record reopened at receipt.deployment.api_url and matched by the LIVE Pages shape:
      record.url == receipt.deployment.api_url AND the numeric build id parsed from that URL ==
      receipt.deployment.id (NEVER a record.id field — the live record has none); error-free IFF
      record.error is absent/None/"" or a dict whose message is None/empty (a nonempty message or
      any other carrier shape rejects); status built; commit == receipt.commit_sha;
      record.created_at <= record.updated_at == receipt.availability_utc
      == receipt.deployment.updated_at; availability parseable.
* day_eligible_for_hit(x) -> bool   — True IFF x is an ADMISSION-minted VerifiedReceipt. Every dict is
    False; a directly-constructed instance is impossible (construction raises) or rejected.
* alarm_available_at_utc(day, verified: VerifiedReceipt | None) -> str
    - VerifiedReceipt => its availability_utc EXACTLY; None => f"{day}T23:59:59Z"; never anything else.

REV 3.1 (codex 2257 WORKS-WITH-ONE-LIVE-SHAPE-FIX): the frozen fixtures synthesized a server
record with `id="..."` and `error=""`, but the LIVE Pages API returns **no id field** (the build
id exists only in the URL) and `error={"message": null}` on success. Fixtures now carry the REAL
shape. Admission contract amended (the ONLY reopened-record change): bind server identity by
`record.url == receipt.deployment.api_url` AND the numeric build id parsed from that pinned URL
== receipt.deployment.id — never require a `record.id` field; the record is error-free IFF its
`error` is absent/None/"" OR a dict whose `message` is None/empty — a nonempty message or any
other error-carrier shape rejects. NO TEST MAY ADD AN `id` FIELD to a reopened Pages fixture.

RED AS AUTHORED against `aaea74d` at the live-shape KATs (the current admission requires
`record.id` and treats the live error dict as an error); the rest of the suite stays green.
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
REL_ENS = "docs/ensemble_latest.json"
REL_CSV = "docs/data.csv"
# The canonical alarm carrier: field name matches the REAL published artifact ("date").
PAYLOAD_ENS = json.dumps({"date": DAY, "regions": {}}).encode()
PAYLOAD_CSV = b"date,region,tier,risk,confidence,methods,agreement\n2026-08-05,kumamoto,2,0.61,0.8,4,0.75\n"
_EXTRA_REL = "docs/unlisted_extra.json"
_EXTRA_BYTES = b"{\"anything\": true}"

# codex fix #1, verbatim: distinguish OMITTED (valid default) from EXPLICITLY ABSENT (None).
_DEFAULT_RECORD = object()


def _live_record():
    """The REAL GitHub Pages build-record shape (codex live reproduction 2257): NO id field —
    the build id exists only in the URL — and success carries error={"message": None}."""
    return {"url": API_URL, "status": "built", "error": {"message": None}, "commit": COMMIT,
            "created_at": DEP["created_at"], "updated_at": DEP["updated_at"]}


def _loaders(ens=PAYLOAD_ENS, csvb=PAYLOAD_CSV, record=_DEFAULT_RECORD, extra_paths=None):
    """Deterministic injected evidence: git-object bytes + the reopened server record.
    `_loaders()` = the valid case; `_loaders(record=None)` = the absent-record negative."""
    rec = _live_record() if record is _DEFAULT_RECORD else record
    if isinstance(rec, dict):
        assert "id" not in rec, "no test may add an `id` field to a reopened Pages fixture"
    blobs = {}
    if ens is not None:
        blobs[REL_ENS] = ens
    if csvb is not None:
        blobs[REL_CSV] = csvb
    blobs.update(extra_paths or {})

    def artifact_loader(commit_sha, relpath):
        if commit_sha == COMMIT and relpath in blobs:
            return blobs[relpath]
        raise ValueError(f"no blob {commit_sha[:8]}:{relpath}")

    def server_record_loader(api_url):
        if api_url == API_URL and rec is not None:
            return dict(rec)
        raise ValueError(f"no server record at {api_url}")

    return artifact_loader, server_record_loader


def _write_paths(td, ens=PAYLOAD_ENS, csvb=PAYLOAD_CSV, extra=None):
    paths = {}
    for rel, data in [(REL_ENS, ens), (REL_CSV, csvb)] + list((extra or {}).items()):
        if data is None:
            continue
        tmp = os.path.join(td, rel.replace("/", "__"))
        with open(tmp, "wb") as fh:
            fh.write(data)
        paths[rel] = tmp
    return paths


def main():
    try:
        import publication_receipt as PR
    except ImportError:
        check("P2R-0 module import", False, "src/publication_receipt.py missing")
        return
    needed = ("build_publication_receipt", "verify_publication_receipt", "admit_receipt",
              "VerifiedReceipt", "day_eligible_for_hit", "alarm_available_at_utc",
              "MANDATORY_ARTIFACTS", "ARTIFACT_ALLOWLIST")
    if not all(hasattr(PR, n) for n in needed) or getattr(PR, "SCHEMA", "") != "geospec-publication-receipt-v2":
        check("P2R-0 rev-3 interface present (admission + carrier policy + schema v2)",
              False, "AWAITING grassmann's rev-3 -- red-first as authored")
        return
    check("P2R-0p carrier policy constants",
          tuple(PR.MANDATORY_ARTIFACTS) == (REL_ENS, REL_CSV)
          and set(PR.MANDATORY_ARTIFACTS) <= set(PR.ARTIFACT_ALLOWLIST))

    with tempfile.TemporaryDirectory() as td:
        paths = _write_paths(td)
        rc = PR.build_publication_receipt(DAY, paths, COMMIT, dict(DEP))

        # -- build-time contract --
        check("P2R-1a schema v2 + day bound + availability == COMPLETION stamp (updated_at, NOT created_at)",
              rc["schema"] == "geospec-publication-receipt-v2" and rc["day"] == DAY
              and rc["availability_utc"] == DEP["updated_at"]
              and rc["artifact_hashes"][REL_ENS] == hashlib.sha256(PAYLOAD_ENS).hexdigest()
              and rc["artifact_hashes"][REL_CSV] == hashlib.sha256(PAYLOAD_CSV).hexdigest())
        check("P2R-1b non-40hex / non-lowercase commit refuses to build",
              raises(lambda: PR.build_publication_receipt(DAY, paths, "abc123", dict(DEP)))
              and raises(lambda: PR.build_publication_receipt(DAY, paths, COMMIT.upper(), dict(DEP))))
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
        check("P2R-1f mandatory-carrier policy at build: omit ensemble OR data.csv refuses; unknown key refuses",
              raises(lambda: PR.build_publication_receipt(
                  DAY, {k: v for k, v in paths.items() if k != REL_ENS}, COMMIT, dict(DEP)))
              and raises(lambda: PR.build_publication_receipt(
                  DAY, {k: v for k, v in paths.items() if k != REL_CSV}, COMMIT, dict(DEP)))
              and raises(lambda: PR.build_publication_receipt(
                  DAY, {**paths, **_write_paths(td, ens=None, csvb=None, extra={_EXTRA_REL: _EXTRA_BYTES})},
                  COMMIT, dict(DEP))))
        wrong_day_ens = json.dumps({"date": "2026-08-04", "regions": {}}).encode()
        check("P2R-1g carrier-day mismatch at build: ensemble['date'] != day refuses",
              raises(lambda: PR.build_publication_receipt(
                  DAY, _write_paths(td, ens=wrong_day_ens), COMMIT, dict(DEP))))
        check("P2R-1h exact deployment keyset + pinned repo URL + timestamp order at build",
              raises(lambda: PR.build_publication_receipt(DAY, paths, COMMIT, {**DEP, "extra_field": 1}))
              and raises(lambda: PR.build_publication_receipt(
                  DAY, paths, COMMIT,
                  {**DEP, "api_url": "https://api.github.com/repos/evil/geospec/pages/builds/1137391428"}))
              and raises(lambda: PR.build_publication_receipt(
                  DAY, paths, COMMIT,
                  {**DEP, "api_url": f"https://api.github.com/repos/kantrarian/geospec/pages/builds/999"}))
              and raises(lambda: PR.build_publication_receipt(
                  DAY, paths, COMMIT, {**DEP, "created_at": "2026-08-07T12:00:00Z"})))  # created > updated

        # -- byte verifier (rev-1 role intact) --
        check("P2R-2 verify passes on exact bytes; mutation fails",
              PR.verify_publication_receipt(rc, {REL_ENS: PAYLOAD_ENS, REL_CSV: PAYLOAD_CSV}) is True
              and raises(lambda: PR.verify_publication_receipt(
                  rc, {REL_ENS: PAYLOAD_ENS + b" ", REL_CSV: PAYLOAD_CSV})))

        # -- ADMISSION (the standing-bearing step) --
        al, sl = _loaders()
        try:
            vr = PR.admit_receipt(rc, DAY, al, sl)
        except Exception as exc:
            check("P2R-9a LIVE-SHAPE GATE: the real Pages record admits (NO id field; "
                  "error={'message': None}; identity = record.url + id parsed from it)",
                  False, f"{type(exc).__name__}: {exc} -- AWAITING the narrow live-shape fix "
                         "(red-first as authored)")
            return
        check("P2R-9a LIVE-SHAPE GATE: the real Pages record admits (NO id field; "
              "error={'message': None}; identity = record.url + id parsed from it)", True)
        check("P2R-3a valid receipt + loaders admit to a VerifiedReceipt with the completion stamp",
              isinstance(vr, PR.VerifiedReceipt) and vr.availability_utc == DEP["updated_at"]
              and vr.day == DAY)

        # codex B1 composed bypass, verbatim: hash-invalid receipt must fail admission AND eligibility
        forged = copy.deepcopy(rc)
        forged["artifact_hashes"][REL_ENS] = "0" * 64
        check("P2R-3b codex-B1 composed bypass CLOSED: hash-invalid receipt fails admission",
              raises(lambda: PR.admit_receipt(forged, DAY, al, sl)))
        check("P2R-3c ...and day_eligible_for_hit(dict) is False for ANY dict (valid or forged)",
              PR.day_eligible_for_hit(forged) is False and PR.day_eligible_for_hit(rc) is False
              and PR.day_eligible_for_hit(vr) is True)

        rand = copy.deepcopy(rc)
        rand["artifact_hashes"][REL_ENS] = hashlib.sha256(b"other").hexdigest()
        check("P2R-3d random (nonzero) wrong hash fails admission",
              raises(lambda: PR.admit_receipt(rand, DAY, al, sl)))
        missing = copy.deepcopy(rc)
        missing["artifact_hashes"]["docs/r5_daily.json"] = hashlib.sha256(b"x").hexdigest()
        check("P2R-3e recorded artifact with no loadable bytes fails admission",
              raises(lambda: PR.admit_receipt(missing, DAY, al, sl)))
        empty = copy.deepcopy(rc)
        empty["artifact_hashes"] = {}
        check("P2R-3f empty artifact set fails admission (a receipt must attest something)",
              raises(lambda: PR.admit_receipt(empty, DAY, al, sl)))
        no_carrier = copy.deepcopy(rc)
        del no_carrier["artifact_hashes"][REL_ENS]
        no_csv = copy.deepcopy(rc)
        del no_csv["artifact_hashes"][REL_CSV]
        check("P2R-3g omitting a MANDATORY carrier fails admission (ensemble; data.csv)",
              raises(lambda: PR.admit_receipt(no_carrier, DAY, al, sl))
              and raises(lambda: PR.admit_receipt(no_csv, DAY, al, sl)))
        al_extra, sl_extra = _loaders(extra_paths={_EXTRA_REL: _EXTRA_BYTES})
        unknown = copy.deepcopy(rc)
        unknown["artifact_hashes"][_EXTRA_REL] = hashlib.sha256(_EXTRA_BYTES).hexdigest()
        check("P2R-3h unknown-but-LOADABLE (hash-valid) extra path fails admission (allowlist, not loadability)",
              raises(lambda: PR.admit_receipt(unknown, DAY, al_extra, sl_extra)))

        # codex B2: server-record reopening — relabelled client dicts + mismatches fail
        check("P2R-4a codex-B2 relabel attack CLOSED: no server record at the named URL -> no admission",
              raises(lambda: PR.admit_receipt(rc, DAY, al, _loaders(record=None)[1])))
        for field, val in (("status", "building"), ("error", {"message": "failed"}),
                           ("commit", "f" * 40), ("updated_at", "2026-08-07T99:99:99Z"),
                           ("url", "https://api.github.com/repos/kantrarian/geospec/pages/builds/999"),
                           ("created_at", "2026-08-07T12:00:00Z")):   # record's created > updated
            bad_rec = _live_record()
            bad_rec[field] = val
            check(f"P2R-4b server-record mismatch on {field} fails admission",
                  raises(lambda r=bad_rec: PR.admit_receipt(rc, DAY, al, _loaders(record=r)[1])))

        # LIVE PAGES SHAPE (codex 2257 attached repair; probe parity)
        no_url = _live_record()
        del no_url["url"]
        check("P2R-9b record MISSING url rejects (identity has nothing to bind)",
              raises(lambda: PR.admit_receipt(rc, DAY, al, _loaders(record=no_url)[1])))
        for desc, err_val, want_admit in (
                ("error absent", "__ABSENT__", True),
                ("error None", None, True),
                ("error ''", "", True),
                ("error {'message': ''}", {"message": ""}, True),
                ("error {'message': 'boom'}", {"message": "boom"}, False),
                ("error 'boom' (nonempty string)", "boom", False),
                ("error 5 (malformed carrier)", 5, False)):
            r9 = _live_record()
            if err_val == "__ABSENT__":
                del r9["error"]
            else:
                r9["error"] = err_val
            if want_admit:
                ok9 = isinstance(PR.admit_receipt(rc, DAY, al, _loaders(record=r9)[1]),
                                 PR.VerifiedReceipt)
            else:
                ok9 = raises(lambda rr=r9: PR.admit_receipt(rc, DAY, al, _loaders(record=rr)[1]))
            check(f"P2R-9c live error-shape rule: {desc} -> "
                  f"{'admits' if want_admit else 'rejects'}", ok9)

        # day binding: transplant detection — BOTH forms
        check("P2R-5 day-transplant CLOSED: admitting a day-D receipt for day E fails",
              raises(lambda: PR.admit_receipt(rc, "2026-08-06", al, sl)))
        stronger = copy.deepcopy(rc)
        stronger["day"] = "2026-08-06"        # codex #3: mutate the receipt field to MATCH the request;
        check("P2R-7a codex stronger transplant CLOSED: edited receipt.day over day-D bytes fails admission "
              "(the reopened carrier date binds the day, not the mutable field)",
              raises(lambda: PR.admit_receipt(stronger, "2026-08-06", al, sl)))

        # sealed capsule (codex #5)
        sealed_ok = False
        try:
            obj = PR.VerifiedReceipt(DAY, DEP["updated_at"], copy.deepcopy(rc))   # forged direct mint
        except Exception:
            sealed_ok = True                   # construction refused: sealed
        else:
            sealed_ok = PR.day_eligible_for_hit(obj) is False   # or the unminted instance is rejected
        check("P2R-8a direct VerifiedReceipt construction cannot confer standing (sealed minting)", sealed_ok)
        extra_top = copy.deepcopy(rc)
        extra_top["grants_standing"] = True
        check("P2R-8b extra top-level receipt field fails admission (exact keyset)",
              raises(lambda: PR.admit_receipt(extra_top, DAY, al, sl)))
        early = copy.deepcopy(rc)
        early["availability_utc"] = "2026-08-07T11:08:05Z"      # rolled back to build START; record intact
        check("P2R-8c receipt-side availability_utc mutated EARLIER fails against the intact record",
              raises(lambda: PR.admit_receipt(early, DAY, al, sl)))
        bad_shape = copy.deepcopy(rc)
        bad_shape["artifact_hashes"][REL_ENS] = ("0" * 63) + "G"   # not lowercase 64-hex
        bad_day = copy.deepcopy(rc)
        bad_day["day"] = "08/05/2026"
        check("P2R-8d shape contract: non-64hex hash and unparseable day fail admission",
              raises(lambda: PR.admit_receipt(bad_shape, DAY, al, sl))
              and raises(lambda: PR.admit_receipt(bad_day, "08/05/2026", al, sl)))

        # availability semantics
        check("P2R-6a with VerifiedReceipt: availability == the completion stamp exactly",
              PR.alarm_available_at_utc(DAY, vr) == DEP["updated_at"])
        check("P2R-6b without: the ceiling, never earlier",
              PR.alarm_available_at_utc(DAY, None) == f"{DAY}T23:59:59Z")


main()
print()
if FAILS:
    print(f"R6-S1 PUBLICATION-RECEIPT REV-3 RED-KAT FAILURES: {FAILS}")
    sys.exit(1)
print("ALL R6-S1 PUBLICATION-RECEIPT REV-3 RED-KATs PASS (carrier-bound day + sealed admission standing)")
