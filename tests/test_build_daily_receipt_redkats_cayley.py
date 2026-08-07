#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PRODUCER red-KATs — build_daily_receipt.py REV 2 (cayley, 2026-08-07) under codex WORKS-WITH-FIX
`d76842a` finding #4: "the promised producer repair has no red bar."

The landed producer still (a) falls back to `commit[:12]` when the Pages URL is absent, (b) treats any
existing pathname as permanently final without validation, (c) writes the destination directly
(non-atomic), (d) records only `created_at`, and (e) self-checks via the rev-1 dict path. This bar
freezes the repaired contract BEFORE implementation.

CONTRACT (grassmann implements src/build_daily_receipt.py REV 2 to THIS, unedited — the decouple)
--------------------------------------------------------------------------------------------------
* build_receipt_for_pages_build(build, *, commit_subject_loader, artifact_loader) -> (day, receipt) | (None, None)
    - build: the `gh api repos/kantrarian/geospec/pages/builds/latest` JSON object;
    - commit_subject_loader(commit_sha) -> str   (production: git log -1 --format=%s <sha>)
    - artifact_loader(commit_sha, relpath) -> bytes | raise   (production: git cat-file blob; the SAME
      loader shape admit_receipt takes — one evidence seam, no parallel path);
    - produces a receipt ONLY when ALL hold, else (None, None) — NEVER a synthetic fallback:
        status == "built"; error message empty/absent; commit is lowercase 40-hex;
        BOTH created_at AND updated_at present/parseable with created_at <= updated_at;
        build["url"] parses as .../repos/kantrarian/geospec/pages/builds/<id> (the pinned repo) —
          id comes ONLY from that URL (no commit[:12], no invented ids);
        the commit subject matches "Daily monitoring YYYY-MM-DD";
        the MANDATORY carriers (publication_receipt.MANDATORY_ARTIFACTS) load at that commit and the
          ensemble payload's ["date"] equals the subject day (subject-day == artifact-day, codex #4);
    - the receipt is built via publication_receipt.build_publication_receipt(day, ..., deployment) with
      deployment {id, api_url, status, error, created_at, updated_at, source="github-pages-build"} —
      availability lands on updated_at (the completion stamp) by the module contract.
* publish_receipt(day, receipt, receipts_dir, *, artifact_loader, server_record_loader) -> str
    - ADMITS the receipt (publication_receipt.admit_receipt with the injected loaders) BEFORE any write;
      a receipt that fails admission is NEVER written (raise or error status — fail closed, codex #4);
    - no existing file  -> ATOMIC publication (temp file in the SAME dir + flush + os.replace; the
      destination NEVER holds partial bytes) -> "written";
    - existing file that ADMITS -> byte-identical no-op -> "valid_existing_noop";
    - existing file that FAILS admission -> SURFACED (status, never silent) and REPAIRED from the same
      independently reopened evidence: the freshly admitted receipt atomically replaces it -> "repaired";
      an invalid existing receipt must neither be accepted NOR block self-heal.
* main() stays fail-open for the DAILY PUBLISH (an exception => no receipt this run, exit 0) — fail-open
  applies to the pipeline, never to evidence: no admission, no write.

REV 2.1 (codex 2257): the server-record fixture now carries the REAL Pages shape (no id field;
error={"message": None}); PB-0d gates the publication path on live-shape admission. Red-first at
PB-0d against `aaea74d`; the build path (already live-shaped) stays green.
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


DAY = "2026-08-05"
COMMIT = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
BUILD_ID = "1137391428"
API_URL = f"https://api.github.com/repos/kantrarian/geospec/pages/builds/{BUILD_ID}"
REL_ENS = "docs/ensemble_latest.json"
REL_CSV = "docs/data.csv"
PAYLOAD_ENS = json.dumps({"date": DAY, "regions": {}}).encode()
PAYLOAD_CSV = b"date,region,tier,risk,confidence,methods,agreement\n2026-08-05,kumamoto,2,0.61,0.8,4,0.75\n"
SUBJECT = f"Daily monitoring {DAY}"

BUILD = {"url": API_URL, "status": "built", "error": {"message": None}, "commit": COMMIT,
         "created_at": "2026-08-07T11:08:05Z", "updated_at": "2026-08-07T11:08:33Z"}


def _subject_loader(subjects=None):
    table = {COMMIT: SUBJECT} if subjects is None else subjects

    def loader(commit_sha):
        if commit_sha in table:
            return table[commit_sha]
        raise ValueError(f"no commit {commit_sha[:8]}")
    return loader


def _artifact_loader(ens=PAYLOAD_ENS, csvb=PAYLOAD_CSV):
    blobs = {REL_ENS: ens, REL_CSV: csvb}

    def loader(commit_sha, relpath):
        if commit_sha == COMMIT and blobs.get(relpath) is not None:
            return blobs[relpath]
        raise ValueError(f"no blob {commit_sha[:8]}:{relpath}")
    return loader


def _server_record_loader():
    # The REAL Pages record shape (codex 2257): NO id field (the id exists only in the URL);
    # success carries error={"message": None}. No test may add an `id` field to this fixture.
    rec = {"url": API_URL, "status": "built", "error": {"message": None}, "commit": COMMIT,
           "created_at": BUILD["created_at"], "updated_at": BUILD["updated_at"]}
    assert "id" not in rec

    def loader(api_url):
        if api_url == API_URL:
            return dict(rec)
        raise ValueError(f"no server record at {api_url}")
    return loader


def main():
    import inspect
    try:
        import publication_receipt as PR
        import build_daily_receipt as BD
    except ImportError as exc:
        check("PB-0 modules import", False, str(exc))
        return
    if getattr(PR, "SCHEMA", "") != "geospec-publication-receipt-v2" or not hasattr(PR, "MANDATORY_ARTIFACTS"):
        check("PB-0 prerequisite: publication_receipt REV 3 present", False, "module rev-3 absent -- red-first")
        return
    try:
        sig_b = set(inspect.signature(BD.build_receipt_for_pages_build).parameters)
        sig_p = set(inspect.signature(BD.publish_receipt).parameters)
        ok_iface = ({"commit_subject_loader", "artifact_loader"} <= sig_b
                    and {"artifact_loader", "server_record_loader"} <= sig_p)
    except Exception:
        ok_iface = False
    if not ok_iface:
        check("PB-0 producer rev-2 seams present (injected loaders on build + publish)",
              False, "AWAITING grassmann's producer rework -- red-first as authored")
        return

    subj, al, sl = _subject_loader(), _artifact_loader(), _server_record_loader()

    # -- production path --
    day, rc = BD.build_receipt_for_pages_build(dict(BUILD), commit_subject_loader=subj, artifact_loader=al)
    check("PB-1 valid built daily build -> receipt for the SUBJECT day with server identity intact",
          day == DAY and isinstance(rc, dict)
          and rc.get("availability_utc") == BUILD["updated_at"]
          and rc.get("deployment", {}).get("api_url") == API_URL
          and rc.get("deployment", {}).get("id") == BUILD_ID
          and rc.get("commit_sha") == COMMIT
          and rc.get("artifact_hashes", {}).get(REL_ENS) == hashlib.sha256(PAYLOAD_ENS).hexdigest())

    def none_for(desc, build_mut=None, **loader_kw):
        b = dict(BUILD)
        if build_mut:
            build_mut(b)
        s = loader_kw.get("subj", subj)
        a = loader_kw.get("al", al)
        try:
            d0, r0 = BD.build_receipt_for_pages_build(b, commit_subject_loader=s, artifact_loader=a)
            check(desc, d0 is None and r0 is None, f"got ({d0!r}, {'receipt' if r0 else None!r})")
        except Exception as exc:
            check(desc, False, f"RAISED {exc} (fail-open contract: produce nothing, do not raise)")

    none_for("PB-2a URL absent -> NO receipt (the commit[:12] fallback is DEAD)",
             build_mut=lambda b: b.pop("url"))
    none_for("PB-2b URL on the wrong repo -> NO receipt (pinned repo shape)",
             build_mut=lambda b: b.__setitem__(
                 "url", f"https://api.github.com/repos/evil/geospec/pages/builds/{BUILD_ID}"))
    none_for("PB-3 non-40hex commit -> NO receipt",
             build_mut=lambda b: b.__setitem__("commit", COMMIT[:12]))
    none_for("PB-4a status != built -> NO receipt", build_mut=lambda b: b.__setitem__("status", "building"))
    none_for("PB-4b errored build -> NO receipt",
             build_mut=lambda b: b.__setitem__("error", {"message": "boom"}))
    none_for("PB-4c missing updated_at -> NO receipt (created_at alone is NOT availability)",
             build_mut=lambda b: b.pop("updated_at"))
    none_for("PB-4d created_at > updated_at -> NO receipt",
             build_mut=lambda b: b.__setitem__("created_at", "2026-08-07T12:00:00Z"))
    none_for("PB-5 subject-day != canonical artifact day -> NO receipt (codex #4 day check)",
             al=_artifact_loader(ens=json.dumps({"date": "2026-08-04", "regions": {}}).encode()))
    none_for("PB-6 non-daily commit subject -> NO receipt",
             subj=_subject_loader({COMMIT: "fix: unrelated maintenance commit"}))
    none_for("PB-6b mandatory carrier unloadable at the commit -> NO receipt",
             al=_artifact_loader(ens=None))

    # -- publication path --
    pkw = dict(artifact_loader=al, server_record_loader=sl)

    # PB-0d LIVE-SHAPE GATE (codex 2257): publish admits against the REAL Pages record shape.
    import publication_receipt as PR
    try:
        PR.admit_receipt(rc, DAY, al, sl)
    except Exception as exc:
        check("PB-0d LIVE-SHAPE GATE: the produced receipt admits against the real Pages record",
              False, f"{type(exc).__name__}: {exc} -- AWAITING the narrow live-shape fix (red-first)")
        return
    check("PB-0d LIVE-SHAPE GATE: the produced receipt admits against the real Pages record", True)

    with tempfile.TemporaryDirectory() as td:
        rdir = os.path.join(td, "receipts")
        status = BD.publish_receipt(DAY, rc, rdir, **pkw)
        dst = os.path.join(rdir, f"{DAY}.json")
        on_disk = json.load(open(dst, encoding="utf-8")) if os.path.exists(dst) else None
        check("PB-7a fresh publish: admitted, written, round-trips",
              status == "written" and on_disk == rc)
        first_bytes = open(dst, "rb").read()

        status2 = BD.publish_receipt(DAY, rc, rdir, **pkw)
        check("PB-8 valid existing receipt: idempotent no-op, byte-identical",
              status2 == "valid_existing_noop" and open(dst, "rb").read() == first_bytes)

        # tamper the published file -> surfaced + repaired from reopened evidence
        broken = dict(json.loads(first_bytes))
        broken["artifact_hashes"] = dict(broken["artifact_hashes"])
        broken["artifact_hashes"][REL_ENS] = "0" * 64
        with open(dst, "w", encoding="utf-8") as fh:
            json.dump(broken, fh)
        status3 = BD.publish_receipt(DAY, rc, rdir, **pkw)
        repaired = json.load(open(dst, encoding="utf-8"))
        check("PB-9 invalid existing receipt: SURFACED (status 'repaired', never silent) + healed to "
              "admitted bytes", status3 == "repaired" and repaired == rc)

    # atomicity: if os.replace is blocked mid-publish, the destination must not exist / hold partial bytes
    with tempfile.TemporaryDirectory() as td2:
        rdir2 = os.path.join(td2, "receipts")
        real_replace = os.replace

        def exploding_replace(src, dstp):
            raise OSError("simulated crash at the atomic boundary")
        os.replace = exploding_replace
        try:
            try:
                BD.publish_receipt(DAY, rc, rdir2, **pkw)
            except Exception:
                pass                                        # raising here is acceptable; writing is not
        finally:
            os.replace = real_replace
        dst2 = os.path.join(rdir2, f"{DAY}.json")
        check("PB-7b ATOMIC publication: blocked os.replace leaves NO destination file (no direct writes)",
              not os.path.exists(dst2))

    # fail-closed evidence gate on publish: a receipt that cannot admit is never written
    with tempfile.TemporaryDirectory() as td3:
        rdir3 = os.path.join(td3, "receipts")
        forged = dict(rc)
        forged["artifact_hashes"] = dict(rc["artifact_hashes"])
        forged["artifact_hashes"][REL_ENS] = "9" * 64
        wrote = False
        try:
            st = BD.publish_receipt(DAY, forged, rdir3, **pkw)
            wrote = (st == "written")
        except Exception:
            pass                                            # raising is a valid refusal
        dst3 = os.path.join(rdir3, f"{DAY}.json")
        check("PB-10 publish refuses a receipt that fails admission (no write, fail closed)",
              not wrote and not os.path.exists(dst3))


main()
print()
if FAILS:
    print(f"PRODUCER REV-2 RED-KAT FAILURES: {FAILS}")
    sys.exit(1)
print("ALL PRODUCER REV-2 RED-KATs PASS (server-identity + atomic + admission-gated publication)")
