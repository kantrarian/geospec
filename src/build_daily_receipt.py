#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the daily SERVER-stamped publication receipt from the GitHub Pages build record (R6 §1 wiring, P2 item 1).

Invoked by run_and_publish.ps1 POST-PUSH. Consumes `publication_receipt.build_publication_receipt`.

Honest timing model: GitHub Pages builds the site a few minutes AFTER the daily push, so the currently-"built"
pages build usually corresponds to a PRIOR daily commit. Each run therefore writes the receipt for whatever daily
commit Pages has *actually built* (read from the build's commit subject), hashing that commit's published docs —
so day D's receipt naturally lands on a later run once Pages has genuinely deployed day D. This is cayley's
"committed on the next run is fine", done without ever backfilling a fake receipt.

FAIL-OPEN + NEVER-BACKFILL: any error, a non-built pages build, a non-daily commit, or an already-present receipt
=> no write. A receipt-less day degrades conservatively (23:59:59Z ceiling, hit-ineligible) and self-heals when a
real built pages build for that commit exists. A receipt is written ONLY from a real, built, server-side pages
build for a real daily-monitoring commit.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)                       # repo-root src/ -> import the R6 §1 module directly
import publication_receipt as PR                   # noqa: E402

OWNER_REPO = "kantrarian/geospec"
RECEIPTS_DIR = os.path.join(REPO, "monitoring", "receipts")
# The published artifact set the daily push git-adds to docs/ (an artifact absent at a given commit is skipped).
ARTIFACTS = ["docs/ensemble_latest.json", "docs/data.csv", "docs/validated_events.json",
             "docs/r4_prospective_record.json", "docs/r5_daily.json"]
_DAILY_RE = re.compile(r"^Daily monitoring (\d{4}-\d{2}-\d{2})")


def _run(args, **kw):
    return subprocess.run(args, capture_output=True, timeout=90, **kw)


def gh_pages_latest():
    """The latest GitHub Pages build record (server-side), or raise."""
    out = _run(["gh", "api", f"repos/{OWNER_REPO}/pages/builds/latest"], text=True)
    if out.returncode != 0:
        raise RuntimeError(f"gh api pages/builds/latest failed: {out.stderr.strip()[:200]}")
    return json.loads(out.stdout)


def _blob_at(commit, relpath):
    """Raw bytes of `relpath` as committed at `commit`, or None if absent there."""
    out = _run(["git", "-C", REPO, "cat-file", "blob", f"{commit}:{relpath}"])
    return out.stdout if out.returncode == 0 else None


def build_receipt_for_pages_build(build):
    """Map a Pages build API object to (day, receipt), or (None, None) to skip (fail-open).

    Writes nothing. A receipt is producible only when the build is BUILT + error-free, its commit is a
    daily-monitoring commit, and at least one published artifact exists at that commit; the receipt hashes the
    docs AS COMMITTED AT that commit (not the current tree, which has since advanced)."""
    if build.get("status") != "built" or (build.get("error") or {}).get("message"):
        return None, None
    commit, created_at = build.get("commit"), build.get("created_at")
    if not (isinstance(commit, str) and commit and isinstance(created_at, str) and created_at):
        return None, None
    subj = _run(["git", "-C", REPO, "log", "-1", "--format=%s", commit], text=True)
    if subj.returncode != 0:
        return None, None
    m = _DAILY_RE.match(subj.stdout.strip())
    if not m:
        return None, None                          # not a daily publish commit -> nothing to receipt
    day = m.group(1)
    build_id = str(build.get("url", "")).rstrip("/").split("/")[-1] or commit[:12]
    deployment = {"id": build_id, "created_at": created_at, "source": "github-pages-build"}
    with tempfile.TemporaryDirectory() as td:
        paths, artifact_bytes = {}, {}
        for rel in ARTIFACTS:
            data = _blob_at(commit, rel)
            if data is None:
                continue
            tmp = os.path.join(td, rel.replace("/", "__"))
            with open(tmp, "wb") as fh:
                fh.write(data)
            paths[rel], artifact_bytes[rel] = tmp, data
        if not paths:
            return None, None
        receipt = PR.build_publication_receipt(paths, commit, deployment)
        # self-check: the receipt must round-trip + be a valid server receipt before we ever write it
        PR.verify_publication_receipt(receipt, artifact_bytes)
        if not PR.day_eligible_for_hit({"publication_receipt": receipt}):
            return None, None
    return day, receipt


def main():
    try:
        day, receipt = build_receipt_for_pages_build(gh_pages_latest())
    except Exception as exc:                        # fail-open: never break the daily publish
        print(f"[receipt] fail-open ({type(exc).__name__}: {exc}) — no receipt this run", flush=True)
        return 0
    if not (day and receipt):
        print("[receipt] no built daily pages-build to receipt this run (conservative; self-heals)", flush=True)
        return 0
    os.makedirs(RECEIPTS_DIR, exist_ok=True)
    dst = os.path.join(RECEIPTS_DIR, f"{day}.json")
    if os.path.exists(dst):
        print(f"[receipt] {day} already receipted; leaving it (never overwrite/backfill)", flush=True)
        return 0
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"[receipt] wrote monitoring/receipts/{day}.json — server stamp "
          f"{receipt['deployment']['created_at']} (pages build {receipt['deployment']['id']})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
