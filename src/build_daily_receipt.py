#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Daily SERVER-stamped publication-receipt producer — REV 2 (R6 §1, P2 item 1).

Implements the contract fixed by `tests/test_build_daily_receipt_redkats_cayley.py` (cayley, geospec 216780a),
UNEDITED, under codex finding #4. Rev-1 defects repaired: id comes ONLY from the pinned Pages build URL (the
`commit[:12]` fallback is DEAD); availability is the COMPLETION stamp (`updated_at`); the subject day must equal
the reopened canonical artifact day; publication ADMITS the receipt (verify-then-admit) BEFORE any write, writes
ATOMICALLY (temp + fsync + os.replace), and surfaces + repairs an invalid existing file instead of silently
accepting it or blocking self-heal.

Evidence flows through the SAME loader seams `publication_receipt` admits (one path, no parallel logic):
  artifact_loader(commit, relpath) -> bytes    (production: git cat-file blob <commit>:<relpath>)
  server_record_loader(api_url)    -> dict      (production: gh api <url>)
  commit_subject_loader(commit)    -> str       (production: git log -1 --format=%s <commit>)

FAIL-OPEN applies to the daily pipeline (an error => no receipt this run, exit 0), NEVER to evidence
(no admission, no write). NEVER backfills.
"""
import datetime
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import publication_receipt as PR   # noqa: E402  (verify-then-admit; the ONLY standing-bearing path)

RECEIPTS_DIR = os.path.join(REPO, "monitoring", "receipts")
_PAGES_URL_RE = re.compile(r"\Ahttps://api\.github\.com/repos/kantrarian/geospec/pages/builds/(\d+)\Z")
_DAILY_RE = re.compile(r"\ADaily monitoring (\d{4}-\d{2}-\d{2})")
_40HEX = re.compile(r"[0-9a-f]{40}\Z")


def _parse_utc(ts):
    if not isinstance(ts, str) or not ts:
        raise ValueError("timestamp")
    return datetime.datetime.fromisoformat(ts[:-1] + "+00:00" if ts.endswith("Z") else ts)


def build_receipt_for_pages_build(build, *, commit_subject_loader, artifact_loader):
    """Map a Pages build API object to `(day, receipt)`, or `(None, None)` to skip. NEVER raises (fail-open
    pipeline). A receipt is produced ONLY from a built, error-free, daily-monitoring build whose pinned-URL id,
    40-hex commit, ordered completion timestamps, and reopened mandatory carriers (with carrier-day == subject
    day) all hold — no synthetic id, no fallback."""
    try:
        if not isinstance(build, dict) or build.get("status") != "built":
            return None, None
        if (build.get("error") or {}).get("message"):                 # errored build
            return None, None
        commit = build.get("commit")
        if not (isinstance(commit, str) and _40HEX.match(commit)):     # lowercase 40-hex only
            return None, None
        created, updated = build.get("created_at"), build.get("updated_at")
        if not (created and updated):                                  # both required; created_at alone is not it
            return None, None
        if _parse_utc(created) > _parse_utc(updated):                  # ordered completion chain
            return None, None
        m = _PAGES_URL_RE.match(str(build.get("url", "")))             # id ONLY from the pinned repo URL
        if not m:
            return None, None
        build_id = m.group(1)
        dm = _DAILY_RE.match(str(commit_subject_loader(commit)).strip())
        if not dm:                                                     # non-daily commit -> no receipt
            return None, None
        day = dm.group(1)
        td = tempfile.mkdtemp()
        try:
            paths = {}
            for rel in PR.MANDATORY_ARTIFACTS:                         # reopen carriers AT the commit (raise -> None)
                data = artifact_loader(commit, rel)
                tmp = os.path.join(td, rel.replace("/", "__"))
                with open(tmp, "wb") as fh:
                    fh.write(data)
                paths[rel] = tmp
            deployment = {"id": build_id, "api_url": build["url"], "status": "built", "error": "",
                          "created_at": created, "updated_at": updated, "source": "github-pages-build"}
            # build_publication_receipt enforces carrier-day == subject day (codex #4) + completion-stamp availability
            receipt = PR.build_publication_receipt(day, paths, commit, deployment)
            return day, receipt
        finally:
            _rmtree(td)
    except Exception:
        return None, None


def _rmtree(path):
    try:
        import shutil
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def _atomic_write(dst, data):
    """Write `data` bytes to `dst` atomically: temp in the SAME dir + flush + fsync + os.replace. If os.replace
    is blocked, `dst` is never created and the temp is cleaned (no partial destination bytes)."""
    d = os.path.dirname(dst) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".receipt-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dst)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def publish_receipt(day, receipt, receipts_dir, *, artifact_loader, server_record_loader):
    """ADMIT the receipt (verify-then-admit) BEFORE any write, then publish atomically. Returns 'written' (fresh),
    'valid_existing_noop' (existing admits — byte-preserving), or 'repaired' (existing FAILS admission — surfaced
    and healed from the freshly admitted receipt). Raises on a receipt that fails admission (fail closed — never
    written)."""
    day = day[:10]
    PR.admit_receipt(receipt, day, artifact_loader, server_record_loader)      # fail-closed BEFORE any write
    os.makedirs(receipts_dir, exist_ok=True)
    dst = os.path.join(receipts_dir, f"{day}.json")
    new_bytes = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if os.path.exists(dst):
        try:
            with open(dst, encoding="utf-8") as fh:
                existing = json.load(fh)
            PR.admit_receipt(existing, day, artifact_loader, server_record_loader)
            return "valid_existing_noop"                                       # existing admits -> leave it
        except Exception:
            _atomic_write(dst, new_bytes)                                      # invalid existing -> repair
            return "repaired"
    _atomic_write(dst, new_bytes)
    return "written"


# --------------------------------------------------------------------------------------------------------------
# Production loaders + fail-open daily entrypoint.
# --------------------------------------------------------------------------------------------------------------
def _git_commit_subject_loader(commit_sha):
    out = subprocess.run(["git", "-C", REPO, "log", "-1", "--format=%s", commit_sha],
                         capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise ValueError(f"git log {commit_sha[:8]} failed")
    return out.stdout.strip()


def _git_artifact_loader(commit_sha, relpath):
    out = subprocess.run(["git", "-C", REPO, "cat-file", "blob", f"{commit_sha}:{relpath}"],
                         capture_output=True, timeout=60)
    if out.returncode != 0:
        raise ValueError(f"git blob {commit_sha[:8]}:{relpath} unavailable")
    return out.stdout


def _gh_server_record_loader(api_url):
    path = api_url.replace("https://api.github.com/", "")
    out = subprocess.run(["gh", "api", path], capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise ValueError(f"gh api {path} failed")
    return json.loads(out.stdout)


def _gh_pages_latest():
    out = subprocess.run(["gh", "api", "repos/kantrarian/geospec/pages/builds/latest"],
                         capture_output=True, text=True, timeout=90)
    if out.returncode != 0:
        raise ValueError(f"gh api pages/builds/latest failed: {out.stderr.strip()[:200]}")
    return json.loads(out.stdout)


def main():
    try:
        build = _gh_pages_latest()
        day, receipt = build_receipt_for_pages_build(
            build, commit_subject_loader=_git_commit_subject_loader, artifact_loader=_git_artifact_loader)
    except Exception as exc:
        print(f"[receipt] fail-open ({type(exc).__name__}: {exc}) — no receipt this run", flush=True)
        return 0
    if not (day and receipt):
        print("[receipt] no built daily pages-build to receipt this run (conservative; self-heals)", flush=True)
        return 0
    try:
        status = publish_receipt(day, receipt, RECEIPTS_DIR,
                                 artifact_loader=_git_artifact_loader,
                                 server_record_loader=_gh_server_record_loader)
        print(f"[receipt] {day}.json -> {status} (server completion stamp {receipt['availability_utc']})", flush=True)
    except Exception as exc:
        print(f"[receipt] admission/publish refused ({type(exc).__name__}: {exc}) — no write", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
