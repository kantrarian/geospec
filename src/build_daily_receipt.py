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
(no admission, no write). NEVER backfills a receipt.

TARGET-BOUND (grassmann proposal 2026-09-14, r2 after codex 2026-09-14T17:05Z; scratch only): receipts are taken for
DECLARED targets (exact scored day + full commit) from THAT commit's own Pages build -- never pages/builds/latest --
and only once the build listing has actually ended within its declared budget. An existing receipt is skipped only
after full admission for the declared day and target. Target declarations and late-observation recovery records are
installed create-once and never replace a first writer. An explicit LATE_RECOVERY writes a separate non-receipt
record (late_observation_recovery), never a receipt.
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
import create_once as CO   # noqa: E402  (atomic no-replace installation for create-once records)
import receipt_target_selector as SEL   # noqa: E402  (target-bound build selection; never /latest)
import late_observation_recovery as LOR   # noqa: E402  (non-receipt late-observation records)

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


def _gh_pages_builds_page(page, per_page):
    """One page of the Pages build listing, newest first. Enumeration completes only when the listing ends."""
    out = subprocess.run(["gh", "api", f"repos/kantrarian/geospec/pages/builds?per_page={int(per_page)}&page={int(page)}"],
                         capture_output=True, text=True, timeout=90)
    if out.returncode != 0:
        raise ValueError(f"gh api pages/builds page {page} failed: {out.stderr.strip()[:200]}")
    return json.loads(out.stdout)


def _git_commit_utc_loader(commit_sha):
    """Diagnostic metadata only: a git commit time never bounds the build listing."""
    out = subprocess.run(["git", "-C", REPO, "log", "-1", "--format=%cI", commit_sha], capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        raise ValueError(f"git log {commit_sha[:8]} failed")
    stamp = datetime.datetime.fromisoformat(out.stdout.strip())
    return stamp.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------------------------------------------
# Target-bound capture (grassmann r2; codex 2026-09-14T16:00Z and T17:05Z: scratch only until owner write/apply authority).
# --------------------------------------------------------------------------------------------------------------
TARGETS_DIR = os.path.join(REPO, "monitoring", "receipts", "targets")
RECOVERIES_DIR = os.path.join(REPO, "monitoring", "receipt_recoveries")
ORDINARY_CAPTURE = "ORDINARY_CAPTURE"
LATE_RECOVERY = "LATE_RECOVERY"
VALID_EXISTING_RECEIPT = "VALID_EXISTING_RECEIPT"
EXISTING_RECEIPT_TARGET_CONFLICT = "EXISTING_RECEIPT_TARGET_CONFLICT"
ORIGINAL_RECEIPT_PRESENT = "ORIGINAL_RECEIPT_PRESENT"
ORIGINAL_RECEIPT_FILE_INVALID = "ORIGINAL_RECEIPT_FILE_INVALID_REPAIR_FIRST"
RECOVERY_BLOCKS_ORDINARY = "LATE_RECOVERY_RECORDED_ORDINARY_CAPTURE_REFUSED"
RECEIPT_NOT_BUILDABLE = "RECEIPT_NOT_BUILDABLE_FROM_SELECTED_BUILD"
EXISTING_ABSENT = "ABSENT"
EXISTING_VALID_FOR_TARGET = "VALID_FOR_DECLARED_TARGET"
EXISTING_VALID_FOR_OTHER_COMMIT = "VALID_FOR_ANOTHER_COMMIT"
EXISTING_INVALID = "INVALID"


class TargetDeclarationConflict(ValueError):
    def __init__(self, detail):
        super().__init__(f"TARGET_DECLARATION_CONFLICT: {detail}")
        self.code = "TARGET_DECLARATION_CONFLICT"


def existing_receipt_state(receipt_path, target, *, artifact_loader, server_record_loader):
    """Classify an existing receipt by FULL admission for the declared day and target -- never by schema text or parsing.
    An admitting receipt for another commit is reported, never overwritten or called the declared target."""
    if not os.path.exists(receipt_path):
        return EXISTING_ABSENT, None
    try:
        with open(receipt_path, encoding="utf-8") as fh:
            existing = json.load(fh)
        verified = PR.admit_receipt(existing, target["scored_day"], artifact_loader, server_record_loader)
    except Exception as exc:  # noqa: BLE001 -- an unadmittable file is INVALID and reaches the named repair path
        return EXISTING_INVALID, f"{type(exc).__name__}: {exc}"[:200]
    if verified.receipt.get("commit_sha") != target["commit_sha"]:
        return EXISTING_VALID_FOR_OTHER_COMMIT, verified.receipt.get("commit_sha")
    return EXISTING_VALID_FOR_TARGET, None


def _diagnostic_commit_utc(commit_utc_loader, commit_sha):
    if commit_utc_loader is None:
        return None
    try:
        return commit_utc_loader(commit_sha)
    except Exception:  # noqa: BLE001 -- diagnostic metadata only
        return None


def capture_target(target, mode, *, provider, commit_subject_loader, artifact_loader, server_record_loader,
                   receipts_dir, recoveries_dir, now_utc, commit_utc_loader=None, original_receipt_state=None,
                   producer=None, per_page=SEL.DEFAULT_PER_PAGE, max_pages=SEL.DEFAULT_MAX_PAGES):
    """Capture ONE declared target, bound to its exact day and full commit. ORDINARY_CAPTURE writes the canonical v2
    receipt through admission (or repairs an unadmittable one); LATE_RECOVERY writes only the distinct late-observation
    recovery record. Lateness is an explicit mode, never inferred from a cutoff. Every non-complete selection or
    refusal writes nothing, and every outcome is named."""
    if mode not in (ORDINARY_CAPTURE, LATE_RECOVERY):
        raise ValueError(f"unknown capture mode {mode!r}")
    day = target["scored_day"]
    receipt_path = os.path.join(receipts_dir, f"{day}.json")
    recovery_file = LOR.recovery_path(recoveries_dir, day)
    base = dict(mode=mode, scored_day=day, commit_sha=target["commit_sha"], wrote=None)
    if mode == ORDINARY_CAPTURE and os.path.exists(recovery_file):
        return dict(base, state=RECOVERY_BLOCKS_ORDINARY)
    existing, existing_detail = existing_receipt_state(receipt_path, target, artifact_loader=artifact_loader,
                                                       server_record_loader=server_record_loader)
    base["existing_receipt"] = existing
    if mode == ORDINARY_CAPTURE:
        if existing == EXISTING_VALID_FOR_TARGET:
            return dict(base, state=VALID_EXISTING_RECEIPT, wrote="valid_existing_noop", path=receipt_path)
        if existing == EXISTING_VALID_FOR_OTHER_COMMIT:
            return dict(base, state=EXISTING_RECEIPT_TARGET_CONFLICT, existing_commit=existing_detail)
    else:
        if existing in (EXISTING_VALID_FOR_TARGET, EXISTING_VALID_FOR_OTHER_COMMIT):
            return dict(base, state=ORIGINAL_RECEIPT_PRESENT)
        if existing == EXISTING_INVALID:
            return dict(base, state=ORIGINAL_RECEIPT_FILE_INVALID, existing_detail=existing_detail)
    builds, coverage = SEL.enumerate_builds(provider, per_page=per_page, max_pages=max_pages,
                                            target_commit_utc=_diagnostic_commit_utc(commit_utc_loader, target["commit_sha"]))
    selection = SEL.select_target_build(target, builds, coverage, commit_subject_loader=commit_subject_loader,
                                        artifact_loader=artifact_loader, server_record_loader=server_record_loader)
    if selection["state"] != SEL.COMPLETE:
        return dict(base, state=selection["state"], selection=selection)
    if mode == ORDINARY_CAPTURE:
        built_day, receipt = build_receipt_for_pages_build(selection["build"], commit_subject_loader=commit_subject_loader,
                                                           artifact_loader=artifact_loader)
        if built_day != day or receipt is None:
            return dict(base, state=RECEIPT_NOT_BUILDABLE, selection=selection)
        status = publish_receipt(day, receipt, receipts_dir, artifact_loader=artifact_loader,
                                 server_record_loader=server_record_loader)
        return dict(base, state=SEL.COMPLETE, wrote=status, path=receipt_path)
    record = LOR.build_recovery_record(selection, commit_subject_loader=commit_subject_loader, artifact_loader=artifact_loader,
                                       observed_utc=now_utc, created_utc=now_utc,
                                       original_receipt_state=original_receipt_state, cause="CAUSE_UNESTABLISHED",
                                       producer=producer or {})
    status = LOR.write_recovery_record(record, recoveries_dir)
    return dict(base, state=SEL.COMPLETE, wrote=status, path=recovery_file)


def declare_target_file(scored_day, commit_sha, targets_dir, *, declared_utc):
    """Create-once, atomic declaration of a published day's receipt target: the exact pushed commit. A first declaration
    is never replaced: the same target is a no-op that keeps the first bytes; a different commit is a typed conflict."""
    target = SEL.declare_target(scored_day, commit_sha, declared_utc=declared_utc, source="run_and_publish.ps1 post-push")
    dst = os.path.join(targets_dir, f"{scored_day}.json")
    if CO.install_no_replace(dst, (json.dumps(target, sort_keys=True, indent=2) + "\n").encode("utf-8")):
        return "declared"
    with open(dst, encoding="utf-8") as fh:
        existing = json.load(fh)
    if existing.get("scored_day") == scored_day and existing.get("commit_sha") == commit_sha:
        return "existing_target_noop"
    raise TargetDeclarationConflict(f"{scored_day} already declared with commit {existing.get('commit_sha')!r}")


def declared_targets(targets_dir):
    rows = []
    if os.path.isdir(targets_dir):
        for name in sorted(os.listdir(targets_dir)):
            if name.endswith(".json"):
                with open(os.path.join(targets_dir, name), encoding="utf-8") as fh:
                    d = json.load(fh)
                rows.append(SEL.declare_target(d["scored_day"], d["commit_sha"], declared_utc=d["declared_utc"], source=d.get("source")))
    return rows


def _now_utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="R6 publication receipts bound to declared targets (never pages/builds/latest)")
    ap.add_argument("--declare-target", action="store_true", help="record the pushed commit as this day's receipt target")
    ap.add_argument("--late-recovery", action="store_true", help="explicit late-observation recovery for one day")
    ap.add_argument("--scored-day")
    ap.add_argument("--commit")
    ap.add_argument("--original-receipt-state")
    ap.add_argument("--write", action="store_true", help="late recovery writes only with this flag; otherwise a dry run")
    args = ap.parse_args(argv)
    loaders = dict(provider=_gh_pages_builds_page, commit_utc_loader=_git_commit_utc_loader,
                   commit_subject_loader=_git_commit_subject_loader, artifact_loader=_git_artifact_loader,
                   server_record_loader=_gh_server_record_loader)
    if args.declare_target:
        try:
            status = declare_target_file(args.scored_day, args.commit, TARGETS_DIR, declared_utc=_now_utc())
            print(f"[receipt] target {args.scored_day} -> {status}", flush=True)
        except Exception as exc:
            print(f"[receipt] target declaration refused ({type(exc).__name__}: {exc})", flush=True)
        return 0
    if args.late_recovery:
        target = SEL.declare_target(args.scored_day, args.commit, declared_utc=_now_utc(), source="explicit late recovery")
        if not args.write:
            builds, coverage = SEL.enumerate_builds(
                loaders["provider"], target_commit_utc=_diagnostic_commit_utc(loaders["commit_utc_loader"], target["commit_sha"]))
            selection = SEL.select_target_build(target, builds, coverage, commit_subject_loader=loaders["commit_subject_loader"],
                                                artifact_loader=loaders["artifact_loader"],
                                                server_record_loader=loaders["server_record_loader"])
            print(json.dumps(dict(mode=LATE_RECOVERY, dry_run=True, state=selection["state"]), sort_keys=True), flush=True)
            return 0 if selection["state"] == SEL.COMPLETE else 2
        result = capture_target(target, LATE_RECOVERY, receipts_dir=RECEIPTS_DIR, recoveries_dir=RECOVERIES_DIR,
                                now_utc=_now_utc(), original_receipt_state=args.original_receipt_state,
                                producer=dict(module="src/build_daily_receipt.py"), **loaders)
        print(json.dumps({k: v for k, v in result.items() if k != "selection"}, sort_keys=True, default=str), flush=True)
        return 0 if result.get("wrote") else 2
    # The daily ordinary capture: fail-open for the publish, never for evidence; one target's failure never stops the next.
    try:
        targets = declared_targets(TARGETS_DIR)
    except Exception as exc:
        print(f"[receipt] fail-open: targets unreadable ({type(exc).__name__}: {exc})", flush=True)
        return 0
    for target in targets:
        try:
            result = capture_target(target, ORDINARY_CAPTURE, receipts_dir=RECEIPTS_DIR, recoveries_dir=RECOVERIES_DIR,
                                    now_utc=_now_utc(), **loaders)
            print(f"[receipt] {target['scored_day']} {result['state']} wrote={result.get('wrote')} "
                  f"existing={result.get('existing_receipt')}", flush=True)
        except Exception as exc:
            print(f"[receipt] {target['scored_day']} fail-open ({type(exc).__name__}: {exc})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
