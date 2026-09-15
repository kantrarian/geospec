#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Target-bound GitHub Pages build selection for ONE declared (scored_day, full commit) -- never /latest.

Proposed runner module (grassmann, 2026-09-14; r2 after codex 2026-09-14T17:05Z), scratch-only until the owner gives
separate write/apply authority.

The daily receipt step used to read `pages/builds/latest` once. When the build of the commit just pushed was
already registered and still building, the previous day's completed build was never considered, and that day never
got a receipt. This module selects the build of the DECLARED target commit instead, and reports why it cannot when
it cannot:

  TARGET_BUILD_COMPLETE      exactly one build of the target commit: built, error-free, pinned URL, ordered clocks,
                             and its independently reopened server record matches the listing
  TARGET_BUILD_PENDING       that build is queued or building
  TARGET_BUILD_ERRORED       that build errored, or carries an error message
  TARGET_BUILD_STATUS_UNKNOWN  any other status
  TARGET_BUILD_ABSENT        the listing was enumerated to its END and no build names the target commit
  DUPLICATE_TARGET_BUILDS    more than one build names the target commit; none is chosen
  TARGET_DAY_MISMATCH        the commit subject or the reopened carrier does not name the declared day
  BUILD_URL_UNPINNED         the build URL is not the pinned repository's Pages build URL
  SERVER_CLOCK_INVALID       a malformed or reversed server clock
  SERVER_RECORD_MISMATCH     the reopened server record disagrees with the listing
  --- unresolved (never read as absence) ---
  QUERY_INCOMPLETE           the declared page budget ran out before the listing ended
  PROVIDER_FAILED            the provider raised, returned a non-list page, or returned more rows than requested
  SERVER_RECORD_UNAVAILABLE  the server record could not be reopened
  TARGET_COMMIT_UNREADABLE   the commit subject or carrier could not be read

r2 (codex item 2): enumeration no longer stops when a listed build predates the target commit's git time. A skewed or
future commit clock made a partial listing look complete, which produced a false TARGET_BUILD_ABSENT and let a receipt
be written past an unseen duplicate. Completeness now requires the provider's listing to actually end within the
declared budget; the commit time is recorded as diagnostic metadata only.

Pure standard library. Every input is injected, so the fixture battery runs without a live provider.
"""
import datetime
import json
import re

PAGES_URL_RE = re.compile(r"\Ahttps://api\.github\.com/repos/kantrarian/geospec/pages/builds/(\d+)\Z")
FULL_COMMIT_RE = re.compile(r"\A[0-9a-f]{40}\Z")
DAY_RE = re.compile(r"\A\d{4}-\d{2}-\d{2}\Z")
CARRIER = "docs/ensemble_latest.json"

COMPLETE = "TARGET_BUILD_COMPLETE"
PENDING = "TARGET_BUILD_PENDING"
ERRORED = "TARGET_BUILD_ERRORED"
STATUS_UNKNOWN = "TARGET_BUILD_STATUS_UNKNOWN"
ABSENT = "TARGET_BUILD_ABSENT"
DUPLICATE = "DUPLICATE_TARGET_BUILDS"
DAY_MISMATCH = "TARGET_DAY_MISMATCH"
URL_UNPINNED = "BUILD_URL_UNPINNED"
CLOCK_INVALID = "SERVER_CLOCK_INVALID"
RECORD_MISMATCH = "SERVER_RECORD_MISMATCH"
QUERY_INCOMPLETE = "QUERY_INCOMPLETE"
PROVIDER_FAILED = "PROVIDER_FAILED"
RECORD_UNAVAILABLE = "SERVER_RECORD_UNAVAILABLE"
COMMIT_UNREADABLE = "TARGET_COMMIT_UNREADABLE"
UNRESOLVED = frozenset({QUERY_INCOMPLETE, PROVIDER_FAILED, RECORD_UNAVAILABLE, COMMIT_UNREADABLE})
COVERAGE_COMPLETE = "COMPLETE"
DEFAULT_PER_PAGE = 100
DEFAULT_MAX_PAGES = 50


class TargetRefused(ValueError):
    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}")
        self.code = code


def parse_utc(ts):
    """A timezone-aware UTC instant from an ISO string; anything else raises ValueError."""
    if not isinstance(ts, str) or not ts:
        raise ValueError(f"timestamp {ts!r} is not a string")
    value = datetime.datetime.fromisoformat(ts[:-1] + "+00:00" if ts.endswith("Z") else ts)
    if value.tzinfo is None:
        raise ValueError(f"timestamp {ts!r} carries no offset")
    return value.astimezone(datetime.timezone.utc)


def declare_target(scored_day, commit_sha, *, declared_utc, source):
    """One declared target. The commit must be the FULL lowercase 40-hex id: a prefix is refused, never expanded."""
    if not (isinstance(scored_day, str) and DAY_RE.match(scored_day)):
        raise TargetRefused("TARGET_DAY_INVALID", repr(scored_day))
    try:
        datetime.date.fromisoformat(scored_day)
    except ValueError:
        raise TargetRefused("TARGET_DAY_INVALID", repr(scored_day)) from None
    if not (isinstance(commit_sha, str) and FULL_COMMIT_RE.match(commit_sha)):
        raise TargetRefused("TARGET_COMMIT_NOT_FULL", repr(commit_sha))
    try:
        parse_utc(declared_utc)
    except ValueError:
        raise TargetRefused("TARGET_DECLARED_UTC_INVALID", repr(declared_utc)) from None
    return dict(scored_day=scored_day, commit_sha=commit_sha, declared_utc=declared_utc, source=source)


def _error_free(rec):
    """GitHub's success shape: `error` absent, None, "", or a dict whose message is None or empty."""
    if not isinstance(rec, dict) or "error" not in rec:
        return isinstance(rec, dict)
    err = rec["error"]
    if err is None or err == "":
        return True
    if isinstance(err, dict):
        return err.get("message") in (None, "")
    return False


def enumerate_builds(provider, *, per_page=DEFAULT_PER_PAGE, max_pages=DEFAULT_MAX_PAGES, target_commit_utc=None):
    """Enumerate the build listing within a DECLARED budget and say how far it got.

    `provider(page, per_page)` returns one page as a list. Coverage is COMPLETE only when the listing actually ENDS
    inside the budget: a page shorter than `per_page`, an empty page included. Exhausting `max_pages` is
    QUERY_INCOMPLETE. A provider exception, a non-list page, or a page longer than requested is PROVIDER_FAILED. None of
    these is ever evidence that the target build is absent. `target_commit_utc` is recorded as diagnostic metadata only
    and never ends enumeration."""
    if not (isinstance(per_page, int) and per_page > 0 and isinstance(max_pages, int) and max_pages > 0):
        raise ValueError(f"per_page and max_pages must be positive integers, got {per_page!r} and {max_pages!r}")
    builds, pages = [], 0

    def coverage(state, **facts):
        return dict(state=state, pages=pages, per_page=per_page, max_pages=max_pages, builds_listed=len(builds),
                    target_commit_utc_diagnostic=target_commit_utc, **facts)

    for page in range(1, max_pages + 1):
        try:
            rows = provider(page, per_page)
        except Exception as exc:  # noqa: BLE001 -- a provider failure is an unresolved state, never an abort or an absence
            return builds, coverage(PROVIDER_FAILED, detail=f"{type(exc).__name__}: {exc}"[:300])
        if not isinstance(rows, list):
            return builds, coverage(PROVIDER_FAILED, detail="page is not a list")
        if len(rows) > per_page:
            return builds, coverage(PROVIDER_FAILED, detail=f"page {page} returned {len(rows)} rows for per_page={per_page}")
        pages += 1
        builds.extend(rows)
        if len(rows) < per_page:
            return builds, coverage(COVERAGE_COMPLETE, reason="LISTING_ENDED")
    return builds, coverage(QUERY_INCOMPLETE, detail=f"max_pages={max_pages} exhausted before the listing ended")


def select_target_build(target, builds, coverage, *, commit_subject_loader, artifact_loader, server_record_loader):
    """Select the build of the declared target, or name exactly why not. Newer builds of other commits are ignored."""
    day, commit = target["scored_day"], target["commit_sha"]

    def result(state, **facts):
        return dict(state=state, scored_day=day, commit_sha=commit, coverage=coverage, **facts)

    try:
        subject = str(commit_subject_loader(commit)).strip()
        carrier_date = json.loads(bytes(artifact_loader(commit, CARRIER)).decode("utf-8")).get("date")
    except Exception as exc:  # noqa: BLE001
        return result(COMMIT_UNREADABLE, detail=f"{type(exc).__name__}: {exc}"[:300])
    if subject != f"Daily monitoring {day}" or carrier_date != day:
        return result(DAY_MISMATCH, subject=subject, carrier_date=carrier_date)
    if coverage.get("state") != COVERAGE_COMPLETE:
        return result(coverage.get("state") or QUERY_INCOMPLETE, detail=coverage.get("detail"))

    matches = [b for b in builds if isinstance(b, dict) and b.get("commit") == commit]
    if not matches:
        return result(ABSENT, builds_seen=len(builds))
    if len(matches) > 1:
        return result(DUPLICATE, build_urls=[b.get("url") for b in matches])
    build = matches[0]
    status = build.get("status")
    if status in ("queued", "building"):
        return result(PENDING, status=status, build_url=build.get("url"))
    if status == "errored" or not _error_free(build):
        return result(ERRORED, status=status, build_url=build.get("url"), error=build.get("error"))
    if status != "built":
        return result(STATUS_UNKNOWN, status=status, build_url=build.get("url"))
    pinned = PAGES_URL_RE.match(str(build.get("url", "")))
    if not pinned:
        return result(URL_UNPINNED, build_url=build.get("url"))
    try:
        created, completed = parse_utc(build.get("created_at")), parse_utc(build.get("updated_at"))
    except ValueError as exc:
        return result(CLOCK_INVALID, detail=str(exc), created_at=build.get("created_at"), updated_at=build.get("updated_at"))
    if created > completed:
        return result(CLOCK_INVALID, detail="created_at is after updated_at",
                      created_at=build.get("created_at"), updated_at=build.get("updated_at"))
    try:
        record = server_record_loader(build["url"])
    except Exception as exc:  # noqa: BLE001
        return result(RECORD_UNAVAILABLE, build_url=build["url"], detail=f"{type(exc).__name__}: {exc}"[:300])
    fields = ("url", "commit", "status", "created_at", "updated_at")
    mismatched = sorted(k for k in fields if not isinstance(record, dict) or record.get(k) != build.get(k))
    if mismatched or not _error_free(record):
        return result(RECORD_MISMATCH, build_url=build["url"], fields=mismatched or ["error"])
    return result(COMPLETE, build=dict(build), build_id=pinned.group(1), server_record=dict(record),
                  server_created_utc=build["created_at"], server_completed_utc=build["updated_at"])
