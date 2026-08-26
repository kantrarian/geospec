#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""THE COMPACT VERIFICATION RUN SUMMARY generator (grassmann).

codex 0057Z P1-2 + 0404Z item 2 + 0410Z item 4.

The manifest pins BYTES; pinning does not prove those bytes were
executed successfully. So every time-varying execution fact lives
here, in one compact artifact, instead of in static generator prose
that goes stale silently and survives a zero-stale-pin regeneration.

Each verification invocation binds its exact argv, interpreter
version, exit code, typed verdict, tested path and blob SHA. The
verdict vocabulary is CLOSED and distinguishes:

  PASS              ran here and succeeded
  COVERED_ELSEWHERE the behaviour is locked, but by another surface
  REFUSE            ran here and refused -- an EXPECTED red is still
                    a red, and is recorded as one
  NOT_RUN           did not execute (missing evidence host, or a
                    result that cannot exist yet)

A missing input is NOT_RUN or REFUSE. It is never a green skip: that
conflation is exactly what let a skipped selftest be reported as a
pass, and the vocabulary now makes it unsayable.

This summary is a PRE-MANIFEST record. It never contains a
manifest-owned admission result -- a summary pinned inside a manifest
cannot honestly assert a PASS that only exists after that manifest
(codex 0410Z item 4). Those live in the downstream receipt emitted by
`w2_restage_verify_batch_grassmann`.

ZERO HTTP.
"""
import hashlib
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
SUMMARY_SCHEMA = "f2g-w2-verification-run-summary-v1"
SUMMARY_PATH = ("docs/f2g_window2_execution/"
                "w2_verification_run_summary_v1.json")
VERDICTS = ("PASS", "COVERED_ELSEWHERE", "REFUSE", "NOT_RUN")
# every verification surface, with the interpreters it must run on
SURFACES = (
    "test_f2g_window2_redkats_grassmann.py",
    "test_w2_fixture_schema_redkats_grassmann.py",
    "w2_acquisition_capture_grassmann.py",
    "w2_producer_grassmann.py",
    "w2_restage_lineage_grassmann.py",
    "w2_disposition_capsule_grassmann.py",
    "w2_restage_verify_batch_grassmann.py",
    "w2_restage_v4_grassmann.py",
    # codex 0445Z item 1: the six separately BOUND cayley locks must
    # be invocations of this summary too -- a bound verification
    # surface that this record never runs is a claim nobody executed
    "test_w2_capsule_pin_bind_redkats_cayley.py",
    "test_w2_report_proof_kinds_redkats_cayley.py",
    "test_w2_boundary_admitted_partition_redkats_cayley.py",
    "test_w2_admitted_absence_redkats_cayley.py",
    "test_w2_authority_serves_every_key_redkats_cayley.py",
    "test_w2_frozen_carrier_set_redkats_cayley.py",
)
SELFTEST_ARG = {
    "w2_disposition_capsule_grassmann.py": ["--selftest"],
    "w2_restage_verify_batch_grassmann.py": ["--selftest"],
    "w2_restage_v4_grassmann.py": ["--selftest"],
}
# codex requires dual-interpreter coverage; cayley's P0 found my
# artifact LABELLING seven runs py3.14 while executing 3.11.9 -- this
# host has NO 3.14 at all, so both "interpreters" were the same one.
# The repair is not a better label but a DISCOVERED one: each
# interpreter is probed, its REAL version recorded, and a REQUIRED
# interpreter that is absent is emitted as an explicit NOT_RUN row.
# An absence must be visible in the artifact, never papered over by
# a label that claims a risk class was tested.
REQUIRED_INTERPRETERS = ("3.14", "3.11")
# ONLY the program's declared pair is reported. 3.13/3.12 exist on
# this host but lack the pinned scientific stack, so running them
# would emit refusals that read as defects in this code rather than
# as an environment gap -- noise that obscures the one fact that
# matters here: 3.14 is ABSENT and was NOT exercised.
CANDIDATE_LAUNCHERS = REQUIRED_INTERPRETERS


def _probe(spec):
    """Return (argv, version) for a launcher spec, or (None, None)."""
    for argv in ((["py", f"-{spec}"]), ):  # explicit only
        p = subprocess.run(list(argv) + [
            "-c", "import sys;print(sys.version.split()[0])"],
            capture_output=True)
        v = p.stdout.decode().strip()
        if p.returncode == 0 and v.startswith(spec):
            return list(argv), v
    return None, None


def discover_interpreters():
    """Only interpreters that ACTUALLY resolve are runnable; every
    REQUIRED one that does not is recorded as NOT_RUN."""
    runnable, missing = [], []
    for spec in CANDIDATE_LAUNCHERS:
        argv, ver = _probe(spec)
        if argv:
            runnable.append((f"py{spec}", argv, ver))
    have = {lbl.replace("py", "") for lbl, _a, _v in runnable}
    for spec in REQUIRED_INTERPRETERS:
        if spec not in have:
            missing.append(spec)
    return runnable, missing


def _git_blob_oid(path):
    """The GIT OBJECT ID (SHA-1). cayley's P0: this was previously
    emitted under the name `blob_sha256`, so the one field designed
    to JOIN this summary to the manifest's blob_sha256 could never
    join it -- a 40-hex SHA-1 under a name promising a 64-hex
    SHA-256. Both are now present, each under its true name."""
    p = subprocess.run(["git", "-C", REPO, "rev-parse",
                        f"HEAD:{path}"], capture_output=True)
    out = p.stdout.decode().strip()
    return out if p.returncode == 0 and out else None


def _blob_sha256(path):
    """The TRUE sha256 of the committed blob bytes -- the value the
    manifest's blob_sha256 can actually be joined against."""
    p = subprocess.run(["git", "-C", REPO, "cat-file", "blob",
                        f"HEAD:{path}"], capture_output=True)
    if p.returncode != 0:
        return None
    return hashlib.sha256(p.stdout).hexdigest()


def _canon_bytes(raw):
    return raw.replace(b"\r\n", b"\n")


def _blob_sha256_canonical(path):
    p = subprocess.run(["git", "-C", REPO, "cat-file", "blob",
                        f"HEAD:{path}"], capture_output=True)
    if p.returncode != 0:
        return None
    return hashlib.sha256(_canon_bytes(p.stdout)).hexdigest()


def _disk_sha256_canonical(abspath):
    if not os.path.isfile(abspath):
        return None
    with open(abspath, "rb") as f:
        return hashlib.sha256(_canon_bytes(f.read())).hexdigest()


def _disk_sha(abspath):
    if not os.path.isfile(abspath):
        return None
    with open(abspath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _classify(rc, tail):
    """A typed verdict, never a bare exit code. An expected red is
    still recorded as a red."""
    t = (tail or "").lower()
    if rc == 0:
        if "skip" in t or "inputs absent" in t:
            return "NOT_RUN"
        # codex 0445Z item 1: detect COVERED-ELSEWHERE LITERALLY so a
        # doctor that lives in another surface is never promoted to a
        # standalone PASS here (cayley's BP-2)
        if "covered-elsewhere" in t or "covered_elsewhere" in t:
            return "COVERED_ELSEWHERE"
        return "PASS"
    if "refus" in t or "vacuous" in t or "stale" in t \
            or "does not match" in t:
        return "REFUSE"
    return "REFUSE"


def _resolved_executable(interp_argv):
    p = subprocess.run(list(interp_argv) +
                       ["-c", "import sys;print(sys.executable)"],
                       capture_output=True)
    return p.stdout.decode().strip() or None


def _utc_now():
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_surface(name, interp_label, interp_argv, host_id=None,
                source_commit=None):
    rel = f"monitoring/src/{name}"
    argv = list(interp_argv) + [name] + SELFTEST_ARG.get(name, [])
    p = subprocess.run(argv, cwd=os.path.join(REPO, "monitoring",
                                              "src"),
                       capture_output=True)
    out = (p.stdout + p.stderr).decode("utf-8", "replace")
    tail = " ".join(out.strip().splitlines()[-2:])[:300]
    # codex 0445Z item 1: a PASS must come from bytes that MATCH the
    # committed blob. The executed disk bytes are canonicalised
    # CRLF->LF and compared to the committed blob's canonical digest;
    # a green result from divergent bytes is REFUSE, not PASS.
    committed = _blob_sha256_canonical(rel)
    executed = _disk_sha256_canonical(
        os.path.join(REPO, "monitoring", "src", name))
    verdict = _classify(p.returncode, tail)
    if verdict == "PASS" and committed and executed             and committed != executed:
        verdict = "REFUSE"
        tail = ("executed bytes diverge from the committed blob; a "
                "green run of uncommitted source is not a PASS :: "
                + tail)
    return {"surface": rel, "argv": argv,
            "host_id": host_id, "source_commit": source_commit,
            "resolved_executable": _resolved_executable(interp_argv),
            "run_utc": _utc_now(),
            "digest_domain": "UTF8_SOURCE_LF_V1",
            "interpreter_label": interp_label,
            "interpreter_version": _interp_version(interp_argv),
            "exit_code": p.returncode,
            "verdict": verdict,
            "canonical_committed_sha256": committed,
            "canonical_executed_sha256": executed,
            "tail": tail,
            "git_blob_oid": _git_blob_oid(rel),
            "blob_sha256": _blob_sha256(rel),
            "disk_sha256": _disk_sha(
                os.path.join(REPO, "monitoring", "src", name))}


def _interp_version(interp_argv):
    p = subprocess.run(list(interp_argv) +
                       ["-c", "import sys;print(sys.version.split()[0])"],
                       capture_output=True)
    return p.stdout.decode().strip() or "UNKNOWN"


def _inventory():
    """The 1,420-key / staged-tree inventory codex ruled must be
    committed INSTEAD of the 5,680-file tree."""
    import w2_restage_v4_grassmann as RES
    inv = {"v4_staged_tree": RES.V4_STAGED,
           "v4_store": RES.V4_STORE}
    if not os.path.isdir(RES.V4_STAGED):
        inv["status"] = "NOT_RUN: the v4 staged tree is absent here"
        return inv
    files = sorted(os.listdir(RES.V4_STAGED))
    per = {}
    for f in files:
        for suf in (".restage.json", ".record.json",
                    ".contract.json", ".artifact.json",
                    ".transcript.json"):
            if f.endswith(suf):
                per[suf] = per.get(suf, 0) + 1
    total_bytes = sum(os.path.getsize(
        os.path.join(RES.V4_STAGED, f)) for f in files)
    bodies = sorted(x for x in os.listdir(RES.V4_STORE)
                    if x.endswith(".body")) \
        if os.path.isdir(RES.V4_STORE) else []
    inv.update({
        "status": "PRESENT",
        "class_counts": dict(sorted(per.items())),
        "file_count": len(files),
        "sorted_relative_path_digest": hashlib.sha256(
            json.dumps(files, separators=(",", ":")).encode()
        ).hexdigest(),
        "total_bytes": total_bytes,
        "distinct_body_count": len(bodies),
        "body_name_digest": hashlib.sha256(
            json.dumps(bodies, separators=(",", ":")).encode()
        ).hexdigest()})
    return inv


def _host_id():
    import platform
    return f"{platform.node()}/{platform.system()}"


def build():
    # codex 0509Z item 2: a multi-host summary may combine rows ONLY
    # when every row records its host and the source commit resolved
    # ONCE at build start -- otherwise a row from another machine
    # reads as if it ran here.
    host_id = _host_id()
    source_commit = subprocess.run(
        ["git", "-C", REPO, "rev-parse", "HEAD"],
        capture_output=True).stdout.decode().strip()
    runnable, missing = discover_interpreters()
    rows = []
    for name in SURFACES:
        for label, argv, _ver in runnable:
            rows.append(run_surface(name, label, argv, host_id,
                                    source_commit))
        for spec in missing:
            rows.append({
                "surface": f"monitoring/src/{name}",
                "argv": None, "interpreter_label": f"py{spec}",
                "interpreter_version": None,
                "exit_code": None, "verdict": "NOT_RUN",
                "tail": (f"no Python {spec} runtime exists on this "
                         "host; this interpreter was NOT exercised "
                         "here and no label may imply otherwise"),
                "git_blob_oid": _git_blob_oid(
                    f"monitoring/src/{name}"),
                "blob_sha256": _blob_sha256(
                    f"monitoring/src/{name}"),
                "host_id": host_id, "source_commit": source_commit,
                "resolved_executable": None,
                "run_utc": _utc_now(),
                "digest_domain": "UTF8_SOURCE_LF_V1",
                "canonical_committed_sha256": _blob_sha256_canonical(
                    f"monitoring/src/{name}"),
                "canonical_executed_sha256": None,
                "disk_sha256": _disk_sha(os.path.join(
                    REPO, "monitoring", "src", name))})
    counts = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    return {"schema": SUMMARY_SCHEMA,
            "host_id": host_id, "source_commit": source_commit,
            "producer_generator_blob_sha256": _blob_sha256(
                "monitoring/src/"
                "w2_verification_run_summary_grassmann.py"),
            "missing_required_interpreters": list(missing),
            "claim_scope": "PRE_MANIFEST_VERIFICATION_RECORD",
            "authorizes": "NOTHING",
            "note": ("execution outcomes only; manifest-owned "
                     "admission results exist ONLY in the downstream "
                     "post-manifest receipt"),
            "repo_head": subprocess.run(
                ["git", "-C", REPO, "rev-parse", "HEAD"],
                capture_output=True).stdout.decode().strip(),
            "verdict_vocabulary": list(VERDICTS),
            "invocations": rows,
            "verdict_counts": dict(sorted(counts.items())),
            "staged_inventory": _inventory(),
            "http_requests": 0}


def main():
    s = build()
    out = os.path.join(REPO, *SUMMARY_PATH.split("/"))
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(s, f, indent=1, sort_keys=True)
        f.write("\n")
    print("verdicts:", s["verdict_counts"])
    for r in s["invocations"]:
        if r["verdict"] != "PASS":
            print(f"  {r['verdict']:17s} {r['interpreter_label']} "
                  f"{os.path.basename(r['surface'])}")
    print("written:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
