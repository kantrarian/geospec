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


EXPECTED_CLASSES = (".restage.json", ".contract.json",
                    ".artifact.json", ".transcript.json")
EXPECTED_KEYS = 1420
EXPECTED_FILES = 5680
EXPECTED_BODIES = 1073


def _inventory():
    """codex 0509Z item 5: the inventory VERIFIES its invariants and
    FAILS CLOSED. Previously it printed 1420x4 / 5680 / 1073 without
    requiring any of them, so `status=PRESENT` could survive a
    materially wrong tree -- an observation dressed as a check."""
    import w2_restage_v4_grassmann as RES
    import w2_no_network_grassmann as NONET
    import w2_disposition_capsule_grassmann as DISP
    import w2_restage_lineage_grassmann as LIN
    import w2_acquisition_capture_grassmann as CAPM
    inv = {"v4_staged_tree": RES.V4_STAGED, "v4_store": RES.V4_STORE,
           "source_store_id": "s4t-w2-capture-20260825",
           "destination_store_id": "s4t-w2-capture-v4"}

    net = NONET.no_network()

    def refuse(why):
        """codex 1327Z P1: EVERY return carries the measured counter,
        including this one. The summary used to read the counter as
        `_inv.get("http_requests", 0)` -- so on the refusal path that
        is active today, a hard-coded 0 was reported under the label
        MEASURED_SENTINEL. An asserted zero wearing a measurement's
        name is worse than an asserted zero."""
        inv["status"] = "REFUSE"
        inv["reason"] = why
        inv["http_requests"] = net.attempts if net.entered else 0
        inv["http_counter_source"] = ("MEASURED_SENTINEL"
                                      if net.entered
                                      else "NOT_MEASURED_NO_OPERATION")
        return inv
    if not os.path.isdir(RES.V4_STAGED):
        return refuse("the v4 staged tree is absent on this host")
    net.__enter__()
    try:
        files = sorted(os.listdir(RES.V4_STAGED))
        per, extras = {}, []
        for f in files:
            hit = [c for c in EXPECTED_CLASSES if f.endswith(c)]
            if not hit:
                extras.append(f)
                continue
            per[hit[0]] = per.get(hit[0], 0) + 1
        if extras:
            return refuse(f"{len(extras)} file(s) outside the four "
                          f"allowed classes, e.g. {extras[:3]}")
        if sorted(per) != sorted(EXPECTED_CLASSES) or \
                any(v != EXPECTED_KEYS for v in per.values()):
            return refuse(f"class counts are not {EXPECTED_KEYS} x 4: "
                          f"{per}")
        if len(files) != EXPECTED_FILES:
            return refuse(f"{len(files)} files, expected "
                          f"{EXPECTED_FILES}")
        v4, v3, tset, bset, aset, claims = [], [], [], [], [], {}
        for f in files:
            if not f.endswith(".restage.json"):
                continue
            with open(os.path.join(RES.V4_STAGED, f),
                      encoding="utf-8") as fh:
                rec = json.load(fh)
            v4.append(rec["v4_key"])
            v3.append(rec["v3_key"])
            tset.append(rec["t_v3_sha256"])
            bset.append(rec["raw_body_sha256"])
            aset.append(rec["artifact_sha256"])
            if "claim" not in rec:
                return refuse(
                    "staged restage records predate the typed claim "
                    "block (they carry a nullable `outcome`); this "
                    "tree is STALE BY DESIGN and is regenerated in "
                    "step 2 -- refusing rather than reporting a "
                    "tree whose records the current contract cannot "
                    "read")
            c = rec["claim"]
            kind = c["artifact_claim_kind"]
            st = c.get("outcome") or c.get("support_outcome")
            claims.setdefault(kind, {})
            claims[kind][st] = claims[kind].get(st, 0) + 1
        if len(set(v4)) != EXPECTED_KEYS or \
                len(set(v3)) != EXPECTED_KEYS:
            return refuse(f"unique keys v4={len(set(v4))} "
                          f"v3={len(set(v3))}, expected "
                          f"{EXPECTED_KEYS} each")
        # every referenced body must EXIST and match its address
        bad = []
        for sha in sorted(set(bset)):
            bp = os.path.join(RES.V4_STORE, sha + ".body")
            if not os.path.isfile(bp):
                bad.append(sha)
                continue
            with open(bp, "rb") as fh:
                if hashlib.sha256(fh.read()).hexdigest() != sha:
                    bad.append(sha)
        if bad:
            return refuse(f"{len(bad)} referenced body/bodies missing "
                          f"or content-address mismatched")
        if len(set(bset)) != EXPECTED_BODIES:
            return refuse(f"{len(set(bset))} distinct bodies, "
                          f"expected {EXPECTED_BODIES}")

        def dg(xs):
            return hashlib.sha256(json.dumps(
                sorted(set(xs)), separators=(",", ":")).encode()
            ).hexdigest()
        inv.update({
            "status": "VERIFIED",
            "class_counts": dict(sorted(per.items())),
            "file_count": len(files),
            "sorted_relative_path_digest": hashlib.sha256(
                json.dumps(files, separators=(",", ":")).encode()
            ).hexdigest(),
            "total_bytes": sum(os.path.getsize(os.path.join(
                RES.V4_STAGED, f)) for f in files),
            "v4_key_digest": dg(v4), "v3_key_digest": dg(v3),
            "original_t_digest_set": dg(tset),
            "body_digest_set": dg(bset),
            "artifact_digest_set": dg(aset),
            "distinct_body_count": len(set(bset)),
            "claims": {k: dict(sorted(v.items()))
                       for k, v in sorted(claims.items())},
            "identities": {
                "transform": CAPM.transform_identity(),
                "restager": _blob_sha256(
                    "monitoring/src/w2_restage_v4_grassmann.py"),
                "verifier": _blob_sha256(
                    "monitoring/src/"
                    "w2_restage_lineage_grassmann.py"),
                "capsule": _blob_sha256(DISP.CAPSULE_PATH),
                "authority": _blob_sha256(DISP.AUTHORITY_PATH)},
            "http_requests": net.attempts,
            "http_counter_source": "MEASURED_SENTINEL"})
        if net.attempts:
            return refuse(f"the offline inventory ATTEMPTED "
                          f"{net.attempts} network connection(s)")
        return inv
    finally:
        net.__exit__()


def _require_measured(inv):
    """The record's counter comes from the inventory's MEASUREMENT or
    the record does not get one. `.get("http_requests", 0)` silently
    manufactured a zero on exactly the path that had not measured."""
    if "http_requests" not in inv or "http_counter_source" not in inv:
        raise RuntimeError(
            "the staged inventory returned no measured network "
            "counter; a summary may not manufacture one")
    return inv["http_requests"], inv["http_counter_source"]


def _host_id():
    import platform
    return f"{platform.node()}/{platform.system()}"


def build():
    # codex 0509Z item 2: a multi-host summary may combine rows ONLY
    # when every row records its host and the source commit resolved
    # ONCE at build start -- otherwise a row from another machine
    # reads as if it ran here.
    host_id = _host_id()
    # codex 1400Z P0 #2: the role is DECLARED from an observed
    # capability -- the v3 store is either present on this host or it
    # is not -- and validated at merge. It is never inferred from a
    # nickname.
    import w2_restage_v4_grassmann as _RES
    _has_store = os.path.isdir(_RES.V3_STORE)
    host_role = "EVIDENCE" if _has_store else "PORTABLE"
    store_identity = V3_STORE_IDENTITY if _has_store else None
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
    _inv = _inventory()
    counts = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    return {"schema": SUMMARY_SCHEMA,
            "host_role": host_role,
            "store_identity": store_identity,
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
            "staged_inventory": _inv,
            # codex 1400Z P1 #4: this zero measures the STAGED
            # INVENTORY and nothing else. build() runs all 14
            # surfaces as SUBPROCESSES, none of them inside a
            # sentinel -- codex measured 0/14 -- and a parent
            # monkey-patch could not instrument a child anyway. The
            # counter stays, correctly scoped; calling it a
            # build-wide measurement would be the label-vs-fact
            # defect again, this time in my favour.
            "http_requests": _require_measured(_inv)[0],
            "http_counter_source": _require_measured(_inv)[1],
            "http_counter_scope": "STAGED_INVENTORY_ONLY",
            "http_counter_caveat": (
                "the surface invocations are subprocesses and are "
                "NOT measured by this counter; this record makes no "
                "whole-build zero-network claim")}



# codex 1327Z P0 #2: snapshot agreement is not SUPPORT COMPLETENESS.
# merge_legs() enforced that legs agree on a frame; it accepted two
# ONE-ROW legs (one 3.11 row, one 3.14 row) and emitted
# `missing_required_interpreters: []` -- twenty-six required cells
# absent while the header read as complete coverage. A closed cell
# contract makes that unsayable: a leg declares SURFACES x
# INTERPRETERS exactly, and every non-execution is an EXPLICIT
# NOT_RUN row rather than an omitted one, because an omitted row is
# indistinguishable from a row nobody thought to require.
VERDICT_VOCABULARY = ("PASS", "COVERED_ELSEWHERE", "REFUSE",
                      "NOT_RUN")
INTERPRETER_LABELS = tuple(f"py{v}" for v in REQUIRED_INTERPRETERS)
# codex 1400Z P0 #1: the EXACT 17-field row schema build() emits.
# `REQUIRED_ROW_FIELDS` was a five-field SUBSET, so a leg of
# synthetic rows carrying no argv, exit code, executable, UTC, git
# object or digests was accepted and returned COMPLETE. A subset
# check cannot establish that a row is a RECORD rather than a label.
ROW_FIELDS = frozenset((
    "argv", "blob_sha256", "canonical_committed_sha256",
    "canonical_executed_sha256", "digest_domain", "disk_sha256",
    "exit_code", "git_blob_oid", "host_id", "interpreter_label",
    "interpreter_version", "resolved_executable", "run_utc",
    "source_commit", "surface", "tail", "verdict"))
# null EXACTLY on NOT_RUN -- nothing executed, so there is no argv,
# exit code, resolved executable, version or executed digest
NULLABLE_ON_NOT_RUN = frozenset((
    "argv", "canonical_executed_sha256", "exit_code",
    "interpreter_version", "resolved_executable"))
HEX64 = frozenset((
    "blob_sha256", "canonical_committed_sha256",
    "canonical_executed_sha256", "disk_sha256"))
SURFACE_PREFIX = "monitoring/src/"
# codex 1400Z P0 #2: a ROLE, declared and validated -- never a
# nickname. My own EVIDENCE_HOST = "devildog" did not equal the real
# host_id "Rmath151409/Windows", so the guard I had just written
# would have REFUSED the genuine devildog leg; my doctor passed only
# because it fed the nickname as host_id, i.e. it was green against
# data that does not exist.
HOST_ROLES = ("EVIDENCE", "PORTABLE")
V3_STORE_IDENTITY = "s4t-w2-capture-20260825"
# Surfaces that REACH host-local evidence when invoked as this record
# invokes them. codex ruled the capsule selftest belongs here: it
# reopens the host-local capsule/archive/store. The batch selftest
# does NOT -- that row claims only the refusal contract.
EVIDENCE_HOST_ONLY = (
    "w2_restage_lineage_grassmann.py",
    "w2_restage_v4_grassmann.py",
    "w2_disposition_capsule_grassmann.py",
)


def required_cells():
    """The closed cell set every leg must declare, exactly."""
    return {(sf, il) for sf in SURFACES for il in INTERPRETER_LABELS}


def _leg_cells(lg):
    """Canonical cells, so an alias path cannot collapse into a
    declared one."""
    return {(_canonical_surface(r["surface"]), r["interpreter_label"])
            for r in lg["invocations"]}


def _canonical_surface(path):
    """codex 1400Z: `alias/<declared-basename>` collapsed by basename
    in the cell set while the duplicate check compared full paths, so
    a 29th aliased row rode in and the leg still read COMPLETE. The
    canonical path must be EXACTLY the declared one."""
    return path[len(SURFACE_PREFIX):] \
        if path.startswith(SURFACE_PREFIX) else None


MERGED_SCHEMA = "f2g-w2-verification-run-summary-merged-v1"


class MergeRefusal(ValueError):
    """Typed refusal; a merged record REFUSES rather than interleaves."""


def _mr(msg):
    raise MergeRefusal("LEG_MERGE_REFUSED: " + str(msg))


def merge_legs(legs):
    """cayley 2026-08-26T1258Z: my committed==executed gate protects a
    ROW against its own divergent bytes; NOTHING protected the RECORD
    against rows generated at different snapshots. Two legs four
    commits apart would union into one record describing two code
    states -- and it would read as one coherent dual-interpreter
    verification while being an interleaving of two.

    cayley proposed the rule (co-generate at one frozen commit) and
    asked for the invariant anyway, because "we were careful to run at
    the same commit" is true until the once it is not. This is that
    invariant: it makes the class UNSAYABLE rather than remembered.

    `legs` is a sequence of leg records (already-parsed dicts).
    Returns the merged record, or raises MergeRefusal.
    """
    legs = list(legs)
    if len(legs) < 2:
        _mr(f"a merge needs at least two legs, got {len(legs)}")
    commits, gens, schemas, hosts = set(), set(), set(), []
    for i, lg in enumerate(legs):
        for f in ("source_commit", "repo_head", "host_id",
                  "producer_generator_blob_sha256", "schema",
                  "host_role", "store_identity", "invocations"):
            if f not in lg:
                _mr(f"leg {i} is missing required field {f!r}; a leg "
                    "without full provenance cannot enter a merge")
        # each leg must still be INTERNALLY honest
        if lg["source_commit"] != lg["repo_head"]:
            _mr(f"leg {i} ({lg['host_id']}) has source_commit "
                f"{lg['source_commit'][:12]} != repo_head "
                f"{lg['repo_head'][:12]} -- it did not run from its "
                "own committed snapshot")
        if lg["schema"] != SUMMARY_SCHEMA:
            _mr(f"leg {i} declares schema {lg['schema']!r}, not "
                f"{SUMMARY_SCHEMA!r} -- two legs AGREEING on an "
                "arbitrary string is not attestation of either")
        if lg["host_role"] not in HOST_ROLES:
            _mr(f"leg {i} declares host_role {lg['host_role']!r}, "
                f"not one of {list(HOST_ROLES)}")
        if lg["host_role"] == "EVIDENCE" and \
                lg["store_identity"] != V3_STORE_IDENTITY:
            _mr(f"leg {i} claims the EVIDENCE role but names store "
                f"{lg['store_identity']!r}, not "
                f"{V3_STORE_IDENTITY!r}")
        if lg["host_role"] != "EVIDENCE" and lg["store_identity"]:
            _mr(f"leg {i} is {lg['host_role']} yet names a store "
                f"{lg['store_identity']!r}")
        commits.add(lg["source_commit"])
        gens.add(lg["producer_generator_blob_sha256"])
        schemas.add(lg["schema"])
        hosts.append(lg["host_id"])
    # THE invariant cayley asked for: one snapshot, or refuse
    if len(commits) != 1:
        _mr("the legs were generated at DIFFERENT snapshots "
            f"{sorted(c[:12] for c in commits)} -- cross-host legs "
            "must be CO-GENERATED at one frozen commit; a union of "
            "these rows would be one record describing two code "
            "states")
    if len(schemas) != 1:
        _mr(f"the legs use different schemas {sorted(schemas)}")
    if len(gens) != 1:
        _mr("the legs were produced by DIFFERENT generator bytes "
            f"{sorted(g[:12] for g in gens)} -- same commit but a "
            "divergent producer still means two records")
    if len(set(hosts)) != len(hosts):
        _mr(f"duplicate host_id in {hosts}; a merge of one host with "
            "itself double-counts rather than adds coverage")
    commit = commits.pop()
    # codex 1400Z: resolve the named commit and RECOMPUTE the
    # generator digest from it. Cross-leg equality only proves the
    # two legs said the same thing; a fabricated 40-hex string and a
    # fabricated generator digest agree with themselves perfectly.
    p = subprocess.run(
        ["git", "-C", REPO, "rev-parse", f"{commit}^{{commit}}"],
        capture_output=True)
    if p.returncode != 0 or p.stdout.decode().strip() != commit:
        _mr(f"the agreed source_commit {commit[:12]} does not "
            "resolve to a commit in this repository")
    gen_path = SURFACE_PREFIX + os.path.basename(__file__)
    raw = subprocess.run(
        ["git", "-C", REPO, "cat-file", "blob",
         f"{commit}:{gen_path}"], capture_output=True).stdout
    if not raw:
        _mr(f"the producer generator is absent at {commit[:12]}")
    recomputed = hashlib.sha256(raw).hexdigest()
    declared = list(gens)[0]
    if recomputed != declared:
        _mr(f"the declared producer_generator_blob_sha256 "
            f"{declared[:12]} does not recompute from "
            f"{commit[:12]}:{gen_path} (got {recomputed[:12]})")
    roles = [lg["host_role"] for lg in legs]
    if roles.count("EVIDENCE") != 1:
        _mr(f"a merged record requires EXACTLY ONE EVIDENCE leg, "
            f"got {roles} -- the evidence-host-only surfaces can be "
            "exercised on exactly one host")
    evidence_host = [lg["host_id"] for lg in legs
                     if lg["host_role"] == "EVIDENCE"][0]
    rows, seen = [], {}
    for lg in legs:
        for r in lg["invocations"]:
            if r.get("source_commit") != commit:
                _mr(f"a row on {lg['host_id']} carries source_commit "
                    f"{str(r.get('source_commit'))[:12]}, not the "
                    f"agreed {commit[:12]} -- every ROW is checked, "
                    "not just the leg header")
            if r.get("host_id") != lg["host_id"]:
                _mr(f"a row claims host {r.get('host_id')!r} inside "
                    f"the {lg['host_id']!r} leg")
            if set(r) != ROW_FIELDS:
                _mr(f"a row on {lg['host_id']} does not match the "
                    f"closed row schema; missing "
                    f"{sorted(ROW_FIELDS - set(r))}, unexpected "
                    f"{sorted(set(r) - ROW_FIELDS)} -- a subset of "
                    "fields makes a row a LABEL, not a record")
            if r["verdict"] not in VERDICT_VOCABULARY:
                _mr(f"row verdict {r['verdict']!r} is outside the "
                    f"declared vocabulary {list(VERDICT_VOCABULARY)}")
            nr = r["verdict"] == "NOT_RUN"
            for f in sorted(ROW_FIELDS):
                nullable = nr and f in NULLABLE_ON_NOT_RUN
                if r[f] in (None, "") and not nullable:
                    _mr(f"row field {f!r} is null on a "
                        f"{r['verdict']} row ({r['surface']}); only "
                        "a NOT_RUN row may omit execution facts")
                if nr and f in NULLABLE_ON_NOT_RUN \
                        and r[f] is not None:
                    _mr(f"row field {f!r} is POPULATED on a NOT_RUN "
                        f"row ({r['surface']}) -- nothing executed, "
                        "so there is no execution fact to record")
                if f in HEX64 and r[f] is not None:
                    v = r[f]
                    if not isinstance(v, str) or len(v) != 64 or \
                            not all(c in "0123456789abcdef"
                                    for c in v):
                        _mr(f"row field {f!r} is not a sha256 digest "
                            f"({v!r})")
            if not isinstance(r["exit_code"], (int, type(None))) \
                    or isinstance(r["exit_code"], bool):
                _mr(f"row exit_code {r['exit_code']!r} is not an int")
            cs = _canonical_surface(r["surface"])
            if cs is None or cs not in SURFACES:
                _mr(f"row surface {r['surface']!r} is not exactly "
                    f"{SURFACE_PREFIX}<declared surface> -- an alias "
                    "path collapses in the cell set while surviving "
                    "the duplicate check")
            if r["interpreter_label"] not in INTERPRETER_LABELS:
                _mr(f"row interpreter {r['interpreter_label']!r} is "
                    f"not one of {list(INTERPRETER_LABELS)}")
            k = (r["host_id"], cs, r["interpreter_label"])
            if k in seen:
                _mr(f"duplicate cell {k}")
            seen[k] = True
            rows.append(r)
    # EXACT cells per leg -- no omissions and no extras. codex's
    # two-one-row case dies here rather than emitting missing=[].
    want = required_cells()
    for lg in legs:
        got = _leg_cells(lg)
        if got != want:
            _mr(f"leg {lg['host_id']} declares {len(got)} cells, not "
                f"the required {len(want)} (SURFACES x "
                f"{list(INTERPRETER_LABELS)}); missing "
                f"{sorted(want - got)[:3]}, unexpected "
                f"{sorted(got - want)[:3]} -- every non-execution "
                "must be an EXPLICIT NOT_RUN row, never an omitted "
                "one")
    counts = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    # a per-surface/interpreter matrix, not a coarse inference over
    # whatever rows happen to be present
    matrix, missing = {}, []
    for sf in SURFACES:
        matrix[sf] = {}
        for il in INTERPRETER_LABELS:
            hit = [r for r in rows
                   if r["surface"].split("/")[-1] == sf
                   and r["interpreter_label"] == il
                   and r["verdict"] in ("PASS", "COVERED_ELSEWHERE")]
            matrix[sf][il] = sorted(r["host_id"] for r in hit)
            if hit:
                continue
            scoped = (sf in EVIDENCE_HOST_ONLY
                      and il != f"py{REQUIRED_INTERPRETERS[-1]}")
            if scoped:
                matrix[sf][il] = "SCOPED_NOT_REQUIRED"
            else:
                missing.append([sf, il])
        # codex 1400Z P0 #2: the requirement is the EVIDENCE ROLE's
        # cell. My first attempt matched a NICKNAME ("devildog")
        # against the real host_id ("Rmath151409/Windows"): it would
        # have refused the genuine evidence leg, and its doctor
        # passed only because the doctor fed it the nickname. A role
        # is declared and validated; a nickname is guessed.
        if sf in EVIDENCE_HOST_ONLY:
            il = f"py{REQUIRED_INTERPRETERS[-1]}"
            by = matrix[sf][il]
            if not isinstance(by, list) or evidence_host not in by:
                if [sf, il] not in missing:
                    missing.append([sf, il])
                matrix[sf][il] = {"covered_by": by,
                                  "required_from": evidence_host,
                                  "status": "NOT_COVERED_BY_"
                                            "EVIDENCE_ROLE"}
    # codex 1400Z P1 #3: `verdict != NOT_RUN` counted a REFUSE as
    # coverage, so an all-REFUSE record reported INCOMPLETE with 26
    # missing cells AND missing_required_interpreters=[] in the same
    # header. Derive from the SAME predicate as the matrix: an
    # interpreter is covered only when every required non-exempt cell
    # for it is covered.
    covered = []
    for v in REQUIRED_INTERPRETERS:
        il = f"py{v}"
        if not any(m[1] == il for m in missing):
            covered.append(v)
    complete = not missing
    return {"schema": MERGED_SCHEMA,
            "coverage_status": ("COMPLETE" if complete
                                else "INCOMPLETE"),
            "coverage_matrix": matrix,
            "missing_required_cells": missing,
            "source_commit": commit,
            "legs": [{"host_id": lg["host_id"],
                      "rows": len(lg["invocations"])}
                     for lg in legs],
            "producer_generator_blob_sha256": gens.pop(),
            "leg_schema": schemas.pop(),
            "invocations": rows,
            "verdict_counts": dict(sorted(counts.items())),
            "interpreters_covered": covered,
            "missing_required_interpreters": [
                v for v in REQUIRED_INTERPRETERS if v not in covered],
            "coverage_claim": (
                "every required surface/interpreter cell is covered"
                if complete else
                f"{len(missing)} required cell(s) NOT covered -- this "
                "record does NOT establish complete dual-interpreter "
                "verification"),
            "claim_scope": "MULTI_HOST_PRE_MANIFEST_VERIFICATION",
            "authorizes": "NOTHING"}


def _merge_selftest():
    """BEHAVIORAL doctors, each MUTATION-TESTED. Legs are built from
    the REAL committed record so a doctor cannot pass against a row
    shape that does not exist -- my nickname guard passed exactly that
    way."""
    real = json.load(open(os.path.join(REPO,
                                       *SUMMARY_PATH.split("/")),
                          encoding="utf-8"))
    COMMIT = real["source_commit"]
    GEN = real["producer_generator_blob_sha256"]
    TEMPLATE = {r["verdict"]: r for r in real["invocations"]}

    def cell(host, sf, il, verdict="PASS"):
        base = TEMPLATE.get(verdict) or TEMPLATE["PASS"]
        r = dict(base)
        r["host_id"] = host
        r["source_commit"] = COMMIT
        r["surface"] = SURFACE_PREFIX + sf
        r["interpreter_label"] = il
        r["verdict"] = verdict
        for f in NULLABLE_ON_NOT_RUN:
            if verdict == "NOT_RUN":
                r[f] = None
            elif r[f] is None:
                r[f] = (0 if f == "exit_code"
                        else "3.11.9" if f == "interpreter_version"
                        else "f" * 64
                        if f == "canonical_executed_sha256"
                        else "x")
        return r

    def full_rows(host, verdict="PASS"):
        return [cell(host, sf, il, verdict)
                for sf in SURFACES for il in INTERPRETER_LABELS]

    def leg(host, commit=None, gen=None, schema=SUMMARY_SCHEMA,
            rows=None, role="PORTABLE", store=None):
        c = COMMIT if commit is None else commit
        return {"host_id": host, "source_commit": c, "repo_head": c,
                "schema": schema, "host_role": role,
                "store_identity": store,
                "producer_generator_blob_sha256":
                    GEN if gen is None else gen,
                "invocations": (full_rows(host) if rows is None
                                else rows)}

    def ev(host="Rmath151409/Windows", **kw):
        return leg(host, role="EVIDENCE", store=V3_STORE_IDENTITY,
                   **kw)

    def refuses(fn, needle):
        try:
            fn()
            return False
        except MergeRefusal as e:
            return needle in str(e)

    def refuses_any(fn):
        """For field-removal/nulling, where an EARLIER guard (the row
        host or row commit check) legitimately fires first. Catching
        only my own typed MergeRefusal means any hit here is a
        DELIBERATE refusal -- there is no accidental-substring risk of
        the kind that made doctor (d) blind."""
        try:
            fn()
            return False
        except MergeRefusal:
            return True
    NCELL = len(SURFACES) * len(INTERPRETER_LABELS)
    ok = merge_legs([ev(), leg("geomen/Windows")])
    assert ok["source_commit"] == COMMIT
    assert len(ok["invocations"]) == 2 * NCELL
    assert ok["coverage_status"] == "COMPLETE", ok["coverage_status"]
    assert ok["missing_required_cells"] == []
    assert ok["missing_required_interpreters"] == []
    # ---- frame ----
    OTHER = "b" * 40
    assert refuses(lambda: merge_legs(
        [ev(), leg("geomen/Windows", commit=OTHER)]),
        "DIFFERENT snapshots")
    bad = leg("geomen/Windows")
    bad["repo_head"] = OTHER
    assert refuses(lambda: merge_legs([ev(), bad]),
                   "did not run from its own committed snapshot")
    sneak = leg("geomen/Windows")
    sneak["invocations"][0] = dict(sneak["invocations"][0],
                                   source_commit=OTHER)
    assert refuses(lambda: merge_legs([ev(), sneak]),
                   "every ROW is checked")
    assert refuses(lambda: merge_legs(
        [ev(), leg("geomen/Windows", gen="h" * 64)]),
        "DIFFERENT generator bytes")
    assert refuses(lambda: merge_legs([ev(), ev()]),
                   "duplicate host")
    assert refuses(lambda: merge_legs([ev()]), "at least two legs")
    liar = leg("geomen/Windows")
    liar["invocations"][0] = dict(liar["invocations"][0],
                                  host_id="Rmath151409/Windows")
    assert refuses(lambda: merge_legs([ev(), liar]), "claims host")
    # ---- codex 1400Z #1: the header is not attestation ----
    assert refuses(lambda: merge_legs(
        [ev(commit=OTHER), leg("geomen/Windows", commit=OTHER)]),
        "does not resolve to a commit")
    assert refuses(lambda: merge_legs(
        [ev(gen="c" * 64), leg("geomen/Windows", gen="c" * 64)]),
        "does not recompute")
    assert refuses(lambda: merge_legs(
        [ev(schema="agreed-but-arbitrary"),
         leg("geomen/Windows", schema="agreed-but-arbitrary")]),
        "is not attestation of either")
    # ---- codex 1400Z #1: closed ROW schema, field by field ----
    for f in sorted(ROW_FIELDS):
        rows = full_rows("geomen/Windows")
        rows[0] = {k: v for k, v in rows[0].items() if k != f}
        assert refuses_any(lambda rows=rows: merge_legs(
            [ev(), leg("geomen/Windows", rows=rows)])), f
        rows2 = full_rows("geomen/Windows")
        rows2[0] = dict(rows2[0], **{f: None})
        assert refuses_any(lambda rows2=rows2: merge_legs(
            [ev(), leg("geomen/Windows", rows=rows2)])), f
    extra = full_rows("geomen/Windows")
    extra[0] = dict(extra[0], unexpected_field=1)
    assert refuses(lambda: merge_legs(
        [ev(), leg("geomen/Windows", rows=extra)]), "unexpected")
    for f in sorted(HEX64):
        rows = full_rows("geomen/Windows")
        rows[0] = dict(rows[0], **{f: "not-a-digest"})
        assert refuses(lambda rows=rows: merge_legs(
            [ev(), leg("geomen/Windows", rows=rows)]),
            "is not a sha256 digest"), f
    # the ALIAS 29th row codex injected
    alias = full_rows("geomen/Windows")
    alias.append(dict(alias[0],
                      surface=SURFACE_PREFIX + "alias/" + SURFACES[0]))
    assert refuses(lambda: merge_legs(
        [ev(), leg("geomen/Windows", rows=alias)]),
        "is not exactly")
    # a NOT_RUN row that carries execution facts it cannot have
    pop = full_rows("geomen/Windows", "NOT_RUN")
    pop[0] = dict(pop[0], exit_code=0)
    assert refuses(lambda: merge_legs(
        [ev(), leg("geomen/Windows", rows=pop)]),
        "POPULATED on a NOT_RUN row")
    # ---- completeness ----
    tiny = [cell("geomen/Windows", SURFACES[0], "py3.14")]
    assert refuses(lambda: merge_legs(
        [ev(), leg("geomen/Windows", rows=tiny)]),
        "every non-execution must be an EXPLICIT NOT_RUN row")
    short = full_rows("geomen/Windows")[:-1]
    assert refuses(lambda: merge_legs(
        [ev(), leg("geomen/Windows", rows=short)]), "declares 27")
    vbad = full_rows("geomen/Windows")
    vbad[0] = dict(vbad[0], verdict="GREEN")
    assert refuses(lambda: merge_legs(
        [ev(), leg("geomen/Windows", rows=vbad)]),
        "outside the declared vocabulary")
    # ---- codex 1400Z #2: the ROLE, not a nickname ----
    assert refuses(lambda: merge_legs(
        [leg("Rmath151409/Windows"), leg("geomen/Windows")]),
        "EXACTLY ONE EVIDENCE leg")
    assert refuses(lambda: merge_legs([ev(), ev("geomen/Windows")]),
                   "EXACTLY ONE EVIDENCE leg")
    assert refuses(lambda: merge_legs(
        [leg("Rmath151409/Windows", role="EVIDENCE", store="wrong"),
         leg("geomen/Windows")]), "not 's4t-w2-capture-20260825'")
    assert refuses(lambda: merge_legs(
        [ev(), leg("geomen/Windows", store="s4t-w2-capture-20260825")
         ]), "yet names a store")
    # codex's exact case: the EVIDENCE leg did NOT run the
    # store-dependent surfaces and the PORTABLE leg claims it did
    d_rows = [cell("Rmath151409/Windows", sf, il,
                   "NOT_RUN" if sf in EVIDENCE_HOST_ONLY else "PASS")
              for sf in SURFACES for il in INTERPRETER_LABELS]
    wrong = merge_legs([ev(rows=d_rows), leg("geomen/Windows")])
    assert wrong["coverage_status"] == "INCOMPLETE", wrong[
        "coverage_status"]
    gaps = {tuple(x) for x in wrong["missing_required_cells"]}
    PY = f"py{REQUIRED_INTERPRETERS[-1]}"
    for sf in EVIDENCE_HOST_ONLY:
        assert (sf, PY) in gaps, sf
        assert wrong["coverage_matrix"][sf][PY]["status"] == \
            "NOT_COVERED_BY_EVIDENCE_ROLE"
    # ---- codex 1400Z #3: the scalar header may not contradict the
    # cell ledger ----
    ref_d = full_rows("Rmath151409/Windows", "REFUSE")
    ref_g = full_rows("geomen/Windows", "REFUSE")
    allref = merge_legs([ev(rows=ref_d),
                         leg("geomen/Windows", rows=ref_g)])
    assert allref["coverage_status"] == "INCOMPLETE"
    assert allref["missing_required_cells"]
    assert sorted(allref["missing_required_interpreters"]) == \
        sorted(REQUIRED_INTERPRETERS), \
        allref["missing_required_interpreters"]
    assert allref["interpreters_covered"] == []
    print("w2 leg-merge selftest: ALL PASS "
          f"({7 + 3 + 2 * len(ROW_FIELDS) + len(HEX64) + 10} "
          "directions -- frame, closed row schema, alias, "
          "completeness, evidence ROLE, ledger consistency)")


USAGE = """w2_verification_run_summary_grassmann

  --generate         build and WRITE the canonical record (the ONLY
                     writing path; atomic replace after a complete
                     build)
  --merge-selftest   read-only; the leg-merge refusal directions
  --argv-selftest    read-only; proves unknown flags cannot write

Any other argument, and the no-argument invocation, REFUSE with exit
2 before build() and before any file is opened."""


def main():
    """codex 1327Z P0 / cayley 1321Z: the writing path is now
    EXPLICIT and the grammar is CLOSED.

    It used to be `if "--merge-selftest" in argv: ... else: main()`,
    so ANY unrecognised argument -- `--selftest`, a typo, an unrelated
    flag -- ran the destructive build and OVERWROTE the canonical
    record with the current host's, exit 0, no warning. cayley hit it
    for real: their `--selftest` (the convention every other module in
    this tree uses) silently replaced the devildog record with a
    geomen one at the same path. They caught it in git status and
    nothing was pushed, but that is precisely the host-substitution
    the multi-host schema exists to make visible -- and my own default
    performed it silently.
    """
    s = build()
    out = os.path.join(REPO, *SUMMARY_PATH.split("/"))
    # atomic: a complete build lands or nothing does. A crash midway
    # through json.dump used to leave a truncated canonical record.
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(s, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, out)
    print("verdicts:", s["verdict_counts"])
    for r in s["invocations"]:
        if r["verdict"] != "PASS":
            print(f"  {r['verdict']:17s} {r['interpreter_label']} "
                  f"{os.path.basename(r['surface'])}")
    print("written:", SUMMARY_PATH)


def _argv_selftest():
    """Prove -- by SUBPROCESS, not by reading argv handling -- that a
    mistyped flag cannot perform a destructive write."""
    import subprocess
    out = os.path.join(REPO, *SUMMARY_PATH.split("/"))
    # SELF-CATCH: this used to read `before = None if absent`, and
    # then compare `after == before` -- so on any host WITHOUT the
    # record it compared None to None and passed having observed
    # NOTHING. A destructive-write doctor that cannot see the file it
    # protects is the "the case never got there" defect I had just
    # warned cayley about, in my own code, one turn later. The record
    # is committed, so its absence means the doctor cannot certify.
    if not os.path.isfile(out):
        raise RuntimeError(
            "the canonical record is ABSENT, so this doctor cannot "
            "observe whether a bad flag modified it -- refusing "
            "rather than passing vacuously")
    with open(out, "rb") as f:
        before = hashlib.sha256(f.read()).hexdigest()
    me = os.path.abspath(__file__)
    for bad in (["--selftest"], ["--bogus"], [], ["--generate=x"],
                ["--Generate"], ["--merge-selftest", "--generate"]):
        r = subprocess.run([sys.executable, me] + bad,
                           capture_output=True)
        assert r.returncode == 2, (bad, r.returncode)
        assert os.path.isfile(out), f"{bad} DELETED the record"
        with open(out, "rb") as f:
            after = hashlib.sha256(f.read()).hexdigest()
        assert after == before, f"{bad} MODIFIED the record"
        assert not os.path.exists(out + ".tmp"), f"{bad} left a temp"
    print("w2 argv selftest: ALL PASS (6 refusing argv shapes, "
          "canonical record byte-unchanged)")


def _cli(argv):
    """Closed grammar: exactly one recognised verb, or exit 2."""
    verbs = [a for a in argv[1:] if a.startswith("--")]
    known = {"--generate", "--merge-selftest", "--argv-selftest"}
    if len(argv) > 1 and (set(verbs) - known or len(argv) != 2):
        sys.stderr.write(f"REFUSED: unrecognised arguments "
                         f"{argv[1:]}\n{USAGE}\n")
        return 2
    if len(argv) != 2:
        sys.stderr.write("REFUSED: a verb is required; the writing "
                         "path is never the default\n" + USAGE + "\n")
        return 2
    if argv[1] == "--merge-selftest":
        _merge_selftest()
    elif argv[1] == "--argv-selftest":
        _argv_selftest()
    elif argv[1] == "--generate":
        main()
    else:
        sys.stderr.write(f"REFUSED: unknown verb {argv[1]!r}\n"
                         + USAGE + "\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv))
