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


EXPECTED_STORE_DESCRIPTOR = (
    "docs/f2g_window2_execution/w2_expected_store_descriptor.json")


def expected_store(commitish):
    """The committed authority for WHICH store counts. Resolved from
    a commit, never from a constant in the running module."""
    raw = subprocess.run(
        ["git", "-C", REPO, "cat-file", "blob",
         f"{commitish}:{EXPECTED_STORE_DESCRIPTOR}"],
        capture_output=True).stdout
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def attest_store(commitish="HEAD"):
    """codex 1534Z P1 #3: EVIDENCE was inferred from
    `os.path.isdir(V3_STORE)`, so ANY existing empty or mispointed
    directory labelled this host EVIDENCE and attached the named
    store identity from a CONSTANT -- a header fact stronger than the
    measurement behind it. Fail-closed: the role is earned by
    CONTENT-ADDRESS verifying the store, or it is not claimed.

    Returns (role, store_identity, attestation).
    """
    import w2_restage_v4_grassmann as _RES

    def portable(why):
        return "PORTABLE", None, {"attested": False, "reason": why}
    root = _RES.V3_STORE
    if not os.path.isdir(root):
        return portable(f"no store directory at {root}")
    names = sorted(f for f in os.listdir(root) if f.endswith(".body"))
    if not names:
        return portable(f"{root} exists but holds no bodies -- an "
                        "empty directory is not an evidence store")
    verified = 0
    for n in names:
        sha = n[:-len(".body")]
        if not _is_hex(sha, 64):
            return portable(f"{n} is not content-addressed")
        with open(os.path.join(root, n), "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() != sha:
                return portable(f"{n} does not match its content "
                                "address")
        verified += 1
    digest = hashlib.sha256(json.dumps(
        names, separators=(",", ":")).encode()).hexdigest()
    # codex 1617Z: INTEGRITY is not IDENTITY. Every body matching its
    # content address says the directory is internally consistent --
    # a single valid body satisfies that -- while the store_id came
    # from a CONSTANT with nothing compared against it. Membership
    # must match a COMMITTED descriptor exactly.
    exp = expected_store(commitish)
    if not isinstance(exp, dict):
        return portable("the expected-store descriptor is absent at "
                        "HEAD; identity cannot be claimed without "
                        "an authority to claim it against")
    if len(names) != exp.get("body_count") or \
            digest != exp.get("body_name_set_digest"):
        return portable(
            f"this directory holds {len(names)} bodies with name-set "
            f"digest {digest[:12]}, but {exp.get('store_id')!r} is "
            f"{exp.get('body_count')} / "
            f"{str(exp.get('body_name_set_digest'))[:12]} -- "
            "internally consistent is not the named store")
    return "EVIDENCE", exp["store_id"], {
        "attested": True,
        "body_count": len(names),
        "bodies_verified": verified,
        "name_set_digest": digest,
        "expected_store_matched": True}


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
    host_role, store_identity, store_attestation = attest_store()
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
            "store_attestation": store_attestation,
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


DIGEST_DOMAIN = "UTF8_SOURCE_LF_V1"
_HEX = "0123456789abcdef"


def _is_hex(v, n):
    return isinstance(v, str) and len(v) == n and \
        all(c in _HEX for c in v)


def _validate_leg_header(lg, n):
    """codex 1534Z P1 #2: malformed HEADERS escaped as raw TypeError.
    An equal integer source_commit/repo_head sliced
    (`'int' object is not subscriptable`); a list host_id hit
    `unhashable type: 'list'`. My earlier shape guard covered
    CONTAINERS but not FIELD TYPES, so the typed contract still had a
    hole one level in. Pure, and runs before any set/slice op."""
    for f in ("host_id", "schema", "host_role"):
        if not isinstance(lg.get(f), str) or not lg[f].strip():
            _mr(f"leg {n} field {f!r} is "
                f"{type(lg.get(f)).__name__}, not a non-empty string")
    for f in ("source_commit", "repo_head"):
        if not _is_hex(lg.get(f), 40):
            _mr(f"leg {n} field {f!r} is not a 40-hex commit "
                f"({lg.get(f)!r})")
    if not _is_hex(lg.get("producer_generator_blob_sha256"), 64):
        _mr(f"leg {n} producer_generator_blob_sha256 is not a "
            "64-hex digest")
    st = lg.get("store_identity")
    if st is not None and (not isinstance(st, str) or not st.strip()):
        _mr(f"leg {n} store_identity is "
            f"{type(st).__name__}, not a string or None")


def _validate_row_values(r, n):
    """Every row field's TYPE and FORMAT. codex accepted
    argv='not-a-list', run_utc=7, git_blob_oid=7, digest_domain=[],
    tail={}, resolved_executable=7 and interpreter_version=[] one at
    a time -- presence and 64-hex shape were checked, the rest were
    not."""
    nr = r["verdict"] == "NOT_RUN"
    for f in ("host_id", "surface", "interpreter_label", "verdict",
              "digest_domain"):
        if not isinstance(r.get(f), str) or not r[f].strip():
            _mr(f"row field {f!r} is {type(r.get(f)).__name__}, not "
                "a non-empty string")
    if r["digest_domain"] != DIGEST_DOMAIN:
        _mr(f"row digest_domain {r['digest_domain']!r} is not "
            f"{DIGEST_DOMAIN!r}")
    if not isinstance(r.get("tail"), str):
        _mr(f"row tail is {type(r.get('tail')).__name__}, not a "
            "string")
    if not _is_hex(r.get("git_blob_oid"), 40):
        _mr(f"row git_blob_oid is not a 40-hex object id "
            f"({r.get('git_blob_oid')!r})")
    if not isinstance(r.get("run_utc"), str) or \
            not r["run_utc"].endswith("Z"):
        _mr(f"row run_utc {r.get('run_utc')!r} is not a UTC "
            "timestamp string")
    if nr:
        return
    a = r.get("argv")
    if not isinstance(a, list) or not a or \
            not all(isinstance(x, str) for x in a):
        _mr(f"row argv is {type(a).__name__}, not a non-empty list "
            "of strings")
    if not isinstance(r.get("resolved_executable"), str) or \
            not r["resolved_executable"].strip():
        _mr("row resolved_executable is not a non-empty string")
    iv = r.get("interpreter_version")
    if not isinstance(iv, str) or not iv.strip():
        _mr(f"row interpreter_version is {type(iv).__name__}, not a "
            "string")
    want = r["interpreter_label"][2:]
    if not iv.startswith(want + "."):
        _mr(f"row interpreter_version {iv!r} does not match its "
            f"declared label {r['interpreter_label']!r}")


_BINDINGS_BY_COMMIT = {}


def _committed_bindings(commit, cache):
    """The 14 declared surfaces at the AGREED commit, recomputed.

    Cached PER COMMIT at module level: a commit's blobs are
    immutable, and re-deriving them inside every merge_legs() call
    meant 28 git subprocesses per call -- ~2660 across the selftest,
    which is why codex's combined 3.11/3.14 run exceeded their
    300-second budget without returning a verdict. That timeout was
    my regression, not their environment.
    """
    if commit in _BINDINGS_BY_COMMIT:
        return _BINDINGS_BY_COMMIT[commit]
    cache = _BINDINGS_BY_COMMIT.setdefault(commit, {})
    for sf in SURFACES:
        path = SURFACE_PREFIX + sf
        oid = subprocess.run(
            ["git", "-C", REPO, "rev-parse", f"{commit}:{path}"],
            capture_output=True)
        raw = subprocess.run(
            ["git", "-C", REPO, "cat-file", "blob", f"{commit}:{path}"],
            capture_output=True)
        if oid.returncode != 0 or raw.returncode != 0 or not raw.stdout:
            _mr(f"{path} is absent at the agreed commit "
                f"{commit[:12]}")
        cache[sf] = {
            "git_blob_oid": oid.stdout.decode().strip(),
            "blob_sha256": hashlib.sha256(raw.stdout).hexdigest(),
            "canonical_committed_sha256":
                hashlib.sha256(_canon_bytes(raw.stdout)).hexdigest()}
    return cache


def _validate_row_provenance(r, sf, commit, cache):
    """codex 1534Z P0 #1: exact field PRESENCE is not exact
    EXECUTION/PROVENANCE binding. A cloned real record was accepted
    with exit_code=9 on a PASS, and with canonical_executed_sha256 or
    blob_sha256 set to 64 zeros -- the hex check proves SYNTAX, not
    the binding, so a PASS could name non-executed bytes or an
    unsuccessful process and still cover a required cell.

    The three committed bindings are RECOMPUTED from the agreed
    commit. disk_sha256, resolved_executable and run_utc stay
    self-attested: the merger cannot reopen another host's disk, and
    pretending otherwise would be a stronger claim than the evidence.
    """
    want = _committed_bindings(commit, cache)[sf]
    for f, v in want.items():
        if r[f] != v:
            _mr(f"row {f!r} for {sf} is {str(r[f])[:12]}, but "
                f"{commit[:12]} gives {v[:12]} -- a recomputable "
                "binding must be RECOMPUTED, not accepted as stated")
    v = r["verdict"]
    if v in ("PASS", "COVERED_ELSEWHERE"):
        if r["exit_code"] != 0:
            _mr(f"a {v} row for {sf} carries exit_code "
                f"{r['exit_code']!r} -- a successful verdict over an "
                "unsuccessful process is not a result")
        if r["canonical_executed_sha256"] != \
                r["canonical_committed_sha256"]:
            _mr(f"a {v} row for {sf} executed bytes that differ from "
                "the committed blob; committed==executed is what "
                "makes the verdict bind to the source")
    elif v == "REFUSE":
        diverged = (r["canonical_executed_sha256"] is not None and
                    r["canonical_executed_sha256"] !=
                    r["canonical_committed_sha256"])
        if r["exit_code"] == 0 and not diverged:
            _mr(f"a REFUSE row for {sf} has exit_code 0 and matching "
                "digests -- nothing about it refused")


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
    try:
        legs = list(legs)
    except TypeError:
        _mr(f"legs is not iterable ({type(legs).__name__})")
    if len(legs) < 2:
        _mr(f"a merge needs at least two legs, got {len(legs)}")
    # cayley 1532Z near-finding, made real: their untyped KeyError
    # traced to their own shell quoting, but probing the SHAPE space
    # found five of eight malformed inputs leaking AttributeError or
    # TypeError out of this function. An untyped exception is NOT a
    # refusal: a caller catching MergeRefusal crashes instead of
    # receiving a verdict, and a crash produces no typed result while
    # leaving whatever it was going to replace looking untouched --
    # the same class as the inventory KeyError I repaired earlier.
    # Shapes are therefore checked BEFORE any field access.
    for n, lg in enumerate(legs):
        if not isinstance(lg, dict):
            _mr(f"leg {n} is a {type(lg).__name__}, not a record")
        inv = lg.get("invocations")
        if not isinstance(inv, list):
            _mr(f"leg {n} has invocations of type "
                f"{type(inv).__name__}, not a list")
        for m, r in enumerate(inv):
            if not isinstance(r, dict):
                _mr(f"leg {n} row {m} is a {type(r).__name__}, not "
                    "a record")
        _validate_leg_header(lg, n)
    commits, gens, schemas, hosts = set(), set(), set(), []
    _evidence_att = []
    for i, lg in enumerate(legs):
        for f in ("source_commit", "repo_head", "host_id",
                  "producer_generator_blob_sha256", "schema",
                  "host_role", "store_identity",
                  "store_attestation", "invocations"):
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
        att = lg.get("store_attestation")
        if not isinstance(att, dict):
            _mr(f"leg {i} carries no store_attestation; a role must "
                "be earned by measurement, not declared")
        if lg["host_role"] == "EVIDENCE":
            if not att.get("attested"):
                _mr(f"leg {i} claims the EVIDENCE role with an "
                    f"UNATTESTED store ({att.get('reason')!r})")
            bc, bv = att.get("body_count"), att.get("bodies_verified")
            if not isinstance(bc, int) or bc <= 0 or bv != bc:
                _mr(f"leg {i} attests body_count={bc!r} "
                    f"bodies_verified={bv!r}; every body must be "
                    "content-address verified")
            if not _is_hex(att.get("name_set_digest"), 64):
                _mr(f"leg {i} attestation carries no name-set digest")
            _evidence_att.append((i, lg, att))
        elif att.get("attested"):
            _mr(f"leg {i} is {lg['host_role']} yet attests a store")
        # NOTE: the store NAME is no longer compared against a
        # module CONSTANT here. codex 1617Z: a constant supplying
        # identity IS the defect, not the check. The committed
        # descriptor at the agreed commit is the sole authority,
        # and that comparison happens below, once commit is known.
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
    # codex 1617Z merger half: resolve the expected store from the
    # AGREED commit and require exact membership. Positive counts,
    # equal counts, 64-hex shape and a constant name all held while
    # a 1/1 attestation with a shaped-but-wrong digest was accepted.
    exp = expected_store(commit)
    if not isinstance(exp, dict):
        _mr(f"the expected-store descriptor is absent at "
            f"{commit[:12]}; an EVIDENCE role cannot be validated "
            "without the authority it claims membership of")
    for i, lg, att in _evidence_att:
        if lg["store_identity"] != exp.get("store_id"):
            _mr(f"leg {i} names store {lg['store_identity']!r}, but "
                f"{commit[:12]} declares {exp.get('store_id')!r}")
        if att.get("body_count") != exp.get("body_count") or \
                att.get("name_set_digest") != \
                exp.get("body_name_set_digest"):
            _mr(f"leg {i} attests {att.get('body_count')} bodies / "
                f"{str(att.get('name_set_digest'))[:12]}, but "
                f"{exp.get('store_id')!r} at {commit[:12]} is "
                f"{exp.get('body_count')} / "
                f"{str(exp.get('body_name_set_digest'))[:12]} -- "
                "content-address integrity is not MEMBERSHIP")
    roles = [lg["host_role"] for lg in legs]
    if roles.count("EVIDENCE") != 1:
        _mr(f"a merged record requires EXACTLY ONE EVIDENCE leg, "
            f"got {roles} -- the evidence-host-only surfaces can be "
            "exercised on exactly one host")
    evidence_host = [lg["host_id"] for lg in legs
                     if lg["host_role"] == "EVIDENCE"][0]
    rows, seen, _bind_cache = [], {}, {}
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
            _validate_row_values(r, lg["host_id"])
            _validate_row_provenance(r, cs, commit, _bind_cache)
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
    PY311 = f"py{REQUIRED_INTERPRETERS[-1]}"
    BY_CELL = {(r["surface"].split("/")[-1], r["interpreter_label"]):
               r for r in real["invocations"]}

    def cell(host, sf, il, verdict="PASS"):
        """Built from the REAL row for THAT EXACT CELL, so a fixture
        cannot assert a shape the artifact does not have. My previous
        version copied one template row across every surface and
        interpreter, which produced a py3.14 row carrying 3.11.9 --
        the validator caught my own fixture, which is the point."""
        base = dict(BY_CELL[(sf, il)])
        # committed bindings are per-SURFACE; the py3.11 row always
        # carries them non-null
        src = BY_CELL[(sf, PY311)]
        for f in ("git_blob_oid", "blob_sha256",
                  "canonical_committed_sha256", "disk_sha256"):
            base[f] = src[f]
        base["digest_domain"] = DIGEST_DOMAIN
        base["host_id"] = host
        base["source_commit"] = COMMIT
        base["surface"] = SURFACE_PREFIX + sf
        base["interpreter_label"] = il
        base["verdict"] = verdict
        base["run_utc"] = real["invocations"][0]["run_utc"]
        if verdict == "NOT_RUN":
            for f in NULLABLE_ON_NOT_RUN:
                base[f] = None
            return base
        base["argv"] = [f"py-{il}", SURFACE_PREFIX + sf]
        base["resolved_executable"] = f"C:/Python/{il}/python.exe"
        base["interpreter_version"] = il[2:] + ".9"
        base["canonical_executed_sha256"] = \
            base["canonical_committed_sha256"]
        base["exit_code"] = 0 if verdict in ("PASS",
                                             "COVERED_ELSEWHERE") \
            else 1
        return base

    def full_rows(host, verdict="PASS"):
        return [cell(host, sf, il, verdict)
                for sf in SURFACES for il in INTERPRETER_LABELS]

    _EXP = expected_store(COMMIT) or {}
    ATT_OK = {"attested": True,
              "body_count": _EXP.get("body_count"),
              "bodies_verified": _EXP.get("body_count"),
              "name_set_digest": _EXP.get("body_name_set_digest"),
              "expected_store_matched": True}
    ATT_NO = {"attested": False, "reason": "no store on this host"}

    def leg(host, commit=None, gen=None, schema=SUMMARY_SCHEMA,
            rows=None, role="PORTABLE", store=None, att=None):
        c = COMMIT if commit is None else commit
        return {"host_id": host, "source_commit": c, "repo_head": c,
                "schema": schema, "host_role": role,
                "store_identity": store,
                "store_attestation":
                    (ATT_OK if role == "EVIDENCE" else ATT_NO)
                    if att is None else att,
                "producer_generator_blob_sha256":
                    GEN if gen is None else gen,
                "invocations": (full_rows(host) if rows is None
                                else rows)}

    def ev(host="Rmath151409/Windows", **kw):
        kw.setdefault("store", _EXP.get("store_id"))
        return leg(host, role="EVIDENCE", **kw)

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
    # (S) SHAPE refusals: an untyped exception is not a refusal
    def twin(**kw):
        t = leg("geomen/Windows")
        t.update(kw)
        return t
    for name, bad in (
            ("invocations not a list", twin(invocations={"a": 1})),
            ("row is an int", twin(invocations=[1, 2, 3])),
            ("row is a string", twin(invocations=["row"])),
            ("row is None", twin(invocations=[None])),
            ("leg is None", None),
            ("leg is a string", "leg"),
            ("leg is a list", [1, 2])):
        try:
            merge_legs([ev(), bad])
            raise AssertionError(f"{name} was ACCEPTED")
        except MergeRefusal:
            pass
    for bad_legs in (None, 42, "legs"):
        try:
            merge_legs(bad_legs)
            raise AssertionError(f"{bad_legs!r} was ACCEPTED")
        except MergeRefusal:
            pass

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
        [ev(), leg("geomen/Windows", gen="a" * 64)]),
        "DIFFERENT generator bytes")
    # a non-hex generator digest is refused by the HEADER validator
    # before it can even be compared across legs
    assert refuses(lambda: merge_legs(
        [ev(), leg("geomen/Windows", gen="h" * 64)]),
        "not a 64-hex digest")
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
    # ---- codex 1534Z P1 #3: the ROLE is earned, not declared ----
    assert refuses(lambda: merge_legs(
        [ev(att=ATT_NO), leg("geomen/Windows")]),
        "UNATTESTED store")
    for bad_att in ({"attested": True, "body_count": 0,
                     "bodies_verified": 0,
                     "name_set_digest": "a" * 64},
                    {"attested": True, "body_count": 1405,
                     "bodies_verified": 1404,
                     "name_set_digest": "a" * 64},
                    {"attested": True, "body_count": 1405,
                     "bodies_verified": 1405,
                     "name_set_digest": "nope"}):
        assert refuses(lambda b=bad_att: merge_legs(
            [ev(att=b), leg("geomen/Windows")]),
            "content-address verified"
            if bad_att.get("bodies_verified") != 1405
            else "name-set digest"), bad_att
    assert refuses(lambda: merge_legs(
        [ev(), leg("geomen/Windows", att=ATT_OK)]),
        "yet attests a store")
    # codex 1617Z doctor 4: an EVIDENCE header with 1/1 and a
    # shaped-but-WRONG digest was ACCEPTED and returned an ordinary
    # INCOMPLETE record -- shape and self-consistency without
    # MEMBERSHIP.
    assert refuses(lambda: merge_legs(
        [ev(att={"attested": True, "body_count": 1,
                 "bodies_verified": 1,
                 "name_set_digest": "0" * 64}),
         leg("geomen/Windows")]),
        "content-address integrity is not MEMBERSHIP")
    assert refuses(lambda: merge_legs(
        [ev(att=dict(ATT_OK, body_count=1404, bodies_verified=1404)),
         leg("geomen/Windows")]),
        "content-address integrity is not MEMBERSHIP")
    assert refuses(lambda: merge_legs(
        [ev(store="some-other-store"), leg("geomen/Windows")]),
        "declares 's4t-w2-capture-20260825'")
    assert refuses(lambda: merge_legs(
        [ev(att="not-a-dict"), leg("geomen/Windows")]),
        "carries no store_attestation")

    # ---- codex 1534Z P0 #1: presence is not PROVENANCE binding ----
    def one_row(**over):
        rows = full_rows("geomen/Windows")
        rows[0] = dict(rows[0], **over)
        return lambda: merge_legs(
            [ev(), leg("geomen/Windows", rows=rows)])
    # codex's exact three accepted forgeries
    assert refuses(one_row(exit_code=9),
                   "over an unsuccessful process is not a result")
    assert refuses(one_row(canonical_executed_sha256="0" * 64),
                   "committed==executed")
    assert refuses(one_row(blob_sha256="0" * 64),
                   "must be RECOMPUTED")
    # every recomputable binding, independently
    for f in ("git_blob_oid",):
        assert refuses(one_row(**{f: "0" * 40}), "must be RECOMPUTED")
    assert refuses(one_row(canonical_committed_sha256="0" * 64),
                   "must be RECOMPUTED")
    # a REFUSE that refused nothing
    rr = full_rows("geomen/Windows", "REFUSE")
    rr[0] = dict(rr[0], exit_code=0)
    rr[0]["canonical_executed_sha256"] = \
        rr[0]["canonical_committed_sha256"]
    assert refuses(lambda: merge_legs(
        [ev(), leg("geomen/Windows", rows=rr)]),
        "nothing about it refused")
    # ---- codex 1534Z P1 #2: every field TYPE, always MergeRefusal --
    for over, needle in (
            ({"argv": "not-a-list"}, "not a non-empty list"),
            ({"argv": []}, "not a non-empty list"),
            ({"argv": [1, 2]}, "not a non-empty list"),
            ({"run_utc": 7}, "not a UTC timestamp"),
            ({"run_utc": "no-zed"}, "not a UTC timestamp"),
            ({"git_blob_oid": 7}, "not a 40-hex object id"),
            ({"digest_domain": []}, "not a non-empty string"),
            ({"digest_domain": "OTHER"}, "is not 'UTF8_SOURCE_LF_V1'"),
            ({"tail": {}}, "not a string"),
            ({"resolved_executable": 7}, "not a non-empty string"),
            ({"interpreter_version": []}, "not a string"),
            ({"interpreter_version": "9.9.9"},
             "does not match its declared label")):
        assert refuses(one_row(**over), needle), over
    # header types -- codex's two raw TypeErrors
    for over in ({"source_commit": 7, "repo_head": 7},
                 {"host_id": ["a"]}, {"host_id": ""},
                 {"schema": 7}, {"host_role": None},
                 {"store_identity": 7},
                 {"producer_generator_blob_sha256": 7}):
        h = leg("geomen/Windows")
        h.update(over)
        try:
            merge_legs([ev(), h])
            raise AssertionError(f"{over} was ACCEPTED")
        except MergeRefusal:
            pass
        except Exception as e:
            raise AssertionError(
                f"{over} raised {type(e).__name__}, not MergeRefusal")

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
        [leg("Rmath151409/Windows", role="EVIDENCE", store="wrong",
             att=ATT_OK), leg("geomen/Windows")]),
        "declares 's4t-w2-capture-20260825'")
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
          f"({7 + 3 + 2 * len(ROW_FIELDS) + len(HEX64) + 10 + 10 + 19 + 8} "
          "directions -- frame, shape, closed row schema, alias, "
          "PROVENANCE binding, field types, store attestation, "
          "completeness, evidence ROLE, ledger consistency)")


USAGE = """w2_verification_run_summary_grassmann

  --generate         build and WRITE the canonical record (the ONLY
                     writing path; atomic replace after a complete
                     build)
  --merge-selftest   read-only; the leg-merge refusal directions
  --argv-selftest    read-only; proves unknown flags cannot write
  --store-selftest   read-only; proves store INTEGRITY is not store
                     IDENTITY (a valid subset must not earn EVIDENCE)

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


def _store_selftest():
    """codex 1617Z's four locks. Three are filesystem-level on the
    PRODUCER path; the fourth is on the MERGER path and lives in the
    merge selftest. Each must DEGRADE to PORTABLE, never earn the
    named identity."""
    import shutil
    import subprocess as sp
    import tempfile
    import w2_restage_v4_grassmann as _RES
    real = _RES.V3_STORE
    names = sorted(f for f in os.listdir(real) if f.endswith(".body"))
    assert len(names) > 3, real
    me = os.path.abspath(__file__)
    # SELF-CATCH: this used to `import
    # w2_verification_run_summary_grassmann` BY NAME, so a mutated
    # copy of this file saved under any other filename spawned
    # children that imported the UNMUTATED original -- the doctor
    # measured a different module than the one it ships in, and both
    # membership mutations came back BLIND for that reason rather
    # than because the guards were sound. Load THIS file by PATH.
    code = ("import importlib.util as u,sys;"
            "sp=u.spec_from_file_location('m'," + repr(me) + ");"
            "S=u.module_from_spec(sp);"
            "sys.path.insert(0," + repr(os.path.dirname(me)) + ");"
            "sp.loader.exec_module(S);"
            "r,i,a=S.attest_store();print(r,'|',i,'|',"
            "a.get('reason',''))")

    def role_at(root):
        env = dict(os.environ, W2_V3_STORE=root)
        r = sp.run([sys.executable, "-c", code], capture_output=True,
                   env=env, cwd=os.path.dirname(me))
        return r.stdout.decode().strip()
    tmps = []
    try:
        # (1) ONE valid body -- internally consistent, not the store
        one = tempfile.mkdtemp(prefix="store_one_")
        tmps.append(one)
        shutil.copy2(os.path.join(real, names[0]),
                     os.path.join(one, names[0]))
        # (2) a PROPER SUBSET of valid bodies
        sub = tempfile.mkdtemp(prefix="store_sub_")
        tmps.append(sub)
        for n in names[:3]:
            shutil.copy2(os.path.join(real, n),
                         os.path.join(sub, n))
        # (3) the FULL set PLUS one extra valid body (hardlinks so
        #     this stays cheap)
        sup = tempfile.mkdtemp(prefix="store_sup_")
        tmps.append(sup)
        for n in names:
            try:
                os.link(os.path.join(real, n), os.path.join(sup, n))
            except OSError:
                shutil.copy2(os.path.join(real, n),
                             os.path.join(sup, n))
        extra = hashlib.sha256(b"an extra valid body").hexdigest()
        with open(os.path.join(sup, extra + ".body"), "wb") as f:
            f.write(b"an extra valid body")
        for label, root in (("one valid body", one),
                            ("proper subset", sub),
                            ("full set + one extra", sup)):
            out = role_at(root)
            assert out.startswith("PORTABLE"), f"{label}: {out}"
            assert "is not the named store" in out or \
                "internally consistent is not" in out, \
                f"{label}: {out}"
            print(f"  {label:22s} -> PORTABLE (degraded, reason "
                  "stated)")
        # the REAL store still earns it
        out = role_at(real)
        assert out.startswith("EVIDENCE"), out
        print(f"  {'the real store':22s} -> EVIDENCE")
    finally:
        for t in tmps:
            shutil.rmtree(t, ignore_errors=True)
    # (4) no AUTHORITY at the named commit -> the role cannot be
    # claimed at all. Parameterised so this guard is doctorable:
    # with a hard-coded "HEAD" the descriptor always existed, and a
    # guard that cannot be reached is a guard that cannot be shown
    # to work.
    add = sp.run(["git", "-C", REPO, "log", "--diff-filter=A",
                  "--format=%H", "--", EXPECTED_STORE_DESCRIPTOR],
                 capture_output=True).stdout.decode().split()
    assert add, "the descriptor has no add-commit"
    before = sp.run(["git", "-C", REPO, "rev-parse", add[-1] + "^"],
                    capture_output=True).stdout.decode().strip()
    role, ident, att = attest_store(before)
    assert role == "PORTABLE" and ident is None, (role, ident)
    assert "descriptor is absent" in att.get("reason", ""), att
    print(f"  {'no authority at commit':22s} -> PORTABLE "
          f"({before[:8]} predates the descriptor)")
    print("w2 store-membership selftest: ALL PASS (3 degrade "
          "directions + no-authority direction + the real store "
          "still earns the role)")


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
    known = {"--generate", "--merge-selftest",
             "--argv-selftest", "--store-selftest"}
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
    elif argv[1] == "--store-selftest":
        _store_selftest()
    elif argv[1] == "--generate":
        main()
    else:
        sys.stderr.write(f"REFUSED: unknown verb {argv[1]!r}\n"
                         + USAGE + "\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv))
