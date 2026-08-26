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
            "http_requests": _require_measured(_inv)[0],
            "http_counter_source": _require_measured(_inv)[1]}



# codex 1327Z P0 #2: snapshot agreement is not SUPPORT COMPLETENESS.
# merge_legs() enforced that legs agree on a frame; it accepted two
# ONE-ROW legs (one 3.11 row, one 3.14 row) and emitted
# `missing_required_interpreters: []` -- twenty-six required cells
# absent while the header read as complete coverage. A closed cell
# contract makes that unsayable: a leg declares SURFACES x
# INTERPRETERS exactly, and every non-execution is an EXPLICIT
# NOT_RUN row rather than an omitted one, because an omitted row is
# indistinguishable from a row nobody thought to require.
VERDICT_VOCABULARY = ("PASS", "COVERED_ELSEWHERE", "REFUSE", "NOT_RUN")
INTERPRETER_LABELS = tuple(f"py{v}" for v in REQUIRED_INTERPRETERS)
REQUIRED_ROW_FIELDS = ("host_id", "source_commit", "surface",
                       "interpreter_label", "verdict")
# Surfaces that can only be exercised where the v3 evidence store
# lives. Per codex's option-(a) ruling these require the EVIDENCE
# host's py3.11 cell; their py3.14 cell is an explicitly scoped
# NOT_RUN and is NOT counted as a coverage gap. DECLARED here rather
# than inferred, so the classification is reviewable.
EVIDENCE_HOST_ONLY = (
    "w2_restage_lineage_grassmann.py",
    "w2_restage_v4_grassmann.py",
)
EVIDENCE_HOST = "devildog"


def required_cells():
    """The closed cell set every leg must declare, exactly."""
    return {(sf, il) for sf in SURFACES for il in INTERPRETER_LABELS}


def _leg_cells(lg):
    return {(r["surface"].split("/")[-1], r["interpreter_label"])
            for r in lg["invocations"]}


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
                  "invocations"):
            if f not in lg:
                _mr(f"leg {i} is missing required field {f!r}; a leg "
                    "without full provenance cannot enter a merge")
        # each leg must still be INTERNALLY honest
        if lg["source_commit"] != lg["repo_head"]:
            _mr(f"leg {i} ({lg['host_id']}) has source_commit "
                f"{lg['source_commit'][:12]} != repo_head "
                f"{lg['repo_head'][:12]} -- it did not run from its "
                "own committed snapshot")
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
            for f in REQUIRED_ROW_FIELDS:
                if r.get(f) in (None, ""):
                    _mr(f"a row on {lg['host_id']} is missing "
                        f"required field {f!r}; an incomplete row "
                        "cannot enter a merged record")
            if r["verdict"] not in VERDICT_VOCABULARY:
                _mr(f"row verdict {r['verdict']!r} is outside the "
                    f"declared vocabulary {list(VERDICT_VOCABULARY)}")
            k = (r["host_id"], r["surface"], r["interpreter_label"])
            if k in seen:
                _mr(f"duplicate row {k}")
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
    covered = sorted({r["interpreter_version"].rsplit(".", 1)[0]
                      for r in rows
                      if r.get("interpreter_version")
                      and r["verdict"] != "NOT_RUN"})
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
    """BEHAVIORAL doctors, each MUTATION-TESTED below. After the
    doctor that keyed on a substring Python itself emits, no invariant
    here is asserted by reading source."""
    def cell(host, commit, sf, il, verdict="PASS"):
        return {"host_id": host, "source_commit": commit,
                "surface": "monitoring/src/" + sf,
                "interpreter_label": il,
                "interpreter_version": il[2:] + ".0",
                "verdict": verdict}

    def full_rows(host, commit, verdict="PASS"):
        """A COMPLETE leg: every declared cell present, explicitly."""
        return [cell(host, commit, sf, il, verdict)
                for sf in SURFACES for il in INTERPRETER_LABELS]

    def leg(host, commit, gen="g" * 64, schema="s-v1", rows=None):
        return {"host_id": host, "source_commit": commit,
                "repo_head": commit, "schema": schema,
                "producer_generator_blob_sha256": gen,
                "invocations": (full_rows(host, commit)
                                if rows is None else rows)}

    def refuses(fn, needle):
        try:
            fn()
            return False
        except MergeRefusal as e:
            return needle in str(e)
    C1, C2 = "a" * 40, "b" * 40
    NCELL = len(SURFACES) * len(INTERPRETER_LABELS)
    ok = merge_legs([leg("devildog", C1), leg("geomen", C1)])
    assert ok["source_commit"] == C1
    assert len(ok["invocations"]) == 2 * NCELL
    assert ok["verdict_counts"] == {"PASS": 2 * NCELL}
    assert [l["host_id"] for l in ok["legs"]] == ["devildog", "geomen"]
    assert ok["coverage_status"] == "COMPLETE", ok["coverage_status"]
    assert ok["missing_required_cells"] == []
    # (1) THE case cayley found: legs at different snapshots
    assert refuses(lambda: merge_legs(
        [leg("devildog", C1), leg("geomen", C2)]),
        "DIFFERENT snapshots")
    # (2) a leg that did not run from its own committed snapshot
    bad = leg("geomen", C1)
    bad["repo_head"] = C2
    assert refuses(lambda: merge_legs([leg("devildog", C1), bad]),
                   "did not run from its own committed snapshot")
    # (3) a ROW at the wrong commit inside a correct-looking leg --
    # the header agreeing is not the rows agreeing
    sneak = leg("geomen", C1)
    sneak["invocations"][0]["source_commit"] = C2
    assert refuses(lambda: merge_legs([leg("devildog", C1), sneak]),
                   "every ROW is checked")
    # (4) divergent generator bytes at the same commit
    assert refuses(lambda: merge_legs(
        [leg("devildog", C1), leg("geomen", C1, gen="h" * 64)]),
        "DIFFERENT generator bytes")
    # (5) one host merged with itself is not coverage
    assert refuses(lambda: merge_legs(
        [leg("devildog", C1), leg("devildog", C1)]), "duplicate host")
    # (6) a single leg is not a merge
    assert refuses(lambda: merge_legs([leg("devildog", C1)]),
                   "at least two legs")
    # (7) a row claiming a host it did not run on
    liar = leg("geomen", C1)
    liar["invocations"][0]["host_id"] = "devildog"
    assert refuses(lambda: merge_legs([leg("devildog", C1), liar]),
                   "claims host")
    # --- codex 1327Z P0 #2: snapshot agreement is not COMPLETENESS ---
    # (8) THE exact case codex constructed: two one-row legs, one
    # 3.11 row and one 3.14 row. This used to emit
    # missing_required_interpreters=[] with 26 required cells absent.
    tiny_d = [cell("devildog", C1, SURFACES[0], "py3.11")]
    tiny_g = [cell("geomen", C1, SURFACES[0], "py3.14")]
    assert refuses(lambda: merge_legs(
        [leg("devildog", C1, rows=tiny_d),
         leg("geomen", C1, rows=tiny_g)]),
        "every non-execution must be an EXPLICIT NOT_RUN row")
    # (9) an OMITTED cell is not the same as a NOT_RUN cell
    short = full_rows("geomen", C1)[:-1]
    assert refuses(lambda: merge_legs(
        [leg("devildog", C1), leg("geomen", C1, rows=short)]),
        "declares 27 cells")
    # (10) an EXTRA cell outside the declared set is refused too
    extra = full_rows("geomen", C1) + [
        cell("geomen", C1, "not_a_declared_surface.py", "py3.11")]
    assert refuses(lambda: merge_legs(
        [leg("devildog", C1), leg("geomen", C1, rows=extra)]),
        "unexpected")
    # (11) a complete-but-UNCOVERED merge is explicitly INCOMPLETE
    # rather than silently green: every cell present as NOT_RUN
    nr_d = full_rows("devildog", C1, "NOT_RUN")
    nr_g = full_rows("geomen", C1, "NOT_RUN")
    inc = merge_legs([leg("devildog", C1, rows=nr_d),
                      leg("geomen", C1, rows=nr_g)])
    assert inc["coverage_status"] == "INCOMPLETE"
    assert inc["missing_required_cells"], "must NAME the gaps"
    assert "does NOT establish" in inc["coverage_claim"]
    # the evidence-host-only surfaces are SCOPED, not counted as gaps
    gaps = {tuple(x) for x in inc["missing_required_cells"]}
    for sf in EVIDENCE_HOST_ONLY:
        assert (sf, "py3.14") not in gaps, sf
        assert (sf, "py3.11") in gaps, sf
    # (12) a row outside the declared verdict vocabulary
    vbad = full_rows("geomen", C1)
    vbad[0] = dict(vbad[0], verdict="GREEN")
    assert refuses(lambda: merge_legs(
        [leg("devildog", C1), leg("geomen", C1, rows=vbad)]),
        "outside the declared vocabulary")
    # (13) a row missing a required field
    fbad = full_rows("geomen", C1)
    fbad[0] = {k: v for k, v in fbad[0].items()
               if k != "interpreter_version"}
    fbad[0]["verdict"] = None
    assert refuses(lambda: merge_legs(
        [leg("devildog", C1), leg("geomen", C1, rows=fbad)]),
        "missing required field")
    print("w2 leg-merge selftest: ALL PASS (13 directions -- 7 frame "
          "+ 6 completeness)")



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
    before = None
    if os.path.isfile(out):
        with open(out, "rb") as f:
            before = hashlib.sha256(f.read()).hexdigest()
    me = os.path.abspath(__file__)
    for bad in (["--selftest"], ["--bogus"], [], ["--generate=x"],
                ["--Generate"], ["--merge-selftest", "--generate"]):
        r = subprocess.run([sys.executable, me] + bad,
                           capture_output=True)
        assert r.returncode == 2, (bad, r.returncode)
        after = None
        if os.path.isfile(out):
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
