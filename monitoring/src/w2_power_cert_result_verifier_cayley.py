#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 POWER-CERT RESULT VERIFIER (cayley) -- the INDEPENDENT
semantic verifier codex's w2r1 cycle-2 finding 2 requires. Manifest
or slot closure is NEVER a semantic PASS; the composed prestart's
power gate consumes THIS verifier's typed verdict.

INDEPENDENCE TERMS
------------------
- Human split: the campaign executor (grassmann) never verifies;
  this tool is run by a different party (cayley/codex) and its
  receipt names the runner.
- Byte source: every input reopens from GIT OBJECTS at the named
  commit (materialized read-only to a temp dir so the pinned
  RUNNER's own authenticated loaders stay the single authority --
  never a re-implementation, never a working tree).
- Content: the package must equal the SHARED pure constructor
  (`build_package_content`, the same function the assembler ran)
  applied to the committed inputs -- rebuild-or-refuse, never a
  field-by-field trust walk.
- Semantics (enforced HERE, typed):
  S1 every ADMITTED_WITH_POWER entry carries a finite
     anticipated-mask envelope + the registered CP-floor threshold;
  S2 terminally excluded MAG entries are TYPED_NON_CERTIFICATION,
     non_blocking, excluded_from_holm;
  S3 every registered grid member outside certified S is
     CANNOT_DETERMINE_NO_POWER and excluded_from_holm;
  S4 admission-table completeness: every registered member exactly
     once, plus the MAG + M-F4 lane entries, nothing else;
  S5 a well-formed typed refusal NEVER counts as a power PASS: the
     verdict separates PACKAGE_VALID (bytes/semantics hold) from
     POWER_GATE (certified-S + admitted coverage), and
     prestart_overall may only consume POWER_GATE.

Returns a typed result; --emit writes
docs/f2g_window2_execution/power_cert/power_cert_verifier_receipt_v1
.json via atomic create-once. Opens no window-2 value; read-only
otherwise; admits nothing. Lambda_geo INCONCLUSIVE.
"""
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import w2_cert_runner_cayley as RUN  # noqa: E402
import w2_power_cert_results_assembly_cayley as ASM  # noqa: E402

VERIFIER_RECEIPT_REL = (ASM.POWER_CERT_DIR
                        + "/power_cert_verifier_receipt_v1.json")
RECEIPT_SCHEMA = "f2g-w2-power-cert-verifier-receipt-v1"


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _git_blob(repo, commit, rel):
    p = subprocess.run(["git", "-C", repo, "cat-file", "blob",
                        f"{commit}:{rel}"], capture_output=True)
    if p.returncode != 0:
        return None
    return p.stdout


def _num(v):
    """A registered numeric value: real number, never bool, never
    NaN/inf."""
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and v == v and v not in (float("inf"), float("-inf")))


def check_admission_semantics(package, grids, *, readiness_sha,
                              records_by_member):
    """cycle-4 R3 (codex cycle-3 finding 3): the INDEPENDENT
    semantic contract oracle. The shared constructor proves byte
    reproduction; THIS function enforces the registered semantics
    against the AUTHENTICATED records, so a constructor defect can
    never certify itself. Typed reasons; pure; directly KAT-able.

    Enforced, exactly:
      - detection envelope keys == {R, k, cp_lb}, values == the
        authenticated record's {R, k, lb}, all registered-numeric;
      - specificity envelope keys == {R, positives, rate, max_rate},
        values == the record surface;
      - threshold == the registered CP floor exactly (0.80,
        non-bool finite);
      - admitted set == certified S (every selector point CERTIFIED
        => admitted; nothing else admitted);
      - every registered member outside S is exactly
        CANNOT_DETERMINE_NO_POWER, Holm-excluded, closed schema;
      - MAG row: exact closed non-certification schema,
        non_blocking, Holm-excluded, 64-hex source;
      - M-F4 row: exactly SEPARATE_GATE_MF4_MATURITY, readiness_state
        READY_FOR_ACCRUAL, source sha == the committed readiness
        record bytes, never power-admitted;
      - closed row schemas everywhere (extra fields refuse)."""
    reasons = []

    def reason(code, detail):
        reasons.append({"code": code, "detail": str(detail)[:300]})

    CP_FLOOR = 0.80
    ROW_KEYS = {
        "ADMITTED_WITH_POWER": {"member", "state",
                                "anticipated_mask_envelope",
                                "threshold_cp_floor",
                                "record_sha256"},
        "CANNOT_DETERMINE_NO_POWER": {"member", "state",
                                      "excluded_from_holm",
                                      "reason"},
        "TYPED_NON_CERTIFICATION_MEMBER": {"member", "state",
                                           "excluded_from_holm",
                                           "reason",
                                           "record_sha256"},
        "MAG_LANE": {"member", "state", "non_blocking",
                     "excluded_from_holm", "reason",
                     "source_sha256"},
        "MF4_LANE": {"member", "state", "excluded_from_holm",
                     "reason", "readiness_state", "source_sha256"},
    }
    want_members = {ASM._member_key(f, e, p)
                    for f, e, p in ASM._grid_members(grids)}
    certified_pts = {
        k for k, m in records_by_member.items()
        if m["record"]["status"] == "CERTIFIED"}
    s_ok = (package.get("certified_s", {}).get("status")
            == "CERTIFIED_S")
    seen_members = set()
    lanes_seen = set()
    for row in package["admission_table"]:
        st = row.get("state")
        if st not in ASM.ADMIT_STATES:
            reason("POWER_VERIFY_STATE_UNKNOWN", st)
            continue
        member = row.get("member", {})
        if "lane" in member:
            lane = member["lane"]
            lanes_seen.add(lane)
            if lane == "mag_primary_set":
                if st != "TYPED_NON_CERTIFICATION" or \
                        row.get("non_blocking") is not True or \
                        row.get("excluded_from_holm") is not True:
                    reason("POWER_VERIFY_MAG_SEMANTICS",
                           "excluded MAG entry must be typed "
                           "non-certification, non-blocking, "
                           "excluded from Holm")
                if set(row) != ROW_KEYS["MAG_LANE"]:
                    reason("POWER_VERIFY_ROW_SCHEMA",
                           f"mag row schema not closed: "
                           f"{sorted(set(row))}")
                src = row.get("source_sha256")
                if not (isinstance(src, str)
                        and re.fullmatch(r"[0-9a-f]{64}", src)):
                    reason("POWER_VERIFY_MAG_SEMANTICS",
                           "mag source_sha256 is not 64-hex")
            elif lane == "mf4_daily_risk":
                if st != "SEPARATE_GATE_MF4_MATURITY":
                    reason("POWER_VERIFY_MF4_SEMANTICS",
                           f"M-F4 lane state {st!r} -- it is NEVER "
                           "power-admitted by this package")
                if row.get("readiness_state") != \
                        "READY_FOR_ACCRUAL":
                    reason("POWER_VERIFY_MF4_SEMANTICS",
                           f"readiness_state "
                           f"{row.get('readiness_state')!r}")
                if readiness_sha is not None and \
                        row.get("source_sha256") != readiness_sha:
                    reason("POWER_VERIFY_MF4_SEMANTICS",
                           "M-F4 source sha diverges from the "
                           "committed readiness record bytes")
                if row.get("excluded_from_holm") is not True:
                    reason("POWER_VERIFY_MF4_SEMANTICS",
                           "M-F4 must be excluded from Holm")
                if set(row) != ROW_KEYS["MF4_LANE"]:
                    reason("POWER_VERIFY_ROW_SCHEMA",
                           f"mf4 row schema not closed: "
                           f"{sorted(set(row))}")
            else:
                reason("POWER_VERIFY_LANE_SET", lane)
            continue
        key = ASM._member_key(member.get("family"),
                              member.get("entry"),
                              member.get("point"))
        if key in seen_members:
            reason("POWER_VERIFY_DUPLICATE_MEMBER", key)
        seen_members.add(key)
        if st == "ADMITTED_WITH_POWER":
            if not s_ok or key not in certified_pts:
                reason("POWER_VERIFY_ADMITTED_OUTSIDE_S", key)
                continue
            if set(row) != ROW_KEYS["ADMITTED_WITH_POWER"]:
                reason("POWER_VERIFY_ROW_SCHEMA",
                       f"admitted row schema not closed: "
                       f"{sorted(set(row))}")
                continue
            rec = records_by_member[key]["record"]
            env = row.get("anticipated_mask_envelope")
            if member.get("entry") == "specificity":
                want = {"R": rec.get("R"),
                        "positives": rec.get("positives"),
                        "rate": rec.get("rate"),
                        "max_rate": rec.get("max_rate")}
            else:
                want = {"R": rec.get("R"), "k": rec.get("k"),
                        "cp_lb": rec.get("lb")}
            if not isinstance(env, dict) or env != want or \
                    not all(_num(v) for v in env.values()):
                reason("POWER_VERIFY_ADMITTED_WITHOUT_ENVELOPE",
                       f"{key}: envelope diverges from the "
                       "authenticated record surface")
            thr = row.get("threshold_cp_floor")
            if not _num(thr) or thr != CP_FLOOR:
                reason("POWER_VERIFY_THRESHOLD_UNREGISTERED",
                       f"{key}: threshold {thr!r} != the registered "
                       f"CP floor {CP_FLOOR}")
            if row.get("record_sha256") != \
                    records_by_member[key]["record_sha256"]:
                reason("POWER_VERIFY_ADMITTED_WITHOUT_ENVELOPE",
                       f"{key}: record digest diverges from the "
                       "authenticated record")
        elif st == "CANNOT_DETERMINE_NO_POWER":
            if key in certified_pts and s_ok:
                reason("POWER_VERIFY_ADMITTED_SET_DIVERGENT",
                       f"{key}: certified in S but not admitted")
            if row.get("excluded_from_holm") is not True:
                reason("POWER_VERIFY_HOLM_EXCLUSION_MISSING", key)
            if set(row) != ROW_KEYS["CANNOT_DETERMINE_NO_POWER"]:
                reason("POWER_VERIFY_ROW_SCHEMA",
                       f"cannot-determine row schema not closed: "
                       f"{sorted(set(row))}")
        elif st == "TYPED_NON_CERTIFICATION":
            if row.get("excluded_from_holm") is not True:
                reason("POWER_VERIFY_HOLM_EXCLUSION_MISSING", key)
            if set(row) != \
                    ROW_KEYS["TYPED_NON_CERTIFICATION_MEMBER"]:
                reason("POWER_VERIFY_ROW_SCHEMA",
                       f"typed row schema not closed: "
                       f"{sorted(set(row))}")
        else:
            reason("POWER_VERIFY_MF4_SEMANTICS",
                   f"member row carries lane-only state {st!r}")
    if seen_members != want_members:
        reason("POWER_VERIFY_TABLE_INCOMPLETE",
               f"missing={len(want_members - seen_members)} "
               f"extra={len(seen_members - want_members)}")
    if lanes_seen != {"mag_primary_set", "mf4_daily_risk"}:
        reason("POWER_VERIFY_LANE_SET", sorted(lanes_seen))
    # admitted set == certified S, both directions
    admitted = {ASM._member_key(r["member"]["family"],
                                r["member"]["entry"],
                                r["member"]["point"])
                for r in package["admission_table"]
                if r.get("state") == "ADMITTED_WITH_POWER"
                and "lane" not in r.get("member", {})}
    if s_ok and admitted != certified_pts:
        reason("POWER_VERIFY_ADMITTED_SET_DIVERGENT",
               f"admitted {len(admitted)} != certified-S "
               f"{len(certified_pts)}")
    return reasons


def verify(repo, commit, *, blob_reader=None,
           selector_loader=None, ancestor_check=None,
           commit_time_utc=None):
    """Typed verification of the committed result package at
    `commit`. Never raises for content findings -- every defect is a
    typed reason; hard I/O problems raise. blob_reader /
    selector_loader are KAT-ONLY fixture seams; production
    callers pass neither and get git objects + the pinned
    runner's committed-selector loader unconditionally."""
    if blob_reader is None:
        def blob_reader(c, rel):
            return _git_blob(repo, c, rel)
    if selector_loader is None:
        selector_loader = RUN.load_selector_committed
    if ancestor_check is None:
        def ancestor_check(a, b):
            return subprocess.run(
                ["git", "-C", repo, "merge-base", "--is-ancestor",
                 str(a), str(b)], capture_output=True
                ).returncode == 0
    if commit_time_utc is None:
        r_ = subprocess.run(
            ["git", "-C", repo, "show", "-s", "--format=%cI",
             str(commit)], capture_output=True, text=True)
        commit_time_utc = r_.stdout.strip() or None
    res = {"schema": "f2g-w2-power-cert-result-verdict-v1",
           "commit": str(commit), "typed_reasons": [],
           "package_valid": False, "power_gate": "REFUSE"}

    def reason(code, detail):
        res["typed_reasons"].append({"code": code,
                                     "detail": str(detail)[:300]})

    pkg_b = blob_reader(commit, ASM.PACKAGE_REL)
    rcpt_b = blob_reader(commit, ASM.RECEIPT_REL)
    if pkg_b is None or rcpt_b is None:
        reason("POWER_VERIFY_RESULT_ABSENT",
               "package/receipt not committed at the target")
        return res
    package = json.loads(pkg_b.decode("utf-8"))
    receipt = json.loads(rcpt_b.decode("utf-8"))

    # receipt binds the exact package bytes
    if receipt.get("schema") != ASM.RECEIPT_SCHEMA or \
            receipt.get("outputs", {}).get(ASM.PACKAGE_REL) != \
            _sha(pkg_b):
        reason("POWER_VERIFY_RECEIPT_BINDING",
               "receipt does not bind the committed package bytes")
    # cycle-4 R2: canonical-UTC timestamps, no later than the
    # result commit
    ts_re = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"
    t0_, t1_ = receipt.get("started_utc"), receipt.get("ended_utc")
    if not (isinstance(t0_, str) and re.fullmatch(ts_re, t0_)
            and isinstance(t1_, str)
            and re.fullmatch(ts_re, t1_) and t0_ <= t1_):
        reason("POWER_VERIFY_RECEIPT_TIMESTAMPS",
               f"non-canonical/misordered: {t0_!r}..{t1_!r}")
    elif isinstance(commit_time_utc, str) and commit_time_utc:
        import datetime as _dt
        try:
            ct = _dt.datetime.fromisoformat(
                commit_time_utc).astimezone(
                _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if t1_ > ct:
                reason("POWER_VERIFY_RECEIPT_TIMESTAMPS",
                       f"ended_utc {t1_} is later than the result "
                       f"commit {ct}")
        except ValueError:
            pass

    # materialize the committed campaign dir + inputs read-only and
    # REBUILD through the shared pure constructor
    outdir_rel = receipt.get("campaign", {}).get("outdir")
    if not isinstance(outdir_rel, str) or \
            not outdir_rel.startswith("docs/"):
        reason("POWER_VERIFY_OUTDIR", repr(outdir_rel))
        return res
    with tempfile.TemporaryDirectory() as td:
        tdir = os.path.join(td, "campaign")
        os.makedirs(tdir)
        names = ["invocation_record.json", "campaign_summary.json"]
        summary_b = blob_reader(commit,
                                outdir_rel + "/campaign_summary.json")
        if summary_b is None:
            reason("POWER_VERIFY_CAMPAIGN_ABSENT", outdir_rel)
            return res
        n = int(json.loads(summary_b.decode("utf-8"))["n_points"])
        names += [f"point_{i:03d}.json" for i in range(n)]
        if blob_reader(commit,
                       outdir_rel + "/campaign_aborted.json"):
            reason("POWER_VERIFY_CAMPAIGN_ABORTED", outdir_rel)
            return res
        for nm in names:
            b = blob_reader(commit, outdir_rel + "/" + nm)
            if b is None:
                reason("POWER_VERIFY_CAMPAIGN_FILE_ABSENT", nm)
                return res
            with open(os.path.join(tdir, nm), "wb") as f:
                f.write(b)

        def blob_at_commit(rel, what):
            b = blob_reader(commit, rel)
            if b is None:
                reason("POWER_VERIFY_INPUT_ABSENT", f"{what}: {rel}")
            return b

        grids_b = blob_at_commit(ASM.GRIDS_REL, "effect grids")
        cal_b = blob_at_commit(ASM.CALENDAR_REL, "calendar v4")
        disp_b = blob_at_commit(ASM.DISPOSITION_REL, "disposition")
        ready_b = blob_at_commit(ASM.READINESS_REL, "readiness")
        man_b = blob_at_commit(ASM.MANIFEST_REL, "manifest")
        if None in (grids_b, cal_b, disp_b, ready_b, man_b):
            return res

        try:
            inv_obj = json.loads(open(os.path.join(
                tdir, "invocation_record.json"),
                encoding="utf-8").read())
            invocation, _pts = RUN._load_invocation(
                tdir, inv_obj.get("invocation_sha256"))
            summary = json.loads(summary_b.decode("utf-8"))
            point_files = [json.loads(open(os.path.join(
                tdir, f"point_{i:03d}.json"),
                encoding="utf-8").read()) for i in range(n)]

            def breader(c, path):
                b = blob_reader(c, path)
                if b is None:
                    raise RUN.RunnerRefusal(
                        f"RUNNER_SELECTOR_INVALID: {path} absent "
                        f"at {c}")
                return b
            selector, _sp, selector_sha = \
                selector_loader(
                    repo, summary["selector_commit"],
                    summary["selector_path"], blob_reader=breader)
            if selector_sha != summary["selector_sha256"]:
                reason("POWER_VERIFY_SELECTOR_IDENTITY",
                       "selector bytes diverge from the summary")
                return res
            # cycle-4 R2: harness pins resolve at the
            # INVOCATION's named manifest commit, which must be an
            # ancestor of the result commit -- never the later
            # manifest.
            inv_mc = invocation["manifest_commit"]
            man_inv_b = blob_reader(inv_mc, ASM.MANIFEST_REL)
            if man_inv_b is None:
                reason("POWER_VERIFY_HARNESS_SLOT",
                       f"manifest unreadable at invocation commit "
                       f"{str(inv_mc)[:12]}")
                return res
            if not ancestor_check(inv_mc, commit):
                reason("POWER_VERIFY_INVOCATION_ANCESTRY",
                       f"invocation manifest_commit "
                       f"{str(inv_mc)[:12]} is not an ancestor of "
                       "the result commit")
                return res
            try:
                harness_pins = ASM.harness_pins_from_manifest_bytes(
                    man_inv_b)
            except ASM.AssemblyRefusal as e:
                reason("POWER_VERIFY_HARNESS_SLOT", e)
                return res
            rebuilt = ASM.build_package_content(
                invocation=invocation, summary=summary,
                point_files=point_files, selector=selector,
                selector_identity={
                    "commit": summary["selector_commit"],
                    "path": summary["selector_path"],
                    "sha256": selector_sha},
                grids=json.loads(grids_b.decode("utf-8")),
                grids_sha=_sha(grids_b),
                calendar=json.loads(cal_b.decode("utf-8")),
                calendar_sha=_sha(cal_b),
                disposition_sha=_sha(disp_b),
                readiness=json.loads(ready_b.decode("utf-8")),
                readiness_sha=_sha(ready_b),
                harness_pins=harness_pins)
        except (ASM.AssemblyRefusal, RUN.RunnerRefusal) as e:
            reason("POWER_VERIFY_REBUILD_REFUSED", e)
            return res

    if json.dumps(rebuilt, sort_keys=True) != \
            json.dumps(package, sort_keys=True):
        reason("POWER_VERIFY_PACKAGE_DIVERGENT",
               "committed package != shared-constructor rebuild "
               "from committed inputs")
        return res

    # cycle-4 R2: the receipt output map is CLOSED and EXACT --
    # every raw byte sha recomputed from committed objects
    want_out = {ASM.PACKAGE_REL: _sha(pkg_b)}
    for nm in (["invocation_record.json", "campaign_summary.json"]
               + [f"point_{i:03d}.json" for i in range(n)]):
        b_ = blob_reader(commit, outdir_rel + "/" + nm)
        want_out[outdir_rel + "/" + nm] = _sha(b_)
    if receipt.get("outputs") != want_out:
        reason("POWER_VERIFY_RECEIPT_OUTPUTS",
               "receipt output map is not the closed exact set of "
               "recomputed raw byte shas")

    # --- semantic gates over the REBUILT (== committed) table,
    # against the AUTHENTICATED records (cycle-4 R3)
    rbm = {}
    for pf in point_files:
        k_ = ASM._member_key(pf["spec"]["family"],
                             pf["spec"]["entry"],
                             pf["spec"]["point"])
        rbm[k_] = {"record": pf["record"],
                   "record_sha256": ASM._canon(pf["record"])}
    for r_ in check_admission_semantics(
            rebuilt, json.loads(grids_b.decode("utf-8")),
            readiness_sha=_sha(ready_b), records_by_member=rbm):
        res["typed_reasons"].append(r_)

    res["package_valid"] = not res["typed_reasons"]
    # S5: POWER_GATE is a SEPARATE verdict -- typed refusals in the
    # package (non-certified S, zero admitted members) keep the gate
    # REFUSE even when the package bytes are perfectly valid.
    admitted = [r for r in rebuilt["admission_table"]
                if r.get("state") == "ADMITTED_WITH_POWER"]
    if res["package_valid"] and \
            rebuilt["certified_s"]["status"] == "CERTIFIED_S" and \
            admitted:
        res["power_gate"] = "PASS"
    res["admitted_members"] = len(admitted)
    res["certified_s_status"] = rebuilt["certified_s"]["status"]
    return res


def main():
    repo = os.path.abspath(sys.argv[1])
    commit = sys.argv[2]
    emit = "--emit" in sys.argv[3:]
    res = verify(repo, commit)
    with open(os.path.abspath(__file__), "rb") as f:
        res["verifier_source_sha256"] = _sha(f.read())
    res["verified_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                        time.gmtime())
    res["verifier_host"] = platform.node()
    res["authorizes"] = ("NOTHING by itself: the composed prestart "
                         "consumes power_gate; no admission, no "
                         "evaluation value")
    print(json.dumps(res, indent=1, sort_keys=True))
    if emit:
        body = json.dumps({"schema": RECEIPT_SCHEMA, **res},
                          indent=1, sort_keys=True) + "\n"
        path = os.path.join(repo, VERIFIER_RECEIPT_REL.replace(
            "/", os.sep))
        RUN._publish_once(path, body)
        print(f"receipt: {VERIFIER_RECEIPT_REL}")
    sys.exit(0 if res["power_gate"] == "PASS" else 1)


if __name__ == "__main__":
    main()
