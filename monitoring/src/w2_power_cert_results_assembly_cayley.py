#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 POWER-CERT RESULT ASSEMBLER (cayley) -- codex w2r1
cycle-2 ruling (2026-08-30T22:40Z) finding 2 (CRITICAL): the result
slot must be generated and verified by reviewed machinery BEFORE any
fire. This module is the PRE-FIRE producer of

    docs/f2g_window2_execution/power_cert/
        power_cert_result_package_v1.json
        power_cert_result_receipt_v1.json

from a COMPLETED certification campaign outdir. Nothing here runs
certification; it assembles, types, and receipts what the pinned
runner/harness produced.

DESIGN TERMS
------------
- One authority per surface: the invocation is authenticated through
  the RUNNER's own loader (canonical invocation_sha256 recompute);
  the selector reopens through the RUNNER's committed-selector path
  (independent rerun verification inside); record identity uses the
  RUNNER's own rule. No re-implementations (content-auth is not
  derivation provenance).
- The package content is built by ONE pure function
  (`build_package_content`) shared verbatim with the independent
  result verifier, so verifier and producer can never drift
  (the MAG r3 lesson: a shared canonical constructor, or the
  verifier is a second guess).
- ADMISSION TABLE semantics (codex finding 2, verbatim intent):
  * terminally excluded MAG entries are TYPED_NON_CERTIFICATION and
    non-blocking (source: the bound terminal-exclusion disposition);
  * registered grid members outside certified S are typed
    CANNOT_DETERMINE_NO_POWER and excluded_from_holm;
  * every member the package marks ADMITTED_WITH_POWER carries the
    anticipated-mask envelope {R, k, cp_lb} and the registered
    threshold (the harness CP floor) -- anything else refuses;
  * M-F4 is NOT in anticipated-mask scope: it gets a typed
    SEPARATE_GATE_MF4_MATURITY entry bound to the successor-
    readiness record; its admission rides its OWN registered gates,
    never this package (flagged for codex in the cycle-3 cover).
  * manifest/slot closure is never a semantic PASS: the independent
    result verifier enforces this table; the composed prestart
    consumes ITS receipt.
- Transactional: validate ALL inputs, build in memory, publish via
  the runner's atomic create-once primitive; a refused assembly
  leaves ZERO artifacts; the receipt publishes LAST.

Opens no window-2 value; no network; runs nothing; admits nothing.
Lambda_geo INCONCLUSIVE.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import w2_cert_runner_cayley as RUN  # noqa: E402
import w2_power_harness_cayley as PH  # noqa: E402

PACKAGE_SCHEMA = "f2g-w2-power-cert-result-package-v1"
RECEIPT_SCHEMA = "f2g-w2-power-cert-result-receipt-v1"
POWER_CERT_DIR = "docs/f2g_window2_execution/power_cert"
PACKAGE_REL = POWER_CERT_DIR + "/power_cert_result_package_v1.json"
RECEIPT_REL = POWER_CERT_DIR + "/power_cert_result_receipt_v1.json"
GRIDS_REL = "docs/f2g_window2_execution/effect_grids_w2_v1.json"
CALENDAR_REL = ("docs/f2g_window2_execution/"
                "calendar_authority_w2_v4.json")
DISPOSITION_REL = ("docs/f2g_window2_execution/"
                   "mag_primary_terminal_exclusion_v1.md")
READINESS_REL = ("docs/f2g_window2_execution/"
                 "mf4_successor_readiness_record_v1.json")
MANIFEST_REL = "docs/f2g_window2_execution/execution_manifest.json"
FAMILIES = ("B1B", "B2A", "B2B", "B3A")
ADMIT_STATES = ("ADMITTED_WITH_POWER", "CANNOT_DETERMINE_NO_POWER",
                "TYPED_NON_CERTIFICATION",
                "SEPARATE_GATE_MF4_MATURITY")


class AssemblyRefusal(ValueError):
    pass


def _refuse(code, detail):
    raise AssemblyRefusal(f"{code}: {detail}")


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _canon(obj):
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def _read(path, what):
    if not os.path.isfile(path):
        _refuse("POWER_RESULT_INPUT_ABSENT", f"{what}: {path}")
    with open(path, "rb") as f:
        return f.read()


def _member_key(family, entry, point):
    return json.dumps({"family": family, "entry": entry,
                       "point": point}, sort_keys=True,
                      separators=(",", ":"))


def _grid_members(grids):
    """Every REGISTERED member: detection points for all four
    families + the two registered B1B specificity gain points (the
    tier-selector shape: gains 3 then 10)."""
    out = []
    for fam in FAMILIES:
        for point in grids["grids"][fam]:
            out.append((fam, "detection", dict(point)))
    for gain in (3, 10):
        out.append(("B1B", "specificity", {"gain": gain}))
    return out


def build_package_content(*, invocation, summary, point_files,
                          selector, selector_identity, grids,
                          grids_sha, calendar, calendar_sha,
                          disposition_sha, readiness, readiness_sha,
                          harness_pins):
    """PURE package-content constructor, shared byte-for-byte by the
    assembler and the independent result verifier. No filesystem, no
    git, no clock -- every identity arrives resolved."""
    if summary.get("schema") != "f2g-w2-cert-campaign-summary-v3":
        _refuse("POWER_RESULT_SUMMARY_SCHEMA",
                repr(summary.get("schema")))
    if summary["invocation_sha256"] != invocation["invocation_sha256"]:
        _refuse("POWER_RESULT_INVOCATION_MISMATCH",
                "summary does not cite the authenticated invocation")
    # --- cycle-4 R2 (codex cycle-3 finding 2): the AUTHENTICATED
    # FIRED INVOCATION is the point/core authority. Every consumed
    # surface must equal it EXACTLY; a coherently rewritten
    # summary/point/selector set that retains the authentic digest
    # refuses typed.
    inv_pts = invocation["ordered_points"]
    if RUN._digest(inv_pts) != invocation["ordered_points_sha256"]:
        _refuse("POWER_RESULT_INVOCATION_CORE_DIVERGENT",
                "invocation points diverge from their own digest")
    for f_, want in (("manifest_commit", None),
                     ("selector_commit", None),
                     ("selector_path", None),
                     ("geometry_path", None),
                     ("ordered_points_sha256", None)):
        if summary.get(f_) != invocation.get(f_):
            _refuse("POWER_RESULT_INVOCATION_CORE_DIVERGENT",
                    f"summary.{f_} diverges from the fired "
                    "invocation")
    if invocation.get("selector_sha256") is not None and \
            summary.get("selector_sha256") != \
            invocation.get("selector_sha256"):
        _refuse("POWER_RESULT_INVOCATION_CORE_DIVERGENT",
                "summary.selector_sha256 diverges from the fired "
                "invocation")
    if int(summary["n_points"]) != len(inv_pts):
        _refuse("POWER_RESULT_INVOCATION_CORE_DIVERGENT",
                "summary.n_points diverges from the fired "
                "invocation point list")
    sel_pts = selector["ordered_points"]
    if len(sel_pts) != len(inv_pts):
        _refuse("POWER_RESULT_INVOCATION_CORE_DIVERGENT",
                "committed selector point count diverges from the "
                "fired invocation")
    for i_, ip in enumerate(inv_pts):
        if sel_pts[i_] != ip:
            _refuse("POWER_RESULT_INVOCATION_CORE_DIVERGENT",
                    f"selector point {i_} diverges from the fired "
                    "invocation")
    if calendar.get("schema") != "f2g-w2-calendar-authority-v4":
        _refuse("POWER_RESULT_CALENDAR_WRONG",
                f"{calendar.get('schema')!r} is not the v4 successor "
                "authority")
    n = int(summary["n_points"])
    if len(point_files) != n or len(summary["per_point"]) != n:
        _refuse("POWER_RESULT_POINT_CENSUS",
                f"{len(point_files)} files / "
                f"{len(summary['per_point'])} summary rows / "
                f"{n} declared")

    # authenticate every point record against invocation + summary
    frame_ids = set()
    capsule_digests = set()
    cal_shas = set()
    by_member = {}
    for i, pf in enumerate(point_files):
        if pf.get("index") != i or \
                pf.get("invocation_sha256") != \
                invocation["invocation_sha256"]:
            _refuse("POWER_RESULT_POINT_IDENTITY",
                    f"point {i} file identity diverges")
        if pf.get("spec") != inv_pts[i]:
            _refuse("POWER_RESULT_INVOCATION_CORE_DIVERGENT",
                    f"point {i} file spec diverges from the fired "
                    "invocation point")
        if pf.get("refusal") is not None or pf.get("record") is None:
            _refuse("POWER_RESULT_POINT_REFUSED",
                    f"point {i}: a refused/absent record can never "
                    "assemble -- rerun or exclude by a NEW campaign, "
                    "never by editing")
        rec = pf["record"]
        srow = summary["per_point"][i]
        if srow["index"] != i or srow["family"] != \
                inv_pts[i]["family"] or srow["point"] != \
                inv_pts[i]["point"] or srow["entry"] != \
                inv_pts[i]["entry"]:
            _refuse("POWER_RESULT_INVOCATION_CORE_DIVERGENT",
                    f"point {i} summary row diverges from the fired "
                    "invocation point")
        if srow["record_sha256"] != _canon(rec):
            _refuse("POWER_RESULT_RECORD_DIGEST",
                    f"point {i} record digest diverges from the "
                    "summary")
        if not RUN._record_identity_ok(pf["spec"], rec):
            _refuse("POWER_RESULT_RECORD_IDENTITY",
                    f"point {i} record family/point diverges")
        if rec.get("tier") != "CERTIFICATION" or \
                rec.get("certifiable") is not True:
            _refuse("POWER_RESULT_TIER",
                    f"point {i} is not a certification-tier record")
        frame_ids.add(rec.get("calendar_frame_id"))
        capsule_digests.add(rec.get("geometry_capsule_digest"))
        cal_shas.add(rec.get("calendar_authority_sha256"))
        key = _member_key(pf["spec"]["family"], pf["spec"]["entry"],
                          pf["spec"]["point"])
        if key in by_member:
            _refuse("POWER_RESULT_DUPLICATE_MEMBER", key)
        by_member[key] = {"spec": pf["spec"], "record": rec,
                          "record_sha256": srow["record_sha256"]}
    if frame_ids and frame_ids != {calendar["frame"]["frame_id"]}:
        _refuse("POWER_RESULT_CALENDAR_WRONG",
                f"record frame ids {sorted(frame_ids)} != the "
                "committed v4 authority frame")
    if cal_shas and cal_shas != {calendar_sha}:
        _refuse("POWER_RESULT_CALENDAR_WRONG",
                "record calendar-authority shas diverge from the "
                "committed v4 authority bytes")
    if len(capsule_digests) > 1:
        _refuse("POWER_RESULT_GEOMETRY_SPLIT",
                f"{len(capsule_digests)} distinct capsule digests "
                "in one campaign")

    # selector: campaign points must be exactly the certified-S shape
    sel_keys = [_member_key(p["family"], p["entry"], p["point"])
                for p in selector["ordered_points"]]
    if sorted(sel_keys) != sorted(by_member):
        _refuse("POWER_RESULT_SELECTOR_SET",
                "campaign members are not exactly the selector's "
                "ordered points")

    # four-family result: every family present, never omitted
    four_family = {}
    for fam in FAMILIES:
        rows = []
        for key in sorted(by_member):
            m = by_member[key]
            if m["spec"]["family"] != fam:
                continue
            rec = m["record"]
            row = {"entry": m["spec"]["entry"],
                   "point": m["spec"]["point"],
                   "status": rec["status"],
                   "record_sha256": m["record_sha256"]}
            if m["spec"]["entry"] == "specificity":
                row["envelope"] = {"R": rec["R"],
                                   "positives": rec["positives"],
                                   "rate": rec["rate"],
                                   "max_rate": rec["max_rate"]}
            else:
                row["envelope"] = {"R": rec["R"], "k": rec["k"],
                                   "cp_lb": rec["lb"]}
            rows.append(row)
        four_family[fam] = ({"points": rows} if rows else
                            {"points": [],
                             "typed_non_certification":
                                 "NO_POINTS_IN_CERTIFIED_CAMPAIGN"})

    # certified S: the selector identity + per-point outcomes
    s_failed = [k for k, m in by_member.items()
                if m["record"]["status"] != "CERTIFIED"]
    s_certified = not s_failed
    certified_s = {
        "selector": dict(selector_identity),
        "n_points": len(sel_keys),
        "all_points_certified": s_certified,
        "status": ("CERTIFIED_S" if s_certified else
                   "TYPED_NON_CERTIFICATION_S"),
        "non_certified_members": sorted(s_failed)}

    # admission table: every registered grid member exactly once.
    # cycle-4 R3: admission requires the WHOLE certified S -- a
    # typed non-certified S admits NOTHING (S5: a well-formed
    # refusal never becomes a power pass), so a CERTIFIED point
    # inside a failed S types, never admits.
    threshold = PH.CP_FLOOR
    table = []
    for fam, entry, point in _grid_members(grids):
        key = _member_key(fam, entry, point)
        m = by_member.get(key)
        if m is None:
            table.append({
                "member": {"family": fam, "entry": entry,
                           "point": point},
                "state": "CANNOT_DETERMINE_NO_POWER",
                "excluded_from_holm": True,
                "reason": "registered member outside certified S"})
        elif m["record"]["status"] == "CERTIFIED" and                 s_certified:
            rec = m["record"]
            env = ({"R": rec["R"], "positives": rec["positives"],
                    "rate": rec["rate"], "max_rate": rec["max_rate"]}
                   if entry == "specificity" else
                   {"R": rec["R"], "k": rec["k"],
                    "cp_lb": rec["lb"]})
            table.append({
                "member": {"family": fam, "entry": entry,
                           "point": point},
                "state": "ADMITTED_WITH_POWER",
                "anticipated_mask_envelope": env,
                "threshold_cp_floor": threshold,
                "record_sha256": m["record_sha256"]})
        else:
            why = (f"campaign status {m['record']['status']}"
                   if m["record"]["status"] != "CERTIFIED" else
                   "CERTIFIED point inside a non-certified S "
                   "(S5: never admitted)")
            table.append({
                "member": {"family": fam, "entry": entry,
                           "point": point},
                "state": "TYPED_NON_CERTIFICATION",
                "excluded_from_holm": True,
                "reason": why,
                "record_sha256": m["record_sha256"]})
    table.append({
        "member": {"lane": "mag_primary_set"},
        "state": "TYPED_NON_CERTIFICATION",
        "non_blocking": True,
        "excluded_from_holm": True,
        "reason": "MAG_TERMINAL_EXCLUSION (owner option 1; typed "
                  "dispositions stand: family_b="
                  "FILTER_SUPPORT_INSUFFICIENT, mag_primary_set="
                  "UNTESTABLE_NO_ADMISSIBLE_PRIMARY)",
        "source_sha256": disposition_sha})
    if readiness.get("schema") != "f2g-w2-mf4-successor-readiness-v1":
        _refuse("POWER_RESULT_READINESS_SCHEMA",
                repr(readiness.get("schema")))
    table.append({
        "member": {"lane": "mf4_daily_risk"},
        "state": "SEPARATE_GATE_MF4_MATURITY",
        "excluded_from_holm": True,
        "reason": "not in anticipated-mask scope; admission rides "
                  "the registered M-F4 maturity/readiness gates, "
                  "never this package",
        "readiness_state": readiness["state"],
        "source_sha256": readiness_sha})

    return {
        "schema": PACKAGE_SCHEMA,
        "identities": {
            "invocation_sha256": invocation["invocation_sha256"],
            "manifest_commit": summary["manifest_commit"],
            "selector": dict(selector_identity),
            "geometry_capsule_digest":
                (sorted(capsule_digests)[0] if capsule_digests
                 else None),
            "calendar_authority_sha256": calendar_sha,
            "calendar_frame_id": calendar["frame"]["frame_id"],
            "effect_grids_sha256": grids_sha,
            "power_harness_slot_pins": harness_pins,
            "ordered_points_sha256":
                summary["ordered_points_sha256"]},
        "four_family_result": four_family,
        "certified_s": certified_s,
        "admission_table": table,
        "semantics": {
            "admitted_requires_envelope_and_threshold": True,
            "manifest_closure_is_never_semantic_pass": True,
            "rule": "any lane/member marked admitted for accrual "
                    "without a registered anticipated-mask envelope "
                    "+ threshold REFUSES the power gate and "
                    "prestart_overall"},
        "claim_ceiling": "synthetic anticipated-mask design power "
                         "only; not detection, not admission, not "
                         "evidence about the Earth; Lambda_geo "
                         "INCONCLUSIVE"}


def harness_pins_from_manifest_bytes(man_b):
    """Shared: the power_harness pin identities from exact manifest
    BYTES (the invocation's named commit -- both the assembler and
    the independent verifier resolve through this one function)."""
    man = json.loads(man_b.decode("utf-8"))
    slot = (man.get("slots") or {}).get("power_harness")
    if not isinstance(slot, dict) or slot.get("status") != "BOUND":
        _refuse("POWER_RESULT_HARNESS_SLOT",
                "power_harness slot absent or not BOUND at the "
                "invocation manifest commit")
    return [{"path": p["path"], "blob_sha256": p["blob_sha256"]}
            for p in slot["pins"]]


def _harness_pins_at(repo, commit):
    r = subprocess.run(["git", "-C", repo, "cat-file", "blob",
                        f"{commit}:{MANIFEST_REL}"],
                       capture_output=True)
    if r.returncode != 0 or not r.stdout:
        _refuse("POWER_RESULT_HARNESS_SLOT",
                f"manifest unreadable at invocation commit "
                f"{commit[:12]}")
    return harness_pins_from_manifest_bytes(r.stdout)


def gather_inputs(repo, outdir):
    """Authenticated input gathering for the production entry --
    every loader is the pinned runner's own."""
    inv_raw = _read(os.path.join(outdir, "invocation_record.json"),
                    "invocation record")
    inv_obj = json.loads(inv_raw.decode("utf-8"))
    invocation, points = RUN._load_invocation(
        outdir, inv_obj.get("invocation_sha256"))
    # cycle-4 R2: `points` IS the authority; it flows into the pure
    # constructor via invocation["ordered_points"] (same object,
    # digest-authenticated above) -- never discarded.
    summary = json.loads(_read(
        os.path.join(outdir, "campaign_summary.json"),
        "campaign summary").decode("utf-8"))
    if os.path.isfile(os.path.join(outdir, "campaign_aborted.json")):
        _refuse("POWER_RESULT_CAMPAIGN_ABORTED",
                "an aborted campaign never assembles")
    point_files = []
    for i in range(int(summary["n_points"])):
        point_files.append(json.loads(_read(
            os.path.join(outdir, f"point_{i:03d}.json"),
            f"point {i}").decode("utf-8")))
    selector, sel_points, selector_sha = \
        RUN.load_selector_committed(repo, summary["selector_commit"],
                                    summary["selector_path"])
    if selector_sha != summary["selector_sha256"]:
        _refuse("POWER_RESULT_SELECTOR_IDENTITY",
                "committed selector bytes diverge from the summary")
    selector_identity = {"commit": summary["selector_commit"],
                         "path": summary["selector_path"],
                         "sha256": selector_sha}
    grids_b = _read(os.path.join(repo, GRIDS_REL.replace(
        "/", os.sep)), "effect grids")
    cal_b = _read(os.path.join(repo, CALENDAR_REL.replace(
        "/", os.sep)), "calendar authority v4")
    disp_b = _read(os.path.join(repo, DISPOSITION_REL.replace(
        "/", os.sep)), "terminal-exclusion disposition")
    ready_b = _read(os.path.join(repo, READINESS_REL.replace(
        "/", os.sep)), "successor-readiness record")
    # cycle-4 R2: harness pins + geometry authority resolve at
    # the INVOCATION's named manifest commit from Git objects (never
    # the later working/result manifest), and that commit must be an
    # ancestor of the current tree state.
    inv_mc = invocation["manifest_commit"]
    anc = subprocess.run(
        ["git", "-C", repo, "merge-base", "--is-ancestor",
         inv_mc, "HEAD"], capture_output=True)
    if anc.returncode != 0:
        _refuse("POWER_RESULT_INVOCATION_CORE_DIVERGENT",
                f"invocation manifest_commit {inv_mc[:12]} is not "
                "an ancestor of the assembling tree")
    return {
        "invocation": invocation, "summary": summary,
        "point_files": point_files, "selector": selector,
        "selector_identity": selector_identity,
        "grids": json.loads(grids_b.decode("utf-8")),
        "grids_sha": _sha(grids_b),
        "calendar": json.loads(cal_b.decode("utf-8")),
        "calendar_sha": _sha(cal_b),
        "disposition_sha": _sha(disp_b),
        "readiness": json.loads(ready_b.decode("utf-8")),
        "readiness_sha": _sha(ready_b),
        "harness_pins": _harness_pins_at(repo, inv_mc)}


def _receipt_outputs(repo, outdir, inputs, pkg_sha):
    rel = os.path.relpath(outdir, repo).replace(os.sep, "/")
    out = {PACKAGE_REL: pkg_sha}
    for nm in (["invocation_record.json", "campaign_summary.json"]
               + [f"point_{i:03d}.json"
                  for i in range(len(inputs["point_files"]))]):
        with open(os.path.join(outdir, nm), "rb") as f:
            out[rel + "/" + nm] = _sha(f.read())
    return out


def assemble(repo, outdir, argv=None):
    """THE production entry: (repo, outdir) only. Validates ALL,
    builds in memory, publishes package then receipt via the
    runner's atomic create-once primitive. A pre-existing package
    or receipt refuses -- assembly is once."""
    t0 = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    inputs = gather_inputs(repo, outdir)
    package = build_package_content(**inputs)
    pkg_body = json.dumps(package, indent=1, sort_keys=True) + "\n"
    pkg_path = os.path.join(repo, PACKAGE_REL.replace("/", os.sep))
    rcpt_path = os.path.join(repo, RECEIPT_REL.replace("/", os.sep))
    for p in (pkg_path, rcpt_path):
        if os.path.exists(p):
            _refuse("POWER_RESULT_ALREADY_PUBLISHED", p)
    os.makedirs(os.path.dirname(pkg_path), exist_ok=True)
    RUN._publish_once(pkg_path, pkg_body)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "invocation_argv": list(argv if argv is not None
                                else sys.argv),
        "host": platform.node(),
        "interpreter": sys.version.replace("\n", " "),
        "started_utc": t0,
        "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                   time.gmtime()),
        "exit_code": 0,
        # cycle-4 R2: the output map is CLOSED and EXACT over
        # every byte artifact of the operation -- invocation,
        # summary, every declared point file, and the package. The
        # independent verifier recomputes each raw sha.
        "outputs": _receipt_outputs(repo, outdir, inputs,
                                    _sha(pkg_body.encode())),
        "campaign": {
            "outdir": os.path.relpath(outdir, repo).replace(
                os.sep, "/"),
            "invocation_sha256":
                package["identities"]["invocation_sha256"],
            "n_points": package["certified_s"]["n_points"]},
        "claim_ceiling": package["claim_ceiling"]}
    rcpt_body = json.dumps(receipt, indent=1, sort_keys=True) + "\n"
    RUN._publish_once(rcpt_path, rcpt_body)
    return {"package_sha256": _sha(pkg_body.encode()),
            "receipt_sha256": _sha(rcpt_body.encode())}


def _selftest():
    """cycle-4 binding + semantics KATs (codex cycle-3 findings
    2+3 folded onto the cycle-3 matrix): the fired invocation is
    the authority (coherent rewrites refuse INVOCATION_CORE_
    DIVERGENT); the strict oracle refuses malformed power semantics
    (codex's exact malformed control locked); receipt output maps,
    timestamps, and invocation ancestry are enforced; the S5 split
    holds."""
    import copy
    import w2_power_cert_result_verifier_cayley as VER
    repo = os.path.abspath(os.path.join(_HERE, "..", ".."))

    def rf(rel):
        with open(os.path.join(repo, rel.replace("/", os.sep)),
                  "rb") as f:
            return f.read()
    grids_b = rf(GRIDS_REL)
    cal4_b = rf(CALENDAR_REL)
    cal3_b = rf("docs/f2g_window2_execution/"
                "calendar_authority_w2_v3.json")
    disp_b = rf(DISPOSITION_REL)
    ready_b = rf(READINESS_REL)
    man_b = rf(MANIFEST_REL)
    grids = json.loads(grids_b.decode("utf-8"))
    cal4 = json.loads(cal4_b.decode("utf-8"))
    ready = json.loads(ready_b.decode("utf-8"))
    cal4_sha = _sha(cal4_b)
    harness_pins = harness_pins_from_manifest_bytes(man_b)
    INV_MC = "c" * 40

    def mk_specs():
        out = []
        for fam, take in (("B1B", 4), ("B2A", 3), ("B2B", 2),
                          ("B3A", 3)):
            for pt in grids["grids"][fam][:take]:
                out.append({"family": fam, "entry": "detection",
                            "point": dict(pt)})
        for g in (3, 10):
            out.append({"family": "B1B", "entry": "specificity",
                        "point": {"gain": g}})
        return out
    specs = mk_specs()

    def mk_rec(spec, status="CERTIFIED"):
        base = {"family": spec["family"], "point": spec["point"],
                "tier": "CERTIFICATION", "n_draws": 9999,
                "certifiable": True,
                "geometry_capsule_digest": "a" * 64,
                "geometry_ref": {"manifest_commit": INV_MC,
                                 "path": "docs/kat/geom.json"},
                "seed_authority_sha256": "d" * 64,
                "calendar_authority_sha256": cal4_sha,
                "calendar_frame_id": "w2-calendar-v4-noncal",
                "baseline_days_sha256": "e" * 64,
                "evaluation_days_sha256": "f" * 64}
        if spec["entry"] == "specificity":
            base.update({"class": "B1B_GAIN_STEP_SPECIFICITY",
                         "status": status, "R": 40, "positives": 1,
                         "rate": 0.025, "passes": True,
                         "max_rate": 0.05, "outcomes": [],
                         "rule": "kat"})
        else:
            base.update({"status": status, "R": 20, "k": 19,
                         "lb": 0.83,
                         "trace": [{"R": 20, "k": 19, "lb": 0.83,
                                    "ub": 0.99}]})
        return base

    def mk_campaign(recs, specs_=None, sel_specs=None):
        sp = specs if specs_ is None else specs_
        sel_art = {"ordered_points":
                   sp if sel_specs is None else sel_specs}
        sel_raw = json.dumps(sel_art, indent=1,
                             sort_keys=True).encode()
        inv = {"schema": "f2g-w2-cert-invocation-v3",
               "ordered_points": sp,
               "ordered_points_sha256": RUN._digest(sp),
               "manifest_commit": INV_MC,
               "geometry_path": "docs/kat/geom.json",
               "selector_commit": "5" * 40,
               "selector_path": "docs/kat/selector.json",
               "selector_sha256": _sha(sel_raw),
               "n_procs": 7, "argv": ["kat"]}
        inv["invocation_sha256"] = RUN._invocation_digest(inv)
        summary = {"schema": "f2g-w2-cert-campaign-summary-v3",
                   "completed_utc": "KAT", "n_points": len(sp),
                   "order_started": list(range(len(sp))),
                   "invocation_sha256": inv["invocation_sha256"],
                   "manifest_commit": INV_MC,
                   "selector_commit": "5" * 40,
                   "selector_path": "docs/kat/selector.json",
                   "selector_sha256": _sha(sel_raw),
                   "geometry_path": "docs/kat/geom.json",
                   "per_point": [
                       {"index": i, "family": sp[i]["family"],
                        "entry": sp[i]["entry"],
                        "point": sp[i]["point"],
                        "status": recs[i]["status"],
                        "record_sha256": _canon(recs[i])}
                       for i in range(len(sp))],
                   "ordered_points_sha256":
                       inv["ordered_points_sha256"]}
        pfs = [{"index": i, "spec": sp[i],
                "invocation_sha256": inv["invocation_sha256"],
                "record": recs[i], "refusal": None}
               for i in range(len(sp))]
        return inv, summary, pfs, sel_art, sel_raw

    recs = [mk_rec(sp_) for sp_ in specs]
    inv, summary, pfs, sel_art, sel_raw = mk_campaign(recs)
    ident = {"commit": "5" * 40, "path": "docs/kat/selector.json",
             "sha256": _sha(sel_raw)}

    def build(**over):
        kw = dict(invocation=inv, summary=summary, point_files=pfs,
                  selector=sel_art, selector_identity=ident,
                  grids=grids, grids_sha=_sha(grids_b),
                  calendar=cal4, calendar_sha=cal4_sha,
                  disposition_sha=_sha(disp_b), readiness=ready,
                  readiness_sha=_sha(ready_b),
                  harness_pins=harness_pins)
        kw.update(over)
        return build_package_content(**kw)

    def rbm_for(pfs_):
        return {_member_key(pf["spec"]["family"],
                            pf["spec"]["entry"],
                            pf["spec"]["point"]):
                {"record": pf["record"],
                 "record_sha256": _canon(pf["record"])}
                for pf in pfs_}

    pkg = build()
    admitted = [r for r in pkg["admission_table"]
                if r["state"] == "ADMITTED_WITH_POWER"]
    assert pkg["certified_s"]["status"] == "CERTIFIED_S"
    assert len(admitted) == len(specs)
    assert len(pkg["admission_table"]) == \
        len(_grid_members(grids)) + 2
    assert VER.check_admission_semantics(
        pkg, grids, readiness_sha=_sha(ready_b),
        records_by_member=rbm_for(pfs)) == []

    def refuse(code, **over):
        try:
            build(**over)
            raise SystemExit("assembly doctor must refuse: " + code)
        except AssemblyRefusal as ex:
            assert code in str(ex), (code, str(ex))

    # census (missing / extra point files)
    refuse("POWER_RESULT_POINT_CENSUS", point_files=pfs[:-1])
    refuse("POWER_RESULT_POINT_CENSUS",
           point_files=pfs + [copy.deepcopy(pfs[-1])])
    # swapped records under coherent summary digests
    r2 = [copy.deepcopy(r) for r in recs]
    r2[0], r2[1] = r2[1], r2[0]
    inv2, sum2, pfs2, _, _ = mk_campaign(r2)
    try:
        build(invocation=inv2, summary=sum2, point_files=pfs2)
        raise SystemExit("swapped records must refuse")
    except AssemblyRefusal as ex:
        assert "POWER_RESULT_RECORD_IDENTITY" in str(ex), str(ex)
    # duplicate member IN THE INVOCATION (digest-coherent)
    specs_dup = [copy.deepcopy(x) for x in specs]
    specs_dup[1] = copy.deepcopy(specs_dup[0])
    recs_dup = [mk_rec(sp_) for sp_ in specs_dup]
    invd, sumd, pfsd, seld, selrd = mk_campaign(recs_dup, specs_dup)
    try:
        build(invocation=invd, summary=sumd, point_files=pfsd,
              selector=seld,
              selector_identity=dict(ident, sha256=_sha(selrd)))
        raise SystemExit("duplicate member must refuse")
    except AssemblyRefusal as ex:
        assert "POWER_RESULT_DUPLICATE_MEMBER" in str(ex), str(ex)
    # wrong-hash: record tampered under a stale summary digest
    pfs4 = [copy.deepcopy(x) for x in pfs]
    pfs4[2]["record"]["lb"] = 0.99
    refuse("POWER_RESULT_RECORD_DIGEST", point_files=pfs4)
    # wrong-calendar (both directions)
    refuse("POWER_RESULT_CALENDAR_WRONG",
           calendar=json.loads(cal3_b.decode("utf-8")),
           calendar_sha=_sha(cal3_b))
    r5 = [copy.deepcopy(r) for r in recs]
    for r_ in r5:
        r_["calendar_authority_sha256"] = "b" * 64
    inv5, sum5, pfs5, _, _ = mk_campaign(r5)
    refuse("POWER_RESULT_CALENDAR_WRONG", invocation=inv5,
           summary=sum5, point_files=pfs5)
    # refused point never assembles
    pfs6 = [copy.deepcopy(x) for x in pfs]
    pfs6[3]["record"] = None
    pfs6[3]["refusal"] = "POWER_GEOMETRY_UNBOUND: kat"
    refuse("POWER_RESULT_POINT_REFUSED", point_files=pfs6)

    # --- cycle-4 R2 KAT (codex item 4): coherent rewrite of
    # summary + point files + selector retaining the AUTHENTIC
    # invocation digest -> INVOCATION_CORE_DIVERGENT
    alt = {"family": "B3A", "entry": "detection",
           "point": dict(grids["grids"]["B3A"][10])}
    specs_rw = [copy.deepcopy(x) for x in specs]
    specs_rw[2] = alt
    recs_rw = [mk_rec(sp_) for sp_ in specs_rw]
    _invx, sum_rw, pfs_rw, sel_rw, selr_rw = mk_campaign(
        recs_rw, specs_rw)
    sum_rw = copy.deepcopy(sum_rw)
    sum_rw["invocation_sha256"] = inv["invocation_sha256"]
    sum_rw["ordered_points_sha256"] = inv["ordered_points_sha256"]
    sum_rw["selector_sha256"] = inv["selector_sha256"]
    for pf in pfs_rw:
        pf["invocation_sha256"] = inv["invocation_sha256"]
    try:
        build(summary=sum_rw, point_files=pfs_rw, selector=sel_rw,
              selector_identity=dict(
                  ident, sha256=inv["selector_sha256"]))
        raise SystemExit("coherent rewrite must refuse")
    except AssemblyRefusal as ex:
        assert "POWER_RESULT_INVOCATION_CORE_DIVERGENT" in str(ex), \
            str(ex)

    # S5 split: FAILED point -> valid package, typed S, gate REFUSE
    r7 = [copy.deepcopy(r) for r in recs]
    r7[5]["status"] = "FAILED"
    inv7, sum7, pfs7, _, _ = mk_campaign(r7)
    pkg7 = build(invocation=inv7, summary=sum7, point_files=pfs7)
    assert pkg7["certified_s"]["status"] == \
        "TYPED_NON_CERTIFICATION_S"
    assert VER.check_admission_semantics(
        pkg7, grids, readiness_sha=_sha(ready_b),
        records_by_member=rbm_for(pfs7)) == []

    # --- cycle-4 R3: strict-oracle doctors (each a doctored COPY)
    def sem(mutate, code, pkg_=None, pfs_=None):
        d = copy.deepcopy(pkg_ or pkg)
        mutate(d)
        got = [r_["code"] for r_ in VER.check_admission_semantics(
            d, grids, readiness_sha=_sha(ready_b),
            records_by_member=rbm_for(pfs_ or pfs))]
        assert code in got, (code, got)

    def adm(d):
        return [r_ for r_ in d["admission_table"]
                if r_.get("state") == "ADMITTED_WITH_POWER"]

    def mag(d):
        return [r_ for r_ in d["admission_table"]
                if r_.get("member", {}).get("lane")
                == "mag_primary_set"][0]

    def mf4(d):
        return [r_ for r_ in d["admission_table"]
                if r_.get("member", {}).get("lane")
                == "mf4_daily_risk"][0]

    # codex's exact malformed control: wrong_key envelope, -999
    # threshold, M-F4 relabeled admitted w/ readiness REFUSE
    def malformed(d):
        row = adm(d)[0]
        row["anticipated_mask_envelope"] = {"wrong_key": 0.0}
        row["threshold_cp_floor"] = -999
        m = mf4(d)
        m["state"] = "ADMITTED_WITH_POWER"
        m["readiness_state"] = "REFUSE"
    dm = copy.deepcopy(pkg)
    malformed(dm)
    got = [r_["code"] for r_ in VER.check_admission_semantics(
        dm, grids, readiness_sha=_sha(ready_b),
        records_by_member=rbm_for(pfs))]
    assert "POWER_VERIFY_ADMITTED_WITHOUT_ENVELOPE" in got \
        and "POWER_VERIFY_THRESHOLD_UNREGISTERED" in got \
        and "POWER_VERIFY_MF4_SEMANTICS" in got, got
    # wrong-value (right keys, wrong number vs authenticated record)
    sem(lambda d: adm(d)[0].__setitem__(
        "anticipated_mask_envelope",
        dict(adm(d)[0]["anticipated_mask_envelope"], cp_lb=0.999)),
        "POWER_VERIFY_ADMITTED_WITHOUT_ENVELOPE")
    # bool / NaN / inf
    sem(lambda d: adm(d)[0].__setitem__("threshold_cp_floor", True),
        "POWER_VERIFY_THRESHOLD_UNREGISTERED")
    sem(lambda d: adm(d)[0]["anticipated_mask_envelope"].
        __setitem__("cp_lb", float("nan")),
        "POWER_VERIFY_ADMITTED_WITHOUT_ENVELOPE")
    sem(lambda d: adm(d)[0]["anticipated_mask_envelope"].
        __setitem__("cp_lb", float("inf")),
        "POWER_VERIFY_ADMITTED_WITHOUT_ENVELOPE")
    # extra field on a closed row
    sem(lambda d: adm(d)[0].__setitem__("bonus", 1),
        "POWER_VERIFY_ROW_SCHEMA")
    # MAG tampers
    sem(lambda d: mag(d).__setitem__("non_blocking", False),
        "POWER_VERIFY_MAG_SEMANTICS")
    sem(lambda d: mag(d).__setitem__("source_sha256", "zz"),
        "POWER_VERIFY_MAG_SEMANTICS")
    # M-F4 readiness sha divergence
    sem(lambda d: mf4(d).__setitem__("source_sha256", "0" * 64),
        "POWER_VERIFY_MF4_SEMANTICS")
    # completeness / Holm-exclusion / lane set
    sem(lambda d: d["admission_table"].pop(len(specs)),
        "POWER_VERIFY_TABLE_INCOMPLETE")
    sem(lambda d: [r_ for r_ in d["admission_table"]
        if r_.get("state") == "CANNOT_DETERMINE_NO_POWER"
        ][0].__setitem__("excluded_from_holm", False),
        "POWER_VERIFY_HOLM_EXCLUSION_MISSING")
    sem(lambda d: d["admission_table"].remove(mf4(d)),
        "POWER_VERIFY_LANE_SET")

    # --- full verifier drive through the KAT seams ----------------
    T = "7" * 40
    outdir_rel = POWER_CERT_DIR
    TS = "2026-08-31T00:00:00Z"

    def store_for(pkg_obj, inv_o, sum_o, pfs_o, *, ts=(TS, TS),
                  outputs_mutate=None):
        pkg_body = (json.dumps(pkg_obj, indent=1, sort_keys=True)
                    + "\n").encode()
        files = {
            outdir_rel + "/invocation_record.json":
                (json.dumps(inv_o, indent=1, sort_keys=True)
                 + "\n").encode(),
            outdir_rel + "/campaign_summary.json":
                (json.dumps(sum_o, indent=1, sort_keys=True)
                 + "\n").encode()}
        for i, pf in enumerate(pfs_o):
            files[outdir_rel + f"/point_{i:03d}.json"] = (
                json.dumps(pf, indent=1, sort_keys=True)
                + "\n").encode()
        outputs = {PACKAGE_REL: _sha(pkg_body)}
        for rel_, b_ in files.items():
            outputs[rel_] = _sha(b_)
        if outputs_mutate:
            outputs_mutate(outputs)
        rcpt = {"schema": RECEIPT_SCHEMA, "invocation_argv": ["kat"],
                "host": "kat", "interpreter": "kat",
                "started_utc": ts[0], "ended_utc": ts[1],
                "exit_code": 0, "outputs": outputs,
                "campaign": {"outdir": outdir_rel,
                             "invocation_sha256":
                                 inv_o["invocation_sha256"],
                             "n_points": len(pfs_o)},
                "claim_ceiling": "kat"}
        st = {(T, PACKAGE_REL): pkg_body,
              (T, RECEIPT_REL): (json.dumps(
                  rcpt, indent=1, sort_keys=True) + "\n").encode(),
              (T, GRIDS_REL): grids_b, (T, CALENDAR_REL): cal4_b,
              (T, DISPOSITION_REL): disp_b,
              (T, READINESS_REL): ready_b,
              (T, MANIFEST_REL): man_b,
              (INV_MC, MANIFEST_REL): man_b}
        for rel_, b_ in files.items():
            st[(T, rel_)] = b_
        return st

    def drive(st, *, sel=None, anc=None, ct="2099-01-01T00:00:00"
              "+00:00"):
        return VER.verify(
            repo, T,
            blob_reader=lambda c, rel: st.get((c, rel)),
            selector_loader=(sel or (lambda *_a, **_k: (
                sel_art, specs, _sha(sel_raw)))),
            ancestor_check=(anc or (lambda a, b: True)),
            commit_time_utc=ct)

    res = drive(store_for(pkg, inv, summary, pfs))
    assert res["package_valid"] is True and \
        res["power_gate"] == "PASS", res["typed_reasons"]
    # coordinated package+receipt mutation still refuses
    pkgX = copy.deepcopy(pkg)
    pkgX["admission_table"][0]["anticipated_mask_envelope"][
        "cp_lb"] = 0.999
    resX = drive(store_for(pkgX, inv, summary, pfs))
    codes = [r_["code"] for r_ in resX["typed_reasons"]]
    assert "POWER_VERIFY_PACKAGE_DIVERGENT" in codes and \
        resX["power_gate"] == "REFUSE", codes
    # S5 at the full drive
    res7 = drive(store_for(pkg7, inv7, sum7, pfs7))
    assert res7["package_valid"] is True and \
        res7["power_gate"] == "REFUSE" and \
        res7["certified_s_status"] == "TYPED_NON_CERTIFICATION_S"
    # cycle-4: receipt outputs not closed-exact
    resO = drive(store_for(pkg, inv, summary, pfs,
                           outputs_mutate=lambda o: o.pop(
                               outdir_rel + "/point_000.json")))
    assert "POWER_VERIFY_RECEIPT_OUTPUTS" in [
        r_["code"] for r_ in resO["typed_reasons"]]
    # cycle-4: non-canonical + late timestamps
    resT = drive(store_for(pkg, inv, summary, pfs,
                           ts=("KAT", "KAT")))
    assert "POWER_VERIFY_RECEIPT_TIMESTAMPS" in [
        r_["code"] for r_ in resT["typed_reasons"]]
    resL = drive(store_for(pkg, inv, summary, pfs),
                 ct="2026-08-30T00:00:00+00:00")
    assert "POWER_VERIFY_RECEIPT_TIMESTAMPS" in [
        r_["code"] for r_ in resL["typed_reasons"]]
    # cycle-4: invocation ancestry refused
    resA = drive(store_for(pkg, inv, summary, pfs),
                 anc=lambda a, b: False)
    assert "POWER_VERIFY_INVOCATION_ANCESTRY" in [
        r_["code"] for r_ in resA["typed_reasons"]]

    print("w2_power_cert_results_assembly selftest: ALL PASS "
          "(cycle-4: fired-invocation authority incl. the coherent-"
          "rewrite doctor; strict oracle locks codex's malformed "
          "control + wrong-key/value/bool/NaN/inf/extra-field/"
          "wrong-state; receipt output-map, timestamp, and ancestry "
          "doctors; coordinated mutation + S5 split retained)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    elif len(sys.argv) == 3:
        out = assemble(os.path.abspath(sys.argv[1]),
                       os.path.abspath(sys.argv[2]))
        print(json.dumps(out, indent=1))
    else:
        raise SystemExit("usage: w2_power_cert_results_assembly_"
                         "cayley.py <repo> <campaign_outdir> | "
                         "--selftest")
