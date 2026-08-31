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
        if pf.get("refusal") is not None or pf.get("record") is None:
            _refuse("POWER_RESULT_POINT_REFUSED",
                    f"point {i}: a refused/absent record can never "
                    "assemble -- rerun or exclude by a NEW campaign, "
                    "never by editing")
        rec = pf["record"]
        srow = summary["per_point"][i]
        if srow["index"] != i or srow["family"] != \
                pf["spec"]["family"] or srow["point"] != \
                pf["spec"]["point"] or srow["entry"] != \
                pf["spec"]["entry"]:
            _refuse("POWER_RESULT_SUMMARY_ROW",
                    f"point {i} summary row diverges from the file")
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
    certified_s = {
        "selector": dict(selector_identity),
        "n_points": len(sel_keys),
        "all_points_certified": not s_failed,
        "status": ("CERTIFIED_S" if not s_failed else
                   "TYPED_NON_CERTIFICATION_S"),
        "non_certified_members": sorted(s_failed)}

    # admission table: every registered grid member exactly once
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
        elif m["record"]["status"] == "CERTIFIED":
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
            table.append({
                "member": {"family": fam, "entry": entry,
                           "point": point},
                "state": "TYPED_NON_CERTIFICATION",
                "excluded_from_holm": True,
                "reason": f"campaign status {m['record']['status']}",
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


def _harness_slot_pins(repo):
    mp = os.path.join(repo, MANIFEST_REL.replace("/", os.sep))
    man = json.loads(_read(mp, "execution manifest").decode("utf-8"))
    slot = (man.get("slots") or {}).get("power_harness")
    if not isinstance(slot, dict) or slot.get("status") != "BOUND":
        _refuse("POWER_RESULT_HARNESS_SLOT",
                "power_harness slot absent or not BOUND")
    return [{"path": p["path"], "blob_sha256": p["blob_sha256"]}
            for p in slot["pins"]]


def gather_inputs(repo, outdir):
    """Authenticated input gathering for the production entry --
    every loader is the pinned runner's own."""
    inv_raw = _read(os.path.join(outdir, "invocation_record.json"),
                    "invocation record")
    inv_obj = json.loads(inv_raw.decode("utf-8"))
    invocation, points = RUN._load_invocation(
        outdir, inv_obj.get("invocation_sha256"))
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
        "harness_pins": _harness_slot_pins(repo)}


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
        "outputs": {
            PACKAGE_REL: _sha(pkg_body.encode()),
        },
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
    """Binding + semantics KATs (codex cycle-2 finding 2): control
    passes, then missing / swapped / extra / duplicate / wrong-hash /
    wrong-calendar / refused-point / coordinated package+receipt
    mutations refuse typed, the S5 split holds (a valid package with
    a typed non-certified S NEVER passes the power gate), and every
    admission-semantics gate refuses on a doctored table. Fixtures
    use the REAL committed grids/calendar/disposition/readiness/
    manifest bytes of this tree; campaign objects mirror the pinned
    runner's published shapes exactly."""
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
    man = json.loads(man_b.decode("utf-8"))
    harness_pins = [{"path": pn["path"],
                     "blob_sha256": pn["blob_sha256"]}
                    for pn in man["slots"]["power_harness"]["pins"]]

    # --- fixture campaign: 12 detection + gains 3 then 10 ---------
    specs = []
    for fam, take in (("B1B", 4), ("B2A", 3), ("B2B", 2),
                      ("B3A", 3)):
        for pt in grids["grids"][fam][:take]:
            specs.append({"family": fam, "entry": "detection",
                          "point": dict(pt)})
    for g in (3, 10):
        specs.append({"family": "B1B", "entry": "specificity",
                      "point": {"gain": g}})
    cal4_sha = _sha(cal4_b)

    def mk_rec(spec, status="CERTIFIED"):
        base = {"family": spec["family"], "point": spec["point"],
                "tier": "CERTIFICATION", "n_draws": 9999,
                "certifiable": True,
                "geometry_capsule_digest": "a" * 64,
                "geometry_ref": {"manifest_commit": "c" * 40,
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

    def mk_campaign(recs):
        inv = {"schema": "f2g-w2-cert-invocation-v3",
               "ordered_points": specs,
               "ordered_points_sha256": RUN._digest(specs),
               "manifest_commit": "c" * 40,
               "geometry_path": "docs/kat/geom.json",
               "selector_commit": "5" * 40,
               "selector_path": "docs/kat/selector.json",
               "n_procs": 7, "argv": ["kat"]}
        inv["invocation_sha256"] = RUN._invocation_digest(inv)
        sel_art = {"ordered_points": specs}
        sel_raw = json.dumps(sel_art, indent=1,
                             sort_keys=True).encode()
        summary = {"schema": "f2g-w2-cert-campaign-summary-v3",
                   "completed_utc": "KAT", "n_points": len(specs),
                   "order_started": list(range(len(specs))),
                   "invocation_sha256": inv["invocation_sha256"],
                   "manifest_commit": "c" * 40,
                   "selector_commit": "5" * 40,
                   "selector_path": "docs/kat/selector.json",
                   "selector_sha256": _sha(sel_raw),
                   "geometry_path": "docs/kat/geom.json",
                   "per_point": [
                       {"index": i, "family": specs[i]["family"],
                        "entry": specs[i]["entry"],
                        "point": specs[i]["point"],
                        "status": recs[i]["status"],
                        "record_sha256": _canon(recs[i])}
                       for i in range(len(specs))],
                   "ordered_points_sha256":
                       inv["ordered_points_sha256"]}
        pfs = [{"index": i, "spec": specs[i],
                "invocation_sha256": inv["invocation_sha256"],
                "record": recs[i], "refusal": None}
               for i in range(len(specs))]
        return inv, summary, pfs, sel_art, sel_raw

    recs = [mk_rec(sp) for sp in specs]
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

    pkg = build()
    admitted = [r for r in pkg["admission_table"]
                if r["state"] == "ADMITTED_WITH_POWER"]
    assert pkg["certified_s"]["status"] == "CERTIFIED_S"
    assert len(admitted) == len(specs)
    n_members = len(_grid_members(grids))
    assert len(pkg["admission_table"]) == n_members + 2
    assert VER.check_admission_semantics(pkg, grids) == []

    def refuse(code, **over):
        try:
            build(**over)
            raise SystemExit("assembly doctor must refuse: " + code)
        except AssemblyRefusal as ex:
            assert code in str(ex), (code, str(ex))

    # missing / extra point files
    refuse("POWER_RESULT_POINT_CENSUS", point_files=pfs[:-1])
    refuse("POWER_RESULT_POINT_CENSUS",
           point_files=pfs + [copy.deepcopy(pfs[-1])])
    # swapped records (summary digests updated coherently)
    r2 = [copy.deepcopy(r) for r in recs]
    r2[0], r2[1] = r2[1], r2[0]
    inv2, sum2, pfs2, _, _ = mk_campaign(r2)
    try:
        build(invocation=inv2, summary=sum2, point_files=pfs2)
        raise SystemExit("swapped records must refuse")
    except AssemblyRefusal as ex:
        assert "POWER_RESULT_RECORD_IDENTITY" in str(ex), str(ex)
    # duplicate member (file+summary coherently duplicated)
    pfs3 = [copy.deepcopy(x) for x in pfs]
    sum3 = copy.deepcopy(summary)
    pfs3[1]["spec"] = copy.deepcopy(pfs3[0]["spec"])
    pfs3[1]["record"] = copy.deepcopy(pfs3[0]["record"])
    sum3["per_point"][1] = dict(sum3["per_point"][0], index=1)
    refuse("POWER_RESULT_DUPLICATE_MEMBER", summary=sum3,
           point_files=pfs3)
    # wrong-hash: record tampered under a stale summary digest
    pfs4 = [copy.deepcopy(x) for x in pfs]
    pfs4[2]["record"]["lb"] = 0.99
    refuse("POWER_RESULT_RECORD_DIGEST", point_files=pfs4)
    # wrong-calendar: the v3 authority at the v4 seat
    refuse("POWER_RESULT_CALENDAR_WRONG",
           calendar=json.loads(cal3_b.decode("utf-8")),
           calendar_sha=_sha(cal3_b))
    # wrong-calendar: v4 seat, records citing another authority
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

    # S5 split: FAILED point -> valid package, typed S, gate REFUSE
    r7 = [copy.deepcopy(r) for r in recs]
    r7[5]["status"] = "FAILED"
    inv7, sum7, pfs7, _, _ = mk_campaign(r7)
    pkg7 = build(invocation=inv7, summary=sum7, point_files=pfs7)
    assert pkg7["certified_s"]["status"] == \
        "TYPED_NON_CERTIFICATION_S"
    assert VER.check_admission_semantics(pkg7, grids) == []

    # --- semantic doctors (each on a doctored COPY) ---------------
    def sem(mutate, code):
        d = copy.deepcopy(pkg)
        mutate(d["admission_table"])
        got = [r_["code"] for r_ in
               VER.check_admission_semantics(d, grids)]
        assert code in got, (code, got)
    sem(lambda t: t[0].update(anticipated_mask_envelope=None),
        "POWER_VERIFY_ADMITTED_WITHOUT_ENVELOPE")
    sem(lambda t: [r_ for r_ in t
        if r_.get("member", {}).get("lane") == "mag_primary_set"
        ][0].update(non_blocking=False),
        "POWER_VERIFY_MAG_SEMANTICS")
    sem(lambda t: t.pop(len(specs)),
        "POWER_VERIFY_TABLE_INCOMPLETE")
    sem(lambda t: [r_ for r_ in t
        if r_.get("state") == "CANNOT_DETERMINE_NO_POWER"
        ][0].update(excluded_from_holm=False),
        "POWER_VERIFY_HOLM_EXCLUSION_MISSING")
    sem(lambda t: t.remove([r_ for r_ in t
        if r_.get("member", {}).get("lane") == "mf4_daily_risk"][0]),
        "POWER_VERIFY_LANE_SET")

    # --- full verifier drive through the KAT seams ----------------
    T = "7" * 40
    outdir_rel = POWER_CERT_DIR

    def store_for(pkg_obj, inv_o, sum_o, pfs_o):
        pkg_body = (json.dumps(pkg_obj, indent=1, sort_keys=True)
                    + "\n").encode()
        rcpt = {"schema": RECEIPT_SCHEMA, "invocation_argv": ["kat"],
                "host": "kat", "interpreter": "kat",
                "started_utc": "KAT", "ended_utc": "KAT",
                "exit_code": 0,
                "outputs": {PACKAGE_REL: _sha(pkg_body)},
                "campaign": {"outdir": outdir_rel,
                             "invocation_sha256":
                                 inv_o["invocation_sha256"],
                             "n_points": len(specs)},
                "claim_ceiling": "kat"}
        st = {(T, PACKAGE_REL): pkg_body,
              (T, RECEIPT_REL): (json.dumps(
                  rcpt, indent=1, sort_keys=True) + "\n").encode(),
              (T, outdir_rel + "/invocation_record.json"):
                  (json.dumps(inv_o, indent=1, sort_keys=True)
                   + "\n").encode(),
              (T, outdir_rel + "/campaign_summary.json"):
                  (json.dumps(sum_o, indent=1, sort_keys=True)
                   + "\n").encode(),
              (T, GRIDS_REL): grids_b, (T, CALENDAR_REL): cal4_b,
              (T, DISPOSITION_REL): disp_b,
              (T, READINESS_REL): ready_b,
              (T, MANIFEST_REL): man_b}
        for i, pf in enumerate(pfs_o):
            st[(T, outdir_rel + f"/point_{i:03d}.json")] = (
                json.dumps(pf, indent=1, sort_keys=True)
                + "\n").encode()
        return st

    def drive(st):
        return VER.verify(
            repo, T,
            blob_reader=lambda c, rel: st.get((c, rel)),
            selector_loader=lambda *_a, **_k: (
                sel_art, specs, _sha(sel_raw)))

    res = drive(store_for(pkg, inv, summary, pfs))
    assert res["package_valid"] is True and \
        res["power_gate"] == "PASS", res["typed_reasons"]
    # coordinated package+receipt mutation: both rewritten
    # coherently -- the shared-constructor rebuild still refuses
    pkgX = copy.deepcopy(pkg)
    pkgX["admission_table"][0]["anticipated_mask_envelope"][
        "cp_lb"] = 0.999
    resX = drive(store_for(pkgX, inv, summary, pfs))
    codes = [r_["code"] for r_ in resX["typed_reasons"]]
    assert "POWER_VERIFY_PACKAGE_DIVERGENT" in codes and \
        resX["power_gate"] == "REFUSE", codes
    # S5 at the full drive: valid bytes, typed S, gate REFUSE
    res7 = drive(store_for(pkg7, inv7, sum7, pfs7))
    assert res7["package_valid"] is True and \
        res7["power_gate"] == "REFUSE" and \
        res7["certified_s_status"] == "TYPED_NON_CERTIFICATION_S"

    print("w2_power_cert_results_assembly selftest: ALL PASS "
          "(control + census/swap/duplicate/wrong-hash/"
          "wrong-calendar/refused doctors; semantic gates S1-S4 "
          "refuse doctored tables; full verifier drive: control "
          "PASS, coordinated package+receipt mutation refuses "
          "PACKAGE_DIVERGENT, S5 split holds -- a valid package "
          "with typed non-certified S never passes the power gate)")


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
