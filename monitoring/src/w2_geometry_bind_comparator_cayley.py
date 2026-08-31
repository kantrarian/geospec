#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""W2 GEOMETRY BIND COMPARATOR (cayley) -- codex ruling
2026-08-31T15:54Z item 1 (CRITICAL).

At the 2026-09-02 bind the REALIZED geometry must be compared against
the geometry the power certificate was computed over. Codex's ruling
is that disclosure is necessary but NOT sufficient:

  - station registry and station-to-segment map: **exact equality**,
    absent a separately preregistered executable equivalence
    envelope. The harness puts station identity into the LOCO seed
    token (`fold=loco:<station>`), so an identity change changes the
    null substreams, and a segment change changes the graph geometry.
    Equal counts or histograms are NOT equivalence -- so ORDER is
    compared too, because the replicate RNG indexes station rows by
    registry order.
  - anticipated mask: a realized mask inside the registered,
    executable envelope may inherit the envelope certificate; outside
    it is the already-specified typed no-run.

This module emits the machine-readable comparator: exact ordered and
canonical station sets, masks and maps, with additions, removals,
movements and both-side digests, and a typed verdict prestart
consumes.

ONE SEMANTIC IS DELIBERATELY NOT DECIDED HERE. Which mask relations
inherit the certificate (exact only? realized-superset? a dominating
bound?) is a preregistered scientific choice, and I found no committed
artifact stating it -- `calendar_authority_w2_v2.md` registers only
that availability is a SEPARATE bound mask that never compacts the
grid. So this comparator CLASSIFIES the mask relation exactly
(EXACT / REALIZED_SUPERSET / REALIZED_SUBSET / DIVERGENT) and consumes
a registered `mask_envelope_policy`. With no policy bound it refuses
typed -- the fail-closed reading of "outside it is the typed no-run" --
rather than silently electing a rule. Routed to codex.

Compares only. Fires nothing, captures nothing, admits nothing,
certifies nothing. Lambda_geo INCONCLUSIVE.
"""
import hashlib
import json
import os
import subprocess
import sys

SCHEMA = "f2g-w2-geometry-bind-comparison-v1"
MASK_RELATIONS = ("EXACT", "REALIZED_SUPERSET", "REALIZED_SUBSET",
                  "DIVERGENT")

# cycle-6 review item 2 (CRITICAL): codex ruled EXACT-ONLY, because
# the bundle declares the anticipated mask to be the MAXIMAL full
# engine grid. A REALIZED_SUBSET is an outage this full-mask artifact
# never power-certified; a REALIZED_SUPERSET is impossible on a
# maximal anticipation and would mean an out-of-grid, excluded or
# duplicate day rather than "more data"; DIVERGENT is outside the
# frame. The policy is an ARTIFACT resolved from its manifest pin --
# a caller dict carrying the same words is not authority.
POLICY_SCHEMA = "f2g-w2-mask-envelope-policy-v1"
POLICY_REL = ("docs/f2g_window2_execution/"
              "mask_envelope_policy_w2_v1.json")
INPUTS_BUNDLE_REL = ("docs/f2g_window2_execution/"
                     "power_geometry_inputs_w2_v1.json")
CALENDAR_V4_REL = ("docs/f2g_window2_execution/"
                   "calendar_authority_w2_v4.json")
MANIFEST_REL = "docs/f2g_window2_execution/execution_manifest.json"
ADMITTED_RELATIONS = ("EXACT",)


class BindComparatorRefusal(ValueError):
    pass


def _canon(obj):
    return json.dumps(obj, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _digest(obj):
    return hashlib.sha256(_canon(obj)).hexdigest()


def _ordered_digest(seq):
    """Digest of a sequence AS ORDERED. Distinct from the set digest
    on purpose: a reordered registry has an identical set digest and
    a different ordered digest, and the replicate RNG indexes station
    rows by that order."""
    return hashlib.sha256(
        _canon([str(x) for x in seq])).hexdigest()


def _set_digest(seq):
    return hashlib.sha256(_canon(sorted(str(x) for x in seq))).hexdigest()


def compare_registries(anticipated, realized):
    """Per carrier: exact ordered equality. Reports membership adds
    and removes AND a separate order-only divergence, so a reviewer
    can see which of the two happened."""
    out = {}
    for ck in sorted(set(anticipated) | set(realized)):
        a = [str(s) for s in anticipated.get(ck, [])]
        r = [str(s) for s in realized.get(ck, [])]
        added = sorted(set(r) - set(a))
        removed = sorted(set(a) - set(r))
        same_set = not added and not removed
        out[ck] = {
            "exact": a == r,
            "added": added,
            "removed": removed,
            "order_only_divergence": bool(same_set and a != r),
            "anticipated_count": len(a),
            "realized_count": len(r),
            "anticipated_ordered_sha256": _ordered_digest(a),
            "realized_ordered_sha256": _ordered_digest(r),
            "anticipated_set_sha256": _set_digest(a),
            "realized_set_sha256": _set_digest(r)}
    return out


def compare_segment_maps(anticipated, realized):
    """Per carrier: exact station->segment equality, with MOVEMENTS
    (same station, different segment) reported separately from
    membership changes -- a movement is the change that silently
    alters the graph while every count stays put."""
    out = {}
    for ck in sorted(set(anticipated) | set(realized)):
        a = {str(k): str(v) for k, v in
             (anticipated.get(ck) or {}).items()}
        r = {str(k): str(v) for k, v in
             (realized.get(ck) or {}).items()}
        moved = sorted(
            ({"station": k, "from": a[k], "to": r[k]}
             for k in set(a) & set(r) if a[k] != r[k]),
            key=lambda m: m["station"])
        out[ck] = {
            "exact": a == r,
            "added": sorted(set(r) - set(a)),
            "removed": sorted(set(a) - set(r)),
            "moved": moved,
            "anticipated_sha256": _digest(a),
            "realized_sha256": _digest(r),
            "anticipated_active_segments": sorted(set(a.values())),
            "realized_active_segments": sorted(set(r.values()))}
    return out


def classify_mask(anticipated_days, realized_days):
    a, r = set(anticipated_days), set(realized_days)
    if a == r:
        rel = "EXACT"
    elif a < r:
        rel = "REALIZED_SUPERSET"
    elif r < a:
        rel = "REALIZED_SUBSET"
    else:
        rel = "DIVERGENT"
    return rel, sorted(r - a), sorted(a - r)


def compare_masks(anticipated, realized):
    out = {}
    for ck in sorted(set(anticipated) | set(realized)):
        a = [str(d) for d in anticipated.get(ck, [])]
        r = [str(d) for d in realized.get(ck, [])]
        rel, gained, lost = classify_mask(a, r)
        out[ck] = {
            "relation": rel,
            "days_gained": gained,
            "days_lost": lost,
            "anticipated_count": len(a),
            "realized_count": len(r),
            "anticipated_sha256": _ordered_digest(a),
            "realized_sha256": _ordered_digest(r)}
    return out


def _git_blob(repo, commit, rel):
    p = subprocess.run(["git", "-C", repo, "cat-file", "blob",
                        f"{commit}:{rel}"], capture_output=True)
    if p.returncode != 0 or not p.stdout:
        return None
    return p.stdout


def build_policy(repo, commit, *, comparator_bytes=None,
                 calendar_bytes=None, bundle_bytes=None):
    """Emit the closed EXACT-ONLY policy, bound to the calendar frame,
    the anticipated-mask digests, the carrier set and this
    comparator's own identity (codex cycle-6 item 2).

    `comparator_bytes` supplies this module's identity. codex
    cycle-6b item 1: a worktree `__file__` read must NOT be the
    comparison authority, so the LOAD path passes the
    manifest-pinned comparator blob and only artifact EMISSION falls
    back to the local file.
    """
    cal_b = (calendar_bytes if calendar_bytes is not None
             else _git_blob(repo, commit, CALENDAR_V4_REL))
    bun_b = (bundle_bytes if bundle_bytes is not None
             else _git_blob(repo, commit, INPUTS_BUNDLE_REL))
    if cal_b is None or bun_b is None:
        raise BindComparatorRefusal(
            "MASK_POLICY_INPUTS_ABSENT: calendar authority or "
            f"geometry inputs bundle unreadable at {str(commit)[:12]}")
    if comparator_bytes is None:
        with open(os.path.abspath(__file__), "rb") as f:
            comparator_bytes = f.read()
    me = comparator_bytes.replace(b"\r\n", b"\n")
    cal = json.loads(cal_b.decode("utf-8"))
    bundle = json.loads(bun_b.decode("utf-8"))
    frame = cal["frame"]
    masks = bundle["carrier_masks"]
    return {
        "schema": POLICY_SCHEMA,
        "admitted_relations": list(ADMITTED_RELATIONS),
        "ruling_basis": "codex cycle-6 combined review "
                        "2026-08-31T18:07Z item 2 (CRITICAL): the "
                        "anticipated mask is the MAXIMAL full engine "
                        "grid, so EXACT is the only relation this "
                        "power work supports. REALIZED_SUBSET is an "
                        "uncertified outage; REALIZED_SUPERSET is "
                        "impossible on a maximal anticipation and "
                        "would signal an out-of-grid, excluded or "
                        "duplicate day, not more data; DIVERGENT is "
                        "outside the frame",
        "calendar_frame": {
            "path": CALENDAR_V4_REL,
            "sha256": hashlib.sha256(cal_b).hexdigest(),
            "frame_id": frame["frame_id"],
            "engine_days": len(frame["engine_days"]),
            "excluded_days": list(frame["excluded_days"])},
        "anticipated_masks": {
            "path": INPUTS_BUNDLE_REL,
            "sha256": hashlib.sha256(bun_b).hexdigest(),
            "carriers": sorted(masks),
            "per_carrier_sha256": {
                ck: _ordered_digest(masks[ck])
                for ck in sorted(masks)},
            "per_carrier_count": {ck: len(masks[ck])
                                  for ck in sorted(masks)}},
        "comparator_identity": {
            "path": "monitoring/src/"
                    "w2_geometry_bind_comparator_cayley.py",
            "sha256_lf": hashlib.sha256(me).hexdigest()},
        "well_formedness": [
            "a realized mask with DUPLICATE days refuses",
            "a realized mask with days OUTSIDE the engine grid "
            "refuses",
            "a realized mask carrying an EXCLUDED (PRESTART) day "
            "refuses"],
        "claim_ceiling": "mask-envelope policy only. It authorizes no "
                         "post-window recertification and relaxes no "
                         "Stage-3 rule; prestart consumes the "
                         "comparator verdict. Lambda_geo INCONCLUSIVE"}


def load_policy(repo, manifest_commit):
    """Resolve the policy from its MANIFEST PIN. A caller-supplied
    dict with the same words is not authority (codex item 2)."""
    man_b = _git_blob(repo, manifest_commit, MANIFEST_REL)
    if man_b is None:
        raise BindComparatorRefusal(
            "MASK_POLICY_UNRESOLVED: manifest unreadable at "
            f"{str(manifest_commit)[:12]}")
    man = json.loads(man_b.decode("utf-8"))
    pin = None
    for slot in man["slots"].values():
        if slot["status"] != "BOUND":
            continue
        for cand in slot["pins"]:
            if cand["path"] == POLICY_REL:
                pin = cand
    if pin is None:
        raise BindComparatorRefusal(
            f"MASK_POLICY_NOT_PINNED: {POLICY_REL} is not a BOUND pin "
            f"at {str(manifest_commit)[:12]} -- the bind path takes "
            "its envelope from an admitted artifact, never a caller")
    body = _git_blob(repo, pin["commit"], pin["path"])
    if body is None or \
            hashlib.sha256(body).hexdigest() != pin["blob_sha256"]:
        raise BindComparatorRefusal(
            "MASK_POLICY_DIVERGENT: pinned policy bytes unreadable or "
            "divergent from the pin")
    policy = json.loads(body.decode("utf-8"))

    # codex cycle-6b item 1 (CRITICAL): checking only the outer
    # schema after the pin meant a self-consistently pinned body
    # carrying just {"schema": ..., "admitted_relations":
    # ["REALIZED_SUPERSET"]} was ACCEPTED, and a realized superset
    # then produced MATCH. A pin proves who wrote the bytes, never
    # that they say the registered thing. The one admissible policy
    # is therefore RECONSTRUCTED from the manifest-pinned calendar,
    # inputs bundle and comparator blob, and the pinned body must
    # equal it canonically.
    def _pinned(rel, what):
        p = None
        for slot in man["slots"].values():
            if slot["status"] != "BOUND":
                continue
            for cand in slot["pins"]:
                if cand["path"] == rel:
                    p = cand
        if p is None:
            raise BindComparatorRefusal(
                f"MASK_POLICY_UNRECONSTRUCTABLE: {what} ({rel}) is "
                "not a BOUND pin, so the admissible policy cannot be "
                "independently rebuilt")
        b = _git_blob(repo, p["commit"], p["path"])
        if b is None or \
                hashlib.sha256(b).hexdigest() != p["blob_sha256"]:
            raise BindComparatorRefusal(
                f"MASK_POLICY_UNRECONSTRUCTABLE: {what} bytes "
                "unreadable or divergent from their pin")
        return p, b

    cal_pin, _cal_b = _pinned(CALENDAR_V4_REL, "calendar authority")
    bun_pin, _bun_b = _pinned(INPUTS_BUNDLE_REL, "inputs bundle")
    cmp_pin, cmp_b = _pinned(
        "monitoring/src/w2_geometry_bind_comparator_cayley.py",
        "comparator")
    # rebuild from the SAME commits the pins name -- never HEAD, and
    # never this worktree's copy of the comparator
    want = build_policy(repo, manifest_commit,
                        comparator_bytes=cmp_b,
                        calendar_bytes=_cal_b,
                        bundle_bytes=_bun_b)
    if json.dumps(policy, sort_keys=True, separators=(",", ":")) != \
            json.dumps(want, sort_keys=True, separators=(",", ":")):
        diff = sorted(set(policy) ^ set(want)) or [
            k for k in sorted(set(policy) & set(want))
            if policy[k] != want[k]]
        raise BindComparatorRefusal(
            "MASK_POLICY_NOT_THE_REGISTERED_POLICY: the pinned body "
            "does not equal the policy independently reconstructed "
            f"from the admitted pins (diverging keys: {diff}) -- a "
            "newly pinned permissive body may not elect a different "
            "rule")
    if list(policy.get("admitted_relations") or []) != \
            list(ADMITTED_RELATIONS):
        raise BindComparatorRefusal(
            "MASK_POLICY_NOT_EXACT_ONLY: admitted_relations "
            f"{policy.get('admitted_relations')!r} is not the ruled "
            f"{list(ADMITTED_RELATIONS)}")
    return policy


def check_mask_wellformed(days, frame):
    """Registered well-formedness, independent of the relation: the
    cases codex named as impossible-on-a-maximal-anticipation."""
    out = []
    seq = [str(d) for d in days]
    if len(set(seq)) != len(seq):
        out.append("duplicate days")
    grid = set(frame["engine_days"])
    if [d for d in seq if d not in grid]:
        out.append("days outside the engine grid")
    exc = set(frame["excluded_days"])
    if [d for d in seq if d in exc]:
        out.append("an excluded PRESTART day")
    return out


def compare(anticipated, realized, *, mask_envelope_policy=None,
            calendar_frame=None):
    """THE bind comparator. `anticipated` and `realized` are each
    {"registries": {...}, "segments": {...}, "masks": {...}}.

    Verdict is REFUSE unless every registry and every segment map is
    EXACTLY equal AND every mask relation is admitted by the
    registered envelope policy. Every refusal is typed; nothing is
    silently tolerated.
    """
    for name, side in (("anticipated", anticipated),
                       ("realized", realized)):
        if not isinstance(side, dict) or \
                set(side) != {"registries", "segments", "masks"}:
            raise BindComparatorRefusal(
                f"GEOMETRY_BIND_INPUT_INVALID: {name} side must carry "
                "exactly registries, segments and masks")
    reasons = []
    regs = compare_registries(anticipated["registries"],
                              realized["registries"])
    segs = compare_segment_maps(anticipated["segments"],
                                realized["segments"])
    msks = compare_masks(anticipated["masks"], realized["masks"])

    for ck, d in regs.items():
        if d["exact"]:
            continue
        if d["order_only_divergence"]:
            reasons.append({
                "code": "GEOMETRY_BIND_REGISTRY_ORDER_DIVERGENT",
                "carrier": ck,
                "detail": "identical station SET in a different "
                          "ORDER -- the replicate RNG indexes station "
                          "rows by registry order, so equal counts "
                          "are not equivalence"})
        else:
            reasons.append({
                "code": "GEOMETRY_BIND_REGISTRY_DIVERGENT",
                "carrier": ck,
                "detail": f"+{len(d['added'])} / -{len(d['removed'])} "
                          "stations; station identity feeds the LOCO "
                          "seed token, so this changes null substreams"})
    for ck, d in segs.items():
        if d["exact"]:
            continue
        reasons.append({
            "code": "GEOMETRY_BIND_SEGMENT_MAP_DIVERGENT",
            "carrier": ck,
            "detail": f"{len(d['moved'])} moved, +{len(d['added'])} / "
                      f"-{len(d['removed'])} stations; a segment "
                      "change alters the graph geometry"})

    policy = mask_envelope_policy
    if policy is None:
        reasons.append({
            "code": "GEOMETRY_BIND_MASK_ENVELOPE_UNREGISTERED",
            "carrier": None,
            "detail": "no mask-envelope policy is bound, so no "
                      "realized mask can be said to fall INSIDE the "
                      "registered envelope; fail closed to the typed "
                      "no-run rather than elect a rule here"})
    else:
        admitted = set(policy.get("admitted_relations") or ())
        if not admitted <= set(MASK_RELATIONS):
            raise BindComparatorRefusal(
                "GEOMETRY_BIND_POLICY_INVALID: admitted relations "
                f"{sorted(admitted - set(MASK_RELATIONS))} are not "
                "registered relation names")
        for ck, d in msks.items():
            if d["relation"] not in admitted:
                reasons.append({
                    "code": "GEOMETRY_BIND_MASK_OUTSIDE_ENVELOPE",
                    "carrier": ck,
                    "detail": f"relation {d['relation']} is not in "
                              f"the registered envelope "
                              f"{sorted(admitted)}; +"
                              f"{len(d['days_gained'])} / -"
                              f"{len(d['days_lost'])} days"})
        # well-formedness is checked whenever the frame is supplied,
        # independent of the relation: on a MAXIMAL anticipation these
        # are the only ways a mask can look like a "superset"
        if calendar_frame is not None:
            for ck in sorted(realized["masks"]):
                bad = check_mask_wellformed(realized["masks"][ck],
                                            calendar_frame)
                if bad:
                    reasons.append({
                        "code": "GEOMETRY_BIND_MASK_MALFORMED",
                        "carrier": ck,
                        "detail": "; ".join(bad)})
    return {
        "schema": SCHEMA,
        "verdict": "MATCH" if not reasons else "REFUSE",
        "typed_reasons": reasons,
        "registries": regs,
        "segment_maps": segs,
        "masks": msks,
        "mask_envelope_policy": (dict(policy) if policy is not None
                                 else None),
        "authorizes": "NOTHING by itself: prestart consumes this "
                      "verdict. A MATCH does not certify, admit, or "
                      "open any value; a REFUSE makes prestart refuse",
        "claim_ceiling": "geometry comparison only; Lambda_geo "
                         "INCONCLUSIVE"}


def _selftest():
    base = {
        "registries": {"cascadia": ["CC.A", "CC.B", "CN.C"],
                       "istanbul_marmara": ["ADVT", "BOTS"]},
        "segments": {"cascadia": {"CC.A": "puget_sound",
                                  "CC.B": "puget_sound",
                                  "CN.C": "vancouver_island"},
                     "istanbul_marmara": {"ADVT": "izmit",
                                          "BOTS": "marmara_west"}},
        "masks": {"cascadia": ["2026-09-04", "2026-09-05"],
                  "istanbul_marmara": ["2026-09-04"]}}
    import copy
    POLICY_EXACT = {"admitted_relations": ["EXACT"],
                    "registered_by": "KAT-ONLY fixture policy"}

    # (+) identical geometry under an exact-only policy MATCHES
    r = compare(base, copy.deepcopy(base),
                mask_envelope_policy=POLICY_EXACT)
    assert r["verdict"] == "MATCH", r["typed_reasons"]
    print("  C0 PASS  identical geometry MATCHES (the comparator is "
          "not an always-refuse)")

    def refuses(mut, code, why):
        alt = copy.deepcopy(base)
        mut(alt)
        res = compare(base, alt, mask_envelope_policy=POLICY_EXACT)
        codes = [x["code"] for x in res["typed_reasons"]]
        if res["verdict"] != "REFUSE" or code not in codes:
            raise SystemExit(
                f"BIND_CONTROL_FAILED ({why}): verdict="
                f"{res['verdict']} codes={codes}")
        return res

    # (-) a REORDERED registry: identical set, identical counts
    def _reorder(a):
        a["registries"]["cascadia"] = ["CC.B", "CC.A", "CN.C"]
    res = refuses(_reorder, "GEOMETRY_BIND_REGISTRY_ORDER_DIVERGENT",
                  "reordered registry")
    d = res["registries"]["cascadia"]
    assert d["anticipated_set_sha256"] == d["realized_set_sha256"]
    assert d["anticipated_ordered_sha256"] != \
        d["realized_ordered_sha256"]
    print("  C1 PASS  a REORDERED registry refuses -- set digests "
          "identical, ordered digests differ (equal counts are not "
          "equivalence)")

    refuses(lambda a: a["registries"]["cascadia"].append("CC.NEW"),
            "GEOMETRY_BIND_REGISTRY_DIVERGENT", "added station")
    refuses(lambda a: a["registries"]["cascadia"].pop(),
            "GEOMETRY_BIND_REGISTRY_DIVERGENT", "removed station")
    print("  C2 PASS  an added or removed station refuses")

    # (-) a MOVED station: every count identical, graph different
    def _move(a):
        a["segments"]["cascadia"]["CC.B"] = "olympic_peninsula"
    res = refuses(_move, "GEOMETRY_BIND_SEGMENT_MAP_DIVERGENT",
                  "moved station")
    mv = res["segment_maps"]["cascadia"]["moved"]
    assert mv == [{"station": "CC.B", "from": "puget_sound",
                   "to": "olympic_peninsula"}], mv
    print("  C3 PASS  a MOVED station refuses and the movement is "
          "reported station-by-station (counts unchanged)")

    # (-) mask relations against an exact-only policy
    res = refuses(lambda a: a["masks"]["cascadia"].append("2026-09-06"),
                  "GEOMETRY_BIND_MASK_OUTSIDE_ENVELOPE",
                  "realized superset")
    assert res["masks"]["cascadia"]["relation"] == "REALIZED_SUPERSET"
    res = refuses(lambda a: a["masks"]["cascadia"].pop(),
                  "GEOMETRY_BIND_MASK_OUTSIDE_ENVELOPE",
                  "realized subset")
    assert res["masks"]["cascadia"]["relation"] == "REALIZED_SUBSET"

    def _diverge(a):
        a["masks"]["cascadia"] = ["2026-09-04", "2026-09-07"]
    res = refuses(_diverge, "GEOMETRY_BIND_MASK_OUTSIDE_ENVELOPE",
                  "divergent mask")
    assert res["masks"]["cascadia"]["relation"] == "DIVERGENT"
    print("  C4 PASS  superset / subset / divergent masks are "
          "classified exactly and refuse under an exact-only policy")

    # a policy admitting supersets lets exactly that one through and
    # nothing else -- so the policy is a real gate, not decoration
    alt = copy.deepcopy(base)
    alt["masks"]["cascadia"].append("2026-09-06")
    ok = compare(base, alt, mask_envelope_policy={
        "admitted_relations": ["EXACT", "REALIZED_SUPERSET"]})
    if ok["verdict"] != "MATCH":
        raise SystemExit(f"POLICY_INERT: {ok['typed_reasons']}")
    alt2 = copy.deepcopy(base)
    alt2["masks"]["cascadia"].pop()
    bad = compare(base, alt2, mask_envelope_policy={
        "admitted_relations": ["EXACT", "REALIZED_SUPERSET"]})
    assert bad["verdict"] == "REFUSE"
    print("  C5 PASS  the envelope policy is a real gate: a "
          "superset-admitting policy passes a superset and still "
          "refuses a subset")

    # NO policy bound -> typed refusal, never a silent election
    res = compare(base, copy.deepcopy(base))
    codes = [x["code"] for x in res["typed_reasons"]]
    assert res["verdict"] == "REFUSE" and \
        "GEOMETRY_BIND_MASK_ENVELOPE_UNREGISTERED" in codes, codes
    print("  C6 PASS  with NO registered mask envelope the comparator "
          "refuses typed even on identical geometry (fail closed; the "
          "semantic is routed to codex, not elected here)")

    # a malformed policy is a hard refusal, not a finding
    try:
        compare(base, copy.deepcopy(base),
                mask_envelope_policy={"admitted_relations": ["NOPE"]})
        raise SystemExit("malformed policy must raise")
    except BindComparatorRefusal as ex:
        assert "POLICY_INVALID" in str(ex), str(ex)
    print("  C7 PASS  a malformed envelope policy raises rather than "
          "reporting a finding")

    # ---- C8..C11: the EXACT-ONLY policy, resolved from its PIN ----
    _here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(_here, "..", ".."))
    try:
        pol = load_policy(repo, "HEAD")
        pinned = True
    except BindComparatorRefusal as e:
        if "NOT_PINNED" not in str(e):
            raise
        pol, pinned = build_policy(repo, "HEAD"), False
    if list(pol["admitted_relations"]) != list(ADMITTED_RELATIONS):
        raise SystemExit(
            f"C8 POLICY_NOT_EXACT_ONLY: {pol['admitted_relations']}")
    print(f"  C8 PASS  the registered policy admits EXACTLY "
          f"{list(ADMITTED_RELATIONS)} "
          f"({'resolved from its manifest pin' if pinned else 'built; pin lands with this packet'})")

    r = compare(base, copy.deepcopy(base), mask_envelope_policy=pol)
    if r["verdict"] != "MATCH":
        raise SystemExit(f"C9 EXACT_REFUSED: {r['typed_reasons']}")
    for mut, why in (
            (lambda a: a["masks"]["cascadia"].append("2026-09-06"),
             "superset"),
            (lambda a: a["masks"]["cascadia"].pop(), "subset"),
            (lambda a: a["masks"].__setitem__(
                "cascadia", ["2026-09-04", "2026-09-07"]),
             "divergent")):
        alt = copy.deepcopy(base)
        mut(alt)
        res = compare(base, alt, mask_envelope_policy=pol)
        if res["verdict"] != "REFUSE":
            raise SystemExit(f"C9 {why.upper()}_ADMITTED under the "
                             "exact-only policy")
    print("  C9 PASS  under the registered policy EXACT passes and "
          "superset / subset / divergent all refuse")

    # well-formedness: duplicate, out-of-grid and excluded days
    frame = {"engine_days": ["2026-09-04", "2026-09-05"],
             "excluded_days": ["2026-09-03"]}
    for days, why in ((["2026-09-04", "2026-09-04"], "duplicate"),
                      (["2026-09-04", "2027-06-01"], "out-of-grid"),
                      (["2026-09-04", "2026-09-03"], "excluded")):
        alt = copy.deepcopy(base)
        alt["masks"]["cascadia"] = days
        res = compare(base, alt, mask_envelope_policy=pol,
                      calendar_frame=frame)
        codes = [x["code"] for x in res["typed_reasons"]]
        if "GEOMETRY_BIND_MASK_MALFORMED" not in codes:
            raise SystemExit(
                f"C10 {why.upper()}_NOT_REFUSED: {codes}")
    ok = compare(base, copy.deepcopy(base), mask_envelope_policy=pol,
                 calendar_frame={"engine_days": base["masks"][
                     "cascadia"] + base["masks"]["istanbul_marmara"],
                     "excluded_days": ["2026-09-03"]})
    if ok["verdict"] != "MATCH":
        raise SystemExit(
            f"C10 WELLFORMED_CONTROL_INERT: a well-formed mask "
            f"refused: {ok['typed_reasons']}")
    print("  C10 PASS  duplicate / out-of-grid / excluded-day masks "
          "refuse typed, and a well-formed mask still passes (the "
          "check is not an always-refuse)")

    res = compare(base, copy.deepcopy(base))
    assert res["verdict"] == "REFUSE"
    print("  C11 PASS  the no-policy refusal is RETAINED as the "
          "negative control (an unbound envelope never admits)")

    # ---- C12 (codex cycle-6b item 1): a SELF-CONSISTENTLY PINNED
    # permissive body must refuse in load_policy(), before compare()
    # is ever reached. This is codex's exact probe: the pin digest
    # matches the bytes, so the pin is honest -- what is dishonest is
    # the CONTENT, and only reconstruction from the admitted inputs
    # can tell the difference.
    if pinned:
        import subprocess as _sp
        import tempfile as _tf

        def _pin_body(obj):
            """Write a body into the object store, pin it in a
            synthesized manifest, and try to load it."""
            raw = json.dumps(obj, indent=1,
                             sort_keys=True).encode() + b"\n"
            oid = _sp.run(["git", "-C", repo, "hash-object", "-w",
                           "--stdin"], input=raw,
                          capture_output=True).stdout.decode().strip()
            man_b = _git_blob(repo, "HEAD", MANIFEST_REL)
            man_o = json.loads(man_b.decode("utf-8"))
            fd, idxf = _tf.mkstemp(prefix="c12-index-")
            os.close(fd)
            os.unlink(idxf)
            env = dict(os.environ, GIT_INDEX_FILE=idxf)

            def g(args, data=None):
                return _sp.run(["git", "-C", repo] + args, input=data,
                               capture_output=True,
                               env=env).stdout.decode().strip()
            try:
                g(["read-tree", "HEAD"])
                g(["update-index", "--cacheinfo",
                   f"100644,{oid},{POLICY_REL}"])
                tree = g(["write-tree"])
                cmt = g(["-c", "user.name=c12", "-c",
                         "user.email=c12@local", "commit-tree", tree,
                         "-p", "HEAD", "-m", "C12 probe"])
            finally:
                try:
                    os.unlink(idxf)
                except OSError:
                    pass
            for slot in man_o["slots"].values():
                if slot["status"] != "BOUND":
                    continue
                for cand in slot["pins"]:
                    if cand["path"] == POLICY_REL:
                        cand["commit"] = cmt
                        cand["blob_sha256"] = hashlib.sha256(
                            raw).hexdigest()
            # the probe's manifest must itself be reachable, so pin
            # it into a second synthesized commit
            man_raw = json.dumps(man_o, indent=1,
                                 sort_keys=True).encode() + b"\n"
            moid = _sp.run(["git", "-C", repo, "hash-object", "-w",
                            "--stdin"], input=man_raw,
                           capture_output=True).stdout.decode().strip()
            fd, idxf = _tf.mkstemp(prefix="c12-index2-")
            os.close(fd)
            os.unlink(idxf)
            env = dict(os.environ, GIT_INDEX_FILE=idxf)
            try:
                g(["read-tree", cmt])
                g(["update-index", "--cacheinfo",
                   f"100644,{moid},{MANIFEST_REL}"])
                tree2 = g(["write-tree"])
                cmt2 = g(["-c", "user.name=c12", "-c",
                          "user.email=c12@local", "commit-tree",
                          tree2, "-p", cmt, "-m", "C12 probe man"])
            finally:
                try:
                    os.unlink(idxf)
                except OSError:
                    pass
            return cmt2

        for rel_ in ("REALIZED_SUPERSET", "REALIZED_SUBSET",
                     "DIVERGENT"):
            c_ = _pin_body({"schema": POLICY_SCHEMA,
                            "admitted_relations": [rel_]})
            try:
                load_policy(repo, c_)
                raise SystemExit(
                    f"C12 PERMISSIVE_POLICY_ADMITTED ({rel_}): a "
                    "self-consistently pinned permissive body loaded")
            except BindComparatorRefusal as e:
                if "NOT_THE_REGISTERED_POLICY" not in str(e) and \
                        "NOT_EXACT_ONLY" not in str(e):
                    raise
        # a doctored BINDING field (right relations, wrong digests)
        doctored = json.loads(json.dumps(pol))
        doctored["anticipated_masks"]["per_carrier_sha256"] = {
            k: "0" * 64 for k in
            doctored["anticipated_masks"]["per_carrier_sha256"]}
        c_ = _pin_body(doctored)
        try:
            load_policy(repo, c_)
            raise SystemExit(
                "C12 DOCTORED_BINDING_ADMITTED: a policy with the "
                "right relations but wrong mask digests loaded")
        except BindComparatorRefusal as e:
            assert "NOT_THE_REGISTERED_POLICY" in str(e), str(e)
        print("  C12 PASS  a SELF-CONSISTENTLY PINNED permissive body "
              "(superset / subset / divergent) and a doctored-binding "
              "policy all REFUSE in load_policy() before compare() is "
              "reached -- the policy is reconstructed from the "
              "admitted pins, not trusted because it is pinned")
    else:
        raise SystemExit(
            "C12 NOT_EXERCISED: the policy is not yet pinned, so the "
            "reconstruction path is untested; pin it before claiming "
            "item 1 is repaired")
    print("w2_geometry_bind_comparator selftest: ALL PASS")


def main():
    _here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(_here, "..", ".."))
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    commit = args[0] if args else "HEAD"
    body = json.dumps(build_policy(repo, commit),
                      indent=1, sort_keys=True) + "\n"
    out = os.path.join(repo, POLICY_REL.replace("/", os.sep))
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
    print(f"wrote {POLICY_REL}")
    print("policy sha256:", hashlib.sha256(body.encode()).hexdigest())


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    elif "--emit-policy" in sys.argv:
        main()
    else:
        raise SystemExit(
            "GEOMETRY_BIND_COMPARATOR: library, --selftest, or "
            "--emit-policy; the 2026-09-02 bind invokes compare() "
            "with the bound capsule, the realized geometry, and the "
            "policy resolved from its manifest pin")
