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
import sys

SCHEMA = "f2g-w2-geometry-bind-comparison-v1"
MASK_RELATIONS = ("EXACT", "REALIZED_SUPERSET", "REALIZED_SUBSET",
                  "DIVERGENT")


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


def compare(anticipated, realized, *, mask_envelope_policy=None):
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
    print("w2_geometry_bind_comparator selftest: ALL PASS")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        raise SystemExit(
            "GEOMETRY_BIND_COMPARATOR: library + --selftest only; the "
            "2026-09-02 bind invokes compare() with the bound capsule "
            "and the realized geometry")
