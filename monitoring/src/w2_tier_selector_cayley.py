#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 TIER-C CANDIDATE SELECTOR (cayley) -- codex 1909Z item 1
BINDING RULE, the narrow window-2 mapping of the REGISTERED two-stage
selector (phaseb_annex_common rev-1.6 sec 3, design pin @ feb20bb,
blob 44f8ddd9...): full-grid Tier-S -> top-8 pre-screen -> post-LOCO
ranking (B1B only under window-2) -> top-3 selection.

The rule, verbatim from the ruling:
1. Tier-S runs every registered detection point at R=50, n_draws=999,
   through the full four-member Holm vector; a pre-LOCO recovery is
   strictly `family in holm_rejects(vector)`.
2. Rank by (-integer_success_count, registered_grid_index);
   denominators must be exactly 50 and outcomes strict booleans.
   Keep min(8, grid_size).
3. B1B only: apply the registered exact partial-recompute LOCO rule
   to those eight at the same Tier-S quality; rank by
   (-post_loco_success_count, registered_grid_index); select three.
   B2A/B2B/B3A select the top three pre-LOCO (a no-op stage 2);
   B2A's three-point grid selects all three.
4. Append the two fixed B1B specificity obligations {gain:3} then
   {gain:10} -- they are never smoke-selected.
5. Campaign order: B2A top-3, B2B top-3, B1B detection top-3, B3A
   top-3, then the two gain points; within each top-3, selector rank
   order.

Tier-S remains PRELIMINARY_SMOKE; selection certifies nothing. The
selector is PURE and deterministic: rule + smoke output -> the same
ordered 14-point list for anyone. Registered text:
docs/f2g_window2_execution/tier_selector_amendment_w2_v1.md.
This module opens no window-2 value.
"""
import hashlib
import json

TIER_S_R = 50
TIER_S_DRAWS = 999
PRESCREEN_KEEP = 8
SELECT_N = 3
FAMILIES_ORDER = ("B2A", "B2B", "B1B", "B3A")
SELECTOR_SCHEMA = "f2g-w2-tier-selector-v1"


class SelectorRefusal(ValueError):
    pass


def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _digest(obj):
    return hashlib.sha256(_canon(obj).encode()).hexdigest()


def _count(outcomes, where):
    if not isinstance(outcomes, list) or len(outcomes) != TIER_S_R:
        raise SelectorRefusal(
            f"SELECTOR_QUALITY_INVALID: {where} denominator "
            f"{len(outcomes) if isinstance(outcomes, list) else '?'} "
            f"!= {TIER_S_R}")
    for o in outcomes:
        if not isinstance(o, bool):
            raise SelectorRefusal(
                f"SELECTOR_QUALITY_INVALID: {where} non-strict "
                f"outcome {o!r}")
    return sum(1 for o in outcomes if o)


def _detection_grid(effect_grid):
    """Registered detection points with their REGISTERED grid indices
    (index in the frozen effect-grid list); gain points excluded."""
    return [(i, p) for i, p in enumerate(effect_grid)
            if "gain" not in p]


def _gain_points(effect_grid):
    """codex 2235Z item 2: the two obligations are the REGISTERED
    {gain: 3} and {gain: 10} exactly -- any other gain values
    refuse."""
    g = [(i, p) for i, p in enumerate(effect_grid) if "gain" in p]
    if len(g) != 2:
        raise SelectorRefusal(
            f"SELECTOR_GRID_INVALID: B1B registers exactly two gain "
            f"points, found {len(g)}")
    out = [p for _, p in sorted(g, key=lambda t: float(
        t[1]["gain"]))]
    if [float(p["gain"]) for p in out] != [3.0, 10.0] or \
            any(set(p) != {"gain"} for p in out):
        raise SelectorRefusal(
            "SELECTOR_GRID_INVALID: specificity obligations must be "
            "exactly {gain: 3} then {gain: 10}")
    return out


def select_candidates(smoke, effect_grids, *, smoke_ref=None,
                      effect_grids_ref=None):
    """smoke = the Tier-S artifact: {"quality": {"R": 50, "n_draws":
    999}, "families": {fam: [{"point", "outcomes"[,
    "post_loco_outcomes"]}, ...]}} covering EVERY registered detection
    point in REGISTERED GRID ORDER; B1B entries carry
    post_loco_outcomes for EXACTLY the stage-1 top-8 (None elsewhere).
    smoke_ref / effect_grids_ref = {commit, path} git references to
    the COMMITTED carriers (codex 2235Z item 2: required for a
    verifiable production artifact; fixture-only runs may omit them).
    Returns the selector artifact with the ordered 14-point list."""
    if not isinstance(smoke, dict) or \
            smoke.get("quality") != {"R": TIER_S_R,
                                     "n_draws": TIER_S_DRAWS}:
        raise SelectorRefusal(
            "SELECTOR_QUALITY_INVALID: smoke quality is not "
            f"{{R: {TIER_S_R}, n_draws: {TIER_S_DRAWS}}}")
    fams = smoke.get("families")
    if not isinstance(fams, dict) or \
            set(fams) != set(FAMILIES_ORDER):
        raise SelectorRefusal(
            "SELECTOR_COVERAGE_INVALID: family set not the "
            "registered four")
    per_point = {}
    top8 = {}
    selected = {}
    for fam in FAMILIES_ORDER:
        grid = effect_grids.get(fam)
        if not isinstance(grid, list) or not grid:
            raise SelectorRefusal(
                f"SELECTOR_GRID_INVALID: {fam} effect grid absent")
        det = _detection_grid(grid)
        entries = fams[fam]
        if not isinstance(entries, list) or \
                len(entries) != len(det):
            raise SelectorRefusal(
                f"SELECTOR_COVERAGE_INVALID: {fam} has "
                f"{len(entries) if isinstance(entries, list) else '?'}"
                f" entries != {len(det)} registered detection points")
        counts = []
        for k, e in enumerate(entries):
            gi, gp = det[k]
            if not isinstance(e, dict) or e.get("point") != gp:
                raise SelectorRefusal(
                    f"SELECTOR_COVERAGE_INVALID: {fam} entry {k} is "
                    "not the registered grid point in registered "
                    "order")
            c = _count(e.get("outcomes"), f"{fam}[{k}]")
            counts.append({"grid_index": gi, "point": gp,
                           "pre_loco_count": c})
        order1 = sorted(range(len(counts)), key=lambda k: (
            -counts[k]["pre_loco_count"], counts[k]["grid_index"]))
        keep = order1[:min(PRESCREEN_KEEP, len(counts))]
        keep_set = set(keep)
        if fam == "B1B":
            for k, e in enumerate(entries):
                pl = e.get("post_loco_outcomes")
                if k in keep_set:
                    if pl is None:
                        raise SelectorRefusal(
                            f"SELECTOR_STAGE2_INVALID: B1B top-8 "
                            f"entry {k} lacks post-LOCO outcomes")
                    counts[k]["post_loco_count"] = _count(
                        pl, f"B1B[{k}].post_loco")
                elif pl is not None:
                    raise SelectorRefusal(
                        f"SELECTOR_STAGE2_INVALID: B1B entry {k} "
                        "carries post-LOCO outcomes outside the "
                        "stage-1 top-8")
            order2 = sorted(keep, key=lambda k: (
                -counts[k]["post_loco_count"],
                counts[k]["grid_index"]))
            pick = order2[:SELECT_N]
        else:
            pick = keep[:SELECT_N]
        if len(pick) < SELECT_N and len(counts) >= SELECT_N:
            raise SelectorRefusal(
                f"SELECTOR_STAGE2_INVALID: {fam} selected "
                f"{len(pick)} < {SELECT_N}")
        per_point[fam] = counts
        top8[fam] = [counts[k]["grid_index"] for k in keep]
        selected[fam] = [counts[k]["grid_index"] for k in pick]
    ordered = []
    for fam in FAMILIES_ORDER:
        det = dict(_detection_grid(effect_grids[fam]))
        for gi in selected[fam]:
            ordered.append({"family": fam, "point": det[gi],
                            "entry": "detection"})
    for gp in _gain_points(effect_grids["B1B"]):
        ordered.append({"family": "B1B", "point": gp,
                        "entry": "specificity"})
    art = {"schema": SELECTOR_SCHEMA,
           "rule": "codex 1909Z item 1 binding rule (annex-common "
                   "sec-3 selector mapped to window-2 full-four-Holm "
                   "+ B1B-only LOCO)",
           "tier_s_label": "PRELIMINARY_SMOKE (selection certifies "
                           "nothing)",
           "quality": {"R": TIER_S_R, "n_draws": TIER_S_DRAWS},
           "effect_grids_sha256": _digest(effect_grids),
           "smoke_sha256": _digest(smoke),
           "geometry_capsule_digest":
               smoke.get("geometry_capsule_digest"),
           "per_point": per_point,
           "top8_grid_indices": top8,
           "selected_grid_indices": selected,
           "ordered_points": ordered,
           "ordered_points_sha256": _digest(ordered),
           "smoke_ref": dict(smoke_ref) if smoke_ref else None,
           "effect_grids_ref": (dict(effect_grids_ref)
                                if effect_grids_ref else None)}
    if len(ordered) != 4 * SELECT_N + 2:
        raise SelectorRefusal(
            f"SELECTOR_STAGE2_INVALID: ordered list has "
            f"{len(ordered)} points != {4 * SELECT_N + 2}")
    return art


EXPECTED_CAMPAIGN_FAMILIES = (["B2A"] * SELECT_N + ["B2B"] * SELECT_N
                              + ["B1B"] * SELECT_N
                              + ["B3A"] * SELECT_N + ["B1B"] * 2)


def verify_selector_artifact(repo, art, *, blob_reader=None):
    """codex 2235Z item 2: a self-consistent ordered-point digest is
    integrity, not selector correctness. This verifier reopens the
    BOUND smoke and effect-grids carriers from their committed git
    objects, independently reruns select_candidates, and requires
    CANONICAL EQUALITY of the full artifact; it enforces the exact
    14-point family/order/entry shape and the registered gains
    3 then 10. A fabricated artifact (missing bindings, uncommitted
    carriers, altered points) refuses typed."""
    if not isinstance(art, dict) or \
            art.get("schema") != SELECTOR_SCHEMA:
        raise SelectorRefusal(
            "SELECTOR_ARTIFACT_INVALID: not a selector artifact")
    for ref_name in ("smoke_ref", "effect_grids_ref"):
        r = art.get(ref_name)
        if not isinstance(r, dict) or set(r) != {"commit", "path"}:
            raise SelectorRefusal(
                f"SELECTOR_ARTIFACT_INVALID: {ref_name} is not a "
                "closed {commit, path} git reference")
    if blob_reader is None:
        import subprocess

        def blob_reader(commit, path):
            p = subprocess.run(
                ["git", "-C", repo, "cat-file", "blob",
                 f"{commit}:{path}"], capture_output=True)
            if p.returncode != 0:
                raise SelectorRefusal(
                    f"SELECTOR_ARTIFACT_INVALID: {path} unreadable "
                    f"at {commit} (uncommitted carriers never bind)")
            return p.stdout
    smoke = json.loads(blob_reader(
        art["smoke_ref"]["commit"],
        art["smoke_ref"]["path"]).decode("utf-8"))
    grids_art = json.loads(blob_reader(
        art["effect_grids_ref"]["commit"],
        art["effect_grids_ref"]["path"]).decode("utf-8"))
    grids = grids_art.get("grids", grids_art)
    rerun = select_candidates(
        smoke, grids, smoke_ref=art["smoke_ref"],
        effect_grids_ref=art["effect_grids_ref"])
    if _canon(rerun) != _canon(art):
        raise SelectorRefusal(
            "SELECTOR_ARTIFACT_INVALID: independent rerun diverges "
            "from the artifact (points/counts/sets are not the "
            "registered rule applied to the bound carriers)")
    op = art["ordered_points"]
    if [p["family"] for p in op] != EXPECTED_CAMPAIGN_FAMILIES or \
            [p["entry"] for p in op] != \
            ["detection"] * (4 * SELECT_N) + ["specificity"] * 2 or \
            [float(p["point"]["gain"]) for p in op[-2:]] != \
            [3.0, 10.0]:
        raise SelectorRefusal(
            "SELECTOR_ARTIFACT_INVALID: campaign shape is not the "
            "registered 14-point order with gains 3 then 10")
    return True


# ---------------------------------------------------------------- selftest
def _selftest():
    def outs(k):
        return [True] * k + [False] * (TIER_S_R - k)

    grids = {
        "B2A": [{"m": 1}, {"m": 2}, {"m": 3}],
        "B2B": [{"m": 1, "dropout": 0.0}, {"m": 2, "dropout": 0.0},
                {"m": 2, "dropout": 0.1}, {"m": 2, "dropout": 0.25},
                {"m": 3, "dropout": 0.0}],
        "B3A": [{"delta": d} for d in (1, 2, 3, 4)],
        "B1B": [{"delta_lat": 0.1 * j, "k": 3, "n_e": 2}
                for j in range(1, 10)]
               + [{"gain": 10.0}, {"gain": 3.0}]}   # gain order mixed

    def smoke_for(counts_by_fam, post_loco=None):
        fams = {}
        for fam in FAMILIES_ORDER:
            det = _detection_grid(grids[fam])
            entries = []
            for k, (gi, gp) in enumerate(det):
                e = {"point": dict(gp),
                     "outcomes": outs(counts_by_fam[fam][k])}
                if fam == "B1B":
                    e["post_loco_outcomes"] = (
                        outs(post_loco[k]) if post_loco and
                        post_loco.get(k) is not None else None)
                entries.append(e)
            fams[fam] = entries
        return {"quality": {"R": 50, "n_draws": 999},
                "geometry_capsule_digest": "kat-digest",
                "families": fams}

    # B1B: stage-1 keeps top 8 of 9; stage-2 post-LOCO REORDERS.
    # pre-LOCO counts rise with j; grid index tie-break checked via
    # equal counts at k=0,1
    b1b_pre = [40, 40, 41, 42, 43, 44, 45, 46, 47]
    keep_expected = [8, 7, 6, 5, 4, 3, 2, 0]     # -count, then index
    post = {8: 10, 7: 45, 6: 44, 5: 46, 4: 20, 3: 19, 2: 18, 0: 17}
    sm = smoke_for({"B2A": [50, 49, 48],
                    "B2B": [10, 30, 30, 20, 40],
                    "B3A": [5, 6, 7, 8],
                    "B1B": b1b_pre}, post)
    art = select_candidates(sm, grids)
    assert art["top8_grid_indices"]["B1B"] == keep_expected
    # post-LOCO ranking: counts 46(k=5),45(k=7),44(k=6) win
    assert art["selected_grid_indices"]["B1B"] == [5, 7, 6]
    # B2A three-point grid selects all three (registered order by rank)
    assert art["selected_grid_indices"]["B2A"] == [0, 1, 2]
    # B2B tie at 30/30 -> lower grid index first; top3 = idx4(40),
    # idx1(30), idx2(30)
    assert art["selected_grid_indices"]["B2B"] == [4, 1, 2]
    # campaign order + the two gain points appended ASCENDING (3, 10)
    op = art["ordered_points"]
    assert [p["family"] for p in op] == \
        ["B2A"] * 3 + ["B2B"] * 3 + ["B1B"] * 3 + ["B3A"] * 3 + \
        ["B1B"] * 2
    assert op[-2]["point"] == {"gain": 3.0} \
        and op[-1]["point"] == {"gain": 10.0}
    assert all(p["entry"] == "specificity" for p in op[-2:])
    assert len(op) == 14
    # determinism
    assert select_candidates(sm, grids)["ordered_points_sha256"] == \
        art["ordered_points_sha256"]

    # doctors
    import copy

    def refuses(mut_fn, code):
        s2 = copy.deepcopy(sm)
        mut_fn(s2)
        try:
            select_candidates(s2, grids)
            return False
        except SelectorRefusal as e:
            return code in str(e)
    assert refuses(lambda s: s["quality"].update(R=40),
                   "SELECTOR_QUALITY_INVALID")
    assert refuses(lambda s: s["families"]["B2A"][0]["outcomes"]
                   .append(True), "SELECTOR_QUALITY_INVALID")
    assert refuses(lambda s: s["families"]["B2A"][0]["outcomes"]
                   .__setitem__(0, 1), "SELECTOR_QUALITY_INVALID")
    assert refuses(lambda s: s["families"]["B2A"].pop(),
                   "SELECTOR_COVERAGE_INVALID")
    assert refuses(lambda s: s["families"]["B2A"][0].update(
        point={"m": 9}), "SELECTOR_COVERAGE_INVALID")
    assert refuses(lambda s: s["families"]["B1B"][8].update(
        post_loco_outcomes=None), "SELECTOR_STAGE2_INVALID")
    # post-LOCO outside the top-8 refuses (index 1 not kept)
    assert refuses(lambda s: s["families"]["B1B"][1].update(
        post_loco_outcomes=outs(5)), "SELECTOR_STAGE2_INVALID")
    # a gain grid with != 2 gain points refuses
    g2 = copy.deepcopy(grids)
    g2["B1B"].append({"gain": 5.0})
    try:
        select_candidates(sm, g2)
        raise AssertionError("three gain points must refuse")
    except SelectorRefusal as e:
        assert "SELECTOR_GRID_INVALID" in str(e)
    # ALTERED gain values refuse (codex 2235Z item 2: the registered
    # obligations are exactly 3 then 10)
    g3 = copy.deepcopy(grids)
    g3["B1B"][-2] = {"gain": 4.0}
    try:
        select_candidates(sm, g3)
        raise AssertionError("altered gain value must refuse")
    except SelectorRefusal as e:
        assert "exactly {gain: 3} then {gain: 10}" in str(e)

    # --- verify_selector_artifact (codex 2235Z item 2) ---
    store = {("c" * 40, "smoke.json"): json.dumps(sm).encode(),
             ("c" * 40, "grids.json"): json.dumps(
                 {"schema": "f2g-w2-effect-grids-v1",
                  "grids": grids}).encode()}

    def reader(commit, path):
        try:
            return store[(commit, path)]
        except KeyError:
            raise SelectorRefusal(
                f"SELECTOR_ARTIFACT_INVALID: {path} unreadable at "
                f"{commit} (uncommitted carriers never bind)")
    refs = {"smoke_ref": {"commit": "c" * 40, "path": "smoke.json"},
            "effect_grids_ref": {"commit": "c" * 40,
                                 "path": "grids.json"}}
    art_b = select_candidates(sm, grids, **refs)
    assert verify_selector_artifact(".", art_b, blob_reader=reader)
    # a FABRICATED minimal artifact refuses (no bindings)
    fab = {"schema": SELECTOR_SCHEMA,
           "ordered_points": [{"family": "B2A", "point": {"m": 999},
                               "entry": "detection"}],
           "ordered_points_sha256": _digest(
               [{"family": "B2A", "point": {"m": 999},
                 "entry": "detection"}])}
    try:
        verify_selector_artifact(".", fab, blob_reader=reader)
        raise AssertionError("fabricated artifact must refuse")
    except SelectorRefusal as e:
        assert "SELECTOR_ARTIFACT_INVALID" in str(e)
    # an ALTERED point refuses via independent rerun divergence
    tam = copy.deepcopy(art_b)
    tam["ordered_points"][0]["point"] = {"m": 999}
    tam["ordered_points_sha256"] = _digest(tam["ordered_points"])
    try:
        verify_selector_artifact(".", tam, blob_reader=reader)
        raise AssertionError("tampered points must refuse")
    except SelectorRefusal as e:
        assert "independent rerun diverges" in str(e)
    # an UNCOMMITTED carrier path refuses through the real reader
    bad_ref = copy.deepcopy(art_b)
    bad_ref["smoke_ref"] = {"commit": "c" * 40,
                            "path": "docs/no-such-smoke.json"}
    import os as _os
    repo_g = _os.path.abspath(_os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
    try:
        verify_selector_artifact(repo_g, bad_ref)
        raise AssertionError("uncommitted carrier must refuse")
    except SelectorRefusal as e:
        assert "unreadable" in str(e)

    print("w2_tier_selector selftest: ALL PASS (hand fixtures; "
          "PRELIMINARY_SMOKE semantics; nothing certified)")


if __name__ == "__main__":
    _selftest()
