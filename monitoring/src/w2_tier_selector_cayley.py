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
# codex 0320Z item 2: the CLOSED Tier-S invocation/result capsule
TIER_S_INVOCATION_SCHEMA = "f2g-w2-tier-s-invocation-v2"
TIER_S_INVOCATION_FIELDS = {
    "schema", "manifest_commit", "effect_grids", "geometry",
    "quality", "seed_authority_sha256", "implementation",
    "grid_order_sha256", "results_ref", "completion_receipt",
    "invocation_sha256"}
TIER_S_RESULTS_SCHEMA = "f2g-w2-tier-s-results-v1"


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


def verify_selector_admission(repo, art, manifest_commit, *,
                              blob_reader=None, git_resolve=None):
    """codex 0238Z item 3: Git-readable is not ADMITTED. Beyond
    verify_selector_artifact's independent rerun, this capsule
    requires: (a) the effect-grid ref's reopened blob to EQUAL the
    blob pinned in the execution manifest at `manifest_commit` for
    that exact path; (b) the smoke ref to be the CLOSED output of the
    admitted Tier-S invocation over the same grids -- schema + exact
    quality + effect-grids digest equality + a reopenable Tier-S
    invocation record whose core digest matches the smoke's recorded
    `invocation_sha256`; (c) every commit resolved to full 40-hex.
    Returns the admitted-identity block the campaign runner binds
    into ITS invocation digest. Three committed, internally
    consistent substitute carriers refuse here as UNADMITTED."""
    import subprocess

    def _resolve(commitish):
        if git_resolve is not None:
            return git_resolve(commitish)
        p = subprocess.run(
            ["git", "-C", repo, "rev-parse",
             f"{commitish}^{{commit}}"], capture_output=True)
        full = p.stdout.decode().strip()
        if p.returncode != 0 or len(full) != 40:
            raise SelectorRefusal(
                f"SELECTOR_UNADMITTED: commit {commitish!r} does not "
                "resolve to an admitted 40-hex lineage")
        return full
    if blob_reader is None:
        def blob_reader(commit, path):
            p = subprocess.run(
                ["git", "-C", repo, "cat-file", "blob",
                 f"{commit}:{path}"], capture_output=True)
            if p.returncode != 0:
                raise SelectorRefusal(
                    f"SELECTOR_UNADMITTED: {path} unreadable at "
                    f"{commit}")
            return p.stdout
    verify_selector_artifact(repo, art, blob_reader=blob_reader)
    mc_full = _resolve(manifest_commit)
    man = json.loads(blob_reader(
        mc_full, "docs/f2g_window2_execution/"
                 "execution_manifest.json").decode("utf-8"))
    # (a) the effect grids must BE the manifest-pinned blob
    g_ref = art["effect_grids_ref"]
    pinned = None
    for slot in man.get("slots", {}).values():
        for pin in slot.get("pins", ()) or ():
            if isinstance(pin, dict) and \
                    pin.get("path") == g_ref["path"]:
                pinned = pin
    if pinned is None:
        raise SelectorRefusal(
            f"SELECTOR_UNADMITTED: {g_ref['path']} is not a pin of "
            f"the execution manifest at {mc_full[:12]}")
    g_raw = blob_reader(_resolve(g_ref["commit"]), g_ref["path"])
    if hashlib.sha256(g_raw).hexdigest() != pinned["blob_sha256"]:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: effect-grid carrier diverges from "
            "the manifest-pinned blob (committed is not admitted)")
    # (b) the smoke must be the closed admitted Tier-S output
    s_ref = art["smoke_ref"]
    s_full = _resolve(s_ref["commit"])
    smoke = json.loads(blob_reader(
        s_full, s_ref["path"]).decode("utf-8"))
    if smoke.get("schema") != "f2g-w2-tier-s-smoke-v1":
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: smoke carrier is not the closed "
            "Tier-S output schema")
    if smoke.get("quality") != {"R": TIER_S_R,
                                "n_draws": TIER_S_DRAWS}:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: smoke quality is not the admitted "
            "Tier-S quality")
    grids_art = json.loads(g_raw.decode("utf-8"))
    grids = grids_art.get("grids", grids_art)
    if smoke.get("effect_grids_sha256") != _digest(grids):
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: smoke does not bind the admitted "
            "effect grids")
    inv_ref = smoke.get("invocation_ref")
    if not isinstance(inv_ref, dict) or \
            set(inv_ref) != {"commit", "path"}:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: smoke lacks a closed Tier-S "
            "invocation reference")
    inv = json.loads(blob_reader(_resolve(inv_ref["commit"]),
                                 inv_ref["path"]).decode("utf-8"))
    # codex 0320Z item 2: a self-hashed dict is NOT an invocation --
    # the CLOSED capsule schema is required, with admitted identities
    if not isinstance(inv, dict) or \
            set(inv) != TIER_S_INVOCATION_FIELDS or \
            inv.get("schema") != TIER_S_INVOCATION_SCHEMA:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: the Tier-S invocation is not the "
            "closed capsule schema (a self-hashed dict attests "
            "nothing)")
    core = {k: v for k, v in inv.items()
            if k != "invocation_sha256"}
    if hashlib.sha256(_canon(core).encode()).hexdigest() != \
            inv.get("invocation_sha256") or \
            inv.get("invocation_sha256") != \
            smoke.get("invocation_sha256"):
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: the smoke's Tier-S invocation "
            "digest does not recompute from the reopened record")
    if _resolve(inv["manifest_commit"]) != mc_full:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: the Tier-S invocation binds a "
            "different manifest lineage")
    ig = inv["effect_grids"]
    if not isinstance(ig, dict) or \
            ig.get("path") != g_ref["path"] or \
            ig.get("blob_sha256") != pinned["blob_sha256"]:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: the Tier-S invocation does not "
            "bind the admitted effect-grid identity")
    if inv.get("quality") != {"R": TIER_S_R,
                              "n_draws": TIER_S_DRAWS}:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: invocation quality is not the "
            "admitted Tier-S quality")
    geo = inv["geometry"]
    if not isinstance(geo, dict) or \
            set(geo) != {"commit", "path", "capsule_digest"}:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: invocation geometry identity not "
            "closed")
    if smoke.get("geometry_capsule_digest") != \
            geo["capsule_digest"]:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: smoke geometry diverges from the "
            "invocation's bound capsule")
    sa = inv["seed_authority_sha256"]
    if not (isinstance(sa, str) and len(sa) == 64):
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: seed authority untyped")
    impl = inv["implementation"]
    impl_pin = None
    for slot in man.get("slots", {}).values():
        for pin in slot.get("pins", ()) or ():
            if isinstance(pin, dict) and \
                    pin.get("path") == impl.get("path"):
                impl_pin = pin
    if impl_pin is None or \
            impl.get("blob_sha256") != impl_pin["blob_sha256"]:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: invocation implementation is not "
            "the manifest-pinned blob")
    det_order = {fam: [p for p in grids[fam] if "gain" not in p]
                 for fam in FAMILIES_ORDER}
    if inv["grid_order_sha256"] != _digest(det_order):
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: invocation grid order diverges "
            "from the admitted detection grids")
    cr = inv["completion_receipt"]
    if not isinstance(cr, dict) or \
            set(cr) != {"fired_utc", "completed_utc"}:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: completion receipt not closed")
    # reopen the per-point RESULTS and independently REBUILD the
    # smoke outcomes (incl B1B post-LOCO) -- fabricated smoke lists
    # cannot survive a results carrier they do not derive from
    r_ref = inv["results_ref"]
    if not isinstance(r_ref, dict) or \
            set(r_ref) != {"commit", "path", "blob_sha256"}:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: results binding not closed")
    r_raw = blob_reader(_resolve(r_ref["commit"]), r_ref["path"])
    if hashlib.sha256(r_raw).hexdigest() != r_ref["blob_sha256"]:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: results bytes diverge from the "
            "invocation's output binding")
    results = json.loads(r_raw.decode("utf-8"))
    if results.get("schema") != TIER_S_RESULTS_SCHEMA:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: results schema mismatch")
    for fam in FAMILIES_ORDER:
        rf = results.get("families", {}).get(fam)
        sf = smoke.get("families", {}).get(fam)
        if not isinstance(rf, list) or not isinstance(sf, list) or \
                len(rf) != len(sf):
            raise SelectorRefusal(
                f"SELECTOR_UNADMITTED: {fam} results/smoke shape "
                "mismatch")
        for k, (re_, se_) in enumerate(zip(rf, sf)):
            if re_.get("point") != se_.get("point"):
                raise SelectorRefusal(
                    f"SELECTOR_UNADMITTED: {fam}[{k}] point "
                    "mismatch between results and smoke")
            rebuilt = [fam in set(r) for r in re_["replicates"]]
            if rebuilt != se_.get("outcomes"):
                raise SelectorRefusal(
                    f"SELECTOR_UNADMITTED: {fam}[{k}] smoke "
                    "outcomes do not REBUILD from the results "
                    "carrier")
            if fam == "B1B" and re_.get("post_loco_replicates") != \
                    se_.get("post_loco_outcomes"):
                raise SelectorRefusal(
                    f"SELECTOR_UNADMITTED: B1B[{k}] post-LOCO "
                    "outcomes do not match the results carrier")
    return {"manifest_commit": mc_full,
            "effect_grids": {"commit": _resolve(g_ref["commit"]),
                             "path": g_ref["path"],
                             "blob_sha256":
                                 hashlib.sha256(g_raw).hexdigest()},
            "smoke": {"commit": s_full, "path": s_ref["path"],
                      "blob_sha256": hashlib.sha256(blob_reader(
                          s_full, s_ref["path"])).hexdigest(),
                      "invocation_sha256":
                          smoke.get("invocation_sha256")}}


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

    # --- verify_selector_admission (codex 0238Z item 3 + 0320Z
    # item 2: the closed Tier-S capsule) ---
    import hashlib as _hl
    _digest_fn = _digest

    def mk_tier_s_capsule(smoke_families, grids_obj, grids_raw,
                          store_map, commit, man_pins, impl_path,
                          geom_digest="kat-digest"):
        import hashlib as _h
        results = {"schema": "f2g-w2-tier-s-results-v1",
                   "families": {}}
        for fam, entries in smoke_families.items():
            results["families"][fam] = [
                {"point": dict(e["point"]),
                 "replicates": [[fam] if o else []
                                for o in e["outcomes"]],
                 "post_loco_replicates":
                     e.get("post_loco_outcomes")}
                for e in entries]
        r_raw = json.dumps(results).encode()
        store_map[(commit, "ts_results.json")] = r_raw
        det_order = {f: [p for p in grids_obj[f] if "gain" not in p]
                     for f in ("B2A", "B2B", "B1B", "B3A")}
        impl_pin = [p for p in man_pins
                    if p["path"] == impl_path][0]
        inv = {"schema": "f2g-w2-tier-s-invocation-v2",
               "manifest_commit": commit,
               "effect_grids": {"commit": commit,
                               "path": "grids.json"
                               if (commit, "grids.json") in store_map
                               else "grids2.json",
                               "blob_sha256": _h.sha256(
                                   grids_raw).hexdigest()},
               "geometry": {"commit": commit, "path": "geom.json",
                            "capsule_digest": geom_digest},
               "quality": {"R": 50, "n_draws": 999},
               "seed_authority_sha256": "b" * 64,
               "implementation": {"path": impl_path,
                                  "blob_sha256":
                                      impl_pin["blob_sha256"]},
               "grid_order_sha256": _digest_fn(det_order),
               "results_ref": {"commit": commit,
                               "path": "ts_results.json",
                               "blob_sha256": _h.sha256(
                                   r_raw).hexdigest()},
               "completion_receipt": {
                   "fired_utc": "2026-08-25T00:00:00Z",
                   "completed_utc": "2026-08-25T11:00:00Z"}}
        inv["invocation_sha256"] = _digest_fn(
            {k: v for k, v in inv.items()
             if k != "invocation_sha256"})
        store_map[(commit, "ts_invocation.json")] = json.dumps(
            inv).encode()
        return inv

    grids_raw = json.dumps({"schema": "f2g-w2-effect-grids-v1",
                            "grids": grids}).encode()
    astore = {("a" * 40, "grids2.json"): grids_raw,
              ("a" * 40, "impl.py"): b"# pinned impl"}
    man_pins = [
        {"path": "grids2.json", "commit": "a" * 40,
         "blob_sha256": _hl.sha256(grids_raw).hexdigest()},
        {"path": "impl.py", "commit": "a" * 40,
         "blob_sha256": _hl.sha256(b"# pinned impl").hexdigest()}]
    fix_man = {"slots": {"x": {"pins": man_pins}}}
    ts_inv = mk_tier_s_capsule(sm["families"], grids, grids_raw,
                               astore, "a" * 40, man_pins, "impl.py")
    sm2 = dict(sm, schema="f2g-w2-tier-s-smoke-v1",
               effect_grids_sha256=_digest(grids),
               invocation_ref={"commit": "a" * 40,
                               "path": "ts_invocation.json"},
               invocation_sha256=ts_inv["invocation_sha256"])
    astore[("a" * 40, "smoke2.json")] = json.dumps(sm2).encode()

    def areader(commit, path):
        if path.endswith("execution_manifest.json"):
            return json.dumps(fix_man).encode()
        try:
            return astore[(commit, path)]
        except KeyError:
            raise SelectorRefusal(
                f"SELECTOR_UNADMITTED: {path} unreadable at {commit}")
    refs2 = {"smoke_ref": {"commit": "a" * 40,
                           "path": "smoke2.json"},
             "effect_grids_ref": {"commit": "a" * 40,
                                  "path": "grids2.json"}}
    art_a = select_candidates(sm2, grids, **refs2)
    astore[("a" * 40, "selector2.json")] = json.dumps(art_a).encode()
    adm = verify_selector_admission(
        ".", art_a, "a" * 40, blob_reader=areader,
        git_resolve=lambda c: c)
    assert adm["effect_grids"]["blob_sha256"] ==         _hl.sha256(grids_raw).hexdigest()
    assert adm["smoke"]["invocation_sha256"] ==         ts_inv["invocation_sha256"]

    def arefuses(art_x, needle, resolve=lambda c: c):
        try:
            verify_selector_admission(".", art_x, "a" * 40,
                                      blob_reader=areader,
                                      git_resolve=resolve)
            return False
        except SelectorRefusal as e:
            return needle in str(e)
    # THREE COMMITTED, INTERNALLY CONSISTENT SUBSTITUTE CARRIERS:
    # a coordinated grid+smoke+selector at other paths -- committed,
    # readable, self-consistent -- refuse as UNADMITTED (the grid is
    # not the manifest-pinned blob)
    g_sub = copy.deepcopy(grids)
    g_sub["B2A"] = [{"m": 999}, {"m": 2}, {"m": 3}]
    g_sub_raw = json.dumps({"schema": "f2g-w2-effect-grids-v1",
                            "grids": g_sub}).encode()
    sm_sub_f = copy.deepcopy(sm2["families"])
    sm_sub_f["B2A"][0]["point"] = {"m": 999}
    sm_sub = dict(sm2, families=sm_sub_f,
                  effect_grids_sha256=_digest(g_sub))
    astore[("a" * 40, "sub_grids.json")] = g_sub_raw
    astore[("a" * 40, "sub_smoke.json")] = json.dumps(
        sm_sub).encode()
    art_sub = select_candidates(
        sm_sub, g_sub,
        smoke_ref={"commit": "a" * 40, "path": "sub_smoke.json"},
        effect_grids_ref={"commit": "a" * 40,
                          "path": "sub_grids.json"})
    assert arefuses(art_sub, "SELECTOR_UNADMITTED")
    # smoke missing the invocation reference -> unadmitted
    sm_noinv = {k: v for k, v in sm2.items()
                if k != "invocation_ref"}
    astore[("a" * 40, "smoke_noinv.json")] = json.dumps(
        sm_noinv).encode()
    art_ni = select_candidates(
        sm_noinv, grids,
        smoke_ref={"commit": "a" * 40, "path": "smoke_noinv.json"},
        effect_grids_ref=refs2["effect_grids_ref"])
    assert arefuses(art_ni, "lacks a closed Tier-S invocation")
    # forged Tier-S invocation digest -> unadmitted
    ts_bad = dict(ts_inv, invocation_sha256="0" * 64)
    sm_bad = dict(sm2, invocation_sha256="0" * 64,
                  invocation_ref={"commit": "a" * 40,
                                  "path": "ts_bad.json"})
    astore[("a" * 40, "ts_bad.json")] = json.dumps(ts_bad).encode()
    astore[("a" * 40, "smoke_bad.json")] = json.dumps(
        sm_bad).encode()
    art_fb = select_candidates(
        sm_bad, grids,
        smoke_ref={"commit": "a" * 40, "path": "smoke_bad.json"},
        effect_grids_ref=refs2["effect_grids_ref"])
    assert arefuses(art_fb, "does not recompute")
    # codex 0320Z item 2 LOCK: the admitted grid + a fabricated
    # all-success smoke + a minimal self-hashed invocation REFUSES
    # (the closed capsule schema kills the dict; a well-formed
    # capsule without a deriving results carrier dies at rebuild)
    min_inv = {"schema": "f2g-w2-tier-s-invocation-v1",
               "purpose": "attests no execution"}
    min_inv["invocation_sha256"] = _digest(
        {k: v for k, v in min_inv.items()
         if k != "invocation_sha256"})
    astore[("a" * 40, "min_inv.json")] = json.dumps(
        min_inv).encode()
    fab_f = {}
    for fam, entries in sm["families"].items():
        fab_f[fam] = [dict(e, outcomes=[True] * 50) for e in entries]
        if fam == "B1B":
            # all-50 ties re-rank stage 1 by grid index -> the
            # fabricated top-8 is indices 0..7
            for k, e in enumerate(fab_f[fam]):
                e["post_loco_outcomes"] = ([True] * 50 if k < 8
                                           else None)
    fab_smoke = dict(sm, families=fab_f,
                     schema="f2g-w2-tier-s-smoke-v1",
                     effect_grids_sha256=_digest(grids),
                     invocation_ref={"commit": "a" * 40,
                                     "path": "min_inv.json"},
                     invocation_sha256=min_inv["invocation_sha256"])
    astore[("a" * 40, "fab_smoke.json")] = json.dumps(
        fab_smoke).encode()
    art_fab = select_candidates(
        fab_smoke, grids,
        smoke_ref={"commit": "a" * 40, "path": "fab_smoke.json"},
        effect_grids_ref=refs2["effect_grids_ref"])
    assert arefuses(art_fab, "self-hashed dict attests nothing")
    # and a WELL-FORMED capsule whose smoke lists do not derive from
    # its results carrier refuses at the rebuild
    fab_inv2 = mk_tier_s_capsule(sm["families"], grids, grids_raw,
                                 astore, "a" * 40, man_pins,
                                 "impl.py")
    fab2 = dict(fab_smoke,
                invocation_ref={"commit": "a" * 40,
                                "path": "ts_invocation.json"},
                invocation_sha256=fab_inv2["invocation_sha256"])
    astore[("a" * 40, "fab2_smoke.json")] = json.dumps(
        fab2).encode()
    art_fab2 = select_candidates(
        fab2, grids,
        smoke_ref={"commit": "a" * 40, "path": "fab2_smoke.json"},
        effect_grids_ref=refs2["effect_grids_ref"])
    assert arefuses(art_fab2, "do not REBUILD")

    # unresolvable lineage -> unadmitted
    def bad_resolve(c):
        raise SelectorRefusal(
            f"SELECTOR_UNADMITTED: commit {c!r} does not resolve "
            "to an admitted 40-hex lineage")
    assert arefuses(art_a, "does not resolve", resolve=bad_resolve)

    print("w2_tier_selector selftest: ALL PASS (hand fixtures; "
          "PRELIMINARY_SMOKE semantics; nothing certified)")


if __name__ == "__main__":
    _selftest()
