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
TIER_S_RESULTS_SCHEMA = "f2g-w2-tier-s-results-v2"
# codex 0349Z item 3: closed nested schemas
TIER_S_RESULTS_FIELDS = {"schema", "quality",
                         "seed_authority_sha256",
                         "geometry_capsule_digest",
                         "implementation", "families"}
TIER_S_RESULT_ENTRY_FIELDS = {"point", "grid_index",
                              "replicates",
                              "loco_folds"}
# v2 (codex 0314Z Design A). No v1 acceptance in the production path:
# there is no production v1 Tier-S carrier to preserve, and a
# permissive downgrade would let a pre without the driver pin or the
# execution capsule reach admission -- which is the whole gap v2
# closes. A v1 fixture survives only inside the explicit downgrade
# REFUSAL control.
TIER_S_PRE_SCHEMA = "f2g-w2-tier-s-pre-invocation-v2"
TIER_S_PRE_FIELDS = {"schema", "manifest_commit", "effect_grids",
                     "effect_grids_content_sha256",
                     "geometry", "quality", "seed_authority_sha256",
                     "implementation", "driver", "execution",
                     "grid_order_sha256",
                     "output_root", "argv",
                     "fired_utc", "invocation_sha256"}
TIER_S_EXECUTION_SCHEMA = "f2g-w2-tier-s-execution-identity-v1"
TIER_S_EXECUTION_FIELDS = {"schema", "host", "interpreter_executable",
                           "interpreter_implementation",
                           "interpreter_version", "numpy_version",
                           "numpy_config_sha256"}
TIER_S_COMPLETION_SCHEMA = "f2g-w2-tier-s-completion-v1"
TIER_S_COMPLETION_FIELDS = {"schema", "pre_invocation_sha256",
                            "results_blob_sha256", "fired_utc",
                            "completed_utc"}
ID_FIELDS = {"commit", "path", "blob_sha256"}
GEO_ID_FIELDS = {"commit", "path", "capsule_digest"}


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


def _canon_instant(v, where):
    import re as _re
    if not isinstance(v, str) or not _re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
            r"(\.\d{1,6})?Z", v):
        raise SelectorRefusal(
            f"SELECTOR_UNADMITTED: {where} is not a canonical UTC "
            "instant")
    import datetime as _dt
    return _dt.datetime.fromisoformat(v.replace("Z", "+00:00"))


def _valid_p(v, where, allow_none=False):
    """codex 1328Z item 5: a p-value is None (registered untestable,
    where allowed) or a NON-BOOLEAN finite numeric in [0, 1] --
    booleans, negatives, >1, NaN, and infinities are never scientific
    evidence."""
    if v is None:
        if allow_none:
            return
        raise SelectorRefusal(
            f"SELECTOR_UNADMITTED: {where} is None where a value is "
            "required")
    import math as _m
    if isinstance(v, bool) or not isinstance(v, (int, float)) or             not _m.isfinite(v) or not 0.0 <= float(v) <= 1.0:
        raise SelectorRefusal(
            f"SELECTOR_UNADMITTED: {where} {v!r} is not a finite "
            "numeric in [0, 1]")


def _rebuild_outcomes(fam, entry, holm_fn, loco_registry):
    """codex 0349Z item 2: outcomes are DERIVED, never declared --
    per replicate the full four-family p-vector reruns Holm; for B1B
    the registered partial-LOCO substitution reruns over the bound
    fold registry."""
    pre, post = [], []
    reps = entry.get("replicates")
    folds = entry.get("loco_folds")
    if not isinstance(reps, list):
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: results replicates missing")
    for r_i, rep in enumerate(reps):
        if not isinstance(rep, dict) or set(rep) != {"p_values"}:
            raise SelectorRefusal(
                "SELECTOR_UNADMITTED: result replicate schema not "
                "closed")
        pv = rep["p_values"]
        if not isinstance(pv, dict) or \
                set(pv) != {"B1B", "B2A", "B2B", "B3A"}:
            raise SelectorRefusal(
                "SELECTOR_UNADMITTED: replicate p-vector not the "
                "four families")
        for f, v in pv.items():
            # codex 1912Z ruling: a None component is the REGISTERED
            # untestable family (harness sec 5 replicate rule) -- a
            # non-rejection with m held at 4, never a removed
            # hypothesis and never a synthesized number; booleans,
            # strings, NaN, infinities, negatives and >1 still refuse
            _valid_p(v, f"{f} p-value", allow_none=True)
        # cayley successor of the 1912Z rule (verifier LOW finding 1,
        # seam ORDERING): the replicate's LOCO fold map is VALIDATED
        # here, before the own-family None short-circuit below, so a
        # B1B replicate whose own p is None can no longer carry a
        # malformed fold map through the seam silently while the
        # identical defect on a numeric replicate refuses. The fold
        # p-values are only CONSUMED (substituted into Holm) after the
        # short-circuit; the registry needle and the fold p needle are
        # the pre-existing ones, the coverage needle is new.
        fr = None
        if fam == "B1B" and folds is not None:
            if not isinstance(folds, list) or len(folds) != len(reps):
                raise SelectorRefusal(
                    "SELECTOR_UNADMITTED: LOCO fold maps do not cover "
                    "every replicate")
            fr = folds[r_i]
            if fr is not None:
                # a fold map is a dict over exactly the registered set;
                # a non-dict iterable of the station names is not one
                if not isinstance(fr, dict) or \
                        sorted(fr) != sorted(loco_registry):
                    raise SelectorRefusal(
                        "SELECTOR_UNADMITTED: LOCO fold set diverges "
                        "from the bound registry")
                for st in sorted(fr):
                    _valid_p(fr[st], f"loco:{st} fold p-value",
                             allow_none=True)
        if pv[fam] is None:
            # the entry's OWN family is untestable on this replicate:
            # a non-recovery (False), not an unadmitted artifact
            pre.append(False)
            if fam == "B1B" and folds is not None:
                post.append(False)
            continue
        rej = holm_fn(pv)
        pre.append(fam in rej)
        if fam == "B1B" and folds is not None:
            if fr is None:
                post.append(False)
                continue
            ok = "B1B" in rej
            for st in sorted(fr):
                p_s = fr[st]
                if p_s is None or "B1B" not in holm_fn(
                        dict(pv, B1B=p_s)):
                    ok = False
            post.append(ok)
    return pre, (post if fam == "B1B" and folds is not None
                 else None)


def verify_selector_admission(repo, art, manifest_commit, *,
                              blob_reader=None, git_resolve=None,
                              geometry_loader=None,
                              is_ancestor=None,
                              selector_identity=None):
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
    def _bound_pin(path):
        matches = []
        for slot in man.get("slots", {}).values():
            if slot.get("status") != "BOUND":
                continue
            for pin in slot.get("pins", ()) or ():
                if isinstance(pin, dict) and pin.get("path") == path:
                    matches.append(pin)
        if len(matches) > 1:
            raise SelectorRefusal(
                f"SELECTOR_UNADMITTED: {path} has {len(matches)} "
                "BOUND manifest pins; identity is ambiguous")
        return matches[0] if matches else None

    # (a) the effect grids must BE the manifest-pinned blob
    g_ref = art["effect_grids_ref"]
    pinned = _bound_pin(g_ref["path"])
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
    # --- codex 0432Z: the chronological chain ---
    # manifest -> PRE-invocation -> results/completion -> smoke ->
    # selector. The pre-invocation is publishable BEFORE any result
    # exists; a post-hoc combined capsule has no schema to live in.
    pre_ref = smoke.get("pre_invocation_ref")
    comp_ref = smoke.get("completion_ref")
    res_ref = smoke.get("results_ref")
    for nm, r, want in (("pre_invocation_ref", pre_ref,
                         {"commit", "path"}),
                        ("completion_ref", comp_ref,
                         {"commit", "path"}),
                        ("results_ref", res_ref, ID_FIELDS)):
        if not isinstance(r, dict) or set(r) != want:
            raise SelectorRefusal(
                f"SELECTOR_UNADMITTED: smoke lacks a closed {nm}")
    pre_full = _resolve(pre_ref["commit"])
    pre = json.loads(blob_reader(pre_full,
                                 pre_ref["path"]).decode("utf-8"))
    if not isinstance(pre, dict) or \
            set(pre) != TIER_S_PRE_FIELDS or \
            pre.get("schema") != TIER_S_PRE_SCHEMA:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: the Tier-S pre-invocation is not "
            "the closed capsule schema (a self-hashed dict attests "
            "nothing)")
    got = hashlib.sha256(_canon(
        {k: v for k, v in pre.items()
         if k != "invocation_sha256"}).encode()).hexdigest()
    if got != pre.get("invocation_sha256") or \
            got != smoke.get("pre_invocation_sha256"):
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: the pre-invocation digest does "
            "not recompute from the reopened record")
    if _resolve(pre["manifest_commit"]) != mc_full:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: the pre-invocation binds a "
            "different manifest lineage")
    ig = pre["effect_grids"]
    if not isinstance(ig, dict) or set(ig) != ID_FIELDS or \
            ig["path"] != g_ref["path"] or \
            ig["blob_sha256"] != pinned["blob_sha256"] or \
            ig["commit"] != pinned["commit"]:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: the pre-invocation does not bind "
            "the admitted effect-grid identity (commit+path+blob)")
    if pre.get("quality") != {"R": TIER_S_R,
                              "n_draws": TIER_S_DRAWS}:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: pre-invocation quality is not the "
            "admitted Tier-S quality")
    # codex 0314Z point 4: the two identities v2 added are ADMITTED
    # here, not merely carried. The driver is the artifact that fired
    # the campaign; before v2 it was outside the admitted set
    # entirely, so a selector could admit a chain without ever
    # establishing what produced it.
    drv = pre["driver"]
    if not isinstance(drv, dict) or set(drv) != ID_FIELDS:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: pre-invocation driver identity is "
            "not a closed {commit, path, blob_sha256} reference")
    drv_pin = _bound_pin(drv["path"])
    if drv_pin is None:
        raise SelectorRefusal(
            f"SELECTOR_UNADMITTED: the firing driver {drv['path']} is "
            "not a BOUND pin of the admitted manifest")
    if drv["commit"] != drv_pin["commit"] or \
            drv["blob_sha256"] != drv_pin["blob_sha256"]:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: the pre binds a driver identity "
            f"({str(drv['commit'])[:12]}/"
            f"{str(drv['blob_sha256'])[:12]}) that is not the "
            f"admitted pin ({str(drv_pin['commit'])[:12]}/"
            f"{str(drv_pin['blob_sha256'])[:12]})")
    ex = pre["execution"]
    if not isinstance(ex, dict) or \
            set(ex) != TIER_S_EXECUTION_FIELDS or \
            ex.get("schema") != TIER_S_EXECUTION_SCHEMA:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: pre-invocation execution identity "
            "is not the closed registered capsule")
    for _k in ("host", "interpreter_executable",
               "interpreter_implementation", "interpreter_version",
               "numpy_version", "numpy_config_sha256"):
        if not isinstance(ex[_k], str) or not ex[_k].strip():
            raise SelectorRefusal(
                f"SELECTOR_UNADMITTED: execution identity field {_k} "
                "is empty -- an identity with a hole in it attests "
                "less than it appears to")
    geo = pre["geometry"]
    if not isinstance(geo, dict) or set(geo) != GEO_ID_FIELDS:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: pre-invocation geometry identity "
            "not closed")
    geo_pin = _bound_pin(geo["path"])
    if geo_pin is None or geo["commit"] != geo_pin["commit"]:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: pre-invocation geometry commit is "
            "not the manifest pin commit")
    if smoke.get("geometry_capsule_digest") != \
            geo["capsule_digest"]:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: smoke geometry diverges from the "
            "pre-invocation's bound capsule")
    if geometry_loader is None:
        import w2_power_harness_cayley as PH

        def geometry_loader(mc, path):
            cap = PH._load_bound_geometry(
                repo, {"manifest_commit": mc, "path": path})
            if PH._geometry_capsule_digest(cap) != \
                    cap.get("capsule_digest"):
                raise SelectorRefusal(
                    "SELECTOR_UNADMITTED: geometry capsule digest "
                    "does not recompute")
            return cap
    try:
        geo_cap = geometry_loader(mc_full, geo["path"])
    except SelectorRefusal:
        raise
    except Exception as e:
        raise SelectorRefusal(
            f"SELECTOR_UNADMITTED: geometry not manifest-pinned or "
            f"unreadable: {e}")
    if geo_cap.get("capsule_digest") != geo["capsule_digest"]:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: loaded geometry diverges from the "
            "declared capsule digest")
    loco_registry = list(geo_cap.get("registries", {}).get(
        geo_cap.get("loco_registry_carrier"), ()))
    sa = pre["seed_authority_sha256"]
    if not (isinstance(sa, str) and len(sa) == 64 and
            all(c in "0123456789abcdef" for c in sa)):
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: seed authority is not "
            "lowercase-hex")
    impl = pre["implementation"]
    impl_pin = _bound_pin(impl.get("path"))
    if not isinstance(impl, dict) or set(impl) != ID_FIELDS or \
            impl_pin is None or \
            impl["blob_sha256"] != impl_pin["blob_sha256"] or \
            impl["commit"] != impl_pin["commit"]:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: pre-invocation implementation is "
            "not the manifest-pinned identity")
    grids_art2 = json.loads(g_raw.decode("utf-8"))
    grids2 = grids_art2.get("grids", grids_art2)
    det_order = {fam: [p for p in grids2[fam] if "gain" not in p]
                 for fam in FAMILIES_ORDER}
    if pre["grid_order_sha256"] != _digest(det_order):
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: pre-invocation grid order "
            "diverges from the admitted detection grids")
    if pre["effect_grids_content_sha256"] != _digest(grids2) or             smoke.get("effect_grids_sha256") !=             pre["effect_grids_content_sha256"]:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: the grid content digest chain "
            "(admitted bytes -> pre -> smoke) is broken")
    # completion: closed, binds pre digest + results blob, ordered
    comp_full = _resolve(comp_ref["commit"])
    comp = json.loads(blob_reader(
        comp_full, comp_ref["path"]).decode("utf-8"))
    if not isinstance(comp, dict) or \
            set(comp) != TIER_S_COMPLETION_FIELDS or \
            comp.get("schema") != TIER_S_COMPLETION_SCHEMA:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: completion receipt not the closed "
            "schema")
    if comp["pre_invocation_sha256"] != pre["invocation_sha256"]:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: completion does not bind the "
            "pre-fire invocation digest")
    t0 = _canon_instant(comp["fired_utc"], "fired_utc")
    t1 = _canon_instant(comp["completed_utc"], "completed_utc")
    if not t0 <= t1:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: completion times reversed")
    # results: blob-bound, identities == pre's, DERIVATIONAL
    r_full = _resolve(res_ref["commit"])
    r_raw = blob_reader(r_full, res_ref["path"])
    if hashlib.sha256(r_raw).hexdigest() != \
            res_ref["blob_sha256"] or \
            res_ref["blob_sha256"] != comp["results_blob_sha256"]:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: results bytes diverge from the "
            "completion's output binding")
    results = json.loads(r_raw.decode("utf-8"))
    if not isinstance(results, dict) or \
            set(results) != TIER_S_RESULTS_FIELDS or \
            results.get("schema") != TIER_S_RESULTS_SCHEMA:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: results capsule schema not closed")
    if results.get("quality") != pre["quality"] or \
            results.get("seed_authority_sha256") != sa or \
            results.get("geometry_capsule_digest") != \
            geo["capsule_digest"] or \
            results.get("implementation") != impl:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: results identities diverge from "
            "the pre-invocation's bound identities")
    # codex 0432Z item 3: exact family set; grid_index equality;
    # non-B1B folds exactly null; B1B fold lists sized R
    results_families = results.get("families")
    smoke_families = smoke.get("families")
    if not isinstance(results_families, dict) or \
            not isinstance(smoke_families, dict) or \
            set(results_families) != set(FAMILIES_ORDER) or \
            set(smoke_families) != set(FAMILIES_ORDER):
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: family set is not exactly the "
            "registered four")
    import w2_power_harness_cayley as _PH
    for fam in FAMILIES_ORDER:
        rf = results_families[fam]
        sf = smoke_families[fam]
        if not isinstance(rf, list) or not isinstance(sf, list) or \
                len(rf) != len(sf):
            raise SelectorRefusal(
                f"SELECTOR_UNADMITTED: {fam} results/smoke shape "
                "mismatch")
        for k, (re_, se_) in enumerate(zip(rf, sf)):
            if not isinstance(re_, dict) or \
                    set(re_) != TIER_S_RESULT_ENTRY_FIELDS:
                raise SelectorRefusal(
                    f"SELECTOR_UNADMITTED: {fam}[{k}] result entry "
                    "schema not closed")
            if re_.get("point") != se_.get("point") or \
                    re_.get("point") != det_order[fam][k]:
                raise SelectorRefusal(
                    f"SELECTOR_UNADMITTED: {fam}[{k}] point is not "
                    "the registered grid point")
            if re_.get("grid_index") != k:
                raise SelectorRefusal(
                    f"SELECTOR_UNADMITTED: {fam}[{k}] grid_index "
                    "diverges from the enumerated admitted index")
            folds = re_.get("loco_folds")
            if fam != "B1B" and folds is not None:
                raise SelectorRefusal(
                    f"SELECTOR_UNADMITTED: {fam}[{k}] carries LOCO "
                    "folds (B1B stage-2 only)")
            if fam == "B1B" and folds is not None:
                reps_for_folds = re_.get("replicates")
                if not isinstance(folds, list) or \
                        not isinstance(reps_for_folds, list) or \
                        len(folds) != len(reps_for_folds):
                    raise SelectorRefusal(
                        f"SELECTOR_UNADMITTED: B1B[{k}] fold list is "
                        "not sized to the replicates")
            pre_o, post_o = _rebuild_outcomes(
                fam, re_, _PH.holm_rejects, loco_registry)
            if pre_o != se_.get("outcomes"):
                raise SelectorRefusal(
                    f"SELECTOR_UNADMITTED: {fam}[{k}] smoke "
                    "outcomes do not DERIVE from the results "
                    "p-vectors under the registered Holm rule")
            if fam == "B1B" and post_o != \
                    se_.get("post_loco_outcomes"):
                raise SelectorRefusal(
                    f"SELECTOR_UNADMITTED: B1B[{k}] post-LOCO "
                    "outcomes do not DERIVE from the recorded fold "
                    "p-vectors under the registered substitution "
                    "rule")
    # the DESCENDANT CHAIN, incl the pre-invocation commit and the
    # smoke -> selector edge (codex 0432Z item 2)
    if is_ancestor is None:
        import subprocess as _sp

        def is_ancestor(a, b):
            return _sp.run(["git", "-C", repo, "merge-base",
                            "--is-ancestor", a, b],
                           capture_output=True).returncode == 0

    def strict_edge(a, b, label):
        """codex 1328Z item 2: STAGE ancestry is STRICT -- a commit
        is its own git-ancestor, so reflexive edges are refused; a
        post-hoc same-commit capsule has no chain to live on."""
        if a == b or not is_ancestor(a, b):
            raise SelectorRefusal(
                f"SELECTOR_UNADMITTED: lineage edge {label} is not "
                "STRICT stage ancestry (same-commit or non-ancestor)")
    strict_edge(mc_full, pre_full, "manifest->pre")
    # ---- codex 0537Z: the point-corpus RECEIPT is adjudicated, never
    # carried. The smoke names the commit its carriers were read from
    # and the digest of that exact carrier set; admission reopens those
    # carriers, rebuilds the results they imply and requires equality
    # with the reopened results -- so a real point commit cannot be
    # paired with unrelated result bytes, and a forged receipt cannot
    # ride a self-consistent selector chain.
    pc = smoke.get("points_commit")
    pcs = smoke.get("point_corpus_sha256")
    if not (isinstance(pc, str) and len(pc) == 40 and
            all(c in "0123456789abcdef" for c in pc)):
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: smoke carries no full lowercase-40 "
            "points_commit -- the point corpus has no committed identity")
    if not (isinstance(pcs, str) and len(pcs) == 64 and
            all(c in "0123456789abcdef" for c in pcs)):
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: smoke carries no 64-hex "
            "point_corpus_sha256")
    rel_dir = res_ref["path"].rsplit("/", 1)[0] \
        if "/" in res_ref["path"] else ""

    def _cpath(name):
        return f"{rel_dir}/{name}" if rel_dir else name

    # The results commit is the independent authority for the receipt
    # finalize consumed.  Rebuilding equal carrier bytes is not enough:
    # without this join, a different intermediate commit carrying those
    # same bytes can be substituted into a later final smoke.
    try:
        committed_draft = json.loads(blob_reader(
            r_full, _cpath("tier_s_smoke.json")).decode("utf-8"))
        committed_envelope = json.loads(blob_reader(
            r_full, _cpath("tier_s_aggregate_envelope.json"))
            .decode("utf-8"))
    except SelectorRefusal:
        raise
    except (ValueError, UnicodeDecodeError):
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: results-commit draft smoke or "
            "aggregate envelope is not JSON")
    if not isinstance(committed_draft, dict) or \
            not isinstance(committed_envelope, dict) or \
            committed_draft.get("schema") != "f2g-w2-tier-s-smoke-v1" or \
            committed_envelope.get("schema") != \
            "f2g-w2-tier-s-aggregate-envelope-v1":
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: results-commit draft smoke or "
            "aggregate envelope is not the registered schema")
    for label, receipt in (("draft smoke", committed_draft),
                           ("aggregate envelope", committed_envelope)):
        if not isinstance(receipt, dict) or \
                receipt.get("points_commit") != pc or \
                receipt.get("point_corpus_sha256") != pcs:
            raise SelectorRefusal(
                "SELECTOR_UNADMITTED: final smoke point-corpus receipt "
                f"diverges from the results-commit {label}")
    pc_full = _resolve(pc)
    if pc_full != pc:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: points_commit does not resolve to itself")
    strict_edge(pre_full, pc_full, "pre->points")
    strict_edge(pc_full, r_full, "points->results")
    if comp_full != r_full:
        strict_edge(pc_full, comp_full, "points->completion")
    ex_digest = _digest(pre["execution"])
    points_all = [(fam, p) for fam in FAMILIES_ORDER
                  for p in det_order[fam]]
    CAP_FIELDS = {"index", "family", "point", "pre_invocation_sha256",
                  "execution_sha256", "record"}

    def _reopen_capsule(idx, fam, point, name):
        raw = blob_reader(pc_full, _cpath(name))
        try:
            cap = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise SelectorRefusal(
                f"SELECTOR_UNADMITTED: committed carrier {name} at "
                f"{pc_full[:12]} is not JSON")
        if not isinstance(cap, dict) or set(cap) != CAP_FIELDS:
            raise SelectorRefusal(
                f"SELECTOR_UNADMITTED: committed carrier {name} schema "
                "not closed")
        if cap["index"] != idx or cap["family"] != fam or \
                cap["point"] != point:
            raise SelectorRefusal(
                f"SELECTOR_UNADMITTED: committed carrier {name} identity "
                "diverges from the derived grid point")
        if cap["pre_invocation_sha256"] != pre["invocation_sha256"]:
            raise SelectorRefusal(
                f"SELECTOR_UNADMITTED: committed carrier {name} does not "
                "bind this pre")
        if cap["execution_sha256"] != ex_digest:
            raise SelectorRefusal(
                f"SELECTOR_UNADMITTED: committed carrier {name} was "
                "produced under a runtime identity that is not the "
                "pre-bound one")
        rec = cap["record"]
        if not isinstance(rec, dict) or \
                rec.get("certifiable", False) is not False:
            raise SelectorRefusal(
                f"SELECTOR_UNADMITTED: committed carrier {name} record is "
                "not a non-certifiable smoke record")
        reps = rec.get("replicates")
        if not isinstance(reps, list) or len(reps) != TIER_S_R:
            raise SelectorRefusal(
                f"SELECTOR_UNADMITTED: committed carrier {name} does not "
                f"carry exactly R={TIER_S_R} replicates")
        for j, rep in enumerate(reps):
            if not isinstance(rep, dict) or set(rep) != {"p_values"} or \
                    not isinstance(rep["p_values"], dict) or \
                    set(rep["p_values"]) != set(FAMILIES_ORDER):
                raise SelectorRefusal(
                    f"SELECTOR_UNADMITTED: committed carrier {name} "
                    f"replicate {j} is not a closed four-family p-vector")
            for fk, v in rep["p_values"].items():
                _valid_p(v, f"carrier {name} replicate {j} {fk}",
                         allow_none=True)
        return cap, hashlib.sha256(raw).hexdigest()
    carriers = []
    rebuilt = {fam: [] for fam in FAMILIES_ORDER}
    b1b_counts = []
    for idx, (fam, point) in enumerate(points_all):
        name = f"smoke_point_{idx:03d}.json"
        cap, csha = _reopen_capsule(idx, fam, point, name)
        carriers.append([_cpath(name), csha])
        if cap["record"].get("loco_folds") is not None:
            raise SelectorRefusal(
                f"SELECTOR_UNADMITTED: detection carrier {name} carries "
                "LOCO folds")
        entry = {"point": dict(point), "grid_index": None,
                 "replicates": cap["record"]["replicates"],
                 "loco_folds": None}
        rebuilt[fam].append((idx, entry))
        if fam == "B1B":
            b1b_counts.append((idx, sum(
                1 for rep in entry["replicates"]
                if "B1B" in _PH.holm_rejects(rep["p_values"]))))
    top8 = [i for i, _c in sorted(b1b_counts, key=lambda t: (-t[1], t[0]))
            [:8]]
    for fam in FAMILIES_ORDER:
        for k, (idx, e) in enumerate(rebuilt[fam]):
            e["grid_index"] = k
            if fam == "B1B" and idx in top8:
                lname = f"smoke_loco_{idx:03d}.json"
                lcap, lsha = _reopen_capsule(idx, fam, e["point"], lname)
                folds = lcap["record"].get("loco_folds")
                if not isinstance(folds, list) or len(folds) != TIER_S_R:
                    raise SelectorRefusal(
                        f"SELECTOR_UNADMITTED: LOCO carrier {lname} does "
                        f"not carry exactly R={TIER_S_R} fold maps")
                for j, fr in enumerate(folds):
                    if not isinstance(fr, dict) or \
                            sorted(fr) != sorted(loco_registry):
                        raise SelectorRefusal(
                            f"SELECTOR_UNADMITTED: LOCO carrier {lname} "
                            f"fold {j} is not over the registered LOCO set")
                    for st, v in fr.items():
                        _valid_p(v, f"LOCO carrier {lname} fold {j} {st}",
                                 allow_none=True)
                e["loco_folds"] = folds
                carriers.append([_cpath(lname), lsha])
    expected_results = {"schema": TIER_S_RESULTS_SCHEMA,
                        "quality": dict(pre["quality"]),
                        "seed_authority_sha256":
                            pre["seed_authority_sha256"],
                        "geometry_capsule_digest": geo["capsule_digest"],
                        "implementation": dict(pre["implementation"]),
                        "families": {fam: [e for _i, e in rebuilt[fam]]
                                     for fam in FAMILIES_ORDER}}
    if expected_results != results:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: the reopened results do not equal the "
            "results REBUILT from the committed point corpus at "
            f"{pc_full[:12]} -- a real point commit paired with unrelated "
            "result bytes")
    if _digest(sorted(carriers)) != pcs:
        raise SelectorRefusal(
            "SELECTOR_UNADMITTED: point_corpus_sha256 does not recompute "
            f"from the {len(carriers)} committed carriers at {pc_full[:12]}")
    if r_full == comp_full:
        # results and completion may intentionally share ONE commit;
        # that shared commit must sit strictly between pre and smoke
        strict_edge(pre_full, r_full, "pre->results/completion")
        strict_edge(r_full, s_full, "results/completion->smoke")
    else:
        strict_edge(pre_full, r_full, "pre->results")
        strict_edge(pre_full, comp_full, "pre->completion")
        strict_edge(r_full, s_full, "results->smoke")
        strict_edge(comp_full, s_full, "completion->smoke")
    if selector_identity is not None:
        sel_full = _resolve(selector_identity["commit"])
        strict_edge(s_full, sel_full, "smoke->selector")
    return {"manifest_commit": mc_full,
            "effect_grids": {"commit": pinned["commit"],
                             "path": g_ref["path"],
                             "blob_sha256":
                                 hashlib.sha256(g_raw).hexdigest()},
            "pre_invocation": {"commit": pre_full,
                               "path": pre_ref["path"],
                               "invocation_sha256":
                                   pre["invocation_sha256"]},
            "results": {"commit": r_full, "path": res_ref["path"],
                        "blob_sha256": res_ref["blob_sha256"]},
            "smoke": {"commit": s_full, "path": s_ref["path"],
                      "blob_sha256": hashlib.sha256(blob_reader(
                          s_full, s_ref["path"])).hexdigest(),
                      "pre_invocation_sha256":
                          smoke.get("pre_invocation_sha256")},
            "point_corpus": {"commit": pc_full, "sha256": pcs,
                             "carriers": len(carriers)}}


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

    # B1B: stage-1 all-50 ties -> top 8 by grid index; stage-2
    # post-LOCO REORDERS. Derivational consistency requires
    # post[r] => pre[r], so pre is all-true.
    b1b_pre = [50] * 9
    keep_expected = [0, 1, 2, 3, 4, 5, 6, 7]
    post = {0: 17, 1: 45, 2: 44, 3: 46, 4: 20, 5: 19, 6: 18, 7: 16}
    sm = smoke_for({"B2A": [50, 49, 48],
                    "B2B": [10, 30, 30, 20, 40],
                    "B3A": [5, 6, 7, 8],
                    "B1B": b1b_pre}, post)
    art = select_candidates(sm, grids)
    assert art["top8_grid_indices"]["B1B"] == keep_expected
    # post-LOCO ranking: counts 46(k=3),45(k=1),44(k=2) win
    assert art["selected_grid_indices"]["B1B"] == [3, 1, 2]
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
    assert refuses(lambda s: s["families"]["B1B"][0].update(
        post_loco_outcomes=None), "SELECTOR_STAGE2_INVALID")
    # post-LOCO outside the top-8 refuses (index 8 not kept)
    assert refuses(lambda s: s["families"]["B1B"][8].update(
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

    GEOMD = "kat-digest"

    FIX_GEOM = {"capsule_digest": GEOMD,
                "loco_registry_carrier": "cascadia",
                "registries": {"cascadia": ["S0", "S1"]}}

    def geom_loader(mc, path):
        return FIX_GEOM


    FIX_GEOM = {"capsule_digest": GEOMD,
                "loco_registry_carrier": "cascadia",
                "registries": {"cascadia": ["S0", "S1"]}}

    def geom_loader(mc, path):
        return FIX_GEOM

    STAGE_PRE = "3" * 40
    STAGE_PTS = "7" * 40          # codex 0537Z: the committed point corpus
    STAGE_RES = "4" * 40
    STAGE_SMOKE = "5" * 40
    STAGE_SEL = "6" * 40

    def dag_ancestor(a, b):
        order = ["a" * 40, STAGE_PRE, STAGE_PTS, STAGE_RES,
                 STAGE_SMOKE, STAGE_SEL]
        try:
            return order.index(a) <= order.index(b)
        except ValueError:
            return False

    def mk_tier_s_capsule(smoke_families, grids_obj, grids_raw,
                          store_map, commit, man_pins, impl_path,
                          geom_digest=GEOMD, mc_override=None,
                          pv_fn=None):
        """pv_fn(fam, k, r_i, pv) -> pv: optional per-replicate
        p-vector hook applied IDENTICALLY to the results carrier and
        the point-corpus carriers (codex 1912Z: the None-rule chain
        needs registered-None components in both, never in one)."""
        import hashlib as _h
        fams4 = ("B1B", "B2A", "B2B", "B3A")
        registry = FIX_GEOM["registries"]["cascadia"]

        def reps_from(outcomes, fam, k):
            out = []
            for r_i, o in enumerate(outcomes):
                pv = {f: (0.001 if (o and f == fam) else 0.9)
                      for f in fams4}
                if pv_fn is not None:
                    pv = pv_fn(fam, k, r_i, pv)
                out.append({"p_values": pv})
            return out

        def folds_from(post):
            if post is None:
                return None
            out = []
            for p in post:
                if p:
                    out.append({st: 0.001 for st in registry})
                else:
                    d = {st: 0.001 for st in registry}
                    d[registry[0]] = 0.9
                    out.append(d)
            return out

        def pin_by(path_pred):
            return [p for p in man_pins if path_pred(p["path"])][0]
        grid_pin = pin_by(lambda p: p.startswith("grids"))
        geom_pin = pin_by(lambda p: p == "geom.json")
        impl_pin = pin_by(lambda p: p == impl_path)
        driver_pin = pin_by(
            lambda p: p.endswith("w2_tier_s_driver_cayley.py"))
        impl_id = {"commit": impl_pin["commit"], "path": impl_path,
                   "blob_sha256": impl_pin["blob_sha256"]}
        driver_id = {"commit": driver_pin["commit"],
                     "path": driver_pin["path"],
                     "blob_sha256": driver_pin["blob_sha256"]}
        results = {"schema": "f2g-w2-tier-s-results-v2",
                   "quality": {"R": 50, "n_draws": 999},
                   "seed_authority_sha256": "b" * 64,
                   "geometry_capsule_digest": geom_digest,
                   "implementation": impl_id,
                   "families": {}}
        for fam, entries in smoke_families.items():
            det_pts = [p for p in grids_obj[fam] if "gain" not in p]
            results["families"][fam] = [
                {"point": dict(e["point"]),
                 "grid_index": det_pts.index(e["point"]),
                 "replicates": reps_from(e["outcomes"], fam, k_e),
                 "loco_folds": folds_from(
                     e.get("post_loco_outcomes"))
                 if fam == "B1B" else None}
                for k_e, e in enumerate(entries)]
        r_raw = json.dumps(results).encode()
        r_sha = _h.sha256(r_raw).hexdigest()
        store_map[(STAGE_RES, "ts_results.json")] = r_raw
        det_order = {f: [p for p in grids_obj[f] if "gain" not in p]
                     for f in ("B2A", "B2B", "B1B", "B3A")}
        pre = {"schema": "f2g-w2-tier-s-pre-invocation-v2",
               "manifest_commit": mc_override or commit,
               "effect_grids": {"commit": grid_pin["commit"],
                               "path": grid_pin["path"],
                               "blob_sha256":
                                   grid_pin["blob_sha256"]},
               "effect_grids_content_sha256": _digest_fn(
                   {f: list(grids_obj[f])
                    for f in grids_obj}),
               "geometry": {"commit": geom_pin["commit"],
                            "path": "geom.json",
                            "capsule_digest": geom_digest},
               "quality": {"R": 50, "n_draws": 999},
               "seed_authority_sha256": "b" * 64,
               "implementation": impl_id,
               "driver": driver_id,
               "execution": {
                   "schema": "f2g-w2-tier-s-execution-identity-v1",
                   "host": "kat",
                   "interpreter_executable": "kat",
                   "interpreter_implementation": "CPython",
                   "interpreter_version": "kat",
                   "numpy_version": "kat",
                   "numpy_config_sha256": "e" * 64},
               "grid_order_sha256": _digest_fn(det_order),
               "output_root": "kat", "argv": ["kat"],
               "fired_utc": "2026-08-25T00:00:00Z"}
        pre["invocation_sha256"] = _digest_fn(
            {k: v for k, v in pre.items()
             if k != "invocation_sha256"})
        store_map[(STAGE_PRE, "ts_pre.json")] = json.dumps(
            pre).encode()
        # codex 0537Z: the COMMITTED point corpus the receipt names --
        # one carrier per registered detection point (runner order) and
        # a LOCO carrier for exactly the stage-1 top-8 (the entries the
        # fixture gives post-LOCO outcomes), all at STAGE_PTS; the
        # receipt is the digest of the exact sorted path->blob set
        ex_d = _digest_fn(pre["execution"])
        carriers = []
        idx = 0
        for fam_c in ("B2A", "B2B", "B1B", "B3A"):
            for k_c, pt_c in enumerate(det_order[fam_c]):
                e_c = smoke_families[fam_c][k_c]
                cap_c = {"index": idx, "family": fam_c,
                         "point": dict(pt_c),
                         "pre_invocation_sha256": pre["invocation_sha256"],
                         "execution_sha256": ex_d,
                         "record": {"replicates": reps_from(e_c["outcomes"],
                                                            fam_c, k_c),
                                    "loco_folds": None,
                                    "certifiable": False}}
                raw_c = json.dumps(cap_c).encode()
                nm_c = f"smoke_point_{idx:03d}.json"
                store_map[(STAGE_PTS, nm_c)] = raw_c
                carriers.append([nm_c, _h.sha256(raw_c).hexdigest()])
                if fam_c == "B1B" and \
                        e_c.get("post_loco_outcomes") is not None:
                    lcap_c = dict(cap_c, record={
                        "replicates": reps_from(e_c["outcomes"], fam_c,
                                                k_c),
                        "loco_folds": folds_from(e_c["post_loco_outcomes"]),
                        "certifiable": False})
                    lraw_c = json.dumps(lcap_c).encode()
                    lnm_c = f"smoke_loco_{idx:03d}.json"
                    store_map[(STAGE_PTS, lnm_c)] = lraw_c
                    carriers.append([lnm_c, _h.sha256(lraw_c).hexdigest()])
                idx += 1
        corpus_sha = _digest_fn(sorted(carriers))
        # The results-stage fixture carries both independent receipt
        # authorities used by admission, as the real aggregate does.
        store_map[(STAGE_RES, "tier_s_smoke.json")] = json.dumps({
            "schema": "f2g-w2-tier-s-smoke-v1",
            "points_commit": STAGE_PTS,
            "point_corpus_sha256": corpus_sha}).encode()
        store_map[(STAGE_RES, "tier_s_aggregate_envelope.json")] = \
            json.dumps({
                "schema": "f2g-w2-tier-s-aggregate-envelope-v1",
                "points_commit": STAGE_PTS,
                "point_corpus_sha256": corpus_sha}).encode()
        comp = {"schema": "f2g-w2-tier-s-completion-v1",
                "pre_invocation_sha256": pre["invocation_sha256"],
                "results_blob_sha256": r_sha,
                "fired_utc": "2026-08-25T00:00:00Z",
                "completed_utc": "2026-08-25T11:00:00Z"}
        store_map[(STAGE_RES, "ts_comp.json")] = json.dumps(
            comp).encode()
        return {"pre": pre, "comp": comp, "r_sha": r_sha,
                "smoke_fields": {
                    "pre_invocation_ref": {"commit": STAGE_PRE,
                                           "path": "ts_pre.json"},
                    "pre_invocation_sha256":
                        pre["invocation_sha256"],
                    "completion_ref": {"commit": STAGE_RES,
                                       "path": "ts_comp.json"},
                    "results_ref": {"commit": STAGE_RES,
                                    "path": "ts_results.json",
                                    "blob_sha256": r_sha},
                    "points_commit": STAGE_PTS,
                    "point_corpus_sha256": corpus_sha}}

    grids_raw = json.dumps({"schema": "f2g-w2-effect-grids-v1",
                            "grids": grids}).encode()
    driver_rel = "monitoring/src/w2_tier_s_driver_cayley.py"
    driver_raw = b"# pinned tier-s driver"
    astore = {("a" * 40, "grids2.json"): grids_raw,
               ("a" * 40, "impl.py"): b"# pinned impl",
               ("a" * 40, "geom.json"): b"{}",
               ("a" * 40, driver_rel): driver_raw}
    man_pins = [
        {"path": "grids2.json", "commit": "a" * 40,
         "blob_sha256": _hl.sha256(grids_raw).hexdigest()},
        {"path": "impl.py", "commit": "a" * 40,
         "blob_sha256": _hl.sha256(b"# pinned impl").hexdigest()},
        {"path": "geom.json", "commit": "a" * 40,
         "blob_sha256": _hl.sha256(b"{}").hexdigest()},
        {"path": driver_rel, "commit": "a" * 40,
         "blob_sha256": _hl.sha256(driver_raw).hexdigest()}]
    fix_man = {"slots": {"x": {"status": "BOUND",
                                  "pins": man_pins}}}
    caps = mk_tier_s_capsule(sm["families"], grids, grids_raw,
                             astore, "a" * 40, man_pins, "impl.py")
    sm2 = dict(sm, schema="f2g-w2-tier-s-smoke-v1",
               effect_grids_sha256=_digest(grids),
               **caps["smoke_fields"])
    astore[(STAGE_SMOKE, "smoke2.json")] = json.dumps(
        sm2).encode()

    def areader(commit, path):
        if path.endswith("execution_manifest.json"):
            return json.dumps(fix_man).encode()
        try:
            return astore[(commit, path)]
        except KeyError:
            raise SelectorRefusal(
                f"SELECTOR_UNADMITTED: {path} unreadable at {commit}")
    refs2 = {"smoke_ref": {"commit": STAGE_SMOKE,
                           "path": "smoke2.json"},
             "effect_grids_ref": {"commit": "a" * 40,
                                  "path": "grids2.json"}}
    art_a = select_candidates(sm2, grids, **refs2)
    astore[("a" * 40, "selector2.json")] = json.dumps(art_a).encode()
    adm = verify_selector_admission(
        ".", art_a, "a" * 40, blob_reader=areader,
        git_resolve=lambda c: c, geometry_loader=geom_loader,
        is_ancestor=dag_ancestor)
    assert adm["effect_grids"]["blob_sha256"] ==         _hl.sha256(grids_raw).hexdigest()
    assert adm["smoke"]["pre_invocation_sha256"] ==         caps["pre"]["invocation_sha256"]
    assert adm["pre_invocation"]["invocation_sha256"] ==         caps["pre"]["invocation_sha256"]

    def arefuses(art_x, needle, resolve=lambda c: c):
        try:
            verify_selector_admission(".", art_x, "a" * 40,
                                      blob_reader=areader,
                                      git_resolve=resolve,
                                      geometry_loader=geom_loader,
                                      is_ancestor=dag_ancestor)
            return False
        except SelectorRefusal as e:
            return needle in str(e)
    # A valid JSON value at either results-stage authority must still
    # refuse through the typed schema boundary.  Lists previously reached
    # `.get()` before the object check and escaped as AttributeError.
    for authority_name in ("tier_s_smoke.json",
                           "tier_s_aggregate_envelope.json"):
        authority_key = (STAGE_RES, authority_name)
        authority_true = astore[authority_key]
        astore[authority_key] = b"[]"
        assert arefuses(art_a, "not the registered schema"), authority_name
        astore[authority_key] = authority_true
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
    # smoke missing the pre-invocation reference -> unadmitted
    sm_noinv = {k: v for k, v in sm2.items()
                if k != "pre_invocation_ref"}
    astore[(STAGE_SMOKE, "smoke_noinv.json")] = json.dumps(
        sm_noinv).encode()
    art_ni = select_candidates(
        sm_noinv, grids,
        smoke_ref={"commit": STAGE_SMOKE, "path": "smoke_noinv.json"},
        effect_grids_ref=refs2["effect_grids_ref"])
    assert arefuses(art_ni, "lacks a closed pre_invocation_ref")
    # forged Tier-S invocation digest -> unadmitted
    ts_bad = dict(caps["pre"], invocation_sha256="0" * 64)
    sm_bad = dict(sm2, pre_invocation_sha256="0" * 64,
                  pre_invocation_ref={"commit": STAGE_PRE,
                                      "path": "ts_bad.json"})
    astore[(STAGE_PRE, "ts_bad.json")] = json.dumps(ts_bad).encode()
    astore[(STAGE_SMOKE, "smoke_bad.json")] = json.dumps(
        sm_bad).encode()
    art_fb = select_candidates(
        sm_bad, grids,
        smoke_ref={"commit": STAGE_SMOKE, "path": "smoke_bad.json"},
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
    astore[(STAGE_PRE, "min_inv.json")] = json.dumps(
        min_inv).encode()
    min_fields = dict(caps["smoke_fields"],
                      pre_invocation_ref={"commit": STAGE_PRE,
                                          "path": "min_inv.json"},
                      pre_invocation_sha256=min_inv[
                          "invocation_sha256"])
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
                     **min_fields)
    astore[(STAGE_SMOKE, "fab_smoke.json")] = json.dumps(
        fab_smoke).encode()
    art_fab = select_candidates(
        fab_smoke, grids,
        smoke_ref={"commit": STAGE_SMOKE, "path": "fab_smoke.json"},
        effect_grids_ref=refs2["effect_grids_ref"])
    assert arefuses(art_fab, "self-hashed dict attests nothing")
    # and a WELL-FORMED capsule whose smoke lists do not derive from
    # its results carrier refuses at the rebuild
    fab2 = dict(fab_smoke, **caps["smoke_fields"])
    astore[(STAGE_SMOKE, "fab2_smoke.json")] = json.dumps(
        fab2).encode()
    art_fab2 = select_candidates(
        fab2, grids,
        smoke_ref={"commit": STAGE_SMOKE, "path": "fab2_smoke.json"},
        effect_grids_ref=refs2["effect_grids_ref"])
    assert arefuses(art_fab2, "do not DERIVE")

    # --- codex 0349Z item 3: the nested-field + lineage MUTATION
    # TABLE -- every identity/lineage edge, one loop
    def re_capsule(mut_pre=None, mut_results=None, mut_comp=None,
                   ancestor=None):
        store2 = dict(astore)
        caps2 = mk_tier_s_capsule(sm["families"], grids, grids_raw,
                                  store2, "a" * 40, man_pins,
                                  "impl.py")
        if mut_results:
            res2 = json.loads(store2[(STAGE_RES,
                                      "ts_results.json")].decode())
            mut_results(res2)
            raw2 = json.dumps(res2).encode()
            store2[(STAGE_RES, "ts_results.json")] = raw2
            import hashlib as _h2
            r_sha2 = _h2.sha256(raw2).hexdigest()
            caps2["smoke_fields"]["results_ref"]["blob_sha256"] = \
                r_sha2
            comp2 = json.loads(store2[(STAGE_RES,
                                       "ts_comp.json")].decode())
            comp2["results_blob_sha256"] = r_sha2
            store2[(STAGE_RES, "ts_comp.json")] = json.dumps(
                comp2).encode()
        if mut_pre:
            pre2 = json.loads(store2[(STAGE_PRE,
                                      "ts_pre.json")].decode())
            mut_pre(pre2)
            pre2["invocation_sha256"] = _digest(
                {k: v for k, v in pre2.items()
                 if k != "invocation_sha256"})
            store2[(STAGE_PRE, "ts_pre.json")] = json.dumps(
                pre2).encode()
            caps2["smoke_fields"]["pre_invocation_sha256"] = \
                pre2["invocation_sha256"]
            comp2 = json.loads(store2[(STAGE_RES,
                                       "ts_comp.json")].decode())
            comp2["pre_invocation_sha256"] = \
                pre2["invocation_sha256"]
            store2[(STAGE_RES, "ts_comp.json")] = json.dumps(
                comp2).encode()
        if mut_comp:
            comp2 = json.loads(store2[(STAGE_RES,
                                       "ts_comp.json")].decode())
            mut_comp(comp2)
            store2[(STAGE_RES, "ts_comp.json")] = json.dumps(
                comp2).encode()
        sm3 = dict(sm, schema="f2g-w2-tier-s-smoke-v1",
                   effect_grids_sha256=_digest(grids),
                   **caps2["smoke_fields"])
        store2[(STAGE_SMOKE, "smoke3.json")] = json.dumps(
            sm3).encode()
        art3 = select_candidates(
            sm3, grids,
            smoke_ref={"commit": STAGE_SMOKE, "path": "smoke3.json"},
            effect_grids_ref=refs2["effect_grids_ref"])

        def rdr(c, p):
            if p.endswith("execution_manifest.json"):
                return json.dumps(fix_man).encode()
            return store2[(c, p)]
        try:
            verify_selector_admission(
                ".", art3, "a" * 40, blob_reader=rdr,
                git_resolve=lambda c: c,
                geometry_loader=geom_loader,
                is_ancestor=ancestor or dag_ancestor)
            return None
        except SelectorRefusal as e:
            return str(e)
    assert re_capsule() is None            # clean capsule passes
    MUTS = [
        (lambda i: i["effect_grids"].update(commit="0" * 40), None,
         None),
        (lambda i: i["implementation"].update(commit="0" * 40),
         None, None),
        (lambda i: i.update(seed_authority_sha256="Z" * 64), None,
         None),
        (lambda i: i["geometry"].update(commit="0" * 40), None,
         None),
        (None, None, lambda c: c.update(fired_utc="2026-08-25")),
        (None, None, lambda c: c.update(
            fired_utc="2026-08-25T12:00:00Z",
            completed_utc="2026-08-25T11:00:00Z")),
        (None, None, lambda c: c.update(
            pre_invocation_sha256="0" * 64)),
        (None, None, lambda c: c.update(
            results_blob_sha256="0" * 64)),
        (None, lambda r: r.update(quality={"R": 40,
                                           "n_draws": 999}), None),
        (None, lambda r: r.update(
            seed_authority_sha256="c" * 64), None),
        (None, lambda r: r["families"]["B2A"][0].update(extra=1),
         None),
        (None, lambda r: r["families"]["B2A"][0].update(
            grid_index=7), None),
        (None, lambda r: r["families"]["B2A"][0].update(
            loco_folds=[{}] * 50), None),
        (None, lambda r: r["families"].update(B5X=[]), None),
        (None, lambda r: r.update(
            families=list(FAMILIES_ORDER)), None),
        (None, lambda r: r["families"]["B2A"][0]["replicates"][0]
            .update(p_values=list(FAMILIES_ORDER)), None),
        (None, lambda r: r["families"]["B1B"][0].update(
            loco_folds=7), None),
        (None, lambda r: r["families"]["B2A"][0]["replicates"][0]
            ["p_values"].update(B2A=0.9), None),
    ]
    for mi, mr, mc_ in MUTS:
        msg = re_capsule(mut_pre=mi, mut_results=mr, mut_comp=mc_)
        assert msg is not None and "SELECTOR_UNADMITTED" in msg, \
            (mi, mr, mc_, msg)
    # lineage edge broken -> refuse
    msg = re_capsule(ancestor=lambda a, b: False)
    assert msg is not None and "STRICT stage ancestry" in msg
    # codex 1328Z item 2: a SAME-COMMIT post-hoc capsule refuses
    msg = re_capsule(ancestor=lambda a, b: True)
    assert msg is None or True   # DAG default governs; explicit:
    same = re_capsule(ancestor=lambda a, b: a == b or True)
    # the real same-commit doctor: collapse every stage commit
    flat_store = dict(astore)
    caps_f = mk_tier_s_capsule(sm["families"], grids, grids_raw,
                               flat_store, "a" * 40, man_pins,
                               "impl.py")
    flat = {}
    for (c, pth), v in flat_store.items():
        flat[("a" * 40, pth)] = v
    sfields = {k: (dict(v, commit="a" * 40)
                   if isinstance(v, dict) and "commit" in v else v)
               for k, v in caps_f["smoke_fields"].items()}
    smf = dict(sm, schema="f2g-w2-tier-s-smoke-v1",
               effect_grids_sha256=_digest(grids), **sfields)
    flat[("a" * 40, "smoke_flat.json")] = json.dumps(smf).encode()
    art_flat = select_candidates(
        smf, grids,
        smoke_ref={"commit": "a" * 40, "path": "smoke_flat.json"},
        effect_grids_ref=refs2["effect_grids_ref"])

    def flat_rdr(c, p):
        if p.endswith("execution_manifest.json"):
            return json.dumps(fix_man).encode()
        return flat[(c, p)]
    try:
        verify_selector_admission(
            ".", art_flat, "a" * 40, blob_reader=flat_rdr,
            git_resolve=lambda c: c, geometry_loader=geom_loader,
            is_ancestor=lambda a, b: True)
        raise AssertionError("same-commit capsule must refuse")
    except SelectorRefusal as e:
        assert "STRICT stage ancestry" in str(e)
    # codex 1328Z item 5: invalid p-values are never evidence
    for bad_p in (-1.0, 1.1, False, float("nan"), float("inf")):
        msg = re_capsule(mut_results=lambda r, b=bad_p:
                         r["families"]["B2A"][0]["replicates"][0]
                         ["p_values"].update(B2A=b))
        assert msg is not None and (
            "not a finite numeric" in msg or "do not DERIVE" in msg
            ), (bad_p, msg)
    msg = re_capsule(mut_results=lambda r:
                     r["families"]["B1B"][0]["loco_folds"][0]
                     .update(S0=False))
    assert msg is not None and ("not a finite numeric" in msg
                                or "do not DERIVE" in msg)
    # forged results whose p-vectors do not derive the smoke refuse
    # (the last mutation above flips one replicate's p-value); the
    # coordinated case -- fabricated smoke + MATCHING fabricated
    # membership -- has no seam left because membership is never read

    # unresolvable lineage -> unadmitted
    def bad_resolve(c):
        raise SelectorRefusal(
            f"SELECTOR_UNADMITTED: commit {c!r} does not resolve "
            "to an admitted 40-hex lineage")
    assert arefuses(art_a, "does not resolve", resolve=bad_resolve)

    # --- codex 1912Z ruling: a registered None INSIDE the four-family
    # p-vector is a NON-REJECTION with m held at 4 (the harness sec 5
    # replicate rule), never a removed hypothesis and never a sentinel.
    # The real class (run tip 03453fdd): every B1B component of the two
    # B2B dropout points came back None (94 of 1,000 B2B p-values);
    # the pre-ruling selector refused the whole chain at _rebuild.
    import w2_power_harness_cayley as _PHt
    _reg = FIX_GEOM["registries"]["cascadia"]
    HAND_PV = {"B2B": 0.015, "B1B": None, "B2A": 0.5, "B3A": 0.7}
    b2b_det = [p for p in grids["B2B"] if "gain" not in p]
    B2B_DROP = [k for k, p in enumerate(b2b_det)
                if p.get("dropout", 0.0) > 0]
    assert B2B_DROP == [2, 3], B2B_DROP
    # sm B2B counts [10, 30, 30, 20, 40]: k=3's first non-recovery
    # replicate is index 20, k=2's is 30
    first_false = {k: sum(1 for o in sm["families"]["B2B"][k]["outcomes"]
                          if o) for k in B2B_DROP}
    assert first_false == {2: 30, 3: 20}, first_false

    def none_pv(fam, k, r_i, pv):
        if fam != "B2B" or k not in B2B_DROP:
            return pv
        pv = dict(pv)
        if k == 3:
            # the dropout-0.25 class: B1B untestable on EVERY replicate
            pv["B1B"] = None
            if r_i == first_false[3]:
                # own-family untestable on a non-recovery replicate
                pv["B2B"] = None
            elif r_i == first_false[3] + 1:
                # the codex hand case on a non-recovery replicate
                pv = dict(HAND_PV)
        elif r_i % 2 == 0:
            # the dropout-0.1 class: B1B untestable on many replicates
            pv["B1B"] = None
        return pv

    store_n = dict(astore)
    caps_n = mk_tier_s_capsule(sm["families"], grids, grids_raw,
                               store_n, "a" * 40, man_pins, "impl.py",
                               pv_fn=none_pv)
    sm_n = dict(sm, schema="f2g-w2-tier-s-smoke-v1",
                effect_grids_sha256=_digest(grids),
                **caps_n["smoke_fields"])
    store_n[(STAGE_SMOKE, "smoke_none.json")] = json.dumps(sm_n).encode()
    art_n = select_candidates(
        sm_n, grids,
        smoke_ref={"commit": STAGE_SMOKE, "path": "smoke_none.json"},
        effect_grids_ref=refs2["effect_grids_ref"])

    def rdr_n(c, p):
        if p.endswith("execution_manifest.json"):
            return json.dumps(fix_man).encode()
        return store_n[(c, p)]

    def admit_n():
        return verify_selector_admission(
            ".", art_n, "a" * 40, blob_reader=rdr_n,
            git_resolve=lambda c: c, geometry_loader=geom_loader,
            is_ancestor=dag_ancestor)
    res_n = json.loads(store_n[(STAGE_RES, "ts_results.json")].decode())
    # anti-vacuity: the fixture CARRIES registered None -- in the results
    # capsule AND in the committed point carriers, identically. k=3: 50
    # B1B + 1 own-family; k=2: 25 B1B -> 76 None values.
    n_none = sum(1 for e in res_n["families"]["B2B"]
                 for rep in e["replicates"]
                 for v in rep["p_values"].values() if v is None)
    car_none = 0
    for idx_c in range(3, 8):    # B2B carriers follow B2A's three
        cap_c = json.loads(store_n[(STAGE_PTS,
                                    f"smoke_point_{idx_c:03d}.json")]
                           .decode())
        assert cap_c["family"] == "B2B", cap_c["family"]
        car_none += sum(1 for rep in cap_c["record"]["replicates"]
                        for v in rep["p_values"].values() if v is None)
    assert n_none == car_none == 76, (n_none, car_none)
    assert res_n["families"]["B2B"][3]["replicates"][20]["p_values"] \
        ["B2B"] is None
    assert res_n["families"]["B2B"][3]["replicates"][21]["p_values"] \
        == HAND_PV
    # (a) the None chain is ADMITTED, and its rebuilt outcomes equal the
    # harness's holm_rejects with the None passed through, replicate by
    # replicate -- which is exactly the smoke the fixture declares
    adm_n = admit_n()
    assert isinstance(adm_n, dict) and adm_n["point_corpus"]["carriers"] \
        == sum(1 for (c, _p) in store_n if c == STAGE_PTS) == 29
    assert adm_n["smoke"]["pre_invocation_sha256"] == \
        caps_n["pre"]["invocation_sha256"]
    for k, e in enumerate(res_n["families"]["B2B"]):
        pre_o, post_o = _rebuild_outcomes("B2B", e, _PHt.holm_rejects,
                                          _reg)
        assert post_o is None
        assert pre_o == ["B2B" in _PHt.holm_rejects(rep["p_values"])
                         for rep in e["replicates"]], k
        assert pre_o == sm["families"]["B2B"][k]["outcomes"], k
    # (c) own-family None -> that replicate is False; the chain admitted
    pre_3, _ = _rebuild_outcomes("B2B", res_n["families"]["B2B"][3],
                                 _PHt.holm_rejects, _reg)
    assert pre_3[20] is False and pre_3[21] is False and \
        pre_3[:20] == [True] * 20, pre_3
    # (b) the hand case: own p 0.015 with ONE None component is NOT a
    # recovery under m=4 (0.015 > 0.05/4 = 0.0125) ...
    assert "B2B" not in _PHt.holm_rejects(HAND_PV)

    def holm_reduced(pv):
        """the REJECTED option 1: None removed from the Holm set."""
        known = sorted((h for h in pv if pv[h] is not None),
                       key=lambda h: pv[h])
        m = len(known)
        rej, still = set(), True
        for i, h in enumerate(known):
            if still and pv[h] <= 0.05 / (m - i):
                rej.add(h)
            else:
                still = False
        return rej
    # ... and a reduced-set implementation (m=3, 0.015 <= 0.05/3) WOULD
    # have rejected it: the control discriminates, at the rebuild seam
    assert "B2B" in holm_reduced(HAND_PV)
    hand_entry = {"replicates": [{"p_values": dict(HAND_PV)}],
                  "loco_folds": None}
    assert _rebuild_outcomes("B2B", hand_entry, _PHt.holm_rejects,
                             _reg) == ([False], None)
    assert _rebuild_outcomes("B2B", hand_entry, holm_reduced,
                             _reg) == ([True], None)
    # ... and at the CHAIN: with the harness rule swapped for the
    # reduced set, the admitted None chain must REFUSE (replicate 21 of
    # k=3 would flip True against the declared False)
    _true_holm = _PHt.holm_rejects
    _PHt.holm_rejects = holm_reduced
    try:
        try:
            admit_n()
            raise AssertionError(
                "a reduced-set Holm must be DETECTED by the None chain")
        except SelectorRefusal as e:
            assert "do not DERIVE" in str(e), str(e)
    finally:
        _PHt.holm_rejects = _true_holm
    assert _PHt.holm_rejects is _true_holm
    assert isinstance(admit_n(), dict)   # restored: admits again
    # (c') B1B with folds: an own-family None replicate is False in BOTH
    # pre and post; a numeric neighbour is unaffected
    fold_ok = {st: 0.001 for st in _reg}
    b1b_none = {"replicates": [
        {"p_values": {"B1B": None, "B2A": 0.9, "B2B": 0.9, "B3A": 0.9}},
        {"p_values": {"B1B": 0.001, "B2A": 0.9, "B2B": 0.9,
                      "B3A": 0.9}}],
        "loco_folds": [dict(fold_ok), dict(fold_ok)]}
    assert _rebuild_outcomes("B1B", b1b_none, _PHt.holm_rejects,
                             _reg) == ([False, True], [False, True])
    assert _rebuild_outcomes("B1B", dict(b1b_none, loco_folds=None),
                             _PHt.holm_rejects, _reg) == \
        ([False, True], None)
    # (c'') seam ORDERING (post-1912Z verifier finding 1): a B1B
    # replicate whose OWN p is None must not carry a malformed fold map
    # through the seam silently while the identical defect on a numeric
    # replicate refuses -- the fold-map checks run BEFORE the own-None
    # short-circuit. Six partners refuse typed on BOTH replicate kinds
    # (the same needle each; both halves of the coverage check and a
    # non-dict fold map are among them), the positive twins are
    # unchanged, and two mutant copies of THIS module's source compiled
    # in memory prove the partners have teeth: the pre-fix order (own-
    # None short-circuit moved back in front of the fold checks) makes
    # every partner return silently, and a coverage check weakened from
    # != to < lets the longer-than-replicates partner through silently
    # while the shorter one still refuses.
    import sys as _sys
    import types as _types
    _self_mod = _sys.modules[__name__]
    own_none_pv = {"B1B": None, "B2A": 0.9, "B2B": 0.9, "B3A": 0.9}
    numeric_pv = {"B1B": 0.001, "B2A": 0.9, "B2B": 0.9, "B3A": 0.9}
    NEEDLE_COVER = "LOCO fold maps do not cover every replicate"
    NEEDLE_REG = "LOCO fold set diverges from the bound registry"
    NEEDLE_NUM = "not a finite numeric"
    SEAM_PARTNERS = (
        ("bogus registry", 1, [{"bogus_station": 0.001}], NEEDLE_REG),
        ("fold p 'x'", 1, [{st: "x" for st in _reg}], NEEDLE_NUM),
        ("folds shorter than replicates", 2, [dict(fold_ok)],
         NEEDLE_COVER),
        ("folds longer than replicates", 1, [dict(fold_ok), dict(fold_ok)],
         NEEDLE_COVER),
        ("folds not a list", 1, {0: dict(fold_ok)}, NEEDLE_COVER),
        ("fold entry a list of the station names", 1, [sorted(_reg)],
         NEEDLE_REG),
    )

    def seam(mod, pv, n_reps, folds):
        """(refusal message | None, output | None) at mod's seam for
        n_reps replicates that all carry pv."""
        try:
            return None, mod._rebuild_outcomes(
                "B1B", {"replicates": [{"p_values": dict(pv)}
                                       for _ in range(n_reps)],
                        "loco_folds": folds}, _PHt.holm_rejects, _reg)
        except mod.SelectorRefusal as e:
            return str(e), None
        except Exception as e:                           # noqa: BLE001
            # an UNTYPED escape is a defect the partner must see as a
            # failed needle, not as a crashed selftest
            return "EXC " + repr(e), None
    # every partner must refuse TYPED (a SelectorRefusal, never an
    # untyped escape that merely carries the needle text) on BOTH
    # replicate kinds -- the EXC prefix is asserted absent
    for label, n_reps, folds, needle in SEAM_PARTNERS:
        msg_o, _out = seam(_self_mod, own_none_pv, n_reps, folds)
        assert msg_o is not None and not msg_o.startswith("EXC ") and \
            "SELECTOR_UNADMITTED" in msg_o and needle in msg_o, \
            (label, msg_o)
        msg_n, _out = seam(_self_mod, numeric_pv, n_reps, folds)
        assert msg_n is not None and not msg_n.startswith("EXC ") and \
            "SELECTOR_UNADMITTED" in msg_n and needle in msg_n, \
            (label, msg_n)
    # positive twins: own-None with a VALID fold map, and with a None
    # fold entry, stay ([False], [False]); the numeric twin stays
    # ([True], [True]) / ([True], [False])
    assert seam(_self_mod, own_none_pv, 1, [dict(fold_ok)]) == \
        (None, ([False], [False]))
    assert seam(_self_mod, own_none_pv, 1, [None]) == \
        (None, ([False], [False]))
    assert seam(_self_mod, numeric_pv, 1, [dict(fold_ok)]) == \
        (None, ([True], [True]))
    assert seam(_self_mod, numeric_pv, 1, [None]) == \
        (None, ([True], [False]))
    # the anti-vacuity mutant: this module's own source with the fold
    # block and the own-None short-circuit swapped back into the
    # pre-fix order, compiled IN MEMORY (nothing written into the
    # tree) and run against the same partners
    with open(__file__, "rb") as f:
        _src = f.read().replace(b"\r\n", b"\n").decode("utf-8")
    _A = ("        # cayley successor of the 1912Z rule (verifier LOW "
          "finding 1,\n")
    _B = "        if pv[fam] is None:\n"
    _E = "        rej = holm_fn(pv)\n        pre.append(fam in rej)\n"
    assert _src.count(_A) == 1 and _src.count(_B) == 1 and \
        _src.count(_E) == 1, (_src.count(_A), _src.count(_B),
                              _src.count(_E))
    _ia, _ib, _ie = _src.index(_A), _src.index(_B), _src.index(_E)
    assert _ia < _ib < _ie
    _mut_src = _src[:_ia] + _src[_ib:_ie] + _src[_ia:_ib] + _src[_ie:]
    assert _mut_src != _src and len(_mut_src) == len(_src) and \
        _mut_src.index(_B) < _mut_src.index(_A) < _mut_src.index(_E)
    _mut = _types.ModuleType("w2_tier_selector_cayley_seam_mutant")
    _mut.__file__ = "<seam-order mutant, in memory>"
    exec(compile(_mut_src, "<seam-order-mutant>", "exec"), _mut.__dict__)
    assert _mut._rebuild_outcomes is not _rebuild_outcomes
    for label, n_reps, folds, needle in SEAM_PARTNERS:
        msg_m, out_m = seam(_mut, own_none_pv, n_reps, folds)
        assert msg_m is None and out_m == ([False] * n_reps,
                                           [False] * n_reps), \
            (label, msg_m, out_m)   # RED under the old order
    # the mutant IS the old order and not a broken module: the numeric
    # defect still refuses there, and the positive twin agrees
    msg_m, _out = seam(_mut, numeric_pv, 1, [{"bogus_station": 0.001}])
    assert msg_m is not None and not msg_m.startswith("EXC ") and \
        NEEDLE_REG in msg_m, msg_m
    msg_m, _out = seam(_mut, numeric_pv, 1, [{st: "x" for st in _reg}])
    assert msg_m is not None and not msg_m.startswith("EXC ") and \
        NEEDLE_NUM in msg_m, msg_m
    assert seam(_mut, own_none_pv, 1, [dict(fold_ok)]) == \
        (None, ([False], [False]))
    assert seam(_mut, numeric_pv, 1, [dict(fold_ok)]) == \
        (None, ([True], [True]))
    # second mutant: the coverage check weakened from != to < (a longer
    # fold list would pass the seam silently). The longer-than-
    # replicates partner must go RED on BOTH replicate kinds under it,
    # the shorter-than-replicates partner must still refuse, and the
    # non-dict fold-map partner must still refuse typed.
    _L = ("            if not isinstance(folds, list) or "
          "len(folds) != len(reps):\n")
    _L2 = _L.replace("len(folds) != len(reps)", "len(folds) < len(reps)")
    assert _src.count(_L) == 1 and _src.count(_L2) == 0
    _mut2_src = _src.replace(_L, _L2)
    _mut2 = _types.ModuleType("w2_tier_selector_cayley_coverage_mutant")
    _mut2.__file__ = "<coverage-lt mutant, in memory>"
    exec(compile(_mut2_src, "<coverage-lt-mutant>", "exec"),
         _mut2.__dict__)
    assert _mut2._rebuild_outcomes is not _rebuild_outcomes
    _longer = [dict(fold_ok), dict(fold_ok)]
    assert seam(_mut2, own_none_pv, 1, _longer) == \
        (None, ([False], [False]))          # RED: passes silently
    assert seam(_mut2, numeric_pv, 1, _longer) == \
        (None, ([True], [True]))            # RED: passes silently
    msg_m, _out = seam(_mut2, own_none_pv, 2, [dict(fold_ok)])
    assert msg_m is not None and not msg_m.startswith("EXC ") and \
        NEEDLE_COVER in msg_m, msg_m
    msg_m, _out = seam(_mut2, own_none_pv, 1, [sorted(_reg)])
    assert msg_m is not None and not msg_m.startswith("EXC ") and \
        NEEDLE_REG in msg_m, msg_m
    del _mut2, _mut2_src
    del _mut, _mut_src, _src
    # (c''') the sibling refusal sites of the same region, driven TYPED
    # (fourth-round verifier: four sites had zero selftest hits, so a
    # typed->untyped mutation there was invisible; and the refusal
    # class identity was pinned nowhere, so aliasing SelectorRefusal to
    # ValueError passed every by-name check). The class is pinned by
    # identity and each site is asserted to raise EXACTLY
    # SelectorRefusal with its needle.
    assert SelectorRefusal is not ValueError and \
        SelectorRefusal.__bases__ == (ValueError,) and \
        issubclass(SelectorRefusal, ValueError)

    def typed(call):
        """('TYPED' | 'SUB' | 'EXC' | None, message) for a direct call:
        TYPED only when the raised class IS SelectorRefusal."""
        try:
            call()
        except SelectorRefusal as e:
            return ("TYPED" if type(e) is SelectorRefusal else "SUB"), \
                str(e)
        except Exception as e:                           # noqa: BLE001
            return "EXC", repr(e)
        return None, None
    _four = {"B1B": 0.5, "B2A": 0.5, "B2B": 0.5, "B3A": 0.5}
    SIBLING_SITES = (
        ("p None where a value is required",
         lambda: _valid_p(None, "B2A p-value"),
         "B2A p-value is None where a value is required"),
        ("results replicates missing",
         lambda: _rebuild_outcomes("B2B", {"replicates": None},
                                   _PHt.holm_rejects, _reg),
         "results replicates missing"),
        ("result replicate schema not closed",
         lambda: _rebuild_outcomes(
             "B2B", {"replicates": [{"p_values": dict(_four),
                                     "extra": 1}]},
             _PHt.holm_rejects, _reg),
         "result replicate schema not closed"),
        ("replicate p-vector not the four families",
         lambda: _rebuild_outcomes(
             "B2B", {"replicates": [{"p_values": {"B1B": 0.5, "B2A": 0.5,
                                                  "B2B": 0.5}}]},
             _PHt.holm_rejects, _reg),
         "replicate p-vector not the four families"),
    )
    for label, call, needle in SIBLING_SITES:
        kind, msg = typed(call)
        assert kind == "TYPED" and "SELECTOR_UNADMITTED" in msg and \
            needle in msg, (label, kind, msg)
    # the same helper on the four-family positive control returns None
    assert typed(lambda: _rebuild_outcomes(
        "B2B", {"replicates": [{"p_values": dict(_four)}]},
        _PHt.holm_rejects, _reg)) == (None, None)
    # (d) invalid classes in a component each refuse TYPED -- at the
    # rebuild seam naming the component, and through the chain
    for bad in (True, "0.5", float("nan"), float("inf"), -1.0, 1.1):
        try:
            _rebuild_outcomes(
                "B2B", {"replicates": [{"p_values": dict(HAND_PV,
                                                         B1B=bad)}],
                        "loco_folds": None}, _PHt.holm_rejects, _reg)
            raise AssertionError(f"{bad!r} must refuse")
        except SelectorRefusal as e:
            assert "B1B p-value" in str(e) and \
                "not a finite numeric" in str(e), (bad, str(e))
        msg = re_capsule(mut_results=lambda r, b=bad:
                         r["families"]["B2B"][0]["replicates"][0]
                         ["p_values"].update(B1B=b))
        assert msg is not None and "SELECTOR_UNADMITTED" in msg and \
            "not a finite numeric" in msg, (bad, msg)
    # (e) a spy holm_fn: the None reaches Holm AS None inside the FULL
    # four-key mapping -- no sentinel, no shrunk family, also through
    # the B1B fold substitution
    seen = []

    def spy(pv):
        seen.append(dict(pv))
        return _PHt.holm_rejects(pv)
    assert _rebuild_outcomes("B2B", hand_entry, spy, _reg) == \
        ([False], None)
    assert seen == [HAND_PV] and seen[0]["B1B"] is None and \
        len(seen[0]) == 4 and not any(v == 1.0 for v in seen[0].values())
    seen[:] = []
    e_b1b = {"replicates": [{"p_values": {"B1B": 0.001, "B2A": None,
                                          "B2B": 0.9, "B3A": 0.9}}],
             "loco_folds": [dict(fold_ok)]}
    assert _rebuild_outcomes("B1B", e_b1b, spy, _reg) == ([True], [True])
    assert len(seen) == 1 + len(_reg) and \
        all(d["B2A"] is None and len(d) == 4 for d in seen)
    # (f) the all-numeric chain is UNCHANGED: every fixture entry
    # rebuilds byte-equal under the pre-ruling rule (allow_none=False,
    # reimplemented here verbatim) and the new one; the clean capsule
    # still admits

    def old_rebuild(fam, entry, holm_fn):
        pre_o, post_o = [], []
        folds = entry.get("loco_folds")
        for r_i, rep in enumerate(entry["replicates"]):
            pv = rep["p_values"]
            for f, v in pv.items():
                _valid_p(v, f"{f} p-value")
            rej = holm_fn(pv)
            pre_o.append(fam in rej)
            if fam == "B1B" and folds is not None:
                fr = folds[r_i]
                if fr is None:
                    post_o.append(False)
                    continue
                ok = "B1B" in rej
                for st in sorted(fr):
                    p_s = fr[st]
                    _valid_p(p_s, f"loco:{st} fold p-value",
                             allow_none=True)
                    if p_s is None or "B1B" not in holm_fn(
                            dict(pv, B1B=p_s)):
                        ok = False
                post_o.append(ok)
        return pre_o, (post_o if fam == "B1B" and folds is not None
                       else None)
    res_num = json.loads(astore[(STAGE_RES, "ts_results.json")].decode())
    n_entries = 0
    for fam_x in FAMILIES_ORDER:
        for e in res_num["families"][fam_x]:
            assert all(v is not None for rep in e["replicates"]
                       for v in rep["p_values"].values())
            new_o = _rebuild_outcomes(fam_x, e, _PHt.holm_rejects, _reg)
            assert json.dumps(new_o) == json.dumps(
                old_rebuild(fam_x, e, _PHt.holm_rejects)), \
                (fam_x, e["grid_index"])
            n_entries += 1
    assert n_entries == 21, n_entries
    assert re_capsule() is None

    print("w2_tier_selector selftest: ALL PASS (hand fixtures incl. the "
          "codex 1912Z None-rule chain; PRELIMINARY_SMOKE semantics; "
          "nothing certified)")


if __name__ == "__main__":
    _selftest()
