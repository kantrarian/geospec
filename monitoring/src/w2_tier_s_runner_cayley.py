#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 TIER-S PRODUCTION RUNNER (cayley) -- codex 0432Z item 1:
the real Tier-S writer with PRE-FIRE chronology. Nothing here is
post-hoc: the closed pre-invocation publishes ATOMICALLY CREATE-ONCE
BEFORE any worker starts and binds the admitted manifest, the exact
effect-grid/geometry/implementation pins, the full detection-grid
order, seed authority, quality, and the intended output root. Every
worker authenticates that pre-fire digest before touching anything.
Per-point result capsules publish create-once; a SEPARATE closed
completion receipt binds the pre-fire digest + the exact results blob
+ ordered canonical times; a DETERMINISTIC aggregator alone emits the
results/smoke/selector carriers. The verifier chain is
manifest -> pre-invocation -> results/completion -> smoke ->
selector, each stage its own commit.

Phases (commits between them are the operator's, so the chain commits
are chronologically real):
  fire_pre        -> commit -> phase1 (all detection points)
  rank stage-1    -> phase2 (B1B top-8 LOCO folds)
  completion      -> commit -> aggregate (results/smoke/selector)

Tier-S output is PRELIMINARY_SMOKE; nothing here certifies anything
or opens any window-2 value. Selftest uses stub smoke functions only.
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

from w2_cert_runner_cayley import (_canon, _digest, _publish_once,
                                   resolve_manifest_commit,
                                   RunnerRefusal)

PRE_SCHEMA = "f2g-w2-tier-s-pre-invocation-v1"
PRE_FIELDS = {"schema", "manifest_commit", "effect_grids",
              "effect_grids_content_sha256",
              "geometry", "quality", "seed_authority_sha256",
              "implementation", "grid_order_sha256", "output_root",
              "argv", "host", "interpreter", "fired_utc",
              "invocation_sha256"}
COMPLETION_SCHEMA = "f2g-w2-tier-s-completion-v1"
COMPLETION_FIELDS = {"schema", "pre_invocation_sha256",
                     "results_blob_sha256", "fired_utc",
                     "completed_utc"}


def _pin_for(manifest, path):
    for slot in manifest.get("slots", {}).values():
        for pin in slot.get("pins", ()) or ():
            if isinstance(pin, dict) and pin.get("path") == path:
                return pin
    raise RunnerRefusal(
        f"RUNNER_TIER_S_UNADMITTED: {path} is not a manifest pin")


def fire_pre(repo, manifest_commit, grids_path, geometry_path,
             impl_path, outdir, *, blob_reader=None, argv=None,
             resolver=None):
    """Phase 0: publish the closed pre-invocation CREATE-ONCE before
    anything else exists. Identities come from the ADMITTED manifest
    pins, never from caller values."""
    mc_full = (resolver or resolve_manifest_commit)(
        repo, manifest_commit)
    if blob_reader is None:
        import subprocess

        def blob_reader(commit, path):
            p = subprocess.run(["git", "-C", repo, "cat-file",
                                "blob", f"{commit}:{path}"],
                               capture_output=True)
            if p.returncode != 0:
                raise RunnerRefusal(
                    f"RUNNER_TIER_S_UNADMITTED: {path} unreadable "
                    f"at {commit}")
            return p.stdout
    man = json.loads(blob_reader(
        mc_full, "docs/f2g_window2_execution/"
                 "execution_manifest.json").decode("utf-8"))
    g_pin = _pin_for(man, grids_path)
    geo_pin = _pin_for(man, geometry_path)
    impl_pin = _pin_for(man, impl_path)
    grids_art = json.loads(blob_reader(
        g_pin["commit"], grids_path).decode("utf-8"))
    grids = grids_art.get("grids", grids_art)
    geo_cap = json.loads(blob_reader(
        geo_pin["commit"], geometry_path).decode("utf-8"))
    det_order = {f: [p for p in grids[f] if "gain" not in p]
                 for f in ("B2A", "B2B", "B1B", "B3A")}
    points = [(fam, p) for fam in ("B2A", "B2B", "B1B", "B3A")
              for p in det_order[fam]]
    pre = {"schema": PRE_SCHEMA,
           "manifest_commit": mc_full,
           "effect_grids": {"commit": g_pin["commit"],
                            "path": grids_path,
                            "blob_sha256": g_pin["blob_sha256"]},
           # codex 1328Z item 1: the canonical PARSED-grid digest,
           # derived from the reopened manifest-pinned bytes -- the
           # smoke inherits it, never a caller value
           "effect_grids_content_sha256": _digest(grids),
           "geometry": {"commit": geo_pin["commit"],
                        "path": geometry_path,
                        "capsule_digest":
                            geo_cap.get("capsule_digest")},
           "quality": {"R": 50, "n_draws": 999},
           "seed_authority_sha256":
               geo_cap.get("seed_authority_sha256"),
           "implementation": {"commit": impl_pin["commit"],
                              "path": impl_path,
                              "blob_sha256":
                                  impl_pin["blob_sha256"]},
           "grid_order_sha256": _digest(det_order),
           "output_root": str(outdir),
           "argv": list(argv if argv is not None else sys.argv),
           "host": platform.node(),
           "interpreter": {"executable": sys.executable,
                           "version": sys.version},
           "fired_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                      time.gmtime())}
    pre["invocation_sha256"] = _digest(
        {k: v for k, v in pre.items() if k != "invocation_sha256"})
    _publish_once(os.path.join(outdir, "tier_s_pre_invocation.json"),
                  json.dumps(pre, indent=1, sort_keys=True,
                             allow_nan=False) + "\n")
    return pre, points


def derive_points(pre, blob_reader):
    """codex 1328Z item 4: the canonical ordered detection points
    derive ONLY from the pre-bound grid identity -- the reopened blob
    must match the pre's recorded blob AND content digests; no phase
    ever accepts a caller points list."""
    g = pre["effect_grids"]
    raw = blob_reader(g["commit"], g["path"])
    if hashlib.sha256(raw).hexdigest() != g["blob_sha256"]:
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: grid bytes diverge from the "
            "pre-bound identity")
    grids_art = json.loads(raw.decode("utf-8"))
    grids = grids_art.get("grids", grids_art)
    if _digest(grids) != pre["effect_grids_content_sha256"]:
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: grid content diverges from "
            "the pre-bound canonical digest")
    det_order = {f: [p for p in grids[f] if "gain" not in p]
                 for f in ("B2A", "B2B", "B1B", "B3A")}
    if pre["grid_order_sha256"] != _digest(det_order):
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: grid order diverges from the "
            "pre-bound digest")
    return [(fam, p) for fam in ("B2A", "B2B", "B1B", "B3A")
            for p in det_order[fam]]


def _check_outdir(pre, outdir):
    if os.path.abspath(outdir) != os.path.abspath(
            pre["output_root"]):
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: outdir diverges from the "
            "pre-bound output root")


def _load_pre(outdir, expected_sha):
    with open(os.path.join(outdir, "tier_s_pre_invocation.json"),
              encoding="utf-8") as f:
        pre = json.load(f)
    got = _digest({k: v for k, v in pre.items()
                   if k != "invocation_sha256"})
    if got != pre.get("invocation_sha256") or got != expected_sha:
        raise RunnerRefusal(
            "RUNNER_TIER_S_PRE_DIGEST_MISMATCH: the pre-invocation "
            "carrier diverges from the fired digest")
    return pre


def run_smoke_point(repo, outdir, idx, expected_pre_sha,
                    blob_reader, with_loco=False, smoke_fn=None):
    """ONE detection point, whole-point, authenticated against the
    PRE-FIRE digest before any work; the point identity derives from
    the PRE-BOUND grid, never a caller list (codex 1328Z item 4)."""
    pre = _load_pre(outdir, expected_pre_sha)
    _check_outdir(pre, outdir)
    points = derive_points(pre, blob_reader)
    if not 0 <= int(idx) < len(points):
        raise RunnerRefusal(
            f"RUNNER_TIER_S_UNADMITTED: point index {idx} outside "
            "the derived grid")
    fam, point = points[int(idx)]
    if smoke_fn is None:
        import w2_power_harness_cayley as PH

        def smoke_fn(f, p, folds):
            return PH.run_point_smoke(
                repo, {"manifest_commit": pre["manifest_commit"],
                       "path": pre["geometry"]["path"]},
                f, p, with_loco_folds=folds)
    rec = smoke_fn(fam, point, with_loco)
    out = {"index": int(idx), "family": fam, "point": point,
           "pre_invocation_sha256": pre["invocation_sha256"],
           "record": rec}
    name = (f"smoke_loco_{idx:03d}.json" if with_loco
            else f"smoke_point_{idx:03d}.json")
    _publish_once(os.path.join(outdir, name),
                  json.dumps(out, indent=1, sort_keys=True,
                             allow_nan=False) + "\n")
    return out


def rank_stage1_b1b(outdir, expected_pre_sha, blob_reader):
    """Deterministic stage-1 ranking of the B1B detection points by
    (-pre-LOCO count, registered grid index) from the published
    phase-1 capsules; the point list derives from the PRE."""
    import w2_power_harness_cayley as PH
    pre = _load_pre(outdir, expected_pre_sha)
    _check_outdir(pre, outdir)
    points = derive_points(pre, blob_reader)
    counts = []
    for idx, (fam, point) in enumerate(points):
        if fam != "B1B":
            continue
        p = os.path.join(outdir, f"smoke_point_{idx:03d}.json")
        if not os.path.exists(p):
            raise RunnerRefusal(
                f"RUNNER_TIER_S_RESULT_MISSING: point {idx}")
        with open(p, encoding="utf-8") as f:
            cap = json.load(f)
        if cap.get("pre_invocation_sha256") != expected_pre_sha:
            raise RunnerRefusal(
                f"RUNNER_TIER_S_PRE_DIGEST_MISMATCH: point {idx}")
        c = sum(1 for rep in cap["record"]["replicates"]
                if "B1B" in PH.holm_rejects(rep["p_values"]))
        counts.append((idx, c))
    order = sorted(counts, key=lambda t: (-t[1], t[0]))
    return [idx for idx, _ in order[:8]]


def write_completion(outdir, pre, results_blob_sha256):
    comp = {"schema": COMPLETION_SCHEMA,
            "pre_invocation_sha256": pre["invocation_sha256"],
            "results_blob_sha256": str(results_blob_sha256),
            "fired_utc": pre["fired_utc"],
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                           time.gmtime())}
    _publish_once(os.path.join(outdir, "tier_s_completion.json"),
                  json.dumps(comp, indent=1, sort_keys=True) + "\n")
    return comp


def _check_point_capsule(cap, idx, fam, point, pre):
    """codex 1328Z item 4: the per-point capsule is CLOSED and its
    identities equal the DERIVED point + the pre -- a relabelled
    p-vector has no seam."""
    if not isinstance(cap, dict) or set(cap) != {
            "index", "family", "point", "pre_invocation_sha256",
            "record"}:
        raise RunnerRefusal(
            f"RUNNER_TIER_S_UNADMITTED: point capsule {idx} schema "
            "not closed")
    if cap["index"] != idx or cap["family"] != fam or \
            cap["point"] != point:
        raise RunnerRefusal(
            f"RUNNER_TIER_S_UNADMITTED: point capsule {idx} "
            "identity diverges from the derived grid point")
    if cap["pre_invocation_sha256"] != pre["invocation_sha256"]:
        raise RunnerRefusal(
            f"RUNNER_TIER_S_PRE_DIGEST_MISMATCH: point {idx}")
    rec = cap.get("record")
    if not isinstance(rec, dict):
        raise RunnerRefusal(
            f"RUNNER_TIER_S_UNADMITTED: point {idx} record absent")
    if "family" in rec and (rec.get("family") != fam or
                            rec.get("point") != point):
        raise RunnerRefusal(
            f"RUNNER_TIER_S_UNADMITTED: point {idx} record "
            "family/point diverges from the derived grid point")
    if "quality" in rec and rec["quality"] != pre["quality"]:
        raise RunnerRefusal(
            f"RUNNER_TIER_S_UNADMITTED: point {idx} record quality "
            "diverges from the pre")
    if "geometry_capsule_digest" in rec and \
            rec["geometry_capsule_digest"] != \
            pre["geometry"]["capsule_digest"]:
        raise RunnerRefusal(
            f"RUNNER_TIER_S_UNADMITTED: point {idx} record geometry "
            "diverges from the pre")
    if "seed_authority_sha256" in rec and \
            rec["seed_authority_sha256"] != \
            pre["seed_authority_sha256"]:
        raise RunnerRefusal(
            f"RUNNER_TIER_S_UNADMITTED: point {idx} record seed "
            "diverges from the pre")
    if rec.get("certifiable", False) is not False:
        raise RunnerRefusal(
            f"RUNNER_TIER_S_UNADMITTED: point {idx} record claims "
            "certifiability")


def aggregate(repo, outdir, expected_pre_sha, top8, loco_registry,
              blob_reader):
    """The DETERMINISTIC aggregator: per-point capsules -> the closed
    derivational results v2 -> completion -> smoke; the selector runs
    separately over the COMMITTED carriers. Missing results refuse;
    nothing is recomputed from caller state."""
    pre = _load_pre(outdir, expected_pre_sha)
    _check_outdir(pre, outdir)
    points = derive_points(pre, blob_reader)
    fams = {"B2A": [], "B2B": [], "B1B": [], "B3A": []}
    for idx, (fam, point) in enumerate(points):
        p = os.path.join(outdir, f"smoke_point_{idx:03d}.json")
        if not os.path.exists(p):
            raise RunnerRefusal(
                f"RUNNER_TIER_S_RESULT_MISSING: point {idx}")
        with open(p, encoding="utf-8") as f:
            cap = json.load(f)
        _check_point_capsule(cap, idx, fam, point, pre)
        folds = None
        if fam == "B1B" and idx in top8:
            lp = os.path.join(outdir, f"smoke_loco_{idx:03d}.json")
            if not os.path.exists(lp):
                raise RunnerRefusal(
                    f"RUNNER_TIER_S_RESULT_MISSING: loco {idx}")
            with open(lp, encoding="utf-8") as f:
                lcap = json.load(f)
            _check_point_capsule(lcap, idx, fam, point, pre)
            folds = lcap["record"]["loco_folds"]
        fams[fam].append({
            "point": dict(point),
            "grid_index": len(fams[fam]) if False else None,
            "replicates": cap["record"]["replicates"],
            "loco_folds": folds})
    # registered grid indices: position within the FULL family grid
    # (detection order == grid order for detection points)
    import w2_power_harness_cayley as PH
    import w2_tier_selector_cayley as TS
    counter = {f: 0 for f in fams}
    for fam in fams:
        for e in fams[fam]:
            e["grid_index"] = counter[fam]
            counter[fam] += 1
    results = {"schema": TS.TIER_S_RESULTS_SCHEMA,
               "quality": dict(pre["quality"]),
               "seed_authority_sha256":
                   pre["seed_authority_sha256"],
               "geometry_capsule_digest":
                   pre["geometry"]["capsule_digest"],
               "implementation": dict(pre["implementation"]),
               "families": fams}
    r_body = json.dumps(results, indent=1, sort_keys=True,
                        allow_nan=False) + "\n"
    _publish_once(os.path.join(outdir, "tier_s_results.json"),
                  r_body)
    r_sha = hashlib.sha256(r_body.encode()).hexdigest()
    comp = write_completion(outdir, pre, r_sha)
    smoke_fams = {}
    for fam, entries in fams.items():
        smoke_fams[fam] = []
        for e in entries:
            pre_out = [fam in PH.holm_rejects(rep["p_values"])
                       for rep in e["replicates"]]
            entry = {"point": dict(e["point"]),
                     "outcomes": pre_out}
            if fam == "B1B":
                if e["loco_folds"] is None:
                    entry["post_loco_outcomes"] = None
                else:
                    post = []
                    for rep, fr in zip(e["replicates"],
                                       e["loco_folds"]):
                        rej = PH.holm_rejects(rep["p_values"])
                        ok = "B1B" in rej
                        if sorted(fr) != sorted(loco_registry):
                            raise RunnerRefusal(
                                "RUNNER_TIER_S_FOLD_SET_INVALID")
                        for st in sorted(fr):
                            if fr[st] is None or "B1B" not in \
                                    PH.holm_rejects(dict(
                                        rep["p_values"],
                                        B1B=fr[st])):
                                ok = False
                        post.append(ok)
                    entry["post_loco_outcomes"] = post
            smoke_fams[fam].append(entry)
    smoke = {"schema": "f2g-w2-tier-s-smoke-v1",
             "quality": dict(pre["quality"]),
             "geometry_capsule_digest":
                 pre["geometry"]["capsule_digest"],
             # codex 1328Z item 1: inherited from the PRE, which
             # derived it from the reopened manifest-pinned bytes
             "effect_grids_sha256":
                 pre["effect_grids_content_sha256"],
             "pre_invocation_sha256": pre["invocation_sha256"],
             "completion_sha256": _digest(comp),
             "results_blob_sha256": r_sha,
             "families": smoke_fams}
    _publish_once(os.path.join(outdir, "tier_s_smoke.json"),
                  json.dumps(smoke, indent=1, sort_keys=True,
                             allow_nan=False) + "\n")
    return results, comp, smoke


def finalize_smoke(outdir, pre_ref, completion_ref, results_ref,
                   blob_reader):
    """After the results/completion COMMIT exists: REOPEN all three
    refs, verify their recorded digests against the draft smoke, and
    only then publish the final smoke create-once (codex 1328Z
    item 1 -- a None/caller/altered digest never survives)."""
    with open(os.path.join(outdir, "tier_s_smoke.json"),
              encoding="utf-8") as f:
        draft = json.load(f)
    pre = json.loads(blob_reader(
        pre_ref["commit"], pre_ref["path"]).decode("utf-8"))
    got_pre = _digest({k: v for k, v in pre.items()
                       if k != "invocation_sha256"})
    if got_pre != draft.get("pre_invocation_sha256"):
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: reopened pre diverges from "
            "the draft smoke")
    comp = json.loads(blob_reader(
        completion_ref["commit"],
        completion_ref["path"]).decode("utf-8"))
    if _digest(comp) != draft.get("completion_sha256"):
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: reopened completion diverges "
            "from the draft smoke")
    r_raw = blob_reader(results_ref["commit"], results_ref["path"])
    if hashlib.sha256(r_raw).hexdigest() != \
            results_ref.get("blob_sha256") or \
            results_ref.get("blob_sha256") != \
            draft.get("results_blob_sha256"):
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: reopened results diverge "
            "from the draft smoke")
    if draft.get("effect_grids_sha256") in (None, ""):
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: the draft smoke carries no "
            "grid content digest")
    smoke = dict(draft,
                 pre_invocation_ref=dict(pre_ref),
                 completion_ref=dict(completion_ref),
                 results_ref=dict(results_ref))
    _publish_once(os.path.join(outdir, "tier_s_smoke_final.json"),
                  json.dumps(smoke, indent=1, sort_keys=True,
                             allow_nan=False) + "\n")
    return smoke


# ---------------------------------------------------------------- selftest
def _selftest():
    import subprocess
    import tempfile
    import w2_power_harness_cayley as PH
    import w2_tier_selector_cayley as TS
    tmp = tempfile.mkdtemp(prefix="w2tiers_")

    # --- the REAL multi-commit git history (codex 1328Z items 1+2:
    # no hand-built chain shapes; every stage its own commit) ---
    repo2 = os.path.join(tmp, "repo")
    os.makedirs(repo2)

    def g(*args):
        r = subprocess.run(["git", "-C", repo2] + list(args),
                           capture_output=True)
        assert r.returncode == 0, (args, r.stderr.decode()[:200])
        return r.stdout.decode().strip()

    def commit_all(msg):
        g("add", "-A")
        subprocess.run(["git", "-C", repo2, "-c",
                        "user.name=kat", "-c", "user.email=k@k",
                        "commit", "-q", "-m", msg],
                       capture_output=True)
        return g("rev-parse", "HEAD")

    def wf(rel, body):
        pth = os.path.join(repo2, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(pth), exist_ok=True)
        with open(pth, "w", encoding="utf-8", newline="\n") as f:
            f.write(body)
    g("init", "-q")
    grids = {"B2A": [{"m": 1}, {"m": 2}, {"m": 3}],
             "B2B": [{"m": 1}, {"m": 2}, {"m": 3}],
             "B3A": [{"d": 1}, {"d": 2}, {"d": 3}],
             "B1B": [{"delta_lat": 0.1, "k": 3, "n_e": 3},
                     {"delta_lat": 0.2, "k": 3, "n_e": 3},
                     {"delta_lat": 0.3, "k": 3, "n_e": 3},
                     {"gain": 3.0}, {"gain": 10.0}]}
    grids_body = json.dumps({"schema": "f2g-w2-effect-grids-v1",
                             "grids": grids}, sort_keys=True)
    geo_body_obj = {"capsule_digest": None,
                    "seed_authority_sha256": "b" * 64,
                    "loco_registry_carrier": "cascadia",
                    "registries": {"cascadia": ["S0", "S1"]}}
    geo_body_obj["capsule_digest"] = hashlib.sha256(json.dumps(
        {k: v for k, v in geo_body_obj.items()
         if k != "capsule_digest"}, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    wf("grids.json", grids_body)
    wf("geom.json", json.dumps(geo_body_obj, sort_keys=True))
    wf("impl.py", "# impl")
    c1 = commit_all("carriers")

    def sha_at(commit, path):
        r = subprocess.run(["git", "-C", repo2, "cat-file", "blob",
                            f"{commit}:{path}"],
                           capture_output=True)
        return hashlib.sha256(r.stdout).hexdigest()
    man = {"slots": {"x": {"status": "BOUND", "pins": [
        {"path": "grids.json", "commit": c1,
         "blob_sha256": sha_at(c1, "grids.json")},
        {"path": "geom.json", "commit": c1,
         "blob_sha256": sha_at(c1, "geom.json")},
        {"path": "impl.py", "commit": c1,
         "blob_sha256": sha_at(c1, "impl.py")}]}}}
    wf("docs/f2g_window2_execution/execution_manifest.json",
       json.dumps(man, sort_keys=True))
    c2 = commit_all("manifest")

    def breader(commit, path):
        r = subprocess.run(["git", "-C", repo2, "cat-file", "blob",
                            f"{commit}:{path}"],
                           capture_output=True)
        if r.returncode != 0:
            raise RunnerRefusal(
                f"RUNNER_TIER_S_UNADMITTED: {path} unreadable at "
                f"{commit}")
        return r.stdout
    outdir = os.path.join(repo2, "tier_s")
    pre, points = fire_pre(repo2, c2, "grids.json", "geom.json",
                           "impl.py", outdir, blob_reader=breader,
                           argv=["kat"])
    assert len(points) == 12
    assert pre["effect_grids_content_sha256"] == _digest(grids)
    c3 = commit_all("pre-invocation")

    def stub_smoke_for(fam, point):
        def fn(f, p, folds):
            assert (f, p) == (fam, point)
            reps = [{"p_values": {x: (0.001 if x == f else 0.9)
                                  for x in ("B1B", "B2A", "B2B",
                                            "B3A")}}
                    for _ in range(50)]
            return {"tier": "PRELIMINARY_SMOKE (never certifiable)",
                    "family": f, "point": dict(p),
                    "quality": {"R": 50, "n_draws": 999},
                    "replicates": reps,
                    "loco_folds": ([{"S0": 0.001, "S1": 0.001}
                                    for _ in range(50)]
                                   if folds else None),
                    "geometry_capsule_digest":
                        pre["geometry"]["capsule_digest"],
                    "seed_authority_sha256": "b" * 64,
                    "certifiable": False}
        return fn
    for i, (fam, point) in enumerate(points):
        run_smoke_point(repo2, outdir, i, pre["invocation_sha256"],
                        breader,
                        smoke_fn=stub_smoke_for(fam, point))
    top8 = rank_stage1_b1b(outdir, pre["invocation_sha256"],
                           breader)
    assert len(top8) == 3
    for i in top8:
        fam, point = points[i]
        run_smoke_point(repo2, outdir, i, pre["invocation_sha256"],
                        breader, with_loco=True,
                        smoke_fn=stub_smoke_for(fam, point))
    results, comp, smoke_draft = aggregate(
        repo2, outdir, pre["invocation_sha256"], top8,
        ["S0", "S1"], breader)
    assert smoke_draft["effect_grids_sha256"] == _digest(grids)
    c4 = commit_all("results+completion")
    smoke = finalize_smoke(
        outdir,
        {"commit": c3, "path": "tier_s/tier_s_pre_invocation.json"},
        {"commit": c4, "path": "tier_s/tier_s_completion.json"},
        {"commit": c4, "path": "tier_s/tier_s_results.json",
         "blob_sha256": hashlib.sha256(open(os.path.join(
             outdir, "tier_s_results.json"), "rb").read())
             .hexdigest()},
        breader)
    c5 = commit_all("smoke")
    art = TS.select_candidates(
        smoke, grids,
        smoke_ref={"commit": c5,
                   "path": "tier_s/tier_s_smoke_final.json"},
        effect_grids_ref={"commit": c1, "path": "grids.json"})
    wf("tier_s/selector.json", json.dumps(art, sort_keys=True))
    c6 = commit_all("selector")
    # THE COMPOSITION LOCK: the actual production functions, real
    # commits, REAL admission -- no hand-built chain shapes anywhere
    def real_geom_loader(mc, path):
        cap = json.loads(breader(c1, path).decode("utf-8"))
        return cap
    adm = TS.verify_selector_admission(
        repo2, art, c2, geometry_loader=real_geom_loader,
        selector_identity={"commit": c6,
                           "path": "tier_s/selector.json",
                           "blob_sha256": "unused"})
    assert adm["manifest_commit"] == c2
    assert adm["pre_invocation"]["commit"] == c3

    # item-1 doctor: a draft smoke with a None grid digest refuses
    # at finalize
    out2 = os.path.join(tmp, "nofinal")
    os.makedirs(out2)
    with open(os.path.join(outdir, "tier_s_smoke.json"),
              encoding="utf-8") as f:
        d = json.load(f)
    d["effect_grids_sha256"] = None
    with open(os.path.join(out2, "tier_s_smoke.json"), "w",
              encoding="utf-8") as f:
        json.dump(d, f)
    try:
        finalize_smoke(
            out2,
            {"commit": c3,
             "path": "tier_s/tier_s_pre_invocation.json"},
            {"commit": c4, "path": "tier_s/tier_s_completion.json"},
            {"commit": c4, "path": "tier_s/tier_s_results.json",
             "blob_sha256": hashlib.sha256(open(os.path.join(
                 outdir, "tier_s_results.json"), "rb").read())
                 .hexdigest()},
            breader)
        raise AssertionError("None grid digest must refuse")
    except RunnerRefusal as e:
        assert "no grid content digest" in str(e) or \
            "diverges" in str(e)

    # item-4 doctors on a SECOND real fire
    out3 = os.path.join(repo2, "tier_s3")
    pre3, pts3 = fire_pre(repo2, c2, "grids.json", "geom.json",
                          "impl.py", out3, blob_reader=breader,
                          argv=["kat"])
    # wrong index refuses at derivation
    try:
        run_smoke_point(repo2, out3, 99, pre3["invocation_sha256"],
                        breader, smoke_fn=stub_smoke_for(
                            *pts3[0]))
        raise AssertionError("wrong index must refuse")
    except RunnerRefusal as e:
        assert "outside the derived grid" in str(e)
    # copied-pre-to-other-root refuses (output_root mismatch)
    out4 = os.path.join(tmp, "otherroot")
    os.makedirs(out4)
    import shutil
    shutil.copy(os.path.join(out3, "tier_s_pre_invocation.json"),
                os.path.join(out4, "tier_s_pre_invocation.json"))
    try:
        run_smoke_point(repo2, out4, 0, pre3["invocation_sha256"],
                        breader, smoke_fn=stub_smoke_for(*pts3[0]))
        raise AssertionError("other-root pre must refuse")
    except RunnerRefusal as e:
        assert "output root" in str(e)
    # swapped/altered point capsule refuses at aggregation: run all
    # points, then doctor capsule 0's point label
    for i, (fam, point) in enumerate(pts3):
        run_smoke_point(repo2, out3, i, pre3["invocation_sha256"],
                        breader,
                        smoke_fn=stub_smoke_for(fam, point))
    t8 = rank_stage1_b1b(out3, pre3["invocation_sha256"], breader)
    for i in t8:
        fam, point = pts3[i]
        run_smoke_point(repo2, out3, i, pre3["invocation_sha256"],
                        breader, with_loco=True,
                        smoke_fn=stub_smoke_for(fam, point))
    cap_p = os.path.join(out3, "smoke_point_000.json")
    with open(cap_p, encoding="utf-8") as f:
        cap0 = json.load(f)
    cap0["point"] = {"m": 2}          # relabel the outer identity
    with open(cap_p, "w", encoding="utf-8") as f:
        json.dump(cap0, f)
    try:
        aggregate(repo2, out3, pre3["invocation_sha256"], t8,
                  ["S0", "S1"], breader)
        raise AssertionError("relabelled capsule must refuse")
    except RunnerRefusal as e:
        assert "identity diverges" in str(e)

    print("w2_tier_s_runner selftest: ALL PASS (real multi-commit "
          "composition through REAL admission; stub smoke fns; "
          "PRELIMINARY_SMOKE; nothing certified)")


if __name__ == "__main__":
    _selftest()
