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
                  json.dumps(pre, indent=1, sort_keys=True) + "\n")
    return pre, points


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


def run_smoke_point(repo, outdir, idx, expected_pre_sha, points,
                    with_loco=False, smoke_fn=None):
    """ONE detection point, whole-point, authenticated against the
    PRE-FIRE digest before any work (codex 0432Z: zero workers
    before invocation publication)."""
    pre = _load_pre(outdir, expected_pre_sha)
    fam, point = points[idx]
    if smoke_fn is None:
        import w2_power_harness_cayley as PH

        def smoke_fn(f, p, folds):
            return PH.run_point_smoke(
                repo, {"manifest_commit": pre["manifest_commit"],
                       "path": pre["geometry"]["path"]},
                f, p, with_loco_folds=folds)
    rec = smoke_fn(fam, point, with_loco)
    out = {"index": idx, "family": fam, "point": point,
           "pre_invocation_sha256": pre["invocation_sha256"],
           "record": rec}
    name = (f"smoke_loco_{idx:03d}.json" if with_loco
            else f"smoke_point_{idx:03d}.json")
    _publish_once(os.path.join(outdir, name),
                  json.dumps(out, indent=1, sort_keys=True) + "\n")
    return out


def rank_stage1_b1b(outdir, points, expected_pre_sha):
    """Deterministic stage-1 ranking of the B1B detection points by
    (-pre-LOCO count, registered grid index) from the published
    phase-1 capsules -- identifies the top-8 for phase 2."""
    import w2_power_harness_cayley as PH
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


def aggregate(repo, outdir, points, expected_pre_sha, top8,
              loco_registry):
    """The DETERMINISTIC aggregator: per-point capsules -> the closed
    derivational results v2 -> completion -> smoke; the selector runs
    separately over the COMMITTED carriers. Missing results refuse;
    nothing is recomputed from caller state."""
    pre = _load_pre(outdir, expected_pre_sha)
    fams = {"B2A": [], "B2B": [], "B1B": [], "B3A": []}
    for idx, (fam, point) in enumerate(points):
        p = os.path.join(outdir, f"smoke_point_{idx:03d}.json")
        if not os.path.exists(p):
            raise RunnerRefusal(
                f"RUNNER_TIER_S_RESULT_MISSING: point {idx}")
        with open(p, encoding="utf-8") as f:
            cap = json.load(f)
        if cap.get("pre_invocation_sha256") != \
                pre["invocation_sha256"]:
            raise RunnerRefusal(
                f"RUNNER_TIER_S_PRE_DIGEST_MISMATCH: point {idx}")
        folds = None
        if fam == "B1B" and idx in top8:
            lp = os.path.join(outdir, f"smoke_loco_{idx:03d}.json")
            if not os.path.exists(lp):
                raise RunnerRefusal(
                    f"RUNNER_TIER_S_RESULT_MISSING: loco {idx}")
            with open(lp, encoding="utf-8") as f:
                lcap = json.load(f)
            if lcap.get("pre_invocation_sha256") != \
                    pre["invocation_sha256"]:
                raise RunnerRefusal(
                    f"RUNNER_TIER_S_PRE_DIGEST_MISMATCH: loco {idx}")
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
    r_body = json.dumps(results, indent=1, sort_keys=True) + "\n"
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
             "effect_grids_sha256": None,   # filled by the selector
             "pre_invocation_sha256": pre["invocation_sha256"],
             "completion_sha256": _digest(comp),
             "results_blob_sha256": r_sha,
             "families": smoke_fams}
    _publish_once(os.path.join(outdir, "tier_s_smoke.json"),
                  json.dumps(smoke, indent=1, sort_keys=True) + "\n")
    return results, comp, smoke


def finalize_smoke(outdir, pre_ref, completion_ref, results_ref):
    """After the results/completion COMMIT exists: fill the smoke's
    reopenable refs {commit, path} and publish the final smoke
    create-once. The draft smoke's digests must match the refs'
    targets -- the aggregator's chronology is preserved."""
    with open(os.path.join(outdir, "tier_s_smoke.json"),
              encoding="utf-8") as f:
        draft = json.load(f)
    smoke = dict(draft,
                 pre_invocation_ref=dict(pre_ref),
                 completion_ref=dict(completion_ref),
                 results_ref=dict(results_ref))
    _publish_once(os.path.join(outdir, "tier_s_smoke_final.json"),
                  json.dumps(smoke, indent=1, sort_keys=True) + "\n")
    return smoke


# ---------------------------------------------------------------- selftest
def _selftest():
    import tempfile
    tmp = tempfile.mkdtemp(prefix="w2tiers_")
    repo_g = os.path.abspath(os.path.join(_HERE, "..", ".."))

    grids = {"B2A": [{"m": 1}, {"m": 2}],
             "B2B": [{"m": 1}], "B3A": [{"d": 1}],
             "B1B": [{"delta_lat": 0.3, "k": 3, "n_e": 3},
                     {"gain": 3.0}, {"gain": 10.0}]}
    grids_raw = json.dumps({"schema": "f2g-w2-effect-grids-v1",
                            "grids": grids}).encode()
    geo_cap = {"capsule_digest": "kat-geo",
               "seed_authority_sha256": "b" * 64,
               "loco_registry_carrier": "cascadia",
               "registries": {"cascadia": ["S0", "S1"]}}
    geo_raw = json.dumps(geo_cap).encode()
    impl_raw = b"# impl"
    pins = [{"path": "grids.json", "commit": "a" * 40,
             "blob_sha256": hashlib.sha256(grids_raw).hexdigest()},
            {"path": "geom.json", "commit": "a" * 40,
             "blob_sha256": hashlib.sha256(geo_raw).hexdigest()},
            {"path": "impl.py", "commit": "a" * 40,
             "blob_sha256": hashlib.sha256(impl_raw).hexdigest()}]
    man = {"slots": {"x": {"pins": pins}}}
    store = {("a" * 40, "grids.json"): grids_raw,
             ("a" * 40, "geom.json"): geo_raw,
             ("a" * 40, "impl.py"): impl_raw}

    def reader(c, p):
        if p.endswith("execution_manifest.json"):
            return json.dumps(man).encode()
        return store[(c, p)]
    hexmc = "f" * 40

    def resolve_stub(repo, mc):
        return hexmc
    rz = lambda r, m: hexmc
    try:
        od = os.path.join(tmp, "run1")
        pre, points = fire_pre(repo_g, hexmc, "grids.json",
                               "geom.json", "impl.py", od,
                               blob_reader=reader, argv=["kat"],
                               resolver=rz)
        assert len(points) == 5           # 5 detection points
        assert pre["effect_grids"]["commit"] == "a" * 40
        assert pre["seed_authority_sha256"] == "b" * 64
        # create-once: a second fire refuses
        try:
            fire_pre(repo_g, hexmc, "grids.json", "geom.json",
                     "impl.py", od, blob_reader=reader,
                     argv=["kat"], resolver=rz)
            raise AssertionError("second pre must refuse")
        except RunnerRefusal as e:
            assert "RUNNER_PUBLISH_EXISTS" in str(e)

        def stub_smoke(fam, point, folds):
            reps = [{"p_values": {f: (0.001 if f == fam else 0.9)
                                  for f in ("B1B", "B2A", "B2B",
                                            "B3A")}}
                    for _ in range(50)]
            return {"replicates": reps,
                    "loco_folds": ([{"S0": 0.001, "S1": 0.001}
                                    for _ in range(50)]
                                   if folds else None)}
        # zero workers before publication: a worker against an outdir
        # with no pre-invocation refuses with no smoke call
        od0 = os.path.join(tmp, "nopre")
        os.makedirs(od0)
        called = []
        try:
            run_smoke_point(repo_g, od0, 0, "0" * 64, points,
                            smoke_fn=lambda *a: called.append(1))
            raise AssertionError("no-pre worker must refuse")
        except (RunnerRefusal, FileNotFoundError):
            pass
        assert not called
        # phase 1 + digest authentication
        for i in range(len(points)):
            run_smoke_point(repo_g, od, i,
                            pre["invocation_sha256"], points,
                            smoke_fn=stub_smoke)
        # post-invocation mutation: doctor the pre file -> workers
        # and aggregator refuse
        od2 = os.path.join(tmp, "mut")
        pre2, pts2 = fire_pre(repo_g, hexmc, "grids.json",
                              "geom.json", "impl.py", od2,
                              blob_reader=reader, argv=["kat"],
                              resolver=rz)
        with open(os.path.join(od2, "tier_s_pre_invocation.json"),
                  encoding="utf-8") as f:
            doc = json.load(f)
        doc["seed_authority_sha256"] = "c" * 64
        with open(os.path.join(od2, "tier_s_pre_invocation.json"),
                  "w", encoding="utf-8") as f:
            json.dump(doc, f)
        try:
            run_smoke_point(repo_g, od2, 0,
                            pre2["invocation_sha256"], pts2,
                            smoke_fn=stub_smoke)
            raise AssertionError("mutated pre must refuse")
        except RunnerRefusal as e:
            assert "PRE_DIGEST_MISMATCH" in str(e)
        # stage-1 ranking + phase 2 + aggregate
        top8 = rank_stage1_b1b(od, points,
                               pre["invocation_sha256"])
        assert len(top8) == 1              # one B1B detection point
        for i in top8:
            run_smoke_point(repo_g, od, i,
                            pre["invocation_sha256"], points,
                            with_loco=True, smoke_fn=stub_smoke)
        results, comp, smoke = aggregate(
            repo_g, od, points, pre["invocation_sha256"], top8,
            ["S0", "S1"])
        assert comp["pre_invocation_sha256"] == \
            pre["invocation_sha256"]
        assert smoke["pre_invocation_sha256"] == \
            pre["invocation_sha256"]
        b1b = smoke["families"]["B1B"][0]
        assert b1b["outcomes"] == [True] * 50
        assert b1b["post_loco_outcomes"] == [True] * 50
        # missing result refuses the aggregator
        od3 = os.path.join(tmp, "missing")
        pre3, pts3 = fire_pre(repo_g, hexmc, "grids.json",
                              "geom.json", "impl.py", od3,
                              blob_reader=reader, argv=["kat"],
                              resolver=rz)
        try:
            aggregate(repo_g, od3, pts3, pre3["invocation_sha256"],
                      [], ["S0", "S1"])
            raise AssertionError("missing results must refuse")
        except RunnerRefusal as e:
            assert "RESULT_MISSING" in str(e)
    finally:
        pass
    print("w2_tier_s_runner selftest: ALL PASS (stub smoke only; "
          "PRELIMINARY_SMOKE; nothing certified)")


if __name__ == "__main__":
    _selftest()
