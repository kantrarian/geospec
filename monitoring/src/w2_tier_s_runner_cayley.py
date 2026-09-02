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
import datetime
import hashlib
import json
import math
import os
import platform
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from w2_cert_runner_cayley import (_canon, _digest, _publish_once,
                                   resolve_manifest_commit,
                                   RunnerRefusal)

PRE_SCHEMA = "f2g-w2-tier-s-pre-invocation-v2"
# v2 (codex 0314Z Design A): `host` and `interpreter` are GONE as
# top-level fields -- they now live only inside the closed `execution`
# capsule, because two copies of a runtime identity are two things
# that can drift apart. `driver` joins them: the artifact that fires
# the campaign was outside the admitted identity entirely (codex
# 0257Z finding 1), so the pre now names it the way it names the
# harness. There is no v1 fallback: no production v1 carrier exists,
# and a permissive downgrade path would reintroduce exactly the
# unbound surface this closes.
PRE_FIELDS = {"schema", "manifest_commit", "effect_grids",
              "effect_grids_content_sha256",
              "geometry", "quality", "seed_authority_sha256",
              "implementation", "driver", "execution",
              "grid_order_sha256", "output_root",
              "argv", "fired_utc", "invocation_sha256"}
# The REGISTERED production driver. A module constant, not a caller
# argument, for the same reason the grid and geometry paths are.
DRIVER_REL = "monitoring/src/w2_tier_s_driver_cayley.py"
RUNNER_REL = "monitoring/src/w2_tier_s_runner_cayley.py"

# The single closed runtime identity. `numpy_config_sha256` is the
# deterministic build-config fingerprint: same-host interpreter drift
# is what made the driver's earlier host-only guard useless, and a
# NumPy rebuilt with different SIMD support is the same hazard one
# layer down. If it cannot be produced we REFUSE rather than omit it
# -- an identity with a hole in it attests less than it appears to.
EXECUTION_SCHEMA = "f2g-w2-tier-s-execution-identity-v1"
EXECUTION_FIELDS = {"schema", "host", "interpreter_executable",
                    "interpreter_implementation",
                    "interpreter_version", "numpy_version",
                    "numpy_config_sha256"}

# grassmann pre-registered these from the manifest-pinned effect-grid
# blob BEFORE any run existed (0302Z), deriving them without importing
# this module; cayley reproduced them through this code path. codex
# 0314Z made them a PRE-RUN GATE: a divergent grid refuses before
# point 0 rather than producing a campaign nobody predicted.
REGISTERED_GRID_ORDER_SHA256 = (
    "00e8e9fdf61e7e12b4aac8a113f61513b8ae60bd45183cf646a07ce44f9fcde8")
REGISTERED_GRIDS_CONTENT_SHA256 = (
    "f76a5acc2814e1b3be99aa338945ff8829ad7f0cc360967370a13710834232d0")
# codex 2303Z finding 2: aggregate publication is ONE create-once envelope
# carrying the exact bytes of its three members; the members are
# materialised FROM it, so a retry after a partial publication completes
# exactly and never overwrites a divergent member.
AGGREGATE_ENVELOPE = "tier_s_aggregate_envelope.json"
AGGREGATE_ENVELOPE_SCHEMA = "f2g-w2-tier-s-aggregate-envelope-v1"
ENVELOPE_FIELDS = {"schema", "pre_invocation_sha256", "points_commit",
                   "point_corpus_sha256", "members"}
AGGREGATE_MEMBERS = ("tier_s_results.json", "tier_s_completion.json",
                     "tier_s_smoke.json")
FAMILIES = ("B1B", "B2A", "B2B", "B3A")
COMPLETION_SCHEMA = "f2g-w2-tier-s-completion-v1"
COMPLETION_FIELDS = {"schema", "pre_invocation_sha256",
                     "results_blob_sha256", "fired_utc",
                     "completed_utc"}


def execution_identity():
    """The live runtime identity, recomputed wherever it is checked.

    Never cached and never passed in: a value handed to a verifier is
    a claim, while a value the verifier computes for itself is a
    measurement. Every worker calls this before doing any work.
    """
    import hashlib as _h
    import json as _j
    try:
        import numpy as _np
        cfg = _np.show_config(mode="dicts")
        fp = _h.sha256(_j.dumps(cfg, sort_keys=True,
                                default=str).encode()).hexdigest()
        ver = str(_np.__version__)
    except Exception as exc:                             # noqa: BLE001
        raise RunnerRefusal(
            "RUNNER_TIER_S_EXECUTION_IDENTITY_UNAVAILABLE: the NumPy "
            f"build fingerprint could not be produced ({str(exc)[:120]})"
            " -- refusing rather than omitting a field the identity "
            "claims to carry")
    return {"schema": EXECUTION_SCHEMA,
            "host": platform.node().strip().lower(),
            "interpreter_executable":
                os.path.normcase(os.path.abspath(sys.executable)),
            "interpreter_implementation":
                platform.python_implementation(),
            "interpreter_version": sys.version,
            "numpy_version": ver,
            "numpy_config_sha256": fp}


def validate_execution(ex):
    if not isinstance(ex, dict) or set(ex) != EXECUTION_FIELDS or \
            ex.get("schema") != EXECUTION_SCHEMA:
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: the execution identity is not "
            "the closed registered capsule")
    return ex


def execution_digest(ex):
    return _digest(validate_execution(ex))


def require_live_execution(pre):
    """Every worker's first act. Compares the LIVE identity against
    the pre-bound one field by field so the refusal names what
    drifted -- 'the run does not match' is not an actionable thing to
    read at 3am."""
    want = validate_execution(pre["execution"])
    got = execution_identity()
    diff = sorted(k for k in EXECUTION_FIELDS if got[k] != want[k])
    if diff:
        raise RunnerRefusal(
            "RUNNER_TIER_S_EXECUTION_DRIFT: "
            + "; ".join(f"{k}: bound {str(want[k])[:40]!r} != live "
                        f"{str(got[k])[:40]!r}" for k in diff))
    return got


def _pin_for(manifest, path):
    matches = []
    for slot in manifest.get("slots", {}).values():
        if slot.get("status") != "BOUND":
            continue
        for pin in slot.get("pins", ()) or ():
            if isinstance(pin, dict) and pin.get("path") == path:
                matches.append(pin)
    if len(matches) != 1:
        raise RunnerRefusal(
            f"RUNNER_TIER_S_UNADMITTED: {path} is not exactly one "
            f"BOUND manifest pin (found {len(matches)})")
    return matches[0]


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
    # codex 0314Z: the PRE-RUN GATE, from grassmann's pre-registration.
    # It fires here, before the pre is published, so a divergent grid
    # cannot produce a campaign at all.
    _og, _cg = _digest(det_order), _digest(grids)
    if _og != REGISTERED_GRID_ORDER_SHA256 or \
            _cg != REGISTERED_GRIDS_CONTENT_SHA256:
        raise RunnerRefusal(
            "RUNNER_TIER_S_GRID_UNREGISTERED: order "
            f"{_og[:12]} / content {_cg[:12]} do not match the "
            f"pre-registered {REGISTERED_GRID_ORDER_SHA256[:12]} / "
            f"{REGISTERED_GRIDS_CONTENT_SHA256[:12]}")
    drv_pin = _pin_for(man, DRIVER_REL)
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
           "driver": {"commit": drv_pin["commit"],
                      "path": DRIVER_REL,
                      "blob_sha256": drv_pin["blob_sha256"]},
           "execution": execution_identity(),
           "grid_order_sha256": _digest(det_order),
           "output_root": str(outdir),
           "argv": list(argv if argv is not None else sys.argv),
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
    if _digest(det_order) != REGISTERED_GRID_ORDER_SHA256:
        raise RunnerRefusal(
            "RUNNER_TIER_S_GRID_UNREGISTERED: the derived order is "
            "not the pre-registered one")
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
    live = require_live_execution(pre)
    rec = smoke_fn(fam, point, with_loco)
    out = {"index": int(idx), "family": fam, "point": point,
           "pre_invocation_sha256": pre["invocation_sha256"],
           "execution_sha256": execution_digest(live),
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


def build_completion(pre, results_blob_sha256):
    return {"schema": COMPLETION_SCHEMA,
            "pre_invocation_sha256": pre["invocation_sha256"],
            "results_blob_sha256": str(results_blob_sha256),
            "fired_utc": pre["fired_utc"],
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                           time.gmtime())}


def write_completion(outdir, pre, results_blob_sha256):
    comp = build_completion(pre, results_blob_sha256)
    _publish_once(os.path.join(outdir, "tier_s_completion.json"),
                  json.dumps(comp, indent=1, sort_keys=True) + "\n")
    return comp


def _publish_members_from_envelope(outdir, env):
    """Materialise the envelope's members: a missing member is published
    from the envelope's exact bytes; an existing member must equal them
    byte-for-byte and is NEVER overwritten. Returns how many were
    published by this call."""
    published = 0
    for name in AGGREGATE_MEMBERS:
        body = env["members"][name]["body"]
        p = os.path.join(outdir, name)
        if os.path.exists(p):
            # codex 0444Z finding 2: compare BYTES, so a non-UTF-8 or
            # otherwise undecodable existing member takes this typed
            # path instead of escaping as a decode error
            with open(p, "rb") as f:
                live = f.read()
            if live != body.encode("utf-8"):
                raise RunnerRefusal(
                    f"RUNNER_TIER_S_AGGREGATE_DIVERGENT: {name} exists "
                    "with bytes that differ from the aggregate envelope "
                    "-- never overwritten; an operator decision")
            continue
        _publish_once(p, body)
        published += 1
    return published


def _validate_envelope_members(env, pre):
    """codex 0149Z finding 2: a recovery envelope is UNTRUSTED bytes on
    disk. Every member is validated -- closed {body, sha256} record,
    text body, lowercase 64-hex digest that recomputes, JSON-object body
    of the admitted per-member schema bound to THIS campaign -- before
    anything is digested for publication. A defect is a typed refusal
    that writes nothing; an operator gets a decision, not a traceback."""
    import w2_tier_selector_cayley as TS
    members = env.get("members")
    if not isinstance(members, dict) or \
            set(members) != set(AGGREGATE_MEMBERS):
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: the aggregate envelope does not "
            "carry exactly the three admitted members")
    parsed = {}
    for name in AGGREGATE_MEMBERS:
        m = members[name]
        if not isinstance(m, dict) or set(m) != {"body", "sha256"}:
            raise RunnerRefusal(
                f"RUNNER_TIER_S_UNADMITTED: envelope member {name} is not "
                "a closed {body, sha256} record")
        body, sha = m["body"], m["sha256"]
        if not isinstance(body, str):
            raise RunnerRefusal(
                f"RUNNER_TIER_S_UNADMITTED: envelope member {name} body "
                "is not text")
        if not isinstance(sha, str) or \
                not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise RunnerRefusal(
                f"RUNNER_TIER_S_UNADMITTED: envelope member {name} digest "
                "is not lowercase 64-hex")
        if hashlib.sha256(body.encode("utf-8")).hexdigest() != sha:
            raise RunnerRefusal(
                f"RUNNER_TIER_S_UNADMITTED: envelope member {name} does "
                "not recompute its own digest")
        try:
            obj = json.loads(body)
        except ValueError:
            raise RunnerRefusal(
                f"RUNNER_TIER_S_UNADMITTED: envelope member {name} body "
                "is not JSON")
        if not isinstance(obj, dict):
            raise RunnerRefusal(
                f"RUNNER_TIER_S_UNADMITTED: envelope member {name} body "
                "is not a JSON object")
        parsed[name] = obj
    res, comp, smoke = (parsed[n] for n in AGGREGATE_MEMBERS)
    if res.get("schema") != TS.TIER_S_RESULTS_SCHEMA or \
            set(res) != {"schema", "quality", "seed_authority_sha256",
                         "geometry_capsule_digest", "implementation",
                         "families"} or \
            res.get("quality") != pre["quality"] or \
            res.get("implementation") != pre["implementation"]:
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: envelope results member is not the "
            "closed results schema bound to this pre")
    if comp.get("schema") != COMPLETION_SCHEMA or \
            set(comp) != COMPLETION_FIELDS or \
            comp.get("pre_invocation_sha256") != pre["invocation_sha256"] \
            or comp.get("results_blob_sha256") != \
            members["tier_s_results.json"]["sha256"]:
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: envelope completion member is not "
            "the closed completion bound to this pre and these results")
    if smoke.get("schema") != "f2g-w2-tier-s-smoke-v1" or \
            smoke.get("pre_invocation_sha256") != pre["invocation_sha256"] \
            or smoke.get("results_blob_sha256") != \
            members["tier_s_results.json"]["sha256"] or \
            smoke.get("completion_sha256") != _digest(comp):
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: envelope smoke member is not the "
            "closed draft smoke bound to this pre, results and completion")
    return res, comp, smoke


def _complete_aggregate_from_envelope(repo, outdir, pre, top8,
                                      loco_registry, blob_reader,
                                      points_commit):
    """Retry path. codex 0352Z: a self-hash proves consistency with the
    edited envelope, not provenance. So the envelope is first checked as
    a WRAPPER (closed record, text bodies, recomputing digests, admitted
    per-member schemas bound to this pre), then the results and the
    smoke are REBUILT from the point capsules and the envelope must
    equal them byte-for-byte, and the completion is BOUND to the pre's
    fired time and the rebuilt results digest -- all before any member
    is written. Then complete only the missing members, exactly."""
    # codex 0444Z finding 2: the envelope file is UNTRUSTED bytes; an
    # unreadable / undecodable / unparsable envelope is a typed refusal
    try:
        with open(os.path.join(outdir, AGGREGATE_ENVELOPE), "rb") as f:
            env = json.loads(f.read().decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: the aggregate envelope in this "
            f"outdir cannot be read as JSON ({type(exc).__name__}) -- "
            "nothing written; an operator decision")
    if not isinstance(env, dict) or \
            set(env) != ENVELOPE_FIELDS or \
            env.get("schema") != AGGREGATE_ENVELOPE_SCHEMA or \
            env.get("pre_invocation_sha256") != pre["invocation_sha256"]:
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: the aggregate envelope in this "
            "outdir is not this campaign's closed envelope")
    # codex 0444Z finding 1: the envelope binds the COMMITTED point
    # corpus it was derived from; the caller must name that very
    # commit, and the rebuild below reads ONLY that commit's blobs
    if env.get("points_commit") != points_commit:
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: the aggregate envelope binds point "
            f"corpus commit {str(env.get('points_commit'))[:12]}; this "
            f"call names {str(points_commit)[:12]} -- not the carrier "
            "identity the envelope was derived from")
    res, comp, smoke = _validate_envelope_members(env, pre)
    members = env["members"]
    results, r_body, r_sha, build_smoke, corpus = _rebuild_aggregate(
        repo, outdir, pre, top8, loco_registry, blob_reader, points_commit)
    if env.get("point_corpus_sha256") != corpus["point_corpus_sha256"]:
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: envelope point-corpus digest "
            "diverges from the COMMITTED carriers at "
            f"{points_commit[:12]}")
    if members["tier_s_results.json"]["body"] != r_body:
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: envelope results diverge from the "
            "results REBUILT from the point capsules -- a self-hash proves "
            "consistency with the edited envelope, not provenance")
    _validate_completion(comp, pre, r_sha)
    if members["tier_s_completion.json"]["body"] != \
            json.dumps(comp, indent=1, sort_keys=True) + "\n":
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: envelope completion body is not "
            "canonical")
    expected_smoke = json.dumps(build_smoke(comp), indent=1,
                                sort_keys=True, allow_nan=False) + "\n"
    if members["tier_s_smoke.json"]["body"] != expected_smoke:
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: envelope smoke diverges from the "
            "smoke REBUILT from the point capsules and the validated "
            "completion")
    if _publish_members_from_envelope(outdir, env) == 0:
        raise RunnerRefusal(
            "RUNNER_PUBLISH_EXISTS: aggregate already complete (envelope "
            "and all three members present and byte-equal; create-once, "
            "never re-derived)")
    return results, comp, smoke


def _check_point_capsule(cap, idx, fam, point, pre):
    """codex 1328Z item 4: the per-point capsule is CLOSED and its
    identities equal the DERIVED point + the pre -- a relabelled
    p-vector has no seam."""
    if not isinstance(cap, dict) or set(cap) != {
            "index", "family", "point", "pre_invocation_sha256",
            "execution_sha256", "record"}:
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
    if cap["execution_sha256"] != execution_digest(pre["execution"]):
        raise RunnerRefusal(
            f"RUNNER_TIER_S_EXECUTION_DRIFT: point {idx} was produced "
            "under a runtime identity that is not the pre-bound one")
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
    # codex 2303Z finding 2: VALUES, not just keys. A JSON-valid string
    # or an out-of-range number passed the key check and serialised
    # fine, then raised during numeric work AFTER two create-once
    # artifacts had been published. Exact closed replicate / fold
    # schemas; every p-value None or a finite real (not bool) in [0,1].
    reps = rec.get("replicates")
    if not isinstance(reps, list):
        raise RunnerRefusal(
            f"RUNNER_TIER_S_UNADMITTED: point {idx} replicates are not "
            "a list")
    for j, rep in enumerate(reps):
        if not isinstance(rep, dict) or set(rep) != {"p_values"} or \
                not isinstance(rep["p_values"], dict) or \
                set(rep["p_values"]) != set(FAMILIES):
            raise RunnerRefusal(
                f"RUNNER_TIER_S_UNADMITTED: point {idx} replicate {j} "
                "is not a closed four-family p-vector")
        for fam_k, v in rep["p_values"].items():
            if not _p_value_ok(v):
                raise RunnerRefusal(
                    f"RUNNER_TIER_S_UNADMITTED: point {idx} replicate "
                    f"{j} {fam_k} p-value {v!r} is not None or a finite "
                    "real in [0,1]")
    folds = rec.get("loco_folds")
    if folds is not None:
        if not isinstance(folds, list):
            raise RunnerRefusal(
                f"RUNNER_TIER_S_UNADMITTED: point {idx} loco_folds is "
                "neither null nor a list")
        for j, fr in enumerate(folds):
            if not isinstance(fr, dict):
                raise RunnerRefusal(
                    f"RUNNER_TIER_S_UNADMITTED: point {idx} fold {j} is "
                    "not a station->p map")
            for st, v in fr.items():
                if not isinstance(st, str) or not _p_value_ok(v):
                    raise RunnerRefusal(
                        f"RUNNER_TIER_S_UNADMITTED: point {idx} fold {j} "
                        f"station {st!r} p-value {v!r} is not None or a "
                        "finite real in [0,1]")


def _p_value_ok(v):
    return v is None or (isinstance(v, (int, float))
                         and not isinstance(v, bool)
                         and math.isfinite(v) and 0.0 <= v <= 1.0)


def _rebuild_aggregate(repo, outdir, pre, top8, loco_registry, blob_reader,
                       points_commit):
    """The DETERMINISTIC re-aggregation, publishing NOTHING: reopen and
    validate every detection / LOCO capsule through the one validator,
    rebuild the closed results v2 (bytes + digest) from the pre-bound
    grids, the registered LOCO set and the Holm rule, and return a
    smoke builder that takes the completion. First publication and
    recovery both go through here (codex 0352Z: an envelope cannot
    authenticate its own semantics; the capsules can)."""
    # codex 0444Z finding 1: the point corpus is a COMMITTED trust root.
    # Live carrier bytes are mutable; the carriers this rebuild reads
    # are the blobs at `points_commit`, and their exact path->blob set
    # is digested into the smoke and the envelope.
    if not isinstance(points_commit, str) or len(points_commit) != 40 or \
            any(c not in "0123456789abcdef" for c in points_commit):
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: points_commit is not a full 40-hex "
            "commit identity")
    r_abs, o_abs = os.path.abspath(repo), os.path.abspath(outdir)
    try:
        common = os.path.commonpath([r_abs, o_abs])
    except ValueError:
        common = None
    if common != r_abs or r_abs == o_abs:
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: the outdir is not inside the "
            "repository, so its carriers have no committed identity")
    rel = os.path.relpath(o_abs, r_abs).replace(os.sep, "/")

    def committed(name):
        try:
            raw = blob_reader(points_commit, f"{rel}/{name}")
        except RunnerRefusal:
            raise
        except Exception as exc:                        # noqa: BLE001
            raise RunnerRefusal(
                f"RUNNER_TIER_S_RESULT_MISSING: {name} is not committed at "
                f"{points_commit[:12]} ({type(exc).__name__})")
        if not raw:
            raise RunnerRefusal(
                f"RUNNER_TIER_S_RESULT_MISSING: {name} is not committed at "
                f"{points_commit[:12]}")
        try:
            return json.loads(raw.decode("utf-8")), \
                hashlib.sha256(raw).hexdigest()
        except (ValueError, UnicodeDecodeError):
            raise RunnerRefusal(
                f"RUNNER_TIER_S_UNADMITTED: committed carrier {name} at "
                f"{points_commit[:12]} is not JSON")
    points = derive_points(pre, blob_reader)
    fams = {"B2A": [], "B2B": [], "B1B": [], "B3A": []}
    carriers = []
    for idx, (fam, point) in enumerate(points):
        name = f"smoke_point_{idx:03d}.json"
        cap, csha = committed(name)
        _check_point_capsule(cap, idx, fam, point, pre)
        carriers.append([f"{rel}/{name}", csha])
        folds = None
        if fam == "B1B" and idx in top8:
            lname = f"smoke_loco_{idx:03d}.json"
            lcap, lsha = committed(lname)
            _check_point_capsule(lcap, idx, fam, point, pre)
            carriers.append([f"{rel}/{lname}", lsha])
            folds = lcap["record"]["loco_folds"]
        fams[fam].append({
            "point": dict(point),
            "grid_index": None,
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
    r_sha = hashlib.sha256(r_body.encode()).hexdigest()
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

    corpus = {"points_commit": points_commit,
              "point_corpus_sha256": _digest(sorted(carriers)),
              "carriers": len(carriers)}

    def build_smoke(comp):
        return {"schema": "f2g-w2-tier-s-smoke-v1",
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
                # codex 0444Z finding 1: the COMMITTED point corpus this
                # smoke was derived from -- the commit and the exact
                # path->blob digest of every carrier read
                "points_commit": points_commit,
                "point_corpus_sha256": corpus["point_corpus_sha256"],
                "families": smoke_fams}
    return results, r_body, r_sha, build_smoke, corpus


_UTC_RE = r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ"


def _validate_completion(comp, pre, r_sha):
    """The completion is the one member that cannot be rebuilt (it
    carries the completion instant), so it is BOUND instead: closed
    fields, the pre's digest, the pre's OWN fired time, the REBUILT
    results digest, canonical UTC instants in order."""
    if not isinstance(comp, dict) or set(comp) != COMPLETION_FIELDS or \
            comp.get("schema") != COMPLETION_SCHEMA:
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: completion is not the closed schema")
    if comp["pre_invocation_sha256"] != pre["invocation_sha256"]:
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: completion does not bind this pre")
    if comp["results_blob_sha256"] != r_sha:
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: completion does not bind the "
            "results REBUILT from the point capsules")
    if comp["fired_utc"] != pre["fired_utc"]:
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: completion fired time is not the "
            "pre's fired time")
    # codex 0444Z finding 3: a digit-shape regex admitted
    # 9999-99-99T99:99:99Z. Parse strictly, require an exact round-trip
    # to the canonical format, then compare the parsed instants.
    inst = {}
    for k in ("fired_utc", "completed_utc"):
        inst[k] = _parse_utc_instant(comp[k], f"completion {k}")
    if not inst["fired_utc"] <= inst["completed_utc"]:
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: completion times reversed")


def _parse_utc_instant(v, where):
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    if not isinstance(v, str) or not re.fullmatch(_UTC_RE, v):
        raise RunnerRefusal(
            f"RUNNER_TIER_S_UNADMITTED: {where} is not a canonical UTC "
            "instant")
    try:
        dt = datetime.datetime.strptime(v, fmt)
    except ValueError:
        raise RunnerRefusal(
            f"RUNNER_TIER_S_UNADMITTED: {where} {v!r} is not a real "
            "instant (impossible month, day, hour, minute or second)")
    if dt.strftime(fmt) != v:
        raise RunnerRefusal(
            f"RUNNER_TIER_S_UNADMITTED: {where} {v!r} does not round-trip "
            "the canonical format")
    return dt


def aggregate(repo, outdir, expected_pre_sha, top8, loco_registry,
              blob_reader, points_commit):
    """The DETERMINISTIC aggregator: per-point capsules -> the closed
    derivational results v2 -> completion -> smoke; the selector runs
    separately over the COMMITTED carriers. Missing results refuse;
    nothing is recomputed from caller state."""
    pre = _load_pre(outdir, expected_pre_sha)
    _check_outdir(pre, outdir)
    if os.path.exists(os.path.join(outdir, AGGREGATE_ENVELOPE)):
        return _complete_aggregate_from_envelope(
            repo, outdir, pre, top8, loco_registry, blob_reader,
            points_commit)
    results, r_body, r_sha, build_smoke, corpus = _rebuild_aggregate(
        repo, outdir, pre, top8, loco_registry, blob_reader, points_commit)
    comp = build_completion(pre, r_sha)      # in memory; nothing published yet
    _validate_completion(comp, pre, r_sha)
    smoke = build_smoke(comp)
    # ---- everything above ran in memory. codex 2303Z finding 2: the
    # first irreversible publication used to happen BEFORE the numeric
    # work (results, completion, then holm), so a value defect stranded
    # two create-once members and the retry refused. Now every member
    # is serialised and re-parsed here, sealed into ONE create-once
    # envelope, and only then materialised -- a failure before the
    # envelope leaves nothing durable; a failure after it is completed
    # exactly by the next call, which RE-DERIVES the members from the
    # point capsules and requires the envelope to agree (codex 0352Z).
    bodies = {"tier_s_results.json": r_body,
              "tier_s_completion.json":
                  json.dumps(comp, indent=1, sort_keys=True) + "\n",
              "tier_s_smoke.json":
                  json.dumps(smoke, indent=1, sort_keys=True,
                             allow_nan=False) + "\n"}
    for name, body in bodies.items():
        json.loads(body)                     # reparse before anything lands
    env = {"schema": AGGREGATE_ENVELOPE_SCHEMA,
           "pre_invocation_sha256": pre["invocation_sha256"],
           "points_commit": points_commit,
           "point_corpus_sha256": corpus["point_corpus_sha256"],
           "members": {name: {"sha256": hashlib.sha256(
                                  body.encode("utf-8")).hexdigest(),
                              "body": body}
                       for name, body in bodies.items()}}
    _publish_once(os.path.join(outdir, AGGREGATE_ENVELOPE),
                  json.dumps(env, indent=1, sort_keys=True,
                             allow_nan=False) + "\n")
    _publish_members_from_envelope(outdir, env)
    return results, comp, smoke


def _is_ancestor_git(repo, a, b):
    import subprocess
    return subprocess.run(["git", "-C", repo, "merge-base", "--is-ancestor",
                           a, b], capture_output=True).returncode == 0


def _hex(s, n):
    return isinstance(s, str) and len(s) == n and \
        all(c in "0123456789abcdef" for c in s)


def finalize_smoke(repo, outdir, pre_ref, completion_ref, results_ref,
                   blob_reader):
    """After the results/completion COMMIT exists: REOPEN all three refs,
    verify their recorded digests against the draft smoke, and only then
    publish the final smoke create-once (codex 1328Z item 1 -- a
    None/caller/altered digest never survives).

    codex 0537Z: the draft smoke used to be read from the LIVE outdir and
    every field not re-verified was copied into the final smoke -- so an
    edited point-corpus receipt (points_commit, point_corpus_sha256)
    passed straight through while pre/completion/results all reopened
    clean. Now the draft smoke and the aggregate envelope are reopened
    FROM THE RESULTS COMMIT; the live draft must EQUAL the committed
    draft byte-for-byte; the draft's receipt must equal the envelope's;
    every envelope member must be the committed member at that commit;
    the point commit must sit STRICTLY between the pre commit and the
    results commit; and the final smoke is published from the COMMITTED
    draft, never from a live file."""
    rc = results_ref["commit"]
    r_dir = results_ref["path"].rsplit("/", 1)[0] \
        if "/" in results_ref["path"] else ""

    def rp(name):
        return f"{r_dir}/{name}" if r_dir else name
    draft_raw = blob_reader(rc, rp("tier_s_smoke.json"))
    env_raw = blob_reader(rc, rp(AGGREGATE_ENVELOPE))
    try:
        draft = json.loads(draft_raw.decode("utf-8"))
        env = json.loads(env_raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: the committed draft smoke or "
            f"aggregate envelope at {str(rc)[:12]} is not JSON")
    live_path = os.path.join(outdir, "tier_s_smoke.json")
    if not os.path.exists(live_path):
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: no draft smoke in the outdir")
    with open(live_path, "rb") as f:
        live = f.read()
    if live != draft_raw:
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: the live draft smoke diverges "
            f"byte-for-byte from the draft committed at {str(rc)[:12]} -- "
            "the committed draft is the authority; nothing is finalised "
            "from a live file")
    if not isinstance(env, dict) or set(env) != ENVELOPE_FIELDS or \
            env.get("schema") != AGGREGATE_ENVELOPE_SCHEMA:
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: the committed aggregate envelope is "
            "not the closed envelope")
    pc, pcs = draft.get("points_commit"), draft.get("point_corpus_sha256")
    if not _hex(pc, 40) or not _hex(pcs, 64) or \
            pc != env.get("points_commit") or \
            pcs != env.get("point_corpus_sha256"):
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: the draft smoke's point-corpus "
            "receipt does not equal the committed envelope's receipt")
    for name in AGGREGATE_MEMBERS:
        member = env["members"].get(name)
        if not isinstance(member, dict) or \
                member.get("body", "").encode("utf-8") != \
                blob_reader(rc, rp(name)):
            raise RunnerRefusal(
                f"RUNNER_TIER_S_UNADMITTED: committed member {name} at "
                f"{str(rc)[:12]} differs from the committed envelope")
    pre_c = pre_ref["commit"]
    if pc == pre_c or not _is_ancestor_git(repo, pre_c, pc):
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: the point corpus commit is not "
            "strictly after the pre commit")
    if pc == rc or not _is_ancestor_git(repo, pc, rc):
        raise RunnerRefusal(
            "RUNNER_TIER_S_UNADMITTED: the results commit is not strictly "
            "after the point corpus commit")
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
    # v2: the REGISTERED grids verbatim. The pre-run gate (codex
    # 0314Z, from grassmann's pre-registration) refuses any other
    # order, and a fixture able to opt out of a production gate is
    # not exercising production.
    with open(os.path.join(
            os.path.dirname(os.path.dirname(_HERE)),
            "docs", "f2g_window2_execution",
            "effect_grids_w2_v1.json"), encoding="utf-8") as _f:
        grids_body = _f.read()
    grids = json.loads(grids_body)["grids"]
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
    wf(DRIVER_REL, "# driver fixture\n")
    wf(RUNNER_REL, "# runner fixture\n")
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
         "blob_sha256": sha_at(c1, "impl.py")},
        {"path": DRIVER_REL, "commit": c1,
         "blob_sha256": sha_at(c1, DRIVER_REL)},
        {"path": RUNNER_REL, "commit": c1,
         "blob_sha256": sha_at(c1, RUNNER_REL)}]}}}
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
    assert len(points) == 80, len(points)
    assert pre['schema'] == PRE_SCHEMA
    validate_execution(pre['execution'])
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
    assert len(top8) == 8, len(top8)
    for i in top8:
        fam, point = points[i]
        run_smoke_point(repo2, outdir, i, pre["invocation_sha256"],
                        breader, with_loco=True,
                        smoke_fn=stub_smoke_for(fam, point))
    c_pts = commit_all("point carriers")   # codex 0444Z: committed corpus
    results, comp, smoke_draft = aggregate(
        repo2, outdir, pre["invocation_sha256"], top8,
        ["S0", "S1"], breader, c_pts)
    assert smoke_draft["effect_grids_sha256"] == _digest(grids)
    c4 = commit_all("results+completion")
    smoke = finalize_smoke(
        repo2, outdir,
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

    # codex 0314Z: selector admission must REJECT an otherwise valid
    # chain whose v2 driver pin or execution capsule was altered.
    # Without these the two identities v2 adds would be carried but
    # never adjudicated -- which is the shape of the gap they close.
    import copy as _copy

    NL = chr(10)
    _mut_n = [0]

    def _admit_with_mutated_pre(mut, label, expect):
        """The mutated chain must be INTERNALLY CONSISTENT, or the
        artifact's independent rerun diverges first and the control
        never reaches the check it exists to exercise. So the pre is
        mutated and committed, then a smoke carrying its digest and
        ref is committed on top: everything agrees, and the only
        thing wrong is the identity under test."""
        _mut_n[0] += 1
        tag = _mut_n[0]
        bad_pre = _copy.deepcopy(pre)
        mut(bad_pre)
        bad_pre["invocation_sha256"] = _digest(
            {k: v for k, v in bad_pre.items()
             if k != "invocation_sha256"})
        p_rel = f"tier_s/pre_mutated_{tag}.json"
        wf(p_rel, json.dumps(bad_pre, indent=1, sort_keys=True) + NL)
        cm1 = commit_all(f"mutated pre {tag}: {label}")
        bad_smoke = dict(
            smoke, pre_invocation_sha256=bad_pre["invocation_sha256"],
            pre_invocation_ref={"commit": cm1, "path": p_rel})
        s_rel = f"tier_s/smoke_mutated_{tag}.json"
        wf(s_rel, json.dumps(bad_smoke, indent=1, sort_keys=True) + NL)
        cm2 = commit_all(f"mutated smoke {tag}: {label}")
        bad_art = TS.select_candidates(
            bad_smoke, grids,
            smoke_ref={"commit": cm2, "path": s_rel},
            effect_grids_ref={"commit": c1, "path": "grids.json"})
        try:
            TS.verify_selector_admission(
                repo2, bad_art, c2, geometry_loader=real_geom_loader,
                selector_identity={"commit": c6,
                                   "path": "tier_s/selector.json",
                                   "blob_sha256": "unused"})
        except TS.SelectorRefusal as e:
            assert expect in str(e), (label, str(e))
            return
        raise AssertionError(
            f"SELECTOR ADMITTED a chain whose {label} was altered")

    _admit_with_mutated_pre(
        lambda p: p["driver"].update({"blob_sha256": "c" * 64}),
        "driver blob", "not the admitted pin")
    _admit_with_mutated_pre(
        lambda p: p["driver"].update({"path": "monitoring/src/"
                                              "not_the_driver.py"}),
        "driver path", "not a BOUND pin")
    _admit_with_mutated_pre(
        lambda p: p["execution"].update({"numpy_config_sha256": ""}),
        "execution NumPy config", "is empty")
    _admit_with_mutated_pre(
        lambda p: p.__setitem__("execution", {"schema": "wrong"}),
        "execution capsule shape", "closed registered capsule")
    # the ADMISSION-PATH v1 downgrade refusal (codex 0338Z). The
    # driver refuses a v1 carrier too, but that is the producer
    # refusing its own input; this is the independent consumer
    # refusing it, which is the one that matters if a v1 pre ever
    # reaches admission by some route the driver never saw.
    _admit_with_mutated_pre(
        lambda p: p.__setitem__(
            "schema", "f2g-w2-tier-s-pre-invocation-v1"),
        "v1 schema downgrade", "closed capsule schema")
    # and a v1-SHAPED pre -- correct old schema AND the old field set,
    # not merely a relabelled v2 -- must refuse just as hard, so the
    # lock cannot be satisfied by a label check alone.
    def _to_v1_shape(p):
        p["schema"] = "f2g-w2-tier-s-pre-invocation-v1"
        ex = p.pop("execution")
        p.pop("driver")
        p["host"] = ex["host"]
        p["interpreter"] = {"executable": ex["interpreter_executable"],
                            "version": ex["interpreter_version"]}
    _admit_with_mutated_pre(_to_v1_shape, "genuine v1 shape",
                            "closed capsule schema")
    print("  selector REFUSES both a relabelled v1 and a GENUINELY "
          "v1-shaped pre (old schema, old field set, no driver, no "
          "execution) -- the downgrade path is closed at admission, "
          "not just at the producer")
    print("  selector REJECTS an altered driver blob, an unadmitted "
          "driver path, an emptied NumPy build fingerprint and a "
          "malformed execution capsule -- the v2 identities are "
          "adjudicated, not merely carried")

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
            repo2, out2,
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
    # the corpus is COMMITTED authority now: the relabelled capsule must
    # be committed to be what aggregate reads
    c_rel = commit_all("relabelled capsule")
    try:
        aggregate(repo2, out3, pre3["invocation_sha256"], t8,
                  ["S0", "S1"], breader, c_rel)
        raise AssertionError("relabelled capsule must refuse")
    except RunnerRefusal as e:
        assert "identity diverges" in str(e)

    print("w2_tier_s_runner selftest: ALL PASS (real multi-commit "
          "composition through REAL admission; stub smoke fns; "
          "PRELIMINARY_SMOKE; nothing certified)")


if __name__ == "__main__":
    _selftest()
