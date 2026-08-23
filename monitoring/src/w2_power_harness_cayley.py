#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 POWER-CERTIFICATION HARNESS, core (cayley) -- prereg v0.3
sec 6 ("full window-1 instrument ... executed through the ENTIRE sec-5
rejection graph") + the frozen family effect grids (annex_b2b sec
"Effect grid", annex_b1b sec "Effect grid + specificity gate").

WHAT THIS IS: the machinery -- synthetic generator under the REGISTERED
law, per-family injection operators, the full-four-member-Holm
replicate rule, and the verbatim Tier stopping contract (R=20 evaluate;
CP-LB >= 0.80 -> CERTIFIED; one-sided 95% exact UPPER < 0.80 ->
FAILED; else extend to R=40; straddle -> typed
CANNOT_DETERMINE_POWER_ESTIMATE, never rounded).

WHAT THIS IS NOT: a certification run. Certified numbers require the
registered scale (Tier-C n_draws=9,999), the PRESTART anticipated-mask
geometry (and later the Stage-3 true masks), and the codex round. The
selftest here is FIXTURE-SCALE MECHANISM VERIFICATION ONLY and is
labeled so in every artifact -- fixture numbers can NEVER populate a
certified MDE or a passed power contract (the pinned Tier-S rule,
applied a fortiori).

Registered generator law (pinned Phase-B constants, carried verbatim):
u_ab(d) = MU0 + GAMMA*G(d) + s_a(d) + s_b(d) + eps_ab(d), G ~ N(0,1)
drawn once per calendar day common to all carriers; values = tanh(u);
MCAR cell dropout. MU0 = atanh(0.30), SIGMA_S = 0.15, SIGMA_E = 0.20,
MCAR = 0.08, GAMMA = 0.05.

Injection operators (frozen annex classes):
- B2A / B2B swap: m in {1,2,3} block swaps at the registered onset.
- B2B churn-robustness: (m=2 swap) x synthetic per-day station dropout
  at {10%, 25%} applied to the injected panel (dropped stations leave
  the measured set AND their incident edges that day -- producer-
  consistent).
- B1B detection: the registered (delta_lat, k, n_e) burst grid.
- B1B specificity (the KOZT class): ONE station's incident-edge raw
  values scaled x{3, 10} from a registered onset; the certified
  contract requires <= 0.05 FAMILYWISE positive rate on this class
  alongside CP-LB >= 0.80 on detection classes.

Replicate rule (sec 5): every replicate runs ALL FOUR families
(B2A/B2B/B1B/B3A) on the same panel; Holm at alpha 0.05 over the four
p-values; recovery for target h iff h rejects under that Holm. A family
returning no p (typed) is a non-rejection with m held at 4
(conservative; disclosed pin).

Geometry is INJECTED (anticipated-mask envelope at PRESTART; reduced
fixtures here) -- carriers, registries, masks, calendar, B1B block
geometry. B2A/B3A run via the PINNED engine functions; B2B/B1B via the
w2 engines. This module opens no window-2 value.
"""
import hashlib
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

import d2_f2g_phase_b_stats as _pb
import w2_b2b as _b2b
import w2_b1b as _b1b

# registered generator law (pinned)
MU0 = math.atanh(0.30)
SIGMA_S = 0.15
SIGMA_E = 0.20
MCAR = 0.08
GAMMA = 0.05

# stopping contract (pinned, verbatim)
CP_FLOOR = 0.80
R_FIRST = 20
R_MAX = 40
ALPHA_HOLM = 0.05
ARTIFACT_MAX_RATE = 0.05
GRAPH = ("B1B", "B2A", "B2B", "B3A")

TIER_LABEL_FIXTURE = "FIXTURE_SMOKE (never certifiable)"


class PowerHarnessError(ValueError):
    pass


# ------------------------------------------------ exact binomial bounds
def cp_lb(k, n):
    """One-sided 95% exact (Clopper-Pearson) LOWER bound."""
    if k <= 0:
        return 0.0
    from scipy.stats import beta
    return float(beta.ppf(0.05, k, n - k + 1))


def cp_ub(k, n):
    """One-sided 95% exact UPPER bound."""
    if k >= n:
        return 1.0
    from scipy.stats import beta
    return float(beta.ppf(0.95, k + 1, n - k))


def certify(success_fn, r_first=R_FIRST, r_max=R_MAX, floor=CP_FLOOR):
    """The verbatim stopping rule. success_fn(r) -> bool, called
    lazily replicate by replicate; the trace records every step."""
    succ = [bool(success_fn(r)) for r in range(r_first)]
    k, n = sum(succ), len(succ)
    trace = [{"R": n, "k": k, "lb": cp_lb(k, n), "ub": cp_ub(k, n)}]
    if trace[-1]["lb"] >= floor:
        status = "CERTIFIED"
    elif trace[-1]["ub"] < floor:
        status = "FAILED"
    else:
        succ += [bool(success_fn(r)) for r in range(r_first, r_max)]
        k, n = sum(succ), len(succ)
        trace.append({"R": n, "k": k, "lb": cp_lb(k, n),
                      "ub": cp_ub(k, n)})
        if trace[-1]["lb"] >= floor:
            status = "CERTIFIED"
        elif trace[-1]["ub"] < floor:
            status = "FAILED"
        else:
            status = "CANNOT_DETERMINE_POWER_ESTIMATE"
    return {"status": status, "R": n, "k": k, "lb": trace[-1]["lb"],
            "ub": trace[-1]["ub"], "floor": floor, "trace": trace}


# ------------------------------------------------ geometry + generator
def fixture_geometry(n_days=84, n_stations=10, seed_tag="fx"):
    """Reduced fixture geometry: ONE carrier, n_days registered days
    (B2A needs > 60 for its 60-day walk-forward baseline), two segments.
    The PRESTART envelope supplies the real anticipated geometry."""
    days = [f"D{i:03d}" for i in range(n_days)]
    sts = [f"S{i:02d}" for i in range(n_stations)]
    return {"carriers": ["cx"], "registry": {"cx": sts},
            "masks": {"cx": days}, "calendar": days,
            "segments": {"cx": {s: ("segA" if i < n_stations // 2
                                    else "segB")
                                for i, s in enumerate(sts)}},
            "eval_start_index": 60,           # walk-forward baseline
            "b1b": {"n_blocks": 14, "block_len": 6,
                    "baseline_positions": 60, "testable_min": 18},
            "seed_tag": seed_tag}


def _edges_of(stations):
    return ["|".join(sorted((a, b))) for i, a in enumerate(stations)
            for b in stations[i + 1:]]


def rep_seed(seed_root, family, point, r):
    material = json.dumps([seed_root, family, point, int(r)],
                          sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def make_panel(geom, family, point, r, seed_root, inject=True):
    """One synthetic replicate panel under the registered law, with the
    family's injection applied to the (single) fixture carrier.
    Returns {"u_panel", "measured", "dropped"} raw pieces plus the
    assembled per-engine shapes via panel_views()."""
    rng = np.random.Generator(np.random.PCG64(
        rep_seed(seed_root, family, point, r)))
    cal = geom["calendar"]
    G = rng.standard_normal(len(cal))
    ck = geom["carriers"][0]
    sts = geom["registry"][ck]
    eds = _edges_of(sts)
    days = geom["masks"][ck]
    s = rng.normal(0.0, SIGMA_S, size=(len(sts), len(days)))
    eps = rng.normal(0.0, SIGMA_E, size=(len(eds), len(days)))
    mcar = rng.random((len(eds), len(days))) < MCAR
    six = {st: i for i, st in enumerate(sts)}
    cpos = {d: i for i, d in enumerate(cal)}
    gvec = np.array([G[cpos[d]] for d in days])
    u = np.empty((len(eds), len(days)))
    for j, e in enumerate(eds):
        a, b = e.split("|")
        u[j] = MU0 + GAMMA * gvec + s[six[a]] + s[six[b]] + eps[j]

    ev0 = geom["eval_start_index"]
    half = len(sts) // 2
    dropped = {d: set() for d in days}
    gain = {}

    if family in ("B2A", "B2B") and inject:
        m = int(point.get("m", 0))
        if m > 0:
            onset = ev0 + max(1, (len(days) - ev0) // 3)
            block = {st: (0 if i < half else 1)
                     for i, st in enumerate(sts)}
            swapped = dict(block)
            for st in sts[half - m:half]:
                swapped[st] = 1
            for st in sts[half:half + m]:
                swapped[st] = 0
            for j, e in enumerate(eds):
                a, b = e.split("|")
                u[j, :onset] += 0.9 if block[a] == block[b] else -0.5
                u[j, onset:] += 0.9 if swapped[a] == swapped[b] \
                    else -0.5
    if family == "B2B":
        rate = float(point.get("dropout", 0.0))
        if rate > 0.0:
            for t in range(ev0, len(days)):
                for st in sts:
                    if rng.random() < rate:
                        dropped[days[t]].add(st)
    if family == "B1B" and inject:
        if "gain" in point:            # the KOZT specificity class
            g = float(point["gain"])
            onset = ev0 + max(1, (len(days) - ev0) // 3)
            target = sts[int(rng.integers(0, len(sts)))]
            gain = {"station": target, "g": g, "onset": onset}
        else:                          # registered burst grid
            k = int(point["k"])
            n_e = int(point["n_e"])
            d_ = float(point["delta_lat"])
            max_start = (len(days) - ev0) - k
            start = ev0 + (0 if max_start <= 0 else
                           int(rng.integers(0, max_start + 1)))
            for e in eds[:n_e]:
                j = eds.index(e)
                u[j, start:start + k] += d_

    vals = np.tanh(u)
    r_series = {}
    for j, e in enumerate(eds):
        a, b = e.split("|")
        row = {}
        for t, d in enumerate(days):
            if mcar[j, t]:
                continue
            if a in dropped[d] or b in dropped[d]:
                continue
            v = float(vals[j, t])
            if gain and gain["station"] in (a, b) \
                    and t >= gain["onset"]:
                v *= gain["g"]
            row[d] = v
        r_series[e] = row
    measured = {d: sorted(set(sts) - dropped[d]) for d in days}
    return {"carrier": ck, "days": days, "stations": sts,
            "edges": eds, "r": r_series, "measured": measured,
            "segments": geom["segments"][ck]}


def panel_views(geom, raw):
    """Per-engine panel shapes from one raw replicate."""
    ck = raw["carrier"]
    ev0 = geom["eval_start_index"]
    eval_days = raw["days"][ev0:]
    graph_carrier = {"registered_days": list(raw["days"]),
                     "stations": raw["stations"],
                     "segments": raw["segments"], "r": raw["r"]}
    b2b_carrier = {"registry": list(raw["stations"]),
                   "registered_days": eval_days,
                   "measured": {d: raw["measured"][d]
                                for d in eval_days},
                   "r": {e: {d: v for d, v in ser.items()
                             if d in set(eval_days)}
                         for e, ser in raw["r"].items()}}
    return {"pb": {"carriers": {ck: graph_carrier}},
            "b2b": {"calendar": eval_days,
                    "carriers": {ck: b2b_carrier}},
            "b1b": {"calendar": list(raw["days"]),
                    "carriers": {ck: {"registry": raw["stations"],
                                      "registered_days": raw["days"],
                                      "r": raw["r"]}}}}


# ------------------------------------------------ replicate -> Holm
def replicate_pvalues(geom, views, n_draws, doc_sha):
    """All four families on the same replicate panel. Typed/None p is
    carried as None (non-rejection, m stays 4 -- disclosed pin)."""
    b = geom["b1b"]
    out = {}
    r1 = _pb.b2a_family(views["pb"], doc_sha256=doc_sha,
                        n_draws=n_draws)
    out["B2A"] = r1.get("p_value")
    r2 = _pb.b3a_family(views["pb"], doc_sha256=doc_sha,
                        n_draws=n_draws)
    out["B3A"] = r2.get("p_value")
    r3 = _b2b.w2_b2b_family(views["b2b"], doc_sha256=doc_sha,
                            n_draws=n_draws)
    out["B2B"] = r3.get("p_value")
    r4 = _b1b.w2_b1b_family(views["b1b"], doc_sha256=doc_sha,
                            n_draws=n_draws, n_blocks=b["n_blocks"],
                            block_len=b["block_len"],
                            baseline_positions=b["baseline_positions"],
                            testable_min=b["testable_min"])
    out["B1B"] = r4.get("p_value")
    return out


def holm_rejects(pvals, alpha=ALPHA_HOLM):
    """Holm over the FULL four-member graph; None p = non-reject with
    m held at 4."""
    m = len(GRAPH)
    known = sorted((h for h in GRAPH if pvals.get(h) is not None),
                   key=lambda h: pvals[h])
    rejected = set()
    still = True
    for i, h in enumerate(known):
        if still and pvals[h] <= alpha / (m - i):
            rejected.add(h)
        else:
            still = False
    return rejected


def run_point(geom, family, point, *, seed_root, n_draws,
              r_first=R_FIRST, r_max=R_MAX):
    """FIXTURE-ONLY detection record (codex 1358Z item 1: the fixture
    and certification entry points are SEPARATE; this path can never
    emit certifiable records -- there is no tier knob to turn)."""
    def success(r):
        raw = make_panel(geom, family, point, r, seed_root)
        pv = replicate_pvalues(geom, panel_views(geom, raw), n_draws,
                               seed_root)
        return family in holm_rejects(pv)
    rec = certify(success, r_first=r_first, r_max=r_max)
    rec.update(family=family, point=point, tier=TIER_LABEL_FIXTURE,
               n_draws=int(n_draws), certifiable=False)
    return rec


def run_artifact_class(geom, point, *, seed_root, n_draws, R):
    """FIXTURE-ONLY B1B specificity record (same separation rule).
    Rate = observed proportion over R (disclosed pin); counts +
    per-replicate outcomes recorded."""
    positives = 0
    outcomes = []
    for r in range(int(R)):
        raw = make_panel(geom, "B1B", point, r, seed_root)
        pv = replicate_pvalues(geom, panel_views(geom, raw), n_draws,
                               seed_root)
        rej = holm_rejects(pv)
        outcomes.append(sorted(rej))
        if rej:
            positives += 1
    rate = positives / int(R)
    return {"class": "B1B_GAIN_STEP_SPECIFICITY", "point": point,
            "R": int(R), "positives": positives, "rate": rate,
            "passes": rate <= ARTIFACT_MAX_RATE,
            "max_rate": ARTIFACT_MAX_RATE,
            "tier": TIER_LABEL_FIXTURE, "outcomes": outcomes,
            "certifiable": False}


# ===================================================================
# CERTIFICATION PATH (codex 1358Z items 1+2; calendar v2 per codex
# 2026-08-23T1400Z ruling 1). Separate entry points; BOUND geometry
# capsules; the registered generator draw order and master/substream
# grammar; the PINNED NON-cal engine seams over the window-2 fixed
# 192-position authority grid (w2-calendar-v2-noncal).
# ===================================================================
BOUND_GEOMETRY_SCHEMA = "f2g-w2-bound-geometry-v2"
CERT_N_DRAWS = 9999

# window-2 calendar authority v2 (codex 1400Z ruling 1; owner PRESTART
# = 2026-08-26). AUTHORITY constants: the committed authority artifact
# and every capsule must match these byte-for-byte -- they are never
# fallbacks and never derived from Phase-B geometry.
W2_FRAME_ID = "w2-calendar-v2-noncal"
W2_BASELINE_START, W2_BASELINE_END = "2026-06-27", "2026-08-25"
W2_EXCLUDED_DAY = "2026-08-26"
W2_EVAL_START, W2_EVAL_END = "2026-08-27", "2027-01-05"
W2_BASELINE_COUNT, W2_EVAL_COUNT = 60, 132
W2_ENGINE_POSITIONS = 192
W2_B1B_N_BLOCKS, W2_B1B_BLOCK_LEN = 16, 12
W2_B1B_BASELINE_POSITIONS = 60
CALENDAR_FRAME_FIELDS = {"frame_id", "baseline_days", "excluded_days",
                         "evaluation_days", "engine_days", "b1b"}
CALENDAR_B1B_FIELDS = {"n_blocks", "block_len", "baseline_positions"}
CARRIER_MASK_FIELDS = {"registered_days", "available_days"}


def _daterange(a, b):
    """Inclusive ISO calendar-day range."""
    import datetime as _dt
    d0, d1 = _dt.date.fromisoformat(a), _dt.date.fromisoformat(b)
    out = []
    while d0 <= d1:
        out.append(d0.isoformat())
        d0 += _dt.timedelta(days=1)
    return out


def days_digest(days):
    """The registered day-list digest convention: sha256 of the
    compact-JSON array of ISO strings."""
    return hashlib.sha256(json.dumps(
        list(days), separators=(",", ":")).encode()).hexdigest()


def w2_calendar_frame():
    """The window-2 calendar frame derived from the authority
    constants. KAT 1 asserts this equals the COMMITTED authority
    artifact byte-for-byte (two independent derivations)."""
    baseline = _daterange(W2_BASELINE_START, W2_BASELINE_END)
    ev = _daterange(W2_EVAL_START, W2_EVAL_END)
    return {"frame_id": W2_FRAME_ID,
            "baseline_days": baseline,
            "excluded_days": [W2_EXCLUDED_DAY],
            "evaluation_days": ev,
            "engine_days": baseline + ev,
            "b1b": {"n_blocks": W2_B1B_N_BLOCKS,
                    "block_len": W2_B1B_BLOCK_LEN,
                    "baseline_positions": W2_B1B_BASELINE_POSITIONS}}


def _validate_calendar_frame(frame):
    """codex 1400Z ruling 1: the fixed authority grid. Every shifted,
    extra, or missing authority date refuses BEFORE generation or an
    engine call; the PRESTART day is never an engine position."""
    def refuse(detail):
        raise PowerHarnessError(f"CALENDAR_AUTHORITY_MISMATCH: {detail}")
    if not isinstance(frame, dict) or \
            set(frame) != CALENDAR_FRAME_FIELDS:
        refuse("frame schema not closed")
    if frame["frame_id"] != W2_FRAME_ID:
        refuse(f"frame_id {frame['frame_id']!r} != {W2_FRAME_ID!r}")
    b, x, ev, eng = (frame["baseline_days"], frame["excluded_days"],
                     frame["evaluation_days"], frame["engine_days"])
    if len(b) != W2_BASELINE_COUNT or len(ev) != W2_EVAL_COUNT or \
            len(eng) != W2_ENGINE_POSITIONS or \
            x != [W2_EXCLUDED_DAY]:
        refuse(f"counts {len(b)}/{len(x)}/{len(ev)}/{len(eng)} != "
               f"{W2_BASELINE_COUNT}/1/{W2_EVAL_COUNT}/"
               f"{W2_ENGINE_POSITIONS}")
    if b[0] != W2_BASELINE_START or b[-1] != W2_BASELINE_END or \
            ev[0] != W2_EVAL_START or ev[-1] != W2_EVAL_END:
        refuse("endpoint mismatch")
    if b != _daterange(W2_BASELINE_START, W2_BASELINE_END) or \
            ev != _daterange(W2_EVAL_START, W2_EVAL_END):
        refuse("shifted/extra/missing authority date")
    if eng != b + ev:
        refuse("engine_days != baseline_days || evaluation_days")
    if W2_EXCLUDED_DAY in eng:
        refuse("PRESTART day appears as an engine position")
    b1b = frame["b1b"]
    if not isinstance(b1b, dict) or set(b1b) != CALENDAR_B1B_FIELDS:
        refuse("b1b authority fields not closed")
    if (b1b["n_blocks"], b1b["block_len"],
            b1b["baseline_positions"]) != \
            (W2_B1B_N_BLOCKS, W2_B1B_BLOCK_LEN,
             W2_B1B_BASELINE_POSITIONS):
        refuse("b1b authority field mismatch")
    if b1b["n_blocks"] * b1b["block_len"] != len(eng) or \
            b1b["baseline_positions"] != len(b) or \
            b1b["baseline_positions"] % b1b["block_len"] != 0:
        refuse("b1b block alignment broken")
    assert b1b["baseline_positions"] // b1b["block_len"] == 5
    assert (len(eng) - b1b["baseline_positions"]) \
        // b1b["block_len"] == 11
    return frame


def _validate_carrier_mask(ck, mask, frame):
    """Non-compression contract: the engine-facing registered_days is
    ALWAYS the full fixed grid; availability is a separate mask; an
    unavailable date keeps its calendar position and is never deleted
    or compacted; the PRESTART day admits no mask entry or value."""
    if not isinstance(mask, dict) or set(mask) != CARRIER_MASK_FIELDS:
        raise PowerHarnessError(
            f"CALENDAR_MASK_COMPRESSION: {ck} mask schema not closed "
            "(registered_days + available_days)")
    if list(mask["registered_days"]) != list(frame["engine_days"]):
        raise PowerHarnessError(
            f"CALENDAR_MASK_COMPRESSION: {ck} registered_days != "
            "engine_days byte-for-byte (missing availability never "
            "compacts the fixed grid)")
    av = [str(d) for d in mask["available_days"]]
    if av != sorted(av) or len(set(av)) != len(av):
        raise PowerHarnessError(
            f"CALENDAR_AUTHORITY_MISMATCH: {ck} available_days "
            "unordered/duplicated")
    for d in av:
        if d in frame["excluded_days"]:
            raise PowerHarnessError(
                f"CALENDAR_EXCLUDED_DAY: {ck} carries availability on "
                f"the PRESTART day {d}")
    off = set(av) - set(frame["engine_days"])
    if off:
        raise PowerHarnessError(
            f"CALENDAR_AUTHORITY_MISMATCH: {ck} available_days "
            f"outside the authority grid: {sorted(off)}")


GEOMETRY_CAPSULE_FIELDS = {
    "schema", "bound", "calendar_authority_mode",
    "calendar_authority_sha256", "calendar_authority_ref",
    "seed_authority_sha256", "calendar_frame", "carrier_masks",
    "registries", "segments", "effect_grids",
    "loco_registry_carrier", "capsule_digest"}


def _geometry_capsule_digest(capsule):
    body = {k: capsule[k] for k in sorted(capsule)
            if k != "capsule_digest"}
    return hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def _validate_geometry_capsule(capsule, family, point):
    """codex 1815Z item 2: CLOSED capsule schema, bound-mode-only,
    recomputed whole-capsule digest, structural geometry checks, and
    family/point membership in the capsule's REGISTERED effect grids.
    All refusals fire BEFORE any replicate runs."""
    def refuse(code, detail):
        raise PowerHarnessError(f"{code}: {detail}")
    if not isinstance(capsule, dict) or \
            set(capsule) != GEOMETRY_CAPSULE_FIELDS:
        refuse("POWER_GEOMETRY_UNBOUND",
               "capsule schema not closed: "
               f"{sorted(set(capsule) ^ GEOMETRY_CAPSULE_FIELDS) if isinstance(capsule, dict) else type(capsule).__name__}")
    if capsule["schema"] != BOUND_GEOMETRY_SCHEMA or \
            capsule["bound"] is not True:
        refuse("POWER_GEOMETRY_UNBOUND", "schema/bound flag invalid")
    if capsule["calendar_authority_mode"] != "bound":
        refuse("POWER_GEOMETRY_UNBOUND",
               f"mode {capsule['calendar_authority_mode']!r} is not "
               "'bound' (fixture mode never certifies)")
    if _geometry_capsule_digest(capsule) != capsule["capsule_digest"]:
        refuse("POWER_GEOMETRY_UNBOUND", "capsule digest mismatch")
    _validate_calendar_frame(capsule["calendar_frame"])
    for ck in sorted(capsule["carrier_masks"]):
        _validate_carrier_mask(ck, capsule["carrier_masks"][ck],
                               capsule["calendar_frame"])
    if sorted(capsule["carrier_masks"]) != \
            sorted(capsule["registries"]) or \
            sorted(capsule["carrier_masks"]) != \
            sorted(capsule["segments"]):
        refuse("POWER_GEOMETRY_UNBOUND",
               "carrier set mismatch across masks/registries/segments")
    grid = capsule["effect_grids"].get(family)
    if grid is None or point not in grid:
        refuse("POWER_POINT_OFF_GRID",
               f"{family} {point} not in the registered effect grid")
    return capsule


def _load_bound_geometry(repo, geometry_ref):
    """The content-addressed, MANIFEST-PINNED capsule loader: the ref
    names {manifest_commit, path}; the path must be a BOUND pin of the
    execution manifest at that commit; the capsule bytes reopen from
    the pin's git object and must hash to the pin. Caller dicts are
    structurally impossible here."""
    if not isinstance(geometry_ref, dict) or \
            set(geometry_ref) != {"manifest_commit", "path"}:
        raise PowerHarnessError(
            "POWER_GEOMETRY_REF_INVALID: certification takes a "
            "{manifest_commit, path} reference, never a capsule dict")
    import subprocess
    mc = geometry_ref["manifest_commit"]
    p = subprocess.run(
        ["git", "-C", repo, "cat-file", "blob",
         f"{mc}:docs/f2g_window2_execution/execution_manifest.json"],
        capture_output=True)
    if p.returncode != 0:
        raise PowerHarnessError(
            f"POWER_GEOMETRY_REF_INVALID: manifest unreadable at {mc}")
    man = json.loads(p.stdout.decode("utf-8"))
    pin = None
    for slot in man["slots"].values():
        if slot["status"] != "BOUND":
            continue
        for cand in slot["pins"]:
            if cand["path"] == geometry_ref["path"]:
                pin = cand
    if pin is None:
        raise PowerHarnessError(
            f"POWER_GEOMETRY_NOT_MANIFEST_PINNED: "
            f"{geometry_ref['path']} is not a BOUND pin at {mc}")
    p = subprocess.run(["git", "-C", repo, "cat-file", "blob",
                        f"{pin['commit']}:{pin['path']}"],
                       capture_output=True)
    if p.returncode != 0 or hashlib.sha256(p.stdout).hexdigest() != \
            pin["blob_sha256"]:
        raise PowerHarnessError(
            "POWER_GEOMETRY_UNBOUND: pinned capsule bytes unreadable "
            "or divergent")
    try:
        return json.loads(p.stdout.decode("utf-8"))
    except ValueError:
        raise PowerHarnessError(
            "POWER_GEOMETRY_UNBOUND: pinned bytes are not a capsule")


def rep_seed_registered(seed_authority_sha, family, r):
    """The REGISTERED master/substream grammar, verbatim from the
    pinned power instrument: one master PCG64 per (authority, family)
    seeded via derive_substream_seed(auth, family, 'full', 'power');
    replicate r's seed = the r-th sequential int64 draw."""
    master = np.random.Generator(np.random.PCG64(
        _pb.derive_substream_seed(seed_authority_sha, family, "full",
                                  "power")))
    return int(master.integers(0, 2 ** 63, size=r + 1,
                               dtype=np.int64)[r])


def make_bound_panels(capsule, family, point, r, inject=True):
    """Joint ALL-carrier generation over the bound geometry in the
    REGISTERED draw order (codex item 2), on the v2 FIXED authority
    grid: one G over the full 192-position engine calendar FIRST,
    then per SORTED carrier the station noise, edge noise, and MCAR
    arrays over that carrier's AVAILABLE days (the separate bound
    availability mask -- an unavailable date keeps its calendar
    position with no value; the grid is never compacted). Injection
    classes apply to the first sorted carrier (the pinned instrument's
    convention). Typed absences are never synthesized.
    Returns (panel, w2_views, debug)."""
    frame = _validate_calendar_frame(capsule["calendar_frame"])
    for _ck in sorted(capsule["carrier_masks"]):
        _validate_carrier_mask(_ck, capsule["carrier_masks"][_ck],
                               frame)
    cal = [str(d) for d in frame["engine_days"]]
    cpos = {d: i for i, d in enumerate(cal)}
    carriers = sorted(capsule["carrier_masks"])
    rng = np.random.Generator(np.random.PCG64(rep_seed_registered(
        capsule["seed_authority_sha256"], family, r)))
    G = rng.standard_normal(len(cal))
    lat = {}
    for ck in carriers:
        sts = list(capsule["registries"][ck])
        eds = _edges_of(sts)
        days = [str(d) for d in
                capsule["carrier_masks"][ck]["available_days"]]
        s = rng.normal(0.0, SIGMA_S, size=(len(sts), len(days)))
        eps = rng.normal(0.0, SIGMA_E, size=(len(eds), len(days)))
        mcar = rng.random((len(eds), len(days))) < MCAR
        six = {st: i for i, st in enumerate(sts)}
        gvec = np.array([G[cpos[d]] for d in days])
        u = np.empty((len(eds), len(days)))
        for j, e in enumerate(eds):
            a, b = e.split("|")
            u[j] = MU0 + GAMMA * gvec + s[six[a]] + s[six[b]] + eps[j]
        lat[ck] = {"u": u, "mcar": mcar, "edges": eds, "days": days,
                   "stations": sts}
    ck0 = carriers[0]
    L0 = lat[ck0]
    ev0_pos = len(frame["baseline_days"])          # 60 (authority)
    day_pos = {d: i for i, d in enumerate(L0["days"])}
    dropped = {d: set() for d in L0["days"]}
    gain = {}
    ev_days0 = [d for d in L0["days"] if cpos[d] >= ev0_pos]
    if family in ("B2A", "B2B") and inject and \
            int(point.get("m", 0)) > 0:
        m = int(point["m"])
        half = len(L0["stations"]) // 2
        onset_day = ev_days0[max(1, len(ev_days0) // 3)]
        onset = day_pos[onset_day]
        block = {st: (0 if i < half else 1)
                 for i, st in enumerate(L0["stations"])}
        swapped = dict(block)
        for st in L0["stations"][half - m:half]:
            swapped[st] = 1
        for st in L0["stations"][half:half + m]:
            swapped[st] = 0
        for j, e in enumerate(L0["edges"]):
            a, b = e.split("|")
            L0["u"][j, :onset] += 0.9 if block[a] == block[b] else -0.5
            L0["u"][j, onset:] += 0.9 if swapped[a] == swapped[b] \
                else -0.5
    if family == "B2B" and float(point.get("dropout", 0.0)) > 0.0:
        rate = float(point["dropout"])
        for d in ev_days0:
            for st in L0["stations"]:
                if rng.random() < rate:
                    dropped[d].add(st)
    if family == "B1B" and inject:
        if "gain" in point:
            gain = {"station": L0["stations"][int(
                rng.integers(0, len(L0["stations"])))],
                "g": float(point["gain"]),
                "onset": day_pos[ev_days0[max(1, len(ev_days0) // 3)]]}
        else:
            k = int(point["k"])
            n_e = int(point["n_e"])
            d_ = float(point["delta_lat"])
            starts = [i for i, d in enumerate(L0["days"])
                      if cpos[d] >= ev0_pos]
            smax = max(1, len(starts) - k)
            s0 = starts[0] + int(rng.integers(0, smax))
            for e in L0["edges"][:n_e]:
                j = L0["edges"].index(e)
                L0["u"][j, s0:s0 + k] += d_
    panel_carriers = {}
    measured = {}
    for ck in carriers:
        L = lat[ck]
        vals = np.tanh(L["u"])
        rr = {}
        for j, e in enumerate(L["edges"]):
            a, b = e.split("|")
            row = {}
            for t, d in enumerate(L["days"]):
                if L["mcar"][j, t]:
                    continue
                if ck == ck0 and (a in dropped.get(d, ())
                                  or b in dropped.get(d, ())):
                    continue
                v = float(vals[j, t])
                if ck == ck0 and gain and gain["station"] in (a, b) \
                        and t >= gain["onset"]:
                    v *= gain["g"]
                row[d] = v
            rr[e] = row
        # engine-facing registered_days = the FULL fixed grid (the
        # non-compression contract); availability lives only in which
        # dates carry values
        panel_carriers[ck] = {
            "registered_days": list(cal),
            "stations": L["stations"],
            "segments": dict(capsule["segments"][ck]), "r": rr}
        measured[ck] = {d: sorted(set(L["stations"])
                                  - (dropped.get(d, set())
                                     if ck == ck0 else set()))
                        for d in L["days"]}
    panel = {"schema": "w2-noncal-panel-v2",
             "frame_id": frame["frame_id"],
             "carriers": panel_carriers}
    eval_cal = list(frame["evaluation_days"])      # exactly the 132
    av = {ck: set(lat[ck]["days"]) for ck in carriers}
    views = {"b2b": {"calendar": eval_cal, "carriers": {}},
             "b1b": {"calendar": cal, "carriers": {}}}
    ev_set = set(eval_cal)
    for ck in carriers:
        c = panel_carriers[ck]
        # per-carrier availability mask: unavailable eval days stay
        # OUT of the carrier's registered set -> typed
        # NO_REGISTERED_SNAPSHOT positions inside the engine
        av_eval = [d for d in eval_cal if d in av[ck]]
        views["b2b"]["carriers"][ck] = {
            "registry": list(c["stations"]),
            "registered_days": av_eval,
            "measured": {d: measured[ck][d] for d in av_eval},
            "r": {e: {d: v for d, v in ser.items() if d in ev_set}
                  for e, ser in c["r"].items()}}
        views["b1b"]["carriers"][ck] = {
            "registry": list(c["stations"]),
            "registered_days": sorted(av[ck]),
            "r": c["r"]}
    return panel, views, {"G": G, "carriers": carriers,
                          "frame": frame}


def _pinned_engine_blob_sha():
    with open(os.path.join(_HERE, "d2_f2g_phase_b_stats.py"),
              "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _mask_digest(capsule):
    return hashlib.sha256(json.dumps(
        {ck: [str(d) for d in
              capsule["carrier_masks"][ck]["available_days"]]
         for ck in sorted(capsule["carrier_masks"])},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _panel_digest(panel):
    return hashlib.sha256(json.dumps(
        panel, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


ENTRYPOINTS_V2 = {
    "B2A": "d2_f2g_phase_b_stats.b2a_family",
    "B3A": "d2_f2g_phase_b_stats.b3a_family",
    "B2B": "w2_b2b.w2_b2b_family",
    "B1B": "w2_b1b.w2_b1b_family"}


def replicate_pvalues_bound(panel, views, n_draws, doc_sha, capsule):
    """The v2 engine seams (codex 1400Z ruling 1): B2A/B3A via the
    PINNED NON-cal entry points over the fixed 192-position authority
    grid (positional walk_forward_split at baseline 60); B2B over
    exactly the 132 evaluation days; B1B geometry from the AUTHORITY
    fields (never fallbacks). Wrapper records carry the frame
    metadata the non-cal results lack; the certification artifact
    refuses absent/divergent records (never trusts a caller label)."""
    frame = capsule["calendar_frame"]
    b1bg = frame["b1b"]
    base_rec = {
        "engine_blob_sha256": _pinned_engine_blob_sha(),
        "frame_id": frame["frame_id"],
        "baseline_days_sha256": days_digest(frame["baseline_days"]),
        "evaluation_days_sha256":
            days_digest(frame["evaluation_days"]),
        "mask_sha256": _mask_digest(capsule),
        "input_panel_sha256": _panel_digest(panel)}
    out, frames = {}, {}
    r1 = _pb.b2a_family(panel, doc_sha256=doc_sha, n_draws=n_draws)
    out["B2A"] = r1.get("p_value")
    frames["B2A"] = dict(base_rec, entrypoint=ENTRYPOINTS_V2["B2A"])
    r2 = _pb.b3a_family(panel, doc_sha256=doc_sha, n_draws=n_draws)
    out["B3A"] = r2.get("p_value")
    frames["B3A"] = dict(base_rec, entrypoint=ENTRYPOINTS_V2["B3A"])
    r3 = _b2b.w2_b2b_family(views["b2b"], doc_sha256=doc_sha,
                            n_draws=n_draws)
    out["B2B"] = r3.get("p_value")
    frames["B2B"] = dict(base_rec, entrypoint=ENTRYPOINTS_V2["B2B"],
                         engine_frame=r3.get("frame"))
    r4 = _b1b.w2_b1b_family(views["b1b"], doc_sha256=doc_sha,
                            n_draws=n_draws,
                            n_blocks=b1bg["n_blocks"],
                            block_len=b1bg["block_len"],
                            baseline_positions=
                                b1bg["baseline_positions"])
    out["B1B"] = r4.get("p_value")
    frames["B1B"] = dict(base_rec, entrypoint=ENTRYPOINTS_V2["B1B"],
                         engine_frame=r4.get("frame"),
                         b1b_geometry=dict(b1bg))
    return out, frames


def _validate_frame_records(frames, capsule, panel):
    """codex 1400Z ruling 1: the certification artifact refuses
    absent or divergent frame metadata. Every expected digest is
    RECOMPUTED here (engine blob from disk, day arrays from the
    capsule frame, mask from the capsule, panel from the input) --
    a record can only pass by matching the recomputation."""
    frame = capsule["calendar_frame"]
    expect = {
        "engine_blob_sha256": _pinned_engine_blob_sha(),
        "frame_id": frame["frame_id"],
        "baseline_days_sha256": days_digest(frame["baseline_days"]),
        "evaluation_days_sha256":
            days_digest(frame["evaluation_days"]),
        "mask_sha256": _mask_digest(capsule),
        "input_panel_sha256": _panel_digest(panel)}
    for fam, ep in ENTRYPOINTS_V2.items():
        rec = frames.get(fam)
        if not isinstance(rec, dict):
            raise PowerHarnessError(
                f"POWER_CALENDAR_FRAME_INVALID: {fam} frame record "
                "absent")
        if rec.get("entrypoint") != ep:
            raise PowerHarnessError(
                f"POWER_CALENDAR_FRAME_INVALID: {fam} entrypoint "
                f"{rec.get('entrypoint')!r} != {ep!r}")
        for k, v in expect.items():
            if rec.get(k) != v:
                raise PowerHarnessError(
                    f"POWER_CALENDAR_FRAME_INVALID: {fam} {k} "
                    "absent or divergent")
    if frames["B1B"].get("b1b_geometry") != frame["b1b"]:
        raise PowerHarnessError(
            "POWER_CALENDAR_FRAME_INVALID: B1B geometry record "
            "diverges from the authority fields")
    return True


def b1b_loco_project(b1b_view, station):
    """LOCO fold projection (amendment v1): remove the named station
    and its incident edges from the B1B view of the SAME replicate --
    every other raw value byte-identical; no panel regeneration."""
    out = {"calendar": list(b1b_view["calendar"]), "carriers": {}}
    for ck, c in b1b_view["carriers"].items():
        reg = [s for s in c["registry"] if s != station]
        out["carriers"][ck] = {
            "registry": reg,
            "registered_days": list(c["registered_days"]),
            "r": {e: dict(ser) for e, ser in c["r"].items()
                  if station not in e.split("|")}}
    return out


def verify_fold_set(folds_run, loco_registry):
    """Amendment v1: the fold set must be EXACTLY the NEW registry --
    missing, extra, duplicate, or wrong-station folds refuse the
    certification artifact (an audit failure, never an ordinary
    non-recovery)."""
    want = sorted(loco_registry)
    if sorted(folds_run) != want or len(folds_run) != len(want):
        raise PowerHarnessError(
            f"POWER_LOCO_FOLD_SET_INVALID: ran {sorted(folds_run)} "
            f"!= registry {want}")
    return True


def _b1b_loco_recovery(views, pv_full, capsule, n_draws, doc_sha,
                       fold_counter=None):
    """recover_B1B per the amendment: full-Holm rejection AND every
    same-replicate fold-substituted Holm rejection. Early-exit without
    folds on full-Holm non-rejection. Typed/no-p fold => False."""
    if "B1B" not in holm_rejects(pv_full):
        return False
    loco_ck = capsule["loco_registry_carrier"]
    registry = capsule["registries"][loco_ck]
    folds_run = []
    ok = True
    for s in sorted(registry):
        folds_run.append(s)
        if fold_counter is not None:
            fold_counter.append(s)
        proj = b1b_loco_project(views["b1b"], s)
        # geometry = the AUTHORITY fields (codex 1400Z: never
        # fallbacks); a view that cannot block-align against them is
        # a typed structural refusal below
        b = capsule["calendar_frame"]["b1b"]
        try:
            r_s = _b1b.w2_b1b_family(
                proj, doc_sha256=doc_sha, n_draws=n_draws,
                fold=f"loco:{s}",
                n_blocks=b["n_blocks"],
                block_len=b["block_len"],
                baseline_positions=b["baseline_positions"])
            p_s = r_s.get("p_value")
        except _b1b.PanelInvalid:
            p_s = None        # typed structural refusal = no-p class
        if p_s is None:
            ok = False        # typed no-p = non-recovery, keep folds
            continue          # running so the fold SET stays exact
        if "B1B" not in holm_rejects(dict(pv_full, B1B=p_s)):
            ok = False
    verify_fold_set(folds_run, registry)
    return ok


def run_point_certification(repo, geometry_ref, family, point,
                            **overrides):
    """THE ONLY PATH to certifiable records (codex 1815Z item 2 shape):
    takes a {manifest_commit, path} REFERENCE to a manifest-pinned
    geometry capsule -- never a caller dict. Reopens the exact pinned
    bytes, closes the capsule schema, recomputes the whole-capsule
    digest, requires bound mode, verifies the calendar-authority bytes
    against the capsule's recorded sha, and validates the family/point
    against the capsule's REGISTERED effect grids -- ALL before any
    replicate. Constructs (never accepts) R=20/40, n_draws=9,999, and
    the registered seed grammar. Results bind the capsule digest and
    point."""
    if overrides:
        raise PowerHarnessError(
            f"POWER_CERTIFICATION_CONFIG_UNBOUND: "
            f"{sorted(overrides)} -- certification constructs its "
            "own R/n_draws/seed authority")
    capsule = _load_bound_geometry(repo, geometry_ref)
    _validate_geometry_capsule(capsule, family, point)
    # calendar-authority bytes must reopen and hash to the capsule's
    # recorded sha (the window-2 bound authority is a PRESTART
    # deliverable -- until committed, this refuses, which is honest)
    import subprocess
    ref = capsule["calendar_authority_ref"]
    p = subprocess.run(["git", "-C", repo, "cat-file", "blob",
                        f"{ref['commit']}:{ref['path']}"],
                       capture_output=True)
    if p.returncode != 0 or hashlib.sha256(p.stdout).hexdigest() != \
            capsule["calendar_authority_sha256"]:
        raise PowerHarnessError(
            "POWER_GEOMETRY_UNBOUND: calendar authority bytes absent "
            "or divergent from the capsule's recorded sha")
    try:
        auth = json.loads(p.stdout.decode("utf-8"))
    except ValueError:
        raise PowerHarnessError(
            "CALENDAR_AUTHORITY_MISMATCH: authority bytes are not a "
            "calendar-authority artifact")
    if auth.get("schema") != "f2g-w2-calendar-authority-v2" or \
            auth.get("frame") != capsule["calendar_frame"]:
        raise PowerHarnessError(
            "CALENDAR_AUTHORITY_MISMATCH: capsule frame diverges "
            "from the committed authority artifact")
    doc_sha = capsule["seed_authority_sha256"]

    def success(r):
        panel, views, _dbg = make_bound_panels(capsule, family,
                                               point, r)
        pv, frames = replicate_pvalues_bound(panel, views,
                                             CERT_N_DRAWS, doc_sha,
                                             capsule)
        _validate_frame_records(frames, capsule, panel)
        if family == "B1B" and "gain" not in point:
            # amendment v1: detection-class B1B recovery = full-Holm
            # AND every same-replicate LOCO-substituted Holm. The
            # gain-step SPECIFICITY class stays pre-LOCO (anti-rescue)
            # and never reaches this branch (run_artifact_class owns
            # it; a detection run on a gain point is off-grid anyway).
            return _b1b_loco_recovery(views, pv, capsule,
                                      CERT_N_DRAWS, doc_sha)
        return family in holm_rejects(pv)
    rec = certify(success, r_first=R_FIRST, r_max=R_MAX)
    frame = capsule["calendar_frame"]
    rec.update(family=family, point=point, tier="CERTIFICATION",
               n_draws=CERT_N_DRAWS, certifiable=True,
               geometry_capsule_digest=capsule["capsule_digest"],
               geometry_ref=dict(geometry_ref),
               seed_authority_sha256=doc_sha,
               calendar_authority_sha256=
                   capsule["calendar_authority_sha256"],
               calendar_frame_id=frame["frame_id"],
               baseline_days_sha256=
                   days_digest(frame["baseline_days"]),
               evaluation_days_sha256=
                   days_digest(frame["evaluation_days"]),
               engine_blob_sha256=_pinned_engine_blob_sha())
    return rec


# ---------------------------------------------------------------- selftest
def _selftest():
    # stopping rule: the three clear terminals
    rec = certify(lambda r: True)
    assert rec["status"] == "CERTIFIED" and rec["R"] == R_FIRST
    assert rec["lb"] >= CP_FLOOR
    rec = certify(lambda r: False)
    assert rec["status"] == "FAILED" and rec["R"] == R_FIRST
    seq = [True] * 17 + [False] * 3 + [True] * 17 + [False] * 3
    rec = certify(lambda r: seq[r])
    assert rec["status"] in ("CERTIFIED", "FAILED",
                             "CANNOT_DETERMINE_POWER_ESTIMATE")
    assert len(rec["trace"]) == 2 and rec["R"] == R_MAX
    assert cp_lb(20, 20) > CP_FLOOR and cp_ub(0, 20) < CP_FLOOR
    assert cp_lb(10, 20) < cp_lb(19, 20)          # monotone

    # generator determinism + injection non-identity
    geom = fixture_geometry()
    raw1 = make_panel(geom, "B2B", {"m": 2}, 0, "seedroot")
    raw2 = make_panel(geom, "B2B", {"m": 2}, 0, "seedroot")
    d1 = hashlib.sha256(json.dumps(raw1["r"], sort_keys=True)
                        .encode()).hexdigest()
    assert d1 == hashlib.sha256(json.dumps(raw2["r"], sort_keys=True)
                                .encode()).hexdigest()
    raw0 = make_panel(geom, "B2B", {"m": 2}, 0, "seedroot",
                      inject=False)
    assert d1 != hashlib.sha256(json.dumps(raw0["r"], sort_keys=True)
                                .encode()).hexdigest()

    # dropout produces producer-consistent thinning (measured shrinks
    # AND incident edges vanish those days)
    rawd = make_panel(geom, "B2B", {"m": 0, "dropout": 0.25}, 1,
                      "seedroot")
    ev_days = rawd["days"][geom["eval_start_index"]:]
    thinned = [d for d in ev_days
               if len(rawd["measured"][d]) < len(rawd["stations"])]
    assert thinned, "25% dropout must thin some eval days"
    for d in thinned[:3]:
        absent = set(rawd["stations"]) - set(rawd["measured"][d])
        for e, ser in rawd["r"].items():
            a, b = e.split("|")
            if a in absent or b in absent:
                assert d not in ser

    # gain-step artifact scales exactly the target's incident eval
    # edges (mechanism, not power)
    rawg = make_panel(geom, "B1B", {"gain": 10.0}, 2, "seedroot")
    rawn = make_panel(geom, "B1B", {"gain": 10.0}, 2, "seedroot",
                      inject=False)
    changed = {e for e in rawg["r"]
               if rawg["r"][e] != rawn["r"][e]}
    stations_in_changed = set()
    for e in changed:
        stations_in_changed.update(e.split("|"))
    common = set.intersection(*[set(e.split("|")) for e in changed])
    assert len(common) == 1, "gain must localize to ONE station"

    # replicate -> four p-values -> Holm; null-ish panel rarely rejects
    raw = make_panel(geom, "B2A", {"m": 0}, 3, "seedroot")
    pv = replicate_pvalues(geom, panel_views(geom, raw), 199,
                           "ab" * 32)
    assert set(pv) == set(GRAPH)
    assert all(v is None or 0.0 < v <= 1.0 for v in pv.values())
    # Holm hand cases incl None handling (m stays 4)
    assert holm_rejects({"B1B": 0.001, "B2A": 0.02, "B2B": 0.3,
                         "B3A": None}) == {"B1B"}
    assert holm_rejects({"B1B": 0.012, "B2A": 0.013, "B2B": 0.016,
                         "B3A": 0.04}) == {"B1B", "B2A", "B2B",
                                           "B3A"}
    assert holm_rejects({h: None for h in GRAPH}) == set()

    # end-to-end fixture-tier records (tiny R + n_draws; MECHANISM
    # only -- the tier label forbids certification claims)
    rec = run_point(geom, "B2B", {"m": 3}, seed_root="sr",
                    n_draws=99, r_first=4, r_max=6)
    assert rec["tier"] == TIER_LABEL_FIXTURE \
        and rec["certifiable"] is False
    assert rec["status"] in ("CERTIFIED", "FAILED",
                             "CANNOT_DETERMINE_POWER_ESTIMATE")
    art = run_artifact_class(geom, {"gain": 3.0}, seed_root="sr",
                             n_draws=99, R=4)
    assert art["certifiable"] is False and 0.0 <= art["rate"] <= 1.0
    assert len(art["outcomes"]) == 4

    # --- codex item 1 doctors: NO path to certifiable from fixtures
    try:
        run_point(geom, "B2B", {"m": 1}, seed_root="sr", n_draws=99,
                  r_first=2, r_max=2, tier="CERTIFICATION")
        raise AssertionError("tier knob must not exist")
    except TypeError:
        pass
    repo_g = os.path.abspath(os.path.join(_HERE, "..", ".."))
    # caller dicts are structurally impossible (codex 1815Z item 2)
    for bad_ref, code, label in (
            ({"schema": "nope"}, "POWER_GEOMETRY_REF_INVALID",
             "caller dict"),
            (dict(fixture_geometry(), schema=BOUND_GEOMETRY_SCHEMA,
                  bound=True), "POWER_GEOMETRY_REF_INVALID",
             "forged bound dict"),
            ({"manifest_commit": "86bbb4d",
              "path": "docs/nonexistent.json"},
             "POWER_GEOMETRY_NOT_MANIFEST_PINNED", "unpinned path"),
            ({"manifest_commit": "86bbb4d",
              "path": "monitoring/src/w2_selection.py"},
             "POWER_GEOMETRY_UNBOUND", "pinned non-capsule")):
        try:
            run_point_certification(repo_g, bad_ref, "B2B", {"m": 2})
            raise AssertionError(f"{label} must refuse")
        except PowerHarnessError as e:
            assert code in str(e), (label, str(e))

    # --- calendar v2 KAT 1: exact arrays, counts, endpoints, and the
    # explicit 08-26 exclusion -- asserted against BOTH the in-module
    # derivation and the COMMITTED authority artifact (two
    # derivations, one contract; never self-consistent only)
    FRAME = w2_calendar_frame()
    ENG = FRAME["engine_days"]
    assert (len(FRAME["baseline_days"]), len(FRAME["excluded_days"]),
            len(FRAME["evaluation_days"]), len(ENG)) == (60, 1, 132,
                                                         192)
    assert FRAME["baseline_days"][0] == "2026-06-27"
    assert FRAME["baseline_days"][-1] == "2026-08-25"
    assert FRAME["excluded_days"] == ["2026-08-26"]
    assert FRAME["evaluation_days"][0] == "2026-08-27"
    assert FRAME["evaluation_days"][-1] == "2027-01-05"
    assert "2026-08-26" not in ENG
    assert ENG == FRAME["baseline_days"] + FRAME["evaluation_days"]
    assert FRAME["b1b"] == {"n_blocks": 16, "block_len": 12,
                            "baseline_positions": 60}
    _validate_calendar_frame(FRAME)
    auth_p = os.path.join(_HERE, "..", "..", "docs",
                          "f2g_window2_execution",
                          "calendar_authority_w2_v2.json")
    with open(auth_p, "rb") as f:
        auth_committed = json.loads(f.read().decode("utf-8"))
    assert auth_committed["schema"] == "f2g-w2-calendar-authority-v2"
    assert auth_committed["frame"] == FRAME, \
        "committed calendar authority diverges from the derivation"
    assert auth_committed["digests"]["engine_days_sha256"] == \
        days_digest(ENG)
    assert auth_committed["digests"]["baseline_days_sha256"] == \
        days_digest(FRAME["baseline_days"])
    assert auth_committed["digests"]["evaluation_days_sha256"] == \
        days_digest(FRAME["evaluation_days"])

    # unit validators: a well-formed bound capsule passes; every
    # doctored field refuses BEFORE any replicate
    def mk_capsule(**mut):
        cap = {"schema": BOUND_GEOMETRY_SCHEMA, "bound": True,
               "calendar_authority_mode": "bound",
               "calendar_authority_sha256": "a" * 64,
               "calendar_authority_ref": {"commit": "x", "path": "y"},
               "seed_authority_sha256": "b" * 64,
               "calendar_frame": json.loads(json.dumps(FRAME)),
               "carrier_masks": {"c1": {
                   "registered_days": list(ENG),
                   "available_days": list(ENG)}},
               "registries": {"c1": ["S0"]},
               "segments": {"c1": {"S0": "sA"}},
               "effect_grids": {"B2B": [{"m": 2}]},
               "loco_registry_carrier": "c1"}
        cap.update(mut)
        cap["capsule_digest"] = _geometry_capsule_digest(cap)
        cap.update({k: v for k, v in mut.items()
                    if k == "capsule_digest"})
        return cap
    _validate_geometry_capsule(mk_capsule(), "B2B", {"m": 2})
    for mut, code, label in (
            ({"calendar_authority_mode": "fixture"},
             "POWER_GEOMETRY_UNBOUND", "fixture mode"),
            ({"bound": False}, "POWER_GEOMETRY_UNBOUND",
             "forged bound"),
            ({"capsule_digest": "0" * 64}, "POWER_GEOMETRY_UNBOUND",
             "digest mismatch")):
        try:
            _validate_geometry_capsule(mk_capsule(**mut), "B2B",
                                       {"m": 2})
            raise AssertionError(f"{label} must refuse")
        except PowerHarnessError as e:
            assert code in str(e), (label, str(e))
    try:
        _validate_geometry_capsule(mk_capsule(), "B2B", {"m": 9})
        raise AssertionError("off-grid point must refuse")
    except PowerHarnessError as e:
        assert "POWER_POINT_OFF_GRID" in str(e)

    # --- calendar v2 KAT 2: removing an AVAILABILITY day never moves
    # the split (60/132 on the fixed grid); a compressed
    # registered_days list refuses CALENDAR_MASK_COMPRESSION
    miss_day = FRAME["baseline_days"][10]
    cap_av = mk_capsule()
    cap_av["carrier_masks"]["c1"]["available_days"] = [
        d for d in ENG if d != miss_day]
    cap_av["capsule_digest"] = _geometry_capsule_digest(cap_av)
    _validate_geometry_capsule(cap_av, "B2B", {"m": 2})
    b_sp, e_sp = _pb.walk_forward_split(
        cap_av["carrier_masks"]["c1"]["registered_days"])
    assert b_sp == FRAME["baseline_days"] and \
        e_sp == FRAME["evaluation_days"]        # grid unmoved
    # the hazard the contract closes: a compacted list silently makes
    # 2026-08-27 baseline position 60 and slides evaluation to 08-28
    comp = [d for d in ENG if d != miss_day]
    bc, ec = _pb.walk_forward_split(comp)
    assert bc[-1] == "2026-08-27" and ec[0] == "2026-08-28"
    cap_comp = mk_capsule()
    cap_comp["carrier_masks"]["c1"]["registered_days"] = comp
    cap_comp["capsule_digest"] = _geometry_capsule_digest(cap_comp)
    try:
        _validate_geometry_capsule(cap_comp, "B2B", {"m": 2})
        raise AssertionError("compressed registered_days must refuse")
    except PowerHarnessError as e:
        assert "CALENDAR_MASK_COMPRESSION" in str(e)

    # --- calendar v2 KAT 3: any 08-26 mask entry refuses (validator
    # AND before generation); shifted/extra/missing authority dates
    # refuse; off-grid availability refuses
    cap_x = mk_capsule()
    cap_x["carrier_masks"]["c1"]["available_days"] = (
        FRAME["baseline_days"] + ["2026-08-26"]
        + FRAME["evaluation_days"])
    cap_x["capsule_digest"] = _geometry_capsule_digest(cap_x)
    for fn, label in ((lambda: _validate_geometry_capsule(
            cap_x, "B2B", {"m": 2}), "validator"),
            (lambda: make_bound_panels(cap_x, "B2B", {"m": 2}, 0),
             "pre-generation")):
        try:
            fn()
            raise AssertionError(
                f"PRESTART-day availability must refuse ({label})")
        except PowerHarnessError as e:
            assert "CALENDAR_EXCLUDED_DAY" in str(e), (label, str(e))
    for fmut, label in (
            ({"baseline_days": ["2026-06-26"]
              + FRAME["baseline_days"][1:]}, "shifted"),
            ({"engine_days": ENG + ["2027-01-06"]}, "extra"),
            ({"evaluation_days": FRAME["evaluation_days"][:-1]},
             "missing")):
        cap_f = mk_capsule()
        cap_f["calendar_frame"] = dict(FRAME, **fmut)
        cap_f["capsule_digest"] = _geometry_capsule_digest(cap_f)
        try:
            _validate_geometry_capsule(cap_f, "B2B", {"m": 2})
            raise AssertionError(f"{label} authority date must refuse")
        except PowerHarnessError as e:
            assert "CALENDAR_AUTHORITY_MISMATCH" in str(e), \
                (label, str(e))
    cap_o = mk_capsule()
    cap_o["carrier_masks"]["c1"]["available_days"] = \
        list(ENG) + ["2027-02-01"]
    cap_o["capsule_digest"] = _geometry_capsule_digest(cap_o)
    try:
        _validate_geometry_capsule(cap_o, "B2B", {"m": 2})
        raise AssertionError("off-grid availability must refuse")
    except PowerHarnessError as e:
        assert "CALENDAR_AUTHORITY_MISMATCH" in str(e)

    # three-carrier BOUND-mechanism capsule (fixture calendar mode so
    # certification never opens; the REAL v2 frame with availability
    # holes exercises the fixed-grid adapter)
    holes = set(ENG[80:90])
    reg3 = {ck: [f"{ck}S{i}" for i in range(6)]
            for ck in ("c1", "c2", "c3")}
    cap3 = {"schema": BOUND_GEOMETRY_SCHEMA, "bound": True,
            "calendar_authority_sha256": "kat-cal-auth",
            "seed_authority_sha256": "kat-seed-auth",
            "calendar_authority_mode": "fixture",
            "calendar_frame": json.loads(json.dumps(FRAME)),
            "carrier_masks": {
                "c1": {"registered_days": list(ENG),
                       "available_days": list(ENG)},
                "c2": {"registered_days": list(ENG),
                       "available_days": [d for d in ENG
                                          if d not in holes]},
                "c3": {"registered_days": list(ENG),
                       "available_days": ENG[:180]}},
            "registries": reg3,
            "segments": {ck: {s: ("sA" if i < 3 else "sB")
                              for i, s in enumerate(reg3[ck])}
                         for ck in reg3}}
    # config-override doctors fire BEFORE any compute
    for ov in ({"n_draws": 99}, {"r_first": 5}, {"tier": "X"}):
        try:
            run_point_certification(
                repo_g, {"manifest_commit": "86bbb4d",
                         "path": "monitoring/src/w2_selection.py"},
                "B2A", {"m": 2}, **ov)
            raise AssertionError(f"override {ov} must refuse")
        except PowerHarnessError as e:
            assert "POWER_CERTIFICATION_CONFIG_UNBOUND" in str(e)

    # registered grammar: sequential master-stream property, verbatim
    ms = np.random.Generator(np.random.PCG64(
        _pb.derive_substream_seed("kat-seed-auth", "B2A", "full",
                                  "power")))
    seq = ms.integers(0, 2 ** 63, size=4, dtype=np.int64)
    assert rep_seed_registered("kat-seed-auth", "B2A", 3) == int(seq[3])
    assert rep_seed_registered("kat-seed-auth", "B2A", 0) != \
        rep_seed_registered("kat-seed-auth", "B3A", 0)

    # joint generation: fixed-grid registered_days, values only on
    # AVAILABLE days, typed absences never synthesized, shared-G
    # record, determinism, sorted carrier order
    panel3, views3, dbg3 = make_bound_panels(cap3, "B2A", {"m": 2}, 0)
    assert dbg3["carriers"] == ["c1", "c2", "c3"]
    assert len(dbg3["G"]) == 192
    for ck in ("c1", "c2", "c3"):
        assert panel3["carriers"][ck]["registered_days"] == ENG
        avs = set(cap3["carrier_masks"][ck]["available_days"])
        for e, ser in panel3["carriers"][ck]["r"].items():
            for d in ser:
                assert d in avs, (ck, d)
    for e, ser in panel3["carriers"]["c2"]["r"].items():
        assert not (set(ser) & holes)          # typed absences stay absent
    panel3b, _v, dbg3b = make_bound_panels(cap3, "B2A", {"m": 2}, 0)
    assert json.dumps(panel3, sort_keys=True) == \
        json.dumps(panel3b, sort_keys=True)
    assert np.array_equal(dbg3["G"], dbg3b["G"])

    # --- calendar v2 KAT 4: the _cal seams are DEAD in the bound
    # path (monkeypatched to fail; the run must still complete), and
    # doctored entrypoint/blob/frame digests refuse
    def _boom(*a, **k):
        raise AssertionError("_cal seam must never be invoked")
    _orig_cal = (_pb.b2a_family_cal, _pb.b3a_family_cal)
    _pb.b2a_family_cal = _boom
    _pb.b3a_family_cal = _boom
    try:
        pv3, frames3 = replicate_pvalues_bound(panel3, views3, 99,
                                               "ab" * 32, cap3)
    finally:
        _pb.b2a_family_cal, _pb.b3a_family_cal = _orig_cal
    assert set(pv3) == set(GRAPH)
    assert all(v is None or 0.0 < v <= 1.0 for v in pv3.values())
    _validate_frame_records(frames3, cap3, panel3)
    for fam, k, v in (
            ("B2A", "entrypoint", "d2_f2g_phase_b_stats"
                                  ".b2a_family_cal"),
            ("B3A", "engine_blob_sha256", "0" * 64),
            ("B2B", "frame_id", "calendar-v2"),
            ("B1B", "baseline_days_sha256", "0" * 64),
            ("B2A", "mask_sha256", "0" * 64),
            ("B3A", "input_panel_sha256", "0" * 64)):
        doct = {f: dict(r) for f, r in frames3.items()}
        doct[fam][k] = v
        try:
            _validate_frame_records(doct, cap3, panel3)
            raise AssertionError(f"doctored {fam}.{k} must refuse")
        except PowerHarnessError as e:
            assert "POWER_CALENDAR_FRAME_INVALID" in str(e)
    doct = {f: dict(r) for f, r in frames3.items() if f != "B3A"}
    try:
        _validate_frame_records(doct, cap3, panel3)
        raise AssertionError("absent frame record must refuse")
    except PowerHarnessError as e:
        assert "POWER_CALENDAR_FRAME_INVALID" in str(e)
    doct = {f: dict(r) for f, r in frames3.items()}
    doct["B1B"]["b1b_geometry"] = dict(FRAME["b1b"], n_blocks=11)
    try:
        _validate_frame_records(doct, cap3, panel3)
        raise AssertionError("doctored b1b geometry must refuse")
    except PowerHarnessError as e:
        assert "POWER_CALENDAR_FRAME_INVALID" in str(e)

    # --- calendar v2 KAT 5: cross-family alignment -- ONE replicate
    # produces ONE 192-position raw frame; B2B is the exact
    # 132-position evaluation projection; B2A/B3A/B1B share the full
    # fixed frame; mask holes are preserved in every view
    assert views3["b2b"]["calendar"] == FRAME["evaluation_days"]
    assert views3["b1b"]["calendar"] == ENG
    ev_set5 = set(FRAME["evaluation_days"])
    for ck in ("c1", "c2", "c3"):
        avs = set(cap3["carrier_masks"][ck]["available_days"])
        b2b_c = views3["b2b"]["carriers"][ck]
        b1b_c = views3["b1b"]["carriers"][ck]
        assert b2b_c["registered_days"] == [
            d for d in FRAME["evaluation_days"] if d in avs]
        assert set(b1b_c["registered_days"]) == avs
        for e, ser in b2b_c["r"].items():
            full = panel3["carriers"][ck]["r"][e]
            assert ser == {d: v for d, v in full.items()
                           if d in ev_set5}     # exact projection
            assert b1b_c["r"][e] == full        # same raw frame
        if ck == "c2":
            for e, ser in b2b_c["r"].items():
                assert not (set(ser) & holes)   # holes preserved

    # --- LOCO amendment v1 KATs (codex 1933Z, grassmann-ratified) ---
    # KAT 1: Holm SUBSTITUTION, not p <= .05 -- the exact hand fixture
    pv_full = {"B1B": 0.001, "B2A": 0.010, "B2B": 0.024, "B3A": 0.8}
    assert "B1B" in holm_rejects(pv_full)
    assert "B1B" not in holm_rejects(dict(pv_full, B1B=0.030))
    assert 0.030 <= 0.05    # the trap the substitution rule closes

    # projection: exactly the named station + incident edges leave;
    # every other raw value byte-identical
    b1b_view = {"calendar": ["D0", "D1"],
                "carriers": {"cx": {
                    "registry": ["A", "B", "C"],
                    "registered_days": ["D0", "D1"],
                    "r": {"A|B": {"D0": 1.0}, "A|C": {"D1": 2.0},
                          "B|C": {"D0": 3.0, "D1": 4.0}}}}}
    proj = b1b_loco_project(b1b_view, "A")
    assert proj["carriers"]["cx"]["registry"] == ["B", "C"]
    assert set(proj["carriers"]["cx"]["r"]) == {"B|C"}
    assert proj["carriers"]["cx"]["r"]["B|C"] == \
        b1b_view["carriers"]["cx"]["r"]["B|C"]   # byte-identical rest

    # fold-set audit: missing / extra / duplicate / wrong-station all
    # refuse the ARTIFACT (never counted as ordinary failure)
    regA = ["S1", "S2", "S3"]
    verify_fold_set(["S3", "S1", "S2"], regA)   # order-free exactness
    for bad, label in ((["S1", "S2"], "missing"),
                       (["S1", "S2", "S3", "S4"], "extra"),
                       (["S1", "S2", "S2"], "duplicate"),
                       (["S1", "S2", "S9"], "wrong-station")):
        try:
            verify_fold_set(bad, regA)
            raise AssertionError(f"{label} fold set must refuse")
        except PowerHarnessError as e:
            assert "POWER_LOCO_FOLD_SET_INVALID" in str(e)

    # early-exit: full-Holm non-rejection runs ZERO folds
    cap_l = dict(mk_capsule(), registries={"c1": ["S0", "S1"]},
                 carrier_masks={"c1": {"registered_days": list(ENG),
                                       "available_days": list(ENG)}},
                 segments={"c1": {"S0": "sA", "S1": "sB"}})
    cap_l["capsule_digest"] = _geometry_capsule_digest(
        {k: v for k, v in cap_l.items() if k != "capsule_digest"})
    counter = []
    r = _b1b_loco_recovery({"b1b": b1b_view},
                           {"B1B": 0.9, "B2A": 0.9, "B2B": 0.9,
                            "B3A": 0.9},
                           cap_l, 99, "ab" * 32, fold_counter=counter)
    assert r is False and counter == []      # licensed early-exit

    # typed-no-p folds: a 2-station loco registry degenerates every
    # projection (structural typed refusal) -> recovery False with the
    # FULL fold set still run (the audit stays exact)
    counter2 = []
    r = _b1b_loco_recovery(
        {"b1b": {"calendar": ["D0", "D1"],
                 "carriers": {"c1": {"registry": ["S0", "S1"],
                                     "registered_days": ["D0", "D1"],
                                     "r": {"S0|S1": {"D0": 1.0}}}}}},
        {"B1B": 0.001, "B2A": 0.9, "B2B": 0.9, "B3A": 0.9},
        cap_l, 99, "ab" * 32, fold_counter=counter2)
    assert r is False and sorted(counter2) == ["S0", "S1"]

    # anti-rescue is structural: run_artifact_class has no LOCO path
    import inspect
    assert "loco" not in inspect.getsource(run_artifact_class).lower()

    print("w2_power_harness selftest: ALL PASS "
          "(fixture-tier mechanism only)")


if __name__ == "__main__":
    _selftest()
