#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RED-first KATs -- fault2graph PHASE B STATISTICS ENGINE bar (grassmann).

Cross-authored per prereg section 6 (V-D governance): grassmann authors, cayley
implements the engine BAR-UNEDITED, codex verifies once. Disagreements with any
pin below go through inbox R1.2, never through editing this file.

Contract sources (the bar encodes the CORRECTED contracts, not prereg v1
verbatim): cayley prereg v1 draft (geospec 2dd451c, sha 16d9e0b7) as repaired
by codex R1.2 306d1a5-successor note f24eb2e (2026-08-20 0237Z):
  R1 B-1 null draws rotate the ENTIRE registered carrier-day vector BEFORE the
     60/remainder split -- ONE common circular offset per carrier per draw --
     then rerun the complete pipeline (split, floor, median/MAD, exclusions,
     z, family max, persistence). NEVER shift already-computed evaluation z
     (a circular shift is a permutation: max|shift(z)| == max|z| identically,
     p degenerates to 1 -- proven in-bar as G7a). 9999 draws, add-one p,
     < 9900 valid -> CANNOT_DETERMINE_NULL_SUPPORT.
  R2 B-2 identifiability gates: unique largest positive-weight component with
     >= 3 nodes (else LCC_TIE), exact station/index identity across compared
     days (else NODESET_MISMATCH), relative eigengap >= 1e-6 (else
     FIEDLER_DEGENERATE), |v2_i| > 1e-10 for every classified coordinate
     (else FIEDLER_ZERO_COORDINATE); orientation: lexicographically first
     component station's coordinate positive.
  R3 max-|z| is the SOLE verdict-bearing B-1 statistic; persistence is
     corrected secondary evidence and can never promote a verdict; a family
     without a sealed power contract types nonpositives
     CANNOT_DETERMINE_NO_POWER, never "no signal".
  R4 B-3 selection is deterministic: sort by (-|z|, station_a, station_b) over
     canonical unordered pairs, K = ceil(0.10*m), exactly first K; m == 0 ->
     INSUFFICIENT_DAILY_EDGES; space null permutes station->segment labels
     within carrier preserving exact segment sizes; NEVER conditions on B-1
     verdict/BY significance.
  R5 substream seeds per prereg rev-2 section (a44819d): seed_material =
     UTF-8("<frozen_doc_sha256_hex_lowercase>||<family>||<fold>||<purpose>")
     with DOUBLE-pipe separators, family tokens {B1, B2, B3}, fold in
     {full} + {loco:<STATION_ID>}, purpose in {null, power}; seed = first 8
     bytes of SHA256(seed_material) as big-endian uint64 -> PCG64; LOCO is
     a conjunctive gate only (every scorable fold must independently pass; a
     missing fold withholds -> LOCO_FOLD_UNSCORABLE; never promotes).

PINNED SEAMS (module monitoring/src/d2_f2g_phase_b_stats.py):
  walk_forward_split(registered_days) -> (baseline_days, eval_days)
  b1_family(panel, *, doc_sha256, n_draws=N_DRAWS, power_contract=...,
            return_null=False) -> result dict
  b2_family(panel, *, doc_sha256, n_draws=N_DRAWS, return_null=False)
  b3_family(panel, *, doc_sha256, n_draws=N_DRAWS, return_null=False)
  loco_gate(full_result, fold_results, alpha) -> dict
  derive_substream_seed(doc_sha256_hex, family, fold, purpose) -> int
  Constants: N_DRAWS=9999, ALPHA_FAMILY=0.05/3, BASELINE_DAYS=60,
             TESTABLE_MIN_BASELINE=45, PERSISTENCE_K=3, Z_PERSIST=3.0,
             TOP_DECILE=0.10, BY_Q=0.05
Family results MUST carry at least: family, p_value (float or None), T_obs,
n_valid_draws, verdict, excluded (dict of typed-exclusion counts), and for
fixtures: return_null=True adds null_T (list); b2 adds day_refusals
[{day, code}]; b3 adds selection {day: [edge, ...]} with edges as "A|B"
canonical sorted pairs. Nonpositive typing rides verdict (a verdict string
containing CANNOT_DETERMINE_NO_POWER / CANNOT_DETERMINE_NULL_SUPPORT / a
typed no-testable-edges refusal is typed, "NEGATIVE" alone is not).

FIXTURE PANEL SCHEMA (fixture-panel-v1, this bar's authority):
  {"carriers": {carrier_key: {
      "registered_days": [ISO ascending],
      "stations": [ids], "segments": {station: segment},
      "r": {"A|B": {day: float}}}}}   # absent (edge, day) = not measured

FIXTURE-ONLY (prereg governing rule): nothing in this bar touches the Phase-A
artifact, any real graph, or any waveform. No claims; Lambda_geo INCONCLUSIVE.
EXPECTED RED until cayley's engine lands. grassmann 2026-08-20.

AMENDMENT 1 (append-only lane of freeze F2G-PB-R2-FREEZE-CODEX-20260820T0301Z;
G5 REV explicitly adjudicated by codex 633be61 item 3; annex authorities
admitted by codex 0400Z pass): G5 is now ANNEX-CONDITIONED and G16a/b/c bind
the rev-1.1 power-annex authorities:
  common  docs/f2g_phase_b_power_annex_common.md @ d3aa25f  sha256 baddf2aa...
  B-1     docs/f2g_phase_b_power_annex_b1.md     @ b816f88  sha256 fb2883f5...
  B-2     docs/f2g_phase_b_power_annex_b2.md     @ b816f88  sha256 dca9dede...
  B-3     docs/f2g_phase_b_power_annex_b3.md     @ b816f88  sha256 9414bd59...
G5 REV 2 semantics (codex 633be61 item 3 verbatim): with the B-1 results
artifact (docs/f2g_phase_b_power_annex_b1_results.json, evidence lane) ABSENT,
G5 reds typed POWER_RESULTS_ABSENT (expected until Tier-C completes). Present
+ zero certified points (terminal MDE_NOT_CERTIFIED_BY_REGISTERED_SEARCH):
G5 passes ONLY when the annex-bound planted run is nonpositive and typed
exactly CANNOT_DETERMINE_NO_POWER -- never NEGATIVE, never a fabricated
positive requirement. Present + >=1 certified point: the representative is the
lexicographically FIRST Pareto-minimal certified point and G5 recomputes the
registered replicate-level post-LOCO recovery bound in-bar (one-sided exact
binomial 95% lower bound >= 0.80 via Clopper-Pearson) from the recorded
replicate outcomes -- never a single cherry-picked fixture.
Results-artifact minimal schema pinned here (R1.2 lane open): keys
certified_points (list), pareto_minimal_certified (list, lex-first =
representative; each point carries grid_point + post_loco {successes,
replicates}), terminal_type (when none certified), equivalence_receipt,
digests (must include the four annex sha256s above). This amendment becomes
the admitted authority only after its commit/digest is routed and checked.

AMENDMENT 2 (append-only; prereg Amendment 1 rev-3 FROZEN
F2G-PB-A1-R3-FREEZE-CODEX-20260820T1325Z, doc 7c3ca7b blob f3d0830b...; the
mandatory sec-A5 red-KAT classes). Amended families B1A/B2A/B3A supersede
B1/B2/B3 (which retain evidence status, no verdict weight). PINNED AMENDED
SEAMS (cayley's engine amendment implements BAR-UNEDITED):
  b1a_family(panel, *, doc_sha256, n_draws=N_DRAWS, power_contract=...,
             return_null=False, exhaustive=False,
             n_blocks=B1A_BLOCKS, block_len=B1A_BLOCK_LEN,
             baseline_len=BASELINE_DAYS, testable_min=TESTABLE_MIN_BASELINE,
             window=B1A_WINDOW, window_min=B1A_WINDOW_MIN) -> result
      T = sum over BOUND carriers of max (testable edge, 7d-window) mean|z|
      (window scored iff >= 4 finite cells); null = ONE common S_n_blocks
      equal-block day permutation per draw applied to EVERY carrier's full
      vector pre-split, complete group INCLUDING identity; exhaustive=True
      enumerates the full group -> result["p_exact"] = #{T_perm >= T_obs}/|G|;
      a bound carrier without a scorable window -> family verdict
      CANNOT_DETERMINE_FAMILY_SCORABILITY (never silent exclusion); fixture
      parameterization is bar-pinned, PRODUCTION values locked as constants
      B1A_BLOCKS=11, B1A_BLOCK_LEN=10, B1A_WINDOW=7, B1A_WINDOW_MIN=4.
  b2a_family(panel, *, doc_sha256, n_draws=N_DRAWS, return_null=False)
      R_total = sum over bound carriers of maximal identical-partition run
      count over CALENDAR-ADJACENT eligible days; gap/mismatch TERMINATES a
      run (never bridged); null = ONE common permutation of the joint
      evaluation-day capsule per draw, days moved ATOMICALLY (partition/
      refusal/frame/gap metadata together), eligibility+runs recomputed AFTER;
      one-sided LOW: p = (1 + #{R_null <= R_obs})/(N_valid + 1); result
      carries runs_by_carrier, day_refusals; return_null adds null_R plus the
      AUDIT VECTOR (codex 1427Z repair 2): null_orders = the ONE common
      capsule position order per draw, null_runs_by_carrier = per-draw
      {carrier: runs} -- the bar recomputes both carriers' runs from the
      returned common order and refuses on any draw-level mismatch.
  b3a_family(panel, *, doc_sha256, n_draws=N_DRAWS, return_null=False,
             exhaustive_space=False)
      selection/typing as admitted B-3 plus K>=2 floor (K==1 day ->
      DAY_K_UNSCORABLE, counts disclosed); C = count of scorable (carrier,
      day)s with f >= (K-1)/K, one-sided HIGH; space null = ONE shared-
      registry station->segment relabeling per draw preserving exact
      per-carrier segment sizes, selections FIXED; exhaustive_space=True ->
      result["exact_space_p"] = {day: exact balanced-label enumeration p}.
  Substream tokens B1A/B2A/B3A, seed root = the FROZEN AMENDMENT sha
  f3d0830b... (NOT the base doc sha), same double-pipe grammar.
"""
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))

# AMENDMENT 1: admitted rev-1.1 power-annex authorities (codex 0400Z pass)
ANNEX_AUTHORITIES = (
    ("docs/f2g_phase_b_power_annex_common.md",
     "d3aa25f4f174d5864f25e036bde07e91417328ca",
     "baddf2aa259689356d4d942b6282824ad6ad6d7f50075c54c992a997f216f20d"),
    ("docs/f2g_phase_b_power_annex_b1.md",
     "b816f88fdce4f2328334e928500b559900b9cee5",
     "fb2883f5a9be84f2197b79291b116a799a3a2c5cc86afc69cc34fea064a4e14a"),
    ("docs/f2g_phase_b_power_annex_b2.md",
     "b816f88fdce4f2328334e928500b559900b9cee5",
     "dca9dede6b91e35e6dd55ed6104dd3b4c29d2f8dfc8b6ef615187df91b5cb05c"),
    ("docs/f2g_phase_b_power_annex_b3.md",
     "b816f88fdce4f2328334e928500b559900b9cee5",
     "9414bd597134ebbcac8dc2d199a111521a23b0d59af6bc40ab0e803cd3495dc2"),
)
ANNEX_SHAS = {sha for _p, _c, sha in ANNEX_AUTHORITIES}
B1_RESULTS = os.path.join(_REPO, "docs",
                          "f2g_phase_b_power_annex_b1_results.json")

# AMENDMENT 2: frozen prereg-amendment authority + amended annex authorities
AMENDMENT1_SHA = ("f3d0830b38869d8b6f0b03d113d45ae0f111e8645bd4a2934582b21e"
                  "48e909e8")
# re-pinned to the ADMITTED rev-1.3/rev-1.1 authorities (codex 1501Z PASS)
AMENDED_ANNEX_AUTHORITIES = (
    ("docs/f2g_phase_b_power_annex_common.md", "60ea20e",
     "b6352e914bf24b3c54663388daa9e71d15a637491c5fec4462abcd6785bc2e8d"),
    ("docs/f2g_phase_b_power_annex_b1a.md", "a0cc87c",
     "e7e08454ec5f4ba45dd3092797afe26cbb12727b02a6027a962cfb99fb946a9b"),
    ("docs/f2g_phase_b_power_annex_b2a.md", "a0cc87c",
     "1df422a477bd418bd26067b5afdd4211bbb1882f6eff5de5f640fd2d891ceaf6"),
    ("docs/f2g_phase_b_power_annex_b3a.md", "a0cc87c",
     "0c5fc14f6ecc4606b4a2548e788f962fd844ef0997ffb271ecf2e456d497f8d0"),
)


def _b1a_small_T(series_by_edge, order, base_len, w, wmin):
    """In-bar independent reimplementation of the B1A statistic for the
    exhaustive-orbit KAT (single carrier): day order applied to the FULL
    vector, positional split, median/1.4826*MAD from baseline, max over
    (edge, w-window) of mean|z| over finite cells (scored iff >= wmin)."""
    best = None
    for vals in series_by_edge.values():
        v = np.asarray(vals, dtype=float)[list(order)]
        base = v[:base_len]
        fb = base[np.isfinite(base)]
        if fb.size == 0:
            continue
        med = float(np.median(fb))
        mad = float(np.median(np.abs(fb - med)))
        if mad == 0:
            continue
        z = (v[base_len:] - med) / (1.4826 * mad)
        for s in range(0, z.size - w + 1):
            win = z[s:s + w]
            fin = win[np.isfinite(win)]
            if fin.size >= wmin:
                m = float(np.mean(np.abs(fin)))
                if best is None or m > best:
                    best = m
    return best


def _balanced_label_exact_p(n_stations, sizes, edges, thresh):
    """In-bar exact enumeration over all balanced station->segment labelings
    (multinomial C(12;4,4,4) = 34650 for the registered frame): probability
    that the FIXED selected edge set has cross-segment count >= thresh."""
    import itertools
    stations = list(range(n_stations))
    hits = total = 0
    for seg1 in itertools.combinations(stations, sizes[0]):
        rest1 = [s for s in stations if s not in seg1]
        for seg2 in itertools.combinations(rest1, sizes[1]):
            lab = {}
            for s in seg1:
                lab[s] = 0
            for s in seg2:
                lab[s] = 1
            for s in rest1:
                if s not in lab:
                    lab[s] = 2
            cross = sum(1 for a, b in edges if lab[a] != lab[b])
            total += 1
            if cross >= thresh:
                hits += 1
    return hits / total, total


def _results(path):
    try:
        return json.loads(open(path, "rb").read().decode("utf-8"))
    except Exception:
        return None


def _digest_leaves(obj):
    """Recursively collect string leaves (codex 0431Z repair 1: check the four
    exact hashes without prescribing irrelevant JSON nesting)."""
    out = set()
    if isinstance(obj, dict):
        for v in obj.values():
            out |= _digest_leaves(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out |= _digest_leaves(v)
    elif isinstance(obj, str):
        out.add(obj)
    return out


def _results_bound(res):
    """codex 0431Z repairs 1+2 + 0517Z closure: the four annex digests must
    appear among the artifact's digest leaves; the equivalence receipt must
    carry Boolean full_equal AND fold_equal both exactly True (a nonempty
    receipt is not proof); and the digests block must bind the EXACT
    registered driver/engine authorities -- key-name substrings are not
    authorities (fake {"claimed_driver": null} bypass closed)."""
    d = res.get("digests")
    eq = res.get("equivalence_receipt")
    if not isinstance(d, dict) or not isinstance(eq, dict):
        return False
    has_auth = (
        d.get("estimation_run_driver_commit") == "d4edfb2"
        and isinstance(d.get("engine"), str)
        and "6034419" in d["engine"]
    )
    return ANNEX_SHAS.issubset(_digest_leaves(d)) \
        and eq.get("full_equal") is True and eq.get("fold_equal") is True \
        and has_auth


def _lex_first_pareto(cert_pts):
    """codex 0431Z repair 3: recompute the lexicographically first
    Pareto-minimal certified point IN-BAR from grid_point coordinates --
    never trust JSON list order. Returns the point dict (membership in
    certified_points holds by construction)."""
    gps = [tuple(p["grid_point"]) for p in cert_pts]
    minimal = [g for g in gps
               if not any(q != g and len(q) == len(g)
                          and all(qi <= gi for qi, gi in zip(q, g))
                          for q in gps)]
    rep_gp = sorted(minimal)[0]
    return next(p for p in cert_pts if tuple(p["grid_point"]) == rep_gp)

HERE_FAILS = []
_N = [0]


def check(name, ok, detail=""):
    _N[0] += 1
    tag = "PASS" if ok else "FAIL"
    print(f"    [{tag}] PB-{name}" + (f" - {detail}" if (detail and not ok) else ""))
    if not ok:
        HERE_FAILS.append(name)


def day(i):
    # deterministic ascending ISO dates
    return f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}"


def mk_panel(n_days, edges, seed, sigma=1.0, base=0.0):
    """fixture-panel-v1: one carrier, gaussian r series per edge."""
    rng = np.random.default_rng(seed)
    days = [day(i) for i in range(n_days)]
    stations = sorted({s for e in edges for s in e.split("|")})
    r = {}
    for e in edges:
        vals = base + sigma * rng.standard_normal(n_days)
        r[e] = {d: float(v) for d, v in zip(days, vals)}
    return {"carriers": {"c_fix": {
        "registered_days": days, "stations": stations,
        "segments": {s: ("seg_a" if i % 2 == 0 else "seg_b")
                     for i, s in enumerate(stations)},
        "r": r}}}


def plant(panel, edges, day_idx, k, delta):
    """add delta to `edges` on k consecutive registered days from day_idx."""
    c = panel["carriers"]["c_fix"]
    for e in edges:
        for i in range(day_idx, day_idx + k):
            d = c["registered_days"][i]
            c["r"][e][d] = c["r"][e][d] + delta
    return panel


DOC = hashlib.sha256(b"pb-fixture-doc-v1").hexdigest()

try:
    import d2_f2g_phase_b_stats as E
    HAVE = True
except ImportError as exc:
    HAVE = False
    print(f"    [RED ] engine module absent ({exc}) -- EXPECTED RED until "
          f"cayley's d2_f2g_phase_b_stats lands; every check below reds")


def g7a():
    """codex CRITICAL finding, proven in-bar and ENGINE-INDEPENDENT: shifting
    already-computed evaluation z can never calibrate max-|z| -- a circular
    shift is a permutation, so every draw's family max equals the observed max
    exactly and the p-value degenerates to 1."""
    rng7 = np.random.default_rng(7)
    z = rng7.standard_normal((10, 40))            # 10 edges x 40 eval days of z
    t_obs = float(np.max(np.abs(z)))
    degenerate = all(
        float(np.max(np.abs(np.stack([np.roll(z[i], int(off[i]))
                                      for i in range(z.shape[0])])))) == t_obs
        for off in rng7.integers(0, 40, size=(200, z.shape[0])))
    check("G7a wrong-null degeneracy (in-bar proof)", degenerate,
          "a circular shift of computed z changed max|z| -- impossible")


def g16a():
    """AMENDMENT 1: the four admitted rev-1.1 annex authorities, recomputed
    from the pinned commits' committed bytes -- engine-independent."""
    ok, det = True, []
    for path, commit, sha in ANNEX_AUTHORITIES:
        try:
            raw = subprocess.run(
                ["git", "-C", _REPO, "cat-file", "blob", f"{commit}:{path}"],
                capture_output=True).stdout
            got = hashlib.sha256(raw).hexdigest()
        except Exception as exc:
            got = f"ERROR:{exc}"
        if got != sha:
            ok = False
            det.append(f"{path}@{commit[:7]}: got={got[:16]} expect={sha[:16]}")
    check("G16a annex authority binding (rev-1.1 bytes)", ok, "; ".join(det))


def g16c():
    """codex 0517Z negative fixture: four exact annex hashes + fake/empty
    authorities + true booleans must be REFUSED -- engine-independent."""
    fake = {
        "digests": {
            "annex_common_rev11_sha256": sorted(ANNEX_SHAS)[0],
            "annex_b1_rev11_sha256": sorted(ANNEX_SHAS)[1],
            "annex_b2_rev11_sha256": sorted(ANNEX_SHAS)[2],
            "annex_b3_rev11_sha256": sorted(ANNEX_SHAS)[3],
            "claimed_driver": None, "claimed_engine": False},
        "equivalence_receipt": {"full_equal": True, "fold_equal": True},
    }
    ok = _results_bound(fake) is False
    real = _results(B1_RESULTS)
    ok_real = real is None or _results_bound(real) is True
    check("G16c fake-empty-authorities REFUSED (exact registered values only)",
          ok and ok_real,
          f"fake_accepted={not ok} real_bound_ok={ok_real}")


def a5_annex_binding():
    """AMENDMENT 2: the amended-family annex authorities (common rev-1.2 +
    B1A/B2A/B3A v1), recomputed from pinned commits -- engine-independent."""
    ok, det = True, []
    for path, commit, sha in AMENDED_ANNEX_AUTHORITIES:
        try:
            raw = subprocess.run(
                ["git", "-C", _REPO, "cat-file", "blob", f"{commit}:{path}"],
                capture_output=True).stdout
            got = hashlib.sha256(raw).hexdigest()
        except Exception as exc:
            got = f"ERROR:{exc}"
        if got != sha:
            ok = False
            det.append(f"{path}@{commit}: got={got[:16]}")
    check("A5j amended-annex authority binding (ADMITTED rev-1.3 + "
          "B1A/B2A/B3A rev-1.1)", ok, "; ".join(det))


_A5_NAMES = (
    "A5a exhaustive-orbit calibration (engine == exact enumeration)",
    "A5b paired-carrier synchrony (2 identical carriers == 2x one, per draw)",
    "A5c bound-carrier unscorable -> family withholding",
    "A5d gap terminates runs (A,A,[gap],A,A == 2) + atomic null recompute",
    "A5e mixed nodeset never bridged",
    "A5f planted two-regime -> B2A p at the add-one floor",
    "A5g B3A exact star/path/mixed balanced-label enumeration",
    "A5h K=1 day -> DAY_K_UNSCORABLE; m=11 -> K=2 scorable",
    "A5i amended substream vectors (B1A/B2A/B3A @ frozen amendment root)",
    "A5k joint-capsule atomicity across carriers (per-draw audit vector)",
)


def _b2_panel(day_specs, stations_all):
    """B2A fixture: 60 trivial baseline days + one eval day per spec.
    spec = list of 'A|B' edges (eligible day), None (gap: nothing measured),
    or a different-frame edge list (nodeset mismatch)."""
    n = 60 + len(day_specs)
    days_ = [day(i) for i in range(n)]
    r = {}
    for i, d in enumerate(days_):
        if i < 60:
            for a in range(len(stations_all) - 1):
                e = "|".join(sorted([stations_all[a], stations_all[a + 1]]))
                r.setdefault(e, {})[d] = 0.5
        else:
            spec = day_specs[i - 60]
            if spec is None:
                continue
            for e in spec:
                r.setdefault(e, {})[d] = 0.8
    return {"carriers": {"c_fix": {
        "registered_days": days_, "stations": sorted(stations_all),
        "segments": {s: "seg_a" for s in stations_all}, "r": r}}}


def amendment2(have_a2):
    if not have_a2:
        for nm in _A5_NAMES:
            check(nm, False, "ENGINE_AMENDMENT_ABSENT (expected red until "
                             "cayley's b1a/b2a/b3a land)")
        return

    # A5a exhaustive-orbit calibration: 4 equal blocks of 10, exact S_4 ------
    import itertools
    e5a = ["AA.S1|AA.S2", "AA.S1|AA.S3"]
    p5a = mk_panel(40, e5a, seed=51)
    plant(p5a, e5a[:1], 30, 5, 6.0)                  # eval-region plant
    series = {e: [p5a["carriers"]["c_fix"]["r"][e][d]
                  for d in p5a["carriers"]["c_fix"]["registered_days"]]
              for e in e5a}
    exceed = 0
    orders = []
    for perm in itertools.permutations(range(4)):
        order = [b * 10 + i for b in perm for i in range(10)]
        orders.append(order)
    t_all = [_b1a_small_T(series, o, 20, 7, 4) for o in orders]
    t_obs = t_all[0]                                  # identity ordering
    exact_p = sum(1 for t in t_all if t is not None and t >= t_obs) / 24.0
    try:
        # codex 1427Z repair 1: the PRODUCTION equal-block constants are
        # locked here -- an override fixture alone cannot prove them
        const_ok = (E.B1A_BLOCKS == 11 and E.B1A_BLOCK_LEN == 10
                    and E.B1A_WINDOW == 7 and E.B1A_WINDOW_MIN == 4)
        r5a = E.b1a_family(p5a, doc_sha256=AMENDMENT1_SHA, exhaustive=True,
                           power_contract={"certified": True}, n_blocks=4,
                           block_len=10, baseline_len=20, testable_min=10,
                           window=7, window_min=4)
        ok5a = const_ok and abs(r5a["p_exact"] - exact_p) < 1e-12 \
            and exact_p >= 1 / 24.0
        check(_A5_NAMES[0], ok5a,
              f"constants_11_10_7_4={const_ok} "
              f"engine={r5a.get('p_exact')} exact={exact_p}")
    except Exception as exc:
        check(_A5_NAMES[0], False, f"{type(exc).__name__}: {exc}")

    # A5b paired-carrier synchrony: joint law, draw-for-draw ------------------
    try:
        p1 = mk_panel(40, e5a, seed=52)
        p2 = {"carriers": {
            "c_a": json.loads(json.dumps(p1["carriers"]["c_fix"])),
            "c_b": json.loads(json.dumps(p1["carriers"]["c_fix"]))}}
        ra = E.b1a_family(p1, doc_sha256=AMENDMENT1_SHA, n_draws=199,
                         power_contract={"certified": True}, return_null=True,
                         n_blocks=4, block_len=10, baseline_len=20,
                         testable_min=10, window=7, window_min=4)
        rb = E.b1a_family(p2, doc_sha256=AMENDMENT1_SHA, n_draws=199,
                         power_contract={"certified": True}, return_null=True,
                         n_blocks=4, block_len=10, baseline_len=20,
                         testable_min=10, window=7, window_min=4)
        na, nb = np.asarray(ra["null_T"]), np.asarray(rb["null_T"])
        ok5b = ra["T_obs"] * 2 == rb["T_obs"] and np.allclose(na * 2, nb) \
            and na.size == nb.size
        check(_A5_NAMES[1], ok5b,
              f"T1={ra.get('T_obs')} T2={rb.get('T_obs')} "
              f"null2x={bool(np.allclose(na * 2, nb))}")
    except Exception as exc:
        check(_A5_NAMES[1], False, f"{type(exc).__name__}: {exc}")

    # A5c bound-carrier unscorable -> family withholding ----------------------
    try:
        p3 = {"carriers": {
            "c_a": json.loads(json.dumps(
                mk_panel(40, e5a, seed=53)["carriers"]["c_fix"])),
            "c_dead": json.loads(json.dumps(
                mk_panel(40, e5a, seed=54)["carriers"]["c_fix"]))}}
        cd = p3["carriers"]["c_dead"]
        for d in cd["registered_days"][20:]:
            for e in list(cd["r"]):
                cd["r"][e].pop(d, None)               # no scorable window
        r5c = E.b1a_family(p3, doc_sha256=AMENDMENT1_SHA, n_draws=99,
                           power_contract={"certified": True}, n_blocks=4,
                           block_len=10, baseline_len=20, testable_min=10,
                           window=7, window_min=4)
        ok5c = "CANNOT_DETERMINE_FAMILY_SCORABILITY" in str(r5c["verdict"])
        check(_A5_NAMES[2], ok5c, f"verdict={r5c.get('verdict')}")
    except Exception as exc:
        check(_A5_NAMES[2], False, f"{type(exc).__name__}: {exc}")

    # A5d gap terminates runs + atomic null recompute -------------------------
    sts = ["DD.S1", "DD.S2", "DD.S3", "DD.S4"]
    pathA = ["DD.S1|DD.S2", "DD.S2|DD.S3", "DD.S3|DD.S4"]
    try:
        pan5d = _b2_panel([pathA, pathA, None, pathA, pathA], sts)
        r5d = E.b2a_family(pan5d, doc_sha256=AMENDMENT1_SHA, n_draws=199,
                           return_null=True)
        nr = set(int(x) for x in r5d["null_R"])
        ok5d = r5d["runs_total"] == 2 and nr <= {1, 2} and len(nr) == 2
        check(_A5_NAMES[3], ok5d,
              f"R_obs={r5d.get('runs_total')} null_R_values={sorted(nr)}")
    except Exception as exc:
        check(_A5_NAMES[3], False, f"{type(exc).__name__}: {exc}")

    # A5e mixed nodeset never bridged -----------------------------------------
    try:
        alt = ["DD.S2|DD.S3", "DD.S3|DD.S4", "DD.S4|DD.S5"]
        pan5e = _b2_panel([pathA, alt, pathA], sts + ["DD.S5"])
        r5e = E.b2a_family(pan5e, doc_sha256=AMENDMENT1_SHA, n_draws=99)
        ok5e = r5e["runs_total"] == 2 and any(
            x.get("code") == "NODESET_MISMATCH"
            for x in r5e.get("day_refusals", []))
        check(_A5_NAMES[4], ok5e,
              f"R={r5e.get('runs_total')} refusals={r5e.get('day_refusals')}")
    except Exception as exc:
        check(_A5_NAMES[4], False, f"{type(exc).__name__}: {exc}")

    # A5f planted two-regime -> p at the add-one floor ------------------------
    try:
        pathB = ["DD.S1|DD.S3", "DD.S3|DD.S2", "DD.S2|DD.S4"]
        pan5f = _b2_panel([pathA] * 12 + [pathB] * 12, sts)
        r5f = E.b2a_family(pan5f, doc_sha256=AMENDMENT1_SHA, n_draws=999)
        floor = 1.0 / (r5f["n_valid_draws"] + 1)
        ok5f = r5f["runs_total"] == 2 and abs(r5f["p_value"] - floor) < 1e-12
        check(_A5_NAMES[5], ok5f,
              f"R={r5f.get('runs_total')} p={r5f.get('p_value')} "
              f"floor={floor}")
    except Exception as exc:
        check(_A5_NAMES[5], False, f"{type(exc).__name__}: {exc}")

    # A5g B3A exact enumeration: star / path / mixed on 12 stations 4/4/4 -----
    try:
        st12 = [f"EE.S{i:02d}" for i in range(12)]
        idx = {s: i for i, s in enumerate(st12)}
        # codex 1427Z repair 3: fixtures + golden rationals are the ON-FILE
        # codex enumerations -- all three locked, no fixture drift possible
        star = [tuple(sorted((0, j))) for j in range(1, 8)]
        pth = [(j, j + 1) for j in range(7)]
        mixed = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (0, 2)]
        golden = {"star": (9660, 34650), "path": (13176, 34650),
                  "mixed": (14016, 34650)}
        ok5g, det5g = True, []
        for name, topo in (("star", star), ("path", pth), ("mixed", mixed)):
            exact, total = _balanced_label_exact_p(12, (4, 4, 4), topo, 6)
            gn, gd = golden[name]
            if total != gd or abs(exact - gn / gd) > 1e-15:
                ok5g = False
                det5g.append(f"{name}: in-bar {exact}*{total} != golden "
                             f"{gn}/{gd}")
            pan = mk_panel(62, [], seed=57)
            c = pan["carriers"]["c_fix"]
            c["stations"] = st12
            c["segments"] = {s: f"seg_{i // 4 + 1}" for i, s in
                             enumerate(st12)}
            c["r"] = {}
            topo_edges = {"|".join(sorted((st12[a], st12[b])))
                          for a, b in topo}
            all_edges = ["|".join(sorted((st12[a], st12[b])))
                         for a in range(12) for b in range(a + 1, 12)]
            for i, d in enumerate(c["registered_days"]):
                for e in all_edges:
                    if i < 60:
                        c["r"].setdefault(e, {})[d] = \
                            0.001 if i % 2 == 0 else -0.001
                    else:
                        c["r"].setdefault(e, {})[d] = \
                            0.5 if e in topo_edges else 0.01
            r5g = E.b3a_family(pan, doc_sha256=AMENDMENT1_SHA,
                               exhaustive_space=True)
            d_ev = c["registered_days"][60]
            got = r5g["exact_space_p"].get(d_ev)
            if got is None or abs(got - exact) > 1e-12:
                ok5g = False
                det5g.append(f"{name}: engine={got} exact={exact}")
        check(_A5_NAMES[6], ok5g, "; ".join(det5g))
    except Exception as exc:
        check(_A5_NAMES[6], False, f"{type(exc).__name__}: {exc}")

    # A5h K floor: m=10 -> K=1 unscorable; m=11 -> K=2 ------------------------
    try:
        e10 = [f"FF.S1|FF.S{j:02d}" for j in range(2, 12)]     # m = 10
        pan10 = mk_panel(62, e10, seed=58, sigma=0.0)
        c10 = pan10["carriers"]["c_fix"]
        for i, d in enumerate(c10["registered_days"]):
            for e in e10:
                c10["r"][e][d] = (0.001 if i % 2 == 0 else -0.001) \
                    if i < 60 else 0.01
        r5h = E.b3a_family(pan10, doc_sha256=AMENDMENT1_SHA, n_draws=99)
        ok5h1 = any("DAY_K_UNSCORABLE" in str(x)
                    for x in r5h.get("day_refusals", []))
        e11 = e10 + ["FF.S1|FF.S12"]                            # m = 11
        pan11 = mk_panel(62, e11, seed=59, sigma=0.0)
        c11 = pan11["carriers"]["c_fix"]
        for i, d in enumerate(c11["registered_days"]):
            for e in e11:
                c11["r"][e][d] = (0.001 if i % 2 == 0 else -0.001) \
                    if i < 60 else 0.01
        r5h2 = E.b3a_family(pan11, doc_sha256=AMENDMENT1_SHA, n_draws=99)
        ok5h2 = not any("DAY_K_UNSCORABLE" in str(x)
                        for x in r5h2.get("day_refusals", []))
        check(_A5_NAMES[7], ok5h1 and ok5h2,
              f"m10_unscorable={ok5h1} m11_scorable={ok5h2}")
    except Exception as exc:
        check(_A5_NAMES[7], False, f"{type(exc).__name__}: {exc}")

    # A5k joint-capsule atomicity across carriers (codex 1427Z repair 2):
    # two BOUND carriers with DIFFERENT gap/refusal masks; the engine returns
    # a per-draw audit vector (null_orders = the ONE common capsule order per
    # draw + null_runs_by_carrier); the bar independently recomputes each
    # carrier's runs from the returned order and the known day payloads -- both
    # carriers must match DRAW-FOR-DRAW (a factorized per-carrier permutation
    # cannot survive this with differing masks) -------------------------------
    try:
        sA, sB = ["GG.S1", "GG.S2", "GG.S3", "GG.S4"], "GG.S5"
        pA = ["GG.S1|GG.S2", "GG.S2|GG.S3", "GG.S3|GG.S4"]
        pB = ["GG.S1|GG.S3", "GG.S3|GG.S2", "GG.S2|GG.S4"]
        mm = ["GG.S2|GG.S3", "GG.S3|GG.S4", "GG.S4|GG.S5"]
        spec_x = [pA, pA, None, pA, pA, pB]           # A,A,gap,A,A,B
        spec_y = [pA, None, pA, pA, mm, pA]           # A,gap,A,A,MM,A
        pay = {"c_x": ["A", "A", "GAP", "A", "A", "B"],
               "c_y": ["A", "GAP", "A", "A", "MM", "A"]}

        def _carrier(specs, stations):
            n = 60 + len(specs)
            days_ = [day(i) for i in range(n)]
            r = {}
            for i, d in enumerate(days_):
                if i < 60:
                    for a in range(len(stations) - 1):
                        e = "|".join(sorted([stations[a], stations[a + 1]]))
                        r.setdefault(e, {})[d] = 0.5
                else:
                    spec = specs[i - 60]
                    if spec is None:
                        continue
                    for e in spec:
                        r.setdefault(e, {})[d] = 0.8
            return {"registered_days": days_,
                    "stations": sorted(set(stations) | {sB}),
                    "segments": {s: "seg_a" for s in stations + [sB]},
                    "r": r}

        pan5k = {"carriers": {"c_x": _carrier(spec_x, sA),
                              "c_y": _carrier(spec_y, sA)}}

        def _runs(order, payload):
            runs, prev = 0, None
            for j in order:
                p = payload[j]
                if p in ("GAP", "MM"):
                    prev = None
                    continue
                if p != prev:
                    runs += 1
                prev = p
            return runs

        r5k = E.b2a_family(pan5k, doc_sha256=AMENDMENT1_SHA, n_draws=199,
                           return_null=True)
        orders = r5k.get("null_orders") or []
        by_car = r5k.get("null_runs_by_carrier") or []
        ok5k = len(orders) == len(by_car) and len(orders) > 0
        for o, rb in zip(orders, by_car):
            for c in ("c_x", "c_y"):
                if int(rb[c]) != _runs(list(o), pay[c]):
                    ok5k = False
                    break
            if not ok5k:
                break
        obs_ok = r5k["runs_by_carrier"]["c_x"] == _runs(range(6), pay["c_x"]) \
            and r5k["runs_by_carrier"]["c_y"] == _runs(range(6), pay["c_y"])
        check(_A5_NAMES[9], ok5k and obs_ok,
              f"draws_audited={len(orders)} audit_ok={ok5k} obs_ok={obs_ok}")
    except Exception as exc:
        check(_A5_NAMES[9], False, f"{type(exc).__name__}: {exc}")

    # A5i amended substream vectors -------------------------------------------
    try:
        ok5i, det5i = True, []
        for fam, fold, purpose in (("B1A", "full", "null"),
                                   ("B2A", "loco:KO.GEML", "null"),
                                   ("B3A", "full", "power")):
            exp = int.from_bytes(hashlib.sha256(
                f"{AMENDMENT1_SHA}||{fam}||{fold}||{purpose}"
                .encode("utf-8")).digest()[:8], "big")
            got = E.derive_substream_seed(AMENDMENT1_SHA, fam, fold, purpose)
            if got != exp:
                ok5i = False
                det5i.append(f"{fam}/{fold}/{purpose}")
        check(_A5_NAMES[8], ok5i, "; ".join(det5i))
    except Exception as exc:
        check(_A5_NAMES[8], False, f"{type(exc).__name__}: {exc}")


def main():
    g7a()                       # engine-independent: runs (and must pass) NOW
    g16a()                      # engine-independent: annex bytes bind NOW
    g16c()                      # engine-independent: authority-bypass lock
    a5_annex_binding()          # engine-independent: amended annexes bind NOW
    if not HAVE:
        for nm in ("G1 seams+constants", "G2 walk-forward split",
                   "G3 testable floor 45", "G4 degenerate baseline",
                   "G5 annex-conditioned planted recovery (REV 2)",
                   "G6 null uniformity",
                   "G7b engine null is NOT the degenerate z-shift",
                   "G8 dependence preservation (common rotation)",
                   "G9 no-testable-edges -> typed CANNOT_DETERMINE",
                   "G10 persistence cannot promote",
                   "G11 no-power -> CANNOT_DETERMINE_NO_POWER",
                   "G12 B-2 identifiability refusals",
                   "G13 B-3 deterministic selection + small-day",
                   "G14 substream seed exact vector",
                   "G15 LOCO conjunctive gate",
                   "G16b power-contract typing (certified flag)"):
            check(nm, False, "ENGINE_ABSENT")
        amendment2(False)
        return

    # G1 seams + registered constants ------------------------------------------
    ok1 = all(callable(getattr(E, f, None)) for f in
              ("walk_forward_split", "b1_family", "b2_family", "b3_family",
               "loco_gate", "derive_substream_seed"))
    ok1 = ok1 and E.N_DRAWS == 9999 and abs(E.ALPHA_FAMILY - 0.05 / 3) < 1e-15 \
        and E.BASELINE_DAYS == 60 and E.TESTABLE_MIN_BASELINE == 45 \
        and E.PERSISTENCE_K == 3 and float(E.Z_PERSIST) == 3.0 \
        and abs(E.TOP_DECILE - 0.10) < 1e-15 and abs(E.BY_Q - 0.05) < 1e-15
    check("G1 seams+constants", ok1)

    # G2 split ------------------------------------------------------------------
    days100 = [day(i) for i in range(100)]
    b, ev = E.walk_forward_split(days100)
    ok2 = list(b) == days100[:60] and list(ev) == days100[60:]
    try:
        E.walk_forward_split(days100[:60])
        ok2b, det2 = False, "accepted 60-day sequence with empty evaluation"
    except Exception as exc:
        ok2b, det2 = "INSUFFICIENT" in str(exc).upper(), f"{exc}"
    check("G2 walk-forward split", ok2 and ok2b, det2 if not ok2b else "")

    # G3 testable floor ---------------------------------------------------------
    p3 = mk_panel(90, ["AA.S1|AA.S2", "AA.S1|AA.S3"], seed=3)
    c3 = p3["carriers"]["c_fix"]
    for i, d in enumerate(c3["registered_days"][:60]):
        if i >= 44:                       # leave exactly 44 finite baseline obs
            del c3["r"]["AA.S1|AA.S3"][d]
    r3 = E.b1_family(p3, doc_sha256=DOC, n_draws=199,
                     power_contract={"passed": True})
    ok3 = r3["excluded"].get("INSUFFICIENT_BASELINE", 0) == 1 \
        and r3["testable_edges"] == 1
    check("G3 testable floor 45", ok3,
          f"excluded={r3['excluded']} testable={r3.get('testable_edges')}")

    # G4 MAD=0 ------------------------------------------------------------------
    p4 = mk_panel(90, ["AA.S1|AA.S2", "AA.S1|AA.S3"], seed=4)
    c4 = p4["carriers"]["c_fix"]
    for d in c4["registered_days"]:
        c4["r"]["AA.S1|AA.S2"][d] = 0.5          # constant -> MAD 0
    r4 = E.b1_family(p4, doc_sha256=DOC, n_draws=199,
                     power_contract={"passed": True})
    ok4 = r4["excluded"].get("DEGENERATE_BASELINE", 0) == 1
    check("G4 degenerate baseline", ok4, f"excluded={r4['excluded']}")

    # G5 REV 2 -- ANNEX-CONDITIONED (codex adjudication 633be61 item 3;
    # the unconditional must-reject expectation is incompatible with frozen
    # section 5 when no MDE is certified: the frozen carrier-rotation null has
    # a structural power ceiling at this data scale -- cayley 556941e, held
    # as EXPECTED POWER-CONTRACT RED by codex pending the annex) ---------------
    edges5 = [f"AA.S{i}|AA.S{j}" for i in range(1, 7) for j in range(i + 1, 7)]
    p5 = plant(mk_panel(100, edges5, seed=5), edges5[:3], 75, 3, 12.0)
    res5 = _results(B1_RESULTS)
    if res5 is None:
        check("G5 annex-conditioned planted recovery (REV 2)", False,
              "POWER_RESULTS_ABSENT -- expected red until the admitted Tier-C "
              "estimation completes and routes the B-1 results artifact")
    else:
        bound5 = _results_bound(res5)
        cert5 = res5.get("certified_points") or []
        if not cert5:
            r5 = E.b1_family(p5, doc_sha256=DOC, n_draws=999,
                             power_contract={
                                 "certified": False,
                                 "terminal": res5.get("terminal_type")})
            v5 = str(r5.get("verdict"))
            ok5 = bound5 \
                and res5.get("terminal_type") \
                == "MDE_NOT_CERTIFIED_BY_REGISTERED_SEARCH" \
                and "CANNOT_DETERMINE_NO_POWER" in v5 \
                and "POSITIVE" not in v5.upper() and v5 != "NEGATIVE"
            check("G5 annex-conditioned planted recovery (REV 2)", ok5,
                  f"bound={bound5} terminal={res5.get('terminal_type')} "
                  f"verdict={v5}")
        else:
            try:
                from scipy.stats import beta as _beta
                rep5 = _lex_first_pareto(cert5)
                claimed5 = res5.get("pareto_minimal_certified") or []
                claim_ok = any(tuple(p.get("grid_point", ())) ==
                               tuple(rep5["grid_point"]) for p in claimed5)
                k5 = int(rep5["post_loco"]["successes"])
                n5 = int(rep5["post_loco"]["replicates"])
                lb5 = float(_beta.ppf(0.05, k5, n5 - k5 + 1)) if k5 > 0 else 0.0
                ok5 = bound5 and claim_ok and lb5 >= 0.80
                check("G5 annex-conditioned planted recovery (REV 2)", ok5,
                      f"bound={bound5} in_claimed_frontier={claim_ok} "
                      f"representative={rep5.get('grid_point')} "
                      f"post-LOCO {k5}/{n5} 95%LB={lb5:.4f}")
            except Exception as exc:
                check("G5 annex-conditioned planted recovery (REV 2)", False,
                      f"{type(exc).__name__}: {exc}")

    # G6 null uniformity --------------------------------------------------------
    ps = []
    for s in range(20):
        pn = mk_panel(100, edges5[:6], seed=100 + s)
        rn = E.b1_family(pn, doc_sha256=DOC, n_draws=499,
                         power_contract={"passed": True})
        ps.append(rn["p_value"])
    frac_sig = sum(1 for p in ps if p is not None and p <= 0.05) / len(ps)
    ok6 = all(p is not None for p in ps) and frac_sig <= 0.25 \
        and 0.25 <= float(np.mean([p for p in ps])) <= 0.80 \
        and min(ps) >= 1.0 / 500
    check("G6 null uniformity", ok6,
          f"frac_sig={frac_sig} mean={float(np.mean(ps)):.3f} min={min(ps)}")

    # G7b engine null must NOT be the degenerate z-shift (G7a ran above) --------
    r7 = E.b1_family(p5, doc_sha256=DOC, n_draws=499,
                     power_contract={"passed": True}, return_null=True)
    null_t = np.asarray(r7["null_T"], dtype=float)
    finite = null_t[np.isfinite(null_t)]
    add_one = (1 + int(np.sum(finite >= r7["T_obs"]))) / (len(finite) + 1)
    ok7b = finite.size > 0 and float(np.std(finite)) > 0 \
        and not np.allclose(finite, r7["T_obs"]) \
        and abs(r7["p_value"] - add_one) < 1e-12
    check("G7b engine null is NOT the degenerate z-shift", ok7b,
          f"std={float(np.std(finite)) if finite.size else 'NA'} "
          f"p={r7['p_value']} add_one={add_one}")

    # G8 dependence preservation: N identical edges == 1 edge, draw-for-draw ----
    p8a = mk_panel(100, ["AA.S1|AA.S2"], seed=8)
    series = dict(p8a["carriers"]["c_fix"]["r"]["AA.S1|AA.S2"])
    p8b = mk_panel(100, edges5[:5], seed=8)
    for e in edges5[:5]:
        p8b["carriers"]["c_fix"]["r"][e] = dict(series)
    ra = E.b1_family(p8a, doc_sha256=DOC, n_draws=299,
                     power_contract={"passed": True}, return_null=True)
    rb = E.b1_family(p8b, doc_sha256=DOC, n_draws=299,
                     power_contract={"passed": True}, return_null=True)
    ok8 = np.array_equal(np.asarray(ra["null_T"]), np.asarray(rb["null_T"])) \
        and ra["T_obs"] == rb["T_obs"]
    check("G8 dependence preservation (common rotation)", ok8,
          "5 perfectly dependent edges produced a different null than 1 edge "
          "-- offsets are not common per carrier-draw")

    # G9 all-excluded -> typed, never a fabricated p ----------------------------
    p9 = mk_panel(90, ["AA.S1|AA.S2"], seed=9)
    for d in p9["carriers"]["c_fix"]["registered_days"]:
        p9["carriers"]["c_fix"]["r"]["AA.S1|AA.S2"][d] = 0.5
    r9 = E.b1_family(p9, doc_sha256=DOC, n_draws=199,
                     power_contract={"passed": True})
    ok9 = r9["p_value"] is None and "CANNOT_DETERMINE" in str(r9["verdict"])
    check("G9 no-testable-edges -> typed CANNOT_DETERMINE", ok9,
          f"verdict={r9.get('verdict')} p={r9.get('p_value')}")

    # G10 persistence is secondary and cannot promote ---------------------------
    r10 = E.b1_family(mk_panel(100, edges5[:6], seed=10), doc_sha256=DOC,
                      n_draws=499, power_contract={"passed": True})
    ok10 = isinstance(r10.get("persistence"), dict) \
        and r10["persistence"].get("verdict_bearing") is False \
        and (r10["p_value"] > E.ALPHA_FAMILY) == ("POSITIVE" not in
                                                  str(r10["verdict"]).upper())
    check("G10 persistence cannot promote", ok10,
          f"persistence={r10.get('persistence')} verdict={r10.get('verdict')}")

    # G11 no power contract -> typed nonpositive --------------------------------
    r11 = E.b1_family(mk_panel(100, edges5[:6], seed=11), doc_sha256=DOC,
                      n_draws=199, power_contract=None)
    ok11 = "CANNOT_DETERMINE_NO_POWER" in str(r11["verdict"])
    check("G11 no-power -> CANNOT_DETERMINE_NO_POWER", ok11,
          f"verdict={r11.get('verdict')}")

    # G12 B-2 identifiability refusals ------------------------------------------
    def two_day_panel(edges_by_day):
        n = 62
        days_ = [day(i) for i in range(n)]
        stations = sorted({s for eds in edges_by_day for e in eds
                           for s in e.split("|")})
        r = {}
        for eds in edges_by_day:
            for e in eds:
                r.setdefault(e, {})
        for i, d in enumerate(days_):
            if i < 60:
                for e in r:
                    r[e][d] = 0.5      # trivially fills baseline
            else:
                for e in edges_by_day[i - 60]:
                    r[e][d] = 0.8
        return {"carriers": {"c_fix": {
            "registered_days": days_, "stations": stations,
            "segments": {s: "seg_a" for s in stations}, "r": r}}}

    tie = ["AA.S1|AA.S2", "AA.S2|AA.S3", "AA.S4|AA.S5", "AA.S5|AA.S6"]
    r12a = E.b2_family(two_day_panel([tie, tie]), doc_sha256=DOC, n_draws=99)
    ok12a = any(x["code"] == "LCC_TIE" for x in r12a["day_refusals"])
    k4 = ["BB.S1|BB.S2", "BB.S1|BB.S3", "BB.S1|BB.S4",
          "BB.S2|BB.S3", "BB.S2|BB.S4", "BB.S3|BB.S4"]
    r12b = E.b2_family(two_day_panel([k4, k4]), doc_sha256=DOC, n_draws=99)
    ok12b = any(x["code"] == "FIEDLER_DEGENERATE" for x in r12b["day_refusals"])
    p3g = ["CC.S1|CC.S2", "CC.S2|CC.S3"]
    r12c = E.b2_family(two_day_panel([p3g, p3g]), doc_sha256=DOC, n_draws=99)
    ok12c = any(x["code"] == "FIEDLER_ZERO_COORDINATE"
                for x in r12c["day_refusals"])
    p4g1 = ["DD.S1|DD.S2", "DD.S2|DD.S3", "DD.S3|DD.S4"]
    p4g2 = ["DD.S2|DD.S3", "DD.S3|DD.S4", "DD.S4|DD.S5"]
    r12d = E.b2_family(two_day_panel([p4g1, p4g2]), doc_sha256=DOC, n_draws=99)
    ok12d = any(x["code"] == "NODESET_MISMATCH" for x in r12d["day_refusals"])
    r12e = E.b2_family(two_day_panel([p4g1, p4g1]), doc_sha256=DOC, n_draws=99)
    ok12e = not r12e.get("day_refusals") and r12e.get("max_switches") == 0
    check("G12 B-2 identifiability refusals",
          ok12a and ok12b and ok12c and ok12d and ok12e,
          f"tie={ok12a} degen={ok12b} zero={ok12c} nodeset={ok12d} "
          f"stable={ok12e} ({r12e.get('max_switches')})")

    # G13 B-3 deterministic selection + small-day -------------------------------
    e13 = [f"EE.S1|EE.S{j}" for j in range(2, 13)]          # m = 11 -> K = 2
    p13 = mk_panel(100, e13, seed=13, sigma=0.0)
    c13 = p13["carriers"]["c_fix"]
    for i, d in enumerate(c13["registered_days"]):
        for e in e13:
            c13["r"][e][d] = (0.001 if i % 2 == 0 else -0.001) if i < 60 else 0.0
    d_ev = c13["registered_days"][60]
    for e in e13:
        c13["r"][e][d_ev] = 0.01                          # all |z| exactly tied
    r13 = E.b3_family(p13, doc_sha256=DOC, n_draws=99)
    sel = r13["selection"][d_ev]
    expect = sorted(e13, key=lambda e: (e.split("|")[0], e.split("|")[1]))[:2]
    ok13a = list(sel) == expect
    p13b = mk_panel(62, ["FF.S1|FF.S2"], seed=14, sigma=0.0)
    c13b = p13b["carriers"]["c_fix"]
    for i, d in enumerate(c13b["registered_days"]):
        c13b["r"]["FF.S1|FF.S2"][d] = (0.001 if i % 2 == 0 else -0.001) \
            if i < 60 else c13b["r"]["FF.S1|FF.S2"][d]
    for d in c13b["registered_days"][60:]:
        del c13b["r"]["FF.S1|FF.S2"][d]                    # m = 0 on eval days
    r13b = E.b3_family(p13b, doc_sha256=DOC, n_draws=99)
    ok13b = any("INSUFFICIENT_DAILY_EDGES" in str(x)
                for x in r13b.get("day_refusals", []))
    check("G13 B-3 deterministic selection + small-day", ok13a and ok13b,
          f"sel={sel} expect={expect} smallday={ok13b}")

    # G14 substream seed exact vector (prereg rev-2 formula verbatim) -----------
    ok14 = True
    det14 = []
    for fam, fold, purpose in (("B1", "full", "null"),
                               ("B2", "loco:KO.GEML", "null"),
                               ("B3", "full", "power")):
        exp = int.from_bytes(hashlib.sha256(
            f"{DOC}||{fam}||{fold}||{purpose}".encode("utf-8"))
            .digest()[:8], "big")
        got = E.derive_substream_seed(DOC, fam, fold, purpose)
        if got != exp:
            ok14 = False
            det14.append(f"{fam}/{fold}/{purpose}: got={got} expect={exp}")
    check("G14 substream seed exact vector", ok14, "; ".join(det14))

    # G15 LOCO conjunctive gate --------------------------------------------------
    a = E.ALPHA_FAMILY
    full_pos = {"p_value": a / 2, "verdict": "POSITIVE"}
    f_pass, f_fail = {"p_value": a / 2}, {"p_value": a * 2}
    g1 = E.loco_gate(full_pos, [f_pass, f_pass, f_pass], a)
    g2 = E.loco_gate(full_pos, [f_pass, f_fail, f_pass], a)
    g3 = E.loco_gate(full_pos, [f_pass, None, f_pass], a)
    g4 = E.loco_gate({"p_value": a * 2, "verdict": "NEGATIVE"},
                     [f_pass, f_pass], a)
    ok15 = g1.get("pass") is True and g2.get("pass") is False \
        and g3.get("pass") is False and g4.get("pass") is False
    check("G15 LOCO conjunctive gate", ok15,
          f"all={g1.get('pass')} onefail={g2.get('pass')} "
          f"missing={g3.get('pass')} promote={g4.get('pass')}")

    # G16b power-contract typing (AMENDMENT 1): a not-certified contract types
    # exactly CANNOT_DETERMINE_NO_POWER; a certified contract must NOT --------
    p16 = mk_panel(100, edges5[:6], seed=16)
    r16n = E.b1_family(p16, doc_sha256=DOC, n_draws=199,
                       power_contract={
                           "certified": False,
                           "terminal":
                               "MDE_NOT_CERTIFIED_BY_REGISTERED_SEARCH"})
    r16c = E.b1_family(p16, doc_sha256=DOC, n_draws=199,
                       power_contract={"certified": True})
    ok16b = "CANNOT_DETERMINE_NO_POWER" in str(r16n.get("verdict")) \
        and "CANNOT_DETERMINE_NO_POWER" not in str(r16c.get("verdict"))
    check("G16b power-contract typing (certified flag)", ok16b,
          f"not_certified={r16n.get('verdict')} "
          f"certified={r16c.get('verdict')}")

    # ---- AMENDMENT 2: the mandatory frozen sec-A5 classes --------------------
    amendment2(all(callable(getattr(E, f, None))
                   for f in ("b1a_family", "b2a_family", "b3a_family")))


main()
print()
if HERE_FAILS:
    print(f"PHASE-B STATS RED-KAT FAILURES ({len(HERE_FAILS)}): {HERE_FAILS}")
    sys.exit(1)
print("ALL PHASE-B STATS RED-KATs PASS")
