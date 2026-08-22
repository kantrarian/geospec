#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 MAG-1 engine, part A: numeric core (cayley) -- per the
FROZEN instantiation docs/f2g_window2_freeze/mag1_instantiation.md +
the byte-authoritative SOS artifact mag1_band_b_sos.json + the typed
input capsules (design freeze CLOSED @ 12161f6/5fba544) and grassmann's
bar seam pin ("w2_mag1 seams"). Seam FIXED as `w2_mag1`. Part A covers
the surfaces with exact W-MAG KATs: SOS binding, the segment/edge-
exclusion filter chain, capsule-driven frame conversion, carrier
endpoint typing, and the internal three-primary Holm. (Part B --
subtraction ledger fit/apply, M1/M2/M3 window machinery -- lands
separately; M2's daily-energy day floor is here because it binds the
filter chain.)

Byte authorities, verified BEFORE use (content-auth rule):
- SOS: the artifact's `sos_serialized` string must hash to the pinned
  `77bceec4...`; the coefficient array is loaded FROM that string. When
  the local SciPy equals the pinned 1.18.0 the array is regenerated via
  `butter(4, [0.001, 0.004], btype='bandpass', fs=1/60, output='sos')`
  and must be ARRAY-EQUAL (typed SOS_REGENERATION_MISMATCH); on any
  other SciPy the committed bytes stand alone (disclosed in the load
  record).
- Capsules: read from git objects at the frozen manifest commit; the
  probe body must hash to the capsule's `probe_body_sha256`.

Frozen filter semantics (codex-closed): sosfiltfilt, padtype='odd',
padlen=27 (fixed integer); causal span 266 (last |h[n]| > 1e-12);
NO silent interpolation -- fill/NaN samples SPLIT the series into
contiguous segments; edge exclusion 532 samples after EACH segment
boundary; usable_N = N - 2*532 per segment (<= 0 -> the segment is
FILTER_SUPPORT_INSUFFICIENT); the 90% day floor (1296/1440 admissible
minutes) applies only to positive surviving support.

Frame rule (codex revision-2/3): conversion is CAPSULE-DRIVEN --
sensor_orientation must be a registered convention (XYZS: geographic
X/Y/Z identity with scalar S EXCLUDED from sqrt(rX^2+rY^2); XYZF:
identity with F excluded); a mutated/omitted orientation, a missing
mapped component array, or a component map that routes the excluded
scalar into the horizontal vector refuses typed (FRAME_NOT_CLOSED /
EXCLUDED_CHANNEL_IN_HORIZONTAL). The conversion returns ONLY the
horizontal pair (+ Z separately) -- the scalar channel cannot leak
structurally.

This module opens no window-2 value.
"""
import hashlib
import json
import math
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

MANIFEST_COMMIT = "5fba5446cd96722fb86021df5065a46b8a1a78f5"
SOS_PATH = "docs/f2g_window2_freeze/mag1_band_b_sos.json"
CAPSULE_PATHS = {
    "vic": "docs/f2g_window2_freeze/mag_capsule_vic.json",
    "new": "docs/f2g_window2_freeze/mag_capsule_new.json"}
SOS_SERIALIZED_SHA = ("77bceec4003b75d11ac671d86fb79342a265a12364fc6101"
                      "e80decdd6e9a7f29")
PINNED_SCIPY = "1.18.0"
PADLEN = 27
CAUSAL_SPAN = 266
SPAN_THRESHOLD = 1e-12
EDGE_EXCLUSION = 532
DAY_MINUTES = 1440
DAY_FLOOR = 1296          # 90% of 1440
ALPHA_LANE = 0.05

# registered per-carrier endpoints (frozen instantiation); the lane's
# three PRIMARIES under internal Holm
CARRIER_ENDPOINTS = {
    "istanbul_marmara": ("M1", "M2"),
    "socal_coachella": ("M1", "M2", "M3"),
    "cascadia": ("M1", "M2", "M3"),
}
UNTESTABLE_CARRIERS = {"turkey_kahramanmaras"}
PRIMARIES = (("istanbul_marmara", "M2"), ("socal_coachella", "M3"),
             ("cascadia", "M3"))

# registered frame conventions: orientation -> (identity components,
# excluded scalar channel)
FRAME_CONVENTIONS = {"XYZS": (("X", "Y", "Z"), "S"),
                     "XYZF": (("X", "Y", "Z"), "F")}


class Mag1Refusal(ValueError):
    """Typed refusal; the code leads the message."""


def _git_blob(repo, ref):
    p = subprocess.run(["git", "-C", repo or
                        os.path.abspath(os.path.join(_HERE, "..",
                                                     "..")),
                        "cat-file", "blob", ref], capture_output=True)
    if p.returncode != 0:
        raise Mag1Refusal(f"ARTIFACT_UNREADABLE: {ref}")
    return p.stdout


def load_sos(repo=None):
    """The byte-authoritative SOS. Returns (sos ndarray, record)."""
    art = json.loads(_git_blob(repo, f"{MANIFEST_COMMIT}:{SOS_PATH}")
                     .decode("utf-8"))
    ser = art["sos_serialized"]
    got = hashlib.sha256(ser.encode("utf-8")).hexdigest()
    if got != SOS_SERIALIZED_SHA or \
            art["sos_serialized_sha256"] != SOS_SERIALIZED_SHA:
        raise Mag1Refusal(f"SOS_SERIALIZATION_SHA_MISMATCH: {got[:12]}")
    sos = np.array(json.loads(ser), dtype=float)
    if sos.shape != (4, 6):
        raise Mag1Refusal(f"SOS_SHAPE_INVALID: {sos.shape}")
    if art["apply"]["padlen_integer"] != PADLEN or \
            art["causal_span"]["last_index_above"] != CAUSAL_SPAN or \
            art["edge_exclusion_samples_per_boundary"] != EDGE_EXCLUSION:
        raise Mag1Refusal("SOS_ARTIFACT_CONSTANTS_MISMATCH")
    record = {"serialized_sha256": got, "regenerated": False,
              "scipy_local": None}
    import scipy
    record["scipy_local"] = scipy.__version__
    if scipy.__version__ == PINNED_SCIPY:
        from scipy.signal import butter
        regen = butter(4, [0.001, 0.004], btype="bandpass", fs=1 / 60,
                       output="sos")
        if not np.array_equal(regen, sos):
            raise Mag1Refusal("SOS_REGENERATION_MISMATCH")
        record["regenerated"] = True
    return sos, record


def segments(values):
    """Contiguous finite runs of a 1-min series (NaN = missing splits;
    never interpolate). Returns [(start_index, ndarray), ...]."""
    v = np.asarray(values, dtype=float)
    out = []
    i = 0
    n = len(v)
    while i < n:
        if not math.isfinite(v[i]):
            i += 1
            continue
        j = i
        while j < n and math.isfinite(v[j]):
            j += 1
        out.append((i, v[i:j]))
        i = j
    return out


def segment_usable_n(seg_len):
    return seg_len - 2 * EDGE_EXCLUSION


def band_b_series(values, sos):
    """Filtered series with NaN at missing AND edge-excluded positions.
    Per segment: usable_N <= 0 -> the whole segment is excluded
    (FILTER_SUPPORT_INSUFFICIENT at segment level, positions NaN)."""
    v = np.asarray(values, dtype=float)
    out = np.full(len(v), np.nan)
    from scipy.signal import sosfiltfilt
    for start, seg in segments(v):
        if segment_usable_n(len(seg)) <= 0:
            continue
        f = sosfiltfilt(sos, seg, padtype="odd", padlen=PADLEN)
        out[start + EDGE_EXCLUSION:
            start + len(seg) - EDGE_EXCLUSION] = \
            f[EDGE_EXCLUSION:len(seg) - EDGE_EXCLUSION]
    return out


def daily_energy(filtered, day_slices):
    """M2 daily magnetic feature over the surviving filtered stream:
    per day, admissible minutes = finite surviving samples; the 90%
    floor (>= 1296/1440) gates admissibility; energy = median of
    squared band-passed values over admissible minutes. Returns
    {day: {"energy": float|None, "surviving": int,
           "typing": None|"DAY_BELOW_FLOOR"|"FILTER_SUPPORT_INSUFFICIENT"}}"""
    out = {}
    f = np.asarray(filtered, dtype=float)
    for day, (a, b) in day_slices.items():
        chunk = f[a:b]
        fin = chunk[np.isfinite(chunk)]
        if fin.size == 0:
            out[day] = {"energy": None, "surviving": 0,
                        "typing": "FILTER_SUPPORT_INSUFFICIENT"}
        elif fin.size < DAY_FLOOR:
            out[day] = {"energy": None, "surviving": int(fin.size),
                        "typing": "DAY_BELOW_FLOOR"}
        else:
            out[day] = {"energy": float(np.median(fin ** 2)),
                        "surviving": int(fin.size), "typing": None}
    return out


def load_capsule(name, repo=None):
    """Typed input capsule + sha-verified probe body (parsed).
    Returns (capsule, body)."""
    if name not in CAPSULE_PATHS:
        raise Mag1Refusal(f"CAPSULE_UNKNOWN: {name}")
    cap = json.loads(
        _git_blob(repo, f"{MANIFEST_COMMIT}:{CAPSULE_PATHS[name]}")
        .decode("utf-8"))
    env_path = cap["probe_envelope"]
    body_path = env_path.replace(".envelope.json", ".json")
    raw = _git_blob(repo, f"{MANIFEST_COMMIT}:{body_path}")
    got = hashlib.sha256(raw).hexdigest()
    if got != cap["probe_body_sha256"]:
        raise Mag1Refusal(f"CAPSULE_BODY_SHA_MISMATCH: {name} "
                          f"{got[:12]}")
    return cap, json.loads(raw.decode("utf-8"))


def convert_frame(capsule, arrays, source_orientation):
    """Capsule-driven typed frame conversion -> (X_north, Y_east,
    Z_down) ndarrays (nulls -> NaN). The excluded scalar channel is
    NEVER returned in the horizontal pair (structural non-leak)."""
    orient = capsule.get("sensor_orientation")
    if orient is None or orient not in FRAME_CONVENTIONS:
        raise Mag1Refusal(f"FRAME_NOT_CLOSED: sensor_orientation="
                          f"{orient!r} has no registered conversion")
    if source_orientation != orient:
        raise Mag1Refusal(
            f"FRAME_NOT_CLOSED: capsule orientation {orient!r} != "
            f"source {source_orientation!r}")
    comps, excluded = FRAME_CONVENTIONS[orient]
    cmap = capsule.get("component_map") or {}
    picks = {}
    for axis, key in (("geographic_X_north", "X"),
                      ("geographic_Y_east", "Y"),
                      ("geographic_Z_down", "Z")):
        src = cmap.get(axis)
        if src is None:
            raise Mag1Refusal(f"FRAME_NOT_CLOSED: component_map "
                              f"lacks {axis}")
        if key != "Z" and src == excluded:
            raise Mag1Refusal(
                f"EXCLUDED_CHANNEL_IN_HORIZONTAL: {src!r} mapped to "
                f"{axis}")
        if src not in comps:
            raise Mag1Refusal(f"FRAME_NOT_CLOSED: {axis} maps to "
                              f"unregistered component {src!r}")
        if src not in arrays:
            raise Mag1Refusal(f"FRAME_NOT_CLOSED: component array "
                              f"{src!r} absent from source")
        picks[key] = np.array(
            [float("nan") if x is None else float(x)
             for x in arrays[src]], dtype=float)
    lens = {len(v) for v in picks.values()}
    if len(lens) != 1:
        raise Mag1Refusal(f"FRAME_NOT_CLOSED: component length "
                          f"mismatch {sorted(lens)}")
    return picks["X"], picks["Y"], picks["Z"]


def horizontal_residual(r_x, r_y):
    """The statistic component: sqrt(rX^2 + rY^2). Takes ONLY the two
    horizontal residual arrays -- the scalar channel has no path in."""
    return np.sqrt(np.asarray(r_x, dtype=float) ** 2
                   + np.asarray(r_y, dtype=float) ** 2)


def endpoints_for(carrier):
    if carrier in UNTESTABLE_CARRIERS:
        raise Mag1Refusal(f"MAG_UNTESTABLE: {carrier} (no coverage; "
                          "amendable pre-freeze only, by disclosed "
                          "amendment)")
    if carrier not in CARRIER_ENDPOINTS:
        raise Mag1Refusal(f"CARRIER_UNKNOWN: {carrier}")
    return CARRIER_ENDPOINTS[carrier]


def holm_internal(p_by_primary, alpha=ALPHA_LANE):
    """Holm over the lane's three primaries. p_by_primary:
    {(carrier, endpoint): p}. Keys must be exactly PRIMARIES."""
    if set(p_by_primary) != set(PRIMARIES):
        raise Mag1Refusal(
            f"HOLM_STRUCTURE_MISMATCH: keys "
            f"{sorted(p_by_primary)} != registered primaries")
    m = len(PRIMARIES)
    order = sorted(p_by_primary, key=lambda k: p_by_primary[k])
    rejected = {}
    still = True
    for i, k in enumerate(order):
        thresh = alpha / (m - i)
        if still and p_by_primary[k] <= thresh:
            rejected[k] = True
        else:
            still = False
            rejected[k] = False
    return {"alpha": alpha, "order": [list(k) for k in order],
            "rejected": {f"{c}:{e}": rejected[(c, e)]
                         for c, e in PRIMARIES}}


# ---------------------------------------------------------------- selftest
def _selftest():
    sos, rec = load_sos()
    assert rec["serialized_sha256"] == SOS_SERIALIZED_SHA
    # regeneration runs EXACTLY when the local SciPy is the pinned one;
    # elsewhere the committed bytes stand alone (disclosed)
    assert rec["regenerated"] == (rec["scipy_local"] == PINNED_SCIPY)

    # causal span 266/267 (impulse through the CAUSAL sosfilt)
    from scipy.signal import sosfilt
    imp = np.zeros(2000)
    imp[0] = 1.0
    h = np.abs(sosfilt(sos, imp))
    above = np.where(h > SPAN_THRESHOLD)[0]
    assert above[-1] == CAUSAL_SPAN, above[-1]
    assert h[CAUSAL_SPAN] > SPAN_THRESHOLD \
        and (h[CAUSAL_SPAN + 1:] <= SPAN_THRESHOLD).all()

    # usable_N boundary: N = 532 / 1064 / 1065
    assert segment_usable_n(532) == -532
    assert segment_usable_n(1064) == 0
    assert segment_usable_n(1065) == 1
    rng = np.random.Generator(np.random.PCG64(5))
    for n, expect in ((532, 0), (1064, 0), (1065, 1)):
        f = band_b_series(rng.normal(size=n), sos)
        assert int(np.isfinite(f).sum()) == expect, (n, expect)

    # single NaN splits (two segments, both edge-excluded); 100-gap same
    v = rng.normal(size=3000)
    assert int(np.isfinite(band_b_series(v, sos)).sum()) == 3000 - 1064
    v1 = v.copy()
    v1[1500] = np.nan
    surv = int(np.isfinite(band_b_series(v1, sos)).sum())
    assert surv == (1500 - 1064) + (1499 - 1064), surv
    v2 = v.copy()
    v2[1500:1600] = np.nan
    surv = int(np.isfinite(band_b_series(v2, sos)).sum())
    assert surv == (1500 - 1064) + (1400 - 1064), surv

    # daily floor: full clean 3-day stream -> middle day fully
    # surviving (1440 >= 1296), first/last days lose the stream edges
    v3 = rng.normal(size=3 * DAY_MINUTES)
    f3 = band_b_series(v3, sos)
    slices = {f"d{i}": (i * DAY_MINUTES, (i + 1) * DAY_MINUTES)
              for i in range(3)}
    de = daily_energy(f3, slices)
    assert de["d1"]["typing"] is None and de["d1"]["surviving"] == 1440
    assert de["d0"]["typing"] == "DAY_BELOW_FLOOR" \
        and de["d0"]["surviving"] == DAY_MINUTES - EDGE_EXCLUSION
    assert de["d2"]["typing"] == "DAY_BELOW_FLOOR"
    empty = daily_energy(np.full(DAY_MINUTES, np.nan),
                         {"d": (0, DAY_MINUTES)})
    assert empty["d"]["typing"] == "FILTER_SUPPORT_INSUFFICIENT"

    # VIC capsule + body: sha-verified load, frame conversion, the
    # four W-MAG frame refusal cases
    cap, body = load_capsule("vic")
    x, y, z = convert_frame(cap, body, body["@info"]
                            ["sensor_orientation"])
    assert len(x) == len(y) == len(z) == 1440
    assert x[0] == 18077.1 and y[0] == 4914.06   # source-exact identity
    hz = horizontal_residual(x[:5] - x[:5].mean(),
                             y[:5] - y[:5].mean())
    assert hz.shape == (5,)
    for doctor, code in (
            (lambda c: c.__setitem__("sensor_orientation", "HDZS"),
             "FRAME_NOT_CLOSED"),
            (lambda c: c.pop("sensor_orientation"),
             "FRAME_NOT_CLOSED"),
            (lambda c: c["component_map"].pop("geographic_Z_down"),
             "FRAME_NOT_CLOSED"),
            (lambda c: c["component_map"].__setitem__(
                "geographic_Y_east", "S"),
             "EXCLUDED_CHANNEL_IN_HORIZONTAL")):
        c = json.loads(json.dumps(cap))
        doctor(c)
        try:
            convert_frame(c, body, "XYZS")
            raise AssertionError(f"doctored capsule must refuse: {code}")
        except Mag1Refusal as e:
            assert code in str(e), (code, str(e))
    body_no_z = dict(body)
    del body_no_z["Z"]
    try:
        convert_frame(cap, body_no_z, "XYZS")
        raise AssertionError("omitted component array must refuse")
    except Mag1Refusal as e:
        assert "FRAME_NOT_CLOSED" in str(e)

    # carrier typing + Holm structure
    assert endpoints_for("cascadia") == ("M1", "M2", "M3")
    try:
        endpoints_for("turkey_kahramanmaras")
        raise AssertionError("kahramanmaras must type MAG_UNTESTABLE")
    except Mag1Refusal as e:
        assert "MAG_UNTESTABLE" in str(e)
    h = holm_internal({("istanbul_marmara", "M2"): 0.01,
                       ("socal_coachella", "M3"): 0.02,
                       ("cascadia", "M3"): 0.2})
    assert h["rejected"] == {"istanbul_marmara:M2": True,
                             "socal_coachella:M3": True,
                             "cascadia:M3": False}
    try:
        holm_internal({("istanbul_marmara", "M2"): 0.01})
        raise AssertionError("wrong Holm structure must refuse")
    except Mag1Refusal as e:
        assert "HOLM_STRUCTURE_MISMATCH" in str(e)

    print("w2_mag1 selftest (part A): ALL PASS")


# ===================================================================
# PART B -- subtraction fit/apply ledger + M1/M2/M3 machinery
#
# Interpretation pins for part B (disclosed, R1.2-able; the two
# STATISTIC-TYPE pins below are heavier than usual and carry an
# explicit R1.2 window -- the frozen texts pin the null geometry, lag
# set, overlap floor, and feature definitions but not the pairing
# statistic's correlation type or the M1 window statistic's exact
# robust form):
# - M2 pairing statistic := max over the frozen lag set {0,+-1,+-2,+-3}
#   of the SPEARMAN (midrank) correlation between the daily magnetic
#   feature and the daily graph series on the exact shared-day mask;
#   one-sided HIGH. [R1.2 WINDOW]
# - M1/M3 window statistic := median of squared residual over the
#   window's admissible minutes (the same robust band-energy form as
#   the M2 daily feature). [R1.2 WINDOW]
# - M2 whole-day rotation commutes with the per-day-local daily
#   feature (energy depends only on that day's minutes and the
#   mag/weather pair rotates JOINTLY), so the null is implemented as
#   rotation of the daily feature vector -- an EXACT equivalence, not
#   an approximation; the weather pairing is preserved structurally.
# - M1 onset-hour class := floor(hour/6) (four 6-hour classes);
#   pseudo-onset candidates overlapping the event window are excluded;
#   pool < n_controls refuses typed (never silently reduced).
# - Subtraction fit floor: masked rows >= 2 x design columns and full
#   column rank (typed SUBTRACTION_*); apply verifies the ledger
#   digest (typed LEDGER_MUTATED) and NEVER recomputes coefficients.
# ===================================================================
from datetime import datetime as _dt

M2_LAGS = (0, 1, -1, 2, -2, 3, -3)
M2_MIN_OVERLAP = 60
M2_EXCLUDED_OFFSET = 3
M1_N_CONTROLS = 999
LST_HARMONIC_HOURS = (24.0, 12.0, 8.0)
SEASONAL_DAYS = (365.25, 182.63)
CAL_EPOCH = "2026-01-01"


def _frac_days(times_iso):
    ep = _dt.fromisoformat(CAL_EPOCH + "T00:00:00")
    out = np.empty(len(times_iso), dtype=float)
    for i, t in enumerate(times_iso):
        d = _dt.fromisoformat(str(t).replace("Z", ""))
        out[i] = (d - ep).total_seconds() / 86400.0
    return out


def build_design_matrix(times_iso, lon_east, weather):
    """Frozen regressor design: intercept + weather columns (aligned
    arrays, producer-acquired per capsule fill policy) + local-solar-
    time harmonics (24/12/8 h sin+cos) + seasonal harmonics (365.25 d,
    182.63 d sin+cos). Returns (X, column_names)."""
    t = _frac_days(times_iso)
    lst_hours = ((t % 1.0) * 24.0 + lon_east / 15.0) % 24.0
    cols = [np.ones(len(t))]
    names = ["intercept"]
    for name in sorted(weather):
        cols.append(np.asarray(weather[name], dtype=float))
        names.append(f"weather:{name}")
    for hh in LST_HARMONIC_HOURS:
        w = 2.0 * math.pi * lst_hours / hh
        cols.append(np.sin(w))
        names.append(f"lst_sin_{hh}h")
        cols.append(np.cos(w))
        names.append(f"lst_cos_{hh}h")
    for dd in SEASONAL_DAYS:
        w = 2.0 * math.pi * t / dd
        cols.append(np.sin(w))
        names.append(f"seasonal_sin_{dd}d")
        cols.append(np.cos(w))
        names.append(f"seasonal_cos_{dd}d")
    return np.column_stack(cols), names


def _ledger_digest(names, coef, meta):
    return hashlib.sha256(json.dumps(
        {"columns": names, "coef": coef, "meta": meta},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fit_subtraction(times_iso, values, lon_east, weather, meta=None):
    """Fit ONCE on the calibration interval; returns the frozen
    ledger. Typed refusals on support/rank."""
    X, names = build_design_matrix(times_iso, lon_east, weather)
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if int(mask.sum()) < 2 * X.shape[1]:
        raise Mag1Refusal(
            f"SUBTRACTION_INSUFFICIENT_SUPPORT: {int(mask.sum())} "
            f"rows < 2x{X.shape[1]} columns")
    Xm = X[mask]
    if np.linalg.matrix_rank(Xm) < X.shape[1]:
        raise Mag1Refusal("SUBTRACTION_DESIGN_RANK_DEFICIENT")
    coef, *_ = np.linalg.lstsq(Xm, y[mask], rcond=None)
    meta = dict(meta or {})
    meta.setdefault("lon_east", lon_east)
    coefs = [float(c) for c in coef]
    return {"columns": names, "coef": coefs, "meta": meta,
            "digest": _ledger_digest(names, coefs, meta)}


def apply_subtraction(ledger, times_iso, values, weather):
    """APPLY-NEVER-REFIT: residual = y - X @ frozen_coef. The ledger
    digest is verified first; coefficients are never recomputed."""
    if ledger.get("digest") != _ledger_digest(
            ledger["columns"], ledger["coef"], ledger["meta"]):
        raise Mag1Refusal("LEDGER_MUTATED")
    X, names = build_design_matrix(times_iso,
                                   ledger["meta"]["lon_east"], weather)
    if names != ledger["columns"]:
        raise Mag1Refusal(f"LEDGER_DESIGN_MISMATCH: {names[:3]}...")
    y = np.asarray(values, dtype=float)
    return y - X @ np.asarray(ledger["coef"], dtype=float)


def fit_m3_reference(local_resid, ref_resid, weather_cols, meta=None):
    """M3 innovation regression, SAME fit-once discipline: local
    residual ~ reference residual + space-weather terms (frozen at
    calibration; codex binding interpretation)."""
    cols = [np.ones(len(local_resid)),
            np.asarray(ref_resid, dtype=float)]
    names = ["intercept", "reference_residual"]
    for name in sorted(weather_cols):
        cols.append(np.asarray(weather_cols[name], dtype=float))
        names.append(f"weather:{name}")
    X = np.column_stack(cols)
    y = np.asarray(local_resid, dtype=float)
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if int(mask.sum()) < 2 * X.shape[1]:
        raise Mag1Refusal("M3_INSUFFICIENT_SUPPORT")
    if np.linalg.matrix_rank(X[mask]) < X.shape[1]:
        raise Mag1Refusal("M3_DESIGN_RANK_DEFICIENT")
    coef, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
    meta = dict(meta or {})
    coefs = [float(c) for c in coef]
    return {"columns": names, "coef": coefs, "meta": meta,
            "digest": _ledger_digest(names, coefs, meta)}


def apply_m3(ledger, local_resid, ref_resid, weather_cols):
    """Innovation under the FROZEN M3 ledger (apply-never-refit)."""
    if ledger.get("digest") != _ledger_digest(
            ledger["columns"], ledger["coef"], ledger["meta"]):
        raise Mag1Refusal("LEDGER_MUTATED")
    cols = [np.ones(len(local_resid)),
            np.asarray(ref_resid, dtype=float)]
    for name in sorted(weather_cols):
        cols.append(np.asarray(weather_cols[name], dtype=float))
    X = np.column_stack(cols)
    return np.asarray(local_resid, dtype=float) \
        - X @ np.asarray(ledger["coef"], dtype=float)


def window_energy(filtered, a, b, min_support=1):
    """M1/M3 window statistic [R1.2 WINDOW pin]: median of squared
    residual over the window's admissible (finite) minutes."""
    chunk = np.asarray(filtered[a:b], dtype=float)
    fin = chunk[np.isfinite(chunk)]
    if fin.size < min_support:
        raise Mag1Refusal(
            f"M1_WINDOW_SUPPORT_INSUFFICIENT: {fin.size} < "
            f"{min_support}")
    return float(np.median(fin ** 2))


def _quarter(day_iso):
    return (int(str(day_iso)[5:7]) - 1) // 3


def m1_controls(event_onset_day, event_onset_hour, duration_days,
                admissible_days, record_days, rng,
                n_controls=M1_N_CONTROLS):
    """Blocked pseudo-onset sampler: candidates share the event's
    season QUARTER, its onset-hour CLASS (floor(hour/6)), and its
    duration; the window must lie fully inside admissible days with NO
    circular wrap; candidates overlapping the event window are
    excluded. Pool < n_controls refuses typed."""
    adm = set(str(d) for d in admissible_days)
    rec = sorted(str(d) for d in record_days)
    q = _quarter(event_onset_day)
    hclass = int(event_onset_hour) // 6
    hours = [h for h in range(24) if h // 6 == hclass]
    from datetime import date as _date, timedelta as _td

    def day_seq(d0):
        dd = _date.fromisoformat(d0)
        return [(dd + _td(days=i)).isoformat()
                for i in range(duration_days)]

    ev_days = set(day_seq(str(event_onset_day)))
    pool = []
    for d0 in rec:
        if _quarter(d0) != q:
            continue
        seq = day_seq(d0)
        if seq[-1] > rec[-1]:            # no wrap: window must fit
            continue
        if not all(s in adm for s in seq):
            continue
        if set(seq) & ev_days:           # exclude event overlap
            continue
        for h in hours:
            pool.append((d0, h))
    if len(pool) < n_controls:
        raise Mag1Refusal(
            f"M1_CONTROL_POOL_INSUFFICIENT: {len(pool)} < "
            f"{n_controls}")
    idx = rng.choice(len(pool), size=n_controls, replace=False)
    return [pool[int(i)] for i in idx]


def m1_p(event_stat, control_stats):
    """One-sided HIGH add-one rank p among the sampled controls."""
    c = np.asarray(control_stats, dtype=float)
    c = c[np.isfinite(c)]
    if c.size == 0:
        raise Mag1Refusal("M1_NO_VALID_CONTROLS")
    return float((1 + int((c >= event_stat).sum())) / (c.size + 1))


def _spearman(a, b):
    """Midrank Spearman correlation (no scipy.stats dependency)."""
    def ranks(x):
        order = np.argsort(x, kind="mergesort")
        rk = np.empty(len(x), dtype=float)
        sx = np.asarray(x)[order]
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and sx[j + 1] == sx[i]:
                j += 1
            rk[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return rk
    ra, rb = ranks(np.asarray(a, float)), ranks(np.asarray(b, float))
    ra -= ra.mean()
    rb -= rb.mean()
    den = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    if den == 0.0:
        return 0.0
    return float((ra * rb).sum() / den)


def _m2_stat(mag_by_day, graph_by_day, days):
    """Max-over-frozen-lags Spearman on the exact shared-day mask
    [R1.2 WINDOW pin]; typed overlap floor per lag."""
    pos = {d: i for i, d in enumerate(days)}
    best = None
    for lag in M2_LAGS:
        pairs = []
        for d in days:
            i = pos[d]
            j = i + lag
            if 0 <= j < len(days):
                m = mag_by_day.get(days[j])
                g = graph_by_day.get(d)
                if m is not None and g is not None:
                    pairs.append((m, g))
        if len(pairs) < M2_MIN_OVERLAP:
            continue
        rho = _spearman([p[0] for p in pairs], [p[1] for p in pairs])
        if best is None or rho > best:
            best = rho
    if best is None:
        raise Mag1Refusal(
            f"M2_OVERLAP_INSUFFICIENT: no lag reaches "
            f"{M2_MIN_OVERLAP} shared days")
    return best


def m2_pairing(mag_by_day, graph_by_day, days):
    """The M2 endpoint: observed statistic + exhaustive whole-day
    rotation null (offsets with |offset| <= 3 excluded; the mag/
    weather pair rotates jointly so rotating the per-day-local feature
    vector is EXACT -- see the part-B pin block). One-sided HIGH,
    add-one p over valid rotations."""
    days = sorted(str(d) for d in days)
    obs = _m2_stat(mag_by_day, graph_by_day, days)
    n = len(days)
    null_stats = []
    for off in range(n):
        if min(off, n - off) <= M2_EXCLUDED_OFFSET:
            continue
        rot = {days[(i + off) % n]: mag_by_day.get(days[i])
               for i in range(n)}
        try:
            null_stats.append(_m2_stat(rot, graph_by_day, days))
        except Mag1Refusal:
            continue
    if not null_stats:
        raise Mag1Refusal("M2_NULL_EMPTY")
    ge = sum(1 for s in null_stats if s >= obs)
    return {"endpoint": "M2", "T_obs": obs,
            "n_null": len(null_stats),
            "p_value": float((1 + ge) / (len(null_stats) + 1)),
            "lags": list(M2_LAGS), "min_overlap": M2_MIN_OVERLAP,
            "excluded_offsets": M2_EXCLUDED_OFFSET}


def _selftest_b():
    rng = np.random.Generator(np.random.PCG64(21))
    n = 4000
    from datetime import timedelta as _tdm
    times = [(_dt(2026, 1, 1) + _tdm(minutes=i)).isoformat()
             for i in range(n)]
    weather = {"symh": rng.normal(size=n)}
    lon = -123.42

    # fit/apply: planted linear model recovers; residual ~ noise
    X, names = build_design_matrix(times, lon, weather)
    truth = rng.normal(size=X.shape[1])
    y = X @ truth + rng.normal(0, 0.01, size=n)
    led = fit_subtraction(times, y, lon, weather)
    resid = apply_subtraction(led, times, y, weather)
    assert float(np.abs(resid).mean()) < 0.05, np.abs(resid).mean()

    # apply-never-refit: NEW data through the SAME ledger uses frozen
    # coefficients (residual biased when the world changed -- no refit)
    y2 = X @ (truth + 1.0) + rng.normal(0, 0.01, size=n)
    r2 = apply_subtraction(led, times, y2, weather)
    assert float(np.abs(r2).mean()) > 0.5   # frozen coefs, honest bias
    hacked = dict(led, coef=[c * 1.01 for c in led["coef"]])
    try:
        apply_subtraction(hacked, times, y, weather)
        raise AssertionError("mutated ledger must refuse")
    except Mag1Refusal as e:
        assert "LEDGER_MUTATED" in str(e)

    # M3 same discipline
    ref = rng.normal(size=n)
    local = 0.7 * ref + weather["symh"] * 0.2 + rng.normal(0, 0.01,
                                                           size=n)
    l3 = fit_m3_reference(local, ref, {"symh": weather["symh"]})
    innov = apply_m3(l3, local, ref, {"symh": weather["symh"]})
    assert float(np.abs(innov).mean()) < 0.05

    # M1 sampler: blocking + pool + overlap-exclusion + no-wrap
    from datetime import date as _date, timedelta as _td
    record = [( _date(2026, 1, 1) + _td(days=i)).isoformat()
              for i in range(88)]           # Q1 2026
    adm = record
    ctrls = m1_controls("2026-02-10", 3, 3, adm, record,
                        np.random.Generator(np.random.PCG64(2)),
                        n_controls=200)
    assert len(ctrls) == 200
    ev = {(_date(2026, 2, 10) + _td(days=i)).isoformat()
          for i in range(3)}
    for d0, h in ctrls:
        assert _quarter(d0) == 0 and h // 6 == 0
        seq = {(_date.fromisoformat(d0) + _td(days=i)).isoformat()
               for i in range(3)}
        assert not (seq & ev) and max(seq) <= record[-1]
    try:
        m1_controls("2026-02-10", 3, 3, adm, record,
                    np.random.Generator(np.random.PCG64(2)),
                    n_controls=10 ** 6)
        raise AssertionError("small pool must refuse")
    except Mag1Refusal as e:
        assert "M1_CONTROL_POOL_INSUFFICIENT" in str(e)
    assert m1_p(10.0, [1.0] * 99) == 1 / 100
    assert abs(m1_p(1.0, [2.0] * 99) - 1.0) < 1e-12

    # M2: planted alignment -> small p; rotation destroys it (the
    # non-identity KAT); overlap floor refuses
    days = [(_date(2026, 3, 1) + _td(days=i)).isoformat()
            for i in range(80)]
    sig = rng.normal(size=80)
    mag_d = {d: float(sig[i]) for i, d in enumerate(days)}
    gr_d = {d: float(sig[i] + rng.normal(0, 0.2))
            for i, d in enumerate(days)}
    res = m2_pairing(mag_d, gr_d, days)
    assert res["p_value"] <= 2 / (res["n_null"] + 1), res
    rot20 = {days[(i + 20) % 80]: mag_d[days[i]] for i in range(80)}
    assert _m2_stat(rot20, gr_d, days) < res["T_obs"]  # non-identity
    try:
        m2_pairing({d: mag_d[d] for d in days[:50]},
                   {d: gr_d[d] for d in days[:50]}, days[:50])
        raise AssertionError("overlap < 60 must refuse")
    except Mag1Refusal as e:
        assert "M2_OVERLAP_INSUFFICIENT" in str(e)

    # window statistic + typed support floor
    f = np.full(100, np.nan)
    f[10:90] = 2.0
    assert window_energy(f, 0, 100) == 4.0
    try:
        window_energy(f, 0, 5, min_support=1)
        raise AssertionError("empty window must refuse")
    except Mag1Refusal as e:
        assert "M1_WINDOW_SUPPORT_INSUFFICIENT" in str(e)

    # frozen constants
    assert M2_LAGS == (0, 1, -1, 2, -2, 3, -3) \
        and M2_MIN_OVERLAP == 60 and M1_N_CONTROLS == 999
    print("w2_mag1 selftest (part B): ALL PASS")


if __name__ == "__main__":
    _selftest()
    _selftest_b()
