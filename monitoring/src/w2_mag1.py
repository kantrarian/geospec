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


if __name__ == "__main__":
    _selftest()
