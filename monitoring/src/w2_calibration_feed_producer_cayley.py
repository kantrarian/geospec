#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0-4 CALIBRATION FEED PRODUCER (cayley) -- codex 1733Z ruling.

Derives the calibration runner's MAG feeds from COMMITTED staged
bytes: per registered observatory, reopen each staged day's
transcript -> body (digest-joined through the inventory and the
named store), extract the raw component arrays exactly as the
pinned transform does, run them through the ONE frame authority
(CAP._canonical_horizontal -> w2_mag1.convert_frame), and JOIN the
frozen three-regressor weather set {sym_h, kp, newell} on the
canonical UTC minute grid. ZERO HTTP; nothing here fits, admits, or
claims -- the output is exactly the runner's registered feed shape.

DERIVATION IS ADJUDICATED BY THE ADMITTED ARTIFACT (the content-auth
!= derivation-provenance rule): for every day this producer's
recomputed frame identity, support mask, null/fill censuses and
sample count must EQUAL the staged artifact's fields, else the whole
build refuses typed. The pinned artifact -- never this module's
parsing -- is the authority; a divergence is a defect surfaced, not
a value judgement made here.

D1 (codex 1733Z, the FROZEN design recovered from
docs/f2g_window2_freeze/mag1_instantiation.md): weather keys exactly
{sym_h, kp, newell}; the full canonical minute grid is RETAINED
(unsupported minutes carry the registered None state -- the frozen
engine's complete-case mask excludes them at fit time; rows are
never compacted or deleted); SYM-H and Newell join by exact
timestamp (positional on the constructed canonical grid, which the
pinned cadence gates already forced on every admitted body -- no
fuzzy/as-of join, no interpolation, no nearest-minute repair); each
Kp value expands only over its own half-open three-hour UTC
interval; an unsupported interval yields None minutes and an ABSENT
staged day refuses -- no carry, ever.

D2 (frozen, arrow per the runner semantics
feeds[target]["m3_reference"] = reference):

    FRN -> TUC ; VIC -> NEW ; IZN, TUC, NEW -> null

Lambda_geo remains INCONCLUSIVE; both scientific slots stay OPEN.
"""
import hashlib
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import w2_accrual_instrument_cayley as ACC
import w2_acquisition_capture_grassmann as CAP

CAL_EPOCH_DAY = "2026-01-01"
MAG_LANE = "MAG_FEED"
WEATHER_LANE = "MAG_WEATHER_FEED"
WEATHER_CARRIERS = ("kp", "omni", "sym_h")
# the frozen target->reference map (mag1_instantiation.md; codex
# 1733Z D2 -- FRN residual regressed on TUC, VIC local with NEW,
# IZN without a second Marmara observatory)
M3_REFERENCE_MAP = {"FRN": "TUC", "VIC": "NEW",
                    "IZN": None, "TUC": None, "NEW": None}
MINUTES_PER_DAY = 1440
KP_INTERVAL_MIN = 180


class FeedProducerRefusal(ValueError):
    """Typed refusal; the code leads the message."""


def _refuse(code, detail):
    raise FeedProducerRefusal(f"{code}: {detail}")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _canon_digest(obj):
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def _read_json(path, what):
    if not os.path.isfile(path):
        _refuse("FEED_STAGED_ABSENT", f"{what}: {path}")
    with open(path, "rb") as f:
        raw = f.read()
    return json.loads(raw.decode("utf-8")), _sha(raw)


def _staged_path(repo, lane, carrier, day, cls):
    stem = f"{lane.lower()}_{carrier}_{day}"
    return os.path.join(
        repo, *ACC.STAGED_PREFIX.split("/"),
        stem + ACC.STAGED_CLASS_SUFFIX[cls])


def canonical_grid(day):
    """The canonical UTC minute grid the pinned cadence gates force
    on every admitted body -- constructed, never parsed."""
    return [f"{day}T{m // 60:02d}:{m % 60:02d}:00Z"
            for m in range(MINUTES_PER_DAY)]


def _load_registry(repo):
    """Authority + inventory + store descriptor, from their
    registered staged paths; per-carrier day lists and cutoffs come
    ONLY from the authority."""
    base = os.path.join(repo, *ACC.STAGED_PREFIX.split("/"))
    auth, auth_sha = _read_json(
        os.path.join(base, ACC.EXPECTED_KEYS_BASENAME), "authority")
    inv, inv_sha = _read_json(
        os.path.join(base, ACC.STAGED_INVENTORY_BASENAME),
        "inventory")
    desc, desc_sha = _read_json(
        os.path.join(base, ACC.STORE_DESCRIPTOR_BASENAME),
        "store descriptor")
    return (auth, inv, desc,
            {"authority_sha256": auth_sha,
             "inventory_sha256": inv_sha,
             "descriptor_sha256": desc_sha})


def _carrier_days_and_cutoff(auth, lane, carrier):
    days = sorted((auth["prestart_expected_keys"]
                   .get(lane, {}) or {}).get(carrier, ()))
    if not days:
        _refuse("FEED_AUTHORITY_CARRIER_ABSENT",
                f"{lane}/{carrier} has no registered days")
    cut = ((auth["static_layer"].get(lane, {}).get("carriers", {})
            .get(carrier, {})) or {}).get("cutoff")
    if not cut:
        _refuse("FEED_AUTHORITY_CUTOFF_ABSENT", f"{lane}/{carrier}")
    for d in days:
        if d < CAL_EPOCH_DAY or d > str(cut):
            _refuse("FEED_AFTER_CUTOFF",
                    f"{lane}/{carrier}/{d} outside "
                    f"[{CAL_EPOCH_DAY}, {cut}]")
    return days, str(cut)


def _read_body(repo, desc, inv, key, transcript):
    ent = (inv.get("objects") or {}).get(key)
    if not isinstance(ent, dict):
        _refuse("FEED_INVENTORY_KEY_ABSENT", key)
    base = os.path.realpath(str(desc["physical_root"]))
    pth = os.path.realpath(os.path.join(
        base, str(ent.get("path") or ent.get("sha256", "") + ".body")))
    if not pth.startswith(base + os.sep):
        _refuse("FEED_STORE_PATH_ESCAPE", key)
    if not os.path.isfile(pth):
        _refuse("FEED_STORE_BODY_ABSENT", f"{key}: {pth}")
    with open(pth, "rb") as f:
        body = f.read()
    got = _sha(body)
    want = transcript.get("raw_body_sha256")
    if got != want or (ent.get("sha256") and ent["sha256"] != got):
        _refuse("FEED_BODY_DIGEST_DIVERGENT",
                f"{key}: body {got[:12]} vs transcript "
                f"{str(want)[:12]} vs inventory "
                f"{str(ent.get('sha256'))[:12]}")
    return body


def _adjudicate(key, name, mine, theirs):
    """The admitted artifact adjudicates this producer's
    recomputation -- exact equality or the build refuses."""
    if mine != theirs:
        _refuse("FEED_ADJUDICATION_DIVERGENT",
                f"{key}: recomputed {name} diverges from the "
                f"admitted artifact ({str(mine)[:60]!r} != "
                f"{str(theirs)[:60]!r})")


# ---------------------------------------------------------------
# MAG value extraction (parse exactly as the pinned transform, then
# the ONE frame authority; the artifact adjudicates)
# ---------------------------------------------------------------

def _mag_day_values(key, body, s, artifact, capsule=None):
    doc = json.loads(CAP._xf_text(body))
    kind = (s.get("source") or {}).get("kind")
    if kind == "usgs-geomag-ws-minute":
        comps = {str(ch.get("id")): ch["values"]
                 for ch in doc["values"]}
        times = doc.get("times")
    elif kind == "intermagnet-gin-minute":
        comps = {k: v for k, v in doc.items()
                 if k != "datetime" and isinstance(v, list)}
        times = doc.get("datetime")
    else:
        _refuse("FEED_UNREGISTERED_MAG_KIND", f"{key}: {kind!r}")
    n = len(times)
    _adjudicate(key, "samples", n, artifact.get("samples"))
    nulls = {cid: sum(1 for v in comps[cid] if v is None)
             for cid in comps}
    _adjudicate(key, "null_by_channel",
                {k: nulls[k] for k in sorted(nulls)},
                artifact.get("null_by_channel"))
    if artifact.get("support_predicate") == \
            CAP.SUPPORT_STRUCTURAL_ALL_NULL:
        # registered absence: no conversion exists for this day
        _adjudicate(key, "support_mask", [False] * n,
                    artifact.get("support_mask"))
        return [None] * MINUTES_PER_DAY, [None] * MINUTES_PER_DAY
    ident, x_arr, y_arr = CAP._canonical_horizontal(
        comps, s, doc, key, capsule)
    mask = [bool(isinstance(x_arr[i], (int, float))
                 and isinstance(y_arr[i], (int, float))
                 and math.isfinite(x_arr[i])
                 and math.isfinite(y_arr[i])) for i in range(n)]
    _adjudicate(key, "canonical_frame", ident,
                artifact.get("canonical_frame"))
    _adjudicate(key, "support_mask", mask,
                artifact.get("support_mask"))
    _adjudicate(key, "definitive_samples", sum(mask),
                artifact.get("definitive_samples"))
    # the inclusive day-next terminal sample (1441st) belongs to the
    # NEXT day's grid; the canonical day contributes exactly 1440
    x = [x_arr[i] if mask[i] else None
         for i in range(MINUTES_PER_DAY)]
    y = [y_arr[i] if mask[i] else None
         for i in range(MINUTES_PER_DAY)]
    return x, y


# ---------------------------------------------------------------
# Weather value extraction (mirrors the pinned parsers row-for-row;
# every census the artifact carries is adjudicated)
# ---------------------------------------------------------------

def _omni_day_columns(key, body, s, artifact):
    """OMNIWeb listing -> per-variable minute columns (None at the
    registered fill), support conjunction -- the exact extraction
    the pinned transform performs, adjudicated against it."""
    rp = dict(s.get("request_params") or {})
    v = rp.get("vars")
    var_list = ([str(x) for x in v]
                if isinstance(v, (list, tuple)) else [str(v)])
    _adjudicate(key, "registered_vars", var_list,
                artifact.get("registered_vars"))
    fills = [CAP._OMNIWEB_VAR_FILL[x] for x in var_list]
    text = CAP._xf_text(body)
    rows, in_data = [], False
    for ln in text.splitlines():
        t = ln.strip()
        if t.startswith("YYYY DOY HR MN"):
            in_data = True
            continue
        if not in_data:
            continue
        if not t or t.startswith("<"):
            in_data = False
            continue
        tok = t.split()
        if len(tok) != 4 + len(var_list):
            _refuse("FEED_OMNI_ROW_MALFORMED", f"{key}: {t[:50]!r}")
        i = len(rows)
        if (int(tok[2]), int(tok[3])) != (i // 60, i % 60):
            _refuse("FEED_OMNI_CADENCE_VIOLATION",
                    f"{key}: row {tok[2]}:{tok[3]} at index {i}")
        rows.append(tok[4:])
    if len(rows) != MINUTES_PER_DAY:
        _refuse("FEED_OMNI_GRID_INCOMPLETE",
                f"{key}: {len(rows)} rows")
    cols = [[] for _ in var_list]
    fill_by_col = [0] * len(var_list)
    support = []
    for vals in rows:
        ok = True
        for i, x in enumerate(vals):
            if x == fills[i]:
                fill_by_col[i] += 1
                cols[i].append(None)
                ok = False
            else:
                cols[i].append(float(x))
        support.append(ok)
    _adjudicate(key, "fill_by_column", fill_by_col,
                artifact.get("fill_by_column"))
    _adjudicate(key, "support_mask", support,
                artifact.get("support_mask"))
    _adjudicate(key, "samples", len(rows), artifact.get("samples"))
    return cols, support


def _kp_day_minutes(key, body, artifact):
    """GFZ Kp day -> 1440 minute values by half-open three-hour ZOH
    (D1: each value expands ONLY over its own interval; an
    unsupported interval yields None minutes)."""
    doc = json.loads(CAP._xf_text(body))
    kp, st = doc.get("Kp"), doc.get("status")
    if not isinstance(kp, list) or len(kp) != 8:
        _refuse("FEED_KP_INTERVALS_MALFORMED",
                f"{key}: {type(kp).__name__}")
    _adjudicate(key, "intervals", len(kp), artifact.get("intervals"))
    _adjudicate(key, "status_counts",
                {v: st.count(v) for v in CAP.KP_STATUS_VOCAB
                 if v in st},
                artifact.get("status_counts"))
    mask = artifact.get("support_mask")
    if not isinstance(mask, list) or len(mask) != 8:
        _refuse("FEED_KP_SUPPORT_MALFORMED", key)
    out = []
    for m in range(MINUTES_PER_DAY):
        iv = m // KP_INTERVAL_MIN
        out.append(float(kp[iv]) if mask[iv] else None)
    return out


def _newell_day_minutes(key, artifact):
    """The admitted OMNI artifact CARRIES the pinned Newell join --
    consumed directly, never recomputed here."""
    blk = artifact.get("newell")
    nw = (blk or {}).get("values") if isinstance(blk, dict) else None
    if not isinstance(nw, list) or \
            len(nw) != artifact.get("samples"):
        _refuse("FEED_NEWELL_ABSENT",
                f"{key}: admitted artifact carries no coherent "
                "newell series")
    if blk.get("supported") != sum(
            1 for v in nw if v is not None):
        _refuse("FEED_NEWELL_SUPPORT_DIVERGENT",
                f"{key}: supported count vs values")
    mask = artifact.get("support_mask")
    for i, v in enumerate(nw):
        if (v is None) != (not mask[i]):
            _refuse("FEED_NEWELL_SUPPORT_DIVERGENT",
                    f"{key}: newell[{i}] vs support mask")
    return list(nw)


# ---------------------------------------------------------------
# assembly
# ---------------------------------------------------------------

def _obs_capsule_lon(repo, carrier):
    """lon_east from the committed frame capsule (execution tree
    first, freeze tree for capsules that live only there)."""
    for rel in (f"docs/f2g_window2_execution/mag_capsules/"
                f"mag_capsule_{carrier}.json",
                f"docs/f2g_window2_freeze/"
                f"mag_capsule_{carrier}.json"):
        p = os.path.join(repo, *rel.split("/"))
        if os.path.isfile(p):
            cap, cap_sha = _read_json(p, f"capsule {carrier}")
            coords = cap.get("coordinates_lat_lon")
            if not (isinstance(coords, (list, tuple))
                    and len(coords) == 2):
                _refuse("FEED_CAPSULE_COORDS_MALFORMED", carrier)
            return float(coords[1]), rel, cap_sha
    _refuse("FEED_CAPSULE_ABSENT", carrier)


def validate_m3_map(feeds):
    """D2 lock: every m3_reference must be EXACTLY the frozen map's
    entry -- inverse pairs, self-references, unknown stations and
    novel pairs all refuse."""
    for obs, feed in feeds.items():
        if obs not in M3_REFERENCE_MAP:
            _refuse("M3_PAIR_UNREGISTERED",
                    f"{obs} is not a frozen-map station")
        want = M3_REFERENCE_MAP[obs]
        got = feed.get("m3_reference")
        if got != want:
            _refuse("M3_PAIR_UNREGISTERED",
                    f"{obs} -> {got!r}; the frozen map registers "
                    f"{want!r}")
        if got is not None:
            if got == obs:
                _refuse("M3_PAIR_UNREGISTERED",
                        f"{obs} self-reference")
            if got not in feeds:
                _refuse("M3_REFERENCE_ABSENT",
                        f"{obs} -> {got} not among the built feeds")


def build_mag_feeds(repo, weather_days_required=True):
    """The complete MAG feed set for run_mag_calibration, derived
    from committed staged bytes. Returns (feeds, provenance)."""
    auth, inv, desc, prov = _load_registry(repo)
    # one shared day list: every MAG and weather carrier must
    # register EXACTLY the same days (the frozen single interval)
    day_sets = {}
    cutoffs = set()
    for lane, carrier in ([(MAG_LANE, c) for c in sorted(
            auth["prestart_expected_keys"].get(MAG_LANE, {}))]
            + [(WEATHER_LANE, c) for c in WEATHER_CARRIERS]):
        days, cut = _carrier_days_and_cutoff(auth, lane, carrier)
        day_sets[f"{lane}/{carrier}"] = days
        cutoffs.add(cut)
    ref_days = None
    for k, days in sorted(day_sets.items()):
        if ref_days is None:
            ref_days = days
        elif days != ref_days:
            _refuse("FEED_DAY_SET_DIVERGENT",
                    f"{k} registers a different day list")
    if len(cutoffs) != 1:
        _refuse("FEED_CUTOFF_DIVERGENT", str(sorted(cutoffs)))
    cutoff = cutoffs.pop()

    def staged(lane, carrier, day):
        art, a_sha = _read_json(
            _staged_path(repo, lane, carrier, day, "artifact"),
            f"{lane}/{carrier}/{day} artifact")
        tr, t_sha = _read_json(
            _staged_path(repo, lane, carrier, day, "transcript"),
            f"{lane}/{carrier}/{day} transcript")
        return art, tr, (a_sha, t_sha)

    roll = hashlib.sha256()
    # ---- weather minute columns over the whole interval ----------
    weather = {"sym_h": [], "kp": [], "newell": []}
    for day in ref_days:
        for carrier in WEATHER_CARRIERS:
            key = f"{WEATHER_LANE}/{carrier}/{day}"
            art, tr, shas = staged(WEATHER_LANE, carrier, day)
            s = ACC.authoritative_static_contract(
                auth, WEATHER_LANE, carrier, day)
            body = _read_body(repo, desc, inv, key, tr)
            roll.update((key + shas[0] + shas[1]).encode())
            if carrier == "kp":
                weather["kp"].extend(
                    _kp_day_minutes(key, body, art))
            elif carrier == "sym_h":
                cols, support = _omni_day_columns(key, body, s, art)
                weather["sym_h"].extend(
                    cols[0][:MINUTES_PER_DAY])
            else:
                # adjudicate the raw columns, then consume the
                # artifact's own pinned Newell join
                _omni_day_columns(key, body, s, art)
                weather["newell"].extend(
                    _newell_day_minutes(key, art)
                    [:MINUTES_PER_DAY])
    n_minutes = len(ref_days) * MINUTES_PER_DAY
    for name, series in weather.items():
        if len(series) != n_minutes:
            _refuse("FEED_WEATHER_LENGTH_DIVERGENT",
                    f"{name}: {len(series)} != {n_minutes}")

    # ---- per-observatory feeds -----------------------------------
    times = []
    for day in ref_days:
        times.extend(canonical_grid(day))
    feeds = {}
    for carrier in sorted(
            auth["prestart_expected_keys"].get(MAG_LANE, {})):
        s0 = ACC.authoritative_static_contract(
            auth, MAG_LANE, carrier, ref_days[0])
        rp = dict(s0.get("request_params") or {})
        obs = str(rp.get("id") or rp.get("observatoryIagaCode")
                  or carrier.upper())
        lon_east, cap_rel, cap_sha = _obs_capsule_lon(repo, carrier)
        xs, ys = [], []
        for day in ref_days:
            key = f"{MAG_LANE}/{carrier}/{day}"
            art, tr, shas = staged(MAG_LANE, carrier, day)
            s = ACC.authoritative_static_contract(
                auth, MAG_LANE, carrier, day)
            body = _read_body(repo, desc, inv, key, tr)
            roll.update((key + shas[0] + shas[1]).encode())
            x, y = _mag_day_values(key, body, s, art)
            xs.extend(x)
            ys.extend(y)
        feeds[obs] = {"observatory": obs, "lon_east": lon_east,
                      "times": list(times),
                      "components": {"X": xs, "Y": ys},
                      "weather": {k: list(v)
                                  for k, v in weather.items()},
                      "m3_reference": M3_REFERENCE_MAP.get(obs)}
    validate_m3_map(feeds)
    prov.update({"cutoff": cutoff, "days": len(ref_days),
                 "minutes": n_minutes,
                 "staged_rolling_sha256": roll.hexdigest(),
                 "producer_source_sha256_normalized":
                     _self_norm_sha()})
    return feeds, prov


def _self_norm_sha():
    with open(os.path.abspath(__file__), "rb") as f:
        return _sha(f.read().replace(b"\r\n", b"\n"))


# ---------------------------------------------------------------
# MF4 amended-lane calibration inputs (codex 1758Z option 1 +
# 0317Z bytes-only boundary): everything derives from grassmann's
# REGISTERED archive capsule and the committed snapshot/receipt
# BYTES -- this producer verifies bindings independently, assembles
# risk_by_region from the committed rows, and hands the ORIGINAL
# bytes onward; the parsed snapshot never exists in this module.
# ---------------------------------------------------------------

def build_mf4_calibration_inputs(repo):
    """Returns the amended-lane calibration inputs + provenance.

    Verification layers, in order: (1) grassmann's full
    verify_capsule() recompute (module authority); (2) this
    module's OWN recompute of every surface it consumes -- rows
    digest, support census adjudication, region partition,
    catalog-binding byte digests; (3) the authenticated adapter
    stack re-proves the git trust anchor at calibrate time. Zero
    HTTP; zero fit; zero writes."""
    import w2_mf4_archive_capsule_gen_grassmann as ARCH
    if os.path.realpath(str(ARCH.REPO)) != os.path.realpath(
            str(repo)):
        _refuse("MF4_INPUTS_REPO_MISMATCH",
                f"module repo {ARCH.REPO} != requested {repo} -- "
                "the amended lane runs inside one checkout")
    ARCH.verify_capsule()
    cap_raw = open(os.path.join(
        repo, *ARCH.CAPSULE_REL.split("/")), "rb").read()
    capsule = json.loads(cap_raw.decode("utf-8"))

    # rows: independent digest + parse + census adjudication
    rows_rel = capsule["rows_file"]["path"]
    rows_raw = open(os.path.join(
        repo, *rows_rel.split("/")), "rb").read()
    if _sha(rows_raw) != capsule["rows_file"]["sha256"] or             len(rows_raw) != capsule["rows_file"]["bytes"]:
        _refuse("MF4_INPUTS_ROWS_DIGEST_DIVERGENT", rows_rel)
    admitted = list(capsule["region_sets"]["admitted_regions"])
    monitor = set(capsule["region_sets"]
                  ["monitor_region_set_at_freeze"])
    interval = capsule["maturity_bounds"]["calibration_interval"]
    risk_by_region = {r: {} for r in admitted}
    census = {r: 0 for r in admitted}
    seen = set()
    n_rows = 0
    for line in rows_raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        n_rows += 1
        row = json.loads(line)
        reg, day = row["region"], row["issue_day"]
        if reg not in monitor:
            _refuse("MF4_INPUTS_ROW_REGION_UNREGISTERED",
                    f"{reg}/{day}")
        if not (interval[0] <= day <= interval[1]):
            _refuse("MF4_INPUTS_ROW_OUTSIDE_INTERVAL",
                    f"{reg}/{day}")
        if (reg, day) in seen:
            _refuse("MF4_INPUTS_ROW_DUPLICATE", f"{reg}/{day}")
        seen.add((reg, day))
        if row["support"] == "SUPPORTED":
            risk = row["combined_risk"]
            if not isinstance(risk, (int, float)) or                     isinstance(risk, bool) or                     not math.isfinite(risk):
                _refuse("MF4_INPUTS_ROW_RISK_INVALID",
                        f"{reg}/{day}: {risk!r}")
            if reg in risk_by_region:
                risk_by_region[reg][day] = float(risk)
                census[reg] += 1
        # non-SUPPORTED rows carry the registered missingness state:
        # absent from the fit input, never imputed
    if n_rows != capsule["rows_file"]["rows"]:
        _refuse("MF4_INPUTS_ROW_COUNT_DIVERGENT",
                f"{n_rows} != {capsule['rows_file']['rows']}")
    # codex 0411Z item 2: the REGISTERED census entry schema is the
    # closed {days_supported, days_total, days_unsupported} triple --
    # consumed under its exact keyset, never a guessed field name
    for r in admitted:
        ent = (capsule.get("support_census") or {}).get(r)
        if not isinstance(ent, dict) or set(ent) != {
                "days_supported", "days_total", "days_unsupported"}:
            _refuse("MF4_INPUTS_CENSUS_SCHEMA",
                    f"{r}: {sorted(ent) if isinstance(ent, dict) else ent!r}")
        if ent["days_supported"] + ent["days_unsupported"] !=                 ent["days_total"]:
            _refuse("MF4_INPUTS_CENSUS_SCHEMA",
                    f"{r}: supported+unsupported != total")
    cap_census = {
        r: int(capsule["support_census"][r]["days_supported"])
        for r in admitted}
    if census != cap_census:
        _refuse("MF4_INPUTS_CENSUS_DIVERGENT",
                f"recomputed {census} != capsule {cap_census}")

    # catalog snapshot/receipt: ORIGINAL BYTES, digest-bound to the
    # capsule's catalog_binding; the adapter's trust anchor proves
    # the git chain itself
    cb = capsule["catalog_binding"]
    snap_raw = open(os.path.join(
        repo, *cb["snapshot_path"].split("/")), "rb").read()
    rcpt_raw = open(os.path.join(
        repo, *cb["receipt_path"].split("/")), "rb").read()
    if _sha(snap_raw) != cb["snapshot_sha256"]:
        _refuse("MF4_INPUTS_SNAPSHOT_DIGEST_DIVERGENT",
                cb["snapshot_path"])
    if _sha(rcpt_raw) != cb["acquisition_receipt_sha256"]:
        _refuse("MF4_INPUTS_RECEIPT_DIGEST_DIVERGENT",
                cb["receipt_path"])

    # codex 0411Z item 2: the capsule bbox is the wrapper
    # {bbox, carrier}; the frozen engine consumes the INNER exact
    # {min_lat, max_lat, min_lon, max_lon} map -- both keysets
    # closed before any adapter call
    bboxes = {}
    for r in admitted:
        w = capsule["bboxes"].get(r)
        if not isinstance(w, dict) or set(w) != {"bbox", "carrier"}:
            _refuse("MF4_INPUTS_BBOX_SCHEMA",
                    f"{r}: wrapper {sorted(w) if isinstance(w, dict) else w!r}")
        inner = w["bbox"]
        if not isinstance(inner, dict) or set(inner) != {
                "min_lat", "max_lat", "min_lon", "max_lon"}:
            _refuse("MF4_INPUTS_BBOX_SCHEMA",
                    f"{r}: inner {sorted(inner) if isinstance(inner, dict) else inner!r}")
        bboxes[r] = dict(inner)
    return {"risk_by_region": risk_by_region,
            "snapshot_bytes": snap_raw,
            "receipt_bytes": rcpt_raw,
            "bboxes": bboxes,
            "regions": admitted,
            "freeze_day": capsule["maturity_bounds"]["freeze_day"],
            "snapshot_end":
                capsule["maturity_bounds"]["snapshot_end"],
            "requested_issue_end": interval[1],
            "provenance": {
                "capsule_sha256": _sha(cap_raw),
                "rows_sha256": capsule["rows_file"]["sha256"],
                "snapshot_sha256": cb["snapshot_sha256"],
                "acquisition_receipt_sha256":
                    cb["acquisition_receipt_sha256"],
                "result_commit": cb["result_commit"],
                "supported_cells": sum(census.values()),
                "producer_source_sha256_normalized":
                    _self_norm_sha()}}


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        from w2_calibration_feed_producer_kats_cayley import main
        raise SystemExit(main())
    raise SystemExit(
        "usage: --selftest (production invocation is "
        "build_mag_feeds(repo) from the runner orchestration; this "
        "module never runs standalone against evidence)")
