"""thd_bound_station_operator.py -- the reviewed SNZO processing operator, shared by the DAILY THD path and the WEEKLY
recalibration (codex review db9a28ff finding 3; grassmann 2026-10-04).

For a BOUND station (today only IU.SNZO) the waveform fetch is location- and channel-bound (no '*' wildcard), traces
are joined only where they abut exactly (or repeat an identical sample), gaps/overlaps/masked samples REFUSE by name
(no interpolation fill), the native rate is checked, and the data is demeaned + linearly detrended exactly as the
production fetch does before the estimator. The operator identity (channel, location, response policy, native rate,
gap/coverage policy, estimator, normalization) is a SHA256 that the weekly recal writes into the station's baseline
record so a later reader can tell which operator produced it. Every other station keeps today's behaviour.

Entry point for both paths: `fetch_bound(network, station, start, end, channel)` returns (data, rate, record) or
(None, 0.0, record) with record['refusal'] naming the reason. `seismic_thd.fetch_continuous_data_for_thd` dispatches
here when (network, station) is bound; `run_thd_recal.run_recal` stamps `operator_record(key)` into the baseline entry.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# gap_policy is part of the operator identity, so its name stays. What "WITHIN_ONE_RETAINED_EPOCH" implements is narrower
# than the name: an identical overlap with a KNOWN retained response-epoch boundary inside it refuses
# (CROSSES_RESPONSE_EPOCH). It is not a proof that the whole window lies in one epoch, and with no retained epochs for the
# selector no epoch rule applies (codex 51e62a83 s.3).
BOUND_STATIONS: Dict[str, Dict] = {
    "IU.SNZO": dict(location="00", channel="BHZ", expected_rate_hz=40.0, client="IRIS",
                    response_policy="NOT_REMOVED",
                    gap_policy="EXACT_ABUT_OR_EXACT_IDENTICAL_OVERLAP_ON_GRID_WITHIN_ONE_RETAINED_EPOCH_NO_FILL",
                    coverage_policy="REQUESTED_WINDOW_FULLY_COVERED_OR_REFUSE",
                    detrend="demean+linear", estimator="SeismicTHDAnalyzer(n_harmonics=5,freq_tolerance=0.1,window_hours=24).compute_thd",
                    normalization="baseline mean=median(daily THD), std=MAD*1.4826 (calibrate_station)",
                    reviewed_by="grassmann thd_bootstrap v2 eae5ed77 / codex measurement-support review 2026-10-03",
                    operator_version="thd-bound-station-operator-v2"),
}
REFUSALS = ("NO_TRACES", "NSLC_MISMATCH", "LOCATION_MISMATCH", "CHANNEL_MISMATCH", "RATE_INVALID", "RATE_MISMATCH",
            "MASKED_SAMPLES", "GAP_OR_OVERLAP", "WINDOW_NOT_COVERED", "PROVIDER_ERROR")


def is_bound(network: str, station: str) -> bool:
    return f"{network}.{station}" in BOUND_STATIONS


def operator_identity(key: str) -> str:
    spec = BOUND_STATIONS[key]
    payload = dict(station=key, **{k: spec[k] for k in sorted(spec) if k != "reviewed_by"})
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def operator_record(key: str) -> Dict:
    """The block the weekly recal writes into the baseline entry of a bound station."""
    spec = dict(BOUND_STATIONS[key])
    return dict(identity=operator_identity(key), **spec)


def _iso(x) -> str:
    return x.isoformat() if hasattr(x, "isoformat") else str(x)


def refusal_summary(rec: Dict) -> str:
    """The refusal with its measured facts on ONE line (codex b7a29458 correction 2: carry the requested window, the
    covered fraction, the gap/overlap seconds and the outcome into refusal rows): the code, the refusing boundary's kind
    and facts, the covered seconds and fraction of an uncovered window, and the requested window. The D3 attempt row
    carries this as its reason text, so the served row says what was seen, not only that it refused."""
    parts = [str(rec.get("refusal") or "UNKNOWN_REFUSAL")]
    d = rec.get("discontinuity") or {}
    if d:
        parts.append(" ".join([str(d.get("kind"))] + ["%s=%s" % (k, d[k]) for k in sorted(d) if k != "kind"]))
    c = rec.get("coverage") or {}
    if c:
        parts.append("covered %s of %s s (fraction %s; missing head %s s, tail %s s)" % (
            c.get("covered_seconds"), c.get("requested_seconds"), c.get("fraction"), c.get("missing_head_seconds"),
            c.get("missing_tail_seconds")))
    if rec.get("requested"):
        parts.append("window [%s, %s)" % tuple(rec["requested"]))
    return "; ".join(parts)


def _utc(x) -> datetime:
    return x.datetime.replace(tzinfo=timezone.utc)


def _discontinuity(kind: str, rec: Dict, **facts) -> None:
    """Name the refusing boundary in the record: the kind and the measured facts, nothing inferred."""
    rec["refusal"] = "GAP_OR_OVERLAP" if kind in ("GAP", "UNEQUAL_OVERLAP", "OFF_GRID", "CROSSES_RESPONSE_EPOCH") else rec["refusal"]
    rec["discontinuity"] = dict(kind=kind, **facts)
    if "at" in facts:
        rec["at"] = facts["at"]
    if "delta_seconds" in facts:
        rec["delta_seconds"] = facts["delta_seconds"]


def _epoch_boundaries(response_epochs):
    """The retained response-epoch boundaries as integer nanoseconds, from (start, end, label) ISO instants."""
    if not response_epochs:
        return []
    from obspy import UTCDateTime
    bounds = set()
    for begin, finish, _label in response_epochs:
        for value in (begin, finish):
            if value is not None:
                bounds.add(int(UTCDateTime(value).ns))
    return sorted(bounds)


def stitch_window(traces, *, network: str, station: str, location: str, channel: str, expected_rate: Optional[float],
                  start: datetime, end: datetime, response_epochs=None) -> Tuple[Optional[np.ndarray], float, Dict]:
    """Join obspy traces into one array covering [start, end) under the reviewed rules. Pure: no I/O.

    thd-bound-station-operator-v2 (cayley 2026-10-10, codex b7a29458): records are placed on ONE native sample grid
    anchored at the earliest record. A record joins when it starts exactly at the end of the assembled support (abut),
    or when it overlaps the support and EVERY overlapping sample equals the support's sample at that grid index -- a
    nested or fully duplicate record contributes nothing and never shrinks the support's end; a partially overlapping
    one contributes only its samples past the end. Refused by name, with the measured facts in `discontinuity`: a GAP
    (no fill, ever), an UNEQUAL_OVERLAP (any differing sample), an OFF_GRID record (start more than a quarter sample
    off the grid), an identical overlap that CROSSES_RESPONSE_EPOCH (a retained epoch boundary inside it, when
    `response_epochs` is given), MASKED_SAMPLES (masked or nonfinite). Everything v1 accepted (exact abut, one
    identical duplicate sample) is joined exactly as v1 joined it, so its output support is byte-identical."""
    rec: Dict = dict(n_traces=len(traces), refusal=None, requested=[_iso(start), _iso(end)],
                     identical_overlap_samples=0, identical_overlaps=0)
    pieces = []
    for tr in traces:
        if (tr.stats.network, tr.stats.station) != (network, station):
            rec["refusal"] = "NSLC_MISMATCH"; return None, 0.0, rec
        if tr.stats.location != location:
            rec["refusal"] = "LOCATION_MISMATCH"; return None, 0.0, rec
        if tr.stats.channel != channel:
            rec["refusal"] = "CHANNEL_MISMATCH"; return None, 0.0, rec
        r = float(tr.stats.sampling_rate)
        if not np.isfinite(r) or r <= 0:
            rec["refusal"] = "RATE_INVALID"; return None, 0.0, rec
        pieces.append(tr)
    if not pieces:
        rec["refusal"] = "NO_TRACES"; return None, 0.0, rec
    rate = float(pieces[0].stats.sampling_rate)
    if any(abs(float(tr.stats.sampling_rate) - rate) > 1e-6 for tr in pieces):
        rec["refusal"] = "RATE_MISMATCH"; return None, 0.0, rec
    if expected_rate is not None and abs(rate - expected_rate) > 1e-6:
        rec["refusal"] = "RATE_MISMATCH"; rec["rate"] = rate; return None, 0.0, rec
    dt = 1.0 / rate
    for tr in pieces:
        if np.ma.isMaskedArray(tr.data) and np.ma.count_masked(tr.data) > 0:
            rec["refusal"] = "MASKED_SAMPLES"; rec["discontinuity"] = dict(kind="MASKED", record_start=str(tr.stats.starttime))
            return None, 0.0, rec
        values = np.asarray(tr.data)
        if values.dtype.kind in "fc" and not np.all(np.isfinite(values)):
            rec["refusal"] = "MASKED_SAMPLES"; rec["discontinuity"] = dict(kind="NONFINITE", record_start=str(tr.stats.starttime))
            return None, 0.0, rec
    # earliest first; for one start the longest first, so a shorter record at the same start is a nested duplicate
    pieces = sorted(pieces, key=lambda tr: (tr.stats.starttime, -tr.stats.npts))
    boundaries = _epoch_boundaries(response_epochs)
    t0 = pieces[0].stats.starttime
    support = [np.asarray(pieces[0].data)]
    n = int(pieces[0].stats.npts)
    flat = None                                                          # the joined support, built lazily for compares
    for tr in pieces[1:]:
        data = np.asarray(tr.data)
        offset = float(tr.stats.starttime - t0) * rate                   # in samples from the grid anchor
        k0 = int(round(offset))
        end_time = t0 + (n - 1) * dt                                     # last assembled sample
        if abs(offset - k0) > 0.25:
            _discontinuity("OFF_GRID", rec, at=str(tr.stats.starttime), offset_samples=round(offset - k0, 6),
                           delta_seconds=float(tr.stats.starttime - end_time))
            return None, 0.0, rec
        if k0 > n:
            _discontinuity("GAP", rec, at=str(end_time), delta_seconds=float(tr.stats.starttime - end_time),
                           gap_seconds=round((k0 - n) * dt, 6), gap_samples=k0 - n)
            return None, 0.0, rec
        overlap = min(n - k0, int(data.size))
        if overlap > 0:
            if flat is None or flat.size != n:
                flat = np.concatenate(support) if len(support) > 1 else support[0]
            mine, theirs = data[:overlap], flat[k0:k0 + overlap]
            if not np.array_equal(mine, theirs):
                differ = np.flatnonzero(mine != theirs)
                _discontinuity("UNEQUAL_OVERLAP", rec, at=str(end_time), delta_seconds=float(tr.stats.starttime - end_time),
                               overlap_samples=overlap, overlap_seconds=round(overlap * dt, 6),
                               differing_samples=int(differ.size), first_difference_index=int(differ[0]))
                return None, 0.0, rec
            lo, hi = int((t0 + k0 * dt).ns), int((t0 + (k0 + overlap - 1) * dt).ns)
            crossing = [b for b in boundaries if lo < b <= hi]
            if crossing:
                from obspy import UTCDateTime
                _discontinuity("CROSSES_RESPONSE_EPOCH", rec, at=str(UTCDateTime(ns=crossing[0])),
                               delta_seconds=float(tr.stats.starttime - end_time), overlap_samples=overlap)
                return None, 0.0, rec
            rec["identical_overlap_samples"] += overlap
            rec["identical_overlaps"] += 1
        remainder = data[overlap:]
        if remainder.size:
            support.append(remainder)
            n += int(remainder.size)
    data = np.concatenate(support) if len(support) > 1 else support[0]
    prev_end = t0 + (n - 1) * dt
    # select exactly [start, end) on the native grid and require full coverage
    from obspy import UTCDateTime
    s0 = UTCDateTime(start); e0 = UTCDateTime(end)
    i0 = int(np.ceil(float(s0 - t0) * rate - 1e-6))
    i1 = int(np.floor(float(e0 - t0) * rate - 1e-6)) + 1
    if i0 < 0 or i1 > data.size or i1 <= i0:
        have_lo, have_hi = max(0, i0), min(int(data.size), i1)
        requested_samples = max(1, i1 - i0)
        rec["refusal"] = "WINDOW_NOT_COVERED"; rec["have"] = [str(t0), str(prev_end)]
        rec["coverage"] = dict(requested_seconds=round(float(e0 - s0), 6),
                               covered_seconds=round(max(0, have_hi - have_lo) * dt, 6),
                               fraction=round(max(0, have_hi - have_lo) / requested_samples, 6),
                               missing_head_seconds=round(max(0, -i0) * dt, 6),
                               missing_tail_seconds=round(max(0, i1 - int(data.size)) * dt, 6))
        return None, 0.0, rec
    sel = data[i0:i1].astype(np.float64)
    rec.update(rate=rate, npts=int(sel.size), start=str(t0 + i0 * dt), end=str(t0 + (i1 - 1) * dt),
               support_sha256=hashlib.sha256(np.ascontiguousarray(data[i0:i1].astype(np.int32)).tobytes()).hexdigest())
    return sel, rate, rec


def _detrend(x: np.ndarray) -> np.ndarray:
    """Same two steps as the production fetch (obspy detrend('demean') then detrend('linear'))."""
    x = x - np.mean(x)
    n = x.size
    t = np.arange(n, dtype=np.float64)
    a, b = np.polyfit(t, x, 1)
    return x - (a * t + b)


def _retained_epochs(trace_id: str):
    """The retained level=response epochs for the exact selector (thd_provider_routing.RESPONSE_METADATA), or None when
    none are retained -- then no epoch rule applies, and that absence is what the record says."""
    try:
        import thd_provider_routing as TPR
    except ImportError:
        return None
    entry = TPR.RESPONSE_METADATA.get(trace_id)
    return list(entry["epochs"]) if entry else None


def fetch_bound(network: str, station: str, start: datetime, end: datetime, channel: Optional[str] = None,
                *, client_factory=None) -> Tuple[Optional[np.ndarray], float, Dict]:
    """Location/channel-bound FDSN fetch for a bound station over [start, end), stitched under the reviewed rules and
    demeaned+linearly detrended. `client_factory(name) -> client` is injectable for tests; production uses obspy."""
    key = f"{network}.{station}"
    spec = BOUND_STATIONS[key]
    chan = channel or spec["channel"]
    rec: Dict = dict(station=key, operator_identity=operator_identity(key), location=spec["location"], channel=chan,
                     client=spec["client"], requested=[start.isoformat(), end.isoformat()], refusal=None)
    if chan != spec["channel"]:
        rec["refusal"] = "CHANNEL_MISMATCH"; return None, 0.0, rec
    try:
        from obspy import UTCDateTime
        if client_factory is None:
            from obspy.clients.fdsn import Client
            client = Client(spec["client"], timeout=120)
        else:
            client = client_factory(spec["client"])
        st = client.get_waveforms(network=network, station=station, location=spec["location"], channel=chan,
                                  starttime=UTCDateTime(start), endtime=UTCDateTime(end))
    except Exception as exc:
        rec["refusal"] = "PROVIDER_ERROR"; rec["error"] = type(exc).__name__
        logger.warning(f"bound fetch {key}: PROVIDER_ERROR {type(exc).__name__}")
        return None, 0.0, rec
    data, rate, srec = stitch_window(list(st), network=network, station=station, location=spec["location"], channel=chan,
                                     expected_rate=spec["expected_rate_hz"], start=start, end=end,
                                     response_epochs=_retained_epochs("%s.%s.%s" % (key, spec["location"], chan)))
    rec.update(srec)
    if data is None:
        logger.warning(f"bound fetch {key}: refused {rec['refusal']}")
        return None, 0.0, rec
    return _detrend(data), rate, rec


# --------------------------------------------------------------------------------------------------- daily operator
TARGET_THD_RATE = 1.0        # Hz, the production daily estimator rate (ensemble.compute_thd_risk)
DAILY_OPERATOR = dict(name="daily_ensemble", window="[target - (window_hours + 1) h, target]",
                      native_rate_hz=40.0, estimator_rate_hz=TARGET_THD_RATE,
                      resampling="scipy.signal.resample_poly(up, down) with gcd-reduced integer factors when native > 1.5x target",
                      preprocessing="demean + linear detrend (bound fetch)", minimum_samples="12 h at the native rate",
                      estimator="SeismicTHDAnalyzer(n_harmonics=5, freq_tolerance=0.1, window_hours=24).analyze_window -> compute_thd_with_noise",
                      operator_version="thd-bound-station-daily-operator-v1")


def daily_operator_identity(key: str) -> str:
    payload = dict(station=key, fetch=operator_identity(key), **DAILY_OPERATOR)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def daily_operator_record(key: str) -> Dict:
    return dict(identity=daily_operator_identity(key), fetch=operator_record(key), **DAILY_OPERATOR)


def daily_measurement(network: str, station: str, target: datetime, *, analyzer, fetch, channel: Optional[str] = None) -> Dict:
    """The PRODUCTION daily THD measurement for a bound station, as one function used by ensemble.compute_thd_risk and
    by calibrate_thd_baselines.compute_daily_thd: requested window = [target - (analyzer.window_hours + 1) h, target],
    `fetch` = the caller's fetch_continuous_data_for_thd (which dispatches bound stations to fetch_bound), the daily
    12-hour floor, resample_poly to TARGET_THD_RATE, then analyzer.analyze_window. Returns a record with thd (None when
    unavailable), the requested window, native and estimator rates, processed sample count and the operator identity."""
    from datetime import timedelta
    key = f"{network}.{station}"
    start = target - timedelta(hours=analyzer.window_hours + 1)
    rec: Dict = dict(station=key, operator_identity=daily_operator_identity(key), requested_window=[start.isoformat(), target.isoformat()],
                     thd=None, native_rate_hz=None, estimator_rate_hz=None, n_native_samples=None, n_processed_samples=None, reason=None)
    # `channel` (grassmann e5a34e44): passed through when the caller names one, so fetch_bound refuses CHANNEL_MISMATCH
    # for anything but the bound channel; a caller that names none keeps the unchanged call shape
    data, sample_rate = fetch(station_network=network, station_code=station, start=start, end=target,
                              **({} if channel is None else {"channel": channel}))
    if data is None or len(data) < sample_rate * 3600 * 12:
        rec["reason"] = "INSUFFICIENT_DATA"; return rec
    rec["native_rate_hz"] = float(sample_rate); rec["n_native_samples"] = int(len(data))
    if sample_rate > TARGET_THD_RATE * 1.5:
        from scipy.signal import resample_poly
        from math import gcd
        up = int(TARGET_THD_RATE * 100); down = int(sample_rate * 100)
        common = gcd(up, down); up //= common; down //= common
        data = resample_poly(data, up, down); sample_rate = TARGET_THD_RATE
    rec["estimator_rate_hz"] = float(sample_rate); rec["n_processed_samples"] = int(len(data))
    result = analyzer.analyze_window(data=data, sample_rate=sample_rate, station=key, window_time=target)
    rec.update(thd=float(result.thd_value), p1=float(result.fundamental_power), f1=float(result.dominant_frequency),
               snr=(float(result.snr) if result.snr is not None else None), result=result)
    return rec
