"""thd_daily_measurement.py -- ONE versioned daily THD measurement for every station, called by the daily scorer
(ensemble.compute_thd_risk) and by the weekly recalibration (calibrate_thd_baselines.compute_daily_thd), so a baseline
is built from the same quantity the daily path compares against it (codex 1614 finding 1; cayley 2026-10-08).
PROPOSAL on a local branch: nothing here is active until a reviewed method change carries it.

Why: for every UNBOUND station the two paths measured different things under one name. Calibration requested
[D, D + 25 h] and ran compute_thd at the NATIVE rate; the daily path requested [D - 25 h, D], resampled to 1 Hz and ran
analyze_window. (The bound station IU.SNZO was already shared: thd_bound_station_operator.daily_measurement.)

`measure` is that daily path, unchanged, for every station: the window [target - (window_hours + 1) h, target]; the
caller's fetch (seismic_thd.fetch_continuous_data_for_thd: the routed, selector-bound request, merge(method=1,
fill_value='interpolate'), demean + linear detrend, response NOT removed); the 12-hour floor at the native rate;
resample_poly to 1 Hz with gcd-reduced integer factors when the native rate exceeds 1.5 Hz; analyzer.analyze_window.
A BOUND station delegates to thd_bound_station_operator.daily_measurement, whose reviewed identity is kept.

The identity binds what decides the number: the selector (location pin and its basis), the route (adapters for the
network), the time bounds, the response and coverage policies, the resampling, the estimator and its parameters, and
the source of the code that performs them. It is recorded with each daily attempt and stamped into each recalibrated
baseline entry. It is deliberately NOT part of the comparability key here: adding it would change the key of every
THD station at activation and, with persistence resetting on a key change, alter issued confirmation although no value
changes. Whether to bind it there is a separate, named decision.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from datetime import timedelta
from typing import Dict

MEASUREMENT_VERSION = "thd-daily-measurement-v1"
TARGET_THD_RATE = 1.0   # Hz, the daily estimator rate
MEASUREMENT = dict(
    version=MEASUREMENT_VERSION,
    window="[target - (window_hours + 1) h, target]; for a scored or calibration day D, target = D 00:00 UTC",
    channel="BHZ",
    fetch="seismic_thd.fetch_continuous_data_for_thd (unbound): routed, selector-bound request; "
          "merge(method=1, fill_value='interpolate'); demean + linear detrend",
    response_policy="NOT_REMOVED (raw counts)",
    coverage_policy="no admission rule; pre-merge coverage recorded as facts (thd-coverage-facts-v2)",
    minimum_samples="12 h at the native rate",
    resampling="scipy.signal.resample_poly(up, down), gcd-reduced integer factors, when native > 1.5 x target",
    estimator_rate_hz=TARGET_THD_RATE,
    estimator="SeismicTHDAnalyzer.analyze_window",
)


def _code_identity(*objects) -> str:
    """Qualified name + sha256 of each object's source (method_comparability.code_identity's rule, kept local so this
    module imports nothing heavy); UNIDENTIFIED when any source cannot be read."""
    parts = []
    for obj in objects:
        target = getattr(obj, "__func__", obj)
        try:
            source = inspect.getsource(target)
        except (OSError, TypeError):
            return "UNIDENTIFIED"
        parts.append("%s@%s" % (getattr(target, "__qualname__", repr(target)),
                                hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]))
    return "|".join(parts)


def _number(x):
    return None if x is None else float(x)


def _analyzer_parameters(analyzer) -> Dict:
    return {name: getattr(analyzer, name) for name in ("fundamental_freq", "n_harmonics", "freq_tolerance", "window_hours")}


def measurement_record(network: str, station: str, analyzer) -> Dict:
    """The descriptor of the measurement for one station, with its identity (sha256 of the canonical JSON)."""
    key = f"{network}.{station}"
    import thd_bound_station_operator as OP
    if key in OP.BOUND_STATIONS:
        return dict(station=key, version=OP.DAILY_OPERATOR["operator_version"], identity=OP.daily_operator_identity(key),
                    delegated_to="thd_bound_station_operator.daily_measurement")
    import seismic_thd
    import thd_provider_routing as TPR
    location, basis = TPR.selector_for(network, station)
    payload = dict(station=key, measurement=MEASUREMENT, analyzer=_analyzer_parameters(analyzer),
                   selector=dict(location=location, basis=basis), routing_version=TPR.ROUTING_VERSION,
                   route=TPR.route(network), coverage_version=TPR.COVERAGE_VERSION,
                   code=_code_identity(measure, seismic_thd.fetch_continuous_data_for_thd, type(analyzer).analyze_window))
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
    return dict(identity=identity, **payload)


def measure(network: str, station: str, target, *, analyzer, fetch, channel: str = "BHZ") -> Dict:
    """The daily THD measurement of `station` for `target` (see the module docstring). Returns a record with thd (None
    when unavailable), the requested window, native and estimator rates, sample counts, the measurement identity and
    the analyzer result. Only BHZ is bound: another channel is refused rather than measured under this identity."""
    key = f"{network}.{station}"
    import thd_bound_station_operator as OP
    if key in OP.BOUND_STATIONS:
        rec = OP.daily_measurement(network, station, target, analyzer=analyzer, fetch=fetch)
        rec["measurement_identity"] = rec.get("operator_identity")
        return rec
    if channel != MEASUREMENT["channel"]:
        raise ValueError("THD_MEASUREMENT_CHANNEL_NOT_BOUND: %r" % (channel,))
    start = target - timedelta(hours=analyzer.window_hours + 1)
    try:   # an identity that cannot be formed is UNIDENTIFIED; it never changes or blocks the measurement
        identity = measurement_record(network, station, analyzer)["identity"]
    except Exception:  # noqa: BLE001 -- recording only
        identity = "UNIDENTIFIED"
    rec: Dict = dict(station=key, measurement_identity=identity,
                     requested_window=[start.isoformat(), target.isoformat()], thd=None, native_rate_hz=None,
                     estimator_rate_hz=None, n_native_samples=None, n_processed_samples=None, reason=None)
    data, sample_rate = fetch(station_network=network, station_code=station, start=start, end=target)
    if data is None or len(data) < sample_rate * 3600 * 12:
        rec["reason"] = "INSUFFICIENT_DATA"
        return rec
    rec["native_rate_hz"] = float(sample_rate)
    rec["n_native_samples"] = int(len(data))
    if sample_rate > TARGET_THD_RATE * 1.5:
        from math import gcd
        from scipy.signal import resample_poly
        up, down = int(TARGET_THD_RATE * 100), int(sample_rate * 100)
        common = gcd(up, down)
        data = resample_poly(data, up // common, down // common)
        sample_rate = TARGET_THD_RATE
    rec["estimator_rate_hz"] = float(sample_rate)
    rec["n_processed_samples"] = int(len(data))
    result = analyzer.analyze_window(data=data, sample_rate=sample_rate, station=key, window_time=target)
    rec.update(thd=_number(result.thd_value), p1=_number(result.fundamental_power),
               f1=_number(result.dominant_frequency), snr=_number(result.snr), result=result)
    return rec
