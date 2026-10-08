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
the source of the code that performs them. It is PROVENANCE: recorded with each daily attempt and stamped into each
recalibrated baseline entry, never a comparability key (codex cfb195ff: source, station, raw-sample and receipt hashes
belong in provenance; a key that moves with them would reset persistence every day).

Comparability is staged (codex cfb195ff decision). Two SEMANTIC parts may enter the THD key, each only when it really
changes: (1) `operator_class` -- selector location, coverage admission, response handling, the estimator definition
and its parameters -- joins the estimator identity when it differs from LEGACY_OPERATOR_CLASS, the operator as it stood
before this module (823a84bc), so an unchanged station keeps a byte-identical key while a real operator change cannot
keep the old class; (2) the calibration CONVENTION of the selected baseline (`calibration_convention`, stamped by the
recal) joins the calibration class the first time a baseline computed under thd-daily-measurement-v1 is selected, and
routine recals within that convention do not move it. A bound station (SNZO) keeps its separately bound identity.
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


# What the measurement admits and how it treats the response: part of the operator class, so a change is a new class.
COVERAGE_ADMISSION = "NONE"        # no partial-day admission rule; coverage is recorded as facts only
RESPONSE_HANDLING = "NOT_REMOVED"  # raw counts; demean + linear detrend only
OPERATOR_CLASS_VERSION = "thd-operator-class-v1"
# The unbound daily operator as it stood at 823a84bc (ensemble.compute_thd_risk inline path): FROZEN history, never
# derived from the current definitions -- deriving it would let a changed definition match itself.
LEGACY_OPERATOR_CLASS = dict(
    selector="*",
    coverage_admission="NONE",
    response="NOT_REMOVED",
    window="[target - (window_hours + 1) h, target]; for a scored or calibration day D, target = D 00:00 UTC",
    minimum_samples="12 h at the native rate",
    resampling="scipy.signal.resample_poly(up, down), gcd-reduced integer factors, when native > 1.5 x target",
    estimator_rate_hz=1.0,
    estimator="SeismicTHDAnalyzer.analyze_window",
    analyzer=dict(fundamental_freq=2.2365360529611736e-05, n_harmonics=5, freq_tolerance=0.1, window_hours=24),
)


def operator_class(network: str, station: str, analyzer) -> Dict:
    """The semantic class of the daily operator for an UNBOUND station: what decides the value, without provenance
    (no basis text, route, source hashes, dates or sample digests)."""
    import thd_provider_routing as TPR
    location, _basis = TPR.selector_for(network, station)
    return dict(selector=location, coverage_admission=COVERAGE_ADMISSION, response=RESPONSE_HANDLING,
                window=MEASUREMENT["window"], minimum_samples=MEASUREMENT["minimum_samples"],
                resampling=MEASUREMENT["resampling"], estimator_rate_hz=MEASUREMENT["estimator_rate_hz"],
                estimator=MEASUREMENT["estimator"], analyzer=_analyzer_parameters(analyzer))


def operator_class_part(network: str, station: str, analyzer):
    """The comparability part for an unbound station's operator: None while it equals LEGACY_OPERATOR_CLASS (the key
    stays byte-identical), else 'operator_class=<version>:<sha256[:16]>'."""
    cls = operator_class(network, station, analyzer)
    if cls == LEGACY_OPERATOR_CLASS:
        return None
    digest = hashlib.sha256(json.dumps(cls, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    return "operator_class=%s:%s" % (OPERATOR_CLASS_VERSION, digest)


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
    identity = ("UNIDENTIFIED" if payload["code"] == "UNIDENTIFIED" else
                hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest())
    return dict(identity=identity, version=MEASUREMENT_VERSION, **payload)


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
