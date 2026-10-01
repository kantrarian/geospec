"""
calibration_eligibility_fixtures.py -- offline fixtures + harness for the calibration-eligibility rule (v3).

Drives the REAL `GeoSpecEnsemble.compute_risk` / `compute_thd_risk` / `compute_fault_correlation_risk` /
`compute_lambda_geo_risk` paths with synthetic inputs: the seismic fetch and THD analyzer are stubbed (no obspy,
no network), the fault-correlation capsule is injected through the existing `capsule_loader` seam, and
`station_baselines.STATION_BASELINES` is pointed at a fixture baseline. No collection, no recalibration.

The same harness runs the PRE-RULE ensemble (a copy of ensemble.py from the base commit, imported as
`ensemble_base`) and the candidate, so "flag off == byte-identical" is a measured comparison, not a claim.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from datetime import datetime
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# ---------------------------------------------------------------------------------------------------------------
# Heavy-module stubs (installed ONLY when the real modules cannot be imported, e.g. no obspy on this host)
# ---------------------------------------------------------------------------------------------------------------

STUBBED: List[str] = []


class _THDResult:
    def __init__(self, thd_value: float, snr: float = 10.0):
        self.thd_value, self.snr = float(thd_value), float(snr)


class _StubTHDAnalyzer:
    """Returns the THD value the harness planted on the instance (set by `build`)."""
    planted_thd: float = 0.0

    def __init__(self, window_hours: int = 24):
        self.window_hours = window_hours

    def analyze_window(self, data, sample_rate, station, window_time):
        return _THDResult(type(self).planted_thd)


class _StubCorrelationResult:
    def __init__(self, ok: bool, names, l2_l1: float, pr: float):
        self.data_quality_ok, self.segment_names = ok, list(names)
        self.eigenvalue_ratios, self.participation_ratio, self.qc_reasons = [l2_l1], pr, []


class _StubFCMonitor:
    planted = None  # (ok, names, l2_l1, pr)

    def __init__(self, window_hours: int = 24, decorrelation_threshold: float = 0.3):
        pass

    def analyze_region(self, region, date, calibration=None):
        ok, names, l2_l1, pr = type(self).planted or (True, ["seg-a", "seg-b"], 1.2, 0.5)
        return _StubCorrelationResult(ok, names, l2_l1, pr)


class _CalibrationUnavailable(Exception):
    def __init__(self, reasons):
        super().__init__("; ".join(reasons))
        self.reasons = list(reasons)


# ---------------------------------------------------------------------------------------------------------------
# Writer-derived facts about fault_correlation.load_calibration_capsule, read from its SOURCE (the module itself
# needs obspy and may not import here): the registered `embargo_days` default, the exact capsule key set, and the
# loader's own refusal text. Fixtures are built from these, never from wording the loader does not write.
# ---------------------------------------------------------------------------------------------------------------

FAULT_CORRELATION_PY = os.path.join(HERE, "fault_correlation.py")


def _loader_def():
    import ast
    with open(FAULT_CORRELATION_PY, "r", encoding="utf-8") as fh:
        source = fh.read()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "load_calibration_capsule":
            return source, node
    raise LookupError("load_calibration_capsule not found in %s" % FAULT_CORRELATION_PY)


def fc_registered_embargo_days() -> int:
    """The loader's registered keyword-only default for `embargo_days`."""
    import ast
    _, node = _loader_def()
    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        if arg.arg == "embargo_days" and default is not None:
            return ast.literal_eval(default)
    raise LookupError("embargo_days default not found")


def fc_capsule_keys() -> set:
    """The exact key set the loader requires (its `expected_keys` set literal)."""
    import ast
    _, node = _loader_def()
    for sub in ast.walk(node):
        if (isinstance(sub, ast.Assign) and len(sub.targets) == 1 and isinstance(sub.targets[0], ast.Name)
                and sub.targets[0].id == "expected_keys"):
            return set(ast.literal_eval(sub.value))
    raise LookupError("expected_keys not found")


def fc_loader_source_text() -> str:
    import ast
    source, node = _loader_def()
    return ast.get_source_segment(source, node) or ""


def fc_capsule(region: str, *, start: str, end: str, valid_through: str) -> dict:
    """A capsule with exactly the loader's key set; content values are synthetic and labelled so."""
    values = dict(schema="SYNTHETIC_FIXTURE", region=region, band_tag="1-10Hz", processing_version="offline-harness",
                  topology_version="fixture", threshold=0.5, calibration_window={"start": start, "end": end},
                  source_commit="0" * 40, input_manifest_sha256="0" * 64, replay_output_sha256="0" * 64,
                  issued_utc="2026-08-25T00:00:00Z", valid_through=valid_through)
    keys = fc_capsule_keys()
    if set(values) != keys:
        raise AssertionError("fixture capsule keys %s != loader keys %s" % (sorted(values), sorted(keys)))
    return values


def _stub_load_calibration_capsule(region, scored_day, *, band_tag=None, processing_version=None,
                                   topology_version=None, capsule_dir=None, expected_sha256=None,
                                   embargo_days=fc_registered_embargo_days()):
    # The stub carries the REAL registered default (read from source above) so the ensemble reads the same policy
    # through inspect.signature here as it does against the real module.
    raise _CalibrationUnavailable(["capsule load not available in the offline harness"])


def _stub_fetch(station_network, station_code, start, end):
    # >= 12 h of 1 Hz samples so compute_thd_risk does not refuse for insufficiency; no resampling at 1 Hz.
    return [0.0] * (3600 * 13), 1.0


def install_stubs() -> List[str]:
    """Install stand-ins for the obspy-backed modules ensemble.py imports, when they are not importable."""
    if STUBBED:
        return STUBBED
    for name in ("fault_correlation", "fault_segments", "seismic_thd", "seismic_data"):
        try:
            __import__(name)
            continue
        except Exception:
            pass
        mod = types.ModuleType(name)
        if name == "fault_correlation":
            mod.FaultCorrelationMonitor = _StubFCMonitor
            mod.CorrelationResult = _StubCorrelationResult
            mod.CalibrationUnavailable = _CalibrationUnavailable
            mod.capsule_registry_entry = lambda region, registry_path=None: (_ for _ in ()).throw(
                _CalibrationUnavailable(["registry entry for %s missing" % region]))
            mod.load_calibration_capsule = _stub_load_calibration_capsule
        elif name == "fault_segments":
            mod.get_segments_for_region = lambda region: ["seg-a", "seg-b", "seg-c"]
            mod.FAULT_SEGMENTS = {}
        elif name == "seismic_thd":
            mod.SeismicTHDAnalyzer = _StubTHDAnalyzer
            mod.THDResult = _THDResult
            mod.fetch_continuous_data_for_thd = _stub_fetch
        elif name == "seismic_data":
            mod.PROCESSING_VERSION = "offline-harness"
            mod._FAULT_CORR_BAND_TAG = "1-10Hz"
        sys.modules[name] = mod
        STUBBED.append(name)
    return STUBBED


def load_module_from_file(path: str, module_name: str):
    """Import an ensemble.py copy (e.g. the base commit's) under `module_name`, with the stubs installed."""
    install_stubs()
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------------------------------------------
# Fixture baselines and scenarios
# ---------------------------------------------------------------------------------------------------------------

SCORED_DAY = datetime(2026, 9, 28)          # the measured Kaikoura example (codex a7d0533d)
KAIKOURA_THD = 0.4497223                    # -> score 0.4097225 under the current mapping (z 2.14, n 0)
MAX_AGE = 50                                # ensemble.MAX_BASELINE_AGE_DAYS at the base commit
MIN_LAG = 30                                # run_thd_recal.EXCLUDE_RECENT_DAYS (the R3 lag); a test pins both


def baseline(station: str, mean: float, std: float, n: int, period: str, notes: str = "",
             calibration_date: Optional[str] = None):
    """v3: `calibration_date` is explicit per fixture (None = unknown); it matters only to a baseline that is
    otherwise calibrated, because every earlier refusal is decided first."""
    from station_baselines import StationBaseline
    return StationBaseline(station=station, mean_thd=mean, std_thd=std, n_samples=n, calibration_period=period,
                           notes=notes, calibration_date=calibration_date)


def fixture_baselines() -> Dict[str, dict]:
    """Each fixture: the baseline (or None), the station/network, the regions it serves, the THD value, and the
    status the rule is EXPECTED to assign. Windows are relative to SCORED_DAY (2026-09-28).
    v3: a calibrated fixture is R3-shaped -- its window ends MIN_LAG (30) d before the day it was calibrated,
    and that day is not after the scored day (v2's "fresh" window ended 1 d before the scored day, which no
    calibration on or before the scored day can produce under the registered lag)."""
    fresh = "2026-05-29 to 2026-08-27"      # window end 32 d before the scored day; calibrated 2026-09-26
    fresh_cal = "2026-09-26"
    at_limit = "2026-05-10 to 2026-08-09"   # window end exactly MAX_AGE (50) d before; calibrated 2026-09-08
    past_limit = "2026-05-09 to 2026-08-08" # window end 51 d before -> stale
    return {
        "missing_no_baseline": dict(baseline=None, station="CAFE", network="IV", regions=["campi_flegrei"],
                                    thd=0.2000, expected="missing"),
        "zero_baseline": dict(baseline=baseline("XX.ZERO", 0.0, 0.0, 31, fresh), station="ZERO", network="XX",
                              regions=["r-zero"], thd=0.4497223, expected="zero"),
        "n0_default_kaikoura": dict(baseline=baseline("IU.SNZO", 0.30, 0.07, 0, "UNCALIBRATED",
                                                      "UNCALIBRATED. estimate based on similar IU broadband"),
                                    station="SNZO", network="IU", regions=["kaikoura"], thd=KAIKOURA_THD,
                                    expected="n0_default"),
        "stale_window": dict(baseline=baseline("IU.COLA", 0.183849, 0.039561, 32, past_limit), station="COLA",
                             network="IU", regions=["anchorage"], thd=0.4497223, expected="stale"),
        "calibrated_at_age_limit": dict(baseline=baseline("MX.TLIG", 0.135859, 0.04906, 87, at_limit,
                                                          calibration_date="2026-09-08"),
                                        station="TLIG", network="MX", regions=["mexico_guerrero"], thd=0.2500,
                                        expected="calibrated"),
        "calibrated_fresh": dict(baseline=baseline("BK.BKS", 0.305512, 0.049728, 31, fresh,
                                                   calibration_date=fresh_cal), station="BKS",
                                 network="BK", regions=["norcal_hayward"], thd=0.4497223, expected="calibrated"),
        "shared_station_tuc": dict(baseline=baseline("IU.TUC", 0.340665, 0.039952, 31, fresh,
                                                     calibration_date=fresh_cal), station="TUC",
                                   network="IU", regions=["ridgecrest", "socal_saf_mojave", "socal_saf_coachella"],
                                   thd=0.4497223, expected="shared_station"),
        # v3 lag re-check: otherwise calibrated baselines that cannot show the registered lag.
        "lag_calibration_date_unknown": dict(baseline=baseline("IU.TATO", 0.30, 0.05, 88, fresh),
                                             station="TATO", network="IU", regions=["hualien"], thd=0.4497223,
                                             expected="missing", code="CALIBRATION_DATE_UNKNOWN"),
        "lag_not_honored": dict(baseline=baseline("IU.COR", 0.30, 0.05, 88, fresh, calibration_date="2026-09-25"),
                                station="COR", network="IU", regions=["cascadia"], thd=0.4497223,
                                expected="missing", code="LAG_NOT_HONORED"),
        "lag_calibrated_after_scored_day": dict(
            baseline=baseline("IU.ANTO", 0.30, 0.05, 88, fresh, calibration_date="2026-09-29"), station="ANTO",
            network="IU", regions=["istanbul_marmara"], thd=0.4497223, expected="missing",
            code="CALIBRATED_AFTER_SCORED_DAY"),
    }


def fc_fixtures() -> Dict[str, dict]:
    """Fault-correlation fixtures. The refusal texts are the loader's own wording ("no registry entry for region
    ...", "scored day ... past valid_through ... (STALE)"); admitted capsules carry the loader's exact key set and
    are re-checked against valid_through and the REGISTERED embargo read from the loader source."""
    embargo = fc_registered_embargo_days()
    fresh = fc_capsule("kaikoura", start="2026-05-01", end="2026-08-24", valid_through="2026-10-31")
    return {
        "fc_missing_capsule": dict(state="unavailable", reasons=["no registry entry for region kaikoura"],
                                   expected="missing", code="NO_ADMISSIBLE_CAPSULE"),
        "fc_expired_capsule": dict(state="unavailable",
                                   reasons=["scored day 2026-09-28 past valid_through 2026-09-09 (STALE)"],
                                   expected="expired", code="CAPSULE_PAST_VALID_THROUGH"),
        "fc_admitted": dict(state="admitted", reasons=[], capsule=fresh, embargo=embargo,
                            expected="calibrated", code="CALIBRATED"),
        "fc_admitted_no_capsule": dict(state="admitted", reasons=[], capsule=None, embargo=embargo,
                                       expected="missing", code="CAPSULE_NOT_SUPPLIED"),
        "fc_admitted_embargo_unreadable": dict(state="admitted", reasons=[], capsule=fresh, embargo=None,
                                               expected="missing", code="EMBARGO_POLICY_UNREADABLE"),
        "fc_admitted_past_valid_through": dict(
            state="admitted", reasons=[], embargo=embargo, expected="expired", code="CAPSULE_PAST_VALID_THROUGH",
            capsule=fc_capsule("kaikoura", start="2026-04-05", end="2026-08-03", valid_through="2026-09-09")),
        "fc_admitted_future_window": dict(
            state="admitted", reasons=[], embargo=embargo, expected="missing", code="FUTURE_WINDOW_END",
            capsule=fc_capsule("kaikoura", start="2026-07-01", end="2026-10-10", valid_through="2026-10-31")),
        "fc_admitted_reversed_window": dict(
            state="admitted", reasons=[], embargo=embargo, expected="missing", code="WINDOW_REVERSED",
            capsule=fc_capsule("kaikoura", start="2026-08-24", end="2026-05-01", valid_through="2026-10-31")),
        "fc_admitted_inside_embargo": dict(
            state="admitted", reasons=[], embargo=embargo, expected="missing", code="CAPSULE_INSIDE_EMBARGO",
            capsule=fc_capsule("kaikoura", start="2026-06-01", end="2026-09-10", valid_through="2026-10-31")),
    }


# Classifier-level Lambda_geo fixtures. `max_age` / `min_lag` are EXPLICIT test policies (the runner registers
# neither -- see ensemble.LAMBDA_GEO_BASELINE_MAX_AGE_DAYS / _MIN_LAG_DAYS); `lg_unregistered_policy` is what the
# runner itself would produce. Every fixture states both policies (no default).
LG_FIXTURES = {
    "lg_no_provenance": dict(provenance=None, max_age=MAX_AGE, min_lag=MIN_LAG, expected="missing", code="NO_PROVENANCE"),
    "lg_n0": dict(provenance={"source": "ngl-baseline", "n_days": 0, "window_end": "2026-09-27"}, max_age=MAX_AGE,
                  min_lag=MIN_LAG, expected="n0_default", code="ZERO_SAMPLE_DEFAULT"),
    "lg_stale": dict(provenance={"source": "ngl-baseline", "n_days": 90, "window_end": "2026-08-08"}, max_age=MAX_AGE,
                     min_lag=MIN_LAG, expected="stale", code="WINDOW_STALE"),
    "lg_calibrated_under_explicit_test_policy": dict(
        provenance={"source": "ngl-baseline", "n_days": 90, "window_start": "2026-05-26", "window_end": "2026-08-24",
                    "calibrated_on": "2026-09-23"},
        max_age=MAX_AGE, min_lag=MIN_LAG, expected="calibrated", code="CALIBRATED"),
    "lg_unregistered_policy": dict(
        provenance={"source": "ngl-baseline", "n_days": 90, "window_start": "2026-05-26", "window_end": "2026-08-24",
                    "calibrated_on": "2026-09-23"},
        max_age=None, min_lag=None, expected="missing", code="NO_REGISTERED_FRESHNESS_POLICY"),
    # v3: a registered freshness bound alone is not enough -- the lag must be registered too.
    "lg_unregistered_lag": dict(
        provenance={"source": "ngl-baseline", "n_days": 90, "window_start": "2026-05-26", "window_end": "2026-08-24",
                    "calibrated_on": "2026-09-23"},
        max_age=MAX_AGE, min_lag=None, expected="missing", code="NO_REGISTERED_LAG_POLICY"),
    "lg_calibration_date_unknown": dict(
        provenance={"source": "ngl-baseline", "n_days": 90, "window_start": "2026-05-26", "window_end": "2026-08-24"},
        max_age=MAX_AGE, min_lag=MIN_LAG, expected="missing", code="CALIBRATION_DATE_UNKNOWN"),
    "lg_lag_not_honored": dict(
        provenance={"source": "ngl-baseline", "n_days": 90, "window_start": "2026-05-26", "window_end": "2026-08-24",
                    "calibrated_on": "2026-09-22"},
        max_age=MAX_AGE, min_lag=MIN_LAG, expected="missing", code="LAG_NOT_HONORED"),
    "lg_calibrated_after_scored_day": dict(
        provenance={"source": "ngl-baseline", "n_days": 90, "window_start": "2026-05-26", "window_end": "2026-08-24",
                    "calibrated_on": "2026-09-30"},
        max_age=MAX_AGE, min_lag=MIN_LAG, expected="missing", code="CALIBRATED_AFTER_SCORED_DAY"),
    "lg_future_window": dict(provenance={"source": "ngl-baseline", "n_days": 90, "window_end": "2026-10-10"},
                             max_age=MAX_AGE, min_lag=MIN_LAG, expected="missing", code="FUTURE_WINDOW_END"),
    "lg_reversed_window": dict(provenance={"source": "ngl-baseline", "n_days": 90, "window_start": "2026-08-24",
                                           "window_end": "2026-05-26"},
                               max_age=MAX_AGE, min_lag=MIN_LAG, expected="missing", code="WINDOW_REVERSED"),
    "lg_boolean_count": dict(provenance={"source": "ngl-baseline", "n_days": True, "window_end": "2026-08-24"},
                             max_age=MAX_AGE, min_lag=MIN_LAG, expected="missing", code="INVALID_SAMPLE_COUNT"),
    "lg_fractional_count": dict(provenance={"source": "ngl-baseline", "n_days": 89.5, "window_end": "2026-08-24"},
                                max_age=MAX_AGE, min_lag=MIN_LAG, expected="missing", code="INVALID_SAMPLE_COUNT"),
}


def input_validation_fixtures() -> Dict[str, dict]:
    """THD input-validation fixtures (codex 638dd6e9 finding 1): n_samples 90, window 2026-05-26 to 2026-08-24
    (35 d before the scored day) unless the fixture varies it. Each EXPECTS a typed refusal except the controls.
    v3: the controls are calibrated 2026-09-23 (exactly the registered 30 d lag after the window end)."""
    period = "2026-05-26 to 2026-08-24"
    cal = "2026-09-23"
    nan, inf = float("nan"), float("inf")
    return {
        "control_nominal": dict(baseline=baseline("IU.T", 0.30, 0.07, 90, period, calibration_date=cal),
                                expected="calibrated", code="CALIBRATED"),
        "control_integral_float_count": dict(baseline=baseline("IU.T", 0.30, 0.07, 90.0, period, calibration_date=cal),
                                             expected="calibrated", code="CALIBRATED"),
        "mean_nan": dict(baseline=baseline("IU.T", nan, 0.07, 90, period), expected="missing", code="NON_FINITE_STATISTIC"),
        "std_pos_inf": dict(baseline=baseline("IU.T", 0.30, inf, 90, period), expected="missing", code="NON_FINITE_STATISTIC"),
        "mean_neg_inf": dict(baseline=baseline("IU.T", -inf, 0.07, 90, period), expected="missing", code="NON_FINITE_STATISTIC"),
        "std_nan": dict(baseline=baseline("IU.T", 0.30, nan, 90, period), expected="missing", code="NON_FINITE_STATISTIC"),
        "mean_boolean": dict(baseline=baseline("IU.T", True, 0.07, 90, period), expected="missing", code="STATISTIC_UNREADABLE"),
        "mean_text": dict(baseline=baseline("IU.T", "0.30", 0.07, 90, period), expected="missing", code="STATISTIC_UNREADABLE"),
        "count_boolean_true": dict(baseline=baseline("IU.T", 0.30, 0.07, True, period), expected="missing", code="INVALID_SAMPLE_COUNT"),
        "count_boolean_false": dict(baseline=baseline("IU.T", 0.30, 0.07, False, period), expected="missing", code="INVALID_SAMPLE_COUNT"),
        "count_fractional": dict(baseline=baseline("IU.T", 0.30, 0.07, 90.5, period), expected="missing", code="INVALID_SAMPLE_COUNT"),
        "count_negative": dict(baseline=baseline("IU.T", 0.30, 0.07, -3, period), expected="missing", code="INVALID_SAMPLE_COUNT"),
        "count_nan": dict(baseline=baseline("IU.T", 0.30, 0.07, nan, period), expected="missing", code="INVALID_SAMPLE_COUNT"),
        "window_future": dict(baseline=baseline("IU.T", 0.30, 0.07, 90, "2026-10-01 to 2026-10-10"),
                              expected="missing", code="FUTURE_WINDOW_END"),
        "window_end_tomorrow": dict(baseline=baseline("IU.T", 0.30, 0.07, 90, "2026-07-01 to 2026-09-29"),
                                    expected="missing", code="FUTURE_WINDOW_END"),
        # v2 expected CALIBRATED (the non-future bound admits age 0). v3: it still passes that bound and is then
        # refused at the lag -- a window ending on the scored day cannot end 30 d before a calibration on or
        # before that day.
        "window_end_on_scored_day": dict(baseline=baseline("IU.T", 0.30, 0.07, 90, "2026-07-01 to 2026-09-28",
                                                           calibration_date="2026-09-28"),
                                         expected="missing", code="LAG_NOT_HONORED"),
        "window_reversed": dict(baseline=baseline("IU.T", 0.30, 0.07, 90, "2026-08-24 to 2026-05-26"),
                                expected="missing", code="WINDOW_REVERSED"),
        "window_end_only": dict(baseline=baseline("IU.T", 0.30, 0.07, 90, "2026-08-24"),
                                expected="missing", code="WINDOW_INCOMPLETE"),
        "window_garbled_start": dict(baseline=baseline("IU.T", 0.30, 0.07, 90, "2026-13-01 to 2026-08-24"),
                                     expected="missing", code="WINDOW_UNREADABLE"),
    }


# ---------------------------------------------------------------------------------------------------------------
# Harness: run the REAL compute_risk with planted inputs
# ---------------------------------------------------------------------------------------------------------------

def build(module, *, region: str, network: str, station: str, thd_baseline, thd_value: float,
          date: datetime = SCORED_DAY, shared: Optional[List[str]] = None, active: Optional[bool] = None,
          lg_ratio: Optional[float] = None, lg_provenance: Optional[dict] = None,
          fc: Optional[dict] = None, thd_unavailable: bool = False):
    """Assess one region-day through `module.GeoSpecEnsemble.compute_risk`.

    `module` is either the candidate `ensemble` or a base-commit copy (which lacks the eligibility kwargs; the
    harness only passes them when the module knows them). Returns the EnsembleResult."""
    import station_baselines as SB
    install_stubs()
    kwargs = {}
    accepts_rule = "eligibility_rule_active" in module.GeoSpecEnsemble.__init__.__code__.co_varnames
    if accepts_rule:
        kwargs = dict(eligibility_rule_active=active, station_regions={f"{network}.{station}": list(shared or [region])})
    ens = module.GeoSpecEnsemble(region, **kwargs)
    # THD inputs: the harness ALWAYS plants the fetch and the analyzer in the module under test (the real
    # seismic_thd may import but needs obspy/network at fetch time); the fixture baseline is what get_baseline
    # returns. The fault-correlation monitor is planted too; the capsule decision goes through capsule_loader.
    key = f"{network}.{station}"
    saved = SB.STATION_BASELINES.get(key, "<absent>")
    if thd_baseline is None:
        SB.STATION_BASELINES.pop(key, None)
    else:
        SB.STATION_BASELINES[key] = thd_baseline
    _StubTHDAnalyzer.planted_thd = thd_value
    original_fetch = module.fetch_continuous_data_for_thd
    module.fetch_continuous_data_for_thd = (lambda **k: (None, 1.0)) if thd_unavailable else _stub_fetch
    ens.thd_analyzer = _StubTHDAnalyzer(window_hours=ens.thd_analyzer.window_hours)
    ens.fault_corr_monitor = _StubFCMonitor()
    try:
        # Fault correlation through the existing capsule_loader seam.
        fc = fc or dict(state="unavailable", reasons=["registry entry for %s missing" % region])
        if fc["state"] == "admitted":
            admitted = fc.get("capsule") or fc_capsule(region, start="2026-05-01", end="2026-08-24",
                                                         valid_through="2026-10-31")
            ens.capsule_loader = lambda r, d, _capsule=admitted: dict(_capsule)
            _StubFCMonitor.planted = (True, ["seg-a", "seg-b"], fc.get("l2_l1", 1.2), fc.get("pr", 0.5))
        else:
            exc_type = sys.modules["fault_correlation"].CalibrationUnavailable
            reasons = list(fc.get("reasons") or [])
            def _raise(r, d, _reasons=reasons, _t=exc_type):
                raise _t(_reasons)
            ens.capsule_loader = _raise
        # Lambda_geo, with provenance only when the module knows the parameter.
        if lg_ratio is not None:
            if "provenance" in module.GeoSpecEnsemble.set_lambda_geo.__code__.co_varnames:
                ens.set_lambda_geo(date, lg_ratio, provenance=lg_provenance)
            else:
                ens.set_lambda_geo(date, lg_ratio)
        return ens.compute_risk(date, thd_station=station, thd_network=network)
    finally:
        if saved == "<absent>":
            SB.STATION_BASELINES.pop(key, None)
        else:
            SB.STATION_BASELINES[key] = saved
        module.fetch_continuous_data_for_thd = original_fetch


def result_dict(result) -> dict:
    """The serialized EnsembleResult, exactly as the runner writes it (to_dict), for byte comparisons."""
    return result.to_dict()
