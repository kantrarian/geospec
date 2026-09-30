"""
calibration_eligibility_fixtures.py -- offline fixtures + harness for calibration-eligibility-v1.

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
            mod.load_calibration_capsule = lambda *a, **k: (_ for _ in ()).throw(
                _CalibrationUnavailable(["capsule load not available in the offline harness"]))
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


def baseline(station: str, mean: float, std: float, n: int, period: str, notes: str = ""):
    from station_baselines import StationBaseline
    return StationBaseline(station=station, mean_thd=mean, std_thd=std, n_samples=n, calibration_period=period,
                           notes=notes)


def fixture_baselines() -> Dict[str, dict]:
    """Each fixture: the baseline (or None), the station/network, the regions it serves, the THD value, and the
    status the rule is EXPECTED to assign. Windows are relative to SCORED_DAY (2026-09-28)."""
    fresh = "2026-06-30 to 2026-09-27"      # window end 1 d before the scored day
    at_limit = "2026-05-10 to 2026-08-09"   # window end exactly MAX_AGE (50) d before
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
        "calibrated_at_age_limit": dict(baseline=baseline("MX.TLIG", 0.135859, 0.04906, 87, at_limit),
                                        station="TLIG", network="MX", regions=["mexico_guerrero"], thd=0.2500,
                                        expected="calibrated"),
        "calibrated_fresh": dict(baseline=baseline("BK.BKS", 0.305512, 0.049728, 31, fresh), station="BKS",
                                 network="BK", regions=["norcal_hayward"], thd=0.4497223, expected="calibrated"),
        "shared_station_tuc": dict(baseline=baseline("IU.TUC", 0.340665, 0.039952, 31, fresh), station="TUC",
                                   network="IU", regions=["ridgecrest", "socal_saf_mojave", "socal_saf_coachella"],
                                   thd=0.4497223, expected="shared_station"),
    }


FC_FIXTURES = {
    "fc_missing_capsule": dict(state="unavailable", reasons=["registry entry for kaikoura missing"], expected="missing"),
    "fc_expired_capsule": dict(state="unavailable", reasons=["capsule for kaikoura expired 2026-08-01"], expected="expired"),
    "fc_admitted": dict(state="admitted", reasons=[], expected="calibrated"),
}

LG_FIXTURES = {
    "lg_no_provenance": dict(provenance=None, expected="missing"),
    "lg_n0": dict(provenance={"source": "ngl-baseline", "n_days": 0, "window_end": "2026-09-27"}, expected="n0_default"),
    "lg_stale": dict(provenance={"source": "ngl-baseline", "n_days": 90, "window_end": "2026-08-08"}, expected="stale"),
    "lg_calibrated": dict(provenance={"source": "ngl-baseline", "n_days": 90, "window_end": "2026-08-29"}, expected="calibrated"),
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
            ens.capsule_loader = lambda r, d: {"capsule": "admitted"}
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
