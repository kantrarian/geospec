"""
calibration_eligibility_runner_parity.py -- drive the REAL daily runner (run_ensemble_daily.run_all_regions) of ONE
source tree with identical synthetic inputs, and write every region's EnsembleResult.to_dict() as canonical JSON.

Purpose (calibration-eligibility-v3 wiring, cayley 2026-10-01): measure that the runner wiring is byte-identical
to the base commit with the rule OFF, and show what it carries with the rule ON. Run it once per tree:

    python calibration_eligibility_runner_parity.py --src <tree>/monitoring/src --out <new>.json [--rule-on]

Offline only: no network, no obspy, no data acquisition, no file written inside either tree. The seismic fetch,
THD analyzer and fault-correlation monitor are stubbed at the ensemble module boundary; Lambda_geo acquisition is
stubbed at the runner module boundary (station catalog, GPS solution); the R5 / pilot / event / validation modules
are blocked so their fail-open paths run. The Lambda_geo calibration is a synthetic file written to a fresh
temporary directory and read ONCE by the candidate's own reader (the base tree's loader is given the same
regions). The inputs are the same for both trees.
"""
import argparse
import functools
import json
import os
import sys
import tempfile
import types
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))

TARGET = datetime(2026, 9, 28)
REGION_LIST = ["ridgecrest", "socal_saf_coachella", "istanbul_marmara", "turkey_kahramanmaras", "kaikoura",
               "anchorage", "norcal_hayward", "cascadia", "kumamoto", "tokyo_kanto", "hualien", "mexico_guerrero"]
# THD stations that answer (AK.SSL and HINET.N.KI2H do not, as on the live host, so their fallbacks run).
THD_VALUES = {"IU.TUC": 0.3711, "IU.ANTO": 0.4202, "IU.SNZO": 0.4497223, "IU.COLA": 0.2954, "BK.BKS": 0.3305,
              "IU.COR": 0.3612, "IU.MAJO": 0.6120, "IU.TATO": 0.2805, "MX.TLIG": 0.1903}
# (mean, std, n, window, calibration_date): R3-shaped where calibrated; COLA stale; SNZO the n=0 default.
BASELINES = {
    "IU.TUC": (0.340665, 0.039952, 85, "2026-05-29 to 2026-08-27", "2026-09-26"),
    "IU.ANTO": (0.30, 0.05, 88, "2026-05-29 to 2026-08-27", "2026-09-26"),
    "IU.SNZO": (0.30, 0.07, 0, "UNCALIBRATED", None),
    "IU.COLA": (0.183849, 0.039561, 32, "2026-05-09 to 2026-08-08", "2026-09-07"),
    "BK.BKS": (0.305512, 0.049728, 91, "2026-05-29 to 2026-08-27", "2026-09-26"),
    "IU.COR": (0.30, 0.05, 90, "2026-05-29 to 2026-08-27", None),
    "IU.MAJO": (0.40, 0.06, 91, "2026-05-29 to 2026-08-27", "2026-09-26"),
    "IU.TATO": (0.31, 0.05, 85, "2026-05-29 to 2026-08-27", "2026-09-26"),
    "MX.TLIG": (0.135859, 0.04906, 46, "2026-05-29 to 2026-08-27", "2026-09-26"),
}
# Lambda_geo: NGL solutions for these regions; cascadia's region baseline is unavailable (fallback median);
# hualien is caller-supplied (no provenance).
LG_MAX = {"ridgecrest": 0.031, "istanbul_marmara": 0.044, "kumamoto": 0.052, "cascadia": 0.027, "kaikoura": 0.019}
LG_BASELINE_FILE = {
    "calibration_timestamp": "2026-09-26T06:10:00",
    "baseline_days": 90,
    "baseline_end_date": "2026-08-27",
    "regions": {
        "ridgecrest": {"available": True, "mean_lambda_geo": 0.020, "n_samples": 85, "quality": "good",
                       "calibration_period": "2026-05-29 to 2026-08-27"},
        "istanbul_marmara": {"available": True, "mean_lambda_geo": 0.025, "n_samples": 62, "quality": "good",
                             "calibration_period": "2026-05-29 to 2026-08-27"},
        "kumamoto": {"available": True, "mean_lambda_geo": 0.030, "n_samples": 41, "quality": "acceptable",
                     "calibration_period": "2026-05-29 to 2026-08-27"},
        "kaikoura": {"available": True, "mean_lambda_geo": 0.012, "n_samples": 77, "quality": "good",
                     "calibration_period": "2026-05-29 to 2026-08-27"},
        "cascadia": {"available": False, "reason": "Insufficient stations: 2 < 3", "data_source": "no_data"},
    },
}
CALLER_SUPPLIED_LG = {"hualien": 3.0}


class _THDResult:
    # The fields the producers read, named as seismic_thd.THDResult names them (thd_bound_station_operator.daily_measurement
    # reads fundamental_power and dominant_frequency); values are SYNTHETIC and never reach the scored output.
    def __init__(self, thd_value, snr=10.0):
        self.thd_value, self.snr = float(thd_value), float(snr)
        self.fundamental_power, self.dominant_frequency = 1.0, 2.24e-05


class _StubAnalyzer:
    def __init__(self, window_hours=24, **_):
        self.window_hours = window_hours

    def analyze_window(self, data, sample_rate, station, window_time):
        return _THDResult(THD_VALUES[station])


class _StubCorrelationResult:
    def __init__(self):
        self.data_quality_ok, self.segment_names = True, ["seg-a", "seg-b"]
        self.eigenvalue_ratios, self.participation_ratio, self.qc_reasons = [1.2], 0.5, []


class _StubFCMonitor:
    def __init__(self, *a, **k):
        pass

    def analyze_region(self, region, date, calibration=None):
        return _StubCorrelationResult()


class _CalibrationUnavailable(Exception):
    def __init__(self, reasons):
        super().__init__("; ".join(reasons))
        self.reasons = list(reasons)


def _stub_fetch(station_network, station_code, start, end, channel="BHZ", attempts=None):
    # `attempts` is the real fetch's optional sink (thd-station-attempts-v1); the stub records nothing in it.
    if f"{station_network}.{station_code}" not in THD_VALUES:
        return None, 0.0
    return [0.0] * (3600 * 13), 1.0


class _StubNGL:
    def __init__(self, *a, **k):
        pass

    def load_station_catalog(self):
        return None


def _stub_acquire(region, ngl, days_back, target_date, geonet):
    if region not in LG_MAX:
        return None
    return types.SimpleNamespace(n_stations=6, lambda_geo_max=LG_MAX[region], data_quality="good")


def _install_heavy_stubs():
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

            def _load(region, scored_day, *, band_tag=None, processing_version=None, topology_version=None,
                      capsule_dir=None, expected_sha256=None, embargo_days=14):
                raise _CalibrationUnavailable(["capsule load not available offline"])
            mod.load_calibration_capsule = _load
        elif name == "fault_segments":
            mod.get_segments_for_region = lambda region: ["seg-a", "seg-b", "seg-c"]
            mod.FAULT_SEGMENTS = {}
        elif name == "seismic_thd":
            mod.SeismicTHDAnalyzer = _StubAnalyzer
            mod.THDResult = _THDResult
            mod.fetch_continuous_data_for_thd = _stub_fetch
        elif name == "seismic_data":
            mod.PROCESSING_VERSION = "offline-harness"
            mod._FAULT_CORR_BAND_TAG = "1-10Hz"
        sys.modules[name] = mod
    # Blocked: their import fails, so the runner's existing fail-open paths run (no network, no side effects).
    for name in ("lambda_geo_pilot", "live_data", "regions", "earthquake_events", "validate_predictions",
                 "stress_release_detector", "precip_residual", "src", "src.precip_residual"):
        sys.modules[name] = None


def run(src, rule_on, workdir):
    src = os.path.abspath(src)
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
    sys.path.insert(0, src)
    _install_heavy_stubs()
    import ensemble
    import station_baselines as SB
    import run_ensemble_daily as RD
    if os.path.dirname(os.path.abspath(RD.__file__)) != src or os.path.dirname(os.path.abspath(ensemble.__file__)) != src:
        raise SystemExit("REFUSED: imported a module from outside --src")
    has_rule = hasattr(ensemble, "CE")
    # Pin the configuration explicitly, never inheriting the shipped constants: the rule is OFF unless --rule-on, and
    # attempt recording (a separate, additive feature with its own suites) is OFF in both modes.
    if has_rule:
        ensemble.CE.ELIGIBILITY_RULE_ACTIVE = False
        if hasattr(ensemble.CE, "EFFECTIVE_SCORED_DAY"):
            ensemble.CE.EFFECTIVE_SCORED_DAY = None
    if hasattr(RD, "RECORD_THD_ATTEMPTS"):
        RD.RECORD_THD_ATTEMPTS = False
    if rule_on:
        if not has_rule:
            raise SystemExit("REFUSED: --rule-on needs a tree with the eligibility rule")
        ensemble.CE.ELIGIBILITY_RULE_ACTIVE = True
        if hasattr(ensemble.CE, "EFFECTIVE_SCORED_DAY"):
            # an activation declares its boundary; the parity day is the first day the rule applies (inclusive)
            ensemble.CE.EFFECTIVE_SCORED_DAY = TARGET.date().isoformat()
    ensemble.SeismicTHDAnalyzer = _StubAnalyzer
    ensemble.FaultCorrelationMonitor = _StubFCMonitor
    ensemble.fetch_continuous_data_for_thd = _stub_fetch
    fields = SB.StationBaseline.__dataclass_fields__
    SB.STATION_BASELINES.clear()
    for key, (mean, std, n, period, cal) in BASELINES.items():
        kwargs = dict(station=key, mean_thd=mean, std_thd=std, n_samples=n, calibration_period=period, notes="fixture")
        if "calibration_date" in fields:
            kwargs["calibration_date"] = cal
        SB.STATION_BASELINES[key] = SB.StationBaseline(**kwargs)
    RD.NGL_LAMBDA_GEO_AVAILABLE = True
    RD.LAMBDA_GEO_PILOT_AVAILABLE = False
    RD.FAULT_POLYGONS = {r: True for r in REGION_LIST}
    RD.NGLLiveAcquisition = _StubNGL
    RD.GeoNetLiveAcquisition = _StubNGL
    RD.acquire_region_data = _stub_acquire
    RD.load_lambda_geo_baselines = lambda: json.loads(json.dumps(LG_BASELINE_FILE["regions"]))
    if hasattr(RD, "read_lambda_geo_calibration"):
        # v4: the candidate reads denominators AND provenance from ONE read of this file (its own reader).
        path = os.path.join(workdir, "lambda_geo_baselines.json")
        with open(path, "x", encoding="utf-8") as handle:
            json.dump(LG_BASELINE_FILE, handle)
        RD.read_lambda_geo_calibration = functools.partial(RD.read_lambda_geo_calibration, baseline_file=path)
    results, _ = RD.run_all_regions(target_date=TARGET, regions=list(REGION_LIST),
                                    lambda_geo_data=dict(CALLER_SUPPLIED_LG), use_seismic=True, fetch_events=False)
    return {region: results[region].to_dict() for region in sorted(results)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rule-on", action="store_true")
    args = parser.parse_args()
    import logging
    logging.disable(logging.CRITICAL)
    with tempfile.TemporaryDirectory(prefix="eligibility_parity_") as workdir:
        payload = run(args.src, args.rule_on, workdir)
    with open(args.out, "x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, indent=1, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
