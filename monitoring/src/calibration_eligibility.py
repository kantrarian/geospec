"""
calibration_eligibility.py -- PROSPECTIVE, VERSIONED calibration-eligibility rule (calibration-eligibility-v1).

NOT ACTIVATED. `ELIGIBILITY_RULE_ACTIVE` is False and stays False until a dated amendment / owner decision flips
it; with the flag off every ordinary-run output is byte-identical to the pre-rule runner (proved by
calibration_eligibility_report.py and test_calibration_eligibility_cayley_20260930.py). Saved historical scores
are never re-scored by this module.

Why (codex a7d0533d item 1, 2026-09-30): a station with NO empirical local baseline (e.g. IU.SNZO: mean 0.30 /
std 0.07 / n_samples 0, calibration_period 'UNCALIBRATED') is z-scored by `ensemble.thd_to_risk_with_baseline`
exactly like a calibrated one, so a DEFAULT baseline acted as calibrated evidence in tiering (Kaikoura 2026-09-28:
THD 0.4497223 -> score 0.4097225, z 2.14, n 0). Numeric availability (a raw value exists) is distinct from
calibrated eligibility (the value can be compared against an honestly established baseline).

What the rule does when ACTIVE: every method observation carries `calibration_status`,
`eligibility_rule_version` and `eligible_for_tiering`; raw values, scores and notes stay visible exactly as
today; the ENSEMBLE tier, `methods_available`, effective weights and confidence count ONLY eligible observations.
An observation whose qualification is unknown is never qualified.

Statuses (exactly these seven):
  missing        no baseline / calibration record, or a window that cannot be read
  zero           a baseline whose mean or std is <= 0 (the runner already falls back to absolute thresholds)
  n0_default     a baseline with n_samples <= 0 -- a default or manual estimate, not an empirical local baseline
  stale          a baseline whose window ended more than `max_age_days` before the scored date
  expired        a calibration capsule the registry reports expired (fault_correlation)
  calibrated     an empirical baseline (n_samples > 0) whose window is readable and within `max_age_days`
  shared_station a calibrated baseline serving more than one region (the support is shared, and eligible)
Only `calibrated` and `shared_station` are eligible for tiering.

The classifier reads STRUCTURED fields (n_samples, std, the parsed window end, the registry state), never the
free-text notes, so relabelling a note cannot qualify a baseline and a default that is later calibrated (n > 0,
dated window) becomes eligible on its data.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional, Sequence

ELIGIBILITY_RULE_VERSION = "calibration-eligibility-v1"
# Prospective rule: OFF until a dated amendment / owner decision. Nothing in the ordinary run flips this.
ELIGIBILITY_RULE_ACTIVE = False

STATUS_MISSING = "missing"
STATUS_ZERO = "zero"
STATUS_N0_DEFAULT = "n0_default"
STATUS_STALE = "stale"
STATUS_EXPIRED = "expired"
STATUS_CALIBRATED = "calibrated"
STATUS_SHARED_STATION = "shared_station"
STATUSES = (STATUS_MISSING, STATUS_ZERO, STATUS_N0_DEFAULT, STATUS_STALE, STATUS_EXPIRED,
            STATUS_CALIBRATED, STATUS_SHARED_STATION)
ELIGIBLE_STATUSES = frozenset((STATUS_CALIBRATED, STATUS_SHARED_STATION))


@dataclass(frozen=True)
class Eligibility:
    status: str
    eligible_for_tiering: bool
    reason: str
    rule_version: str = ELIGIBILITY_RULE_VERSION

    def to_dict(self) -> dict:
        return {"status": self.status, "eligible_for_tiering": bool(self.eligible_for_tiering),
                "reason": self.reason, "rule_version": self.rule_version}


def rule_active(override: Optional[bool] = None) -> bool:
    """The flag, or an explicit per-instance override (tests / a dated amendment's run configuration)."""
    return bool(ELIGIBILITY_RULE_ACTIVE if override is None else override)


def _make(status: str, reason: str) -> Eligibility:
    if status not in STATUSES:
        raise ValueError("unknown calibration status %r" % (status,))
    return Eligibility(status=status, eligible_for_tiering=status in ELIGIBLE_STATUSES, reason=reason)


def window_end(calibration_period) -> Optional[datetime]:
    """The END of a 'YYYY-MM-DD to YYYY-MM-DD' calibration window as a naive datetime, or None when unreadable
    (e.g. 'UNCALIBRATED', 'unknown', '')."""
    try:
        end_str = str(calibration_period).split(" to ")[-1].strip()
        return datetime.strptime(end_str, "%Y-%m-%d")
    except (TypeError, ValueError, AttributeError):
        return None


def baseline_age_days(calibration_period, target_date) -> Optional[int]:
    """Days between the window END and `target_date` (same arithmetic as ensemble._baseline_age_days); None when
    the window cannot be read -- the caller treats None as UNKNOWN qualification, never as fresh."""
    end = window_end(calibration_period)
    if end is None or target_date is None:
        return None
    td = target_date.replace(tzinfo=None) if getattr(target_date, "tzinfo", None) else target_date
    return (td - end).days


def classify_thd_baseline(baseline, target_date, *, max_age_days: int,
                          shared_regions: Sequence[str] = ()) -> Eligibility:
    """Classify a station THD baseline (station_baselines.StationBaseline or None) for `target_date`.

    `shared_regions` names every region the station serves on this run (from the runner's REGIONS map); a
    calibrated baseline serving more than one region is `shared_station` (eligible, disclosed)."""
    if baseline is None:
        return _make(STATUS_MISSING, "no station baseline")
    mean = _float(getattr(baseline, "mean_thd", None))
    std = _float(getattr(baseline, "std_thd", None))
    n = _int(getattr(baseline, "n_samples", None))
    if mean is None or std is None:
        return _make(STATUS_MISSING, "baseline mean/std unreadable")
    if mean <= 0.0 or std <= 0.0:
        return _make(STATUS_ZERO, "baseline mean %.6g / std %.6g: no z-score is possible" % (mean, std))
    if n is None or n <= 0:
        return _make(STATUS_N0_DEFAULT, "n_samples=%s: a default or manual estimate, not an empirical local baseline"
                     % ("unreadable" if n is None else n))
    age = baseline_age_days(getattr(baseline, "calibration_period", None), target_date)
    if age is None:
        return _make(STATUS_MISSING, "calibration window %r unreadable: qualification unknown"
                     % (getattr(baseline, "calibration_period", None),))
    if age > max_age_days:
        return _make(STATUS_STALE, "window ended %d d before the scored day (> %d d)" % (age, max_age_days))
    regions = sorted({str(r) for r in (shared_regions or ()) if r})
    if len(regions) > 1:
        return _make(STATUS_SHARED_STATION, "calibrated (n=%d, window end %d d) shared by %s"
                     % (n, age, ",".join(regions)))
    return _make(STATUS_CALIBRATED, "calibrated (n=%d, window end %d d before the scored day)" % (n, age))


def classify_fc_calibration(state: str, reasons: Iterable[str] = ()) -> Eligibility:
    """Classify the fault-correlation calibration capsule: `state` is 'admitted' (a capsule was honestly admitted
    for the day) or 'unavailable' (fault_correlation.CalibrationUnavailable, with its reasons)."""
    reasons = [str(r) for r in (reasons or ())]
    if state == "admitted":
        return _make(STATUS_CALIBRATED, "calibration capsule admitted")
    if state == "unavailable":
        if any("expired" in r.lower() for r in reasons):
            return _make(STATUS_EXPIRED, "; ".join(reasons) or "capsule expired")
        return _make(STATUS_MISSING, "; ".join(reasons) or "no admissible capsule")
    raise ValueError("fault-correlation calibration state must be 'admitted' or 'unavailable', got %r" % (state,))


def classify_lambda_geo(provenance, target_date, *, max_age_days: int) -> Eligibility:
    """Classify a Lambda_geo ratio's baseline provenance: a dict with `n_days` (baseline sample days) and
    `window_end` ('YYYY-MM-DD') plus an optional `source`; None means the runner supplied a ratio with no
    baseline record, whose qualification is unknown and therefore not eligible."""
    if not isinstance(provenance, dict):
        return _make(STATUS_MISSING, "no Lambda_geo baseline provenance")
    source = str(provenance.get("source") or "unnamed source")
    n = _int(provenance.get("n_days"))
    if n is None or n <= 0:
        return _make(STATUS_N0_DEFAULT, "%s: n_days=%s" % (source, "unreadable" if n is None else n))
    age = baseline_age_days(provenance.get("window_end"), target_date)
    if age is None:
        return _make(STATUS_MISSING, "%s: baseline window end unreadable" % source)
    if age > max_age_days:
        return _make(STATUS_STALE, "%s: window ended %d d before the scored day (> %d d)" % (source, age, max_age_days))
    return _make(STATUS_CALIBRATED, "%s: n_days=%d, window end %d d before the scored day" % (source, n, age))


def attach(method_result, eligibility: Eligibility):
    """Bind an Eligibility onto a MethodResult (the three fields the served view reads). Returns the result."""
    method_result.calibration_status = eligibility.status
    method_result.eligibility_rule_version = eligibility.rule_version
    method_result.eligible_for_tiering = bool(eligibility.eligible_for_tiering)
    method_result.eligibility_reason = eligibility.reason
    return method_result


def counts_for_tier(method_result, active: bool) -> bool:
    """Whether a component contributes to the ensemble tier / methods_available / confidence.

    Rule off: exactly the pre-rule condition (available and not frozen).
    Rule on:  additionally requires `eligible_for_tiering is True` -- an unclassified observation (None) is an
    UNKNOWN qualification and does not count."""
    base = bool(getattr(method_result, "available", False)) and not bool(getattr(method_result, "frozen", False))
    if not active:
        return base
    return base and getattr(method_result, "eligible_for_tiering", None) is True


def _float(value) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _int(value) -> Optional[int]:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None
