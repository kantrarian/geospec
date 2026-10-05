"""
calibration_eligibility.py -- PROSPECTIVE, VERSIONED calibration-eligibility rule (calibration-eligibility-v3).

ACTIVATED AND DATE-GATED (docs/AMENDMENT_2026-10-05_method_qualification.md, owner approval 2026-10-05):
`ELIGIBILITY_RULE_ACTIVE` is True and `EFFECTIVE_SCORED_DAY` is "2026-10-07", so the production daily path applies the
rule only to scored days on or after 2026-10-07 (rule_active_for_scored_day). Every scored day before the boundary keeps
the rule-off behaviour, and a rule-off output is byte-identical to the pre-rule runner (proved by
calibration_eligibility_report.py and test_calibration_eligibility_cayley_20260930.py). Saved historical scores are
never re-scored by this module. Changing either constant is a new dated amendment, never an in-place edit.

Why (codex a7d0533d item 1, 2026-09-30): a station with NO empirical local baseline (e.g. IU.SNZO: mean 0.30 /
std 0.07 / n_samples 0, calibration_period 'UNCALIBRATED') is z-scored by `ensemble.thd_to_risk_with_baseline`
exactly like a calibrated one, so a DEFAULT baseline acted as calibrated evidence in tiering (Kaikoura 2026-09-28:
THD 0.4497223 -> score 0.4097225, z 2.14, n 0). Numeric availability (a raw value exists) is distinct from
calibrated eligibility (the value can be compared against an honestly established baseline).

v2 (codex 638dd6e9 finding 1, 2026-09-30): v1 qualified a NaN mean, an infinite std and a calibration window that
ends AFTER the scored day (age -12), and coerced booleans / fractional counts into sample sizes. v2 validates its
inputs before it qualifies anything:
  * baseline statistics must be finite real numbers (bool is not a number here);
  * sample counts must be integral and non-negative (bool, fractional and negative counts are refused);
  * a window the schema carries must be COMPLETE and ORDERED (THD 'START to END'; LG window_start when supplied;
    the FC capsule's calibration_window), and it must not end after the scored day (an explicit non-future bound);
  * each method's EXISTING registered freshness policy is carried in by the caller, never invented here:
      seismic_thd       ensemble.MAX_BASELINE_AGE_DAYS (incident 2026-07-31 staleness guard)
      fault_correlation the admitted capsule's own valid_through, and the loader's registered embargo
                        (fault_correlation.load_calibration_capsule's `embargo_days` default)
      lambda_geo        NO registered baseline-age bound exists in the runner; `max_age_days=None` means
                        "unregistered" and the observation is NOT eligible (NO_REGISTERED_FRESHNESS_POLICY) until a
                        dated amendment registers one. v1 borrowed seismic_thd's 50 d for lambda_geo; v2 does not.
Every refusal carries a typed code ("CODE: detail"); an unreadable, unverifiable or missing input is never eligible.

v3 (codex 0106 / be9c2a19 activation prerequisites, 2026-10-01): the R3 30-day lag is RE-CHECKED from a
structured calibration date instead of being assumed. A baseline that is otherwise fresh must also carry the day
it was calibrated, and its window must end at least the REGISTERED lag before that day:
  seismic_thd   StationBaseline.calibration_date (set from the dated recal file name) against
                run_thd_recal.EXCLUDE_RECENT_DAYS, read from that module, not restated here;
  lambda_geo    provenance["calibrated_on"] against a lag the caller registers; the runner registers none
                (ensemble.LAMBDA_GEO_BASELINE_MIN_LAG_DAYS = None), so after a future freshness bound the ratio
                would still be NOT eligible (NO_REGISTERED_LAG_POLICY) until a dated amendment registers both.
The calibration day must also not be after the scored day (a later recal cannot qualify an earlier score), so
together the two checks guarantee the window ends at least the registered lag before the scored day.
Precedence is unchanged for every v2 outcome: unreadable / future / stale are decided first, and the lag check
can only move an otherwise calibrated baseline to `missing` (CALIBRATION_DATE_UNKNOWN,
CALIBRATED_AFTER_SCORED_DAY or LAG_NOT_HONORED).

What the rule does when ACTIVE: every method observation carries `calibration_status`,
`eligibility_rule_version` and `eligible_for_tiering`; raw values, scores and notes stay visible exactly as
today; the ENSEMBLE tier, `methods_available`, effective weights and confidence count ONLY eligible observations.
An observation whose qualification is unknown is never qualified.

Statuses (exactly these seven):
  missing        no baseline / calibration record, or an input that is unreadable, invalid, future-dated or not
                 verifiable under a registered policy
  zero           a baseline whose (finite) mean or std is <= 0 (the runner already falls back to absolute thresholds)
  n0_default     a baseline with n_samples == 0 -- a default or manual estimate, not an empirical local baseline
  stale          a baseline whose window ended more than the registered max age before the scored date
  expired        a calibration capsule past its valid_through (fault_correlation)
  calibrated     an empirical baseline whose inputs validate and whose window is fresh under its registered policy
  shared_station a calibrated baseline serving more than one region (the support is shared, and eligible)
Only `calibrated` and `shared_station` are eligible for tiering.

The classifier reads STRUCTURED fields (n_samples, the statistics, the parsed window, the capsule dates), never the
free-text notes, so relabelling a note cannot qualify a baseline and a default that is later calibrated (n > 0,
dated window) becomes eligible on its data.
"""
from __future__ import annotations

import inspect
import math
import numbers
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Optional, Sequence, Tuple

ELIGIBILITY_RULE_VERSION = "calibration-eligibility-v3"
# Set True by the 2026-10-05 amendment (owner decision); applied only from EFFECTIVE_SCORED_DAY. Nothing in the
# ordinary run flips this.
ELIGIBILITY_RULE_ACTIVE = True
# The amendment's effective scored-day boundary (ISO date, e.g. "2026-10-12"). None = UNSET. The production daily path
# applies the rule to scored day D only when ELIGIBILITY_RULE_ACTIVE AND D >= this boundary, so a replay of an earlier
# day keeps the rule-off behaviour and never rescores issued history. An ACTIVE rule with an UNSET boundary refuses.
# Both constants change together, in ONE reviewed commit.
EFFECTIVE_SCORED_DAY = "2026-10-07"

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

# Typed reason codes (the first token of every reason).
NO_BASELINE = "NO_BASELINE"
NO_PROVENANCE = "NO_PROVENANCE"
STATISTIC_UNREADABLE = "STATISTIC_UNREADABLE"
NON_FINITE_STATISTIC = "NON_FINITE_STATISTIC"
NON_POSITIVE_STATISTIC = "NON_POSITIVE_STATISTIC"
INVALID_SAMPLE_COUNT = "INVALID_SAMPLE_COUNT"
ZERO_SAMPLE_DEFAULT = "ZERO_SAMPLE_DEFAULT"
WINDOW_UNREADABLE = "WINDOW_UNREADABLE"
WINDOW_INCOMPLETE = "WINDOW_INCOMPLETE"
WINDOW_REVERSED = "WINDOW_REVERSED"
FUTURE_WINDOW_END = "FUTURE_WINDOW_END"
TARGET_DATE_UNKNOWN = "TARGET_DATE_UNKNOWN"
WINDOW_STALE = "WINDOW_STALE"
NO_REGISTERED_FRESHNESS_POLICY = "NO_REGISTERED_FRESHNESS_POLICY"
NO_ADMISSIBLE_CAPSULE = "NO_ADMISSIBLE_CAPSULE"
CAPSULE_PAST_VALID_THROUGH = "CAPSULE_PAST_VALID_THROUGH"
CAPSULE_NOT_SUPPLIED = "CAPSULE_NOT_SUPPLIED"
CAPSULE_FIELDS_UNREADABLE = "CAPSULE_FIELDS_UNREADABLE"
CAPSULE_INSIDE_EMBARGO = "CAPSULE_INSIDE_EMBARGO"
EMBARGO_POLICY_UNREADABLE = "EMBARGO_POLICY_UNREADABLE"
CALIBRATION_DATE_UNKNOWN = "CALIBRATION_DATE_UNKNOWN"
LAG_NOT_HONORED = "LAG_NOT_HONORED"
CALIBRATED_AFTER_SCORED_DAY = "CALIBRATED_AFTER_SCORED_DAY"
NO_REGISTERED_LAG_POLICY = "NO_REGISTERED_LAG_POLICY"
CALIBRATED = "CALIBRATED"
SHARED_SUPPORT = "SHARED_SUPPORT"

# fault_correlation.load_calibration_capsule's OWN refusal text for a capsule past its valid_through
# ("scored day ... past valid_through ... (STALE)"). Matched against the writer's wording, which a test reads
# from the loader source; v1 matched the word "expired", which the loader never writes.
FC_PAST_VALID_THROUGH_PHRASE = "past valid_through"


@dataclass(frozen=True)
class Eligibility:
    status: str
    eligible_for_tiering: bool
    reason: str
    code: str = ""
    rule_version: str = ELIGIBILITY_RULE_VERSION

    def to_dict(self) -> dict:
        return {"status": self.status, "eligible_for_tiering": bool(self.eligible_for_tiering),
                "code": self.code, "reason": self.reason, "rule_version": self.rule_version}


def rule_active(override: Optional[bool] = None) -> bool:
    """The flag, or an explicit per-instance override (tests / a dated amendment's run configuration)."""
    return bool(ELIGIBILITY_RULE_ACTIVE if override is None else override)


def _iso_day(value, name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            parsed = None
        if parsed is not None and parsed.isoformat() == value:
            return parsed
    raise ValueError("%s must be an ISO calendar date (YYYY-MM-DD), a date or a datetime, got %r" % (name, value))


def rule_active_for_scored_day(scored_day) -> bool:
    """The rule for ONE scored day on the production daily path: on only when ELIGIBILITY_RULE_ACTIVE and the day is
    on or after EFFECTIVE_SCORED_DAY. Before the boundary the legacy (rule-off) behaviour holds, so replaying an issued
    day cannot rescore it. An active rule without a declared boundary refuses rather than guessing one."""
    day = _iso_day(scored_day, "scored_day")
    if not ELIGIBILITY_RULE_ACTIVE:
        return False
    if EFFECTIVE_SCORED_DAY is None:
        raise ValueError("ELIGIBILITY_RULE_ACTIVE requires EFFECTIVE_SCORED_DAY: an active rule must declare its "
                         "effective scored-day boundary")
    return day >= _iso_day(EFFECTIVE_SCORED_DAY, "EFFECTIVE_SCORED_DAY")


def _make(status: str, code: str, detail: str = "") -> Eligibility:
    if status not in STATUSES:
        raise ValueError("unknown calibration status %r" % (status,))
    reason = "%s: %s" % (code, detail) if detail else code
    return Eligibility(status=status, eligible_for_tiering=status in ELIGIBLE_STATUSES, reason=reason, code=code)


# ---------------------------------------------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------------------------------------------

def _statistic(value) -> Tuple[Optional[float], Optional[str]]:
    """(finite float, None) or (None, problem). bool is not a statistic; NaN and +/-infinity are not finite."""
    if value is None:
        return None, "absent"
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None, "not a number (%s)" % type(value).__name__
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None, "not representable as a float"
    if not math.isfinite(number):
        return None, "non-finite (%r)" % (number,)
    return number, None


def _count(value) -> Tuple[Optional[int], Optional[str]]:
    """(non-negative int, None) or (None, problem): bool, fractional, non-finite and negative counts are refused;
    an integral float (e.g. 90.0 from a JSON writer) is accepted as the integer it states."""
    if value is None:
        return None, "absent"
    if isinstance(value, bool):
        return None, "boolean is not a count"
    if isinstance(value, numbers.Integral):
        number = int(value)
    elif isinstance(value, numbers.Real):
        as_float = float(value)
        if not math.isfinite(as_float):
            return None, "non-finite (%r)" % (as_float,)
        if not as_float.is_integer():
            return None, "fractional (%r)" % (as_float,)
        number = int(as_float)
    else:
        return None, "not a number (%s)" % type(value).__name__
    if number < 0:
        return None, "negative (%d)" % number
    return number, None


def _parse_day(value) -> Optional[date]:
    """A calendar day from 'YYYY-MM-DD', a date or a datetime (a datetime keeps its own calendar day: the label,
    not a conversion -- the same convention as ensemble._baseline_age_days). None when unreadable."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.strptime(value.strip(), "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def _parse_period(period) -> Tuple[Optional[date], Optional[date], Optional[Tuple[str, str]]]:
    """A COMPLETE, ORDERED 'START to END' window, or (None, None, (code, detail))."""
    if not isinstance(period, str):
        return None, None, (WINDOW_UNREADABLE, "calibration window %r is not text" % (period,))
    parts = [p.strip() for p in period.split(" to ")]
    if len(parts) != 2:
        if len(parts) == 1 and _parse_day(parts[0]) is not None:
            return None, None, (WINDOW_INCOMPLETE, "calibration window %r has an end but no start" % (period,))
        return None, None, (WINDOW_UNREADABLE, "calibration window %r unreadable: qualification unknown" % (period,))
    start, end = _parse_day(parts[0]), _parse_day(parts[1])
    if start is None or end is None:
        return None, None, (WINDOW_UNREADABLE, "calibration window %r unreadable: qualification unknown" % (period,))
    if start > end:
        return None, None, (WINDOW_REVERSED, "calibration window %r starts after it ends" % (period,))
    return start, end, None


def _registered_policy(value, name: str, *, allow_none: bool) -> Optional[int]:
    """A caller-supplied REGISTERED day count. A malformed policy is a caller defect and raises; None is accepted
    only where the method has no registered policy (lambda_geo)."""
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) < 0:
        raise ValueError("%s must be a registered non-negative integer day count, got %r" % (name, value))
    return int(value)


def registered_default(func, parameter: str):
    """The default value a callable REGISTERS for `parameter` (e.g. fault_correlation.load_calibration_capsule's
    `embargo_days`), read from its signature; None when the callable or the parameter cannot be read. Used so the
    rule carries a method's existing policy instead of restating a number."""
    try:
        found = inspect.signature(func).parameters.get(parameter)
    except (TypeError, ValueError):
        return None
    if found is None or found.default is inspect.Parameter.empty:
        return None
    return found.default


# ---------------------------------------------------------------------------------------------------------------
# Helpers kept from v1 (read by tests and callers)
# ---------------------------------------------------------------------------------------------------------------

def registered_constant(module_name: str, attribute: str) -> int:
    """A day count a module REGISTERS as a constant (e.g. run_thd_recal.EXCLUDE_RECENT_DAYS, the R3 lag), read from
    the module itself rather than restated. Imported only when the rule is active. A missing or malformed value
    is a wiring defect and raises."""
    import importlib
    module = importlib.import_module(module_name)
    if not hasattr(module, attribute):
        raise ValueError("%s registers no %s" % (module_name, attribute))
    return _registered_policy(getattr(module, attribute), "%s.%s" % (module_name, attribute), allow_none=False)


def _lag_check(window_end_day: date, calibrated_on, target_day: date, min_lag: int,
               label: str) -> Optional[Eligibility]:
    """None when the baseline was calibrated on or before the scored day AND its window ends at least `min_lag`
    days before that calibration day; else the typed refusal."""
    calibrated = _parse_day(calibrated_on)
    if calibrated is None:
        return _make(STATUS_MISSING, CALIBRATION_DATE_UNKNOWN,
                     "%scalibration date %r unknown: the registered %d d lag cannot be re-checked"
                     % (label, calibrated_on, min_lag))
    if calibrated > target_day:
        return _make(STATUS_MISSING, CALIBRATED_AFTER_SCORED_DAY,
                     "%scalibrated %s, %d d after the scored day %s" % (label, calibrated,
                                                                      (calibrated - target_day).days, target_day))
    lag = (calibrated - window_end_day).days
    if lag < min_lag:
        return _make(STATUS_MISSING, LAG_NOT_HONORED,
                     "%swindow ends %s, %d d before calibration on %s (< registered lag %d d)"
                     % (label, window_end_day, lag, calibrated, min_lag))
    return None


def window_end(calibration_period) -> Optional[datetime]:
    """The END of a 'YYYY-MM-DD to YYYY-MM-DD' calibration window as a naive datetime, or None when unreadable
    (e.g. 'UNCALIBRATED', 'unknown', ''). Same reading as ensemble._baseline_age_days; the classifier itself uses
    the stricter complete/ordered `_parse_period`."""
    try:
        end_str = str(calibration_period).split(" to ")[-1].strip()
        return datetime.strptime(end_str, "%Y-%m-%d")
    except (TypeError, ValueError, AttributeError):
        return None


def baseline_age_days(calibration_period, target_date) -> Optional[int]:
    """Days between the window END and `target_date` (same arithmetic as ensemble._baseline_age_days); None when
    the window cannot be read -- the caller treats None as UNKNOWN qualification, never as fresh."""
    end = window_end(calibration_period)
    target = _parse_day(target_date)
    if end is None or target is None:
        return None
    return (target - end.date()).days


# ---------------------------------------------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------------------------------------------

def classify_thd_baseline(baseline, target_date, *, max_age_days: int, min_lag_days: int,
                          shared_regions: Sequence[str] = ()) -> Eligibility:
    """Classify a station THD baseline (station_baselines.StationBaseline or None) for `target_date`.

    `max_age_days` is the REGISTERED seismic_thd policy (ensemble.MAX_BASELINE_AGE_DAYS); `min_lag_days` the
    REGISTERED R3 lag (run_thd_recal.EXCLUDE_RECENT_DAYS), re-checked against `baseline.calibration_date` (v3).
    `shared_regions` names
    every region the station serves on this run (from the runner's REGIONS map); a calibrated baseline serving more
    than one region is `shared_station` (eligible, disclosed)."""
    max_age = _registered_policy(max_age_days, "max_age_days", allow_none=False)
    min_lag = _registered_policy(min_lag_days, "min_lag_days", allow_none=False)
    if baseline is None:
        return _make(STATUS_MISSING, NO_BASELINE, "no station baseline")
    mean, mean_problem = _statistic(getattr(baseline, "mean_thd", None))
    std, std_problem = _statistic(getattr(baseline, "std_thd", None))
    if mean_problem or std_problem:
        code = NON_FINITE_STATISTIC if "non-finite" in "%s %s" % (mean_problem, std_problem) else STATISTIC_UNREADABLE
        return _make(STATUS_MISSING, code, "baseline mean %s / std %s" % (mean_problem or "ok", std_problem or "ok"))
    if mean <= 0.0 or std <= 0.0:
        return _make(STATUS_ZERO, NON_POSITIVE_STATISTIC,
                     "baseline mean %.6g / std %.6g: no z-score is possible" % (mean, std))
    raw_n = getattr(baseline, "n_samples", None)
    n, n_problem = _count(raw_n)
    if n_problem:
        return _make(STATUS_MISSING, INVALID_SAMPLE_COUNT, "n_samples=%r: %s" % (raw_n, n_problem))
    if n == 0:
        return _make(STATUS_N0_DEFAULT, ZERO_SAMPLE_DEFAULT,
                     "n_samples=0: a default or manual estimate, not an empirical local baseline")
    start, end, window_problem = _parse_period(getattr(baseline, "calibration_period", None))
    if window_problem:
        return _make(STATUS_MISSING, *window_problem)
    target = _parse_day(target_date)
    if target is None:
        return _make(STATUS_MISSING, TARGET_DATE_UNKNOWN, "scored day %r unreadable" % (target_date,))
    age = (target - end).days
    if age < 0:
        return _make(STATUS_MISSING, FUTURE_WINDOW_END,
                     "window ends %s, %d d after the scored day %s" % (end, -age, target))
    if age > max_age:
        return _make(STATUS_STALE, WINDOW_STALE, "window ended %d d before the scored day (> %d d)" % (age, max_age))
    lag_refusal = _lag_check(end, getattr(baseline, "calibration_date", None), target, min_lag, "")
    if lag_refusal is not None:
        return lag_refusal
    regions = sorted({str(r) for r in (shared_regions or ()) if r})
    if len(regions) > 1:
        return _make(STATUS_SHARED_STATION, SHARED_SUPPORT, "calibrated (n=%d, window %s..%s, end %d d) shared by %s"
                     % (n, start, end, age, ",".join(regions)))
    return _make(STATUS_CALIBRATED, CALIBRATED,
                 "n=%d, window %s..%s, end %d d before the scored day" % (n, start, end, age))


def classify_fc_calibration(state: str, reasons: Iterable[str] = (), *, capsule=None, scored_day=None,
                            embargo_days=None) -> Eligibility:
    """Classify the fault-correlation calibration capsule.

    `state` 'unavailable': the loader refused (fault_correlation.CalibrationUnavailable, with its reasons); a
    refusal in the loader's own past-valid_through wording is `expired`, anything else `missing`.
    `state` 'admitted': the capsule the loader admitted is RE-CHECKED against its own registered policy -- a
    complete, ordered calibration_window that does not end after the scored day, the scored day on or before
    valid_through, and the loader's registered embargo (`embargo_days`, read by the caller from
    load_calibration_capsule's signature). A capsule, scored day or embargo that cannot be read is not eligible."""
    reasons = [str(r) for r in (reasons or ())]
    if state == "unavailable":
        if any(FC_PAST_VALID_THROUGH_PHRASE in r for r in reasons):
            return _make(STATUS_EXPIRED, CAPSULE_PAST_VALID_THROUGH, "; ".join(reasons))
        return _make(STATUS_MISSING, NO_ADMISSIBLE_CAPSULE, "; ".join(reasons) or "no admissible capsule")
    if state != "admitted":
        raise ValueError("fault-correlation calibration state must be 'admitted' or 'unavailable', got %r" % (state,))
    if not isinstance(capsule, dict):
        return _make(STATUS_MISSING, CAPSULE_NOT_SUPPLIED, "admitted, but no capsule was supplied to verify")
    if isinstance(embargo_days, bool) or not isinstance(embargo_days, numbers.Integral) or int(embargo_days) < 0:
        return _make(STATUS_MISSING, EMBARGO_POLICY_UNREADABLE,
                     "registered embargo %r unreadable: qualification unknown" % (embargo_days,))
    window = capsule.get("calibration_window")
    if not (isinstance(window, dict) and set(window) == {"start", "end"}):
        return _make(STATUS_MISSING, CAPSULE_FIELDS_UNREADABLE, "calibration_window must be exactly {start, end}")
    start, end = _parse_day(window.get("start")), _parse_day(window.get("end"))
    valid_through = _parse_day(capsule.get("valid_through"))
    if start is None or end is None or valid_through is None:
        return _make(STATUS_MISSING, CAPSULE_FIELDS_UNREADABLE,
                     "window %r / valid_through %r unreadable" % (window, capsule.get("valid_through")))
    if start > end:
        return _make(STATUS_MISSING, WINDOW_REVERSED, "calibration window %s..%s starts after it ends" % (start, end))
    target = _parse_day(scored_day)
    if target is None:
        return _make(STATUS_MISSING, TARGET_DATE_UNKNOWN, "scored day %r unreadable" % (scored_day,))
    if end > target:
        return _make(STATUS_MISSING, FUTURE_WINDOW_END, "window ends %s, after the scored day %s" % (end, target))
    if target > valid_through:
        return _make(STATUS_EXPIRED, CAPSULE_PAST_VALID_THROUGH,
                     "scored day %s past valid_through %s" % (target, valid_through))
    lag = (target - end).days
    if lag < int(embargo_days):
        return _make(STATUS_MISSING, CAPSULE_INSIDE_EMBARGO,
                     "window ends %d d before the scored day (< registered embargo %d d)" % (lag, int(embargo_days)))
    return _make(STATUS_CALIBRATED, CALIBRATED, "capsule window %s..%s, valid_through %s, lag %d d >= embargo %d d"
                 % (start, end, valid_through, lag, int(embargo_days)))


def classify_lambda_geo(provenance, target_date, *, max_age_days: Optional[int],
                        min_lag_days: Optional[int]) -> Eligibility:
    """Classify a Lambda_geo ratio's baseline provenance: a dict with `n_days` (baseline sample days) and
    `window_end` ('YYYY-MM-DD'), optionally `window_start` (checked complete and ordered when supplied) and
    `source`. None means the runner supplied a ratio with no baseline record (qualification unknown).

    `max_age_days` is the REGISTERED lambda_geo baseline-age bound; the runner registers none, so the ensemble
    passes None and a ratio whose inputs otherwise validate is `missing` (NO_REGISTERED_FRESHNESS_POLICY). The
    argument is keyword-only and has no default: a caller states the policy or states that there is none.
    v3: `min_lag_days` is the REGISTERED lag between the window end and `provenance["calibrated_on"]`; None means
    unregistered, and a ratio that passes freshness is then still `missing` (NO_REGISTERED_LAG_POLICY)."""
    max_age = _registered_policy(max_age_days, "max_age_days", allow_none=True)
    min_lag = _registered_policy(min_lag_days, "min_lag_days", allow_none=True)
    if not isinstance(provenance, dict):
        return _make(STATUS_MISSING, NO_PROVENANCE, "no Lambda_geo baseline provenance")
    source = str(provenance.get("source") or "unnamed source")
    raw_n = provenance.get("n_days")
    n, n_problem = _count(raw_n)
    if n_problem:
        return _make(STATUS_MISSING, INVALID_SAMPLE_COUNT, "%s: n_days=%r: %s" % (source, raw_n, n_problem))
    if n == 0:
        return _make(STATUS_N0_DEFAULT, ZERO_SAMPLE_DEFAULT, "%s: n_days=0" % source)
    end = _parse_day(provenance.get("window_end"))
    if end is None:
        return _make(STATUS_MISSING, WINDOW_UNREADABLE,
                     "%s: window_end %r unreadable" % (source, provenance.get("window_end")))
    if "window_start" in provenance:
        start = _parse_day(provenance.get("window_start"))
        if start is None:
            return _make(STATUS_MISSING, WINDOW_UNREADABLE,
                         "%s: window_start %r unreadable" % (source, provenance.get("window_start")))
        if start > end:
            return _make(STATUS_MISSING, WINDOW_REVERSED, "%s: window %s..%s starts after it ends" % (source, start, end))
    target = _parse_day(target_date)
    if target is None:
        return _make(STATUS_MISSING, TARGET_DATE_UNKNOWN, "%s: scored day %r unreadable" % (source, target_date))
    age = (target - end).days
    if age < 0:
        return _make(STATUS_MISSING, FUTURE_WINDOW_END,
                     "%s: window ends %s, %d d after the scored day %s" % (source, end, -age, target))
    if max_age is None:
        return _make(STATUS_MISSING, NO_REGISTERED_FRESHNESS_POLICY,
                     "%s: lambda_geo has no registered baseline-age bound; qualification unknown" % source)
    if age > max_age:
        return _make(STATUS_STALE, WINDOW_STALE,
                     "%s: window ended %d d before the scored day (> %d d)" % (source, age, max_age))
    if min_lag is None:
        return _make(STATUS_MISSING, NO_REGISTERED_LAG_POLICY,
                     "%s: lambda_geo has no registered calibration lag; qualification unknown" % source)
    lag_refusal = _lag_check(end, provenance.get("calibrated_on"), target, min_lag, "%s: " % source)
    if lag_refusal is not None:
        return lag_refusal
    return _make(STATUS_CALIBRATED, CALIBRATED, "%s: n_days=%d, window end %d d before the scored day, calibrated %s"
                 % (source, n, age, provenance.get("calibrated_on")))


def attach(method_result, eligibility: Eligibility):
    """Bind an Eligibility onto a MethodResult (the fields the served view reads). Returns the result."""
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
