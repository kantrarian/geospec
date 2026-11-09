"""Offline LG integration policy. No production registration or default activation.

The producer must bind these fields to retained inputs; this consumer does not
invent identities or reconstruct absent observations. Historical issued scores
are untouched. The existing amplitude-to-score mapping is deliberately unchanged.
"""
from dataclasses import dataclass, replace
from datetime import date, datetime
import math
import re

import calibration_eligibility as CE

VERSION = 'lg-integration-candidate-v1'
STATISTIC = 'mean-of-daily-max-over-rule-epochs'


@dataclass(frozen=True)
class CandidatePolicy:
    effective_scored_day: str
    baseline_max_age_days: int = 45
    calibration_min_lag_days: int = 30
    observation_max_age_days: int = 45
    min_baseline_samples: int = 30

    def __post_init__(self):
        if _day(self.effective_scored_day) is None:
            raise ValueError('explicit candidate effective_scored_day required')
        for key in ('baseline_max_age_days', 'calibration_min_lag_days',
                    'observation_max_age_days', 'min_baseline_samples'):
            value = getattr(self, key)
            if type(value) is not int or value < (1 if key == 'min_baseline_samples' else 0):
                raise ValueError('invalid candidate policy ' + key)
        if self.baseline_max_age_days < self.calibration_min_lag_days:
            raise ValueError('baseline age must accommodate the calibration lag')


def _day(value):
    if isinstance(value, datetime):
        return value.date()
    if type(value) is date:
        return value
    if type(value) is str:
        try:
            parsed = date.fromisoformat(value)
            return parsed if parsed.isoformat() == value else None
        except ValueError:
            pass
    return None


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _hash(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def classify(provenance, target_date, ratio, policy):
    """Accept a matched, as-of LG input; refusal never changes its measured value."""
    def refuse(code, detail):
        return CE.Eligibility('missing', False, code + ': ' + detail, code, VERSION)

    if not isinstance(policy, CandidatePolicy):
        raise TypeError('explicit CandidatePolicy required')
    target = _day(target_date)
    if target is None:
        return refuse('TARGET_DATE_UNKNOWN', 'invalid scored day')
    if target < _day(policy.effective_scored_day):
        return refuse('LG_BEFORE_CANDIDATE_BOUNDARY', 'no retrospective qualification')
    prior = CE.classify_lambda_geo(provenance, target,
                                  max_age_days=policy.baseline_max_age_days,
                                  min_lag_days=policy.calibration_min_lag_days)
    if not prior.eligible_for_tiering:
        return replace(prior, rule_version=VERSION)
    if provenance['n_days'] < policy.min_baseline_samples:
        return refuse('LG_INSUFFICIENT_BASELINE_SAMPLES', 'too few accepted epochs')
    if not _hash(provenance.get('calibration_sha256')):
        return refuse('LG_BASELINE_BYTES_UNBOUND', 'retain the selected calibration digest')
    if provenance.get('statistic') != STATISTIC:
        return refuse('LG_BASELINE_STATISTIC_UNKNOWN', 'matched support rule and valid zeros required')
    denominator = provenance.get('baseline_value')
    if not _finite(denominator) or denominator <= 0:
        return refuse('LG_DENOMINATOR_UNAVAILABLE', 'finite positive regional baseline required')
    measurement = provenance.get('measurement')
    if not isinstance(measurement, dict):
        return refuse('LG_MEASUREMENT_PROVENANCE_UNKNOWN', 'retain the selected GNSS observation')
    live_id, baseline_id = measurement.get('operator_identity'), provenance.get('operator_identity')
    if not _hash(live_id) or not _hash(baseline_id):
        return refuse('LG_OPERATOR_IDENTITY_UNBOUND', 'both carrier/operator identities required')
    if live_id != baseline_id:
        return refuse('LG_OPERATOR_MISMATCH', 'numerator and denominator use different support')
    if not _hash(measurement.get('input_sha256')):
        return refuse('LG_INPUT_BYTES_UNBOUND', 'retain the input manifest digest')
    observed, through = _day(measurement.get('observation_day')), _day(measurement.get('support_through_day'))
    if observed is None or through is None:
        return refuse('LG_OBSERVATION_TIME_UNKNOWN', 'observation and smoothing support dates required')
    if observed > through or through > target:
        return refuse('LG_FUTURE_SUPPORT', 'no support after the scored day')
    if (target - observed).days > policy.observation_max_age_days:
        return refuse('LG_OBSERVATION_STALE', 'selected observation exceeds candidate age bound')
    if measurement.get('selection_status') != 'OK':
        return refuse('LG_SUPPORT_SELECTION_UNAVAILABLE', 'full-support selection did not succeed')
    numerator = measurement.get('lambda_max')
    if not _finite(numerator) or numerator < 0:
        return refuse('LG_NUMERATOR_UNAVAILABLE', 'finite nonnegative numerator required; zero is valid')
    quotient = numerator / denominator
    retained = measurement.get('unclamped_ratio')
    if (not math.isfinite(quotient) or not _finite(retained) or
            not math.isclose(quotient, retained, rel_tol=1e-12, abs_tol=0.0) or
            not _finite(ratio) or not math.isclose(ratio, max(0.1, min(50.0, quotient)),
                                                 rel_tol=1e-12, abs_tol=0.0)):
        return refuse('LG_RATIO_LINEAGE_MISMATCH', 'ratio must reproduce from the retained numerator and baseline')
    return CE.Eligibility('calibrated', True,
                          'CALIBRATED: matched LG input under explicit offline candidate policy',
                          'CALIBRATED', VERSION)
