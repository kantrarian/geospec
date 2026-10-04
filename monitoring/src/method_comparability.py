"""
method_comparability.py -- PROSPECTIVE comparison contract for the ensemble (method-comparability-v1).

NOT ACTIVATED. Everything here is emitted ONLY while the calibration-eligibility rule is active
(calibration_eligibility.rule_active), so every rule-OFF output stays byte-identical to the issued reports. Issued
history is never re-scored or rewritten.

Why (owner "proceed with your recs", codex docs/geospec/METHOD_QUALIFICATION_DELIVERY_PLAN_CODEX_20261004.md, M1):
`combined_risk` is a weighted mean over the methods that QUALIFIED for the tier. When two regions qualify different
method sets, their numbers are conditional means over different evidence, not the same calibrated risk. On
2026-10-02 turkey_kahramanmaras (THD alone, 0.8115) was the report's `max_risk_region` ahead of istanbul_marmara
(LG+THD, 0.3478), although both rest on the same IU.ANTO THD score.

Contract:
  1. Qualification is decided ONLY by calibration_eligibility.counts_for_tier (input validity, provenance,
     calibration, support, freeze) -- never by a score's value or its agreement with another method. A qualified
     zero keeps its weight (VALID_ZERO); an unqualified value is excluded however high or low it is.
  2. Every component carries exactly one qualification state (STATES below) with the typed reason the rule gave.
     An observation whose qualification is unknown is never qualified.
  3. Each region carries its method set: included methods (stable order and label), excluded methods with state
     and reason, nominal and effective weights, weighted coverage, valid-zero methods, the support each included
     method rests on (a THD station and every region it serves: shared support is not independent regional
     confirmation) and a comparability key.
  4. `combined_risk` is a CONDITIONAL MEAN OVER THE INCLUDED METHODS. Two regions' values are comparable only when
     their comparability keys are equal: the same method set, under the same rule and contract versions, with the
     same calibration class per method. A cross-region maximum is reported per comparability group; a single
     maximum across groups is withheld (save_results).
  5. Persistence counts only prior days issued under the same REGIME (rule version + contract version). A regime
     change is recorded as a transition and is not carried across. Method-set and measurement-support changes
     start a new confirmation sequence. This remains an offline prospective candidate.
"""
from __future__ import annotations

import hashlib
import inspect
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import calibration_eligibility as CE

COMPARISON_CONTRACT_VERSION = "method-comparability-v1"

METHOD_ORDER = ("lambda_geo", "fault_correlation", "seismic_thd")
METHOD_LABELS = {"lambda_geo": "LG", "fault_correlation": "FC", "seismic_thd": "THD"}

# Qualification states (exactly these nine).
VALID = "VALID"                      # qualified, nonzero score
VALID_ZERO = "VALID_ZERO"            # qualified, score exactly 0.0: keeps its weight
UNAVAILABLE = "UNAVAILABLE"          # no value was computed (no data / no admissible input)
FROZEN = "FROZEN"                    # incident freeze: computed, excluded from the tier
EXPIRED = "EXPIRED"                  # calibration past its registered validity (FC capsule valid_through)
STALE = "STALE"                      # baseline window older than the registered max age
INVALID_DEFAULT = "INVALID_DEFAULT"  # a default or invalid baseline (n_samples 0, non-positive statistics)
NOT_QUALIFIED = "NOT_QUALIFIED"      # judged and refused (no provenance, no registered policy, lag, future window...)
UNKNOWN = "UNKNOWN"                  # a value exists but no rule judged it: never qualified while the rule is on
STATES = (VALID, VALID_ZERO, UNAVAILABLE, FROZEN, EXPIRED, STALE, INVALID_DEFAULT, NOT_QUALIFIED, UNKNOWN)
INCLUDED_STATES = frozenset((VALID, VALID_ZERO))

_REFUSED_STATUS_STATE = {
    CE.STATUS_EXPIRED: EXPIRED,
    CE.STATUS_STALE: STALE,
    CE.STATUS_N0_DEFAULT: INVALID_DEFAULT,
    CE.STATUS_ZERO: INVALID_DEFAULT,
}

RISK_BASIS = "CONDITIONAL_MEAN_OVER_INCLUDED_METHODS"
# A cross-region maximum is a DESCRIPTIVE method-score statistic within one comparable group, never a calibrated
# regional risk or a ranking of hazard (codex db9a28ff finding 2).
MAXIMUM_BASIS = "DESCRIPTIVE_METHOD_SCORE_WITHIN_ONE_COMPARABLE_GROUP_NOT_CALIBRATED_REGIONAL_RISK"
UNIDENTIFIED = "UNIDENTIFIED"


def code_identity(*objects) -> str:
    """The estimator / normalization identity of the code that produced a value: each object's qualified name and
    the sha256 of its source, so a change of the code changes the identity (derived, never a hand-kept version).
    UNIDENTIFIED when any source cannot be read."""
    parts = []
    for obj in objects:
        target = getattr(obj, "__func__", obj)
        try:
            source = inspect.getsource(target)
        except (OSError, TypeError):
            return UNIDENTIFIED
        name = getattr(target, "__qualname__", getattr(target, "__name__", type(target).__name__))
        parts.append("%s@%s" % (name, hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]))
    return "+".join(parts) or UNIDENTIFIED
NO_METHODS_LABEL = "NONE"
PRE_CONTRACT_REGIME = "PRE_CONTRACT"

# Decision D-1 (owner): whether a change of comparability key between consecutive days resets persistence.
# Codex D-1 recommendation: confirmation belongs to the same evidence basis, not only the same rule version.
PERSISTENCE_RESETS_ON_METHOD_SET_CHANGE = True


class ContractViolation(RuntimeError):
    """The comparison contract and the tier disagree about a component: refuse rather than emit both."""


def _code(reason: Optional[str]) -> Optional[str]:
    if not reason:
        return None
    return reason.split(":", 1)[0].strip() or None


def qualification(component, rule_active: bool) -> Dict:
    """The one qualification state of a component, derived from the same predicate the tier uses."""
    judged = getattr(component, "eligibility_rule_version", None) is not None
    status = getattr(component, "calibration_status", None) if judged else None
    reason = getattr(component, "eligibility_reason", None) if judged else None
    available = bool(getattr(component, "available", False))
    included = CE.counts_for_tier(component, rule_active)
    if bool(getattr(component, "frozen", False)):
        state = FROZEN
    elif included:
        state = VALID_ZERO if float(component.risk_score) == 0.0 else VALID
    elif status in _REFUSED_STATUS_STATE:
        state = _REFUSED_STATUS_STATE[status]
    elif not available:
        state = UNAVAILABLE
    elif judged:
        state = NOT_QUALIFIED
    else:
        state = UNKNOWN
    if included != (state in INCLUDED_STATES):
        raise ContractViolation("%s: counts_for_tier=%s but state %s" % (component.name, included, state))
    return {
        "state": state,
        "included": included,
        "code": _code(reason),
        "reason": reason,
        "calibration_status": status,
        "available": available,
    }


def label_for(included: Sequence[str]) -> str:
    names = [n for n in METHOD_ORDER if n in included]
    return "+".join(METHOD_LABELS[n] for n in names) or NO_METHODS_LABEL


def regime_id(rule_version: Optional[str] = None, contract_version: str = COMPARISON_CONTRACT_VERSION) -> str:
    return "%s|%s" % (rule_version or CE.ELIGIBILITY_RULE_VERSION, contract_version)


def method_set(components: Mapping, weights: Mapping[str, float], rule_active: bool,
               support: Optional[Mapping[str, Dict]] = None) -> Dict:
    """The region's method-set block. `support` maps a method to {'identity': str, 'calibration_class': str,
    'shared_with': [regions]} where the compute path knows it; anything absent is recorded as UNIDENTIFIED."""
    support = dict(support or {})
    names = [n for n in METHOD_ORDER if n in components]
    unexpected = sorted(set(components) - set(METHOD_ORDER))
    if unexpected:
        raise ContractViolation("unknown methods %s" % unexpected)
    states = {n: qualification(components[n], rule_active) for n in names}
    included = [n for n in names if states[n]["included"]]
    nominal = {n: float(weights.get(n, 0.0)) for n in names}
    total_nominal = sum(nominal.values())
    included_nominal = sum(nominal[n] for n in included)
    effective = {n: nominal[n] / included_nominal for n in included} if included_nominal > 0 else {}
    coverage = included_nominal / total_nominal if total_nominal > 0 else 0.0
    support_rows = {}
    classes = []
    for n in included:
        row = dict(support.get(n) or {})
        identity = row.get("identity") or "UNIDENTIFIED"
        klass = row.get("calibration_class") or "UNIDENTIFIED"
        estimator = row.get("estimator") or UNIDENTIFIED
        shared = sorted(set(row.get("shared_with") or ()))
        support_rows[n] = {"identity": identity, "calibration_class": klass, "estimator": estimator,
                           "shared_with": shared,
                           "independent_of_other_regions": (len(shared) == 1 if shared else None)}
        classes.append("%s:%s" % (METHOD_LABELS[n], klass))
    label = label_for(included)
    # Comparison identity: the method set, the rule/contract regime, the calibration class, the ESTIMATOR /
    # normalization code identity and the EFFECTIVE WEIGHTS of each included method (codex db9a28ff finding 2):
    # identical method names under different weights or operator versions never share a group.
    estimators = ",".join("%s:%s" % (METHOD_LABELS[n], support_rows[n]["estimator"]) for n in included)
    weights_key = ",".join("%s=%r" % (METHOD_LABELS[n], effective[n]) for n in included)
    key = "%s|%s|%s|%s|%s" % (label, regime_id(), ",".join(classes) or NO_METHODS_LABEL,
                             estimators or NO_METHODS_LABEL, weights_key or NO_METHODS_LABEL)
    return {
        "contract_version": COMPARISON_CONTRACT_VERSION,
        "rule_version": CE.ELIGIBILITY_RULE_VERSION,
        "regime": regime_id(),
        "label": label,
        "included": included,
        "excluded": {n: {k: states[n][k] for k in ("state", "code", "reason")}
                     for n in names if not states[n]["included"]},
        "states": {n: states[n]["state"] for n in names},
        "valid_zero_methods": [n for n in included if states[n]["state"] == VALID_ZERO],
        "nominal_weights": nominal,
        "effective_weights": effective,
        "weighted_coverage": coverage,
        "support": support_rows,
        "comparability_key": key,
        "comparability_complete": bool(included) and all(
            r["calibration_class"] != "UNIDENTIFIED" and r["identity"] != "UNIDENTIFIED"
            and r["estimator"] != UNIDENTIFIED
            for r in support_rows.values()),
        "risk_basis": RISK_BASIS,
    }


def comparison_groups(region_rows: Mapping[str, Mapping]) -> Dict[str, Dict]:
    """Group regions by comparability key; the maximum is reported only WITHIN a group. DEGRADED regions (no
    qualified method) form no group: there is no risk to compare. An exact tie lists every tied region in
    `max_risk_regions` and leaves `max_risk_region` None (two regions on one shared station tie by construction)."""
    groups: Dict[str, Dict] = {}
    for region in sorted(region_rows):
        row = region_rows[region]
        ms = row.get("method_set")
        if not ms or not ms.get("included") or not ms.get("comparability_complete"):
            continue
        g = groups.setdefault(ms["comparability_key"], {"label": ms["label"], "maximum_basis": MAXIMUM_BASIS,
                                                         "regions": [],
                                                         "max_risk_region": None, "max_risk": None,
                                                         "max_risk_regions": []})
        g["regions"].append(region)
        risk = float(row.get("combined_risk", 0.0))
        if g["max_risk"] is None or risk > g["max_risk"]:
            g["max_risk"], g["max_risk_region"], g["max_risk_regions"] = risk, region, [region]
        elif risk == g["max_risk"]:
            g["max_risk_regions"].append(region)   # an exact tie is reported, never broken by name order
    for g in groups.values():
        if len(g["max_risk_regions"]) > 1:
            g["max_risk_region"] = None
    return groups


def prior_regime(prior_region_row: Optional[Mapping]) -> Optional[str]:
    """The regime a prior issued row was produced under; None for a hole (no row)."""
    if prior_region_row is None:
        return None
    ms = prior_region_row.get("method_set")
    if not ms:
        return PRE_CONTRACT_REGIME
    return ms.get("regime") or PRE_CONTRACT_REGIME


def prior_key(prior_region_row: Optional[Mapping]) -> Optional[str]:
    if prior_region_row is None:
        return None
    ms = prior_region_row.get("method_set")
    return ms.get("comparability_key") if ms else None


def persistence_basis(ms: Mapping):
    """Station/carrier replacement must not inherit confirmation, even within a descriptive comparison group.
    Derive from retained fields rather than trusting a possibly stale precomputed key in an issued row.
    """
    if not ms.get("comparability_complete"):
        return None
    included = ms.get("included") or []
    support = ms.get("support") or {}
    identities = tuple((name, (support.get(name) or {}).get("identity")) for name in included)
    if not included or any(not identity or identity == "UNIDENTIFIED" for _, identity in identities):
        return None
    return (ms.get("comparability_key"), identities, tuple(sorted((ms.get("effective_weights") or {}).items())))


def regime_persistence(current_tier: int, current_ms: Mapping, prior_rows: Sequence[Optional[Mapping]],
                       required_consecutive: int) -> Dict:
    """Persistence under the contract. `prior_rows` are the ISSUED prior-day rows, nearest first (index 0 = one day
    back); None marks a hole. Tiers are read exactly as issued; only same-regime days are counted."""
    current = current_ms["regime"]
    consecutive = 1
    transition = None
    for back, row in enumerate(prior_rows, start=1):
        if row is None:
            break
        regime = prior_regime(row)
        if regime != current:
            transition = {"from": regime, "to": current, "days_back": back}
            break
        if PERSISTENCE_RESETS_ON_METHOD_SET_CHANGE and prior_key(row) != current_ms["comparability_key"]:
            transition = {"from": prior_key(row), "to": current_ms["comparability_key"], "days_back": back,
                          "kind": "METHOD_SET"}
            break
        if persistence_basis(current_ms) is None or persistence_basis(row.get("method_set") or {}) != persistence_basis(current_ms):
            transition = {"kind": "MEASUREMENT_SUPPORT", "days_back": back,
                          "reason": "support unknown or changed; start a new confirmation sequence"}
            break
        tier = row.get("tier", 0)
        if tier is not None and tier >= 1 and current_tier >= 1:
            consecutive += 1
        else:
            break
    yesterday = prior_rows[0] if prior_rows else None
    return {
        "regime": current,
        "consecutive_days": consecutive if current_tier >= 1 else 0,
        "is_confirmed": (consecutive >= required_consecutive) if current_tier >= 1 else False,
        "regime_transition": transition,
        "method_set_label": current_ms["label"],
        "method_set_changed": (yesterday is not None and prior_key(yesterday) != current_ms["comparability_key"]),
        "method_set_history": [None if r is None else ((r.get("method_set") or {}).get("label")) for r in
                               reversed(list(prior_rows))] + [current_ms["label"]],
        "resets_on_method_set_change": PERSISTENCE_RESETS_ON_METHOD_SET_CHANGE,
    }
