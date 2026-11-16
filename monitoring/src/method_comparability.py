"""
method_comparability.py -- PROSPECTIVE comparison contract for the ensemble (method-comparability-v2; v1 + point 6).

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
  6. observation-update-confirmation-v1 (2026-11-14): report-day persistence is kept as counted; CONFIRMATION also
     needs each included method's retained observation to have advanced over those report days. A repeated input
     is two report days and one observation, with an explicit reason; unknown identity blocks confirmation. Every
     included method counts, a qualified zero included (codex ruling A). Introduced as contract v2.
  7. observation-update-confirmation-v2 (2026-11-16; codex 5851b0649 / 8c8d84db9): LG repeat and revision compare the
     SELECTED OBSERVATION's dependency content (lg-observation-content-v1, written by lg_observation_identity), not
     the report's whole input manifest, whose digests embed the scored day's as-of instant. A row without that
     content, or with a schema, encoding, epoch or digest that does not bind, is UNKNOWN; the as-of-bearing digests are
     never a fallback. The decision rule (only ADVANCED counts), THD's identity, the scores and the weights are
     unchanged. Introduced as contract v3: the explicit REGIME_TRANSITION reset stops every chain at the last v2 day,
     so no pre-v3 row is ever compared (WATCH+ regions, THD-only ones included, restart their confirmation sequence
     once; no score or tier moves).
"""
from __future__ import annotations

import hashlib
import inspect
from datetime import date, datetime, timezone
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import calibration_eligibility as CE

# v2 (2026-11-14; codex 93df4f774 item 2) = v1 + observation-update confirmation (point 6 above). The bump is a regime
# change: no v1 report day is carried into a v2 confirmation, and the reset is disclosed as REGIME_TRANSITION.
# v3 (2026-11-16; codex 5851b0649 answer 2) = v2 with the LG selected-observation identity (point 7). Same mechanism.
COMPARISON_CONTRACT_VERSION = "method-comparability-v3"

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


def compose_identity(*parts) -> str:
    """Join identity parts with '|'; UNIDENTIFIED when any part is empty or UNIDENTIFIED, so an unreadable source
    never hides inside a composite key."""
    if not parts or any(not p or p == UNIDENTIFIED for p in parts):
        return UNIDENTIFIED
    return "|".join(parts)
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


# ---- observation-update-confirmation-v1 (cayley 2026-11-14; codex 191be4566, 896778785, 603fefee0) -----------------
# Report-day persistence (`consecutive_days`) is counted exactly as before. CONFIRMATION additionally needs every
# included method's OBSERVATION to have advanced across the counted report days: two reports of one retained LG epoch
# are two report days and ONE observation, never a second measurement. Novelty is read only from retained measurement
# identity -- LG (point 7, v3): the selected observation epoch with its dependency-content digest
# (lg-observation-content-v1); THD: the operator's
# retained returned-trace epoch, sample digest and selected-station identity. Requested windows are consistency
# checks only, never evidence of new observations. Retrieval times,
# report or publication times, file mtimes, calibration dates and an unchanged-content refresh are never read as
# observation evidence. The same epoch with different digests is a REVISION, not a new observation. An identity that
# is not retained stays UNKNOWN and blocks confirmation. Distinct epochs are NOT statistically independent evidence
# (adjacent LG epochs share most of their input window); nothing here says they are.
CONFIRMATION_RULE_VERSION = "observation-update-confirmation-v2"   # v2: LG reads lg-observation-content-v1 (point 7)
STEP_ADVANCED = "ADVANCED"                                 # a later retained epoch
STEP_ADVANCED_BY_WINDOW = "ADVANCED_BY_OPERATOR_WINDOW"    # obsolete candidate label, never emitted
STEP_REPEATED = "REPEATED_INPUT"                           # same epoch, same retained digests
STEP_REVISED = "REVISED_SAME_EPOCH"                        # same epoch, different digests: a revision, not an update
STEP_REGRESSED = "EPOCH_REGRESSED"                         # an earlier epoch than the older report's
STEP_UNKNOWN = "UNKNOWN_OBSERVATION_IDENTITY"              # not retained, not registered, or inconsistent
UPDATE_STEPS = frozenset((STEP_ADVANCED,))
BLOCKING_STEPS = frozenset((STEP_UNKNOWN, STEP_REGRESSED))
UPDATES_BELOW_REQUIRED = "OBSERVATION_UPDATES_BELOW_REQUIRED"
REPORT_DAYS_BELOW_REQUIRED = "REPORT_DAYS_BELOW_REQUIRED"
# why a report-day chain ended (codex 93df4f774 item 2: the regime transition is a disclosed reason, not an outage)
CHAIN_GAP = "GAP"
CHAIN_REGIME_TRANSITION = "REGIME_TRANSITION"
CHAIN_METHOD_SET_CHANGE = "METHOD_SET_CHANGE"
CHAIN_SUPPORT_CHANGE = "MEASUREMENT_SUPPORT_CHANGE"
CHAIN_PRIOR_BELOW_WATCH = "PRIOR_DAY_BELOW_WATCH"
CHAIN_NO_EARLIER_REPORT = "NO_EARLIER_REPORT"
LG_OBSERVATION_BASIS = "RETAINED_LG_SELECTED_OBSERVATION_CONTENT"
# Written by lg_observation_identity (runner side) into measurement['observation_content']; matched here by literal
# so this module imports nothing of the producer.
LG_OBSERVATION_CONTENT_SCHEMA = "lg-observation-content-v1"
LG_OBSERVATION_CONTENT_ENCODING = "lg-observation-content-c14n-v1"
THD_OBSERVATION_BASIS = "RETAINED_THD_TRACE_EPOCH"


def _field(obj, key):
    """A field of an issued JSON row (mapping) or of the current EnsembleResult / MethodResult (attribute)."""
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _iso_day(value) -> Optional[str]:
    if not isinstance(value, str) or len(value) != 10:
        return None
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        return None


def _utc_time(value):
    if not isinstance(value, str) or "T" not in value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        # These retained provider records declare UTC; older request strings omit Z.
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _thd_retained_observation(row):
    """No report-day fallback. Bind actual returned data to the included station.

    A requested window may advance while returned data remain unchanged.
    Missing/ambiguous retained evidence cannot establish observation novelty.
    """
    block = _field(row, "thd_attempts")
    if not isinstance(block, Mapping):
        return None
    stations = [s for s in block.get("stations") or [] if isinstance(s, Mapping) and s.get("selected")]
    if len(stations) != 1:
        return None
    station = stations[0]
    ms = _field(row, "method_set")
    support = (ms.get("support") or {}).get("seismic_thd") if isinstance(ms, Mapping) else None
    if not isinstance(support, Mapping) or station.get("station") != support.get("identity"):
        return None
    mid = station.get("measurement_identity")
    if station.get("outcome") != "VALUE" or not isinstance(mid, str) or not mid:
        return None
    providers = [p for p in station.get("providers") or []
                 if isinstance(p, Mapping) and p.get("outcome") == "DATA_RETURNED"]
    if len(providers) != 1:
        return None
    provider = providers[0]
    window, epoch = provider.get("window_requested"), provider.get("epoch")
    if not all(isinstance(v, (list, tuple)) and len(v) == 2 for v in (window, epoch)):
        return None
    start, end = map(_utc_time, epoch)
    request_start, request_end = map(_utc_time, window)
    day = _iso_day(block.get("scored_day"))
    if any(v is None for v in (start, end, request_start, request_end)) or day is None:
        return None
    if not (start < end and request_start < request_end and end >= request_start
            and start < request_end and end <= request_end
            and request_end == _utc_time(day + "T00:00:00")):
        return None
    digest, trace = provider.get("raw_samples_sha256"), provider.get("trace_id")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        return None
    if not isinstance(trace, str) or not trace.startswith(str(station.get("station")) + "."):
        return None
    return {"status": "KNOWN", "basis": THD_OBSERVATION_BASIS,
            "epoch": end.replace(tzinfo=None).isoformat(), "retained_window": True,
            "content": {"station": station["station"], "measurement_identity": mid,
                        "trace_id": trace, "raw_samples_sha256": digest,
                        "returned_start": start.replace(tzinfo=None).isoformat()}}


def _lg_retained_observation(component) -> Dict:
    """LG under point 7: KNOWN only with an lg-observation-content-v1 summary whose status is KNOWN, whose schema and
    encoding are exactly the registered ones, whose epoch is exactly the measurement's observation_day and whose digest
    is a lowercase sha256. Anything else is UNKNOWN by name, even when a nonempty hash string exists; the report's
    as-of-bearing input digests are never read as observation content."""
    def unknown(reason):
        return {"status": "UNKNOWN", "basis": LG_OBSERVATION_BASIS, "epoch": None, "reason": reason}
    prov = _field(component, "baseline_provenance")
    m = prov.get("measurement") if isinstance(prov, Mapping) else None
    if not isinstance(m, Mapping):
        return unknown("no retained LG measurement")
    epoch = _iso_day(m.get("observation_day"))
    oc = m.get("observation_content")
    if epoch is None or m.get("observation_day") != epoch:
        return unknown("no retained observation_day")
    if not isinstance(oc, Mapping):
        return unknown("no selected-observation content (a pre-v3 row, or a run that did not record it)")
    if oc.get("schema") != LG_OBSERVATION_CONTENT_SCHEMA or oc.get("encoding") != LG_OBSERVATION_CONTENT_ENCODING:
        return unknown("selected-observation content schema or encoding is not the registered one")
    if oc.get("status") != "KNOWN":
        return unknown("selected-observation content UNKNOWN: %s" % (oc.get("reason") or "no reason recorded"))
    if oc.get("epoch") != epoch:
        return unknown("selected-observation content is for another epoch")
    digest = oc.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        return unknown("selected-observation content digest is not a sha256")
    return {"status": "KNOWN", "basis": LG_OBSERVATION_BASIS, "epoch": epoch,
            "content": {"schema": LG_OBSERVATION_CONTENT_SCHEMA, "encoding": LG_OBSERVATION_CONTENT_ENCODING,
                        "sha256": digest}}


def observation_identity(method: str, row) -> Dict:
    """The retained observation identity of one method in one report row: the issued JSON row of a prior day, or the
    current EnsembleResult (whose components carry the same fields that row will be issued with)."""
    components = _field(row, "components")
    component = components.get(method) if isinstance(components, Mapping) else None
    if method == "lambda_geo":
        return _lg_retained_observation(component)
    if method == "seismic_thd":
        retained = _thd_retained_observation(row)
        if retained is None:
            return {"status": "UNKNOWN", "basis": THD_OBSERVATION_BASIS, "epoch": None,
                    "reason": "no consistent retained selected THD trace epoch and sample identity"}
        return retained
    return {"status": "UNKNOWN", "basis": "NOT_REGISTERED", "epoch": None,
            "reason": "no registered observation identity for this method"}


def observation_step(newer: Mapping, older: Mapping) -> str:
    """How one method's observation changed from an older counted report day to the next newer one."""
    if newer.get("status") != "KNOWN" or older.get("status") != "KNOWN":
        return STEP_UNKNOWN
    a, b = newer.get("epoch"), older.get("epoch")
    if a is None or b is None:
        return STEP_UNKNOWN
    if a > b:
        return STEP_ADVANCED
    if a < b:
        return STEP_REGRESSED
    return STEP_REPEATED if newer.get("content") == older.get("content") else STEP_REVISED


def observation_updates(method: str, chain: Sequence) -> Dict:
    """One included method over the counted report days, newest first (chain[0] = the current report)."""
    ids = [observation_identity(method, row) for row in chain]
    steps = [observation_step(newer, older) for newer, older in zip(ids, ids[1:])]
    known = ids[0]["status"] == "KNOWN"
    blocking = sorted(set(s for s in steps if s in BLOCKING_STEPS) | (set() if known else {STEP_UNKNOWN}))
    return {"basis": ids[0]["basis"], "report_days": len(ids), "epochs": [i.get("epoch") for i in ids],
            "steps": steps, "observation_updates": (1 + sum(s in UPDATE_STEPS for s in steps)) if known else None,
            "revisions": sum(s == STEP_REVISED for s in steps), "blocking": blocking}


def _not_confirmed_reason(method: str, upd: Mapping, required: int) -> Optional[str]:
    if upd["blocking"]:
        return "%s:%s" % (upd["blocking"][0], method)
    if upd["observation_updates"] >= required:
        return None
    for code in (STEP_REPEATED, STEP_REVISED):
        if code in upd["steps"]:
            return "%s:%s" % (code, method)
    return "%s:%s" % (UPDATES_BELOW_REQUIRED, method)


def regime_persistence(current_tier: int, current_ms: Mapping, prior_rows: Sequence[Optional[Mapping]],
                       required_consecutive: int, current_row=None) -> Dict:
    """Persistence under the contract. `prior_rows` are the ISSUED prior-day rows, nearest first (index 0 = one day
    back); None marks a hole. Tiers are read exactly as issued; only same-regime days are counted. `current_row` is the
    current report's result, read only for its retained observation identities; without it no LG observation identity
    is known and an LG region cannot confirm."""
    current = current_ms["regime"]
    consecutive = 1
    transition = None
    counted = []
    stop = None   # why the report-day chain ended (named in the reason when too few report days remain)
    for back, row in enumerate(prior_rows, start=1):
        if row is None:
            stop = CHAIN_GAP
            break
        regime = prior_regime(row)
        if regime != current:
            transition = {"from": regime, "to": current, "days_back": back}
            stop = CHAIN_REGIME_TRANSITION
            break
        if PERSISTENCE_RESETS_ON_METHOD_SET_CHANGE and prior_key(row) != current_ms["comparability_key"]:
            transition = {"from": prior_key(row), "to": current_ms["comparability_key"], "days_back": back,
                          "kind": "METHOD_SET"}
            stop = CHAIN_METHOD_SET_CHANGE
            break
        if persistence_basis(current_ms) is None or persistence_basis(row.get("method_set") or {}) != persistence_basis(current_ms):
            transition = {"kind": "MEASUREMENT_SUPPORT", "days_back": back,
                          "reason": "support unknown or changed; start a new confirmation sequence"}
            stop = CHAIN_SUPPORT_CHANGE
            break
        tier = row.get("tier", 0)
        if tier is not None and tier >= 1 and current_tier >= 1:
            consecutive += 1
            counted.append(row)
        else:
            stop = CHAIN_PRIOR_BELOW_WATCH
            break
    else:
        stop = CHAIN_NO_EARLIER_REPORT
    yesterday = prior_rows[0] if prior_rows else None
    chain = [current_row] + counted
    updates = {m: observation_updates(m, chain) for m in (current_ms.get("included") or [])}
    report_days_met = current_tier >= 1 and consecutive >= required_consecutive
    reasons = []
    if current_tier >= 1:
        if not report_days_met:
            # too few report days already decides it: name why the chain ended, then add only what independently
            # blocks (unknown or regressed)
            reasons.append("%s:%s" % (REPORT_DAYS_BELOW_REQUIRED, stop))
            reasons += ["%s:%s" % (u["blocking"][0], m) for m, u in updates.items() if u["blocking"]]
        else:
            reasons += [r for r in (_not_confirmed_reason(m, u, required_consecutive)
                                    for m, u in updates.items()) if r]
    return {
        "regime": current,
        "consecutive_days": consecutive if current_tier >= 1 else 0,
        "is_confirmed": bool(report_days_met and not reasons),
        "report_days": consecutive if current_tier >= 1 else 0,
        "report_days_met": report_days_met,
        "confirmation_rule": CONFIRMATION_RULE_VERSION,
        "observation_updates": updates,
        "not_confirmed_reasons": reasons,
        "regime_transition": transition,
        "method_set_label": current_ms["label"],
        "method_set_changed": (yesterday is not None and prior_key(yesterday) != current_ms["comparability_key"]),
        "method_set_history": [None if r is None else ((r.get("method_set") or {}).get("label")) for r in
                               reversed(list(prior_rows))] + [current_ms["label"]],
        "resets_on_method_set_change": PERSISTENCE_RESETS_ON_METHOD_SET_CHANGE,
    }
