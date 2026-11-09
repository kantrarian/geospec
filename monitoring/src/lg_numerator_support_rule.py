"""lg_numerator_support_rule.py v3 (EDGE_MARGIN 5 since 2026-11-09, grassmann; identity fields edge_policy/edge_margin change with it) -- the M2 LG support rule with BOTH codex guards integrated: the numerator guard
(review db9a28ff finding 1) and the calibration guard (review eb83ad47 finding 2). grassmann 2026-10-04.
NOT wired into production; candidate for the reviewed source successor. Pure functions over the arrays the production
operator already builds (union time grid + per-station real-sample mask from StrainComputer.compute_velocities, and the
(n_times, n_triangles) grid from LambdaGeoComputer); estimator, units, smoothing and the risk mapping are unchanged.

v3 changes (eb83ad47): `denominator_from_grid` keeps valid ZERO daily maxima in the mean (an all-zero mean is
unavailable because division needs a positive denominator), rejects an epoch with an invalid triangle (NaN-only,
infinite or negative) by reason instead of laundering it through a positive max, derives first/last epoch from the
ACCEPTED epochs, and validates shape / daily time axis / policy before computing. `operator_identity` validates every
metadata field BEFORE stringification (a None provider can no longer become the string "None"), requires >= 3 distinct
named stations, non-null smoothing values and integer policies. The acquisition window `days_back` stays an identity
field: both sides must be replayed with a genuinely equal configuration (the round-3 replay uses 90 on both sides) or
the mismatch refuses.

Rule (unchanged): the numerator epoch is the LATEST epoch with real samples from EVERY station of the DECLARED carrier,
at least `edge_margin` rows from the grid end, with no station's series ending (or starting) within `edge_margin`
rows of it; isolated interior gaps stay interpolated as today. No station is pruned; when no such epoch exists within
`max_lag_days` the component is UNAVAILABLE by name. The denominator uses the same epochs. The 45-day lag is an
OFFLINE proposal, not a registered freshness policy.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from datetime import date, datetime
from typing import Dict, Optional, Sequence

import numpy as np

EDGE_MARGIN = 5          # savgol half-window (3) + velocity central difference (1) + strain-rate central difference (1); measured: cutting a tail moves T-0..T-4, T-5 is stable (cayley 5f9a3070)
MAX_LAG_DAYS = 45        # OFFLINE proposal
CLAMP = (0.1, 50.0)      # unchanged production clamp
REFUSALS = ("OPERATOR_IDENTITY_UNBOUND", "OPERATOR_MISMATCH", "SUPPORT_POLICY_INVALID", "INPUT_SHAPE_INVALID",
            "INPUT_TIME_INVALID", "INSUFFICIENT_STATIONS", "DENOMINATOR_UNAVAILABLE", "FUTURE_EPOCH",
            "NO_FULL_SUPPORT_EPOCH", "FULL_SUPPORT_EPOCH_TOO_OLD", "NUMERATOR_UNAVAILABLE", "INSUFFICIENT_VALID_DAYS")
_HEX64 = re.compile(r"[a-f0-9]{64}")
IDENTITY_FIELDS = ("provider", "frame", "units", "station_codes", "triangulation", "smoothing", "derivative",
                   "edge_policy", "gap_policy", "estimator", "reduction", "baseline_statistic", "processing_version",
                   "code_version", "days_back", "edge_margin")
_STRING_FIELDS = tuple(f for f in IDENTITY_FIELDS if f not in ("station_codes", "smoothing", "days_back", "edge_margin"))


def operator_identity(**metadata) -> str:
    """SHA256 over the ORDERED metadata that defines carrier + operator. Every field is validated BEFORE any
    stringification: strings must be nonempty str, the carrier >= 3 distinct named stations (sorted into the key),
    smoothing a nonempty dict with no None values, days_back an int >= 1, edge_margin an int >= EDGE_MARGIN."""
    for key in IDENTITY_FIELDS:
        if key == "edge_margin":
            metadata.setdefault("edge_margin", EDGE_MARGIN)
        if metadata.get(key) is None:
            raise ValueError(f"operator identity field {key} is unbound")
    for key in _STRING_FIELDS:
        v = metadata[key]
        if type(v) is not str or not v.strip():
            raise ValueError(f"operator identity field {key} must be a nonempty string")
    st = metadata["station_codes"]
    if (not isinstance(st, (list, tuple)) or len(st) < 3 or any(type(x) is not str or not x.strip() for x in st)
            or len(set(st)) != len(st)):
        raise ValueError("operator station carrier must contain >= 3 distinct named stations")
    sm = metadata["smoothing"]
    if not isinstance(sm, dict) or not sm or any(v is None for v in sm.values()):
        raise ValueError("smoothing metadata unbound")
    for key, minimum in (("days_back", 1), ("edge_margin", EDGE_MARGIN)):
        v = metadata[key]
        if type(v) is not int or isinstance(v, bool) or v < minimum:
            raise ValueError(f"invalid operator policy {key}")
    payload = {k: metadata[k] for k in IDENTITY_FIELDS}
    payload["station_codes"] = sorted(st)
    payload["smoothing"] = dict(sm)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def production_operator_identity(*, provider: str, station_codes: Sequence[str], code_version: str, days_back: int,
                                 smoothing_window: int = 7) -> str:
    """The identity of the UNCHANGED production operator (live_data.py StrainComputer / LambdaGeoComputer) plus this
    rule's support policy. `provider` is 'NGL' or 'GeoNet'; `code_version` the method commit of the live_data.py bytes;
    `days_back` the ACTUAL acquisition window used for the replay/calibration (an identity field)."""
    return operator_identity(provider=provider, frame="ENU-displacement-mm", units="mm/day-strain-rate-commutator",
                             station_codes=list(station_codes), triangulation="scipy-Delaunay-lat-lon-km-local",
                             smoothing=dict(filter="savgol", window=int(smoothing_window), polyorder=2),
                             derivative="central-difference-1day", edge_policy=f"full-support-interior-margin-{EDGE_MARGIN}",
                             gap_policy="interior-linear-interp-no-tail-hold", estimator="LambdaGeoComputer-commutator-norm",
                             reduction="nanmax-over-triangles", baseline_statistic="mean-of-daily-max-over-rule-epochs",
                             processing_version="lg-support-rule-v3", code_version=code_version, days_back=int(days_back))


def _shape_ok(times, support, grid) -> Optional[str]:
    if (not isinstance(support, np.ndarray) or support.ndim != 2 or support.dtype != np.bool_ or support.shape[1] < 1
            or not isinstance(grid, np.ndarray) or grid.ndim != 2 or grid.shape[0] != support.shape[0] or grid.shape[1] < 1
            or len(times) != support.shape[0] or len(times) == 0 or not np.issubdtype(grid.dtype, np.number)
            or np.iscomplexobj(grid)):
        return "INPUT_SHAPE_INVALID"
    if (not all(isinstance(t, datetime) for t in times)
            or any((times[i + 1].date() - times[i].date()).days != 1 for i in range(len(times) - 1))):
        return "INPUT_TIME_INVALID"
    return None


def valid_denominator_epochs(support: np.ndarray, edge_margin: int = EDGE_MARGIN) -> np.ndarray:
    """Indices i such that every station has a real sample AT i, i is >= edge_margin rows from either grid end, and no
    station's series starts after i-edge_margin or ends before i+edge_margin. Isolated interior gaps stay interpolated."""
    n_t, n_s = support.shape
    full = support.sum(axis=1) == n_s
    first_real = np.array([int(np.argmax(support[:, s])) if support[:, s].any() else n_t for s in range(n_s)])
    last_real = np.array([int(n_t - 1 - np.argmax(support[::-1, s])) if support[:, s].any() else -1 for s in range(n_s)])
    ok = np.zeros(n_t, dtype=bool)
    for i in range(edge_margin, n_t - edge_margin):
        ok[i] = bool(full[i] and (first_real <= i - edge_margin).all() and (last_real >= i + edge_margin).all())
    return np.where(ok)[0]


def _epoch_max(row: np.ndarray):
    """(value, reason): the daily max over finite triangles, or None with NUMERATOR_UNAVAILABLE when the epoch holds an
    infinite or negative triangle or no finite triangle at all. A finite zero is a valid value."""
    row = np.asarray(row, dtype=float)
    finite = row[np.isfinite(row)]
    if not len(finite) or np.isinf(row).any() or (finite < 0).any():
        return None, "NUMERATOR_UNAVAILABLE"
    return float(np.max(finite)), None


def denominator_from_grid(times: Sequence[datetime], support, grid, *, edge_margin: int = EDGE_MARGIN,
                          min_valid_days: int = 30, min_stations: int = 3) -> Dict:
    """The denominator statistic under the SAME rule: mean of the daily max over rule epochs, valid zeros INCLUDED,
    invalid epochs rejected with reasons, first/last epoch taken from the ACCEPTED epochs. available=False carries a
    typed reason and the same detail block."""
    def refused(reason, **detail):
        return dict(available=False, reason=reason, **detail)
    if (type(edge_margin) is not int or edge_margin < EDGE_MARGIN or type(min_valid_days) is not int or min_valid_days < 1
            or type(min_stations) is not int or min_stations < 3):
        return refused("SUPPORT_POLICY_INVALID", n_samples=0)
    bad = _shape_ok(times, support, grid)
    if bad:
        return refused(bad, n_samples=0)
    if support.shape[1] < min_stations:
        return refused("INSUFFICIENT_STATIONS", n_samples=0, n_stations=int(support.shape[1]))
    idx = valid_denominator_epochs(support, edge_margin)
    values, accepted, rejected = [], [], []
    for i in idx:
        v, reason = _epoch_max(grid[i])
        if reason:
            rejected.append(dict(epoch=times[int(i)].date().isoformat(), reason=reason))
        else:
            values.append(v); accepted.append(times[int(i)].date().isoformat())
    detail = dict(n_samples=len(values), n_rule_epochs=int(len(idx)), accepted_epochs=accepted, rejected_epochs=rejected,
                  n_zero_days=int(sum(x == 0 for x in values)))
    if not len(idx):
        return refused("NO_FULL_SUPPORT_EPOCH", **detail)
    if len(values) < min_valid_days:
        return refused("INSUFFICIENT_VALID_DAYS", **detail)
    with np.errstate(over="ignore", invalid="ignore"):
        mean = float(np.mean(values)); std = float(np.std(values))
    if not np.isfinite(mean) or mean <= 0 or not np.isfinite(std):
        return refused("DENOMINATOR_UNAVAILABLE", **detail)
    return dict(available=True, mean_lambda_geo=mean, std_lambda_geo=std, median_lambda_geo=float(np.median(values)),
                first_epoch=accepted[0], last_epoch=accepted[-1], statistic="mean-of-daily-max-over-rule-epochs", **detail)


@dataclass
class NumeratorSelection:
    status: str
    index: Optional[int]
    epoch: Optional[str]
    days_before_target: Optional[int]
    stations_with_real_samples: Optional[int]
    of_stations: int
    lambda_max: Optional[float]
    unclamped_ratio: Optional[float]
    clamped_ratio: Optional[float]
    clamped: Optional[bool]
    reason: str


def _refuse(status, n_s, reason):
    return NumeratorSelection(status, None, None, None, None, int(n_s), None, None, None, None, reason)


def select_numerator(times: Sequence[datetime], support, grid, target, *, denominator,
                     edge_margin: int = EDGE_MARGIN, max_lag_days: int = MAX_LAG_DAYS, min_stations: int = 3,
                     live_identity: Optional[str] = None, denominator_identity: Optional[str] = None) -> NumeratorSelection:
    n_s = support.shape[1] if isinstance(support, np.ndarray) and support.ndim == 2 else 0
    if not all(type(x) is str and _HEX64.fullmatch(x) for x in (live_identity, denominator_identity)):
        return _refuse("OPERATOR_IDENTITY_UNBOUND", n_s, "both operator identities must be bound SHA256 digests")
    if live_identity != denominator_identity:
        return _refuse("OPERATOR_MISMATCH", n_s, "live operator identity != denominator operator identity")
    if (type(edge_margin) is not int or edge_margin < EDGE_MARGIN or type(max_lag_days) is not int or max_lag_days < 0
            or type(min_stations) is not int or min_stations < 3):
        return _refuse("SUPPORT_POLICY_INVALID", n_s, "invalid edge margin, age bound or station minimum")
    bad = _shape_ok(times, support, grid)
    if bad:
        return _refuse(bad, n_s, "aligned nonempty times, boolean support, real triangle grid and a daily union grid required")
    if type(target) is not date or any(times[i].date() >= times[i + 1].date() for i in range(len(times) - 1)):
        return _refuse("INPUT_TIME_INVALID", n_s, "strictly increasing unique observation days and date target required")
    if n_s < min_stations:
        return _refuse("INSUFFICIENT_STATIONS", n_s, f"{n_s} stations < {min_stations}")
    if (not isinstance(denominator, (int, float, np.integer, np.floating)) or isinstance(denominator, (bool, np.bool_))
            or not np.isfinite(denominator) or denominator <= 0):
        return _refuse("DENOMINATOR_UNAVAILABLE", n_s, "no finite positive denominator")
    if any(t.date() > target for t in times):
        return _refuse("FUTURE_EPOCH", n_s, "grid contains epochs after the scored day")
    cand = valid_denominator_epochs(support, edge_margin)
    if len(cand) == 0:
        return _refuse("NO_FULL_SUPPORT_EPOCH", n_s, "no epoch has real samples from every station clear of the grid edge")
    i = int(cand[-1])
    lag = (target - times[i].date()).days
    if lag > max_lag_days:
        return NumeratorSelection("FULL_SUPPORT_EPOCH_TOO_OLD", i, times[i].date().isoformat(), lag, n_s, n_s, None, None,
                                  None, None, f"latest full-support epoch is {lag} d before the scored day (> {max_lag_days})")
    num, reason = _epoch_max(grid[i])
    if reason:
        return _refuse("NUMERATOR_UNAVAILABLE", n_s, "no valid nonnegative finite numerator at the selected epoch")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        q = num / float(denominator)
    if not np.isfinite(q):
        return _refuse("NUMERATOR_UNAVAILABLE", n_s, "numerator ratio is not finite")
    c = max(CLAMP[0], min(CLAMP[1], q))
    return NumeratorSelection("OK", i, times[i].date().isoformat(), lag, int(support[i].sum()), n_s, num, q, c,
                              bool(q != c), "latest full-support interior epoch")


def as_dict(sel: NumeratorSelection) -> dict:
    return asdict(sel)
