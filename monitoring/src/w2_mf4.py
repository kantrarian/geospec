#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 M-F4 engine (cayley) -- monitor risk-delta skill per the
FROZEN annex docs/f2g_window2_freeze/annex_mf4.md (design freeze CLOSED
@ 12161f6/5fba544) and grassmann's bar seam pin ("w2_mf4
accrual/scoring seams"). Seam FIXED as `w2_mf4`.

Surfaces:
- features(): issue-time-only feature vector {drisk, roll_z_risk,
  recent_event}; typed no-prediction refusals; any row dated AFTER the
  issue day refuses ISSUE_TIME_VIOLATION (fail-closed).
- calibrate(): fit-ONCE ledger (scaler + L2 logistic C=1.0
  max_iter=2000 on the three features; persistence baseline =
  recent_event alone) over issue days [2025-10-18,
  calibration_issue_end] with calibration_issue_end = min(freeze_day-H,
  snapshot_end-H) -- the codex label-maturity bound. Requesting a later
  end refuses CALIBRATION_LABEL_NOT_MATURE. Training digest = sha256 of
  the canonical training rows. Events later than the snapshot coverage
  CANNOT touch training rows structurally (labels end at d+H <=
  snapshot_end; recent_event windows end before the tail) -- the bar's
  byte-lock KAT verifies this.
- predict_row()/verify_row()/append_row(): one immutable signed row per
  (region, issue_day) with p_model, p_persistence, features, issue
  timestamp, sha256 row digest; mutation -> PREDICTION_ROW_MUTATED;
  duplicate -> PREDICTION_ROW_DUPLICATE. Typed no-prediction days emit
  a typing row (recorded, never silent).
- score_endpoint(): equal-weight MACRO mean over admitted regions of
  AUC_r(model) - AUC_r(persistence), midrank ties;
  REGION_UNSCORABLE_ZERO_CLASS typed; NO-DROP rule: > 1/3 of admitted
  regions unscorable -> ENDPOINT_UNSCORABLE (no verdict). Inference:
  synchronized circular calendar-block bootstrap over issue days, block
  length 14, B = 999, seeds derived from the freeze sha (lane=MF4);
  rejection iff the one-sided 95% percentile lower bound of the macro
  mean > 0.

Interpretation pins (disclosed, R1.2-able):
- roll_z uses the (up to) 7 MOST RECENT AVAILABLE days strictly before
  the issue day (>= 4 required) -- "prior 7 available days" read as
  availability-ordered, not a calendar window.
- recent_event window (d-7, d) is open on both ends; labels use
  (d, d+H] half-open-left per the annex.
- In-replicate zero-class regions are excluded from that replicate's
  macro under the same > 1/3 no-drop rule (replicate invalid -> NaN,
  outside the valid count); valid-replicate floor = the pinned Phase-B
  _valid_floor fraction applied to B.
- Scaler std guard: a zero-variance training feature scales by 1.0
  (disclosed; the annex's 1e-9 guard applies to roll_z internally).

The calibration-catalog receipt, ledger pinning, admission gate
(NOT_ADMITTED_DATA_CONTINUITY at the ~08-25 renewal), embargo, and
late-row clock enforcement live in the accrual/barrier instrument.
This module opens no window-2 value.
"""
import hashlib
import json
import math
import os
import sys
from datetime import date, timedelta

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

import d2_f2g_phase_b_stats as _pb

H_DAYS = 7
BLOCK_LEN = 14
B_REPLICATES = 999
CAL_START = "2025-10-18"
MAG_MIN = 4.0
FEATURES = ("drisk", "roll_z_risk", "recent_event")
ROLL_MIN_PRIOR = 4
ROLL_WINDOW = 7


class Mf4Refusal(ValueError):
    """Typed refusal; the code leads the message."""


def _d(s):
    return date.fromisoformat(str(s))


def _in_bbox(ev, bbox):
    return (bbox["min_lat"] <= ev["lat"] <= bbox["max_lat"]
            and bbox["min_lon"] <= ev["lon"] <= bbox["max_lon"])


def _qualifying_days(events, bbox):
    return sorted(_d(ev["day"]) for ev in events
                  if ev.get("mag", 0.0) >= MAG_MIN and _in_bbox(ev, bbox))


def recent_event(events, bbox, issue_day):
    d0 = _d(issue_day)
    return int(any(d0 - timedelta(days=ROLL_WINDOW) < e < d0
                   for e in _qualifying_days(events, bbox)))


def label(events, bbox, issue_day):
    d0 = _d(issue_day)
    return int(any(d0 < e <= d0 + timedelta(days=H_DAYS)
                   for e in _qualifying_days(events, bbox)))


def features(risk_series, events, bbox, issue_day):
    """Issue-time-only features; typed refusals otherwise."""
    d0 = _d(issue_day)
    days = sorted(_d(k) for k in risk_series)
    if any(dd > d0 for dd in days):
        raise Mf4Refusal(f"ISSUE_TIME_VIOLATION: row dated after "
                         f"{issue_day}")
    if d0 not in days:
        raise Mf4Refusal(f"NO_PREDICTION_MISSING_ISSUE_DAY: {issue_day}")
    prev = d0 - timedelta(days=1)
    if prev not in days:
        raise Mf4Refusal(f"NO_PREDICTION_MISSING_PRIOR: {issue_day}")
    risk = {_d(k): float(v) for k, v in risk_series.items()}
    prior = [dd for dd in days if dd < d0][-ROLL_WINDOW:]
    if len(prior) < ROLL_MIN_PRIOR:
        raise Mf4Refusal(
            f"NO_PREDICTION_INSUFFICIENT_HISTORY: {len(prior)} prior "
            f"available days < {ROLL_MIN_PRIOR}")
    vals = np.array([risk[dd] for dd in prior], dtype=float)
    roll_z = (risk[d0] - float(vals.mean())) / (float(vals.std()) + 1e-9)
    return {"drisk": risk[d0] - risk[prev], "roll_z_risk": roll_z,
            "recent_event": recent_event(events, bbox, issue_day)}


def calibration_issue_end(freeze_day, snapshot_end):
    return min(_d(freeze_day) - timedelta(days=H_DAYS),
               _d(snapshot_end) - timedelta(days=H_DAYS))


def calibrate(risk_by_region, events_snapshot, bboxes, regions,
              freeze_day, snapshot_end, requested_issue_end=None):
    """Fit-once ledger over [CAL_START, calibration_issue_end]."""
    bound = calibration_issue_end(freeze_day, snapshot_end)
    if requested_issue_end is not None and \
            _d(requested_issue_end) > bound:
        raise Mf4Refusal(
            f"CALIBRATION_LABEL_NOT_MATURE: requested "
            f"{requested_issue_end} > matured bound {bound.isoformat()}")
    end = _d(requested_issue_end) if requested_issue_end else bound

    rows = []
    for r in sorted(regions):
        series = risk_by_region[r]
        d0 = _d(CAL_START)
        while d0 <= end:
            iso = d0.isoformat()
            try:
                x = features(
                    {k: v for k, v in series.items() if _d(k) <= d0},
                    events_snapshot, bboxes[r], iso)
            except Mf4Refusal:
                d0 += timedelta(days=1)
                continue
            rows.append({"region": r, "issue_day": iso,
                         "x": [x[f] for f in FEATURES],
                         "y": label(events_snapshot, bboxes[r], iso)})
            d0 += timedelta(days=1)
    if not rows:
        raise Mf4Refusal("CALIBRATION_EMPTY: no admissible training rows")
    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True,
                   separators=(",", ":")).encode()).hexdigest()

    X = np.array([r_["x"] for r_ in rows], dtype=float)
    y = np.array([r_["y"] for r_ in rows], dtype=int)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0.0] = 1.0
    Xs = (X - mean) / std
    from sklearn.linear_model import LogisticRegression
    if len(set(y.tolist())) < 2:
        raise Mf4Refusal("CALIBRATION_SINGLE_CLASS")
    # L2 is sklearn's default penalty (the explicit penalty= kwarg is
    # deprecated in 1.8); semantics identical to the annex's "L2
    # logistic (C=1.0, max_iter=2000)"
    model = LogisticRegression(C=1.0, max_iter=2000)
    model.fit(Xs, y)
    basel = LogisticRegression(C=1.0, max_iter=2000)
    basel.fit(Xs[:, [2]], y)   # recent_event alone (persistence)
    return {"calibration_start": CAL_START,
            "calibration_issue_end": end.isoformat(),
            "training_digest": digest, "n_rows": len(rows),
            "regions": sorted(regions),
            "scaler_mean": mean.tolist(), "scaler_std": std.tolist(),
            "coef": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
            "baseline_coef": basel.coef_[0].tolist(),
            "baseline_intercept": float(basel.intercept_[0])}


def _sigmoid(t):
    return 1.0 / (1.0 + math.exp(-t))


def _row_digest(row):
    body = {k: row[k] for k in sorted(row) if k != "row_digest"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True,
                   separators=(",", ":")).encode()).hexdigest()


def predict_row(ledger, risk_series, events_view, bbox, region,
                issue_day, issued_utc):
    """One immutable signed prediction row (apply-never-refit); typed
    no-prediction days emit a typing row, never silence."""
    try:
        x = features(risk_series, events_view, bbox, issue_day)
    except Mf4Refusal as exc:
        row = {"region": region, "issue_day": str(issue_day),
               "typing": str(exc), "issued_utc": str(issued_utc)}
        row["row_digest"] = _row_digest(row)
        return row
    xs = [(x[f] - m) / s for f, m, s in
          zip(FEATURES, ledger["scaler_mean"], ledger["scaler_std"])]
    p_model = _sigmoid(sum(c * v for c, v in zip(ledger["coef"], xs))
                       + ledger["intercept"])
    p_pers = _sigmoid(ledger["baseline_coef"][0] * xs[2]
                      + ledger["baseline_intercept"])
    row = {"region": region, "issue_day": str(issue_day),
           "p_model": p_model, "p_persistence": p_pers,
           "features": {f: x[f] for f in FEATURES},
           "issued_utc": str(issued_utc)}
    row["row_digest"] = _row_digest(row)
    return row


def verify_row(row):
    if row.get("row_digest") != _row_digest(row):
        raise Mf4Refusal(
            f"PREDICTION_ROW_MUTATED: {row.get('region')} "
            f"{row.get('issue_day')}")
    return True


def append_row(rows, row):
    verify_row(row)
    key = (row["region"], row["issue_day"])
    if any((r_["region"], r_["issue_day"]) == key for r_ in rows):
        raise Mf4Refusal(f"PREDICTION_ROW_DUPLICATE: {key}")
    rows.append(row)
    return rows


def auc_midrank(scores, labels):
    """Mann-Whitney AUC with midrank ties; None if single-class."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n0 == 0 or n1 == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0   # midrank
        i = j + 1
    r1 = float(ranks[y == 1].sum())
    return (r1 - n1 * (n1 + 1) / 2.0) / (n0 * n1)


def _macro(day_index, region_rows, admitted):
    """Macro mean over regions scorable on the given day multiset;
    returns (macro, unscorable list) or (None, unscorable) when the
    no-drop rule trips."""
    deltas = []
    unscorable = []
    for r in admitted:
        rows = region_rows[r]
        sm, sp, yy = [], [], []
        for di in day_index:
            if di in rows:
                pm, pp, y = rows[di]
                sm.append(pm)
                sp.append(pp)
                yy.append(y)
        a_m = auc_midrank(sm, yy) if yy else None
        a_p = auc_midrank(sp, yy) if yy else None
        if a_m is None or a_p is None:
            unscorable.append(r)
            continue
        deltas.append(a_m - a_p)
    if 3 * len(unscorable) > len(admitted):
        return None, unscorable
    return (float(np.mean(deltas)) if deltas else None), unscorable


def score_endpoint(rows, events_snapshot, bboxes, admitted, eval_days,
                   freeze_sha, b=B_REPLICATES, block_len=BLOCK_LEN):
    """The M-F4 endpoint. Labels from the maturity-tail snapshot; macro
    AUC-delta; no-drop rule; synchronized circular block bootstrap."""
    admitted = sorted(admitted)
    eval_days = sorted(str(d0) for d0 in eval_days)
    region_rows = {r: {} for r in admitted}
    for row in rows:
        if "typing" in row or row["region"] not in region_rows:
            continue
        verify_row(row)
        d0 = row["issue_day"]
        if d0 in eval_days:
            y = label(events_snapshot, bboxes[row["region"]], d0)
            region_rows[row["region"]][d0] = (row["p_model"],
                                              row["p_persistence"], y)

    macro_obs, unscorable = _macro(eval_days, region_rows, admitted)
    out = {"endpoint": "M-F4",
           "estimand": "equal-weight macro mean of AUC_r(model) - "
                       "AUC_r(persistence), midrank ties, one-sided "
                       "LB > 0",
           "admitted_regions": admitted,
           "unscorable_regions": unscorable,
           "block_len": block_len, "B": int(b),
           "alpha": 0.05}
    if macro_obs is None:
        out.update(macro_obs=None, lb95=None, reject=None,
                   verdict="ENDPOINT_UNSCORABLE (no-drop rule: "
                           f"{len(unscorable)}/{len(admitted)} "
                           "regions unscorable)"
                   if 3 * len(unscorable) > len(admitted)
                   else "ENDPOINT_UNSCORABLE (no scorable region)")
        return out
    out["macro_obs"] = macro_obs
    for r in unscorable:
        out.setdefault("region_typings", {})[r] = \
            "REGION_UNSCORABLE_ZERO_CLASS"

    n = len(eval_days)
    n_blocks_draw = math.ceil(n / block_len)
    rng = _pb._rng(freeze_sha, "MF4", "full", "bootstrap")
    reps = []
    for _ in range(int(b)):
        starts = rng.integers(0, n, size=n_blocks_draw)
        idx = [eval_days[(int(s) + i) % n] for s in starts
               for i in range(block_len)][:n]
        m_rep, _u = _macro(idx, region_rows, admitted)
        reps.append(float("nan") if m_rep is None else m_rep)
    valid = np.array([v for v in reps if not math.isnan(v)])
    out["n_valid_replicates"] = int(valid.size)
    if valid.size < _pb._valid_floor(b):
        out.update(lb95=None, reject=None,
                   verdict="CANNOT_DETERMINE_BOOTSTRAP_SUPPORT")
        return out
    lb = float(np.percentile(valid, 5.0))
    out["lb95"] = lb
    out["reject"] = bool(lb > 0.0)
    out["verdict"] = ("REJECT (macro LB95 > 0)" if lb > 0.0
                      else "NO_REJECT (macro LB95 <= 0)")
    return out


# ---------------------------------------------------------------- selftest
def _selftest():
    bbox = {"min_lat": 30.0, "max_lat": 40.0,
            "min_lon": -125.0, "max_lon": -115.0}
    bboxes = {"ra": bbox, "rb": bbox, "rc": bbox}

    # feature refusals
    rs = {f"2026-06-{i:02d}": 0.1 * i for i in range(1, 15)}
    try:
        features(rs, [], bbox, "2026-06-10")   # rows after issue day
        raise AssertionError("future rows must refuse")
    except Mf4Refusal as e:
        assert "ISSUE_TIME_VIOLATION" in str(e)
    rs10 = {k: v for k, v in rs.items() if k <= "2026-06-10"}
    x = features(rs10, [], bbox, "2026-06-10")
    assert abs(x["drisk"] - 0.1) < 1e-12 and x["recent_event"] == 0
    try:
        features({"2026-06-01": 1.0, "2026-06-03": 1.0}, [], bbox,
                 "2026-06-03")
        raise AssertionError("missing prior must refuse")
    except Mf4Refusal as e:
        assert "NO_PREDICTION_MISSING_PRIOR" in str(e)

    # calibration + the byte-lock KAT
    rng = np.random.Generator(np.random.PCG64(3))
    days = [(date(2025, 10, 10) + timedelta(days=i)).isoformat()
            for i in range(120)]
    risk = {r: {d0: float(rng.uniform(0, 1)) for d0 in days}
            for r in ("ra", "rb")}
    ev = [{"day": (date(2025, 11, 1) + timedelta(days=7 * i))
           .isoformat(), "lat": 35.0, "lon": -120.0, "mag": 4.5}
          for i in range(8)]
    freeze_day, snap_end = "2026-02-10", "2026-02-08"
    led = calibrate(risk, ev, bboxes, ["ra", "rb"], freeze_day,
                    snap_end)
    assert led["calibration_issue_end"] == "2026-02-01"  # snap_end-7
    tail_ev = ev + [{"day": "2026-02-09", "lat": 35.0, "lon": -120.0,
                     "mag": 5.0}]   # inside the 7-day tail
    led2 = calibrate(risk, tail_ev, bboxes, ["ra", "rb"], freeze_day,
                     snap_end)
    assert led2["training_digest"] == led["training_digest"]
    assert led2["coef"] == led["coef"] \
        and led2["intercept"] == led["intercept"]
    try:
        calibrate(risk, ev, bboxes, ["ra"], freeze_day, snap_end,
                  requested_issue_end="2026-02-02")
        raise AssertionError("immature end must refuse")
    except Mf4Refusal as e:
        assert "CALIBRATION_LABEL_NOT_MATURE" in str(e)

    # immutable signed rows
    row = predict_row(led, rs10, ev, bbox, "ra", "2026-06-10",
                      "2026-06-10T01:00:00Z")
    assert "p_model" in row and "p_persistence" in row
    verify_row(row)
    hacked = dict(row, p_model=0.99)
    try:
        verify_row(hacked)
        raise AssertionError("mutation must refuse")
    except Mf4Refusal as e:
        assert "PREDICTION_ROW_MUTATED" in str(e)
    rows = append_row([], row)
    try:
        append_row(rows, row)
        raise AssertionError("duplicate must refuse")
    except Mf4Refusal as e:
        assert "PREDICTION_ROW_DUPLICATE" in str(e)

    # AUC midrank sanity: perfect, anti, ties
    assert auc_midrank([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert auc_midrank([4, 3, 2, 1], [0, 0, 1, 1]) == 0.0
    assert auc_midrank([1, 1, 1, 1], [0, 0, 1, 1]) == 0.5
    assert auc_midrank([1, 2], [1, 1]) is None

    # scoring: zero-class typing, no-drop rule, planted skill
    eval_days = [(date(2026, 6, 1) + timedelta(days=i)).isoformat()
                 for i in range(56)]

    # sparse events (every 14 days) -> labels ~half/half so replicate
    # zero-class stays rare; skill = p_model aligned with the TRUE
    # label, persistence flat 0.5
    ev_sc = [{"day": (date(2026, 6, 1) + timedelta(days=3 + 14 * i))
              .isoformat(), "lat": 35.0, "lon": -120.0, "mag": 4.2}
             for i in range(4)]

    def mk_rows(regions, skill):
        rr = []
        rng2 = np.random.Generator(np.random.PCG64(9))
        for r in regions:
            for d0 in eval_days:
                lab = label(ev_sc, bbox, d0)
                p = (0.9 if lab else 0.1) if skill else \
                    float(rng2.uniform(0, 1))
                row = {"region": r, "issue_day": d0, "p_model": p,
                       "p_persistence": 0.5,
                       "features": {"drisk": 0.0, "roll_z_risk": 0.0,
                                    "recent_event": 0},
                       "issued_utc": f"{d0}T01:00:00Z"}
                row["row_digest"] = _row_digest(row)
                rr.append(row)
        return rr

    # NOTE: events are shared across regions (same bbox) -> rc gets
    # labels too; give rc its own empty bbox for the zero-class case
    bboxes_sc = {"ra": bbox, "rb": bbox,
                 "rc": {"min_lat": 0, "max_lat": 1,
                        "min_lon": 0, "max_lon": 1}}
    rr = mk_rows(["ra", "rb", "rc"], skill=True)
    res = score_endpoint(rr, ev_sc, bboxes_sc, ["ra", "rb", "rc"],
                         eval_days, "ef" * 32, b=199)
    assert res["unscorable_regions"] == ["rc"], res
    assert res["region_typings"]["rc"] == "REGION_UNSCORABLE_ZERO_CLASS"
    assert res["reject"] is True and res["lb95"] > 0, res

    # no-drop: 2 of 3 unscorable -> ENDPOINT_UNSCORABLE
    bboxes_nd = dict(bboxes_sc,
                     rb={"min_lat": 0, "max_lat": 1,
                         "min_lon": 0, "max_lon": 1})
    res = score_endpoint(rr, ev_sc, bboxes_nd, ["ra", "rb", "rc"],
                         eval_days, "ef" * 32, b=99)
    assert res["reject"] is None \
        and "ENDPOINT_UNSCORABLE" in res["verdict"], res

    # no-skill: model random -> reject False (deterministic seed)
    rr2 = mk_rows(["ra", "rb"], skill=False)
    res = score_endpoint(rr2, ev_sc, {"ra": bbox, "rb": bbox},
                         ["ra", "rb"], eval_days, "ef" * 32, b=199)
    assert res["reject"] is False, res

    # frozen constants
    assert (H_DAYS, BLOCK_LEN, B_REPLICATES) == (7, 14, 999)
    assert CAL_START == "2025-10-18" and MAG_MIN == 4.0
    print("w2_mf4 selftest: ALL PASS")


if __name__ == "__main__":
    _selftest()
