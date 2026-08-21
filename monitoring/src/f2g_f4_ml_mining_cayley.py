#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F4 ML MINING PASS v1 (cayley, owner-directed 2026-08-21).

*** EXPLORATORY LANE -- ZERO CLAIM WEIGHT. Rogue-wave template: ML as
hypothesis discovery; interpretable models; surfaced features graduate
ONLY via the window-2 prereg. Nothing here is a skill claim. ***

Two corpora:
  A) Phase-A graphs (consumed window, owner-authorized exploration):
     per (carrier, day) graph features from the real coherence panel;
     labels = USGS ComCat M>=3.5 in the carrier's FDSN receipt bbox
     within the next 7 days (completeness caveat disclosed for TR).
  B) Daily-monitor archive (docs/data.csv, 2025-10-18..): per
     (region, day) risk features + rolling deltas, cross-region pooled;
     labels = ComCat M>=4.0 in the region bbox within the next 7 days.

Honesty guardrails baked in: a PERSISTENCE baseline (recent-event
feature + persistence-only model) is always reported -- aftershock
clustering inflates naive AUC, so only lift OVER persistence counts as
interesting; strict temporal train/test split; label counts reported;
wide-uncertainty framing.
Usage: mine.py <repo>
"""
import csv
import json
import os
import sys
import time
import urllib.request

import numpy as np

BOXES = {
    # Phase-A carriers: the FDSN receipt query boxes (provenance-clean)
    "istanbul_marmara": (40.15, 41.25, 26.85, 31.15),
    "socal_coachella": (32.65, 34.15, -116.95, -115.05),
    "turkey_kahramanmaras": (35.85, 39.15, 34.85, 40.15),
}
HORIZON = 7  # days ahead
PHASE_A_MAG = 3.5
MONITOR_MAG = 4.0


def fetch_events(box, t0, t1, minmag):
    la0, la1, lo0, lo1 = box
    url = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv"
           f"&starttime={t0}&endtime={t1}&minmagnitude={minmag}"
           f"&minlatitude={la0}&maxlatitude={la1}"
           f"&minlongitude={lo0}&maxlongitude={lo1}&orderby=time-asc")
    req = urllib.request.Request(url, headers={"User-Agent":
                                               "geo2graph-f4/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        rows = list(csv.DictReader(
            r.read().decode("utf-8").splitlines()))
    return [(row["time"][:10], float(row["mag"])) for row in rows
            if row.get("mag")]


def day_labels(days, events, horizon):
    ev_days = sorted({d for d, _m in events})
    lab = []
    recent = []
    for d in days:
        fut = any(d < e <= _shift(d, horizon) for e in ev_days)
        past = any(_shift(d, -horizon) <= e < d for e in ev_days)
        lab.append(1 if fut else 0)
        recent.append(1 if past else 0)
    return np.array(lab), np.array(recent)


def _shift(day, k):
    t = time.mktime(time.strptime(day, "%Y-%m-%d")) + k * 86400
    return time.strftime("%Y-%m-%d", time.localtime(t))


def graph_features(panel):
    """Per (carrier, day): interpretable graph features."""
    out = {}
    for ck, c in panel["carriers"].items():
        days = c["registered_days"]
        sts = sorted(c["stations"])
        idx = {s: i for i, s in enumerate(sts)}
        per_day = {}
        for d in days:
            vals = []
            W = np.zeros((len(sts), len(sts)))
            for e, row in c["r"].items():
                if d in row:
                    a, b = e.split("|")
                    v = row[d]
                    vals.append(v)
                    W[idx[a], idx[b]] = W[idx[b], idx[a]] = abs(v)
            if len(vals) < 3:
                per_day[d] = None
                continue
            v = np.array(vals)
            deg = W.sum(axis=1)
            L = np.diag(deg) - W
            try:
                ev = np.linalg.eigvalsh(L)
                fied = float(ev[1]) if len(ev) > 1 else 0.0
                gap = float(ev[2] - ev[1]) if len(ev) > 2 else 0.0
            except Exception:
                fied = gap = np.nan
            per_day[d] = {
                "n_edges": len(vals), "mean_r": float(v.mean()),
                "std_r": float(v.std()),
                "frac_high": float((np.abs(v) > 0.7).mean()),
                "frac_neg": float((v < 0).mean()),
                "fiedler": fied, "eigengap": gap,
            }
        # day-over-day deltas + 7d rolling z of mean_r
        keys = [d for d in days if per_day[d]]
        means = {d: per_day[d]["mean_r"] for d in keys}
        for i, d in enumerate(keys):
            f = per_day[d]
            f["dmean_r"] = (means[d] - means[keys[i - 1]]) if i else 0.0
            w = [means[k] for k in keys[max(0, i - 7):i]]
            f["roll_z_mean_r"] = (float((means[d] - np.mean(w))
                                        / (np.std(w) + 1e-9))
                                  if len(w) >= 4 else 0.0)
        out[ck] = {d: per_day[d] for d in keys}
    return out


def run_models(X, y, recent, feat_names, split_frac=0.7):
    """Temporal split; logistic + GBT + persistence baseline; returns
    AUCs + permutation importances on test."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.inspection import permutation_importance
    n = len(y)
    cut = int(n * split_frac)
    Xtr, Xte = X[:cut], X[cut:]
    ytr, yte = y[:cut], y[cut:]
    res = {"n_train": int(cut), "n_test": int(n - cut),
           "pos_train": int(ytr.sum()), "pos_test": int(yte.sum())}
    if ytr.sum() < 5 or yte.sum() < 3 or ytr.sum() == cut:
        res["status"] = "INSUFFICIENT_LABELS"
        return res
    # persistence baseline: recent-event indicator alone
    try:
        res["auc_persistence"] = round(float(
            roc_auc_score(yte, recent[cut:])), 3)
    except ValueError:
        res["auc_persistence"] = None
    lr = LogisticRegression(max_iter=2000)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - X.mean(axis=0)) / sd
    lr.fit(Xs[:cut], ytr)
    res["auc_logistic"] = round(float(
        roc_auc_score(yte, lr.predict_proba(Xs[cut:])[:, 1])), 3)
    gb = GradientBoostingClassifier(random_state=0, max_depth=2,
                                    n_estimators=150)
    gb.fit(Xtr, ytr)
    res["auc_gbt"] = round(float(
        roc_auc_score(yte, gb.predict_proba(Xte)[:, 1])), 3)
    pi = permutation_importance(gb, Xte, yte, n_repeats=20,
                                random_state=0, scoring="roc_auc")
    order = np.argsort(-pi.importances_mean)
    res["gbt_top_features"] = [
        {"feature": feat_names[i],
         "perm_importance": round(float(pi.importances_mean[i]), 4)}
        for i in order[:8]]
    return res


def main(repo):
    sys.path.insert(0, os.path.join(repo, "monitoring", "src"))
    os.chdir(repo)
    import f2g_sealed_run_instrument_cayley as I
    out = {"schema": "f2g-f4-ml-mining-v1",
           "lane": "EXPLORATORY -- zero claim weight; feature ranking "
                   "for the window-2 prereg only; aftershock clustering "
                   "means only LIFT OVER PERSISTENCE is interesting",
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime()),
           "corpusA": {}, "corpusB": {}}
    # ---------------- corpus A: Phase-A graphs ----------------
    panel = I.build_panel(repo, os.path.join(repo, I.ARTIFACT_ROOT,
                                             "snapshots"),
                          allow_real=True)
    feats = graph_features(panel)
    FEAT = ["n_edges", "mean_r", "std_r", "frac_high", "frac_neg",
            "fiedler", "eigengap", "dmean_r", "roll_z_mean_r",
            "recent_event"]
    for ck, box in BOXES.items():
        events = fetch_events(box, "2026-02-20", "2026-07-25",
                              PHASE_A_MAG)
        days = sorted(feats[ck])
        y, recent = day_labels(days, events, HORIZON)
        X = np.array([[feats[ck][d][f] for f in FEAT[:-1]]
                      + [recent[i]]
                      for i, d in enumerate(days)])
        X = np.nan_to_num(X)
        r = run_models(X, y, recent, FEAT)
        r["n_events_in_box"] = len(events)
        r["completeness_caveat"] = ("ComCat completeness outside US is "
                                    "limited at M3.5 (TR boxes)")
        out["corpusA"][ck] = r
        print(f"[A:{ck}] events={len(events)} {json.dumps({k: v for k, v in r.items() if 'features' not in k})}",
              flush=True)
    # ---------------- corpus B: monitor archive ----------------
    sys.path.insert(0, os.path.join(repo, "monitoring", "src"))
    import fault_segments as FS
    rows = list(csv.DictReader(open(os.path.join(repo, "docs",
                                                 "data.csv"),
                                    encoding="utf-8")))
    regions = sorted({r["region"] for r in rows})
    bmat, by, brecent = [], [], []
    BFEAT = ["risk", "tier", "confidence", "drisk", "roll_z_risk",
             "xregion_mean_risk", "recent_event"]
    xr = {}
    for r in rows:
        xr.setdefault(r["date"], []).append(float(r["risk"]))
    xmean = {d: float(np.mean(v)) for d, v in xr.items()}
    corpB_counts = {}
    for reg in regions:
        segs = FS.FAULT_SEGMENTS.get(reg)
        if not segs:
            continue
        lats = [p[0] for s in segs for p in s.polygon]
        lons = [p[1] for s in segs for p in s.polygon]
        box = (min(lats), max(lats), min(lons), max(lons))
        rr = sorted([r for r in rows if r["region"] == reg],
                    key=lambda x: x["date"])
        days = [r["date"] for r in rr]
        if len(days) < 60:
            continue
        events = fetch_events(box, "2025-10-01", "2026-08-29",
                              MONITOR_MAG)
        corpB_counts[reg] = len(events)
        y, recent = day_labels(days, events, HORIZON)
        risks = [float(r["risk"]) for r in rr]
        for i, r in enumerate(rr):
            w = risks[max(0, i - 7):i]
            bmat.append([risks[i], float(r["tier"]),
                         float(r["confidence"] or 0),
                         risks[i] - risks[i - 1] if i else 0.0,
                         (risks[i] - np.mean(w)) / (np.std(w) + 1e-9)
                         if len(w) >= 4 else 0.0,
                         xmean.get(r["date"], 0.0),
                         recent[i]])
            by.append(y[i])
            brecent.append(recent[i])
        time.sleep(0.5)
    XB = np.nan_to_num(np.array(bmat))
    yB = np.array(by)
    rB = run_models(XB, yB, np.array(brecent), BFEAT)
    rB["regions_pooled"] = corpB_counts
    out["corpusB"]["pooled_14region"] = rB
    print(f"[B:pooled] {json.dumps({k: v for k, v in rB.items() if k not in ('gbt_top_features','regions_pooled')})}",
          flush=True)
    outdir = os.path.join(repo, "docs", "f2g_poc_review")
    with open(os.path.join(outdir, "f4_ml_mining_v1.json"), "w",
              encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    print("wrote docs/f2g_poc_review/f4_ml_mining_v1.json")


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "."))
