#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 CALIBRATION-LEDGER production runner (cayley) -- the
orchestration that, AT THE AVAILABILITY CUTOFF, fits the frozen
apply-never-refit ledgers and emits them with input-bound receipts.
Fills the `calibration_ledgers` execution-manifest slot when run for
real; until then this module is fixture-verified machinery.

FEED CONTRACT (the producer targets these shapes -- the panel-shape
precedent):
- MF4 feed: {"risk_by_region": {region: {iso_day: float}},
   "catalog_snapshot": [{"day","lat","lon","mag"}...],
   "snapshot_end": iso_day, "freeze_day": iso_day,
   "bboxes": {region: bbox}, "regions": [...]}
- MAG feed (one per observatory): {"observatory": iaga,
   "lon_east": deg, "times": [iso minute stamps],
   "components": {"X": [...], "Y": [...]},
   "weather": {name: [...aligned...]},
   "m3_reference": iaga-or-None}
  Calibration interval: 2026-01-01 -> the cutoff (the caller slices;
  the engines refuse unsupported/rank-deficient designs typed).

PROVENANCE RULE (the content-auth != derivation-provenance lesson):
every receipt binds (a) the INPUT CARRIER digests -- each feed's
canonical-JSON sha256 computed BEFORE fitting, (b) the PRODUCER
identity handed in by the producer (their code blob sha -- recorded,
not attested here), (c) THIS runner's own executed-source sha, and
(d) the output ledger digests. verify_receipt() recomputes the output
digests from the written artifacts and refuses on any divergence
(typed CALIBRATION_RECEIPT_MISMATCH). Cutoff ordering vs
evaluation_start is validated by the BARRIER at PRESTART assembly (the
cutoff exists before evaluation_start does); this runner records the
cutoff verbatim.

This module opens no window-2 value: calibration uses strictly
pre-evaluation bytes by construction of the interval.
"""
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import w2_mf4 as MF4
import w2_mag1 as MAG

OUT_DIR = "docs/f2g_window2_execution/calibration"


class CalibrationRunnerError(ValueError):
    """Typed refusal; the code leads the message."""


def _canon_digest(obj):
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def _self_sha():
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(
            f.read().replace(b"\r\n", b"\n")).hexdigest()


def _write(repo, rel, obj):
    p = os.path.join(repo, rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")
    return rel


def run_mf4_calibration(repo, feed, cutoff, producer_identity):
    """MF4 fit-once ledger + receipt. Input digest FIRST, then fit."""
    for k in ("risk_by_region", "catalog_snapshot", "snapshot_end",
              "freeze_day", "bboxes", "regions"):
        if k not in feed:
            raise CalibrationRunnerError(f"MF4_FEED_INCOMPLETE: {k}")
    feed_digest = _canon_digest(feed)
    ledger = MF4.calibrate(feed["risk_by_region"],
                           feed["catalog_snapshot"], feed["bboxes"],
                           feed["regions"], feed["freeze_day"],
                           feed["snapshot_end"])
    led_path = _write(repo, f"{OUT_DIR}/mf4_ledger.json", ledger)
    receipt = {"schema": "f2g-w2-calibration-receipt-v1",
               "lane": "MF4", "cutoff": str(cutoff),
               "input_feed_sha256": feed_digest,
               "producer_identity": dict(producer_identity),
               "runner_source_sha256_normalized": _self_sha(),
               "ledger_path": led_path,
               "ledger_sha256": _canon_digest(ledger),
               "training_digest": ledger["training_digest"],
               "n_rows": ledger["n_rows"]}
    rec_path = _write(repo, f"{OUT_DIR}/mf4_ledger.receipt.json",
                      receipt)
    return {"ledger": led_path, "receipt": rec_path,
            "ledger_sha256": receipt["ledger_sha256"]}


def run_mag_calibration(repo, feeds, cutoff, producer_identity):
    """MAG subtraction ledgers per observatory per horizontal
    component, plus M3 reference regressions per the frozen
    instantiation. Input digests FIRST, then fits."""
    out = {"observatories": {}, "m3": {}}
    residuals = {}
    for obs in sorted(feeds):
        feed = feeds[obs]
        for k in ("observatory", "lon_east", "times", "components",
                  "weather"):
            if k not in feed:
                raise CalibrationRunnerError(
                    f"MAG_FEED_INCOMPLETE: {obs}:{k}")
        feed_digest = _canon_digest(
            {k: v for k, v in feed.items() if k != "m3_reference"})
        obs_rec = {"input_feed_sha256": feed_digest, "components": {}}
        residuals[obs] = {}
        for comp in ("X", "Y"):
            if comp not in feed["components"]:
                raise CalibrationRunnerError(
                    f"MAG_FEED_INCOMPLETE: {obs}:components:{comp}")
            led = MAG.fit_subtraction(
                feed["times"], feed["components"][comp],
                feed["lon_east"], feed["weather"],
                meta={"observatory": obs, "component": comp,
                      "cutoff": str(cutoff)})
            rel = _write(repo,
                         f"{OUT_DIR}/mag_{obs.lower()}_{comp}"
                         f"_ledger.json", led)
            obs_rec["components"][comp] = {
                "ledger_path": rel, "ledger_sha256": _canon_digest(led),
                "ledger_digest_field": led["digest"]}
            residuals[obs][comp] = MAG.apply_subtraction(
                led, feed["times"], feed["components"][comp],
                feed["weather"])
        out["observatories"][obs] = obs_rec
    # M3 references: local residual ~ reference residual + weather
    for obs in sorted(feeds):
        ref = feeds[obs].get("m3_reference")
        if not ref:
            continue
        if ref not in residuals:
            raise CalibrationRunnerError(
                f"MAG_M3_REFERENCE_ABSENT: {obs} -> {ref}")
        for comp in ("X", "Y"):
            led = MAG.fit_m3_reference(
                residuals[obs][comp], residuals[ref][comp],
                {n: feeds[obs]["weather"][n]
                 for n in sorted(feeds[obs]["weather"])},
                meta={"local": obs, "reference": ref,
                      "component": comp, "cutoff": str(cutoff)})
            rel = _write(repo,
                         f"{OUT_DIR}/mag_m3_{obs.lower()}_on_"
                         f"{ref.lower()}_{comp}_ledger.json", led)
            out["m3"][f"{obs}:{ref}:{comp}"] = {
                "ledger_path": rel,
                "ledger_sha256": _canon_digest(led)}
    receipt = {"schema": "f2g-w2-calibration-receipt-v1",
               "lane": "MAG", "cutoff": str(cutoff),
               "producer_identity": dict(producer_identity),
               "runner_source_sha256_normalized": _self_sha(),
               "results": out}
    rec_path = _write(repo, f"{OUT_DIR}/mag_ledgers.receipt.json",
                      receipt)
    return {"receipt": rec_path, "results": out}


def verify_receipt(repo, receipt_rel):
    """Recompute every output ledger digest from the written artifacts;
    refuse typed on divergence."""
    with open(os.path.join(repo, receipt_rel.replace("/", os.sep)),
              encoding="utf-8") as f:
        rec = json.load(f)

    def check(path_rel, want):
        with open(os.path.join(repo, path_rel.replace("/", os.sep)),
                  encoding="utf-8") as f:
            got = _canon_digest(json.load(f))
        if got != want:
            raise CalibrationRunnerError(
                f"CALIBRATION_RECEIPT_MISMATCH: {path_rel} "
                f"{got[:12]} != {want[:12]}")

    n = 0
    if rec["lane"] == "MF4":
        check(rec["ledger_path"], rec["ledger_sha256"])
        n = 1
    else:
        for obs in rec["results"]["observatories"].values():
            for c in obs["components"].values():
                check(c["ledger_path"], c["ledger_sha256"])
                n += 1
        for m3 in rec["results"]["m3"].values():
            check(m3["ledger_path"], m3["ledger_sha256"])
            n += 1
    return {"verified_ledgers": n, "lane": rec["lane"]}


# ---------------------------------------------------------------- selftest
def _selftest():
    import tempfile
    import numpy as np
    from datetime import date, datetime, timedelta
    repo = tempfile.mkdtemp(prefix="w2_cal_kat_")
    rng = np.random.Generator(np.random.PCG64(17))
    producer = {"name": "kat-producer", "code_blob_sha256": "ab" * 32}

    # MF4: synthetic feed -> ledger + receipt -> verify -> tamper
    days = [(date(2025, 10, 10) + timedelta(days=i)).isoformat()
            for i in range(120)]
    bbox = {"min_lat": 30, "max_lat": 40, "min_lon": -125,
            "max_lon": -115}
    feed = {"risk_by_region": {r: {d: float(rng.uniform(0, 1))
                                   for d in days} for r in ("ra", "rb")},
            "catalog_snapshot": [
                {"day": (date(2025, 11, 1) + timedelta(days=7 * i))
                 .isoformat(), "lat": 35.0, "lon": -120.0, "mag": 4.5}
                for i in range(8)],
            "snapshot_end": "2026-02-08", "freeze_day": "2026-02-10",
            "bboxes": {"ra": bbox, "rb": bbox},
            "regions": ["ra", "rb"]}
    res = run_mf4_calibration(repo, feed, "2026-02-09", producer)
    v = verify_receipt(repo, res["receipt"])
    assert v == {"verified_ledgers": 1, "lane": "MF4"}
    # determinism: rerun -> identical ledger digest
    res2 = run_mf4_calibration(repo, feed, "2026-02-09", producer)
    assert res2["ledger_sha256"] == res["ledger_sha256"]
    # tamper a written ledger -> receipt verification refuses
    lp = os.path.join(repo, res["ledger"].replace("/", os.sep))
    led = json.load(open(lp))
    led["intercept"] = 99.9
    json.dump(led, open(lp, "w"))
    try:
        verify_receipt(repo, res["receipt"])
        raise AssertionError("tampered ledger must refuse")
    except CalibrationRunnerError as e:
        assert "CALIBRATION_RECEIPT_MISMATCH" in str(e)
    try:
        run_mf4_calibration(repo, {k: v_ for k, v_ in feed.items()
                                   if k != "bboxes"},
                            "2026-02-09", producer)
        raise AssertionError("incomplete feed must refuse")
    except CalibrationRunnerError as e:
        assert "MF4_FEED_INCOMPLETE" in str(e)

    # MAG: two observatories, one M3 pair, per-component ledgers
    n = 3000
    times = [(datetime(2026, 1, 1) + timedelta(minutes=i)).isoformat()
             for i in range(n)]
    weather = {"symh": rng.normal(size=n).tolist()}

    def obs_feed(name, ref):
        return {"observatory": name, "lon_east": -120.0,
                "times": times,
                "components": {
                    "X": rng.normal(20000, 5, size=n).tolist(),
                    "Y": rng.normal(4000, 5, size=n).tolist()},
                "weather": weather, "m3_reference": ref}
    feeds = {"FRN": obs_feed("FRN", "TUC"),
             "TUC": obs_feed("TUC", None)}
    res = run_mag_calibration(repo, feeds, "2026-08-24", producer)
    v = verify_receipt(repo, res["receipt"])
    assert v["verified_ledgers"] == 6      # 2 obs x2 comps + 2 M3
    assert "FRN:TUC:X" in res["results"]["m3"]
    try:
        run_mag_calibration(
            repo, {"FRN": obs_feed("FRN", "NEW")}, "2026-08-24",
            producer)
        raise AssertionError("absent M3 reference must refuse")
    except CalibrationRunnerError as e:
        assert "MAG_M3_REFERENCE_ABSENT" in str(e)

    print("w2_calibration_runner selftest: ALL PASS")


if __name__ == "__main__":
    _selftest()
