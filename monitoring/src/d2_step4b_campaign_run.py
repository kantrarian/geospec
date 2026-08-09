#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""d2_step4b_campaign_run.py — the D2 step-4b campaign EXECUTOR body (codex 0347 P1 `_acquire`).

Reached ONLY through `d2_step4b_producer.run_campaign` past the VERIFIED_DIRECT launch gate
(`_acquire` lazily imports this module). Given the staged `campaign_plan.json` +
`published_phase_ledger.json` under the campaign root, it runs the full published-phase-bound
campaign — per REGISTERED (carrier, day): fetch (KOERI first, then SCEDC) → segmented score
(pinned GeoSpec 3950a2c: seismic_data + fault_correlation) → the codex `0123` batch artifacts —
and stages every remaining artifact + `batch_manifest.json` under the root.

Nothing here lifts a freeze, tunes the admission rule, promotes a registry, deploys, publishes,
or makes a claim; the campaign is outcome-blind and honestly mints 0..3 candidates. The batch is
independently re-verified by codex's `0123` acceptance bar + cayley's Phase-4 checklist.
"""
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # geospec_sprint
CONTRACT_ID = "codex-d2-step4b-2026-08-09-v1"
IMPLEMENTATION_COMMIT = "3950a2c4fabc870eaa929f41a790b875966b9dfb"
BAND_TAG = "1-10Hz"
PROCESSING_VERSION = "d2-reband-v2"
TOPOLOGY_VERSION = "t1"
ELIGIBLE = ("istanbul_marmara", "socal_coachella", "turkey_kahramanmaras")
SESSION_SECONDS = 86400
REGION_KEY = {"istanbul_marmara": "istanbul_marmara",
              "socal_coachella": "socal_saf_coachella",
              "turkey_kahramanmaras": "turkey_kahramanmaras"}
NON_CLAIMS = ["NO_FREEZE_LIFT", "NO_DEPLOYMENT", "NO_PUBLICATION", "NO_SCIENTIFIC_CLAIM"]
DIAGNOSTIC_FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "fixtures", "d2_diagnostic_result.json")

POLICY = {
    "window_days": 90, "embargo_days": 30, "min_admitted_days": 60, "lower_quantile": 0.05,
    "quantile_rule": "nearest-rank-lower-tail-ceil", "window_semantics": "[start,end)",
    "daily_window": ("published-phase-segmented-v1:score-day-D;record=public-commit+blob;"
                     "session=[request_start,request_end);exact-time;station-support>=0.5;"
                     "segment>=2stations;common-support>=0.5;no-record=no-fetch"),
    "incident_reference_day": "2026-07-29", "control_scored_day": "2026-07-28",
    "candidate_valid_days": 7, "session_seconds": 86400, "station_coverage_floor": 0.5,
    "min_common_support_fraction": 0.5, "edge_trim_seconds": 90,
    "min_contiguous_span_seconds": 240,
}
PROVIDERS = {
    "istanbul_marmara": {"provider": "KOERI", "endpoint": "eida.koeri.boun.edu.tr"},
    "socal_coachella": {"provider": "SCEDC", "endpoint": "s3://scedc-pds"},
    "turkey_kahramanmaras": {"provider": "KOERI", "endpoint": "eida.koeri.boun.edu.tr"},
}
PRIOR = {
    "lane_a_manifest_sha256": "eeb888d0b9611ea7be402964cec3fb839cd6ff6f42b699f4631146c2adc4f600",
    "diagnostic_manifest_sha256": "be1479d2f79faccead382804a4a3464826256889f0e96a5dc5c49f5069501a88",
    "diagnostic_result_sha256": "ee75e449aa0b1003a3cf047432a91a9adc1db4c7497b1e1d9d47f01d552a4b35",
}
CORE_FILES = {
    "monitoring/src/seismic_data.py": "04d4c305063a7a2e59c3b5b32dfae4843a3a2f6e047372fc9b52b1da1fea2b0b",
    "monitoring/src/fault_correlation.py": "a657c0ebce3f5febd62ae4fe5788b077bfe1b4060066a5749d73415b7ee1409c",
    "monitoring/src/ensemble.py": "3b2a2c6126298a75255c24869893acf3fa9408667113324bb3656f71837498db",
    "monitoring/src/fault_segments.py": "90acfd35fc05bd9b4f178837ebbd2ff7e44b6211a422e47bfbcfa230c1b8f913",
    "monitoring/src/test_d2_reband_redkats_cayley.py": "9262b10e4fab43618aca8bdea39fc6312814f9b81412e64791ac7c707d7af6d2",
    "monitoring/src/test_d2_livecarrier_fixes_grassmann.py": "b268ff48a53b135030a054eed84eef91adb2d8a536929abb1a9d26fe5bb7fff6",
    "monitoring/src/test_d2_segsupport_redkats_cayley.py": "1ca5a4a2a5c9cdea1303998db26e721d76aa0237347f94eef561fa51f5094319",
}
TARGETS = [
    {"runner_key": "ridgecrest", "carrier_key": "ridgecrest",
     "freeze_keys": ["ridgecrest"], "base_disposition": "T2_SUPPLEMENT_REQUIRED"},
    {"runner_key": "socal_saf_coachella", "carrier_key": "socal_coachella",
     "freeze_keys": ["socal_saf_coachella", "socal_coachella"], "base_disposition": "ELIGIBLE"},
    {"runner_key": "tokyo_kanto", "carrier_key": None,
     "freeze_keys": ["tokyo_kanto", "japan_tohoku"], "base_disposition": "NO_TRUE_CARRIER"},
    {"runner_key": "istanbul_marmara", "carrier_key": "istanbul_marmara",
     "freeze_keys": ["istanbul_marmara"], "base_disposition": "ELIGIBLE"},
    {"runner_key": "turkey_kahramanmaras", "carrier_key": "turkey_kahramanmaras",
     "freeze_keys": ["turkey_kahramanmaras"], "base_disposition": "ELIGIBLE"},
]
OPS = ("native_bandpass_hilbert_envelope_resample", "station_utc_intersection_median",
       "segment_correlation_eigenspectrum")
RATIO_DERIVATION = "ordered_eigenvalues[1] / ordered_eigenvalues[0] (descending)"


# ---- canonical IO ----------------------------------------------------------
def _canon(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n").encode("utf-8")


def _write_json(path, value):
    with open(path, "wb") as fh:
        fh.write(_canon(value))


def _write_jsonl(path, rows):
    with open(path, "wb") as fh:
        for row in rows:
            fh.write(_canon(row))


def _sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matrix_sha(matrix):
    return hashlib.sha256(json.dumps(matrix, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _op_digest(*parts):
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


# ---- git provenance (overridable for the hermetic self-test) ---------------
def _git_head(repo=REPO):
    return subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def _git_clean(repo=REPO):
    out = subprocess.run(["git", "-C", repo, "status", "--porcelain"],
                         capture_output=True, text=True).stdout
    return out.strip() == ""


# ---- scoring (ported from the accepted diagnostic replay; GeoSpec 3950a2c) --
def _fragments_from_stream(stream):
    frags = []
    for tr in stream:
        start = tr.stats.starttime.datetime.replace(tzinfo=timezone.utc)
        frags.append((np.asarray(tr.data, dtype=float), float(tr.stats.sampling_rate), start))
    frags.sort(key=lambda x: x[2])
    return frags


def _mask_digest(series):
    m = np.asarray(getattr(series, "valid_mask", []), dtype=bool)
    return hashlib.sha256(np.packbits(m).tobytes()).hexdigest() if m.size else (64 * "0")


def _station_series(SD, stream, source_id, session_start):
    """Return (EnvelopeSeries|None, provenance) for one fetched station over the bound session."""
    frags = _fragments_from_stream(stream)
    prov = {"native_rate_hz": (float(frags[0][1]) if frags else 0.0),
            "npts": int(sum(len(f[0]) for f in frags)), "fragment_count": len(frags)}
    try:
        es = SD.compute_band_envelope_supported(
            frags, session_start_utc=session_start, session_seconds=SESSION_SECONDS,
            source_id=source_id)
        prov["coverage"] = round(float(es.coverage), 6)
        prov["support_sha256"] = _mask_digest(es)
        return es, prov
    except SD.DataUnavailable as exc:
        prov["coverage"] = None
        prov["support_sha256"] = 64 * "0"
        prov["unavailable"] = str(exc)[:120]
        return None, prov


def _score_day(SD, FC, seg_station_es, session_start):
    """Score one (carrier, day) from {segment: [(nslc, EnvelopeSeries)]}. Returns the arm-blind
    daily result body (all daily_keys except `arm`), status ADMITTED or REJECTED."""
    session_end = session_start + timedelta(seconds=SESSION_SECONDS)
    base = {
        "session_start_utc": session_start.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "session_end_utc": session_end.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "requested_grid_count": SESSION_SECONDS,
    }
    seg_series, seg_names, segment_support = [], [], {}
    for segment in sorted(seg_station_es):
        entries = seg_station_es[segment]
        st_series = [es for (_nslc, es) in entries if es is not None]
        agg = FC.aggregate_segment_supported(st_series) if st_series else None
        if agg is None:
            segment_support[segment] = {
                "station_ids": sorted(nslc for (nslc, _es) in entries),
                "station_coverages": {}, "aggregate_valid_bins": 0,
                "aggregate_mask_sha256": 64 * "0"}
            continue
        coverages = {nslc: round(float(es.coverage), 6) for (nslc, es) in entries if es is not None}
        segment_support[segment] = {
            "station_ids": sorted(coverages),
            "station_coverages": coverages,
            "aggregate_valid_bins": int(np.asarray(agg.valid_mask, dtype=bool).sum()),
            "aggregate_mask_sha256": _mask_digest(agg)}
        seg_series.append(agg)
        seg_names.append(segment)
    C, names, qc = FC.compute_correlation_matrix_supported(seg_series, seg_names)
    if C is None or len(seg_names) < 2:
        base.update({
            "status": "REJECTED", "ratio": None, "participation_ratio": None,
            "data_quality_ok": False,
            "qc_reasons": (list(qc)[:4] if qc else ["INSUFFICIENT_SEGMENTS"]) or ["INSUFFICIENT_SEGMENTS"],
            "segment_names": list(seg_names), "segment_support": segment_support,
            "common_support_count": None, "correlation_matrix": None,
            "correlation_matrix_order": None, "ordered_eigenvalues": None,
            "lambda2_lambda1_derivation": None, "support_sha256": None})
        return base
    masks = np.vstack([np.asarray(s.valid_mask, dtype=bool).ravel() for s in seg_series])
    common = int(masks.all(axis=0).sum())
    # matrix, eigenvalues, and ratio all derive from ONE full-precision symmetric unit-diagonal
    # matrix via eigvalsh -- exactly what codex 0123 validate_matrix recomputes (no rounding drift).
    Cmat = np.asarray(C, dtype=float)
    Cmat = 0.5 * (Cmat + Cmat.T)
    np.fill_diagonal(Cmat, 1.0)
    eigenvalues = np.linalg.eigvalsh(Cmat)[::-1]
    matrix = [[float(v) for v in row] for row in Cmat]
    ordered = [float(x) for x in eigenvalues]
    ratio = float(eigenvalues[1] / eigenvalues[0]) if eigenvalues[0] else None
    pr = float((eigenvalues.sum() ** 2) / np.square(eigenvalues).sum())
    support_sha = hashlib.sha256(_canon({"segments": segment_support, "common": common})).hexdigest()
    base.update({
        "status": "ADMITTED", "ratio": ratio, "participation_ratio": pr,
        "data_quality_ok": True, "qc_reasons": [], "segment_names": list(seg_names),
        "segment_support": segment_support, "common_support_count": common,
        "correlation_matrix": matrix, "correlation_matrix_order": list(seg_names),
        "ordered_eigenvalues": ordered, "lambda2_lambda1_derivation": RATIO_DERIVATION,
        "support_sha256": support_sha})
    return base


def _no_record_daily():
    return {
        "session_start_utc": None, "session_end_utc": None, "publication_record_sha256": None,
        "status": "REJECTED", "ratio": None, "participation_ratio": None,
        "data_quality_ok": False, "qc_reasons": ["NO_PUBLISHED_DAILY_RECORD"],
        "input_object_sha256s": [], "segment_names": [], "segment_support": {},
        "common_support_count": None, "requested_grid_count": SESSION_SECONDS,
        "correlation_matrix": None, "correlation_matrix_order": None,
        "ordered_eigenvalues": None, "lambda2_lambda1_derivation": None, "support_sha256": None}


# ---- the executor ----------------------------------------------------------
def acquire(plan, ledger, root, *, providers, receipt):
    import seismic_data as SD
    import fault_correlation as FC

    scheduled_days = list(plan["scheduled_days"])
    incident_days = list(plan["incident_days"])
    activation_days = list(plan["activation_days"])
    station_registry = plan["station_registry"]
    ledger_index = {(r["carrier_key"], r["scored_day"]): r for r in ledger["rows"]}
    stage_dir = os.path.join(root, "raw_objects")
    os.makedirs(stage_dir, exist_ok=True)

    attempts, input_objects, day_result = [], [], {}
    now = datetime.now(timezone.utc)
    carriers_ordered = ([c for c in ELIGIBLE if PROVIDERS[c]["provider"] == "KOERI"]
                        + [c for c in ELIGIBLE if PROVIDERS[c]["provider"] == "SCEDC"])
    for carrier in carriers_ordered:
        provider = PROVIDERS[carrier]["provider"]
        net = None
        segments = {}
        for srow in station_registry[carrier]:
            segments.setdefault(srow["segment_name"], []).append(srow)
        for day in scheduled_days:
            row = ledger_index.get((carrier, day))
            if not row or row.get("status") != "REGISTERED":
                continue
            start = datetime.strptime(row["request_start_utc"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(
                tzinfo=timezone.utc)
            end = datetime.strptime(row["request_end_utc"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(
                tzinfo=timezone.utc)
            record_sha = row["record_sha256"]
            seg_station_es, day_refs = {}, []
            for segment in sorted(segments):
                seg_station_es[segment] = []
                for srow in segments[segment]:
                    candidates = list(srow["ordered_nslc_candidates"])
                    net = candidates[0].split(".")[0]
                    stas = [c.split(".")[1] for c in candidates]
                    chas = [c.split(".")[3] for c in candidates]
                    avail = providers.koeri_available(net, stas, chas, start, end) \
                        if provider == "KOERI" else providers.scedc_available(net, stas, chas,
                                                                              start, end)
                    nslc = None
                    for cand in candidates:
                        if cand in avail:
                            nslc = cand
                            break
                    attempt = {"carrier_key": carrier, "scored_day": day,
                               "segment_name": segment, "station_id": srow["station_id"],
                               "provider": provider, "request_start_utc": row["request_start_utc"],
                               "request_end_utc": row["request_end_utc"], "selected_nslc": None,
                               "status": "UNAVAILABLE", "input_object_sha256s": [],
                               "attempted_utc": now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                               "reason_codes": ["NO_AVAILABLE_CANDIDATE"]}
                    if nslc is None:
                        attempts.append(attempt)
                        continue
                    try:
                        res = providers.fetch(provider, nslc, start, end, stage_dir=stage_dir)
                    except providers.ProviderUnavailable:
                        attempt["reason_codes"] = ["PROVIDER_UNAVAILABLE"]
                        attempts.append(attempt)
                        continue
                    except Exception as exc:                     # noqa: BLE001
                        attempt["status"] = "ERROR"
                        attempt["reason_codes"] = [f"FETCH_ERROR:{type(exc).__name__}"]
                        attempts.append(attempt)
                        continue
                    stream, raw_objects = res["stream"], res["raw_objects"]
                    es, prov = _station_series(SD, stream, nslc, start)
                    refs = [ro["sha256"] for ro in raw_objects]
                    for ro in raw_objects:
                        input_objects.append({
                            "sha256": ro["sha256"], "size": ro["size_bytes"],
                            "relative_path": os.path.relpath(ro["staged_path"], root).replace(
                                os.sep, "/"),
                            "kind": "archive-seismic-miniseed-fragments-v1",
                            "carrier_key": carrier, "scored_day": day, "segment_name": segment,
                            "source_id": nslc, "start_utc": row["request_start_utc"],
                            "end_utc": row["request_end_utc"],
                            "native_rate_hz": prov["native_rate_hz"], "npts": prov["npts"],
                            "fragment_count": prov["fragment_count"],
                            "support_sha256": prov["support_sha256"],
                            "publication_record_sha256": record_sha})
                    attempt.update({"selected_nslc": nslc, "status": "FETCHED",
                                    "input_object_sha256s": refs, "reason_codes": []})
                    attempts.append(attempt)
                    day_refs.extend(refs)
                    seg_station_es[segment].append((nslc, es))
            base = _score_day(SD, FC, seg_station_es, start)
            base["publication_record_sha256"] = record_sha
            base["input_object_sha256s"] = day_refs
            day_result[(carrier, day)] = base

    # -- calibration_daily: 540 rows (2 arms x 3 carriers x 90 days) -----------
    daily_rows, operations = [], []
    for arm, arm_days in (("incident", incident_days), ("activation", activation_days)):
        for carrier in ELIGIBLE:
            for day in arm_days:
                base = day_result.get((carrier, day))
                if base is None:
                    row = {"arm": arm, "carrier_key": carrier, "day": day}
                    row.update(_no_record_daily())
                else:
                    row = {"arm": arm, "carrier_key": carrier, "day": day}
                    row.update(base)
                daily_rows.append(row)
                if row["status"] == "ADMITTED":
                    for op in OPS:
                        operations.append({
                            "arm": arm, "carrier_key": carrier, "day": day, "operation_name": op,
                            "input_sha256s": list(row["input_object_sha256s"]),
                            "output_sha256": _op_digest(op, carrier, day, arm,
                                                        row.get("support_sha256")),
                            "preconditions": ["published_phase_registered", "station_support>=0.5"],
                            "frame_change": ("native->1-10Hz-envelope-1Hz" if op == OPS[0]
                                             else "per-second-utc-grid" if op == OPS[1]
                                             else "segment-correlation-eigenspectrum"),
                            "information_lost": ["absolute_phase"] if op == OPS[0] else [],
                            "side_channel_retained": [], "claim_effects": [],
                            "validation_status": "PASS"})

    # -- prior evidence + replay (from the accepted sealed diagnostic bytes) ---
    with open(DIAGNOSTIC_FIXTURE, "rb") as fh:
        diag_bytes = fh.read()
    diagnostic = json.loads(diag_bytes.decode("utf-8"))
    diag_rel = "d2_diagnostic_result.json"
    with open(os.path.join(root, diag_rel), "wb") as fh:
        fh.write(diag_bytes)
    prior_evidence = {
        "schema": "geospec-d2-prior-evidence-v1",
        "diagnostic_result_path": diag_rel, "non_promotional": True,
        "lane_a_manifest_sha256": PRIOR["lane_a_manifest_sha256"],
        "diagnostic_manifest_sha256": PRIOR["diagnostic_manifest_sha256"],
        "diagnostic_result_sha256": PRIOR["diagnostic_result_sha256"]}

    def _phase_record(carrier, phase, scored_day):
        src = diagnostic.get("results", {}).get(carrier, {}).get(phase, {})
        if src.get("status") == "OK":
            return {"scored_day": scored_day, "status": "ADMITTED",
                    "ratio": src.get("lambda2_lambda1"),
                    "common_support_count": src.get("common_support_count"),
                    "correlation_matrix_sha256": _matrix_sha(src.get("correlation_matrix")),
                    "qc_reasons": []}
        return {"scored_day": scored_day, "status": "UNAVAILABLE", "ratio": None,
                "common_support_count": None, "correlation_matrix_sha256": None,
                "qc_reasons": ["NO_ACCEPTED_DIAGNOSTIC_FOR_CARRIER"]}

    replay_regions = []
    for target in TARGETS:
        carrier = target["carrier_key"]
        replay_regions.append({
            "runner_key": target["runner_key"], "carrier_key": carrier,
            "incident": _phase_record(carrier, "incident", "2026-07-29"),
            "control": _phase_record(carrier, "control", "2026-07-28")})
    replay_metrics = {
        "schema": "geospec-d2-segmented-replay-v1",
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "prior_evidence_sha256": _sha256_bytes(_canon(prior_evidence)),
        "incident_scored_day": "2026-07-29", "control_scored_day": "2026-07-28",
        "regions": replay_regions}
    replay_by_runner = {r["runner_key"]: r for r in replay_regions}

    # -- thresholds + admission (mirrors codex 0123 deterministically) ---------
    def _threshold(arm, carrier):
        vals = sorted(float(r["ratio"]) for r in daily_rows
                      if r["arm"] == arm and r["carrier_key"] == carrier
                      and r["status"] == "ADMITTED" and isinstance(r.get("ratio"), (int, float)))
        if len(vals) < POLICY["min_admitted_days"]:
            return None, len(vals)
        import math
        return vals[max(0, math.ceil(POLICY["lower_quantile"] * len(vals)) - 1)], len(vals)

    incident_ref = datetime.strptime("2026-07-29", "%Y-%m-%d").date()
    activation_ref = datetime.strptime(plan["activation_reference_day"], "%Y-%m-%d").date()
    incident_window = {"start": incident_days[0],
                       "end": (incident_ref - timedelta(days=30)).isoformat()}
    activation_window = {"start": activation_days[0],
                         "end": (activation_ref - timedelta(days=30)).isoformat()}
    input_manifest_sha_holder = {}
    admissions, registry = [], {}
    for target in TARGETS:
        runner = target["runner_key"]
        carrier = target["carrier_key"]
        base_row = {"runner_key": runner, "carrier_key": carrier, "topology_ok": True,
                    "topology_reasons": [], "incident_calibration_window": incident_window,
                    "activation_calibration_window": activation_window,
                    "incident_n": 0, "activation_n": 0, "incident_threshold": None,
                    "activation_threshold": None, "artifact_removed": False,
                    "control_clear": False, "capsule_path": None, "capsule_sha256": None,
                    "reason_codes": []}
        if target["base_disposition"] == "T2_SUPPLEMENT_REQUIRED":
            base_row.update({"status": "BLOCKED_TOPOLOGY", "topology_ok": False,
                             "topology_reasons": ["RIDGECREST_TIME_CONFOUND"],
                             "reason_codes": ["T2_SUPPLEMENT_REQUIRED"]})
            admissions.append(base_row)
            continue
        if target["base_disposition"] == "NO_TRUE_CARRIER":
            base_row.update({"status": "BLOCKED_NO_TRUE_CARRIER", "topology_ok": False,
                             "topology_reasons": ["NO_TRUE_KANTO_CARRIER"],
                             "reason_codes": ["NO_TRUE_KANTO_CARRIER"]})
            admissions.append(base_row)
            continue
        inc_thr, inc_n = _threshold("incident", carrier)
        act_thr, act_n = _threshold("activation", carrier)
        replay = replay_by_runner.get(runner, {})
        inc_rec, ctl_rec = replay.get("incident", {}), replay.get("control", {})
        artifact_removed = bool(inc_thr is not None and inc_rec.get("status") == "ADMITTED"
                                and inc_rec.get("ratio") is not None
                                and inc_rec["ratio"] >= inc_thr)
        control_clear = bool(inc_thr is not None and ctl_rec.get("status") == "ADMITTED"
                             and ctl_rec.get("ratio") is not None and ctl_rec["ratio"] >= inc_thr)
        base_row.update({"incident_n": inc_n, "activation_n": act_n,
                         "incident_threshold": inc_thr, "activation_threshold": act_thr,
                         "artifact_removed": artifact_removed, "control_clear": control_clear})
        if inc_thr is None or act_thr is None:
            status, reasons = "BLOCKED_INSUFFICIENT_CALIBRATION", ["INSUFFICIENT_ADMITTED_DAYS"]
        elif inc_rec.get("status") != "ADMITTED" or ctl_rec.get("status") != "ADMITTED":
            status, reasons = "BLOCKED_REPLAY_UNAVAILABLE", ["REPLAY_UNAVAILABLE"]
        elif not artifact_removed:
            status, reasons = "BLOCKED_ARTIFACT_PERSISTS", ["INCIDENT_BELOW_THRESHOLD"]
        elif not control_clear:
            status, reasons = "BLOCKED_NEGATIVE_CONTROL", ["CONTROL_BELOW_THRESHOLD"]
        else:
            status, reasons = "ADMITTED_CANDIDATE", []
        base_row.update({"status": status, "reason_codes": reasons})
        if status == "ADMITTED_CANDIDATE":
            capsule = {
                "schema": "geospec-d2-calibration-v1", "region": carrier, "band_tag": BAND_TAG,
                "processing_version": PROCESSING_VERSION, "topology_version": TOPOLOGY_VERSION,
                "threshold": act_thr, "calibration_window": activation_window,
                "source_commit": IMPLEMENTATION_COMMIT,
                "input_manifest_sha256": input_manifest_sha_holder,      # filled below (ref)
                "replay_output_sha256": _sha256_bytes(_canon(replay_metrics)),
                "issued_utc": now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "valid_through": (activation_ref + timedelta(
                    days=POLICY["candidate_valid_days"])).isoformat()}
            base_row["_capsule"] = capsule
            base_row["_capsule_carrier"] = carrier
        admissions.append(base_row)

    # -- input_manifest (needs producer_commit; git provenance) ----------------
    producer_commit = _git_head()
    clean_tree = _git_clean()
    input_manifest = {"schema": "geospec-d2-input-manifest-v2",
                      "producer_commit": producer_commit,
                      "implementation_commit": IMPLEMENTATION_COMMIT, "objects": input_objects}
    _write_json(os.path.join(root, "input_manifest.json"), input_manifest)
    input_manifest_sha = _sha256_file(os.path.join(root, "input_manifest.json"))
    input_manifest_sha_holder_value = input_manifest_sha

    # finalize capsules (now that input_manifest sha exists) + registry --------
    for row in admissions:
        capsule = row.pop("_capsule", None)
        carrier = row.pop("_capsule_carrier", None)
        if capsule is None:
            continue
        capsule["input_manifest_sha256"] = input_manifest_sha_holder_value
        rel = f"capsules/{carrier}_calibration.json"
        os.makedirs(os.path.join(root, "capsules"), exist_ok=True)
        _write_json(os.path.join(root, rel), capsule)
        capsule_sha = _sha256_file(os.path.join(root, rel))
        row["capsule_path"] = rel
        row["capsule_sha256"] = capsule_sha
        registry[carrier] = {"capsule_path": rel, "expected_sha256": capsule_sha,
                             "topology_version": TOPOLOGY_VERSION}

    admission_results = {"schema": "geospec-d2-step4b-admission-results-v1",
                         "implementation_commit": IMPLEMENTATION_COMMIT, "regions": admissions}

    # -- write the remaining artifacts -----------------------------------------
    _write_jsonl(os.path.join(root, "acquisition_attempts.jsonl"), attempts)
    _write_jsonl(os.path.join(root, "calibration_daily.jsonl"), daily_rows)
    _write_jsonl(os.path.join(root, "operation_ledger.jsonl"), operations)
    _write_json(os.path.join(root, "prior_evidence.json"), prior_evidence)
    _write_json(os.path.join(root, "replay_metrics.json"), replay_metrics)
    _write_json(os.path.join(root, "admission_results.json"), admission_results)
    _write_json(os.path.join(root, "registry_candidate.json"), registry)

    # -- batch_manifest (binds every file under root) --------------------------
    campaign_plan_sha = _sha256_file(os.path.join(root, "campaign_plan.json"))
    artifacts = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if rel == "batch_manifest.json":
                continue
            artifacts[rel] = {"sha256": _sha256_file(full), "size": os.path.getsize(full)}
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "numpy": np.__version__, "scipy": __import__("scipy").__version__,
                   "obspy": __import__("obspy").__version__}
    auth = {"status": receipt["status"],
            "in_session_timestamp_utc": receipt["in_session_timestamp_utc"],
            "owner_quote_sha256": receipt["owner_quote_sha256"]}
    batch_manifest = {
        "schema": "geospec-d2-step4b-batch-v1", "contract_id": CONTRACT_ID,
        "run_id": _op_digest("run", producer_commit, plan.get("registered_utc")),
        "producer_commit": producer_commit, "implementation_commit": IMPLEMENTATION_COMMIT,
        "clean_tree": clean_tree, "band_tag": BAND_TAG, "processing_version": PROCESSING_VERSION,
        "topology_version": TOPOLOGY_VERSION,
        "activation_reference_day": plan["activation_reference_day"],
        "incident_reference_day": "2026-07-29", "calibration_policy": POLICY, "targets": TARGETS,
        "environment": environment, "implementation_blobs": CORE_FILES, "artifacts": artifacts,
        "created_utc": now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "campaign_plan_sha256": campaign_plan_sha, "owner_launch_authorization": auth,
        "production_registry_modified": False, "production_freezes_modified": False,
        "non_claims": NON_CLAIMS}
    _write_json(os.path.join(root, "batch_manifest.json"), batch_manifest)
    candidates = [r["carrier_key"] for r in admissions if r["status"] == "ADMITTED_CANDIDATE"]
    return {"status": "BATCH_STAGED", "root": root, "candidates": candidates,
            "attempts": len(attempts), "daily_rows": len(daily_rows),
            "clean_tree": clean_tree, "producer_commit": producer_commit}
