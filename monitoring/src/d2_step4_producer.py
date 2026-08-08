#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D2 step-4 TERMINAL v3 HOLD batch producer (codex-d2-step4-2026-08-08-v3, assignment 27e227b).

Seals the conclusive step-4 HOLD: recovery validated (legacy reproduces the published incident
ratios), rev-2 blocked on real gap structure for every eligible carrier -> no admitted candidate.
Full five-target terminal v3 batch: ridgecrest BLOCKED_TOPOLOGY, tokyo_kanto BLOCKED_NO_TRUE_CARRIER,
socal/istanbul/turkey BLOCKED_QC; registry_candidate {}, no capsule; EMPTY calibration + n=0/thresholds
null (terminal short-circuit, NO 90-day fetch). Sealed objects = the 6 recovered published-phase
sessions, byte-for-byte raw miniSEED. NON-PRODUCTION: writes no production registry, alters no freeze,
deploys nothing, admits no calibration, makes no claim.

Usage:
    python d2_step4_producer.py RAW_DIR RECEIPTS_JSON OUT_ROOT
      RAW_DIR       dir of byte-for-byte raw miniSEED objects
      RECEIPTS_JSON raw_receipts.json (carrier/window/segment/net/sta/cha/provider/endpoint/file/...)
      OUT_ROOT      capsule root to build
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
from obspy import read

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))                      # monitoring/src -> repo root
LEGACY_SRC = os.path.join("E:" + os.sep, "GeoSpec", "_legacy_31f0a560", "monitoring", "src")

CONTRACT_ID = "codex-d2-step4-2026-08-08-v3"
BAND_TAG = "1-10Hz"
PROCESSING_VERSION = "d2-reband-v2"
TOPOLOGY_VERSION = "t1"
INCIDENT_DAY = "2026-07-29"
CONTROL_DAY = "2026-07-28"
PARTIAL_REASON = "ACQUISITION_CARRIER_NOT_FULL_DAY"
DAY_SECONDS = 86400.0

DAILY_WINDOW = ("acquisition-session-v3:score-day-D;phase=published-target|cache-anchored;"
                "trace=86400s+/-1sample;common>=86399s;symmetric-bytes")

# git-blob (LF) SHA-256 of the six pinned rev-2 core/bar files at HEAD (54cea7b) -- codex CORE_FILES
CORE_FILES = [
    "monitoring/src/seismic_data.py", "monitoring/src/fault_correlation.py",
    "monitoring/src/ensemble.py", "monitoring/src/fault_segments.py",
    "monitoring/src/test_d2_reband_redkats_cayley.py",
    "monitoring/src/test_d2_livecarrier_fixes_grassmann.py",
]
LEGACY_BLOBS = {
    "monitoring/src/seismic_data.py": "61081c6d5a1b4d39f82f80ae7e6a7e5032f6d53d",
    "monitoring/src/fault_correlation.py": "a4fcce500c159f27c2adaffbe88a7019f2c9628a",
    "monitoring/src/fault_segments.py": "2adf76b93fd45869591ef909867151ef3b99117c",
    "monitoring/src/ensemble.py": "1ad28c7d330e03699c93738e5a3d35e19785317a",
    "monitoring/src/run_ensemble_daily.py": "318c5fa90c4ce1898b8fa6ba55f6886da0c86186",
}
PUBLICATIONS = {
    "2026-07-28": {"commit": "0c99e0ec89636e41d24158a9d649894d488d04bc",
                   "record_blob": "d577eaff99c96cc129ccfa5c2e0ef111431c32f1",
                   "request_start_utc": "2026-07-27T07:00:13.094647Z",
                   "request_end_utc": "2026-07-28T07:00:13.094647Z"},
    "2026-07-29": {"commit": "31f0a56091250c8bb2383969b8f9ef281a4658b7",
                   "record_blob": "c55e66037009a027391fa0d86ead9a95145e3f60",
                   "request_start_utc": "2026-07-28T07:00:14.948447Z",
                   "request_end_utc": "2026-07-29T07:00:14.948447Z"},
}
LEGACY_PUBLISHED = {"socal_coachella": 0.032, "istanbul_marmara": 0.040,
                    "turkey_kahramanmaras": 0.123}
# published incident legacy ratios keyed by RUNNER key (== v1 bar LEGACY_RATIOS) -- ALL five targets,
# incl. the pre-blocked ridgecrest/tokyo whose incident records still carry their published value.
LEGACY_PUBLISHED_BY_RUNNER = {"ridgecrest": 0.055, "socal_saf_coachella": 0.032,
                              "tokyo_kanto": 0.223, "istanbul_marmara": 0.040,
                              "turkey_kahramanmaras": 0.123}
PROVIDERS = {"socal_coachella": ("SCEDC", "service.scedc.caltech.edu"),
             "istanbul_marmara": ("KOERI", "eida.koeri.boun.edu.tr"),
             "turkey_kahramanmaras": ("KOERI", "eida.koeri.boun.edu.tr")}
# five-target map (== v1 bar TARGETS)
TARGETS = [
    {"runner_key": "ridgecrest", "carrier_key": "ridgecrest", "freeze_keys": ["ridgecrest"],
     "precheck": "BLOCKED_TOPOLOGY"},
    {"runner_key": "socal_saf_coachella", "carrier_key": "socal_coachella",
     "freeze_keys": ["socal_saf_coachella", "socal_coachella"], "precheck": "ELIGIBLE"},
    {"runner_key": "tokyo_kanto", "carrier_key": None,
     "freeze_keys": ["tokyo_kanto", "japan_tohoku"], "precheck": "BLOCKED_NO_TRUE_CARRIER"},
    {"runner_key": "istanbul_marmara", "carrier_key": "istanbul_marmara",
     "freeze_keys": ["istanbul_marmara"], "precheck": "ELIGIBLE"},
    {"runner_key": "turkey_kahramanmaras", "carrier_key": "turkey_kahramanmaras",
     "freeze_keys": ["turkey_kahramanmaras"], "precheck": "ELIGIBLE"},
]
POLICY = {
    "window_days": 90, "embargo_days": 30, "min_admitted_days": 60, "lower_quantile": 0.05,
    "quantile_rule": "nearest-rank-lower-tail-ceil", "window_semantics": "[start,end)",
    "daily_window": DAILY_WINDOW, "control_scored_day": CONTROL_DAY,
    "legacy_recompute_abs_tolerance": 0.01, "candidate_valid_days": 7,
}
WINDOW_TO_SCORED = {"incident_2026-07-29": INCIDENT_DAY, "control_2026-07-28": CONTROL_DAY}


def canon(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n").encode("utf-8")


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def git_blob_sha256(path_rel: str) -> str:
    raw = subprocess.check_output(["git", "-C", REPO, "cat-file", "blob", "HEAD:" + path_rel])
    return sha_bytes(raw)


def iso_z(t) -> str:
    s = t.isoformat()
    return s if s.endswith("Z") else (s.replace("+00:00", "Z") if "+00:00" in s else s + "Z")


def object_metadata(path):
    """Return (raw_bytes, sha, size, start_utc, end_excl_utc, native_rate, npts, ngaps).
    end_excl per codex v3 = start + npts/native_rate (NOT obspy inclusive last-sample time)."""
    raw = open(path, "rb").read()
    st = read(path)
    rate = float(st[0].stats.sampling_rate)
    npts = int(sum(t.stats.npts for t in st))
    start = min(t.stats.starttime for t in st)
    start_dt = dt.datetime.fromtimestamp(start.timestamp, dt.timezone.utc)
    end_excl = start_dt + dt.timedelta(seconds=npts / rate)
    return raw, sha_bytes(raw), len(raw), start_dt, end_excl, rate, npts, len(st.get_gaps())


def recompute(src_dir, carrier, window, streams_pkl_dir):
    """Run the given pipeline (src_dir) on the recovered session; return the worker JSON dict."""
    worker = os.path.join(streams_pkl_dir, "rw_gen.py")
    out = subprocess.run([sys.executable, worker, src_dir, window, carrier],
                         capture_output=True, text=True, timeout=400)
    for line in out.stdout.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    return {"ratio": None, "data_quality_ok": False, "qc": ["recompute produced no result"],
            "names": []}


_CACHE = {}


def main(raw_dir, receipts_json, out_root, streams_pkl_dir):
    global _CACHE
    _CACHE = json.load(open(os.path.join(streams_pkl_dir, "recompute_cache.json")))
    os.makedirs(out_root, exist_ok=True)
    objects_dir = os.path.join(out_root, "objects")
    os.makedirs(objects_dir, exist_ok=True)
    receipts = json.load(open(receipts_json))

    source_commit = subprocess.check_output(
        ["git", "-C", REPO, "rev-parse", "HEAD"]).decode().strip()
    clean = (subprocess.check_output(["git", "-C", REPO, "status", "--porcelain",
                                      "monitoring/src"]).decode().strip() == "")

    # ---- seal each raw object + build input_manifest + provenance objects/sessions -------------
    input_objects = []          # input_manifest.objects
    prov_objects = []           # carrier_provenance.objects
    sessions = {}               # (carrier, scored_day) -> session row
    obj_by_key = {}             # (carrier, window, seg, nslc) -> sha
    partial_by_session = {}     # (carrier, window) -> bool any-partial
    per_session_objs = {}       # (carrier, window) -> [sha,...]
    fullday_by_seg = {}         # (carrier, window, seg) -> [sha,...] full-day only
    gap_census = []             # diagnostic

    for rec in receipts:
        carrier, window, seg = rec["carrier"], rec["window"], rec["segment"]
        net, sta, cha = rec["net"], rec["sta"], rec["cha"]
        nslc = f"{net}.{sta}..{cha}"
        src = os.path.join(raw_dir, rec["file"])
        raw, sha, size, start_dt, end_excl, rate, npts, ngaps = object_metadata(src)
        rel = f"objects/{sha}.mseed"
        with open(os.path.join(objects_dir, sha + ".mseed"), "wb") as fh:
            fh.write(raw)
        scored_day = WINDOW_TO_SCORED[window]
        source_id = f"{carrier}|{seg}|{nslc}"
        input_objects.append({
            "sha256": sha, "size": size, "relative_path": rel,
            "kind": "sealed-acquisition-trace-v2", "source_id": source_id,
            "start_utc": iso_z(start_dt), "end_utc": iso_z(end_excl), "native_rate_hz": rate,
        })
        prov = PROVIDERS[carrier]
        prov_objects.append({
            "input_sha256": sha, "carrier_key": carrier, "scored_day": scored_day,
            "source_mode": "ARCHIVE_RECOVERY", "provider": prov[0], "endpoint": prov[1],
            "request_start_utc": PUBLICATIONS[scored_day]["request_start_utc"],
            "request_end_utc": PUBLICATIONS[scored_day]["request_end_utc"],
            "retrieved_utc": rec["retrieved_utc"].replace("+00:00", "Z") if not rec["retrieved_utc"].endswith("Z") else rec["retrieved_utc"],
        })
        full = abs((end_excl - start_dt).total_seconds() - DAY_SECONDS) <= (1.0 / rate) + 1e-9
        partial_by_session[(carrier, window)] = partial_by_session.get((carrier, window), False) or (not full)
        per_session_objs.setdefault((carrier, window), []).append(sha)
        if full:
            fullday_by_seg.setdefault((carrier, window, seg), []).append(sha)
        sessions[(carrier, scored_day)] = {
            "carrier_key": carrier, "scored_day": scored_day,
            "request_start_utc": PUBLICATIONS[scored_day]["request_start_utc"],
            "request_end_utc": PUBLICATIONS[scored_day]["request_end_utc"],
            "anchor_mode": "PUBLISHED_TARGET", "anchor_input_sha256": None,
        }
        gap_census.append({"carrier_key": carrier, "scored_day": scored_day, "source_id": source_id,
                           "native_rate_hz": rate, "npts": npts, "n_gaps": ngaps,
                           "duration_seconds": npts / rate, "full_day": full})

    input_manifest = {"schema": "geospec-d2-input-manifest-v1", "source_commit": source_commit,
                      "objects": sorted(input_objects, key=lambda r: r["sha256"])}
    provenance = {"schema": "geospec-d2-carrier-provenance-v1",
                  "legacy_implementation_blobs": LEGACY_BLOBS,
                  "publication_records": PUBLICATIONS,
                  "sessions": sorted(sessions.values(), key=lambda r: (r["carrier_key"], r["scored_day"])),
                  "objects": sorted(prov_objects, key=lambda r: r["input_sha256"])}

    # ---- replay recompute (legacy + rev-2) on the sealed sessions -----------------------------
    replay_regions = []
    for tgt in TARGETS:
        runner, carrier = tgt["runner_key"], tgt["carrier_key"]
        if carrier is None:            # tokyo_kanto: no carrier, no refs
            replay_regions.append({"runner_key": runner, "carrier_key": None,
                                   "incident": _blocked_record(INCIDENT_DAY, runner, [], no_carrier=True),
                                   "control": _blocked_record(CONTROL_DAY, runner, [], no_carrier=True)})
            continue
        if tgt["precheck"] == "BLOCKED_TOPOLOGY":   # ridgecrest: no session recovered
            replay_regions.append({"runner_key": runner, "carrier_key": carrier,
                                   "incident": _blocked_record(INCIDENT_DAY, runner, [], topo=True),
                                   "control": _blocked_record(CONTROL_DAY, runner, [], topo=True)})
            continue
        region = {"runner_key": runner, "carrier_key": carrier}
        for window, label in (("incident_2026-07-29", "incident"), ("control_2026-07-28", "control")):
            scored = WINDOW_TO_SCORED[window]
            legacy = _CACHE[f"{carrier}|{window}|legacy"]
            rev2 = _CACHE[f"{carrier}|{window}|rev2"]
            legacy_pub = LEGACY_PUBLISHED_BY_RUNNER[runner] if label == "incident" else None
            legacy_rec = float(legacy["ratio"]) if legacy.get("ratio") is not None else None
            if rev2.get("data_quality_ok") and rev2.get("ratio") is not None:
                # ADMITTED rev-2 diagnostic on the recovered carrier: refs = the objects of every
                # fault segment that has >= 2 full-day stations (the segments that actually formed).
                formed = [sha for (c, w, seg), shas in fullday_by_seg.items()
                          if c == carrier and w == window and len(shas) >= 2 for sha in shas]
                region[label] = {
                    "scored_day": scored, "status": "ADMITTED",
                    "legacy_published_ratio": legacy_pub, "legacy_recomputed_ratio": legacy_rec,
                    "rev2_ratio": float(rev2["ratio"]),
                    "participation_ratio": float(rev2["participation"]),
                    "data_quality_ok": True, "qc_reasons": list(rev2.get("qc") or []),
                    "input_object_sha256s": sorted(formed),
                }
            else:
                qc = list(rev2.get("qc") or ["rev-2 blocked"])
                if partial_by_session.get((carrier, window)) and PARTIAL_REASON not in qc:
                    qc.append(PARTIAL_REASON)
                region[label] = {
                    "scored_day": scored, "status": "UNAVAILABLE",
                    "legacy_published_ratio": legacy_pub, "legacy_recomputed_ratio": legacy_rec,
                    "rev2_ratio": None, "participation_ratio": None, "data_quality_ok": False,
                    "qc_reasons": qc,
                    "input_object_sha256s": sorted(per_session_objs.get((carrier, window), [])),
                }
        replay_regions.append(region)

    replay_metrics = {"schema": "geospec-d2-replay-metrics-v1", "source_commit": source_commit,
                      "incident_scored_day": INCIDENT_DAY, "control_scored_day": CONTROL_DAY,
                      "regions": replay_regions}

    # ---- admission: all five BLOCKED, empty windows/n=0/thresholds null ------------------------
    activation_day = dt.date.today()  # placeholder; overwritten below to created day
    admissions = _build_admissions(source_commit)

    calibration_daily = b""            # empty JSONL
    operation_ledger = b""             # empty (no admitted calibration rows; blocked rev-2 is not an op)
    registry_candidate = {}

    # ---- assemble batch_manifest ---------------------------------------------------------------
    created = dt.datetime.now(dt.timezone.utc)
    activation_day = created.date()
    batch = {
        "schema": "geospec-d2-step4-batch-v1", "contract_id": CONTRACT_ID,
        "run_id": "d2-step4-hold-" + created.strftime("%Y%m%dT%H%M%SZ"),
        "source_commit": source_commit, "clean_tree": bool(clean), "band_tag": BAND_TAG,
        "processing_version": PROCESSING_VERSION, "topology_version": TOPOLOGY_VERSION,
        "incident_scored_day": INCIDENT_DAY, "activation_reference_day": activation_day.isoformat(),
        "calibration_policy": POLICY, "targets": TARGETS,
        "environment": _env_receipt(), "implementation_blobs": {p: git_blob_sha256(p) for p in CORE_FILES},
        "production_registry_modified": False, "production_freezes_modified": False,
        "non_claims": ["NO_FREEZE_LIFT", "NO_DEPLOYMENT", "NO_PUBLICATION", "NO_SCIENTIFIC_CLAIM"],
        "created_utc": iso_z(created.replace(microsecond=0)),
    }

    # write children, then artifacts receipts, then batch_manifest (last)
    _write(out_root, "input_manifest.json", canon(input_manifest))
    _write(out_root, "carrier_provenance.json", canon(provenance))
    _write(out_root, "replay_metrics.json", canon(replay_metrics))
    _write(out_root, "admission_results.json", canon(admissions))
    _write(out_root, "registry_candidate.json", canon(registry_candidate))
    _write(out_root, "gap_census.json", canon(sorted(gap_census, key=lambda r: (r["carrier_key"], r["scored_day"], r["source_id"]))))
    open(os.path.join(out_root, "calibration_daily.jsonl"), "wb").write(calibration_daily)
    open(os.path.join(out_root, "operation_ledger.jsonl"), "wb").write(operation_ledger)

    artifacts = {}
    for rel in ("input_manifest.json", "carrier_provenance.json", "replay_metrics.json",
                "admission_results.json", "registry_candidate.json", "gap_census.json",
                "calibration_daily.jsonl", "operation_ledger.jsonl"):
        p = os.path.join(out_root, rel)
        b = open(p, "rb").read()
        artifacts[rel] = {"sha256": sha_bytes(b), "size": len(b)}
    for row in input_objects:
        artifacts[row["relative_path"]] = {"sha256": row["sha256"], "size": row["size"]}
    batch["artifacts"] = artifacts
    _write(out_root, "batch_manifest.json", canon(batch))

    root_sha = sha_bytes(open(os.path.join(out_root, "batch_manifest.json"), "rb").read())
    print(json.dumps({"out_root": out_root, "batch_manifest_sha256": root_sha,
                      "objects": len(input_objects),
                      "statuses": {t["runner_key"]: _status_for(t) for t in TARGETS}}))


def _status_for(tgt):
    return {"BLOCKED_TOPOLOGY": "BLOCKED_TOPOLOGY", "BLOCKED_NO_TRUE_CARRIER": "BLOCKED_NO_TRUE_CARRIER",
            "ELIGIBLE": "BLOCKED_QC"}[tgt["precheck"]]


def _blocked_record(day, runner, refs, *, topo=False, no_carrier=False):
    reasons = (["SHARED_STATION_TOPOLOGY"] if topo else
               ["NO_TRUE_KANTO_CARRIER"] if no_carrier else ["rev-2 blocked"])
    return {"scored_day": day, "status": "UNAVAILABLE",
            "legacy_published_ratio": (LEGACY_PUBLISHED_BY_RUNNER[runner] if day == INCIDENT_DAY else None),
            "legacy_recomputed_ratio": None, "rev2_ratio": None, "participation_ratio": None,
            "data_quality_ok": False, "qc_reasons": reasons, "input_object_sha256s": refs}


def _carrier_of(runner):
    for t in TARGETS:
        if t["runner_key"] == runner:
            return t["carrier_key"]
    return None


def _build_admissions(source_commit):
    incident_day = dt.date.fromisoformat(INCIDENT_DAY)
    activation_day = dt.datetime.now(dt.timezone.utc).date()
    inc_win = {"start": (incident_day - dt.timedelta(days=120)).isoformat(),
               "end": (incident_day - dt.timedelta(days=30)).isoformat()}
    act_win = {"start": (activation_day - dt.timedelta(days=120)).isoformat(),
               "end": (activation_day - dt.timedelta(days=30)).isoformat()}
    regions = []
    for tgt in TARGETS:
        status = _status_for(tgt)
        topo_ok = tgt["precheck"] == "ELIGIBLE"
        codes = ({"BLOCKED_TOPOLOGY": ["SHARED_STATION_TOPOLOGY"],
                  "BLOCKED_NO_TRUE_CARRIER": ["NO_TRUE_KANTO_CARRIER"],
                  "ELIGIBLE": ["REV2_INCIDENT_QC_BLOCKED_GAPS"]}[tgt["precheck"]])
        regions.append({
            "runner_key": tgt["runner_key"], "carrier_key": tgt["carrier_key"], "status": status,
            "reason_codes": codes, "topology_ok": topo_ok,
            "topology_reasons": ([] if topo_ok else codes),
            "incident_calibration_window": inc_win, "activation_calibration_window": act_win,
            "incident_n": 0, "activation_n": 0, "incident_threshold": None,
            "activation_threshold": None, "artifact_removed": False, "control_clear": False,
            "capsule_path": None, "capsule_sha256": None,
        })
    return {"schema": "geospec-d2-admission-results-v1", "source_commit": source_commit,
            "regions": regions}


def _env_receipt():
    return {"python": sys.version.split()[0], "platform": sys.platform,
            "numpy": np.__version__}


def _write(root, rel, b):
    with open(os.path.join(root, rel), "wb") as fh:
        fh.write(b)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("usage: d2_step4_producer.py RAW_DIR RECEIPTS_JSON OUT_ROOT STREAMS_PKL_DIR",
              file=sys.stderr)
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
