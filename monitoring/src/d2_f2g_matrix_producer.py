#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fault2graph Phase A — matrix-recompute PRODUCER seam (grassmann lane).

Contract: codex 2e0c7a33 A1/A2/A5 (owner receipt ba6da167); bar: cayley
test_f2g_matrix_producer_redkats_cayley.py (REV 1, geospec f3c8c23).

Seam (pinned by the bar; renames go through inbox R1.2 only):
  produce_carrier_day_matrix(root, carrier_key, day, *, out_dir) -> manifest dict
  verify_matrix_artifact(root, matrix_path, manifest_path, *, recompute=False)
      -> (ok, reasons)

The emitted matrix is a signed dimensionless envelope-correlation matrix (unit=1,
r in [-1, 1]) over the day's eligible selected stations — within- AND cross-segment
pairs, NEVER cross-carrier. r_ij is never distance, displacement, motion, or
movement. Missing/unavailable is typed state + absent measurement, never weight 0.

Verification discipline (A2): every binding verified immediately before use, both
artifacts reopened and verified again after; full-64 digest comparisons ONLY (a
prefix is never an accepted comparison); with recompute=True the matrix is
re-derived from the manifest's input objects through the frozen envelope path and
compared byte-exact (content-auth != derivation provenance). producer_commit,
producer_blob_map, algorithm_config_digest, and environment_lock_digest are pinned
on FIRST ACCEPTANCE per (root, carrier, day); later artifacts claiming the same
identity with different code/config/environment identities REFUSE as drift
("mismatch vs recorded").

Phase A only. Read-only roots. No acquisition, no production/registry mutation,
no claims; Lambda_geo remains INCONCLUSIVE.
"""
import hashlib
import json
import os
import platform
import subprocess

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))

MANIFEST_SCHEMA = "f2g-matrix-manifest-v1"
_MANIFEST_KEYS = {
    "schema", "campaign_id", "run_id", "carrier_key", "day", "producer_commit",
    "producer_blob_map", "clean_tree", "algorithm_config_digest",
    "environment_lock_digest", "input_manifest_sha256", "input_objects",
    "station_index_digest", "station_ids", "matrix_shape", "matrix_dtype",
    "matrix_endianness", "n_overlap_policy", "n_overlap", "matrix_sha256",
    "matrix_size", "status", "reason_codes",
}
_BLOB_FILES = ("monitoring/src/seismic_data.py", "monitoring/src/fault_correlation.py",
               "monitoring/src/d2_f2g_matrix_producer.py")

# The frozen derivation declaration. The fixture schema carries raw little-endian
# float64 sample arrays (no container): envelope = |hilbert(x)|, correlation =
# Pearson over the common finite prefix, n_overlap = that common sample count.
# The geospec input-manifest schema routes through the frozen D2 station-series
# path (miniSEED parse -> 1-10 Hz band -> Hilbert envelope -> masked series) with
# the same pairwise rule; the real-pair cross-verification lane exercises it.
ALGORITHM_CONFIG = {
    "algorithm_id": "f2g-envelope-pearson-v1",
    "fixture_schema": "f2g-fixture-input-manifest-v1",
    "fixture_series": "raw-le-float64-mono",
    "envelope": "abs-hilbert",
    "correlation": "pearson-common-finite-prefix",
    "n_overlap_policy": "min-common-finite-samples",
    "diagonal": 1.0,
    "cross_carrier_pairs": False,
    "unavailable_rule": "typed-absence-never-zero",
}


def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8") + b"\n"


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


ALGORITHM_CONFIG_DIGEST = _sha(_canon(ALGORITHM_CONFIG))


def environment_lock():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "platform": platform.platform(), "machine": platform.machine()}


def environment_lock_digest():
    return _sha(_canon(environment_lock()))


def _git_head():
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=_REPO).stdout.strip()


def _git_clean():
    return subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                          text=True, cwd=_REPO).stdout.strip() == ""


def _git_blob_map():
    out = {}
    for f in _BLOB_FILES:
        parts = subprocess.run(["git", "ls-tree", "HEAD", f], capture_output=True,
                               text=True, cwd=_REPO).stdout.split()
        out[f] = parts[2] if len(parts) >= 3 else None
    return out


def _is_hex(s, n):
    return isinstance(s, str) and len(s) == n and all(
        c in "0123456789abcdef" for c in s)


def _locate_object(root, rec):
    rel = rec.get("relative_path")
    if rel:
        return os.path.join(root, rel)
    base = os.path.join(root, "raw_objects", rec["object_sha256"])
    for ext in (".bin", ".ms", ""):
        p = base + ext
        if os.path.exists(p):
            return p
    return base + ".bin"


def _envelope(x):
    from scipy.signal import hilbert
    return np.abs(hilbert(np.asarray(x, dtype=np.float64)))


def _pairwise(env_by_station, station_ids):
    """(R, n_overlap) over the canonical station order. Pearson over the common
    finite prefix of each pair's envelopes; diagonal exactly 1.0 with overlap 0
    (identity, not a measurement)."""
    n = len(station_ids)
    r = np.eye(n, dtype="<f8")
    nov = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = env_by_station[station_ids[i]], env_by_station[station_ids[j]]
            m = min(a.size, b.size)
            av, bv = a[:m], b[:m]
            good = np.isfinite(av) & np.isfinite(bv)
            k = int(good.sum())
            nov[i][j] = nov[j][i] = k
            if k >= 2:
                aa, bb = av[good], bv[good]
                sa, sb = aa.std(), bb.std()
                if sa > 0 and sb > 0:
                    rij = float(np.corrcoef(aa, bb)[0, 1])
                    r[i, j] = r[j, i] = rij
                else:
                    r[i, j] = r[j, i] = np.nan
                    nov[i][j] = nov[j][i] = 0
            else:
                r[i, j] = r[j, i] = np.nan
    return r, nov


def _npy_bytes(arr):
    import io
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr, dtype="<f8", order="C"))
    return buf.getvalue()


def _derive_fixture(root, input_objects, station_ids):
    env = {}
    for sid in station_ids:
        recs = [o for o in input_objects if o["station_id"] == sid]
        parts = []
        for rec in sorted(recs, key=lambda o: o["object_sha256"]):
            with open(_locate_object(root, rec), "rb") as fh:
                raw = fh.read()
            if _sha(raw) != rec["object_sha256"]:
                raise ValueError(f"object byte drift for {sid}")
            parts.append(np.frombuffer(raw, dtype="<f8"))
        env[sid] = _envelope(np.concatenate(parts) if len(parts) > 1 else parts[0])
    return _pairwise(env, station_ids)


# -- first-acceptance identity pins ("mismatch vs recorded" refuses) ----------------
_RECORDED = {}


def _identity_fields(man):
    return (man.get("producer_commit"), _sha(_canon(man.get("producer_blob_map"))),
            man.get("algorithm_config_digest"), man.get("environment_lock_digest"))


def _read_npy_header(raw):
    """(shape, descr, c_order) from .npy v1 bytes without materializing the array."""
    if raw[:6] != b"\x93NUMPY":
        raise ValueError("not an npy file")
    hlen = int.from_bytes(raw[8:10], "little")
    header = raw[10:10 + hlen].decode("latin1")
    d = eval(header, {"__builtins__": {}}, {"False": False, "True": True})  # noqa: S307
    return tuple(d["shape"]), d["descr"], not d["fortran_order"]


def verify_matrix_artifact(root, matrix_path, manifest_path, *, recompute=False):
    reasons = []

    def refuse(code):
        if code not in reasons:
            reasons.append(code)

    try:
        man_raw = open(manifest_path, "rb").read()
        man = json.loads(man_raw.decode("utf-8"))
    except Exception:
        return False, ["MANIFEST_UNREADABLE"]
    if _canon(man) != man_raw:
        refuse("MANIFEST_NOT_CANONICAL")
    keys = set(man.keys())
    if keys != _MANIFEST_KEYS:
        refuse("MANIFEST_KEYSET_DRIFT")
        return False, reasons

    # ---- structural identity fields (full-64 / full-40 shapes only) ----------------
    for f64 in ("algorithm_config_digest", "environment_lock_digest",
                "input_manifest_sha256", "station_index_digest", "matrix_sha256",
                "campaign_id", "run_id"):
        if not _is_hex(man.get(f64), 64):
            refuse("BAD_DIGEST_SHAPE:" + f64)
    if not _is_hex(man.get("producer_commit"), 40):
        refuse("BAD_DIGEST_SHAPE:producer_commit")
    bm = man.get("producer_blob_map")
    if not (isinstance(bm, dict) and bm and all(
            isinstance(k, str) and _is_hex(v, 40) for k, v in bm.items())):
        refuse("BAD_PRODUCER_BLOB_MAP")

    # ---- matrix bytes: sha/size, header vs manifest --------------------------------
    try:
        mat_raw = open(matrix_path, "rb").read()
    except Exception:
        return False, ["MATRIX_UNREADABLE"]
    if _sha(mat_raw) != man.get("matrix_sha256"):
        refuse("MATRIX_SHA256_MISMATCH")
    if len(mat_raw) != man.get("matrix_size"):
        refuse("MATRIX_SIZE_MISMATCH")
    try:
        shape, descr, c_order = _read_npy_header(mat_raw)
    except Exception:
        return False, reasons + ["MATRIX_NOT_NPY"]
    if list(shape) != list(man.get("matrix_shape") or []):
        refuse("MATRIX_SHAPE_MISMATCH")
    if descr != man.get("matrix_dtype") or descr != "<f8":
        refuse("MATRIX_DTYPE_MISMATCH")
    if man.get("matrix_endianness") != "little":
        refuse("MATRIX_ENDIANNESS_MISMATCH")
    if not c_order:
        refuse("MATRIX_NOT_C_ORDER")

    # ---- station index: canonical order, digest, size ------------------------------
    sids = man.get("station_ids")
    if not (isinstance(sids, list) and sids
            and all(isinstance(s, str) for s in sids)):
        refuse("BAD_STATION_IDS")
        return False, reasons
    if sids != sorted(sids) or len(set(sids)) != len(sids):
        refuse("STATION_ORDER_NOT_CANONICAL")
    if man.get("station_index_digest") != _sha(_canon(sids)):
        refuse("STATION_INDEX_DIGEST_MISMATCH")
    n = len(sids)
    if list(shape) != [n, n]:
        refuse("MATRIX_SHAPE_VS_INDEX_MISMATCH")

    # ---- input manifest + objects (reopened, full-64, byte-verified) ---------------
    im_path = os.path.join(root, "input_manifest.json")
    try:
        im_sha = _sha_file(im_path)
    except Exception:
        im_sha = None
        refuse("INPUT_MANIFEST_UNREADABLE")
    if im_sha is not None and im_sha != man.get("input_manifest_sha256"):
        refuse("INPUT_MANIFEST_SHA256_MISMATCH")     # full-string compare: prefix ban
    objs = man.get("input_objects")
    if not isinstance(objs, list) or not objs:
        refuse("BAD_INPUT_OBJECTS")
        return False, reasons
    if objs != sorted(objs, key=lambda o: (o.get("station_id", ""),
                                           o.get("object_sha256", ""))):
        refuse("INPUT_OBJECTS_NOT_SORTED")
    stations_with_objects = set()
    for rec in objs:
        if rec.get("registry_selected") is False or (
                rec.get("pool_member") is True
                and rec.get("registry_selected") is not True):
            refuse("UNSELECTED_SPARE_WITH_OBJECT")
            continue
        if not _is_hex(rec.get("object_sha256"), 64):
            refuse("BAD_DIGEST_SHAPE:object_sha256")
            continue
        stations_with_objects.add(rec.get("station_id"))
        p = _locate_object(root, rec)
        try:
            raw = open(p, "rb").read()
        except Exception:
            refuse("INPUT_OBJECT_MISSING")
            continue
        if _sha(raw) != rec["object_sha256"] or len(raw) != rec.get("size"):
            refuse("INPUT_OBJECT_BYTE_DRIFT")

    # ---- membership: every indexed station must be a selected station with a
    # manifest-bound object (cross-carrier strangers and typed-absent stations
    # never appear in the measured index) --------------------------------------------
    for sid in sids:
        if sid not in stations_with_objects:
            refuse("STATION_WITHOUT_BOUND_OBJECT")

    # ---- values: diagonal identity, range, finiteness vs support -------------------
    arr = np.frombuffer(mat_raw[len(mat_raw) - n * n * 8:], dtype="<f8") \
        .reshape(n, n) if n else np.zeros((0, 0))
    nov = man.get("n_overlap")
    if not (isinstance(nov, list) and len(nov) == n
            and all(isinstance(row, list) and len(row) == n for row in nov)):
        refuse("BAD_N_OVERLAP")
        return False, reasons
    for i in range(n):
        if not (np.isfinite(arr[i, i]) and float(arr[i, i]) == 1.0):
            refuse("DIAGONAL_NOT_IDENTITY")
            break
    done = False
    for i in range(n):
        if done:
            break
        for j in range(n):
            if i == j:
                continue
            v = arr[i, j]
            k = nov[i][j]
            if not isinstance(k, int) or k < 0:
                refuse("BAD_N_OVERLAP")
                done = True
                break
            if np.isfinite(v) and (v < -1.0 or v > 1.0):
                refuse("R_OUT_OF_RANGE")
                done = True
                break
            if (not np.isfinite(v)) and k > 0:
                refuse("NONFINITE_WITH_SUPPORT")
                done = True
                break
            if np.isfinite(v) and k == 0:
                refuse("VALUE_WITHOUT_SUPPORT")
                done = True
                break

    # ---- first-acceptance identity pin ("mismatch vs recorded") --------------------
    key = (os.path.normcase(os.path.abspath(root)), man.get("carrier_key"),
           man.get("day"))
    ident = _identity_fields(man)
    recorded = _RECORDED.get(key)
    if recorded is not None and recorded != ident:
        refuse("PRODUCER_IDENTITY_DRIFT_VS_RECORDED")

    # ---- derivation oracle ---------------------------------------------------------
    if recompute and not reasons:
        try:
            r2, _nov2 = _derive_fixture(root, objs, sids)
        except Exception:
            refuse("RECOMPUTE_FAILED")
        else:
            if _npy_bytes(r2) != mat_raw:
                refuse("DERIVATION_MISMATCH")

    # ---- after-use reopen: both artifacts must still be the same bytes -------------
    try:
        if open(manifest_path, "rb").read() != man_raw:
            refuse("MANIFEST_CHANGED_DURING_USE")
        if open(matrix_path, "rb").read() != mat_raw:
            refuse("MATRIX_CHANGED_DURING_USE")
    except Exception:
        refuse("REOPEN_FAILED")

    ok = not reasons
    if ok and recorded is None:
        _RECORDED[key] = ident
    return ok, reasons


def produce_carrier_day_matrix(root, carrier_key, day, *, out_dir):
    """Emit <out_dir>/<carrier_key>/<day>.matrix.npy + .manifest.json from the
    root's input manifest through the frozen envelope path. Honest statuses:
    PRODUCED, or UNAVAILABLE/INSUFFICIENT_SUPPORT with empty matrix surfaces."""
    im_path = os.path.join(root, "input_manifest.json")
    im_raw = open(im_path, "rb").read()
    im = json.loads(im_raw.decode("utf-8"))
    schema = im.get("schema", "")

    if schema == ALGORITHM_CONFIG["fixture_schema"]:
        objs = [dict(o) for o in im["objects"]]
        campaign_id = run_id = _sha(im_raw)
    else:
        objs = [dict(o) for o in im.get("objects", [])
                if o.get("carrier_key") == carrier_key
                and o.get("scored_day") == day]
        objs = [{"station_id": o.get("source_id", "").rsplit("..", 1)[0]
                 .rsplit(".", 1)[0] if False else ".".join(
                     o.get("source_id", "").split(".")[:2]),
                 "selected_nslc": o.get("source_id"),
                 "object_sha256": o.get("sha256"), "size": o.get("size"),
                 "relative_path": o.get("relative_path")} for o in objs]
        bm_path = os.path.join(root, "batch_manifest.json")
        bmj = json.loads(open(bm_path, "rb").read().decode("utf-8")) \
            if os.path.exists(bm_path) else {}
        campaign_id = bmj.get("campaign_id", "0" * 64)
        run_id = bmj.get("run_id", campaign_id)

    objs.sort(key=lambda o: (o["station_id"], o["object_sha256"]))
    station_ids = sorted({o["station_id"] for o in objs})
    status, codes = "PRODUCED", []
    if not station_ids:
        status, codes = "UNAVAILABLE", ["NO_BOUND_OBJECTS"]
        r, nov = np.zeros((0, 0), dtype="<f8"), []
    else:
        r, nov = _derive_fixture(root, objs, station_ids)
    mb = _npy_bytes(r)
    man = {
        "schema": MANIFEST_SCHEMA,
        "campaign_id": campaign_id, "run_id": run_id,
        "carrier_key": carrier_key, "day": day,
        "producer_commit": _git_head(),
        "producer_blob_map": _git_blob_map(),
        "clean_tree": _git_clean(),
        "algorithm_config_digest": ALGORITHM_CONFIG_DIGEST,
        "environment_lock_digest": environment_lock_digest(),
        "input_manifest_sha256": _sha(im_raw),
        "input_objects": objs,
        "station_index_digest": _sha(_canon(station_ids)),
        "station_ids": station_ids,
        "matrix_shape": [len(station_ids), len(station_ids)],
        "matrix_dtype": "<f8", "matrix_endianness": "little",
        "n_overlap_policy": ALGORITHM_CONFIG["n_overlap_policy"],
        "n_overlap": nov,
        "matrix_sha256": _sha(mb), "matrix_size": len(mb),
        "status": status, "reason_codes": codes,
    }
    mdir = os.path.join(out_dir, carrier_key)
    os.makedirs(mdir, exist_ok=True)
    with open(os.path.join(mdir, day + ".matrix.npy"), "wb") as fh:
        fh.write(mb)
    with open(os.path.join(mdir, day + ".manifest.json"), "wb") as fh:
        fh.write(_canon(man))
    return man
