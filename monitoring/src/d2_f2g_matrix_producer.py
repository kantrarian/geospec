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
IDENTITY_ARTIFACT = "f2g_producer_identity.json"
IDENTITY_SCHEMA = "f2g-producer-identity-v1"
_IDENTITY_KEYS = {"schema", "producer_commit", "producer_blob_map",
                  "algorithm_config_digest", "environment_lock_digest",
                  "clean_tree"}
# Real production input-manifest schemas: the frozen D2 station-series path ONLY
# (closure 2b). "im-v2-resume" is the bar's pinned token for the same dispatch.
REAL_SCHEMAS = {"im-v2-resume", "geospec-d2-step4b-input-manifest-v2-resume"}
_MANIFEST_KEYS = {
    "schema", "campaign_id", "run_id", "carrier_key", "day", "producer_commit",
    "producer_blob_map", "clean_tree", "algorithm_config_digest",
    "environment_lock_digest", "input_manifest_sha256", "input_objects",
    "station_index_digest", "station_ids", "matrix_shape", "matrix_dtype",
    "matrix_endianness", "n_overlap_policy", "n_overlap", "matrix_sha256",
    "matrix_size", "status", "reason_codes",
}
_BLOB_FILES = ("monitoring/src/seismic_data.py", "monitoring/src/fault_correlation.py",
               "monitoring/src/d2_step4b_providers.py",
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
    "real_schemas": sorted(REAL_SCHEMAS),
    "real_path": "d2_step4b_campaign_run._station_series/"
                 "seismic_data-band-envelope; pearson over common valid_mask; "
                 "n_overlap = common-valid count",
    "n_overlap_policy": "min-common-finite-samples",
    "diagonal": 1.0,
    "cross_carrier_pairs": False,
    "unavailable_rule": "typed-absence-never-zero",
    # codex R1.3 adoption 15dfa7bd (dual-path finding 6d039f65): the persisted
    # v2 surface is the oracle -- reproduce what production SCORED. Per-unit
    # stream assembly dispatches ONLY on the hash-bound ROOT input manifest's
    # reuse_disposition; the result manifest can never choose its own semantics.
    "acquisition_dispatch": {
        "authority": "root-input-manifest.reuse_disposition",
        "FETCHED_NEW": "provider-assembly",
        "LEGACY_REUSED_AFTER_REFETCH_ATTESTATION": "provider-assembly",
        "REUSED_VERIFIED": "d2_step4b_providers.parse_staged-single-object-untrimmed",
        "reused_multi_object": "REUSED_MULTI_OBJECT_UNSUPPORTED-typed-refusal",
        "unknown_or_mixed": "typed-refusal",
    },
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
    try:
        import scipy
        scipy_v = scipy.__version__
    except ImportError:
        scipy_v = "absent"
    try:
        import obspy
        obspy_v = getattr(obspy, "__version__", "unavailable")
    except ImportError:
        obspy_v = "absent"
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy_v, "obspy": obspy_v,
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
    """FIXTURE-SCHEMA ONLY (closure 2b): raw little-endian float64 sample arrays.
    Unreachable for any real input-manifest schema by construction."""
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


def _real_universe(im, carrier_key, day):
    """The FULL candidate object universe for (carrier, day) from a real-schema
    input manifest: every bound object, with station identity derived from
    source_id. This -- and never a result manifest -- is the eligibility
    authority (P16)."""
    out = []
    for o in im.get("objects", []):
        if o.get("carrier_key") != carrier_key or o.get("scored_day") != day:
            continue
        rec = dict(o)
        rec["station_id"] = ".".join(str(o.get("source_id", "")).split(".")[:2])
        rec["selected_nslc"] = o.get("source_id")
        rec["object_sha256"] = o.get("sha256")
        out.append(rec)
    out.sort(key=lambda r: (r["station_id"], r["object_sha256"]))
    return out


def _slim_record(rec):
    """The manifest input_objects record shape (single source of truth used by
    BOTH production and the P16 recompute reconstruction)."""
    out = {"station_id": rec["station_id"], "selected_nslc": rec["selected_nslc"],
           "object_sha256": rec["object_sha256"], "size": rec["size"]}
    if rec.get("relative_path"):
        out["relative_path"] = rec["relative_path"]
    return out


def _provider_of(root, rec, carrier_key):
    """P17 dispatch key: the object record's declared provider; else the pinned
    plan's per-carrier provider when a campaign_plan.json rides the root; else
    None (KOERI-style read+trim assembly, the conservative default)."""
    p = rec.get("provider")
    if p:
        return p
    plan_path = os.path.join(root, "campaign_plan.json")
    if os.path.exists(plan_path):
        try:
            plan = json.loads(open(plan_path, "rb").read().decode("utf-8"))
            return plan.get("providers", {}).get(carrier_key, {}).get("provider")
        except Exception:
            return None
    return None


_FRESH_DISPOSITIONS = {"FETCHED_NEW", "LEGACY_REUSED_AFTER_REFETCH_ATTESTATION"}


def _acquisition_lane(sid, recs):
    """Per-unit acquisition-history dispatch (codex adoption 15dfa7bd), keyed
    STRICTLY on the root-manifest records' reuse_disposition. Returns
    (lane, refusal_code): lane in {"fresh", "reused"} with code None, or lane
    None with a typed refusal code. absent = unknown = refusal; mixed
    dispositions within a unit refuse; a multi-object REUSED_VERIFIED unit
    refuses as loss guard -- objects[0] emulation may never silently discard
    bound bytes."""
    vals = sorted({str(r.get("reuse_disposition")) for r in recs})
    if len(vals) > 1:
        return None, "REUSE_DISPOSITION_MIXED:" + sid
    v = vals[0]
    if v in _FRESH_DISPOSITIONS:
        return "fresh", None
    if v == "REUSED_VERIFIED":
        if len(recs) != 1:
            return None, "REUSED_MULTI_OBJECT_UNSUPPORTED:" + sid
        return "reused", None
    return None, "REUSE_DISPOSITION_UNKNOWN:" + sid + ":" + v


def _assemble_stream(root, recs, provider, obspy):
    """P17: provider-dispatched post-staging assembly, mirroring the live fetch:
    SCEDC = read every day-volume + merge(method=0) + trim(request window) +
    split; KOERI/other = read + trim (live KOERI responses are window-limited;
    trim is the faithful no-op bound)."""
    from obspy import UTCDateTime
    stream = None
    for rec in sorted(recs, key=lambda o: o["object_sha256"]):
        p = _locate_object(root, rec)
        raw = open(p, "rb").read()
        if _sha(raw) != rec["object_sha256"]:
            raise ValueError("object byte drift for " + rec["station_id"])
        st = obspy.read(p)                         # typed failure if not miniSEED
        stream = st if stream is None else stream + st
    start, end = recs[0].get("start_utc"), recs[0].get("end_utc")
    if provider == "SCEDC":
        stream.merge(method=0)
        if start and end:
            stream.trim(UTCDateTime(start), UTCDateTime(end))
        stream = stream.split()
    elif start and end:
        stream.trim(UTCDateTime(start), UTCDateTime(end))
    return stream


def _derive_real(root, records_by_station, station_ids, session_start,
                 carrier_key=None):
    """REAL v2 schema (closures 2b + P17 + codex adoption 15dfa7bd): per-unit
    acquisition-history dispatch on the root manifest's reuse_disposition --
    fresh units through the provider assembly, REUSED_VERIFIED single-object
    units through the ACTUAL d2_step4b_providers.parse_staged seam untrimmed
    (the executor's resume semantics) -- then the FROZEN D2 station-series path
    (d2_step4b_campaign_run._station_series over the seismic_data band-envelope),
    then Pearson over the common valid_mask samples; n_overlap = the
    common-valid count. Returns (r, n_overlap, ineligible) with ineligible a
    dict {station_id: typed refusal code}. Raises typed errors when real
    dependencies are absent or bytes do not parse -- never a silent fixture
    read, never a wrong-frame answer."""
    try:
        import obspy
    except ImportError as exc:
        raise RuntimeError("REAL_DEPS_ABSENT:obspy") from exc
    import d2_step4b_campaign_run as CR
    import d2_step4b_providers as PV
    import seismic_data as SD

    es_by_station = {}
    ineligible = {}
    for sid in station_ids:
        recs = records_by_station[sid]
        nslc = recs[0]["selected_nslc"]
        lane, code = _acquisition_lane(sid, recs)
        if lane is None:
            # Dispatch refusal (unknown/mixed disposition or multi-object
            # REUSED): the station is never measured this day.
            ineligible[sid] = code
            continue
        if lane == "reused":
            rec = recs[0]
            p = _locate_object(root, rec)
            raw = open(p, "rb").read()
            if _sha(raw) != rec["object_sha256"]:
                raise ValueError("object byte drift for " + sid)
            stream = PV.parse_staged(p)
        else:
            provider = _provider_of(root, recs[0], carrier_key)
            stream = _assemble_stream(root, recs, provider, obspy)
        es = CR._station_series(SD, stream, nslc, session_start)
        if es is None:
            # The frozen path refused the series (quality/coverage gate). The
            # station is NOT eligible this day: typed absence from the measured
            # index -- never a zero/nan row for a station without a measurement.
            ineligible[sid] = "SERIES_UNAVAILABLE:" + sid
        else:
            es_by_station[sid] = es
    station_ids = [s for s in station_ids if s not in ineligible]

    n = len(station_ids)
    r = np.eye(n, dtype="<f8")
    nov = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a = es_by_station[station_ids[i]]
            b = es_by_station[station_ids[j]]
            common = a.valid_mask & b.valid_mask
            k = int(common.sum())
            if k >= 2:
                va, vb = a.values[common], b.values[common]
                if va.std() > 0 and vb.std() > 0:
                    r[i, j] = r[j, i] = float(np.corrcoef(va, vb)[0, 1])
                    nov[i][j] = nov[j][i] = k
                else:
                    r[i, j] = r[j, i] = np.nan
            else:
                r[i, j] = r[j, i] = np.nan
    return r, nov, ineligible


# -- identity AUTHORITY (closure 2a): the durable per-root artifact is the ONLY
# clearance; every verification (first call, any process) compares against the
# REOPENED artifact. The process-local first-acceptance registry below remains
# ONLY as an auxiliary same-process drift alarm, never clearance. -------------------
_RECORDED = {}


def _identity_fields(man):
    return (man.get("producer_commit"), _sha(_canon(man.get("producer_blob_map"))),
            man.get("algorithm_config_digest"), man.get("environment_lock_digest"))


def write_producer_identity(root):
    """Emit the canonical durable expected-identity artifact for this producer at
    <root>/f2g_producer_identity.json BEFORE matrix production. Refuses on a dirty
    tree (the artifact must bind clean_tree=true). Returns the artifact digest."""
    if not _git_clean():
        raise RuntimeError("IDENTITY_ARTIFACT_REQUIRES_CLEAN_TREE")
    doc = {"schema": IDENTITY_SCHEMA,
           "producer_commit": _git_head(),
           "producer_blob_map": _git_blob_map(),
           "algorithm_config_digest": ALGORITHM_CONFIG_DIGEST,
           "environment_lock_digest": environment_lock_digest(),
           "clean_tree": True}
    raw = _canon(doc)
    with open(os.path.join(root, IDENTITY_ARTIFACT), "wb") as fh:
        fh.write(raw)
    return _sha(raw)


def _load_identity_authority(root, refuse):
    """Reopen the durable identity artifact; absent/malformed = fail-closed."""
    p = os.path.join(root, IDENTITY_ARTIFACT)
    try:
        raw = open(p, "rb").read()
    except Exception:
        refuse("IDENTITY_AUTHORITY_ABSENT")
        return None
    try:
        doc = json.loads(raw.decode("utf-8"))
    except Exception:
        refuse("IDENTITY_AUTHORITY_UNREADABLE")
        return None
    if _canon(doc) != raw:
        refuse("IDENTITY_AUTHORITY_NOT_CANONICAL")
    if set(doc.keys()) != _IDENTITY_KEYS or doc.get("schema") != IDENTITY_SCHEMA:
        refuse("IDENTITY_AUTHORITY_KEYSET_DRIFT")
        return None
    if doc.get("clean_tree") is not True:
        refuse("IDENTITY_AUTHORITY_NOT_CLEAN_TREE")
    return doc


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

    # ---- identity AUTHORITY (closure 2a): the reopened durable artifact is the
    # only clearance -- first call and after restart alike ---------------------------
    authority = _load_identity_authority(root, refuse)
    if authority is not None:
        if man.get("producer_commit") != authority.get("producer_commit"):
            refuse("IDENTITY_MISMATCH_VS_AUTHORITY:producer_commit")
        if man.get("producer_blob_map") != authority.get("producer_blob_map"):
            refuse("IDENTITY_MISMATCH_VS_AUTHORITY:producer_blob_map")
        if man.get("algorithm_config_digest") != authority.get(
                "algorithm_config_digest"):
            refuse("IDENTITY_MISMATCH_VS_AUTHORITY:algorithm_config_digest")
        if man.get("environment_lock_digest") != authority.get(
                "environment_lock_digest"):
            refuse("IDENTITY_MISMATCH_VS_AUTHORITY:environment_lock_digest")

    # ---- auxiliary same-process drift alarm (never clearance) ----------------------
    key = (os.path.normcase(os.path.abspath(root)), man.get("carrier_key"),
           man.get("day"))
    ident = _identity_fields(man)
    recorded = _RECORDED.get(key)
    if recorded is not None and recorded != ident:
        refuse("PRODUCER_IDENTITY_DRIFT_VS_RECORDED")

    # ---- derivation oracle (schema-dispatched; fixture reader NEVER runs for a
    # real-schema root) ---------------------------------------------------------------
    if recompute and not reasons:
        try:
            im_doc = json.loads(open(im_path, "rb").read().decode("utf-8"))
            schema = im_doc.get("schema", "")
            if schema == ALGORITHM_CONFIG["fixture_schema"]:
                r2, _nov2 = _derive_fixture(root, objs, sids)
            elif schema in REAL_SCHEMAS:
                # P16: rebuild the FULL candidate universe from the REOPENED
                # hash-bound root manifest -- the result manifest NEVER decides
                # its own measurement domain.
                universe = _real_universe(im_doc, man.get("carrier_key"),
                                          man.get("day"))
                if not universe:
                    raise ValueError("EMPTY_CANDIDATE_UNIVERSE")
                cand_sids = sorted({o["station_id"] for o in universe})
                starts = sorted(o.get("start_utc") for o in universe
                                if o.get("start_utc"))
                if not starts:
                    raise ValueError("REAL_SCHEMA_WITHOUT_SESSION_START")
                from datetime import datetime, timezone
                session_start = datetime.strptime(
                    starts[0], "%Y-%m-%dT%H:%M:%S.%fZ").replace(
                        tzinfo=timezone.utc)
                by_station = {}
                for o in universe:
                    by_station.setdefault(o["station_id"], []).append(o)
                r2, nov2, inel = _derive_real(
                    root, by_station, cand_sids, session_start,
                    carrier_key=man.get("carrier_key"))
                eligible = [s for s in cand_sids if s not in inel]
                exp_objs = sorted((_slim_record(o) for o in universe
                                   if o["station_id"] not in inel),
                                  key=lambda o: (o["station_id"],
                                                 o["object_sha256"]))
                exp_codes = sorted(inel.values())
                exp_status = "PRODUCED" if len(eligible) >= 2 \
                    else "INSUFFICIENT_ELIGIBLE_STATIONS"
                if man.get("station_ids") != eligible:
                    refuse("ELIGIBILITY_SET_MISMATCH")
                if man.get("input_objects") != exp_objs:
                    refuse("OBJECT_SET_MISMATCH")
                # exact equality: the recomputed refusal codes (frozen-gate
                # SERIES_UNAVAILABLE + dispatch refusals) and NOTHING else
                if sorted(man.get("reason_codes") or []) != exp_codes:
                    refuse("ELIGIBILITY_REASON_MISMATCH")
                if man.get("status") != exp_status:
                    refuse("ELIGIBILITY_STATUS_MISMATCH")
                if man.get("n_overlap") != nov2:
                    refuse("N_OVERLAP_MISMATCH")
            else:
                raise ValueError(f"UNSUPPORTED_INPUT_MANIFEST_SCHEMA:{schema}")
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

    session_start = None
    if schema == ALGORITHM_CONFIG["fixture_schema"]:
        objs = [dict(o) for o in im["objects"]]
        campaign_id = run_id = _sha(im_raw)
    elif schema in REAL_SCHEMAS:
        objs = _real_universe(im, carrier_key, day)
        starts = sorted(o.get("start_utc") for o in objs if o.get("start_utc"))
        if starts:
            from datetime import datetime, timezone
            session_start = datetime.strptime(
                starts[0], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
        bm_path = os.path.join(root, "batch_manifest.json")
        bmj = json.loads(open(bm_path, "rb").read().decode("utf-8")) \
            if os.path.exists(bm_path) else {}
        campaign_id = bmj.get("campaign_id", _sha(im_raw))
        run_id = bmj.get("run_id", campaign_id)
    else:
        raise ValueError(f"UNSUPPORTED_INPUT_MANIFEST_SCHEMA:{schema}")

    objs.sort(key=lambda o: (o["station_id"], o["object_sha256"]))
    station_ids = sorted({o["station_id"] for o in objs})
    status, codes = "PRODUCED", []
    if not station_ids:
        status, codes = "UNAVAILABLE", ["NO_BOUND_OBJECTS"]
        r, nov = np.zeros((0, 0), dtype="<f8"), []
    elif schema == ALGORITHM_CONFIG["fixture_schema"]:
        r, nov = _derive_fixture(root, objs, station_ids)
    else:
        by_station = {}
        for o in objs:
            by_station.setdefault(o["station_id"], []).append(o)
        if session_start is None:
            raise ValueError("REAL_SCHEMA_WITHOUT_SESSION_START")
        r, nov, ineligible = _derive_real(root, by_station, station_ids,
                                          session_start,
                                          carrier_key=carrier_key)
        if ineligible:
            # typed absence: frozen-gate refusal (SERIES_UNAVAILABLE) or
            # acquisition-dispatch refusal (unknown/mixed disposition,
            # multi-object REUSED loss guard); the station leaves the measured
            # index AND the used-objects record.
            station_ids = [s for s in station_ids if s not in ineligible]
            codes += sorted(ineligible.values())
        objs = [_slim_record(o) for o in objs
                if o["station_id"] not in ineligible]
        objs.sort(key=lambda o: (o["station_id"], o["object_sha256"]))
        if len(station_ids) < 2:
            status = "INSUFFICIENT_ELIGIBLE_STATIONS"
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
