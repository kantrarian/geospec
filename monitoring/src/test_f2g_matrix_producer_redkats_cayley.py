"""RED-first KATs -- fault2graph Phase A MATRIX PRODUCER bar (cayley).

REV 2 (codex verify-once `4b365e4e` closures 2a/2b; grassmann ACK `479eddc0`):
  IDENTITY AUTHORITY (2a -- TOFU is not attestation): a canonical DURABLE
  expected-identity artifact `<root>/f2g_producer_identity.json` (schema
  f2g-producer-identity-v1: producer_commit, producer_blob_map,
  algorithm_config_digest, environment_lock_digest, clean_tree=true; canonical
  bytes) is the ONLY identity clearance. Every verification -- first call and
  after process restart -- compares the manifest's identity fields to the
  REOPENED artifact. Process-local first-acceptance may remain as an auxiliary
  same-process drift alarm, never clearance.
    P8a  invented identity REFUSES on the FIRST call (no prior acceptance)
    P8b  identity artifact ABSENT -> fail-closed REFUSAL (no authority, no pass)
    P9a  post-restart (fresh subprocess): valid identity PASSES, invented
         identity REFUSES -- the durable artifact is the authority, not memory
  REAL-PATH DISPATCH (2b -- the fixture reader must be unreachable for real
  data): input-manifest schema dispatch is pinned -- fixture schema
  `f2g-fixture-input-manifest-v1` alone may use `_derive_fixture`; the real v2
  schema `im-v2-resume` MUST parse miniSEED through the frozen D2 path
  (`d2_step4b_campaign_run._station_series` ->
  `seismic_data.compute_band_envelope_supported`) with ObsPy/SciPy in the
  environment lock; any other schema is a typed refusal.
    P15a (runs everywhere) real-schema input NEVER reaches `_derive_fixture`
         (spy) and NEVER yields a silently fixture-derived matrix; with real
         deps absent the outcome must be a typed refusal, not a wrong answer
    P15b (obspy-gated) small real-format miniSEED fixture: the produced matrix
         equals an INDEPENDENTLY INVOKED frozen-D2 oracle (envelopes via
         _station_series/compute_band_envelope_supported; pearson over the
         common valid_mask samples; n_overlap = common-valid count). ORACLE
         NOTE: authored on an obspy-less host -- if the frozen correlation
         semantic differs from pearson-over-common-valid, R1.2 the exact
         frozen call through inbox and the bar amends; the reachability lock
         (P15a) is load-bearing regardless.
  STATUS: these classes are EXPECTED RED against producer `d3e76f4` (grassmann
  runs the red proof, lands repairs bar-unedited, then green). P1-P14 unchanged.

REV 1 (codex Phase-A contract `2e0c7a33` A2/A5; owner receipt ba6da167; design note
`e123c2c` V-A). Cross-authorship per A5: grassmann implements the producer, THIS bar
is cayley's red-KAT surface for it. Codex reviews the builder separately.

PINNED SEAM (the producer implements EXACTLY this; naming R1.2 goes through inbox,
never a silent rename):
  module  monitoring/src/d2_f2g_matrix_producer.py
  fn      produce_carrier_day_matrix(root, carrier_key, day, *, out_dir) -> manifest dict
          emits <out_dir>/<carrier_key>/<day>.matrix.npy  (numpy .npy v1, C-order,
          little-endian float64 '<f8', shape NxN) and <day>.manifest.json
          (canonical UTF-8 JSON: sorted keys, finite numbers only, one terminal LF)
  fn      verify_matrix_artifact(root, matrix_path, manifest_path, *, recompute=False)
          -> (ok: bool, reasons: list[str])
          consumer seam: verifies EVERY binding immediately before use, reopens both
          artifacts, verifies again after; recompute=True re-derives the matrix from
          the manifest's input objects through the FROZEN D2 envelope path and
          compares byte-exact (a doctored matrix with an internally consistent hash
          is NOT admissible -- content-auth != derivation provenance).

MANIFEST REQUIRED KEYS (A2; exact set -- missing or extra key REFUSES):
  schema, campaign_id, run_id, carrier_key, day, producer_commit, producer_blob_map,
  clean_tree, algorithm_config_digest, environment_lock_digest, input_manifest_sha256,
  input_objects  (sorted records {station_id, selected_nslc, object_sha256, size}),
  station_index_digest, station_ids, matrix_shape, matrix_dtype, matrix_endianness,
  n_overlap_policy, n_overlap, matrix_sha256, matrix_size, status, reason_codes

INVARIANTS UNDER TEST (refusal classes; A2 + the A5 minimum list, producer side):
  P1  seam presence + nominal synthetic fixture verifies end-to-end
  P2  station order not canonical UTF-8 lexical -> REFUSE (label/order relabel)
  P3  matrix-byte doctor w/ CONSISTENT updated sha -> recompute REFUSES (derivation)
  P4  cross-carrier station in the index -> REFUSE (never cross-carrier pairs)
  P5  UNAVAILABLE station present as zero row/col -> REFUSE (typed-absence rule:
      missing is typed state + absent measurement, never weight 0)
  P6  input byte drift: source object bytes changed after manifest -> REFUSE
  P7  matrix shape/dtype/endianness disagree with manifest -> REFUSE
  P8  algorithm_config_digest / environment_lock_digest drift -> REFUSE
  P9  producer_commit or producer_blob_map mismatch vs recorded -> REFUSE
  P10 PREFIX BAN: 12-hex prefix match with differing tail -> REFUSE (full-64 only)
  P11 non-finite cell claiming positive n_overlap support -> REFUSE
  P12 manifest keyset drift (missing key / extra key) -> REFUSE
  P13 r_ij out of [-1, 1] -> REFUSE (dimensionless signed correlation, unit=1)
  P14 unselected pool spare carrying an input object -> REFUSE (A1: no
      waveform-derived edge without a manifest-bound object for a SELECTED station)

STATUS: RED-FIRST. The seam does not exist yet; P1 reports the seam-absent RED and
the battery is authored-executable (fixtures built, mutations defined) so the bar
goes green the moment grassmann's producer lands and satisfies it. No root I/O:
this bar is hermetic-synthetic; the REAL-pair lane (pins from the beside-root
inventory f445f877...) is the cross-verification packet's job, not this file's.

Phase A only. Read-only roots. No acquisition, no production/registry mutation, no
claims. r_ij is never distance, displacement, motion, or movement.
"""

import hashlib
import importlib
import io
import json
import os
import struct
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FAILS = []

SEAM_MODULE = "d2_f2g_matrix_producer"
IDENTITY_ARTIFACT = "f2g_producer_identity.json"
IDENTITY = {"schema": "f2g-producer-identity-v1",
            "producer_commit": "a" * 40,
            "producer_blob_map": {"monitoring/src/seismic_data.py": "b" * 40},
            "algorithm_config_digest": "c" * 64,
            "environment_lock_digest": "d" * 64,
            "clean_tree": True}
FIXTURE_SCHEMA = "f2g-fixture-input-manifest-v1"
REAL_SCHEMA = "im-v2-resume"
MANIFEST_KEYS = {
    "schema", "campaign_id", "run_id", "carrier_key", "day", "producer_commit",
    "producer_blob_map", "clean_tree", "algorithm_config_digest",
    "environment_lock_digest", "input_manifest_sha256", "input_objects",
    "station_index_digest", "station_ids", "matrix_shape", "matrix_dtype",
    "matrix_endianness", "n_overlap_policy", "n_overlap", "matrix_sha256",
    "matrix_size", "status", "reason_codes",
}


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    suffix = f" - {detail}" if detail and not ok else ""
    print(f"    [{tag}] {name}{suffix}")
    if not ok:
        FAILS.append(name)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8") + b"\n"


# ---- synthetic fixture -------------------------------------------------------------
# Deterministic synthetic "waveforms": pure sinusoid mixtures whose pairwise envelope
# correlation the producer must reproduce through ITS OWN frozen path. The fixture
# pins bytes + identities; it never pins the numeric answer (that would be the
# self-consistent-synthetic trap) -- P3's oracle is the producer's recompute, not a
# constant baked here.
def _mk_fixture(td):
    stations = ["KO.A01", "KO.A02", "KO.B01", "KO.B02"]      # canonical lexical order
    rng = np.random.default_rng(20260818)
    raws = {}
    for i, s in enumerate(stations):
        t = np.arange(0, 600.0, 0.04)
        w = (np.sin(2 * np.pi * (1.5 + 0.3 * i) * t)
             + 0.25 * rng.standard_normal(t.size))
        raws[s] = w.astype("<f8").tobytes()
    objs = []
    os.makedirs(os.path.join(td, "raw_objects"), exist_ok=True)
    for s in stations:
        b = raws[s]
        with open(os.path.join(td, "raw_objects", sha(b) + ".bin"), "wb") as fh:
            fh.write(b)
        objs.append({"station_id": s, "selected_nslc": s + "..HHZ",
                     "object_sha256": sha(b), "size": len(b)})
    objs.sort(key=lambda o: (o["station_id"], o["object_sha256"]))
    im = {"schema": "f2g-fixture-input-manifest-v1", "objects": objs}
    with open(os.path.join(td, "input_manifest.json"), "wb") as fh:
        fh.write(canon(im))
    # REV 2 (2a): the durable expected-identity artifact IS the identity
    # authority; fixtures carry it with the values the manifests claim
    with open(os.path.join(td, IDENTITY_ARTIFACT), "wb") as fh:
        fh.write(canon(IDENTITY))
    return stations, objs


def _npy_bytes(arr):
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr, dtype="<f8", order="C"))
    return buf.getvalue()


def _mk_artifact(td, stations, objs, *, mutate=None):
    """A structurally valid emitted artifact pair (matrix + manifest) the CONSUMER
    seam must judge. mutate(matrix_array, manifest_dict) applies a violation and
    returns (matrix_array, manifest_dict); sha/size fields are RE-BOUND after the
    mutation unless the mutation deliberately breaks them."""
    n = len(stations)
    rng = np.random.default_rng(7)
    a = rng.standard_normal((n, n))
    r = np.corrcoef(a)                                        # any valid r-matrix
    np.fill_diagonal(r, 1.0)
    nov = [[600 if i != j else 0 for j in range(n)] for i in range(n)]
    man = {
        "schema": "f2g-matrix-manifest-v1",
        "campaign_id": "f" * 64, "run_id": "f" * 64,
        "carrier_key": "c_fix", "day": "2026-03-02",
        "producer_commit": "a" * 40,
        "producer_blob_map": {"monitoring/src/seismic_data.py": "b" * 40},
        "clean_tree": True,
        "algorithm_config_digest": "c" * 64,
        "environment_lock_digest": "d" * 64,
        "input_manifest_sha256": sha(open(os.path.join(td, "input_manifest.json"),
                                          "rb").read()),
        "input_objects": objs,
        "station_index_digest": sha(canon(stations)),
        "station_ids": list(stations),
        "matrix_shape": [n, n], "matrix_dtype": "<f8", "matrix_endianness": "little",
        "n_overlap_policy": "min-common-finite-samples", "n_overlap": nov,
        "matrix_sha256": None, "matrix_size": None,
        "status": "PRODUCED", "reason_codes": [],
    }
    if mutate:
        r, man = mutate(np.array(r), man)
    mb = _npy_bytes(r)
    if man.get("matrix_sha256") is None:
        man["matrix_sha256"] = sha(mb)
    if man.get("matrix_size") is None:
        man["matrix_size"] = len(mb)
    mdir = os.path.join(td, "out", man["carrier_key"])
    os.makedirs(mdir, exist_ok=True)
    mp = os.path.join(mdir, man["day"] + ".matrix.npy")
    fp = os.path.join(mdir, man["day"] + ".manifest.json")
    with open(mp, "wb") as fh:
        fh.write(mb)
    with open(fp, "wb") as fh:
        fh.write(canon(man))
    return mp, fp


def main():
    try:
        prod = importlib.import_module(SEAM_MODULE)
        have = all(hasattr(prod, f) for f in
                   ("produce_carrier_day_matrix", "verify_matrix_artifact"))
    except ModuleNotFoundError:
        prod, have = None, False

    check("F2G-P1a seam present: d2_f2g_matrix_producer.produce_carrier_day_matrix "
          "+ verify_matrix_artifact (RED until grassmann's producer lands)", have,
          "module or functions absent -- authored red-first per contract 2e0c7a33 A5")
    if not have:
        # The battery below is authored-executable; without the seam every class is
        # RED by construction (cannot be judged). Print them as explicit reds so the
        # green transition is a visible, countable event -- never a silent skip.
        for name in ("P1b nominal fixture verifies end-to-end",
                     "P2 non-canonical station order REFUSES",
                     "P3 doctored matrix w/ consistent sha REFUSES on recompute",
                     "P4 cross-carrier station REFUSES",
                     "P5 UNAVAILABLE-as-zero REFUSES",
                     "P6 source-object byte drift REFUSES",
                     "P7 shape/dtype/endianness mismatch REFUSES",
                     "P8 config/environment digest drift REFUSES",
                     "P9 producer commit/blob-map mismatch REFUSES",
                     "P10 prefix-match-only comparison REFUSES",
                     "P11 non-finite cell with positive n_overlap REFUSES",
                     "P12 manifest keyset drift REFUSES",
                     "P13 r_ij outside [-1,1] REFUSES",
                     "P14 unselected spare w/ input object REFUSES"):
            check(f"F2G-{name}", False, "seam absent")
        return

    td = tempfile.mkdtemp()
    stations, objs = _mk_fixture(td)

    def verdict(mutate=None, recompute=False):
        mp, fp = _mk_artifact(td, stations, objs, mutate=mutate)
        ok, reasons = prod.verify_matrix_artifact(td, mp, fp, recompute=recompute)
        return ok, reasons

    ok_nom, det_nom = verdict()
    check("F2G-P1b nominal fixture verifies end-to-end", ok_nom, str(det_nom))

    def refused(name, mutate, recompute=False):
        ok, reasons = verdict(mutate=mutate, recompute=recompute)
        check(name, not ok, f"ACCEPTED despite violation ({reasons})")

    refused("F2G-P2 non-canonical station order REFUSES",
            lambda r, m: (r, {**m, "station_ids": list(reversed(m["station_ids"])),
                              "station_index_digest":
                                  sha(canon(list(reversed(m["station_ids"]))))}))
    def _doctor_in_bounds(r, m):
        # off-diagonal, symmetric, still inside [-1, 1]: only DERIVATION can catch it
        r[0, 1] = r[1, 0] = float(np.clip(r[0, 1] + 1e-3, -0.999, 0.999))
        return r, m
    refused("F2G-P3 doctored matrix w/ consistent sha REFUSES on recompute",
            _doctor_in_bounds, recompute=True)
    refused("F2G-P4 cross-carrier station REFUSES",
            lambda r, m: (r, {**m, "station_ids":
                              m["station_ids"][:-1] + ["XX.OTHERCARRIER"]}))
    refused("F2G-P5 UNAVAILABLE-as-zero REFUSES",
            lambda r, m: ((np.pad(r, ((0, 1), (0, 1)))),
                          {**m,
                           "station_ids": m["station_ids"] + ["KO.DARK"],
                           "matrix_shape": [len(m["station_ids"]) + 1] * 2,
                           "n_overlap": [row + [0] for row in m["n_overlap"]]
                                        + [[0] * (len(m["station_ids"]) + 1)],
                           "station_index_digest":
                               sha(canon(m["station_ids"] + ["KO.DARK"]))}))
    def _drift_source(r, m):
        p = os.path.join(td, "raw_objects", m["input_objects"][0]["object_sha256"]
                         + ".bin")
        with open(p, "ab") as fh:
            fh.write(b"drift")
        return r, m
    refused("F2G-P6 source-object byte drift REFUSES", _drift_source)
    _mk_fixture(td)   # restore drifted fixture bytes for later classes
    refused("F2G-P7 shape/dtype/endianness mismatch REFUSES",
            lambda r, m: (r, {**m, "matrix_dtype": ">f8",
                              "matrix_endianness": "big"}))
    refused("F2G-P8 config/environment digest drift REFUSES",
            lambda r, m: (r, {**m, "algorithm_config_digest": "e" * 64}))
    refused("F2G-P9 producer commit/blob-map mismatch REFUSES",
            lambda r, m: (r, {**m, "producer_blob_map":
                              {"monitoring/src/seismic_data.py": "9" * 40}}))
    refused("F2G-P10 prefix-match-only comparison REFUSES",
            lambda r, m: (r, {**m, "input_manifest_sha256":
                              m["input_manifest_sha256"][:12] + "0" * 52}))
    refused("F2G-P11 non-finite cell with positive n_overlap REFUSES",
            lambda r, m: ((np.where(np.eye(len(r)) == 0, r, r) * np.nan), m))
    refused("F2G-P12 manifest keyset drift REFUSES",
            lambda r, m: (r, {k: v for k, v in m.items() if k != "clean_tree"}))
    refused("F2G-P13 r_ij outside [-1,1] REFUSES",
            lambda r, m: ((r * 3.0), m))
    refused("F2G-P14 unselected spare w/ input object REFUSES",
            lambda r, m: (r, {**m, "input_objects": m["input_objects"] + [{
                "station_id": "KO.SPARE", "selected_nslc": "KO.SPARE..HHZ",
                "object_sha256": "5" * 64, "size": 1,
                "pool_member": True, "registry_selected": False}]}))

    # ---- REV 2: identity authority (2a) -- EXPECTED RED on d3e76f4 ----------------
    INVENTED = {"producer_commit": "f" * 40,
                "producer_blob_map": {"monitoring/src/seismic_data.py": "e" * 40},
                "algorithm_config_digest": "9" * 64,
                "environment_lock_digest": "8" * 64}

    def _fresh_pair(day, ident_fields=None, drop_identity_artifact=False):
        t2 = tempfile.mkdtemp()
        s2, o2 = _mk_fixture(t2)
        if drop_identity_artifact:
            os.remove(os.path.join(t2, IDENTITY_ARTIFACT))
        def mut(r, m):
            m = {**m, "day": day}
            if ident_fields:
                m.update(ident_fields)
            return r, m
        mp2, fp2 = _mk_artifact(t2, s2, o2, mutate=mut)
        return t2, mp2, fp2

    t8, mp8, fp8 = _fresh_pair("2026-03-05", INVENTED)
    ok8a, det8a = prod.verify_matrix_artifact(t8, mp8, fp8, recompute=False)
    check("F2G-P8a INVENTED identity REFUSES on the FIRST call (durable "
          "artifact is the authority; TOFU is not attestation)", not ok8a,
          f"ACCEPTED invented identity ({det8a})")
    t8b, mp8b, fp8b = _fresh_pair("2026-03-06", drop_identity_artifact=True)
    ok8b, det8b = prod.verify_matrix_artifact(t8b, mp8b, fp8b, recompute=False)
    check("F2G-P8b identity artifact ABSENT -> fail-closed REFUSAL", not ok8b,
          f"ACCEPTED without any identity authority ({det8b})")

    import subprocess
    def _sub_verify(root, mp_, fp_):
        code = ("import sys; sys.path.insert(0, sys.argv[4]); "
                "import d2_f2g_matrix_producer as P; "
                "ok, rs = P.verify_matrix_artifact(sys.argv[1], sys.argv[2], "
                "sys.argv[3], recompute=False); print('VERDICT', ok)")
        r_ = subprocess.run([sys.executable, "-c", code, root, mp_, fp_, HERE],
                            capture_output=True, text=True, timeout=180)
        return "VERDICT True" in r_.stdout, (r_.stdout + r_.stderr)[-160:]
    t9, mp9, fp9 = _fresh_pair("2026-03-07")
    ok9v, out9v = _sub_verify(t9, mp9, fp9)
    t9i, mp9i, fp9i = _fresh_pair("2026-03-08", INVENTED)
    ok9i, out9i = _sub_verify(t9i, mp9i, fp9i)
    check("F2G-P9a post-restart authority: valid identity PASSES and invented "
          "identity REFUSES in a FRESH process", ok9v and not ok9i,
          f"valid={ok9v} invented_accepted={ok9i} ({out9i if ok9i else out9v})")

    # ---- REV 2: real-path dispatch (2b) -- EXPECTED RED on d3e76f4 ----------------
    t15 = tempfile.mkdtemp()
    os.makedirs(os.path.join(t15, "raw_objects"), exist_ok=True)
    fake = b"MSEEDMSEEDMSEED!"                     # 16 bytes: parses as 2 float64s
    with open(os.path.join(t15, "raw_objects", sha(fake) + ".ms"), "wb") as fh:
        fh.write(fake)
    real_objs = [{"sha256": sha(fake), "size": len(fake),
                  "relative_path": f"raw_objects/{sha(fake)}.ms",
                  "kind": "archive-seismic-miniseed-fragments-v1",
                  "carrier_key": "c_fix", "scored_day": "2026-03-02",
                  "segment_name": "seg_a", "source_id": "KO.A01..HHZ",
                  "start_utc": "2026-03-01T07:00:13.094647Z",
                  "end_utc": "2026-03-02T07:00:13.094647Z"}]
    with open(os.path.join(t15, "input_manifest.json"), "wb") as fh:
        fh.write(canon({"schema": REAL_SCHEMA, "producer_commit": "a" * 40,
                        "implementation_commit": "a" * 40,
                        "objects": real_objs}))
    with open(os.path.join(t15, IDENTITY_ARTIFACT), "wb") as fh:
        fh.write(canon(IDENTITY))
    fixture_calls = []
    orig_df = getattr(prod, "_derive_fixture", None)
    if orig_df is not None:
        def _spy(*a, **k):
            fixture_calls.append(1)
            return orig_df(*a, **k)
        prod._derive_fixture = _spy
    try:
        try:
            prod.produce_carrier_day_matrix(t15, "c_fix", "2026-03-02",
                                            out_dir=os.path.join(t15, "out"))
            produced = True
        except Exception:
            produced = False
    finally:
        if orig_df is not None:
            prod._derive_fixture = orig_df
    check("F2G-P15a real schema (im-v2-resume) NEVER reaches _derive_fixture "
          "and never yields a silently fixture-derived matrix (typed refusal "
          "acceptable when real deps absent)",
          not fixture_calls and not produced,
          f"fixture_reader_calls={len(fixture_calls)} produced={produced}")

    try:
        import obspy  # noqa: F401
        run_p15b = True
    except ImportError:
        run_p15b = False
        print("    [CAP ] F2G-P15b frozen-D2 oracle - obspy absent on this "
              "host (grassmann's venv runs it live)")
    if run_p15b:
        from datetime import datetime, timezone
        import obspy as _ob
        import d2_step4b_campaign_run as CR
        import seismic_data as SD
        t15b = tempfile.mkdtemp()
        os.makedirs(os.path.join(t15b, "raw_objects"), exist_ok=True)
        session_start = datetime(2026, 3, 2, 0, 0, 0, tzinfo=timezone.utc)
        rng15 = np.random.default_rng(15)
        objs15, streams = [], {}
        for sid in ("KO.A01", "KO.A02"):
            tr = _ob.Trace(data=(np.sin(2 * np.pi * 3.0 * np.arange(0, 600, 0.02))
                                 + 0.1 * rng15.standard_normal(30000)))
            tr.stats.sampling_rate = 50.0
            tr.stats.starttime = _ob.UTCDateTime("2026-03-02T00:00:00")
            net, sta = sid.split(".")
            tr.stats.network, tr.stats.station, tr.stats.channel = net, sta, "HHZ"
            import io as _io
            buf = _io.BytesIO()
            _ob.Stream([tr]).write(buf, format="MSEED")
            b15 = buf.getvalue()
            with open(os.path.join(t15b, "raw_objects", sha(b15) + ".ms"),
                      "wb") as fh:
                fh.write(b15)
            objs15.append({"sha256": sha(b15), "size": len(b15),
                           "relative_path": f"raw_objects/{sha(b15)}.ms",
                           "kind": "archive-seismic-miniseed-fragments-v1",
                           "carrier_key": "c_fix", "scored_day": "2026-03-02",
                           "segment_name": "seg_a",
                           "source_id": sid + "..HHZ",
                           "start_utc": "2026-03-02T00:00:00.000000Z",
                           "end_utc": "2026-03-02T00:10:00.000000Z"})
            streams[sid] = _ob.Stream([tr])
        with open(os.path.join(t15b, "input_manifest.json"), "wb") as fh:
            fh.write(canon({"schema": REAL_SCHEMA, "producer_commit": "a" * 40,
                            "implementation_commit": "a" * 40,
                            "objects": objs15}))
        with open(os.path.join(t15b, IDENTITY_ARTIFACT), "wb") as fh:
            fh.write(canon(IDENTITY))
        ok15b, det15b = False, ""
        try:
            man15 = prod.produce_carrier_day_matrix(
                t15b, "c_fix", "2026-03-02", out_dir=os.path.join(t15b, "out"))
            es = {sid: CR._station_series(SD, streams[sid], sid + "..HHZ",
                                          session_start)
                  for sid in streams}
            a_, b_ = es["KO.A01"], es["KO.A02"]
            common = a_.valid_mask & b_.valid_mask
            va, vb = a_.values[common], b_.values[common]
            oracle = float(np.corrcoef(va, vb)[0, 1])
            mtx = np.load(os.path.join(t15b, "out", "c_fix",
                                       "2026-03-02.matrix.npy"))
            ids15 = man15["station_ids"]
            got = float(mtx[ids15.index("KO.A01"), ids15.index("KO.A02")])
            ok15b = np.isfinite(got) and got == oracle \
                and man15["n_overlap"][0][1] == int(common.sum())
            det15b = f"got={got} oracle={oracle} n={int(common.sum())}"
        except Exception as exc:
            det15b = f"{type(exc).__name__}: {exc}"
        check("F2G-P15b real miniSEED through the frozen D2 path equals the "
              "independently invoked oracle (envelopes via _station_series; "
              "pearson over common valid_mask)", ok15b, det15b)


main()
print()
if FAILS:
    print(f"F2G MATRIX-PRODUCER RED-KAT FAILURES ({len(FAILS)}): {FAILS}")
    sys.exit(1)
print("ALL F2G MATRIX-PRODUCER RED-KATs PASS")
