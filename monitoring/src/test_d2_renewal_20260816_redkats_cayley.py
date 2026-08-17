#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D2 CALIBRATION RENEWAL red-KATs — REV 2 (cayley, 2026-08-16) for codex's frozen
renewal contract `codex-d2-campaign-v2-renewal-2026-08-16-v1` (`80bb3c6`), recut per
codex review `f7fd6f3` (R1 WORKS-WITH-FIX, five findings, each repaired below).

REV 2 REPAIRS (codex f7fd6f3, finding -> checks):
  #1 CRITICAL  RN-5d/RN-5e: batch-root recompute from reopened capsule bytes + registry
      candidate binding (path/sha/contract/batch-root/verification receipts) with
      per-field flip refusals + the separate lift/effective boundary: a day already
      refused stale is NOT retroactively admitted (no-backfill, executable).
  #2 CRITICAL  RN-2c now uses a canonical CORRECT-contract bundle fixture (module pin
      temporarily pointed at the fixture sha): the builder must ACCEPT those bytes,
      return canonical plan bytes carrying core_blobs == core_blob_map() == git-HEAD,
      and REFUSE a one-byte mutation — an always-raise builder or dummy pin cannot
      green. RN-5c's nominal capsule uses git HEAD as source_commit and the validator
      takes expected_source_commit (flip -> refuse).
  #3 MAJOR     RN-2e pins build/validate_renewal_phase_ledger: exact 138-day union
      coverage, bundle sha binding, old-v2-ledger + one-day-mutation refusals, and
      pool/ledger created_utc BEFORE every provider attempted_utc (chronology closed
      end-to-end in RN-4f against real acquire attempts).
  #4 MAJOR     RN-4f drives the REAL `d2_step4b_campaign_run.acquire` with hermetic
      scripted providers: a true post-sort overlap must end as a station attempt with
      reason TRUE_OVERLAP_UNRULED and NO admitted data/mask/matrix contribution; an
      either-arm-below-60 carrier must produce ZERO provider calls; attempts must
      postdate pool+ledger creation. Helpers alone can no longer green.
  #5 MAJOR     RN-4e/RN-5c are table-driven: batch form refuses duplicates, reorder,
      extra/missing targets, wrong contract, nonterminal states; the capsule validator
      refuses independent mutation of EVERY required scalar/binding.

REV 3 (codex R2 `00c4e7b` / `169dca4`, four bypasses repaired):
  #1 CRITICAL RN-5d: compute_batch_root REQUIRES capsule_path+manifest_path, REOPENS both
     byte artifacts, REFUSES either declared-digest mismatch, derives the root only from
     reopened-and-matched bytes (zeros-digest and missing-manifest negatives executable).
  #2 CRITICAL RN-2c/2e: ledger-FIRST construction — the final bundle bytes BIND
     phase_ledger_sha256, the plan CARRIES and MATCHES the reopened ledger digest, and
     the builder REFUSES a bundle without the binding.
  #3 MAJOR RN-4d: reuse entries validate against an EXPECTED immutable-object record
     (well-formed-but-WRONG identity/session/digest negatives refuse).
  #4 MAJOR RN-4f: chronology is ENFORCED IN acquire against
     max(plan.created_utc, ledger.created_utc) — a deliberately earlier injected clock
     must REFUSE, and every attempt must postdate the LATER creation time.
  MINOR owned: cayley's 0222 envelope carried a literal ${TS} (quoted-heredoc expansion
  defect); live-read timestamps hereafter; no chronology inferred from that envelope.

REV 4 (grassmann finding `974ef1a`: minted capsules carry the frozen v2 label
`source_commit=292b1069...` and FAIL RN-5c — a producer/validator mismatch this bar
never drove; the bar validated the VALIDATOR, not the MINT — the R2 lesson one level
deeper, owned). REV 4 implements the contract-forced option B (contract §3 verbatim bans
the 292b1069 attestation on renewal capsules; option A would contradict the frozen
contract). SELF-VOIDING clause: if codex nonetheless rules option A, RN-5f/5g are struck
and RN-5c amended per that ruling instead.
  RN-5f pins `d2_renewal_plan.capsule_source_commit(plan)`: returns the PRODUCING git
     HEAD when plan.contract_id is the renewal contract, and the frozen
     IMPLEMENTATION_COMMIT for the v2 path (v2 semantics preserved); and the executor's
     mint must CALL it — the literal `"source_commit": IMPLEMENTATION_COMMIT` may no
     longer appear in campaign_run.py.
  RN-5g validates any RE-MINTED renewal capsules present in a staged renewal root:
     each must pass validate_renewal_capsule(expected_source_commit = the root's
     recorded producer_commit) — the sealed zero-I/O re-mint (option B step 3, owner
     one-liner) produces exactly these artifacts.

REV 4.3 (grassmann `2b27621` finding 1: a SECOND mint site in the RESUME executor
  d2_step4b_resume.py -- its own inline capsule dict with the stale
  source_commit=CR.IMPLEMENTATION_COMMIT -- and per RC1c the resume path IS the real
  production edge; production never calls CR.acquire, so the REV 4.2-proven finalize
  mint is dead code in production and the remint capsules came out stale AGAIN. The
  bar validated the seam on the WRONG EDGE -- the R2 lesson at a third level:
  validator -> mint -> WHICH mint. Owned.):
  RN-5f-d drives the REAL production edge: run_resume_campaign on the hermetic
  candidate fixture, spy-wrapped CR.mint_capsule, and the full REV 4.2 rigor on the
  RESUME-written root (one spy call per candidate, ONE producer commit, complete
  snapshot equality, manifest-bytes cross-link, source_commit == producer ==
  input_manifest.producer_commit). RED on the current tree by construction: resume
  emits candidates without ever calling the seam.
  RN-5f-e: a raising replacement minter must PROPAGATE through the RESUME edge as the
  injected sentinel type+value; a resume path that never calls the seam, swallows, or
  translates is a failed check.

REV 4.2 (codex `c37ef5c` WORKS-WITH-FIX, two bar false-greens hardened; the landed
  implementation 3183889 passes both repaired assertions):
  RN-5f-b: COMPLETE equality only -- the reopened written capsule must equal a JSON
  round-trip SNAPSHOT of the spy-captured mint result with NO fields excluded (the
  issued_utc/input_manifest_sha256 exclusion was an executable false-green: a producer
  overwriting input_manifest_sha256 AFTER mint still passed; snapshotting also kills
  the in-place-mutation variant), PLUS the mandatory cross-link
  capsule.input_manifest_sha256 == sha256(reopened input_manifest.json bytes).
  RN-5f-c: ONLY the injected sentinel type+value counts as propagation; any translated
  exception is a failed check (except-BaseException accepted translation).
  validate_replacement_root(): the same manifest-bytes cross-link is enforced on every
  admitted capsule (absence/mismatch = refusal); a new always-running battery fixture
  proves the cross-link itself discriminates.

REV 4.1 (codex `8e4d8fc`, two executable counterexamples repaired):
  RN-5g now parses the PRODUCER'S REAL SCHEMA (admission_results.regions is a LIST of
  rows; any other type refuses), selects rows by status==ADMITTED_CANDIDATE, binds the
  reopened capsule's region to the row's carrier_key, and requires the plan's blob-key
  set to equal CORE_FILES exactly before value comparison (empty/missing/extra/git-fail
  all refuse). Its core is factored into validate_replacement_root() and LOCKED by an
  always-running fixture battery: nominal synthetic-root PASS + dict-shape, empty-map,
  missing-key, extra-key, wrong-blob, and region-mismatch refusals.
  RN-5f now EXECUTES the producer-to-minter edge: a candidate-producing hermetic
  real-acquire fixture (POLICY.min_admitted_days patched to 2 inside the KAT only —
  RN-1c still locks the real value; one correlated low-ratio day pins the threshold
  minimum so the sealed diagnostic replay clears the gates) drives the REAL acquire to
  an emitted capsule; the bar spy-wraps CR.mint_capsule, requires one spy call per
  emitted candidate, reopens the written capsule and requires it to equal the spy
  result carrying the single supplied producer commit; a raising replacement minter
  must PROPAGATE through the producer. Source inspection is demoted to diagnostics.

SEAMS PINNED (naming decisions implementing the contract):
  module `monitoring/src/d2_renewal_plan.py`:
    RENEWAL_CONTRACT_ID / RENEWAL_ANCHOR / V2_POOL_SHA256 / RENEWAL_BUNDLE_SHA256
    renewal_arms() -> dict
    build_renewal_candidate_pool(v2_pool_bytes) -> (pool, pool_bytes)
    build_renewal_plan(bundle_bytes) -> (plan, plan_bytes)   [canonical bytes; core_blobs]
    build_renewal_phase_ledger(bundle_bytes) -> (ledger, ledger_bytes)
    validate_renewal_phase_ledger(ledger, bundle) -> bool
    core_blob_map() -> {repo-relative core file: git blob sha}
    classify_station_refusal(frs) -> str            ("TRUE_OVERLAP_UNRULED" | other)
    mark_coverage_infeasible(potentials) -> bool
    validate_reuse_entry(entry, *, campaign_start_utc, contract_id, expected) -> bool
        (expected = the immutable-object record {provider_identity, session, sha256};
         well-formed-but-wrong values REFUSE; locally reopened bytes checked when given)
    validate_batch_form(batch) -> bool
    validate_renewal_capsule(capsule, *, expected_source_commit) -> bool
    compute_batch_root(entries) -> str
        (entries REQUIRE capsule_path+manifest_path; BOTH reopened; declared
         capsule_sha256/manifest_sha256 must MATCH the reopened bytes or REFUSE;
         root derived only from reopened-and-matched bytes)
    validate_registry_candidate(record, *, expected) -> bool
    renewal_admits(day, capsule, lift_effective_utc) -> bool  (no-backfill boundary)
    capsule_source_commit(plan) -> str   (REV 4: renewal plan -> producing git HEAD;
                                          v2 plan -> frozen IMPLEMENTATION_COMMIT)
  executor delta (d2_step4b_campaign_run.acquire): renewal-contract plans accepted;
    true-overlap station attempts carry TRUE_OVERLAP_UNRULED; infeasible carriers are
    never fetched.

COMPOSED FROZEN SUITES (unchanged-semantics enforcement, must stay green):
  order-canon, step4b executor, step4b provider, phase-0.75 registry.
EXPECTED NOW: composition + arithmetic + POLICY + loader locks PASS; every renewal seam
check RED (module absent / executor delta absent). GREEN = all PASS post-implementation.
NO PROVIDER I/O anywhere in this bar; the fresh fire-time owner go naming the contract id
and A remains required before any external request.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import types
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

FAILS = []

CONTRACT_ID = "codex-d2-campaign-v2-renewal-2026-08-16-v1"
ANCHOR = "2026-08-16"
V2_POOL_SHA = "15d0e32c51c027dc144c5c6d57ec5f100a59374f6248abfc5c56ee38628ddc67"
STALE_PREFIX = "292b1069"
TARGET_ORDER = ["istanbul_marmara", "socal_coachella", "turkey_kahramanmaras"]
CORE_FILES = ["monitoring/src/seismic_data.py", "monitoring/src/fault_correlation.py",
              "monitoring/src/ensemble.py", "monitoring/src/d2_step4b_campaign_run.py",
              "monitoring/src/d2_renewal_plan.py"]


def check(desc, ok, detail=""):
    print(f"    [{'PASS' if ok else 'FAIL'}] {desc}" + (f" - {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(desc)


def awaiting(*descs):
    for d in descs:
        check(d, False, "AWAITING implementation (RN-0 red)")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def canon(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                       allow_nan=False) + "\n").encode()


def days_between(a, b):
    out = []
    d = date.fromisoformat(a)
    end = date.fromisoformat(b)
    while d < end:
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def git_head_blobs():
    out = {}
    repo = os.path.join(HERE, "..", "..")
    for f in CORE_FILES:
        parts = subprocess.run(["git", "ls-tree", "HEAD", f], capture_output=True,
                               text=True, cwd=repo).stdout.split()
        out[f] = parts[2] if len(parts) >= 3 else None
    return out


def git_head(repo=None):
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=repo or os.path.join(HERE, "..", "..")).stdout.strip()


def run_suite(name, timeout=1800):
    r = subprocess.run([sys.executable, "-B", os.path.join(HERE, name)],
                       capture_output=True, text=True, timeout=timeout, cwd=HERE)
    return r.returncode == 0, (r.stdout + r.stderr)[-200:]


def install_obspy_stub():
    if "obspy" in sys.modules:
        return
    import numpy as _np
    from scipy.signal import hilbert as _hilb
    _o = types.ModuleType("obspy")
    _o.Stream, _o.Trace, _o.UTCDateTime = type("S", (), {}), type("T", (), {}), object
    _o.__version__ = "stub-hermetic"        # resume.py:996 reads it unguarded
    _f = types.ModuleType("obspy.clients.fdsn")
    _f.Client = type("C", (), {"__init__": lambda s, *a, **k: None})
    _c = types.ModuleType("obspy.clients")
    _c.fdsn = _f
    _o.clients = _c
    _sg = types.ModuleType("obspy.signal")
    _sf = types.ModuleType("obspy.signal.filter")
    _sf.envelope = lambda a: _np.abs(_hilb(a))
    _sf.bandpass = lambda d, fmin, fmax, df, corners=4, zerophase=False: d
    _sg.filter = _sf
    for nm, md in (("obspy", _o), ("obspy.clients", _c), ("obspy.clients.fdsn", _f),
                   ("obspy.signal", _sg), ("obspy.signal.filter", _sf)):
        sys.modules[nm] = md


# ---- executor-drive doubles (shapes proven by the frozen executor bar) -----------
class _Stats:
    def __init__(self, start, rate):
        self.starttime = types.SimpleNamespace(datetime=start.replace(tzinfo=None))
        self.sampling_rate = rate


class _Trace:
    def __init__(self, start, npts, rate):
        import numpy as np
        self.stats = _Stats(start, rate)
        self.data = np.zeros(int(npts))


def _frames_of(body):
    return json.loads(body.decode("utf-8").split("TRACES:", 1)[1])


def _traces_of(body):
    out = []
    for iso, npts, rate in _frames_of(body):
        start = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        out.append(_Trace(start, npts, rate))
    return out


def _body(frames):
    return ("TRACES:" + json.dumps(frames)).encode("utf-8")


class _SpyProviders:
    """Hermetic scripted providers (executor-bar pattern) with a CALL COUNTER and one
    overlap-scripted station. No network; stages its own bodies."""

    class ProviderUnavailable(Exception):
        pass

    def __init__(self, overlap_nslc=None):
        self.overlap_nslc = overlap_nslc
        self.calls = []                      # (provider, nslc, day)

    def koeri_available(self, net, stas, chas, start, end):
        return {f"{net}.{s}..{c}" for s in stas for c in chas}

    def scedc_available(self, net, stas, chas, start, end):
        return {f"{net}.{s}..{c}" for s in stas for c in chas}

    def parse_staged(self, staged_path):
        with open(staged_path, "rb") as f:
            return _traces_of(f.read())

    def fetch(self, provider, nslc, start, end, *, stage_dir, **kw):
        day = end.date().isoformat()
        self.calls.append((provider, nslc, day))
        s = start.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
        if nslc == self.overlap_nslc:
            s2 = (start + timedelta(seconds=150)).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
            frames = [[s, 30000, 100.0], [s2, 30000, 100.0]]     # TRUE overlap post-sort
        else:
            frames = [[s, 8, 40.0]]
        body = _body(frames)
        name = f"{nslc}_{day}_{sha(body)[:12]}.ms"
        path = os.path.join(stage_dir, name)
        with open(path, "wb") as f:
            f.write(body)
        return {"stream": _traces_of(body),
                "raw_objects": [{"source": f"fake://{nslc}/{day}", "staged_path": path,
                                 "size_bytes": len(body), "sha256": sha(body)}]}


RECEIPT = {"status": "VERIFIED_DIRECT", "in_session_timestamp_utc": "2026-08-16T01:03:00Z",
           "owner_quote_sha256": 64 * "0"}   # fixture; the REAL fire receipt is owner-gated


def _mk_renewal_fixture_root(RP, td, *, infeasible_carrier=None):
    """Minimal renewal plan+ledger fixture in the executor-bar shape, renewal contract id,
    THREE scheduled days, one carrier (istanbul), 2 segments x 2 stations."""
    days = ["2026-03-02", "2026-03-03", "2026-03-04"]
    registry = {}
    rows = []
    for seg_i, seg in enumerate(("seg_a", "seg_b")):
        for st_i in range(2):
            sta = f"S{seg_i}{st_i}"
            rows.append({"segment_name": seg, "station_id": f"KO.{sta}",
                         "ordered_nslc_candidates": [f"KO.{sta}..BHZ"]})
    registry["istanbul_marmara"] = rows
    plan = {"schema": "geospec-d2-step4b-campaign-plan-v1",
            "contract_id": CONTRACT_ID,
            "registered_utc": "2026-08-16T01:20:00.000000Z",
            "activation_reference_day": ANCHOR, "incident_reference_day": "2026-07-29",
            "carriers": ["istanbul_marmara"],
            "providers": {"istanbul_marmara": {"provider": "KOERI",
                                               "endpoint": "eida.koeri.boun.edu.tr"}},
            "station_registry": registry,
            "incident_days": days, "activation_days": days, "scheduled_days": days,
            "acquisition_order": ["KOERI"], "free_sources_only": True,
            "outcomes_inspected_before_schedule": False,
            "core_blobs": RP.core_blob_map() if RP else {},
            "created_utc": "2026-08-16T01:20:00.000000Z"}
    if infeasible_carrier:
        plan["coverage_infeasible"] = {infeasible_carrier: {"incident_potential": 59,
                                                            "activation_potential": 96}}
    lrows = []
    for day in days:
        end = f"{day}T07:00:13.094647Z"
        start_dt = datetime.strptime(end, "%Y-%m-%dT%H:%M:%S.%fZ") - timedelta(seconds=86400)
        lrows.append({"carrier_key": "istanbul_marmara", "scored_day": day,
                      "status": "REGISTERED", "publication_commit": 40 * "a",
                      "publication_repo_path": "docs/ensemble_latest.json",
                      "publication_record_artifact": "publication_records/x.json",
                      "record_git_blob": 40 * "b", "record_sha256": 64 * "c",
                      "request_start_utc": start_dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z",
                      "request_end_utc": end, "reason_codes": []})
    ledger = {"schema": "geospec-d2-published-phase-ledger-v1", "rows": lrows,
              "created_utc": "2026-08-16T01:21:00.000000Z"}
    with open(os.path.join(td, "campaign_plan.json"), "wb") as f:
        f.write(canon(plan))
    with open(os.path.join(td, "published_phase_ledger.json"), "wb") as f:
        f.write(canon(ledger))
    return plan, ledger


def _mk_candidate_fixture_root(RP, td):
    """Candidate-producing hermetic fixture: 1 carrier (istanbul), 3 scheduled days,
    2 segments x 2 stations, REAL-quality bodies (13 h @ 25 Hz, coverage ~0.54). Day 1
    serves ONE identical body to all stations/segments (cross-segment correlation ->
    LOW ratio, pinning the nearest-rank threshold minimum); days 2-3 serve independent
    noise (HIGH ratios). With POLICY.min_admitted_days patched to 2 in the KAT, the
    admitted set yields finite thresholds whose minimum is the low day-1 ratio, so the
    sealed diagnostic replay ratios clear artifact/control gates."""
    days = ["2026-03-02", "2026-03-03", "2026-03-04"]
    registry = {}
    rows = []
    for seg_i, seg in enumerate(("seg_a", "seg_b")):
        for st_i in range(2):
            sta = f"S{seg_i}{st_i}"
            rows.append({"segment_name": seg, "station_id": f"KO.{sta}",
                         "ordered_nslc_candidates": [f"KO.{sta}..BHZ"]})
    registry["istanbul_marmara"] = rows
    plan = {"schema": "geospec-d2-step4b-campaign-plan-v1",
            "contract_id": CONTRACT_ID,
            "registered_utc": "2026-08-16T01:20:00.000000Z",
            "activation_reference_day": ANCHOR, "incident_reference_day": "2026-07-29",
            "carriers": ["istanbul_marmara"],
            "providers": {"istanbul_marmara": {"provider": "KOERI",
                                               "endpoint": "eida.koeri.boun.edu.tr"}},
            "station_registry": registry,
            "incident_days": days, "activation_days": days, "scheduled_days": days,
            "acquisition_order": ["KOERI"], "free_sources_only": True,
            "outcomes_inspected_before_schedule": False,
            "core_blobs": RP.core_blob_map() if RP else {},
            "created_utc": "2026-08-16T01:20:00.000000Z"}
    lrows = []
    for day in days:
        end = f"{day}T07:00:13.094647Z"
        start_dt = datetime.strptime(end, "%Y-%m-%dT%H:%M:%S.%fZ") - timedelta(seconds=86400)
        lrows.append({"carrier_key": "istanbul_marmara", "scored_day": day,
                      "status": "REGISTERED", "publication_commit": 40 * "a",
                      "publication_repo_path": "docs/ensemble_latest.json",
                      "publication_record_artifact": "publication_records/x.json",
                      "record_git_blob": 40 * "b", "record_sha256": 64 * "c",
                      "request_start_utc": start_dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z",
                      "request_end_utc": end, "reason_codes": []})
    ledger = {"schema": "geospec-d2-published-phase-ledger-v1", "rows": lrows,
              "created_utc": "2026-08-16T01:21:00.000000Z"}
    with open(os.path.join(td, "campaign_plan.json"), "wb") as f:
        f.write(canon(plan))
    with open(os.path.join(td, "published_phase_ledger.json"), "wb") as f:
        f.write(canon(ledger))
    return plan, ledger


class _CandidateProviders:
    """Serves REAL-quality synthetic bodies (13 h @ 25 Hz starting 5 h into the
    session). Day 1: one shared correlated body for every station (low ratio).
    Days 2-3: per-station seeded noise (high ratio). Zero network."""

    class ProviderUnavailable(Exception):
        pass

    SPAN = 46800          # 13 h
    RATE = 25.0
    OFFSET = 5 * 3600

    def __init__(self):
        import numpy as np
        rng = np.random.default_rng(20260817)
        self._shared = rng.standard_normal(int(self.SPAN * self.RATE))
        self._np = np

    def koeri_available(self, net, stas, chas, start, end):
        return {f"{net}.{s}..{c}" for s in stas for c in chas}

    def scedc_available(self, net, stas, chas, start, end):
        return {f"{net}.{s}..{c}" for s in stas for c in chas}

    def parse_staged(self, staged_path):
        import pickle
        with open(staged_path, "rb") as f:
            frames = pickle.loads(f.read())
        return [_Trace(datetime.fromisoformat(iso), npts, rate)
                for iso, npts, rate, *_ in frames]

    def fetch(self, provider, nslc, start, end, *, stage_dir, **kw):
        import pickle
        np = self._np
        day = end.date().isoformat()
        frag_start = (start + timedelta(seconds=self.OFFSET))
        if day == "2026-03-02":
            data = self._shared
        else:
            seed = abs(hash((nslc, day))) % (2 ** 32)
            data = np.random.default_rng(seed).standard_normal(int(self.SPAN * self.RATE))
        # Staged bytes carry the timing frames (provenance reparse needs only
        # rate/npts/support timing per H3); scoring consumes the returned stream.
        frames = [[frag_start.isoformat(), int(self.SPAN * self.RATE), self.RATE,
                   nslc, day]]
        body = pickle.dumps(frames)
        name = f"{nslc}_{day}_{sha(body)[:12]}.pkl"
        path = os.path.join(stage_dir, name)
        with open(path, "wb") as f:
            f.write(body)
        stream = [_Trace(frag_start, int(self.SPAN * self.RATE), self.RATE)]
        stream[0].data = data
        return {"stream": stream,
                "raw_objects": [{"source": f"fake://{nslc}/{day}", "staged_path": path,
                                 "size_bytes": len(body), "sha256": sha(body)}]}


def validate_replacement_root(root_env, RP):
    """RN-5g core (REV 4.1): producer-schema-faithful replacement-root validation.
    Returns (ok: bool, detail: str). REFUSES: non-list regions, missing carrier_key/
    capsule_path on candidate rows, capsule-region/row mismatch, blob-key set !=
    CORE_FILES exactly, any git ls-tree failure or value mismatch, validator failure.
    REV 4.2 (codex c37ef5c): every admitted capsule must also carry
    input_manifest_sha256 == sha256(reopened input_manifest.json BYTES)."""
    with open(os.path.join(root_env, "input_manifest.json"), "rb") as fh:
        im_bytes = fh.read()
    im = json.loads(im_bytes.decode("utf-8"))
    im_digest = sha(im_bytes)
    producer = im.get("producer_commit")
    adm = json.loads(open(os.path.join(root_env, "admission_results.json"),
                          encoding="utf-8").read())
    regions = adm.get("regions")
    if not isinstance(regions, list):
        return False, f"regions is {type(regions).__name__}, not the producer's list"
    plan_doc = json.loads(open(os.path.join(root_env, "campaign_plan.json"),
                               encoding="utf-8").read())
    blobs = plan_doc.get("core_blobs")
    if not isinstance(blobs, dict) or set(blobs.keys()) != set(CORE_FILES):
        return False, f"core_blobs key set != CORE_FILES (got {sorted(blobs) if isinstance(blobs, dict) else blobs})"
    repo = os.path.join(HERE, "..", "..")
    for f2 in CORE_FILES:
        r = subprocess.run(["git", "ls-tree", str(producer), f2], capture_output=True,
                           text=True, cwd=repo)
        parts = r.stdout.split()
        if r.returncode != 0 or len(parts) < 3 or parts[2] != blobs[f2]:
            return False, f"blob mismatch/failure at {f2}"
    caps = 0
    for row in regions:
        if not isinstance(row, dict) or row.get("status") != "ADMITTED_CANDIDATE":
            continue
        carrier = row.get("carrier_key")
        cp = row.get("capsule_path")
        if not carrier or not cp:
            return False, f"candidate row missing carrier_key/capsule_path: {row.get('runner_key')}"
        cpath = cp if os.path.isabs(cp) else os.path.join(root_env, cp)
        cap = json.loads(open(cpath, encoding="utf-8").read())
        if cap.get("region") != carrier:
            return False, f"capsule region {cap.get('region')} != row carrier {carrier}"
        if cap.get("source_commit") != producer:
            return False, f"capsule source_commit != producer_commit for {carrier}"
        if cap.get("input_manifest_sha256") != im_digest:
            return False, ("capsule input_manifest_sha256 != sha256(reopened manifest "
                           f"bytes) for {carrier}")
        if RP.validate_renewal_capsule(cap, expected_source_commit=producer) is not True:
            return False, f"validator refuses capsule for {carrier}"
        caps += 1
    if caps == 0:
        return False, "no ADMITTED_CANDIDATE capsules found"
    return True, f"caps={caps} producer={str(producer)[:9]}"


def _mk_synthetic_replacement_root(RP, cap_template, head):
    """Nominal RN-5g lock fixture: a minimal synthetic root in the REAL producer
    schema (regions as a LIST) with one HEAD-bound capsule and exact HEAD blobs."""
    td = tempfile.mkdtemp()
    os.makedirs(os.path.join(td, "capsules"), exist_ok=True)
    im_body = json.dumps({"producer_commit": head}).encode("utf-8")
    with open(os.path.join(td, "input_manifest.json"), "wb") as fh:
        fh.write(im_body)
    cap = dict(cap_template)
    cap["region"] = "istanbul_marmara"
    cap["source_commit"] = head
    cap["input_manifest_sha256"] = sha(im_body)   # REV 4.2 cross-link bound to bytes
    open(os.path.join(td, "capsules", "istanbul_marmara_calibration.json"), "w",
         encoding="utf-8").write(json.dumps(cap, sort_keys=True))
    regions = [{"runner_key": "istanbul_marmara", "carrier_key": "istanbul_marmara",
                "status": "ADMITTED_CANDIDATE",
                "capsule_path": "capsules/istanbul_marmara_calibration.json"},
               {"runner_key": "ridgecrest", "carrier_key": "ridgecrest",
                "status": "BLOCKED_TOPOLOGY", "capsule_path": None}]
    open(os.path.join(td, "admission_results.json"), "w", encoding="utf-8").write(
        json.dumps({"schema": "x", "regions": regions}))
    blobs = {f: git_head_blobs()[f] for f in CORE_FILES}
    open(os.path.join(td, "campaign_plan.json"), "w", encoding="utf-8").write(
        json.dumps({"contract_id": CONTRACT_ID, "core_blobs": blobs}))
    return td


def main():
    install_obspy_stub()

    # ================= RN-0 — pinned renewal seams =================
    RP = None
    try:
        import d2_renewal_plan as RP_mod
        RP = RP_mod
    except ImportError as e:
        check("RN-0 pinned module d2_renewal_plan with the full REV-2 seam set "
              "(arms/pool/plan/ledger builders, core_blob_map, classify/infeasible/"
              "reuse/batch/capsule/root/registry/no-backfill validators)",
              False, f"import failed: {e}")
    if RP is not None:
        seams = ("renewal_arms", "build_renewal_candidate_pool", "build_renewal_plan",
                 "build_renewal_phase_ledger", "validate_renewal_phase_ledger",
                 "core_blob_map", "classify_station_refusal", "mark_coverage_infeasible",
                 "validate_reuse_entry", "validate_batch_form", "validate_renewal_capsule",
                 "compute_batch_root", "validate_registry_candidate", "renewal_admits")
        need = (getattr(RP, "RENEWAL_CONTRACT_ID", None) == CONTRACT_ID
                and getattr(RP, "RENEWAL_ANCHOR", None) == ANCHOR
                and getattr(RP, "V2_POOL_SHA256", None) == V2_POOL_SHA
                and isinstance(getattr(RP, "RENEWAL_BUNDLE_SHA256", None), str)
                and all(callable(getattr(RP, f, None)) for f in seams))
        check("RN-0 pinned module d2_renewal_plan with the full REV-2 seam set "
              "(arms/pool/plan/ledger builders, core_blob_map, classify/infeasible/"
              "reuse/batch/capsule/root/registry/no-backfill validators)", need,
              f"missing={[f for f in seams if not callable(getattr(RP, f, None))]}")
        if not need:
            RP = None

    # ================= RN-1 — §6.1 arithmetic + identity =================
    exp_incident = days_between("2026-03-01", "2026-06-29")
    exp_activation = days_between("2026-03-19", "2026-07-17")
    exp_union = sorted(set(exp_incident) | set(exp_activation))
    A = date.fromisoformat(ANCHOR)
    check("RN-1a bar-side arithmetic self-check: incident 120d, activation "
          "[A-150,A-30)=120d, union 138d, A+7d=2026-08-23",
          len(exp_incident) == 120 and len(exp_activation) == 120 and len(exp_union) == 138
          and exp_activation[0] == (A - timedelta(days=150)).isoformat()
          and (A + timedelta(days=7)).isoformat() == "2026-08-23")
    if RP is not None:
        try:
            arms = RP.renewal_arms()
            ok = (list(arms["incident"]["days"]) == exp_incident
                  and list(arms["activation"]["days"]) == exp_activation
                  and arms["contract_id"] == CONTRACT_ID
                  and arms["activation_reference_day"] == ANCHOR
                  and arms["incident_reference"] == "2026-07-29"
                  and int(arms["embargo_days"]) == 30
                  and arms["valid_through"] == "2026-08-23"
                  and arms["expiry_utc"] == "2026-08-24T00:00:00Z")
            check("RN-1b renewal_arms() == the contract exactly", ok)
        except Exception as e:
            check("RN-1b renewal_arms() == the contract exactly", False, f"raised {e}")
    else:
        awaiting("RN-1b renewal_arms() == the contract exactly")
    try:
        import d2_step4b_campaign_run as CR
        import math
        pol = CR.POLICY
        vals = sorted([10.0, 3.0, 7.0, 1.0, 9.0, 5.0] * 10)
        k = max(0, math.ceil(pol["lower_quantile"] * len(vals)) - 1)
        check("RN-1c frozen executor POLICY unchanged (floor 60, 5% nearest-rank-ceil "
              "-> idx 2 of 60, valid days 7)",
              pol["min_admitted_days"] == 60 and pol["lower_quantile"] == 0.05
              and pol["candidate_valid_days"] == 7 and k == 2 and vals[k] == 1.0)
    except Exception as e:
        check("RN-1c frozen executor POLICY unchanged (floor 60, 5% nearest-rank-ceil "
              "-> idx 2 of 60, valid days 7)", False, f"raised {e}")

    # ================= RN-2 — §6.2 pool + plan + ledger + chronology =================
    v2_bytes = open(os.path.join(HERE, "d2_campaign_v2_candidate_pool.json"), "rb").read()
    check("RN-2a v2 candidate pool bytes == pinned V2_POOL_SHA256",
          sha(v2_bytes) == V2_POOL_SHA)

    prelim_bundle = {
        "schema": "geospec-d2-campaign-v2-renewal-bundle-v1", "contract_id": CONTRACT_ID,
        "activation_reference_day": ANCHOR,
        "arms": {"incident": {"days": exp_incident}, "activation": {"days": exp_activation}},
        "campaign_plan": {"eligible_carriers": TARGET_ORDER},
        "phase_ledger_sha256": None, "created_utc": "2026-08-16T01:15:00.000000Z"}

    if RP is not None:
        # RN-2b pool copy-only + refusals
        try:
            pool, pool_bytes = RP.build_renewal_candidate_pool(v2_bytes)
            v2_pool = json.loads(v2_bytes)

            def strip_env(p):
                return {k: v for k, v in p.items()
                        if k not in ("contract_id", "created_utc", "pool_digest",
                                     "schema", "source_pool_sha256")}
            refusals = []
            for bad in (json.loads(v2_bytes), v2_bytes + b" "):
                try:
                    RP.build_renewal_candidate_pool(bad)
                    refusals.append(False)
                except Exception:
                    refusals.append(True)
            check("RN-2b renewal pool = copy-only re-envelope of the v2 pool + dict/"
                  "mutation refusals + new contract envelope",
                  strip_env(pool) == strip_env(v2_pool)
                  and pool.get("contract_id") == CONTRACT_ID
                  and pool.get("source_pool_sha256") == V2_POOL_SHA and all(refusals)
                  and isinstance(pool.get("created_utc"), str))
        except Exception as e:
            check("RN-2b renewal pool = copy-only re-envelope of the v2 pool + dict/"
                  "mutation refusals + new contract envelope", False, f"raised {e}")

        # RN-2c/RN-2e REV3: LEDGER-FIRST construction with the plan carrying the binding.
        try:
            saved = RP.RENEWAL_BUNDLE_SHA256
            try:
                prelim_bytes = canon(prelim_bundle)
                RP.RENEWAL_BUNDLE_SHA256 = sha(prelim_bytes)
                ledger, ledger_bytes = RP.build_renewal_phase_ledger(prelim_bytes)
                ldays = sorted({r["scored_day"] for r in ledger["rows"]})
                ok_cov = ldays == exp_union and ledger.get("contract_id") == CONTRACT_ID \
                    and isinstance(ledger.get("created_utc"), str)
                final_bundle = dict(prelim_bundle)
                final_bundle["phase_ledger_sha256"] = sha(ledger_bytes)
                fb_bytes = canon(final_bundle)
                RP.RENEWAL_BUNDLE_SHA256 = sha(fb_bytes)
                plan, plan_bytes = RP.build_renewal_plan(fb_bytes)
                head = git_head_blobs()
                ok_accept = (plan["contract_id"] == CONTRACT_ID
                             and plan["activation_reference_day"] == ANCHOR
                             and plan_bytes == canon(plan)
                             and plan.get("core_blobs") == RP.core_blob_map()
                             and all(plan["core_blobs"].get(f) == head[f] and head[f]
                                     for f in CORE_FILES))
                # codex R2 #2: the plan CARRIES and MATCHES the reopened ledger digest
                ok_bind = plan.get("phase_ledger_sha256") == sha(ledger_bytes) \
                    == final_bundle["phase_ledger_sha256"]
                # builder REFUSES a bundle without the binding
                unbound = dict(prelim_bundle)          # phase_ledger_sha256 is None
                ub_bytes = canon(unbound)
                RP.RENEWAL_BUNDLE_SHA256 = sha(ub_bytes)
                try:
                    RP.build_renewal_plan(ub_bytes)
                    ok_unbound = False
                except Exception:
                    ok_unbound = True
                RP.RENEWAL_BUNDLE_SHA256 = sha(fb_bytes)
                mut = bytearray(fb_bytes)
                mut[-2] ^= 1
                try:
                    RP.build_renewal_plan(bytes(mut))
                    ok_mut = False
                except Exception:
                    ok_mut = True
                try:
                    RP.build_renewal_plan(canon({"contract_id":
                                                 "codex-d2-campaign-v2-2026-08-10-v1"}))
                    ok_wrong = False
                except Exception:
                    ok_wrong = True
                check("RN-2c build_renewal_plan ACCEPTS the canonical ledger-bound bundle "
                      "(pin==fixture sha), returns canonical plan bytes carrying "
                      "core_blobs==core_blob_map()==git-HEAD AND "
                      "phase_ledger_sha256==sha(reopened ledger bytes); REFUSES an "
                      "unbound bundle, a one-byte mutation, and a wrong contract",
                      ok_accept and ok_bind and ok_unbound and ok_mut and ok_wrong,
                      f"accept={ok_accept} bind={ok_bind} unbound_refused={ok_unbound} "
                      f"mut={ok_mut} wrong={ok_wrong}")
                ok_val = RP.validate_renewal_phase_ledger(ledger, final_bundle) is True
                old_v2 = dict(ledger)
                old_v2["contract_id"] = "codex-d2-campaign-v2-2026-08-10-v1"
                ok_old = RP.validate_renewal_phase_ledger(old_v2, final_bundle) is False
                lmut = json.loads(json.dumps(ledger))
                lmut["rows"] = lmut["rows"][1:]
                ok_lmut = RP.validate_renewal_phase_ledger(lmut, final_bundle) is False
                check("RN-2e renewal published-phase ledger: LEDGER-FIRST flow, exact "
                      "138-day union coverage, final-bundle sha binding; REFUSES the old "
                      "v2 ledger and a one-day mutation",
                      ok_cov and ok_val and ok_old and ok_lmut,
                      f"cov={ok_cov} val={ok_val} old={ok_old} mut={ok_lmut}")
            finally:
                RP.RENEWAL_BUNDLE_SHA256 = saved
        except Exception as e:
            check("RN-2c build_renewal_plan ACCEPTS the canonical ledger-bound bundle "
                  "(pin==fixture sha), returns canonical plan bytes carrying "
                  "core_blobs==core_blob_map()==git-HEAD AND "
                  "phase_ledger_sha256==sha(reopened ledger bytes); REFUSES an "
                  "unbound bundle, a one-byte mutation, and a wrong contract",
                  False, f"raised {e}")
            check("RN-2e renewal published-phase ledger: LEDGER-FIRST flow, exact "
                  "138-day union coverage, final-bundle sha binding; REFUSES the old "
                  "v2 ledger and a one-day mutation", False, f"raised {e}")
    else:
        awaiting("RN-2b renewal pool = copy-only re-envelope of the v2 pool + dict/"
                 "mutation refusals + new contract envelope",
                 "RN-2c build_renewal_plan ACCEPTS the canonical ledger-bound bundle "
                 "(pin==fixture sha), returns canonical plan bytes carrying "
                 "core_blobs==core_blob_map()==git-HEAD AND "
                 "phase_ledger_sha256==sha(reopened ledger bytes); REFUSES an "
                 "unbound bundle, a one-byte mutation, and a wrong contract",
                 "RN-2e renewal published-phase ledger: LEDGER-FIRST flow, exact "
                 "138-day union coverage, final-bundle sha binding; REFUSES the old "
                 "v2 ledger and a one-day mutation")

    ok2d, tail = run_suite("test_campaign_v2_phase075_registry_redkats_cayley.py")
    check("RN-2d COMPOSE: frozen v2 selection-semantics bar green", ok2d, tail)

    # ================= RN-3 — §6.3 repaired shell + overlap hold =================
    ok3a, tail = run_suite("test_d2_koeri_order_canon_redkats_cayley.py")
    check("RN-3a COMPOSE: order-canonicalization bar green (O-1..O-4 stays out of scope)",
          ok3a, tail)

    if RP is not None:
        try:
            import numpy as np
            t0 = datetime(2026, 3, 2, tzinfo=timezone.utc)
            overlap = [(np.zeros(30000), 100.0, t0),
                       (np.zeros(30000), 100.0, t0 + timedelta(seconds=150))]
            disjoint = [(np.zeros(30000), 100.0, t0),
                        (np.zeros(30000), 100.0, t0 + timedelta(seconds=400))]
            check("RN-3c classify_station_refusal: true post-sort overlap -> "
                  "TRUE_OVERLAP_UNRULED; disjoint pair -> not",
                  RP.classify_station_refusal(overlap) == "TRUE_OVERLAP_UNRULED"
                  and RP.classify_station_refusal(disjoint) != "TRUE_OVERLAP_UNRULED")
        except Exception as e:
            check("RN-3c classify_station_refusal: true post-sort overlap -> "
                  "TRUE_OVERLAP_UNRULED; disjoint pair -> not", False, f"raised {e}")
        check("RN-3d stale v1-era source attestation tripwire: literal 292b1069 absent "
              "from the plan-builder module source",
              STALE_PREFIX not in open(os.path.join(HERE, "d2_renewal_plan.py"),
                                       encoding="utf-8").read())
    else:
        awaiting("RN-3c classify_station_refusal: true post-sort overlap -> "
                 "TRUE_OVERLAP_UNRULED; disjoint pair -> not",
                 "RN-3d stale v1-era source attestation tripwire: literal 292b1069 absent "
                 "from the plan-builder module source")

    # ================= RN-4 — §6.4 fresh-root gates through the REAL executor =========
    ok4a, tail = run_suite("test_d2_step4b_executor_redkats_cayley.py")
    check("RN-4a COMPOSE: frozen v2 executor bar green", ok4a, tail)
    ok4b, tail = run_suite("test_d2_step4b_provider_redkats_cayley.py")
    check("RN-4b COMPOSE: frozen v2 provider bar green", ok4b, tail)

    if RP is not None:
        # RN-4f REV2: REAL acquire drive — overlap hold + zero-fetch infeasible + chronology
        try:
            import d2_step4b_campaign_run as CR
            td = tempfile.mkdtemp()
            plan, ledger = _mk_renewal_fixture_root(RP, td)
            prov = _SpyProviders(overlap_nslc="KO.S00..BHZ")
            err = None
            try:
                CR.acquire(plan, ledger, td, providers=prov, receipt=RECEIPT)
            except SystemExit as e:
                err = f"acquire SystemExit {e}"
            except Exception as e:
                err = f"acquire raised {type(e).__name__}: {e}"
            ok4f = False
            det = err or ""
            if err is None:
                attempts = [json.loads(l) for l in open(
                    os.path.join(td, "acquisition_attempts.jsonl"), encoding="utf-8")]
                s00 = [a for a in attempts if a.get("station_id") == "KO.S00"
                       or "KO.S00" in str(a.get("selected_nslc", ""))]
                held = any("TRUE_OVERLAP_UNRULED" in (a.get("reason_codes") or [])
                           for a in s00)
                admitted_with_overlap = any(
                    a.get("status") == "FETCHED" and "KO.S00" in str(a) and
                    "TRUE_OVERLAP_UNRULED" not in (a.get("reason_codes") or [])
                    for a in s00)
                later_created = max(plan["created_utc"], ledger["created_utc"])
                chron = all(a.get("attempted_utc", "") > later_created for a in attempts)
                ok4f = held and not admitted_with_overlap and chron
                det = f"held={held} admitted_anyway={admitted_with_overlap} chron={chron}"
            check("RN-4f REAL-acquire overlap hold: the scripted true-overlap station ends "
                  "as TRUE_OVERLAP_UNRULED with no admitted contribution, and every "
                  "attempt postdates max(plan, ledger) creation", ok4f, det)
        except Exception as e:
            check("RN-4f REAL-acquire overlap hold: the scripted true-overlap station ends "
                  "as TRUE_OVERLAP_UNRULED with no admitted contribution, and every "
                  "attempt postdates max(plan, ledger) creation", False, f"raised {e}")
        # RN-4g REV3 (codex R2 #4): acquire ENFORCES chronology — an injected clock that
        # predates plan/ledger creation must REFUSE (no attempts, no root artifacts).
        try:
            import d2_step4b_campaign_run as CR
            td3 = tempfile.mkdtemp()
            plan3, ledger3 = _mk_renewal_fixture_root(RP, td3)
            early = datetime(2026, 8, 15, 0, 0, 0, tzinfo=timezone.utc)   # before creation
            state = {"t": early}

            def early_clock():
                state["t"] += timedelta(seconds=1)
                return state["t"]

            refused = False
            try:
                CR.acquire(plan3, ledger3, td3, providers=_SpyProviders(),
                           receipt=RECEIPT, clock=early_clock)
            except (SystemExit, Exception):
                refused = True
            no_attempts = not os.path.exists(os.path.join(td3,
                                                          "acquisition_attempts.jsonl")) \
                or not open(os.path.join(td3, "acquisition_attempts.jsonl"),
                            encoding="utf-8").read().strip()
            check("RN-4g acquire ENFORCES chronology: an injected clock earlier than "
                  "max(plan, ledger) created_utc REFUSES with zero attempts recorded",
                  refused and no_attempts,
                  f"refused={refused} no_attempts={no_attempts}")
        except Exception as e:
            check("RN-4g acquire ENFORCES chronology: an injected clock earlier than "
                  "max(plan, ledger) created_utc REFUSES with zero attempts recorded",
                  False, f"raised {e}")
        try:
            import d2_step4b_campaign_run as CR
            td2 = tempfile.mkdtemp()
            plan2, ledger2 = _mk_renewal_fixture_root(RP, td2,
                                                      infeasible_carrier="istanbul_marmara")
            prov2 = _SpyProviders()
            try:
                CR.acquire(plan2, ledger2, td2, providers=prov2, receipt=RECEIPT)
            except BaseException:
                pass
            check("RN-4c COVERAGE_INFEASIBLE is enforced AT THE OPERATION: an either-arm-"
                  "below-60 carrier produces ZERO provider calls and zero fetch entries",
                  RP.mark_coverage_infeasible({"incident_potential": 59,
                                               "activation_potential": 96}) is True
                  and RP.mark_coverage_infeasible({"incident_potential": 96,
                                                   "activation_potential": 96}) is False
                  and len(prov2.calls) == 0, f"provider_calls={len(prov2.calls)}")
        except Exception as e:
            check("RN-4c COVERAGE_INFEASIBLE is enforced AT THE OPERATION: an either-arm-"
                  "below-60 carrier produces ZERO provider calls and zero fetch entries",
                  False, f"raised {e}")
        # RN-4d REV3 (codex R2 #3): validation against the EXPECTED immutable-object
        # record — well-formed-but-WRONG values must refuse.
        try:
            base = {"reuse": True, "attested_utc": "2026-08-16T02:00:00Z",
                    "provider_identity": "s3://scedc-pds/obj1",
                    "contract_id": CONTRACT_ID,
                    "session": {"start": "2026-03-01T07:00:00Z", "end": "2026-03-02T07:00:00Z"},
                    "sha256": "a" * 64}
            expected = {"provider_identity": "s3://scedc-pds/obj1",
                        "session": {"start": "2026-03-01T07:00:00Z",
                                    "end": "2026-03-02T07:00:00Z"},
                        "sha256": "a" * 64}
            kw = {"campaign_start_utc": "2026-08-16T01:30:00Z",
                  "contract_id": CONTRACT_ID, "expected": expected}
            cases = [
                (dict(base), True),
                ({**base, "attested_utc": "2026-08-15T00:00:00Z"}, False),
                ({**base, "contract_id": "codex-d2-campaign-v2-2026-08-10-v1"}, False),
                ({**base, "provider_identity": ""}, False),
                ({**base, "sha256": "b" * 63}, False),
                ({k: v for k, v in base.items() if k != "session"}, False),
                # codex R2 #3 well-formed-but-WRONG negatives:
                ({**base, "provider_identity": "s3://scedc-pds/obj2"}, False),
                ({**base, "session": {"start": "2026-03-01T08:00:00Z",
                                      "end": "2026-03-02T08:00:00Z"}}, False),
                ({**base, "sha256": "c" * 64}, False),
            ]
            ok4d = all(RP.validate_reuse_entry(c, **kw) is exp for c, exp in cases)
            check("RN-4d reuse attestation validates against the EXPECTED immutable-object "
                  "record: pre-campaign/wrong-contract/empty/malformed refuse AND "
                  "well-formed-but-WRONG identity, session, and digest refuse; only the "
                  "exact expected re-attestation accepts", ok4d)
        except Exception as e:
            check("RN-4d reuse attestation validates against the EXPECTED immutable-object "
                  "record: pre-campaign/wrong-contract/empty/malformed refuse AND "
                  "well-formed-but-WRONG identity, session, and digest refuse; only the "
                  "exact expected re-attestation accepts", False, f"raised {e}")
        # RN-4e REV2: table-driven batch form
        try:
            def batch(carriers):
                return {"contract_id": CONTRACT_ID,
                        "carriers": [{"carrier_key": k, "state": s} for k, s in carriers]}
            T = "ADMITTED_CANDIDATE"
            good = batch([(k, T) for k in TARGET_ORDER])
            zero = batch([(k, "COVERAGE_INFEASIBLE") for k in TARGET_ORDER])
            cases = [
                (good, True),
                (zero, True),                                     # honest 0-candidate form
                (batch([(TARGET_ORDER[0], T)]), False),           # favorable subset
                (batch([(k, T) for k in TARGET_ORDER] + [(TARGET_ORDER[0], T)]), False),
                (batch([(k, T) for k in reversed(TARGET_ORDER)]), False),   # reordered
                (batch([(k, T) for k in TARGET_ORDER[:2] + ["kumamoto"]]), False),
                ({**good, "contract_id": "x"}, False),
                (batch([(TARGET_ORDER[0], "RUNNING")] +
                       [(k, T) for k in TARGET_ORDER[1:]]), False),          # nonterminal
            ]
            ok4e = all(RP.validate_batch_form(c) is exp for c, exp in cases)
            check("RN-4e batch form is table-driven: exactly the three frozen targets in "
                  "the outcome-blind order with terminal states; duplicates, reorder, "
                  "substitutes, wrong contract, and nonterminal states REFUSE; the "
                  "honest 0-candidate form accepts", ok4e)
        except Exception as e:
            check("RN-4e batch form is table-driven: exactly the three frozen targets in "
                  "the outcome-blind order with terminal states; duplicates, reorder, "
                  "substitutes, wrong contract, and nonterminal states REFUSE; the "
                  "honest 0-candidate form accepts", False, f"raised {e}")
    else:
        awaiting("RN-4f REAL-acquire overlap hold: the scripted true-overlap station ends "
                 "as TRUE_OVERLAP_UNRULED with no admitted contribution, and every "
                 "attempt postdates max(plan, ledger) creation",
                 "RN-4g acquire ENFORCES chronology: an injected clock earlier than "
                 "max(plan, ledger) created_utc REFUSES with zero attempts recorded",
                 "RN-4c COVERAGE_INFEASIBLE is enforced AT THE OPERATION: an either-arm-"
                 "below-60 carrier produces ZERO provider calls and zero fetch entries",
                 "RN-4d reuse attestation validates against the EXPECTED immutable-object "
                 "record: pre-campaign/wrong-contract/empty/malformed refuse AND "
                 "well-formed-but-WRONG identity, session, and digest refuse; only the "
                 "exact expected re-attestation accepts",
                 "RN-4e batch form is table-driven: exactly the three frozen targets in "
                 "the outcome-blind order with terminal states; duplicates, reorder, "
                 "substitutes, wrong contract, and nonterminal states REFUSE; the "
                 "honest 0-candidate form accepts")

    # ================= RN-5 — §6.5 capsule/registry/batch-root/stale/no-backfill ======
    import importlib
    try:
        FC = importlib.import_module("fault_correlation")
        SD = importlib.import_module("seismic_data")
        head_commit = git_head()
        cap = {"schema": "geospec-d2-calibration-v1", "region": "istanbul_marmara",
               "band_tag": "1-10Hz", "processing_version": SD.PROCESSING_VERSION,
               "topology_version": "t1", "threshold": 0.21,
               "calibration_window": {"start": "2026-03-19", "end": "2026-07-17"},
               "source_commit": head_commit,
               "input_manifest_sha256": "a" * 64, "replay_output_sha256": "b" * 64,
               "issued_utc": "2026-08-16T02:00:00Z", "valid_through": "2026-08-23"}
        capdir = tempfile.mkdtemp()
        body = json.dumps(cap, sort_keys=True).encode()
        cappath = os.path.join(capdir, "istanbul_marmara.json")
        open(cappath, "wb").write(body)
        pin = sha(body)

        def load(day, expected=pin):
            try:
                FC.load_calibration_capsule(
                    "istanbul_marmara", day, band_tag="1-10Hz",
                    processing_version=SD.PROCESSING_VERSION, topology_version="t1",
                    capsule_dir=capdir, expected_sha256=expected)
                return "ADMIT"
            except FC.CalibrationUnavailable:
                return "REFUSE"

        check("RN-5a stale boundary LOCK: valid_through 2026-08-23 ADMITS scored day "
              "08-23, REFUSES 08-24 (expiry 2026-08-24T00:00:00Z)",
              load("2026-08-23") == "ADMIT" and load("2026-08-24") == "REFUSE")
        mut = dict(cap)
        mut["threshold"] = 0.5
        open(cappath, "wb").write(json.dumps(mut, sort_keys=True).encode())
        check("RN-5b registered-digest binding LOCK: mutation under the unchanged "
              "registry pin REFUSES", load("2026-08-20") == "REFUSE")
        open(cappath, "wb").write(body)     # restore for RN-5d fixtures
    except Exception as e:
        check("RN-5a stale boundary LOCK: valid_through 2026-08-23 ADMITS scored day "
              "08-23, REFUSES 08-24 (expiry 2026-08-24T00:00:00Z)", False, f"raised {e}")
        check("RN-5b registered-digest binding LOCK: mutation under the unchanged "
              "registry pin REFUSES", False, f"raised {e}")
        cap, capdir = None, None

    if RP is not None and cap is not None:
        # RN-5c REV2: per-field mutation table (every required scalar/binding)
        try:
            kwv = {"expected_source_commit": git_head()}
            muts = [({}, True)]
            for field, val in [("region", "kumamoto"), ("band_tag", "2-20Hz"),
                               ("processing_version", "bogus"), ("topology_version", "t2"),
                               ("threshold", float("nan")), ("threshold", None),
                               ("calibration_window", {"start": "2026-03-13",
                                                       "end": "2026-07-11"}),
                               ("input_manifest_sha256", "a" * 63),
                               ("replay_output_sha256", ""),
                               ("issued_utc", "2027-01-01T00:00:00Z"),
                               ("valid_through", "2026-08-30"),
                               ("source_commit", STALE_PREFIX + "e" * 32),
                               ("source_commit", "f" * 40)]:
                m = dict(cap)
                m[field] = val
                muts.append((m, False))
            ok5c = all(
                RP.validate_renewal_capsule({**cap, **m} if m else dict(cap), **kwv) is exp
                for m, exp in [({}, True)] ) and all(
                RP.validate_renewal_capsule(m, **kwv) is exp for m, exp in muts[1:])
            check("RN-5c capsule validator is table-driven: EVERY required scalar/binding "
                  "mutation refuses independently (region/band/processing/topology/window/"
                  "non-finite threshold/manifest sha/replay sha/future issued/wrong "
                  "valid_through/stale-or-mismatched source_commit); nominal HEAD-bound "
                  "capsule accepts", ok5c)
        except Exception as e:
            check("RN-5c capsule validator is table-driven: EVERY required scalar/binding "
                  "mutation refuses independently (region/band/processing/topology/window/"
                  "non-finite threshold/manifest sha/replay sha/future issued/wrong "
                  "valid_through/stale-or-mismatched source_commit); nominal HEAD-bound "
                  "capsule accepts", False, f"raised {e}")
        # RN-5d REV3 (codex R2 #1): BOTH artifacts reopened; declared-digest mismatch
        # REFUSES; root only from reopened-and-matched bytes.
        try:
            entries = []
            for i, carrier in enumerate(TARGET_ORDER):
                cbody = json.dumps({**cap, "region": carrier}, sort_keys=True).encode()
                p = os.path.join(capdir, f"{carrier}_renewal.json")
                open(p, "wb").write(cbody)
                mbody = canon({"carrier": carrier, "objects": [{"sha256": "e" * 64}]})
                mp = os.path.join(capdir, f"{carrier}_manifest.json")
                open(mp, "wb").write(mbody)
                entries.append({"carrier_key": carrier, "capsule_path": p,
                                "capsule_sha256": sha(cbody),
                                "manifest_path": mp, "manifest_sha256": sha(mbody)})
            root1 = RP.compute_batch_root(entries)
            root2 = RP.compute_batch_root([dict(e2) for e2 in entries])
            # declared-digest mismatch refusals (codex's exact probes):
            zeros = [dict(entries[0], capsule_sha256="0" * 64)] + entries[1:]
            try:
                RP.compute_batch_root(zeros)
                ok_zero = False
            except Exception:
                ok_zero = True
            ghost = [dict(entries[0], manifest_path=os.path.join(capdir, "ghost.json"))] \
                + entries[1:]
            try:
                RP.compute_batch_root(ghost)
                ok_ghost = False
            except Exception:
                ok_ghost = True
            wrongm = [dict(entries[0], manifest_sha256="f" * 64)] + entries[1:]
            try:
                RP.compute_batch_root(wrongm)
                ok_wrongm = False
            except Exception:
                ok_wrongm = True
            open(entries[0]["capsule_path"], "ab").write(b" ")
            tampered = [{**entries[0], "capsule_sha256":
                         sha(open(entries[0]["capsule_path"], "rb").read())}] + entries[1:]
            root3 = RP.compute_batch_root(tampered)
            record = {"capsule_path": entries[0]["capsule_path"],
                      "capsule_sha256": entries[0]["capsule_sha256"],
                      "region": TARGET_ORDER[0], "topology_version": "t1",
                      "contract_id": CONTRACT_ID, "batch_root_sha256": root1,
                      "verification_receipts": ["r1", "r2", "r3"],
                      "lift_effective_utc": None}
            expected = dict(record)
            ok_rec = RP.validate_registry_candidate(record, expected=expected) is True
            flips = []
            for f2 in ("capsule_sha256", "contract_id", "batch_root_sha256",
                       "topology_version", "region"):
                bad = dict(record)
                bad[f2] = "TAMPERED"
                flips.append(RP.validate_registry_candidate(bad, expected=expected) is False)
            empty_rx = dict(record)
            empty_rx["verification_receipts"] = []
            flips.append(RP.validate_registry_candidate(empty_rx, expected=expected) is False)
            check("RN-5d batch root: BOTH capsule and manifest REOPENED and matched "
                  "against declared digests (zeros-capsule-sha, missing-manifest, and "
                  "wrong-manifest-sha all REFUSE; root1==root2; tamper changes the root) "
                  "and the registry candidate binds with per-field flip refusals",
                  root1 == root2 and root3 != root1 and ok_zero and ok_ghost
                  and ok_wrongm and ok_rec and all(flips),
                  f"eq={root1 == root2} tamper={root3 != root1} zero={ok_zero} "
                  f"ghost={ok_ghost} wrongm={ok_wrongm} rec={ok_rec}")
        except Exception as e:
            check("RN-5d batch root: BOTH capsule and manifest REOPENED and matched "
                  "against declared digests (zeros-capsule-sha, missing-manifest, and "
                  "wrong-manifest-sha all REFUSE; root1==root2; tamper changes the root) "
                  "and the registry candidate binds with per-field flip refusals",
                  False, f"raised {e}")
        # RN-5e REV2: no-backfill — lift/effective boundary is executable
        try:
            lift = "2026-08-20T14:00:00Z"
            ok5e = (RP.renewal_admits("2026-08-18", cap, lift) is False   # pre-lift gap day
                    and RP.renewal_admits("2026-08-21", cap, lift) is True
                    and RP.renewal_admits("2026-08-21", cap, None) is False  # unlifted
                    and RP.renewal_admits("2026-08-24", cap, lift) is False)  # stale
            check("RN-5e no-backfill is executable: a gap day already refused stale "
                  "(before lift_effective) is NEVER retroactively admitted; unlifted "
                  "capsules admit nothing; post-expiry stays stale", ok5e)
        except Exception as e:
            check("RN-5e no-backfill is executable: a gap day already refused stale "
                  "(before lift_effective) is NEVER retroactively admitted; unlifted "
                  "capsules admit nothing; post-expiry stays stale", False, f"raised {e}")
        # RN-5f REV4.1 (codex 8e4d8fc #2): the REAL producer-to-minter edge.
        # (a) direct branch KATs on the pure seam; (b) candidate-producing hermetic
        # real-acquire drive with a spy-wrapped minter; (c) raising-minter propagation.
        # Source inspection is DIAGNOSTIC ONLY (reported, never load-bearing).
        try:
            import d2_step4b_campaign_run as CR
            head = git_head()
            mint = getattr(CR, "mint_capsule", None)
            ok_branch = False
            if callable(mint):
                kwm = dict(carrier="istanbul_marmara", threshold=0.21,
                           calibration_window={"start": "2026-03-19", "end": "2026-07-17"},
                           replay_output_sha256="b" * 64, valid_through="2026-08-23",
                           input_manifest_sha256="a" * 64,
                           issued_utc="2026-08-17T14:00:00Z", producer_commit=head)
                cap_r = mint({"contract_id": CONTRACT_ID}, **kwm)
                cap_v = mint({"contract_id": "codex-d2-step4b-2026-08-09-v1"}, **kwm)
                ok_branch = (isinstance(cap_r, dict)
                             and cap_r.get("source_commit") == head
                             and RP.validate_renewal_capsule(
                                 cap_r, expected_source_commit=head) is True
                             and RP.validate_renewal_capsule(
                                 cap_r, expected_source_commit="f" * 40) is False
                             and cap_v.get("source_commit") == CR.IMPLEMENTATION_COMMIT)
            check("RN-5f-a pure mint seam branch KATs: renewal plan -> supplied producing "
                  "commit (validates; wrong commit refuses); v2 plan -> "
                  "IMPLEMENTATION_COMMIT preserved",
                  ok_branch, "mint_capsule seam absent" if not callable(mint) else "")
        except Exception as e:
            check("RN-5f-a pure mint seam branch KATs: renewal plan -> supplied producing "
                  "commit (validates; wrong commit refuses); v2 plan -> "
                  "IMPLEMENTATION_COMMIT preserved", False, f"raised {e}")
        try:
            import d2_step4b_campaign_run as CR
            saved_pol = dict(CR.POLICY)
            spy_calls = []
            real_mint = getattr(CR, "mint_capsule", None)
            td_c = tempfile.mkdtemp()
            plan_c, ledger_c = _mk_candidate_fixture_root(RP, td_c)
            drive_err = None
            try:
                CR.POLICY["min_admitted_days"] = 2
                if callable(real_mint):
                    def spy_mint(plan_arg, **kw):
                        out = real_mint(plan_arg, **kw)
                        # REV 4.2: SNAPSHOT (json round-trip) so post-mint mutation of
                        # the same dict object cannot retro-edit the spy record.
                        spy_calls.append({"producer_commit": kw.get("producer_commit"),
                                          "carrier": kw.get("carrier"),
                                          "out": json.loads(json.dumps(out))})
                        return out
                    CR.mint_capsule = spy_mint
                try:
                    CR.acquire(plan_c, ledger_c, td_c, providers=_CandidateProviders(),
                               receipt=RECEIPT)
                except BaseException as e:
                    drive_err = f"{type(e).__name__}: {e}"
            finally:
                CR.POLICY.clear()
                CR.POLICY.update(saved_pol)
                if callable(real_mint):
                    CR.mint_capsule = real_mint
            adm_p = os.path.join(td_c, "admission_results.json")
            regions_c = []
            if os.path.exists(adm_p):
                regions_c = json.loads(open(adm_p, encoding="utf-8").read()).get(
                    "regions", [])
            cand = [r for r in regions_c if isinstance(r, dict)
                    and r.get("status") == "ADMITTED_CANDIDATE"]
            ok_drive = False
            det = f"drive_err={drive_err} candidates={len(cand)} spy_calls={len(spy_calls)}"
            if cand and not drive_err:
                row = cand[0]
                cpath = os.path.join(td_c, row.get("capsule_path") or "missing")
                emitted = json.loads(open(cpath, encoding="utf-8").read()) \
                    if os.path.exists(cpath) else None
                im_c = json.loads(open(os.path.join(td_c, "input_manifest.json"),
                                       encoding="utf-8").read())
                producers = {c["producer_commit"] for c in spy_calls}
                spy_out = spy_calls[0]["out"] if spy_calls else None
                with open(os.path.join(td_c, "input_manifest.json"), "rb") as fh:
                    im_bytes = fh.read()
                ok_drive = (len(spy_calls) == len(cand) and len(producers) == 1
                            and emitted is not None and spy_out is not None
                            and emitted == spy_out
                            and emitted.get("input_manifest_sha256") == sha(im_bytes)
                            and emitted.get("source_commit") == next(iter(producers))
                            and emitted.get("source_commit")
                            == im_c.get("producer_commit"))
                det += (f" emitted_src={str((emitted or {}).get('source_commit'))[:9]}"
                        f" manifest_src={str(im_c.get('producer_commit'))[:9]}")
            check("RN-5f-b REAL-acquire candidate drive: the hermetic fixture reaches "
                  "ADMITTED_CANDIDATE, the spy-wrapped minter is called once per emitted "
                  "candidate with ONE producer commit, the WRITTEN capsule equals the "
                  "SNAPSHOTTED spy result COMPLETELY (no fields excluded), its "
                  "input_manifest_sha256 == sha256(reopened manifest bytes), and its "
                  "source_commit == that commit == input_manifest.producer_commit",
                  ok_drive, det)
            # (c) raising replacement minter must PROPAGATE through the producer
            ok_prop = False
            if callable(real_mint) and cand:
                td_p = tempfile.mkdtemp()
                plan_p, ledger_p = _mk_candidate_fixture_root(RP, td_p)
                class _Boom(Exception):
                    pass
                def boom_mint(*a, **k):
                    raise _Boom("bar sentinel")
                saved_pol2 = dict(CR.POLICY)
                try:
                    CR.POLICY["min_admitted_days"] = 2
                    CR.mint_capsule = boom_mint
                    try:
                        CR.acquire(plan_p, ledger_p, td_p,
                                   providers=_CandidateProviders(), receipt=RECEIPT)
                        ok_prop = False
                    except _Boom as e:
                        # REV 4.2: only the injected sentinel type+value counts.
                        ok_prop = str(e) == "bar sentinel"
                    except BaseException:
                        ok_prop = False
                finally:
                    CR.POLICY.clear()
                    CR.POLICY.update(saved_pol2)
                    CR.mint_capsule = real_mint
            check("RN-5f-c a raising replacement minter PROPAGATES through the real "
                  "producer path as the injected sentinel type+value -- translation "
                  "to any other exception is a failed check (no silent swallow)", ok_prop,
                  "unreachable while RN-5f-b red" if not (callable(real_mint) and cand)
                  else "")
            # diagnostics only (never load-bearing): source shape
            src = open(os.path.join(HERE, "d2_step4b_campaign_run.py"),
                       encoding="utf-8").read()
            print(f"    [diag] mint-call-sites={src.count('mint_capsule')} "
                  f"stale-literal-present={'source_commit' + chr(34) + ': IMPLEMENTATION_COMMIT' in src}")
        except Exception as e:
            check("RN-5f-b REAL-acquire candidate drive: the hermetic fixture reaches "
                  "ADMITTED_CANDIDATE, the spy-wrapped minter is called once per emitted "
                  "candidate with ONE producer commit, the WRITTEN capsule equals the "
                  "SNAPSHOTTED spy result COMPLETELY (no fields excluded), its "
                  "input_manifest_sha256 == sha256(reopened manifest bytes), and its "
                  "source_commit == that commit == input_manifest.producer_commit",
                  False, f"raised {e}")
            check("RN-5f-c a raising replacement minter PROPAGATES through the real "
                  "producer path as the injected sentinel type+value -- translation "
                  "to any other exception is a failed check (no silent swallow)", False, f"raised {e}")
        # RN-5f-d/e REV 4.3 (grassmann 2b27621 finding 1): the RESUME executor is
        # the REAL production edge (RC1c) with its OWN capsule assembly -- drive IT.
        try:
            import d2_step4b_campaign_run as CR
            import d2_step4b_resume as RS
            saved_pol3 = dict(CR.POLICY)
            spy_calls_r = []
            real_mint2 = getattr(CR, "mint_capsule", None)
            td_r = tempfile.mkdtemp()
            plan_r, ledger_r = _mk_candidate_fixture_root(RP, td_r)
            drive_err_r = None
            try:
                CR.POLICY["min_admitted_days"] = 2
                if callable(real_mint2):
                    def spy_mint_r(plan_arg, **kw):
                        out = real_mint2(plan_arg, **kw)
                        spy_calls_r.append({"producer_commit": kw.get("producer_commit"),
                                            "carrier": kw.get("carrier"),
                                            "out": json.loads(json.dumps(out))})
                        return out
                    CR.mint_capsule = spy_mint_r
                try:
                    RS.run_resume_campaign(plan_r, td_r,
                                           providers=_CandidateProviders(),
                                           receipt=RECEIPT)
                except BaseException as e:
                    drive_err_r = f"{type(e).__name__}: {e}"
            finally:
                CR.POLICY.clear()
                CR.POLICY.update(saved_pol3)
                if callable(real_mint2):
                    CR.mint_capsule = real_mint2
            adm_pr = os.path.join(td_r, "admission_results.json")
            regions_r = []
            if os.path.exists(adm_pr):
                regions_r = json.loads(open(adm_pr, encoding="utf-8").read()).get(
                    "regions", [])
            cand_r = [r for r in regions_r if isinstance(r, dict)
                      and r.get("status") == "ADMITTED_CANDIDATE"]
            ok_resume = False
            det_r = (f"drive_err={drive_err_r} candidates={len(cand_r)} "
                     f"spy_calls={len(spy_calls_r)}")
            if cand_r and not drive_err_r:
                row_r = cand_r[0]
                cpath_r = os.path.join(td_r, row_r.get("capsule_path") or "missing")
                emitted_r = json.loads(open(cpath_r, encoding="utf-8").read()) \
                    if os.path.exists(cpath_r) else None
                im_pr = os.path.join(td_r, "input_manifest.json")
                im_bytes_r = open(im_pr, "rb").read() if os.path.exists(im_pr) else b""
                im_r = json.loads(im_bytes_r.decode("utf-8")) if im_bytes_r else {}
                producers_r = {c["producer_commit"] for c in spy_calls_r}
                spy_out_r = spy_calls_r[0]["out"] if spy_calls_r else None
                ok_resume = (len(spy_calls_r) == len(cand_r)
                             and len(producers_r) == 1
                             and emitted_r is not None and spy_out_r is not None
                             and emitted_r == spy_out_r
                             and emitted_r.get("input_manifest_sha256") == sha(im_bytes_r)
                             and emitted_r.get("source_commit") == next(iter(producers_r))
                             and emitted_r.get("source_commit")
                             == im_r.get("producer_commit"))
                det_r += (f" emitted_src={str((emitted_r or {}).get('source_commit'))[:9]}"
                          f" manifest_src={str(im_r.get('producer_commit'))[:9]}")
            check("RN-5f-d RESUME-edge candidate drive (the REAL production edge, "
                  "RC1c): run_resume_campaign on the hermetic fixture reaches "
                  "ADMITTED_CANDIDATE, the spy-wrapped CR.mint_capsule seam is called "
                  "once per emitted candidate with ONE producer commit, the WRITTEN "
                  "capsule equals the SNAPSHOTTED spy result COMPLETELY, its "
                  "input_manifest_sha256 == sha256(reopened manifest bytes), and its "
                  "source_commit == that commit == input_manifest.producer_commit (a "
                  "stale IMPLEMENTATION_COMMIT literal in the resume assembly cannot "
                  "pass)", ok_resume, det_r)
            ok_prop_r = False
            if callable(real_mint2) and cand_r:
                td_r2 = tempfile.mkdtemp()
                plan_r2, _lg2 = _mk_candidate_fixture_root(RP, td_r2)
                class _BoomR(Exception):
                    pass
                def boom_mint_r(*a, **k):
                    raise _BoomR("bar sentinel resume")
                saved_pol4 = dict(CR.POLICY)
                try:
                    CR.POLICY["min_admitted_days"] = 2
                    CR.mint_capsule = boom_mint_r
                    try:
                        RS.run_resume_campaign(plan_r2, td_r2,
                                               providers=_CandidateProviders(),
                                               receipt=RECEIPT)
                        ok_prop_r = False
                    except _BoomR as e:
                        ok_prop_r = str(e) == "bar sentinel resume"
                    except BaseException:
                        ok_prop_r = False
                finally:
                    CR.POLICY.clear()
                    CR.POLICY.update(saved_pol4)
                    CR.mint_capsule = real_mint2
            check("RN-5f-e a raising replacement minter PROPAGATES through the RESUME "
                  "edge as the injected sentinel type+value -- a resume path that "
                  "never calls the seam, swallows, or translates is a failed check",
                  ok_prop_r,
                  "unreachable while RN-5f-d red" if not (callable(real_mint2)
                                                          and cand_r) else "")
            src_r = open(os.path.join(HERE, "d2_step4b_resume.py"),
                         encoding="utf-8").read()
            print(f"    [diag] resume mint_capsule-refs={src_r.count('mint_capsule')} "
                  f"resume-stale-literal="
                  f"{'source_commit' + chr(34) + ': CR.IMPLEMENTATION_COMMIT' in src_r}")
        except Exception as e:
            check("RN-5f-d RESUME-edge candidate drive (the REAL production edge, "
                  "RC1c): run_resume_campaign on the hermetic fixture reaches "
                  "ADMITTED_CANDIDATE, the spy-wrapped CR.mint_capsule seam is called "
                  "once per emitted candidate with ONE producer commit, the WRITTEN "
                  "capsule equals the SNAPSHOTTED spy result COMPLETELY, its "
                  "input_manifest_sha256 == sha256(reopened manifest bytes), and its "
                  "source_commit == that commit == input_manifest.producer_commit (a "
                  "stale IMPLEMENTATION_COMMIT literal in the resume assembly cannot "
                  "pass)", False, f"raised {e}")
            check("RN-5f-e a raising replacement minter PROPAGATES through the RESUME "
                  "edge as the injected sentinel type+value -- a resume path that "
                  "never calls the seam, swallows, or translates is a failed check",
                  False, f"raised {e}")
        # RN-5g REV4.1 (codex 8e4d8fc #1): producer-schema replacement-root check
        # + ALWAYS-RUNNING lock fixtures proving the check itself discriminates.
        try:
            head = git_head()
            nominal = _mk_synthetic_replacement_root(RP, cap, head)
            ok_nom, det_nom = validate_replacement_root(nominal, RP)
            refusals = []
            # dict-shape regions
            r1 = _mk_synthetic_replacement_root(RP, cap, head)
            adm = json.loads(open(os.path.join(r1, "admission_results.json")).read())
            adm["regions"] = {"istanbul_marmara": adm["regions"][0]}
            open(os.path.join(r1, "admission_results.json"), "w").write(json.dumps(adm))
            refusals.append(not validate_replacement_root(r1, RP)[0])
            # empty blob map
            r2 = _mk_synthetic_replacement_root(RP, cap, head)
            pl = json.loads(open(os.path.join(r2, "campaign_plan.json")).read())
            pl["core_blobs"] = {}
            open(os.path.join(r2, "campaign_plan.json"), "w").write(json.dumps(pl))
            refusals.append(not validate_replacement_root(r2, RP)[0])
            # missing key
            r3 = _mk_synthetic_replacement_root(RP, cap, head)
            pl = json.loads(open(os.path.join(r3, "campaign_plan.json")).read())
            pl["core_blobs"].pop(CORE_FILES[0])
            open(os.path.join(r3, "campaign_plan.json"), "w").write(json.dumps(pl))
            refusals.append(not validate_replacement_root(r3, RP)[0])
            # extra key
            r4 = _mk_synthetic_replacement_root(RP, cap, head)
            pl = json.loads(open(os.path.join(r4, "campaign_plan.json")).read())
            pl["core_blobs"]["monitoring/src/extra.py"] = "a" * 40
            open(os.path.join(r4, "campaign_plan.json"), "w").write(json.dumps(pl))
            refusals.append(not validate_replacement_root(r4, RP)[0])
            # wrong blob value
            r5 = _mk_synthetic_replacement_root(RP, cap, head)
            pl = json.loads(open(os.path.join(r5, "campaign_plan.json")).read())
            pl["core_blobs"][CORE_FILES[0]] = "b" * 40
            open(os.path.join(r5, "campaign_plan.json"), "w").write(json.dumps(pl))
            refusals.append(not validate_replacement_root(r5, RP)[0])
            # capsule-region / row-carrier mismatch
            r6 = _mk_synthetic_replacement_root(RP, cap, head)
            cp6 = os.path.join(r6, "capsules", "istanbul_marmara_calibration.json")
            c6 = json.loads(open(cp6).read())
            c6["region"] = "socal_coachella"
            open(cp6, "w").write(json.dumps(c6, sort_keys=True))
            refusals.append(not validate_replacement_root(r6, RP)[0])
            # capsule manifest-digest cross-link mismatch (codex c37ef5c HIGH-1)
            r7 = _mk_synthetic_replacement_root(RP, cap, head)
            cp7 = os.path.join(r7, "capsules", "istanbul_marmara_calibration.json")
            c7 = json.loads(open(cp7).read())
            c7["input_manifest_sha256"] = "f" * 64
            open(cp7, "w").write(json.dumps(c7, sort_keys=True))
            refusals.append(not validate_replacement_root(r7, RP)[0])
            check("RN-5g-lock the replacement-root check DISCRIMINATES: nominal "
                  "producer-schema synthetic root PASSES; dict-shape regions, empty blob "
                  "map, missing key, extra key, wrong blob, capsule/row region "
                  "mismatch, and capsule-manifest digest mismatch ALL refuse", ok_nom and all(refusals),
                  f"nominal={ok_nom}({det_nom}) refusals={refusals}")
        except Exception as e:
            check("RN-5g-lock the replacement-root check DISCRIMINATES: nominal "
                  "producer-schema synthetic root PASSES; dict-shape regions, empty blob "
                  "map, missing key, extra key, wrong blob, capsule/row region "
                  "mismatch, and capsule-manifest digest mismatch ALL refuse", False, f"raised {e}")
        try:
            root_env = os.environ.get("D2_RENEWAL_ROOT")
            if root_env and os.path.isdir(root_env):
                ok5g, det5g = validate_replacement_root(root_env, RP)
                check("RN-5g replacement-root validation (producer schema; capsule == "
                      "manifest == plan-HEAD blob vector + capsule manifest-digest "
                      "cross-link on reopened bytes)",
                      ok5g, det5g)
            else:
                check("RN-5g replacement-root validation (producer schema; capsule == "
                      "manifest == plan-HEAD blob vector + capsule manifest-digest "
                      "cross-link on reopened bytes)",
                      False, "PENDING ROOT (set D2_RENEWAL_ROOT; verification lanes run "
                             "this against the sealed replacement root)")
        except Exception as e:
            check("RN-5g replacement-root validation (producer schema; capsule == "
                  "manifest == plan-HEAD blob vector + capsule manifest-digest "
                  "cross-link on reopened bytes)",
                  False, f"raised {e}")
    else:
        awaiting("RN-5c capsule validator is table-driven: EVERY required scalar/binding "
                 "mutation refuses independently (region/band/processing/topology/window/"
                 "non-finite threshold/manifest sha/replay sha/future issued/wrong "
                 "valid_through/stale-or-mismatched source_commit); nominal HEAD-bound "
                 "capsule accepts",
                 "RN-5d batch root: BOTH capsule and manifest REOPENED and matched "
                 "against declared digests (zeros-capsule-sha, missing-manifest, and "
                 "wrong-manifest-sha all REFUSE; root1==root2; tamper changes the root) "
                 "and the registry candidate binds with per-field flip refusals",
                 "RN-5e no-backfill is executable: a gap day already refused stale "
                 "(before lift_effective) is NEVER retroactively admitted; unlifted "
                 "capsules admit nothing; post-expiry stays stale",
                 "RN-5f capsule_source_commit(plan): renewal plan -> producing git HEAD, "
                 "v2 plan -> frozen IMPLEMENTATION_COMMIT (v2 semantics preserved); the "
                 "executor mint CALLS the seam and the literal "
                 "source_commit: IMPLEMENTATION_COMMIT mint is gone from campaign_run",
                 "RN-5f-d RESUME-edge candidate drive (the REAL production edge, "
                 "RC1c) with full snapshot equality + manifest cross-link",
                 "RN-5f-e a raising replacement minter PROPAGATES through the RESUME "
                 "edge as the injected sentinel type+value",
                 "RN-5g staged-root capsules pass validate_renewal_capsule against "
                 "the root's recorded producer_commit (re-mint acceptance)")


main()
print()
if FAILS:
    print(f"D2 RENEWAL RED-KAT FAILURES: {FAILS}")
    sys.exit(1)
print("ALL D2 RENEWAL RED-KATs PASS (REV 2: end-to-end bindings per codex f7fd6f3; "
      "provider I/O still gated on the fresh fire-time owner go naming the contract id "
      "and A)")
