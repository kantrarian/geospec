#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D2 STEP-4B EXECUTOR red-KATs (cayley, 2026-08-09) — freezes codex `0447` findings H1–H4
as executable gates. Contract `codex-d2-step4b-2026-08-09-v1`; companion to the closed SB
(REV 2, `d1a58b3`) and PV (`67aec6c`) bars. H1–H2 gate the OUTWARD FETCH; H3–H4 gate batch
acceptance. HERMETIC: no network, no git writes — providers/scoring/clock all injected;
`seismic_data`/`fault_correlation`/`obspy` are stubbed in sys.modules (executor logic under
test here, not DSP — the DSP is locked by the accepted 3950a2c suites). The owner-consent
hold is separate and untouched by this bar.

CONTRACT SEAMS (grassmann implements UNEDITED; additive to the landed executor `5766093`)
=========================================================================================
* H1 — injectable, honestly-read clock:
  `d2_step4b_campaign_run.acquire(plan, ledger, root, *, providers, receipt, clock=None)`;
  `clock` = zero-arg callable returning aware-UTC datetime (default: live now()).
    - `campaign_started_utc` = ONE clock() read at entry (post-gate), recorded in
      `batch_manifest.campaign_started_utc`; must satisfy
      `campaign_started_utc.date() == plan["activation_reference_day"]` — the STARTED day,
      not creation/issue day, carries the activation-day requirement;
    - every attempt's `attempted_utc` = a FRESH clock() read at that attempt;
    - every capsule's `issued_utc` = a FRESH clock() read at its mint;
    - `created_utc` = a FRESH clock() read after final artifact assembly;
    - ordering: campaign_started < every attempted < every issued < created under a strictly
      advancing clock (no reuse of the launch instant anywhere);
    - `valid_through = activation_reference + 7d` unchanged (inclusive stored date);
      `expiry_utc` = 00:00:00Z on the day AFTER valid_through (codex 0500 #2). A capsule
      whose mint-time clock() >= expiry_utc makes the region row
      `BLOCKED_CANDIDATE_WINDOW_EXPIRED` with reason `CANDIDATE_WINDOW_EXPIRED`, NO capsule
      file, NO registry_candidate entry (no post-outcome extension). Boundary: a mint at
      valid_through 23:59:59.999999Z is LIVE; the following day 00:00:00.000000Z BLOCKS.
* H2 — the acquired plan IS the staged plan:
  `run_campaign` (producer), after receipt verification and BEFORE any provider import or
  `_acquire` call, reopens `root/campaign_plan.json`, parses it, and
    - refuses (SystemExit) if a caller-supplied plan differs from the staged plan —
      zero `_acquire` calls, zero provider I/O;
    - passes the REOPENED object (not the caller object) to `_acquire`; `plan=None` is
      allowed and uses the reopened object exclusively.
* H3 — per-object provenance from the object's OWN bytes:
  each `input_manifest.objects` row's `native_rate_hz`, `npts`, `fragment_count`, and
  `support_sha256` are derived by REOPENING that staged object and parsing IT alone.
  The reparse goes through the injected providers object: `providers.parse_staged(
  staged_path) -> stream` (production providers implement it via obspy.read on the staged
  bytes; this bar's fake decodes its own bodies) — the executor itself stays obspy-free:
    - `native_rate_hz` = first fragment's rate; `npts` = sum of its own fragment lengths;
      `fragment_count` = its own fragment count;
    - `support_sha256` = sha256 of the canonical JSON (sort_keys, ',:', +LF) of the object's
      own timestamp-support frame list [[fragment_start_utc_iso, npts_i, rate_i], ...]
      in time order (timestamp-derived, no DSP);
    - the combined session stream remains what scoring consumes; daily/segment records keep
      the combined support. Two SCEDC day-volumes with different contents must produce two
      DIFFERENT provenance rows.
* H4 — operation receipts bind real stage outputs + declared losses:
  for every ADMITTED (arm, carrier, day) the three ledger rows carry
  `output_sha256 = sha256(canonical(receipt))` with receipts recomputable from the committed
  artifacts (canonical = json sort_keys/',:'/allow_nan=False + trailing LF, the executor's
  `_canon`):
    - r1 (native_bandpass_hilbert_envelope_resample):
      {"operation", "arm", "carrier_key", "day",
       "input_sha256s": sorted(daily.input_object_sha256s),
       "native_rates": {sha256: native_rate_hz from input_manifest},
       "station_coverages": {segment: daily.segment_support[segment]["station_coverages"]}}
    - r2 (station_utc_intersection_median):
      {"operation", "arm", "carrier_key", "day",
       "segment_support": daily.segment_support,
       "common_support_count": daily.common_support_count}
    - r3 (segment_correlation_eigenspectrum):
      {"operation", "arm", "carrier_key", "day",
       "correlation_matrix": daily.correlation_matrix,
       "correlation_matrix_order": daily.correlation_matrix_order,
       "ordered_eigenvalues": daily.ordered_eigenvalues, "ratio": daily.ratio,
       "participation_ratio": daily.participation_ratio,
       "derivation": daily.lambda2_lambda1_derivation}
  and the declared-loss fields are EXACTLY (sorted):
    - r1: information_lost=["absolute_phase","native_frequency_detail"],
          side_channel_retained=["native_rate_hz","raw_object_sha256s","station_support_masks"],
          claim_effects=[]
    - r2: information_lost=["station_specific_amplitude_structure"],
          side_channel_retained=["aggregate_mask_sha256","station_coverages","station_identities"],
          claim_effects=[]
    - r3: information_lost=["full_operator_structure"],
          side_channel_retained=["correlation_matrix","ordered_eigenvalues","participation_ratio"],
          claim_effects=["NO_EIGENVECTOR_CLAIM","SCALAR_SUMMARY_ONLY"]

RED AS AUTHORED (REV 2, codex 0500 repairs #1+#2 applied; #3 landed as PV bar REV 2): exactly
['H1-GATE (clock seam)', 'H1b (expiry; explicitly marked blocked-by-missing-seam)',
 'H2a (plan-mismatch refuse)', 'H2b (reopened-plan authority)',
 'H3 (per-object provenance)', 'H4a (receipt binding)', 'H4b (declared losses)'] — seven.

REV 3 (2026-08-09, codex 0824 folded into the frozen suite as GREEN locks — the typed-fatal
fix landed at 1305f5e before this revision; import-hermetic here via the sys.modules stubs,
unlike the standalone producer KAT which needs obspy-importable scoring modules):
* H5 — host resource exhaustion never mints or relabels:
  `ScoringResourceUnavailable` (RuntimeError subclass) exists; `_station_series` retries
  ONCE after gc on MemoryError (exactly two scorer calls) and returns the envelope on
  success; a SECOND MemoryError raises the typed fatal (never None — resource failure is
  not data unavailability); `SD.DataUnavailable` still maps to None; a persistent OOM
  anywhere in scoring aborts `acquire` and leaves NO batch_manifest.json.
"""
import hashlib
import inspect
import json
import os
import sys
import tempfile
import types
from datetime import date, datetime, timedelta, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

FAILS = []


def check(desc, ok, detail=""):
    print(f"    [{'PASS' if ok else 'FAIL'}] {desc}" + (f" - {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(desc)


def _canon(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n").encode("utf-8")


def _sha(b):
    return hashlib.sha256(b).hexdigest()


# ---- stubs -------------------------------------------------------------------
class _ES:
    def __init__(self, coverage, mask):
        self.coverage = coverage
        self.valid_mask = mask


def _install_stubs():
    sd = types.ModuleType("seismic_data")

    class DataUnavailable(Exception):
        pass

    def compute_band_envelope_supported(frags, *, session_start_utc, session_seconds,
                                        source_id):
        return _ES(0.9, np.ones(86400, dtype=bool))

    sd.DataUnavailable = DataUnavailable
    sd.compute_band_envelope_supported = compute_band_envelope_supported
    sys.modules["seismic_data"] = sd

    fc = types.ModuleType("fault_correlation")

    def aggregate_segment_supported(series):
        if not series:
            return None
        mask = np.ones(86400, dtype=bool)
        for s in series:
            mask &= np.asarray(s.valid_mask, dtype=bool)
        return _ES(min(s.coverage for s in series), mask)

    def compute_correlation_matrix_supported(seg_series, seg_names):
        if len(seg_series) < 2:
            return None, list(seg_names), ["INSUFFICIENT_SEGMENTS"]
        return [[1.0, 0.3], [0.3, 1.0]], list(seg_names), []

    fc.aggregate_segment_supported = aggregate_segment_supported
    fc.compute_correlation_matrix_supported = compute_correlation_matrix_supported
    sys.modules["fault_correlation"] = fc

    for name in ("obspy", "scipy"):
        try:
            __import__(name)
        except ImportError:
            m = types.ModuleType(name)
            m.__version__ = "0.0-stub"
            sys.modules[name] = m


# ---- trace/body encoding: body <-> frames (the bar's ground truth) -----------
class _Stats:
    def __init__(self, start, rate):
        self.starttime = types.SimpleNamespace(datetime=start.replace(tzinfo=None))
        self.sampling_rate = rate


class _Trace:
    def __init__(self, start, npts, rate):
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


def _support_sha_of(body):
    return _sha(_canon(_frames_of(body)))


def _body(frames):
    return ("TRACES:" + json.dumps(frames)).encode("utf-8")


# ---- fake providers -----------------------------------------------------------
class _FakeProviders:
    """Post-P2-shaped provider object; stages files itself, serves scripted bodies."""

    class ProviderUnavailable(Exception):
        pass

    def __init__(self, special_day=None):
        self.special_day = special_day     # (carrier uses SCEDC) day with TWO day-volumes

    def koeri_available(self, net, stas, chas, start, end):
        return {f"{net}.{s}..{c}" for s in stas for c in chas}

    def scedc_available(self, net, stas, chas, start, end):
        return {f"{net}.{s}..{c}" for s in stas for c in chas}

    def parse_staged(self, staged_path):
        with open(staged_path, "rb") as f:
            return _traces_of(f.read())

    def _mk(self, nslc, day, bodies, stage_dir):
        objs, traces = [], []
        for i, body in enumerate(bodies):
            name = f"{nslc}_{day}_{i}_{_sha(body)[:12]}.ms"
            path = os.path.join(stage_dir, name)
            with open(path, "wb") as f:
                f.write(body)
            objs.append({"source": f"fake://{nslc}/{day}/{i}", "staged_path": path,
                         "size_bytes": len(body), "sha256": _sha(body)})
            traces.extend(_traces_of(body))
        return {"stream": traces, "raw_objects": objs}

    def fetch(self, provider, nslc, start, end, *, stage_dir, **kw):
        day = end.date().isoformat()          # the SCORED day (end-anchored session)
        s = start.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
        s2 = (start + timedelta(seconds=600)).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
        s3 = (start + timedelta(seconds=1200)).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
        if provider == "SCEDC" and self.special_day == day:
            return self._mk(nslc, day, [_body([[s, 10, 40.0]]),
                                        _body([[s2, 5, 40.0], [s3, 3, 40.0]])], stage_dir)
        return self._mk(nslc, day, [_body([[s, 8, 40.0]])], stage_dir)


# ---- fixtures ------------------------------------------------------------------
def _mk_fixtures(P, root):
    incident_days = P.schedule_days("2026-07-29")
    activation_days = P.schedule_days("2026-08-09")
    union = sorted(set(incident_days) | set(activation_days))
    registry = {}
    for carrier, net in (("istanbul_marmara", "KO"), ("turkey_kahramanmaras", "KO"),
                         ("socal_coachella", "CI")):
        rows = []
        for seg_i, seg in enumerate(("seg_a", "seg_b")):
            for st_i in range(2):
                sta = f"S{seg_i}{st_i}"
                rows.append({"segment_name": seg, "station_id": f"{net}.{sta}",
                             "ordered_nslc_candidates": [f"{net}.{sta}..BHZ"]})
        registry[carrier] = rows
    plan = {
        "schema": "geospec-d2-step4b-campaign-plan-v1",
        "contract_id": "codex-d2-step4b-2026-08-09-v1",
        "registered_utc": "2026-08-09T02:20:00.000000Z",
        "activation_reference_day": "2026-08-09", "incident_reference_day": "2026-07-29",
        "carriers": ["istanbul_marmara", "socal_coachella", "turkey_kahramanmaras"],
        "providers": {"istanbul_marmara": {"provider": "KOERI",
                                           "endpoint": "eida.koeri.boun.edu.tr"},
                      "socal_coachella": {"provider": "SCEDC", "endpoint": "s3://scedc-pds"},
                      "turkey_kahramanmaras": {"provider": "KOERI",
                                               "endpoint": "eida.koeri.boun.edu.tr"}},
        "station_registry": registry, "incident_days": incident_days,
        "activation_days": activation_days, "scheduled_days": union,
        "acquisition_order": ["KOERI", "SCEDC"], "free_sources_only": True,
        "outcomes_inspected_before_schedule": False}
    rows = []
    for carrier in plan["carriers"]:
        for day in union:
            end = f"{day}T07:00:13.094647Z"
            start_dt = datetime.strptime(end, "%Y-%m-%dT%H:%M:%S.%fZ") - timedelta(
                seconds=86400)
            rows.append({"carrier_key": carrier, "scored_day": day, "status": "REGISTERED",
                         "publication_commit": 40 * "a", "publication_repo_path":
                         "docs/ensemble_latest.json",
                         "publication_record_artifact": "publication_records/x.json",
                         "record_git_blob": 40 * "b", "record_sha256": 64 * "c",
                         "request_start_utc":
                         start_dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z",
                         "request_end_utc": end, "reason_codes": []})
    ledger = {"schema": "geospec-d2-published-phase-ledger-v1", "rows": rows}
    with open(os.path.join(root, "campaign_plan.json"), "wb") as f:
        f.write(_canon(plan))
    with open(os.path.join(root, "published_phase_ledger.json"), "wb") as f:
        f.write(_canon(ledger))
    return plan, ledger


RECEIPT = {"status": "VERIFIED_DIRECT", "in_session_timestamp_utc": "2026-08-09T02:04:49Z",
           "owner_quote_sha256":
           "0658bdf0b498b551c433bb3f932a87a9c06e28929703c22d9468507b1fc7d3f8"}


def _mk_clock(start, step_seconds):
    state = {"t": start - timedelta(seconds=step_seconds)}

    def clock():
        state["t"] = state["t"] + timedelta(seconds=step_seconds)
        return state["t"]
    return clock


def _read_json(root, rel):
    with open(os.path.join(root, rel), "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def _read_jsonl(root, rel):
    out = []
    with open(os.path.join(root, rel), "rb") as f:
        for line in f.read().decode("utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def _parse_utc(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)


def main():
    _install_stubs()
    try:
        import d2_step4b_producer as P
        import d2_step4b_campaign_run as RUN
    except ImportError as exc:
        check("H0 producer + executor import", False, str(exc))
        return

    special_day = P.schedule_days("2026-08-09")[0]        # first activation day (SCEDC 2-vol)

    # ================= H1: injectable honest clock ============================
    sig = inspect.signature(RUN.acquire).parameters
    if "clock" not in sig:
        check("H1-GATE acquire exposes the injectable clock seam "
              "(clock=zero-arg aware-UTC callable)", False,
              "AWAITING clock seam -- red-first as authored (codex 0447 H1)")
        check("H1b mint on/after expiry_utc -> BLOCKED_CANDIDATE_WINDOW_EXPIRED "
              "(boundary: valid_through 23:59:59.999999Z live, next-day 00:00:00Z blocks)",
              False, "BLOCKED by the missing clock seam (codex 0500 #1 explicit marker)")
    else:
        with tempfile.TemporaryDirectory() as td:
            plan, ledger = _mk_fixtures(P, td)
            clock = _mk_clock(datetime(2026, 8, 9, 23, 50, 0, tzinfo=timezone.utc), 1)
            res = RUN.acquire(plan, ledger, td, providers=_FakeProviders(special_day),
                              receipt=RECEIPT, clock=clock)
            bm = _read_json(td, "batch_manifest.json")
            attempts = _read_jsonl(td, "acquisition_attempts.jsonl")
            adm = _read_json(td, "admission_results.json")["regions"]
            started = _parse_utc(bm["campaign_started_utc"])
            attempted = [_parse_utc(a["attempted_utc"]) for a in attempts]
            issued = []
            for r in adm:
                if r.get("capsule_path"):
                    cap = _read_json(td, r["capsule_path"])
                    issued.append(_parse_utc(cap["issued_utc"]))
            created = _parse_utc(bm["created_utc"])
            check("H1a fresh clock reads: started < every attempted < every issued < "
                  "created (all distinct under an advancing clock); started.date() == "
                  "activation day; honest post-midnight completion allowed",
                  started.date() == date(2026, 8, 9)
                  and attempted and issued
                  and started < min(attempted) and max(attempted) < min(issued)
                  and max(issued) < created
                  and len({started, created} | set(issued)) == 2 + len(issued)
                  and created.date() > date(2026, 8, 9),
                  f"started={started} n_att={len(attempted)} issued={issued} created={created}")
        def _expiry_run(clock_fn):
            with tempfile.TemporaryDirectory() as td:
                plan, ledger = _mk_fixtures(P, td)
                RUN.acquire(plan, ledger, td, providers=_FakeProviders(special_day),
                            receipt=RECEIPT, clock=clock_fn)
                adm = _read_json(td, "admission_results.json")["regions"]
                reg = _read_json(td, "registry_candidate.json")
                caps = []
                for r in adm:
                    if r.get("capsule_path"):
                        caps.append(_read_json(td, r["capsule_path"]))
                return adm, reg, caps

        # interior expired case (advancing +3h clock; started on activation day)
        adm_i, reg_i, caps_i = _expiry_run(
            _mk_clock(datetime(2026, 8, 9, 0, 10, 0, tzinfo=timezone.utc), 10800))
        exp_i = [r for r in adm_i if r["status"] == "BLOCKED_CANDIDATE_WINDOW_EXPIRED"]
        interior_ok = (len(exp_i) >= 1 and reg_i == {} and caps_i == []
                       and all(r.get("capsule_path") is None for r in adm_i)
                       and all("CANDIDATE_WINDOW_EXPIRED" in r["reason_codes"]
                               for r in exp_i))
        # boundary LIVE: valid_through (2026-08-16) 23:59:59.999999Z still mints
        live_t = datetime(2026, 8, 16, 23, 59, 59, 999999, tzinfo=timezone.utc)
        adm_l, reg_l, caps_l = _expiry_run(lambda: live_t)
        live_ok = (len(caps_l) >= 1 and len(reg_l) >= 1
                   and all(c["issued_utc"] == "2026-08-16T23:59:59.999999Z"
                           for c in caps_l)
                   and any(r["status"] == "ADMITTED_CANDIDATE" for r in adm_l))
        # boundary BLOCKED: 2026-08-17T00:00:00.000000Z (== expiry_utc) blocks
        adm_b, reg_b, caps_b = _expiry_run(
            lambda: datetime(2026, 8, 17, 0, 0, 0, 0, tzinfo=timezone.utc))
        blocked_ok = (reg_b == {} and caps_b == []
                      and any(r["status"] == "BLOCKED_CANDIDATE_WINDOW_EXPIRED"
                              for r in adm_b))
        check("H1b mint on/after expiry_utc -> BLOCKED_CANDIDATE_WINDOW_EXPIRED "
              "(boundary: valid_through 23:59:59.999999Z live, next-day 00:00:00Z blocks)",
              interior_ok and live_ok and blocked_ok,
              f"interior={interior_ok} live={live_ok} blocked={blocked_ok} "
              f"statuses_i={[r['status'] for r in adm_i]}")

    # ================= H2: staged-plan binding ================================
    with tempfile.TemporaryDirectory() as td:
        plan, ledger = _mk_fixtures(P, td)
        calls = []
        real_acquire = P._acquire
        try:
            P._acquire = lambda pl, lg, rt: calls.append(pl) or {"status": "SPY"}
            plan_b = json.loads(json.dumps(plan))
            plan_b["activation_reference_day"] = "2026-08-10"
            refused = False
            try:
                P.run_campaign(plan=plan_b, launch_authorization=RECEIPT, root=td)
            except SystemExit:
                refused = True
            except BaseException:
                pass
            check("H2a caller plan != staged campaign_plan.json -> SystemExit BEFORE "
                  "_acquire (zero acquisition calls)",
                  refused and calls == [], f"refused={refused} calls={len(calls)}")
            calls.clear()
            try:
                P.run_campaign(plan=None, launch_authorization=RECEIPT, root=td)
            except BaseException:
                pass
            staged = _read_json(td, "campaign_plan.json")
            check("H2b plan=None -> _acquire receives the REOPENED staged plan object "
                  "(reopened authority, exact equality)",
                  len(calls) == 1 and calls[0] == staged,
                  f"calls={len(calls)} equal={bool(calls and calls[0] == staged)}")
        finally:
            P._acquire = real_acquire

    # ================= H3 + H4: run once, inspect artifacts ===================
    with tempfile.TemporaryDirectory() as td:
        plan, ledger = _mk_fixtures(P, td)
        kw = {"providers": _FakeProviders(special_day), "receipt": RECEIPT}
        if "clock" in sig:
            kw["clock"] = _mk_clock(datetime(2026, 8, 9, 12, 0, 0, tzinfo=timezone.utc), 1)
        RUN.acquire(plan, ledger, td, **kw)
        manifest = _read_json(td, "input_manifest.json")["objects"]
        daily = _read_jsonl(td, "calibration_daily.jsonl")
        ops = _read_jsonl(td, "operation_ledger.jsonl")

        # ---- H3: the two SCEDC day-volumes carry their OWN provenance --------
        srow = [r for r in manifest if r["carrier_key"] == "socal_coachella"
                and r["scored_day"] == special_day]
        by_sha = {r["sha256"]: r for r in srow}
        start_dt = _parse_utc(f"{special_day}T07:00:13.094647Z") - timedelta(seconds=86400)
        s = start_dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
        s2 = (start_dt + timedelta(seconds=600)).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
        s3 = (start_dt + timedelta(seconds=1200)).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
        vol1 = _body([[s, 10, 40.0]])
        vol2 = _body([[s2, 5, 40.0], [s3, 3, 40.0]])
        r1 = by_sha.get(_sha(vol1))
        r2 = by_sha.get(_sha(vol2))
        ok_h3 = (r1 is not None and r2 is not None
                 and r1["npts"] == 10 and r1["fragment_count"] == 1
                 and r2["npts"] == 8 and r2["fragment_count"] == 2
                 and r1["support_sha256"] == _support_sha_of(vol1)
                 and r2["support_sha256"] == _support_sha_of(vol2)
                 and r1["native_rate_hz"] == 40.0 and r2["native_rate_hz"] == 40.0)
        check("H3 per-object provenance derives from each staged object's OWN bytes "
              "(two SCEDC day-volumes -> distinct npts/fragment_count/timestamp-support "
              "digests recomputed by the bar)",
              ok_h3,
              f"r1={{npts:{r1 and r1['npts']},fc:{r1 and r1['fragment_count']}}} "
              f"r2={{npts:{r2 and r2['npts']},fc:{r2 and r2['fragment_count']}}}")

        # ---- H4: op receipts bind real stage outputs -------------------------
        rates = {r["sha256"]: r["native_rate_hz"] for r in manifest}
        daily_ix = {(r["arm"], r["carrier_key"], r["day"]): r for r in daily}
        LOSS = {
            RUN.OPS[0]: (["absolute_phase", "native_frequency_detail"],
                         ["native_rate_hz", "raw_object_sha256s", "station_support_masks"],
                         []),
            RUN.OPS[1]: (["station_specific_amplitude_structure"],
                         ["aggregate_mask_sha256", "station_coverages", "station_identities"],
                         []),
            RUN.OPS[2]: (["full_operator_structure"],
                         ["correlation_matrix", "ordered_eigenvalues", "participation_ratio"],
                         ["NO_EIGENVECTOR_CLAIM", "SCALAR_SUMMARY_ONLY"]),
        }
        bind_ok, loss_ok, n_checked = True, True, 0
        for op in ops:
            row = daily_ix[(op["arm"], op["carrier_key"], op["day"])]
            name = op["operation_name"]
            ids = {"operation": name, "arm": op["arm"], "carrier_key": op["carrier_key"],
                   "day": op["day"]}
            if name == RUN.OPS[0]:
                receipt = dict(ids, input_sha256s=sorted(row["input_object_sha256s"]),
                               native_rates={h: rates[h]
                                             for h in row["input_object_sha256s"]},
                               station_coverages={seg: sup["station_coverages"]
                                                  for seg, sup in
                                                  row["segment_support"].items()})
            elif name == RUN.OPS[1]:
                receipt = dict(ids, segment_support=row["segment_support"],
                               common_support_count=row["common_support_count"])
            else:
                receipt = dict(ids, correlation_matrix=row["correlation_matrix"],
                               correlation_matrix_order=row["correlation_matrix_order"],
                               ordered_eigenvalues=row["ordered_eigenvalues"],
                               ratio=row["ratio"],
                               participation_ratio=row["participation_ratio"],
                               derivation=row["lambda2_lambda1_derivation"])
            if op["output_sha256"] != _sha(_canon(receipt)):
                bind_ok = False
            lost, side, claims = LOSS[name]
            if (sorted(op.get("information_lost", [])) != lost
                    or sorted(op.get("side_channel_retained", [])) != side
                    or sorted(op.get("claim_effects", [])) != claims):
                loss_ok = False
            n_checked += 1
        check("H4a every operation output_sha256 == sha256(canonical stage receipt) "
              "recomputed by the bar from calibration_daily + input_manifest",
              n_checked > 0 and bind_ok, f"checked={n_checked}")
        check("H4b declared losses/side-channels/claim-effects are populated EXACTLY "
              "(scalarization declares NO_EIGENVECTOR_CLAIM + SCALAR_SUMMARY_ONLY)",
              n_checked > 0 and loss_ok, f"checked={n_checked}")

    # ================= H5: resource exhaustion is typed-fatal (codex 0824) ====
    if not hasattr(RUN, "ScoringResourceUnavailable"):
        check("H5-GATE typed ScoringResourceUnavailable seam present", False,
              "AWAITING typed-fatal seam (codex 0824)")
    else:
        class _ScriptedSD:
            class DataUnavailable(Exception):
                pass

            def __init__(self, n_oom):
                self.calls = 0
                self.n_oom = n_oom

            def compute_band_envelope_supported(self, frags, *, session_start_utc,
                                                session_seconds, source_id):
                self.calls += 1
                if self.calls <= self.n_oom:
                    raise MemoryError("injected commit-limit OOM")
                return _ES(0.9, np.ones(86400, dtype=bool))

        sd1 = _ScriptedSD(1)
        frags_stream = [_Trace(datetime(2026, 5, 1, 7, 0, 0, tzinfo=timezone.utc), 8, 40.0)]
        es1 = RUN._station_series(sd1, frags_stream, "KO.SAUV..HHZ",
                                  datetime(2026, 5, 1, 7, 0, 0, tzinfo=timezone.utc))
        check("H5a TRANSIENT OOM: retry-after-gc returns the envelope after EXACTLY two "
              "scorer calls", es1 is not None and sd1.calls == 2, f"calls={sd1.calls}")

        sd2 = _ScriptedSD(2)
        typed = False
        try:
            RUN._station_series(sd2, frags_stream, "KO.SAUV..HHZ",
                                datetime(2026, 5, 1, 7, 0, 0, tzinfo=timezone.utc))
        except RUN.ScoringResourceUnavailable:
            typed = True
        except Exception:
            pass
        check("H5b PERSISTENT OOM raises typed ScoringResourceUnavailable (never None -- "
              "resource failure is not data unavailability)",
              typed and sd2.calls == 2, f"typed={typed} calls={sd2.calls}")

        sd3 = _ScriptedSD(0)

        def _du(*a, **k):
            raise _ScriptedSD.DataUnavailable("no data")
        sd3.compute_band_envelope_supported = _du
        es3 = RUN._station_series(sd3, frags_stream, "KO.SAUV..HHZ",
                                  datetime(2026, 5, 1, 7, 0, 0, tzinfo=timezone.utc))
        check("H5c DataUnavailable still maps to None (data vs resource cleanly separated)",
              es3 is None)

        stub_sd = sys.modules["seismic_data"]
        real_compute = stub_sd.compute_band_envelope_supported

        def _persistent_oom(frags, *, session_start_utc, session_seconds, source_id):
            raise MemoryError("injected persistent commit-limit OOM")
        try:
            stub_sd.compute_band_envelope_supported = _persistent_oom
            with tempfile.TemporaryDirectory() as td:
                plan5, ledger5 = _mk_fixtures(P, td)
                kw5 = {"providers": _FakeProviders(None), "receipt": RECEIPT}
                if "clock" in sig:
                    kw5["clock"] = _mk_clock(
                        datetime(2026, 8, 9, 12, 0, 0, tzinfo=timezone.utc), 1)
                raised5 = False
                try:
                    RUN.acquire(plan5, ledger5, td, **kw5)
                except RUN.ScoringResourceUnavailable:
                    raised5 = True
                except Exception:
                    pass
                no_batch = not os.path.isfile(os.path.join(td, "batch_manifest.json"))
                check("H5d FULL PATH: a persistent scoring OOM aborts acquire with the typed "
                      "fatal and mints NO batch_manifest.json",
                      raised5 and no_batch, f"raised={raised5} no_batch={no_batch}")
        finally:
            stub_sd.compute_band_envelope_supported = real_compute


main()
print()
if FAILS:
    print(f"D2 STEP-4B EXECUTOR RED-KAT FAILURES: {FAILS}")
    sys.exit(1)
print("ALL D2 STEP-4B EXECUTOR RED-KATs PASS (honest injected clock + staged-plan binding + "
      "per-object provenance + output-bound operation receipts)")
