#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D2 STEP-4B RESUMABILITY red-KATs (cayley, 2026-08-09) — freezes codex extension
`codex-d2-step4b-resume-2026-08-09-v1` (1303) + the 1309 pre-I/O causal-order clarification
as five executable families RK1–RK5. Parent contract `codex-d2-step4b-2026-08-09-v1`; pinned
DSP `3950a2c` unchanged. HERMETIC: providers/scoring/clock injected; process death is
simulated as an injected exception propagating out of `run_campaign` (equivalent for lock
release + state durability — the kernel writer lock and fsync'd state behave identically).
Nothing here authorizes a reboot, provider I/O, re-fire, mint, lift, or claim; the actual
resume still waits on asylum's reboot timing + fresh direct go (cayley 1257 / codex R0.4).

CONTRACT SEAMS (grassmann implements UNEDITED; additive to `0f3df30`)
=====================================================================
* NEW MODULE `monitoring/src/d2_step4b_resume.py`:
    - ResumeIntegrityError(RuntimeError) — integrity violations NEVER degrade to
      data-unavailability and NEVER mint a manifest.
    - campaign_id_of(plan_bytes: bytes, ledger_bytes: bytes) -> 64-hex ==
      sha256(b"geospec-d2-step4b-resume-v1\\0" + sha256(plan_bytes).digest()
             + sha256(ledger_bytes).digest()).hexdigest()   [R0.1; binary digests]
    - stage_raw_atomic(raw: bytes, dest_dir: str) -> {"relative_path", "sha256",
      "size_bytes"}: sibling unique temp + flush/fsync + atomic install to
      raw_objects/<sha256>.ms; EXISTING destination is verified byte-for-byte and REUSED
      WITHOUT opening for write; mismatch -> ResumeIntegrityError.               [R1.2]
    - append_event(root, event: dict) -> committed event carrying prev_event_sha256 +
      event_sha256 (sha256 of the canonical event sans its own hash, chained); atomic
      replace + fsync of resume_state.json.                                       [R1.3]
    - load_resume_state(root) -> validated state (full chain re-derivation; ANY tamper,
      truncation, or unresolvable/escaping referenced path -> ResumeIntegrityError).
* PRODUCER `run_campaign(plan, launch_authorization, root, ..., providers=None)` — same
  single gated entry; `providers=None` lazily imports the real d2_step4b_providers module
  (production default — the existing SB/PV/executor bars keep passing unchanged); an
  injected providers object replaces it wholesale (this bar's harness seam). Resume is
  automatic from root state:
    - COMPLETED root (batch_manifest.json present): verify and return the completed result;
      NEVER mutate/re-finalize (byte-identical manifest after the call).          [R0.3]
    - identity drift (campaign_plan.json or published_phase_ledger.json bytes differing
      from the hashes bound in resume_state) -> refusal BEFORE provider import/I/O. [R0.2]
    - each invocation re-verifies its own VERIFIED_DIRECT receipt (invalid -> SystemExit,
      zero provider calls, even on a resumable root).                             [R0.4]
    - under the writer lock, PROCESS_STARTED is committed+reopened BEFORE any provider
      import/availability/fetch (spy-proven); UNIT_ATTEMPT_STARTED (attempt_id, per-unit
      ordinal, IN_PROGRESS) committed before that unit's first provider I/O; the terminal
      event completes that exact attempt_id.                                      [1309]
    - crash-left IN_PROGRESS attempts are retained as INDETERMINATE_PROCESS_DEATH on
      resume — never rewritten as UNAVAILABLE/ERROR/FETCHED; the re-run gets a NEW attempt
      ordinal; history is never erased.                                           [1309.3]
    - terminal units (FETCHED/UNAVAILABLE/ERROR) are never availability-probed or fetched
      again; FETCHED reuse requires reopen + exact size/sha + H3 reparse
      (ResumeIntegrityError on any mismatch); a digest-named file with NO committed event
      is an ORPHAN — never reused, never in input_manifest/closure.               [R2]
    - `campaign_process_ledger.jsonl` (core artifact): one row per PROCESS_STARTED ever
      committed (crashed segments included; process_ended_utc null + observed_dead_utc on
      ungraceful death); input objects bind acquired_process_id / verified_process_id /
      reuse_disposition (FETCHED_NEW | REUSED_VERIFIED); attempts rows bind attempt_ids /
      terminal_attempt_id / indeterminate_attempt_count.                          [R3]
    - batch_manifest.run_id == campaign_id; campaign_started_utc == FIRST process start,
      never reset by a resume; activation day / valid_through / 00:00Z expiry derive from
      the ORIGINAL activation reference (a late resume may honestly expire; never remint).
    - R4 LEGACY (the preserved 218): a digest-named file present WITHOUT a committed event
      is UNATTESTED_LEGACY. Under a fresh authorized process the frozen selection + exact
      provider request run normally; if the served response's size+sha equal the legacy
      file, commit the new exact receipt with
      reuse_disposition=LEGACY_REUSED_AFTER_REFETCH_ATTESTATION; if it differs, the served
      bytes win and the legacy file is quarantined at the SIBLING `root + ".legacy_quarantine/"`
      (outside closure). Without attestation a legacy byte can never reach input_manifest,
      a daily row, a threshold, or a candidate.
    - R5 closure: resume_state.json + campaign_process_ledger.jsonl are manifest-bound;
      no temp/orphan/lock/partial file inside the final root; all final artifacts rebuilt
      deterministically from reopened state; batch_manifest.json written LAST (atomic).

RED AS AUTHORED vs GeoSpec `0f3df30`: exactly the six family gates
['RK0-GATE', 'RK1-BLOCKED', 'RK2-BLOCKED', 'RK3-BLOCKED', 'RK4-BLOCKED', 'RK5-BLOCKED']
(the resume module does not exist yet; every family activates behind RK0).
"""
import hashlib
import json
import os
import sys
import tempfile
import types
from datetime import datetime, timedelta, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

FAILS = []


def check(desc, ok, detail=""):
    print(f"    [{'PASS' if ok else 'FAIL'}] {desc}" + (f" - {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(desc)


def raises(fn, exc=Exception):
    try:
        fn()
        return False
    except exc:
        return True


def _canon(v):
    return (json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n").encode()


def _sha(b):
    return hashlib.sha256(b).hexdigest()


# ---- stubs (same shapes as the executor bar) ---------------------------------
class _ES:
    def __init__(self, coverage, mask):
        self.coverage = coverage
        self.valid_mask = mask


def _install_stubs():
    sd = types.ModuleType("seismic_data")

    class DataUnavailable(Exception):
        pass

    sd.DataUnavailable = DataUnavailable
    sd.compute_band_envelope_supported = (
        lambda frags, *, session_start_utc, session_seconds, source_id:
        _ES(0.9, np.ones(86400, dtype=bool)))
    sys.modules["seismic_data"] = sd
    fc = types.ModuleType("fault_correlation")

    def agg(series):
        if not series:
            return None
        mask = np.ones(86400, dtype=bool)
        for s in series:
            mask &= np.asarray(s.valid_mask, dtype=bool)
        return _ES(min(s.coverage for s in series), mask)

    fc.aggregate_segment_supported = agg
    fc.station_eligible = lambda es: es is not None and es.coverage >= 0.5
    # ^ fixture-compat amendment (fix-A scorer repair calls FC.station_eligible;
    #   stub series carry 0.9 coverage -> all eligible, behavior unchanged)
    fc.compute_correlation_matrix_supported = (
        lambda seg_series, seg_names:
        ((None, list(seg_names), ["INSUFFICIENT_SEGMENTS"]) if len(seg_series) < 2
         else ([[1.0, 0.3], [0.3, 1.0]], list(seg_names), [])))
    sys.modules["fault_correlation"] = fc
    for name in ("obspy", "scipy"):
        try:
            __import__(name)
        except ImportError:
            m = types.ModuleType(name)
            m.__version__ = "0.0-stub"
            sys.modules[name] = m


class _Stats:
    def __init__(self, start, rate):
        self.starttime = types.SimpleNamespace(datetime=start.replace(tzinfo=None))
        self.sampling_rate = rate


class _Trace:
    def __init__(self, start, npts, rate):
        self.stats = _Stats(start, rate)
        self.data = np.zeros(int(npts))


def _frames_of(body):
    # bodies carry a "|<nslc>" uniqueness suffix after the JSON frame list
    return json.loads(body.decode().split("TRACES:", 1)[1].split("|", 1)[0])


def _traces_of(body):
    return [_Trace(datetime.fromisoformat(i.replace("Z", "+00:00")), n, r)
            for i, n, r in _frames_of(body)]


def _body_for(nslc, end):
    s = (end - timedelta(seconds=86400)).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    return ("TRACES:" + json.dumps([[s, 8, 40.0]]) + "|" + nslc).encode()


class _Die(BaseException):
    """Injected process death. BaseException-derived so ordinary fetch-error handling
    (except Exception -> ERROR outcome) can NEVER absorb it — it propagates out of
    run_campaign exactly like a real process death (lock released, state left as-is)."""


class _Prov:
    """Deterministic fake provider with staged files, call log, and scripted death."""

    class ProviderUnavailable(Exception):
        pass

    def __init__(self, die_after_fetches=None, unavailable_nslc=(), serve_override=None):
        self.fetches = []
        self.probes = []
        self.die_after = die_after_fetches
        self.unavailable = set(unavailable_nslc)
        self.serve_override = serve_override or {}

    def koeri_available(self, net, stas, chas, s, e):
        self.probes.append(("KOERI", tuple(stas), s))
        return {f"{net}.{st}..{c}" for st in stas for c in chas}

    def scedc_available(self, net, stas, chas, s, e):
        self.probes.append(("SCEDC", tuple(stas), s))
        return {f"{net}.{st}..{c}" for st in stas for c in chas}

    def parse_staged(self, path):
        with open(path, "rb") as f:
            return _traces_of(f.read())

    def fetch(self, provider, nslc, s, e, *, stage_dir, **kw):
        self.fetches.append((nslc, e.date().isoformat()))
        if self.die_after is not None and len(self.fetches) > self.die_after:
            raise _Die(f"injected death at fetch #{len(self.fetches)}")
        if nslc in self.unavailable:
            raise self.ProviderUnavailable(f"scripted unavailable: {nslc}")
        body = self.serve_override.get(nslc) or _body_for(nslc, e)
        # durable staging through stage_raw_atomic is the PRODUCER's job post-R1;
        # the fake serves bytes plus a provider-side temp file only
        name = _sha(body)[:24] + ".ms.part"
        path = os.path.join(stage_dir, name)
        with open(path, "wb") as f:
            f.write(body)
        return {"stream": _traces_of(body), "raw_objects": [{
            "source": f"fake://{provider}/{nslc}/{e.date().isoformat()}",
            "staged_path": path, "size_bytes": len(body), "sha256": _sha(body)}]}


RECEIPT = {"status": "VERIFIED_DIRECT", "in_session_timestamp_utc": "2026-08-09T02:04:49Z",
           "owner_quote_sha256": 64 * "b"}


def _mk_clock(start, step):
    state = {"t": start - timedelta(seconds=step)}

    def clock():
        state["t"] = state["t"] + timedelta(seconds=step)
        return state["t"]
    return clock


def _fixtures(P, root, n_days=2):
    """Small campaign: 1 carrier x n_days x 2 segments x 2 stations (fast resume cycles)."""
    incident_days = P.schedule_days("2026-07-29")[:n_days]
    activation_days = P.schedule_days("2026-08-09")[:n_days]
    union = sorted(set(incident_days) | set(activation_days))
    plan = {"schema": "geospec-d2-step4b-campaign-plan-v1",
            "contract_id": "codex-d2-step4b-2026-08-09-v1",
            "registered_utc": "2026-08-09T02:20:00.000000Z",
            "activation_reference_day": "2026-08-09",
            "incident_reference_day": "2026-07-29",
            "carriers": ["istanbul_marmara"],
            "providers": {"istanbul_marmara": {"provider": "KOERI",
                                               "endpoint": "eida.koeri.boun.edu.tr"}},
            "station_registry": {"istanbul_marmara": [
                {"segment_name": "s0", "station_id": "KO.A",
                 "ordered_nslc_candidates": ["KO.A..HHZ"]},
                {"segment_name": "s0", "station_id": "KO.B",
                 "ordered_nslc_candidates": ["KO.B..HHZ"]},
                {"segment_name": "s1", "station_id": "KO.C",
                 "ordered_nslc_candidates": ["KO.C..HHZ"]},
                {"segment_name": "s1", "station_id": "KO.D",
                 "ordered_nslc_candidates": ["KO.D..HHZ"]}]},
            "incident_days": incident_days, "activation_days": activation_days,
            "scheduled_days": union, "acquisition_order": ["KOERI", "SCEDC"],
            "free_sources_only": True, "outcomes_inspected_before_schedule": False}
    rows = []
    for day in union:
        end = f"{day}T07:00:13.094647Z"
        start = (datetime.strptime(end, "%Y-%m-%dT%H:%M:%S.%fZ")
                 - timedelta(seconds=86400)).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
        rows.append({"carrier_key": "istanbul_marmara", "scored_day": day,
                     "status": "REGISTERED", "record_sha256": 64 * "c",
                     "request_start_utc": start, "request_end_utc": end,
                     "publication_commit": 40 * "a",
                     "publication_repo_path": "docs/ensemble_latest.json",
                     "publication_record_artifact": "publication_records/x.json",
                     "record_git_blob": 40 * "b", "reason_codes": []})
    ledger = {"schema": "geospec-d2-published-phase-ledger-v1", "rows": rows}
    plan_b, ledger_b = _canon(plan), _canon(ledger)
    with open(os.path.join(root, "campaign_plan.json"), "wb") as f:
        f.write(plan_b)
    with open(os.path.join(root, "published_phase_ledger.json"), "wb") as f:
        f.write(ledger_b)
    return plan, ledger, plan_b, ledger_b


def _run(P, root, prov, step=1, start=None):
    return P.run_campaign(
        plan=None, launch_authorization=RECEIPT, root=root,
        clock=_mk_clock(start or datetime(2026, 8, 9, 12, 0, 0, tzinfo=timezone.utc), step),
        providers=prov)


def _read_json(root, rel):
    with open(os.path.join(root, rel), "rb") as f:
        return json.loads(f.read().decode())


def _read_jsonl(root, rel):
    with open(os.path.join(root, rel), "rb") as f:
        return [json.loads(x) for x in f.read().decode().splitlines() if x.strip()]


def main():
    _install_stubs()
    try:
        import d2_step4b_producer as P
    except ImportError as exc:
        check("RK0-GATE producer imports", False, str(exc))
        return
    try:
        import d2_step4b_resume as RS
        need = ("ResumeIntegrityError", "campaign_id_of", "stage_raw_atomic",
                "append_event", "load_resume_state")
        assert all(hasattr(RS, n) for n in need), "resume seams incomplete"
        run_sig = P.run_campaign.__doc__ or ""
        import inspect
        assert "providers" in inspect.signature(P.run_campaign).parameters, \
            "run_campaign lacks injectable providers (resume harness seam)"
    except (ImportError, AssertionError) as exc:
        check("RK0-GATE resume module + seams (d2_step4b_resume.py: ResumeIntegrityError, "
              "campaign_id_of, stage_raw_atomic, append_event, load_resume_state; "
              "run_campaign grows injectable providers=)", False,
              f"AWAITING resume implementation -- red-first as authored ({exc})")
        for fam, what in (("RK1", "identity/authority/pre-I/O causal order"),
                          ("RK2", "atomic bytes/provenance/orphans"),
                          ("RK3", "idempotent re-entry/indeterminate retention"),
                          ("RK4", "multi-process scientific equivalence"),
                          ("RK5", "finality/legacy attestation/locking")):
            check(f"{fam}-BLOCKED {what}", False,
                  "BLOCKED by the missing resume module (explicit family marker)")
        return

    # =========================== RK1 =========================================
    plan_probe = b"PLAN-BYTES"
    ledger_probe = b"LEDGER-BYTES"
    expect = hashlib.sha256(b"geospec-d2-step4b-resume-v1\0"
                            + hashlib.sha256(plan_probe).digest()
                            + hashlib.sha256(ledger_probe).digest()).hexdigest()
    check("RK1a campaign_id_of implements the exact R0.1 construction (binary digests)",
          RS.campaign_id_of(plan_probe, ledger_probe) == expect)

    with tempfile.TemporaryDirectory() as tp:
        root = os.path.join(tp, "c")
        os.makedirs(root)
        _fixtures(P, root)
        prov1 = _Prov(die_after_fetches=0)          # dies at the FIRST fetch
        died = raises(lambda: _run(P, root, prov1), _Die)
        state = RS.load_resume_state(root)          # dict with "events" (validated chain)
        events = state["events"]
        kinds = [e.get("kind") for e in events]
        started_idx = next((i for i, k in enumerate(kinds) if "PROCESS_STARTED" in str(k)),
                           None)
        check("RK1b PROCESS_STARTED is durably committed BEFORE any provider I/O (a process "
              "that dies at its first fetch still left the start event + attempt intent)",
              died and started_idx is not None and len(events) >= 2
              and any("UNIT_ATTEMPT_STARTED" in str(k) for k in kinds),
              f"kinds={kinds[:6]}")
        # drift refusal: tamper the staged plan AFTER state exists -> refuse pre-I/O
        with open(os.path.join(root, "campaign_plan.json"), "ab") as f:
            f.write(b" ")
        prov2 = _Prov()
        refused = raises(lambda: _run(P, root, prov2), BaseException)
        check("RK1c plan/ledger drift on a resumable root refuses BEFORE provider I/O",
              refused and prov2.fetches == [] and prov2.probes == [],
              f"fetches={len(prov2.fetches)} probes={len(prov2.probes)}")
        # authorization is per-process even on resume
        prov3 = _Prov()
        with open(os.path.join(root, "campaign_plan.json"), "rb+") as f:
            data = f.read()[:-1]
            f.seek(0)
            f.truncate()
            f.write(data)                            # restore exact plan bytes
        bad = raises(lambda: P.run_campaign(plan=None, launch_authorization=None,
                                            root=root, providers=prov3), BaseException)
        check("RK1d a resume segment re-verifies its OWN receipt (invalid -> refusal, zero "
              "provider calls)", bad and prov3.fetches == [] and prov3.probes == [])

    # =========================== RK2 =========================================
    with tempfile.TemporaryDirectory() as tp:
        dest = os.path.join(tp, "raw_objects")
        os.makedirs(dest)
        body = b"RAW-OBJECT-BYTES-1"
        r1 = RS.stage_raw_atomic(body, dest)
        p1 = os.path.join(dest, os.path.basename(r1["relative_path"]))
        mtime1 = os.stat(p1).st_mtime_ns
        r2 = RS.stage_raw_atomic(body, dest)         # same digest -> verified reuse, no write
        ok_reuse = (r2["sha256"] == r1["sha256"] == _sha(body)
                    and os.stat(p1).st_mtime_ns == mtime1
                    and open(p1, "rb").read() == body
                    and not any(n.endswith((".tmp", ".part")) for n in os.listdir(dest)))
        with open(p1, "r+b") as f:                   # corrupt the installed object
            f.write(b"X")
        tampered = raises(lambda: RS.stage_raw_atomic(body, dest), RS.ResumeIntegrityError)
        check("RK2a atomic staging: fsync+rename install; same-digest re-stage is verified "
              "REUSE without write (no temp residue); corrupted existing bytes at the digest "
              "name -> ResumeIntegrityError (never silently overwritten)",
              r1["size_bytes"] == len(body) and ok_reuse and tampered,
              f"reuse={ok_reuse} tampered={tampered}")

    with tempfile.TemporaryDirectory() as tp:
        root = os.path.join(tp, "c")
        os.makedirs(root)
        _fixtures(P, root)
        ev1 = RS.append_event(root, {"kind": "PROCESS_STARTED", "process_id": "p1",
                                     "process_started_utc": "2026-08-09T12:00:00.000000Z"})
        ev2 = RS.append_event(root, {"kind": "UNIT_ATTEMPT_STARTED", "unit_id": 64 * "a",
                                     "attempt_id": 64 * "d", "status": "IN_PROGRESS"})
        chain_ok = (ev2.get("prev_event_sha256") == ev1.get("event_sha256")
                    and len(ev2["event_sha256"]) == 64)
        sp = os.path.join(root, "resume_state.json")
        raw_state = open(sp, "rb").read()
        open(sp, "wb").write(raw_state.replace(b"p1", b"pX", 1))
        tamper_ok = raises(lambda: RS.load_resume_state(root), RS.ResumeIntegrityError)
        open(sp, "wb").write(raw_state)
        escape = raises(lambda: RS.append_event(root, {
            "kind": "UNIT_TERMINAL", "unit_id": 64 * "a", "attempt_id": 64 * "d",
            "status": "FETCHED",
            "objects": [{"relative_path": "../evil.ms", "sha256": 64 * "e",
                         "size_bytes": 1}]}), RS.ResumeIntegrityError)
        check("RK2b hash-chained events: prev/event digests link; a single tampered byte in "
              "resume_state -> ResumeIntegrityError on load; an escaping referenced path "
              "refuses", chain_ok and tamper_ok and escape,
              f"chain={chain_ok} tamper={tamper_ok} escape={escape}")

    # =========================== RK3 + RK4 ===================================
    with tempfile.TemporaryDirectory() as tp:
        # ---- run A: uninterrupted reference -------------------------------------
        rootA = os.path.join(tp, "A")
        os.makedirs(rootA)
        _fixtures(P, rootA)
        provA = _Prov(unavailable_nslc={"KO.D..HHZ"})
        resA = _run(P, rootA, provA)
        # ---- run B: same fixtures, die mid-campaign, then resume ----------------
        rootB = os.path.join(tp, "B")
        os.makedirs(rootB)
        _fixtures(P, rootB)
        provB1 = _Prov(die_after_fetches=2, unavailable_nslc={"KO.D..HHZ"})
        died = raises(lambda: _run(P, rootB, provB1), _Die)
        provB2 = _Prov(unavailable_nslc={"KO.D..HHZ"})
        resB = _run(P, rootB, provB2,
                    start=datetime(2026, 8, 9, 14, 0, 0, tzinfo=timezone.utc))
        fetchedB1 = {f[0] + "|" + f[1] for f in provB1.fetches[:2]}
        refetched = {f[0] + "|" + f[1] for f in provB2.fetches}
        check("RK3a resume executes ONLY nonterminal units: no unit fetched by the crashed "
              "segment is fetched again (terminal FETCHED reused; scripted-UNAVAILABLE not "
              "re-probed into a different outcome)",
              died and fetchedB1.isdisjoint(refetched),
              f"first={sorted(fetchedB1)} refetched={sorted(refetched)}")
        attempts = _read_jsonl(rootB, "acquisition_attempts.jsonl")
        keys = [(a["carrier_key"], a["scored_day"], a["segment_name"], a["station_id"])
                for a in attempts]
        indeterminate = [a for a in attempts
                         if a.get("indeterminate_attempt_count", 0) >= 1]
        check("RK3b ONE summary row per logical unit (no duplicates); the death-interrupted "
              "unit retains its INDETERMINATE history (count >= 1) with a NEW terminal "
              "attempt — never rewritten",
              len(keys) == len(set(keys)) and len(indeterminate) >= 1
              and all(a.get("terminal_attempt_id") for a in attempts),
              f"units={len(keys)} indeterminate={len(indeterminate)}")
        dailyA = _read_jsonl(rootA, "calibration_daily.jsonl")
        dailyB = _read_jsonl(rootB, "calibration_daily.jsonl")
        admA = _read_json(rootA, "admission_results.json")["regions"]
        admB = _read_json(rootB, "admission_results.json")["regions"]
        sci = lambda rows: [{k: v for k, v in r.items()} for r in rows]
        adm_sci = lambda rows: [
            {k: v for k, v in r.items()
             if k not in ("capsule_path", "capsule_sha256")} for r in rows]
        check("RK4a SCIENTIFIC EQUIVALENCE: uninterrupted vs interrupted+resumed produce "
              "IDENTICAL calibration_daily rows and admission payloads (only process/time "
              "provenance may differ)",
              sci(dailyA) == sci(dailyB) and adm_sci(admA) == adm_sci(admB),
              f"daily_eq={sci(dailyA) == sci(dailyB)}")
        bmB = _read_json(rootB, "batch_manifest.json")
        pl = _read_jsonl(rootB, "campaign_process_ledger.jsonl")
        manifest_objs = _read_json(rootB, "input_manifest.json")["objects"]
        reused = [o for o in manifest_objs
                  if o.get("reuse_disposition") == "REUSED_VERIFIED"]
        plan_bytes = open(os.path.join(rootB, "campaign_plan.json"), "rb").read()
        ledger_bytes = open(os.path.join(rootB, "published_phase_ledger.json"), "rb").read()
        check("RK4b honest multi-process ledger: >=2 process rows (crashed segment "
              "included, ended_utc null + observed_dead_utc set); pre-crash objects are "
              "REUSED_VERIFIED with acquired != verified process; run_id == campaign_id; "
              "campaign_started_utc == FIRST process start",
              len(pl) >= 2
              and any(r.get("process_ended_utc") is None
                      and r.get("observed_dead_utc") for r in pl)
              and len(reused) >= 1
              and all(o["acquired_process_id"] != o["verified_process_id"] for o in reused)
              and bmB["run_id"] == RS.campaign_id_of(plan_bytes, ledger_bytes)
              and bmB["campaign_started_utc"]
              == min(r["process_started_utc"] for r in pl),
              f"processes={len(pl)} reused={len(reused)}")

        # =========================== RK5 =====================================
        bm_bytes_before = open(os.path.join(rootB, "batch_manifest.json"), "rb").read()
        provC = _Prov()
        res2 = P.run_campaign(plan=None, launch_authorization=RECEIPT, root=rootB,
                              providers=provC)
        bm_bytes_after = open(os.path.join(rootB, "batch_manifest.json"), "rb").read()
        check("RK5a a COMPLETED root is immutable: re-invocation verifies and returns "
              "(byte-identical manifest, zero provider calls, never re-finalized)",
              bm_bytes_before == bm_bytes_after and provC.fetches == []
              and provC.probes == [] and isinstance(res2, dict),
              f"fetches={len(provC.fetches)}")
        arts = bmB["artifacts"]
        bound = set(arts)
        on_disk = set()
        for dp, _dn, fns in os.walk(rootB):
            for fn in fns:
                rel = os.path.relpath(os.path.join(dp, fn), rootB).replace(os.sep, "/")
                if rel != "batch_manifest.json":
                    on_disk.add(rel)
        check("RK5b exact closure holds through resume: every root file manifest-bound "
              "(resume_state + process ledger included); no temp/part/lock/orphan inside",
              on_disk == bound
              and "resume_state.json" in bound
              and "campaign_process_ledger.jsonl" in bound
              and not any(r.endswith((".tmp", ".part", ".lock")) for r in on_disk),
              f"unbound={sorted(on_disk - bound)[:4]} missing={sorted(bound - on_disk)[:4]}")

    # ---- RK5c/d: R4 legacy attestation ---------------------------------------
    with tempfile.TemporaryDirectory() as tp:
        root = os.path.join(tp, "L")
        os.makedirs(os.path.join(root, "raw_objects"))
        _fixtures(P, root, n_days=1)
        end = datetime.strptime(_read_json(root, "published_phase_ledger.json")["rows"][0]
                                ["request_end_utc"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=timezone.utc)
        legacy_match = _body_for("KO.A..HHZ", end)          # will match the refetch
        legacy_stale = b"TRACES:" + b"[]|STALE-KO.B"        # will NOT match
        for body, in ((legacy_match,), (legacy_stale,)):
            with open(os.path.join(root, "raw_objects", _sha(body) + ".ms"), "wb") as f:
                f.write(body)
        stale_for_b = {"KO.B..HHZ": _body_for("KO.B..HHZ", end)}   # fresh differs from stale
        prov = _Prov(serve_override=stale_for_b)
        _run(P, root, prov)
        objs = _read_json(root, "input_manifest.json")["objects"]
        by_sha = {o["sha256"]: o for o in objs}
        attested = by_sha.get(_sha(legacy_match), {})
        check("RK5c LEGACY refetch-attestation: an UNATTESTED_LEGACY byte identical to the "
              "fresh authorized response is admitted as "
              "LEGACY_REUSED_AFTER_REFETCH_ATTESTATION",
              attested.get("reuse_disposition") == "LEGACY_REUSED_AFTER_REFETCH_ATTESTATION",
              f"disposition={attested.get('reuse_disposition')}")
        quarantine = root + ".legacy_quarantine"
        stale_in_manifest = _sha(legacy_stale) in by_sha
        stale_quarantined = (os.path.isdir(quarantine)
                             and any(_sha(open(os.path.join(quarantine, n), "rb").read())
                                     == _sha(legacy_stale)
                                     for n in os.listdir(quarantine)))
        stale_in_root = any(_sha(legacy_stale) in n
                            for n in os.listdir(os.path.join(root, "raw_objects")))
        check("RK5d a legacy byte the fresh response does NOT reproduce never enters the "
              "manifest/closure: quarantined at the sibling path, out of the final root",
              (not stale_in_manifest) and stale_quarantined and (not stale_in_root),
              f"in_manifest={stale_in_manifest} quarantined={stale_quarantined} "
              f"in_root={stale_in_root}")


main()
print()
if FAILS:
    print(f"D2 STEP-4B RESUME RED-KAT FAILURES: {FAILS}")
    sys.exit(1)
print("ALL D2 STEP-4B RESUME RED-KATs PASS (identity+pre-I/O causal order + atomic "
      "chained state + idempotent re-entry + multi-process scientific equivalence + "
      "immutable finality + legacy attestation)")
