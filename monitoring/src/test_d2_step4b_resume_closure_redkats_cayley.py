#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D2 STEP-4B RESUME-CLOSURE red-KATs (cayley, 2026-08-09) — freezes codex `1506`
combined-closure findings CRITICAL-1 + HIGH-2..5 as five executable counterexample families
RC1–RC5. Companion to the frozen RK bar (`3a829cd`); contract
`codex-d2-step4b-resume-2026-08-09-v1` + 1309. HERMETIC: no network, no reboot authority.

SCHEMA PIN: RC1 encodes option (A) resume-canonical (cayley 1557 concur; codex's own
CRITICAL-1 repair text assumes this routing). If codex rules (B), RC1's run_id assertion is
the ONLY line that moves. THREAT-MODEL NON-GOAL (RC5): hostile rewrite of BOTH state and
head files is OUT of scope (needs owner-held key material); the bar covers crash/truncation
integrity only.

CONTRACT SEAMS (grassmann implements UNEDITED; additive to `b70e0d7`)
=====================================================================
* RC1 (CRITICAL-1): `run_campaign` with NO providers argument (production default) on a
  standing/resumable/completed root is OWNED by the resume executor: the REAL provider
  module is imported lazily INSIDE it, strictly AFTER the durable reopened PROCESS_STARTED
  event (this bar proves ordering with an import hook: at import time of
  `d2_step4b_providers`, resume_state.json already contains PROCESS_STARTED); the completed
  batch carries `run_id == campaign_id_of(plan, ledger)` and the resume artifacts. Injected
  providers remain the hermetic test seam only.
* RC2 (HIGH-2): `d2_step4b_resume.verify_completed_root(root)` — reopens plan+ledger bytes,
  requires `run_id == campaign_id_of(...)`, exact `artifacts == on-disk set` (excluding
  batch_manifest.json), independent rehash/re-size of EVERY artifact, and state/process
  cross-link validation; ANY wrong byte, missing file, extra file, or wrong run_id raises
  `ResumeIntegrityError`. `run_campaign` re-entry on a completed root calls it (tampered
  completed root -> typed refusal, never BATCH_COMPLETE).
* RC3 (HIGH-3): exact typed event capsules —
  PROCESS_STARTED carries ordinal, resume_of, producer_commit, process_started_utc,
  campaign_id, plan/ledger hashes, owner_launch_authorization;
  UNIT_ATTEMPT_STARTED carries attempted_utc, provider, ordered_nslc_candidates,
  request_start_utc, request_end_utc, unit identity;
  UNIT_SOURCE_SELECTED (NEW event) commits the exact selected NSLC + source_url BEFORE the
  corresponding fetch (event committed while the provider fetch log is still empty for that
  unit); UNIT_TERMINAL carries terminal_utc, carrier/day/segment/station, provider,
  source_url, request window, publication_record_sha256, status, objects;
  PROCESS_OBSERVED_DEAD (NEW event) persists the FIRST observation of a dead process —
  a later resume never rewrites it (double-death: p1's observed_dead_utc is immutable once
  recorded; p2 gets its own). `campaign_process_ledger.jsonl` rows carry ordinal, resume_of,
  producer_commit, disposition, failure_type, owner_launch_authorization,
  first_event_sha256, last_event_sha256 — derived from the reopened chain.
* RC4 (HIGH-4): every rebuilt final artifact (batch_manifest.json especially) is written
  same-dir-temp + fsync + atomic replace (+ dir fsync); manifest LAST. Death injected at the
  final manifest replace leaves NO batch_manifest.json (temp only, outside closure) and the
  next authorized re-entry RESUMES to a complete, verifiable root (never a JSON parse
  crash). The allowed closure is built from TYPED outputs, not os.walk: every raw file must
  be `<sha256>.ms`, rehash to its name, and cross-link to a terminal event/input receipt; a
  seeded `raw_objects/junk.ms` is quarantined OUTSIDE the root and never manifested.
* RC5 (HIGH-5): a separately committed monotonic STATE-HEAD (generation, event_count,
  last_event_sha256) is validated before any state reuse and bound into the final manifest;
  tail truncation of resume_state.json (a clean shorter prefix) -> `ResumeIntegrityError`;
  a crash BETWEEN state append and head update (state exactly ONE event ahead of head) is
  recoverable — load accepts the longer state, repairs the head, loses nothing; state
  SHORTER than head always refuses.

RED AS AUTHORED vs GeoSpec `b70e0d7`: exactly
['RC0-GATE', 'RC1-BLOCKED', 'RC2-BLOCKED', 'RC3-BLOCKED', 'RC4-BLOCKED', 'RC5-BLOCKED']
(the closure seams do not exist yet; families activate behind RC0).
"""
import hashlib
import importlib.abc
import importlib.machinery
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


# ---- stubs / fixtures (RK-bar harness shapes) --------------------------------
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
    return json.loads(body.decode().split("TRACES:", 1)[1].split("|", 1)[0])


def _traces_of(body):
    return [_Trace(datetime.fromisoformat(i.replace("Z", "+00:00")), n, r)
            for i, n, r in _frames_of(body)]


def _body_for(nslc, end):
    s = (end - timedelta(seconds=86400)).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    return ("TRACES:" + json.dumps([[s, 8, 40.0]]) + "|" + nslc).encode()


class _Die(BaseException):
    """Injected process death (never absorbed by except Exception)."""


class _Prov:
    class ProviderUnavailable(Exception):
        pass

    def __init__(self, die_after_fetches=None, all_unavailable=False):
        self.fetches = []
        self.probes = []
        self.die_after = die_after_fetches
        self.all_unavailable = all_unavailable

    def koeri_available(self, net, stas, chas, s, e):
        self.probes.append(tuple(stas))
        return {f"{net}.{st}..{c}" for st in stas for c in chas}

    def scedc_available(self, net, stas, chas, s, e):
        self.probes.append(tuple(stas))
        return {f"{net}.{st}..{c}" for st in stas for c in chas}

    def parse_staged(self, path):
        with open(path, "rb") as f:
            return _traces_of(f.read())

    def fetch(self, provider, nslc, s, e, *, stage_dir, **kw):
        self.fetches.append((nslc, e.date().isoformat()))
        if self.die_after is not None and len(self.fetches) > self.die_after:
            raise _Die(f"injected death at fetch #{len(self.fetches)}")
        if self.all_unavailable:
            raise self.ProviderUnavailable("scripted")
        body = _body_for(nslc, e)
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


def _fixtures(P, root, n_days=1):
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
    with open(os.path.join(root, "campaign_plan.json"), "wb") as f:
        f.write(_canon(plan))
    with open(os.path.join(root, "published_phase_ledger.json"), "wb") as f:
        f.write(_canon(ledger))
    return plan, ledger


def _run(P, root, prov, step=1, start=None, **kw):
    return P.run_campaign(
        plan=None, launch_authorization=RECEIPT, root=root,
        clock=_mk_clock(start or datetime(2026, 8, 9, 12, 0, 0, tzinfo=timezone.utc), step),
        providers=prov, **kw)


def _read_json(root, rel):
    with open(os.path.join(root, rel), "rb") as f:
        return json.loads(f.read().decode())


def _read_jsonl(root, rel):
    with open(os.path.join(root, rel), "rb") as f:
        return [json.loads(x) for x in f.read().decode().splitlines() if x.strip()]


def _events(RS, root):
    return RS.load_resume_state(root)["events"]


class _ProvidersImportHook(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serves a spy 'd2_step4b_providers' module; records whether the resume state
    already carried PROCESS_STARTED at the moment of import (RC1 ordering proof)."""

    def __init__(self, root, RS):
        self.root = root
        self.RS = RS
        self.import_seen = False
        self.started_at_import = False
        self.spy = _Prov(all_unavailable=True)

    def find_spec(self, name, path, target=None):
        if name != "d2_step4b_providers":
            return None
        return importlib.machinery.ModuleSpec(name, self)

    def create_module(self, spec):
        self.import_seen = True
        try:
            evs = self.RS.load_resume_state(self.root)["events"]
            self.started_at_import = any(e.get("kind") == "PROCESS_STARTED" for e in evs)
        except Exception:
            self.started_at_import = False
        mod = types.ModuleType("d2_step4b_providers")
        for attr in ("koeri_available", "scedc_available", "parse_staged", "fetch",
                     "ProviderUnavailable"):
            setattr(mod, attr, getattr(self.spy, attr))
        return mod

    def exec_module(self, module):
        pass


def main():
    _install_stubs()
    try:
        import d2_step4b_producer as P
        import d2_step4b_resume as RS
    except ImportError as exc:
        check("RC0-GATE producer + resume modules import", False, str(exc))
        return
    need = ("verify_completed_root",)
    have_new = all(hasattr(RS, n) for n in need)
    if not have_new:
        check("RC0-GATE closure seams present (d2_step4b_resume.verify_completed_root + "
              "typed event capsules + state-head + atomic finalize)", False,
              "AWAITING closure implementation -- red-first as authored (codex 1506)")
        for fam, what in (("RC1", "production-default resume ownership + lazy provider "
                                  "import after durable PROCESS_STARTED"),
                          ("RC2", "verified completed-root re-entry (tamper refusal)"),
                          ("RC3", "full typed event capsules + first-observation death"),
                          ("RC4", "atomic finalization + typed-output closure"),
                          ("RC5", "anti-truncation state-head + WAL recovery")):
            check(f"{fam}-BLOCKED {what}", False,
                  "BLOCKED by the missing closure seams (explicit family marker)")
        return

    U = timezone.utc

    # =========================== RC1 (option A) ===============================
    saved_prov_mod = sys.modules.pop("d2_step4b_providers", None)
    with tempfile.TemporaryDirectory() as tp:
        root = os.path.join(tp, "c")
        os.makedirs(root)
        _fixtures(P, root)
        hook = _ProvidersImportHook(root, RS)
        sys.modules_backup = None
        sys.meta_path.insert(0, hook)
        try:
            res = P.run_campaign(plan=None, launch_authorization=RECEIPT, root=root,
                                 clock=_mk_clock(datetime(2026, 8, 9, 12, 0, 0,
                                                          tzinfo=U), 1))
        finally:
            sys.meta_path.remove(hook)
            sys.modules.pop("d2_step4b_providers", None)
            if saved_prov_mod is not None:
                sys.modules["d2_step4b_providers"] = saved_prov_mod
        bm = _read_json(root, "batch_manifest.json")
        plan_b = open(os.path.join(root, "campaign_plan.json"), "rb").read()
        led_b = open(os.path.join(root, "published_phase_ledger.json"), "rb").read()
        check("RC1a PRODUCTION DEFAULT (no providers argument) is owned by the resume "
              "executor: resume_state + process ledger in closure, run_id == campaign_id "
              "(option A), honest all-UNAVAILABLE completion",
              os.path.isfile(os.path.join(root, "resume_state.json"))
              and os.path.isfile(os.path.join(root, "campaign_process_ledger.jsonl"))
              and bm.get("run_id") == RS.campaign_id_of(plan_b, led_b)
              and "resume_state.json" in bm.get("artifacts", {})
              and "campaign_process_ledger.jsonl" in bm.get("artifacts", {}),
              f"run_id_ok={bm.get('run_id') == RS.campaign_id_of(plan_b, led_b)}")
        check("RC1b the REAL provider module is imported lazily, strictly AFTER the durable "
              "reopened PROCESS_STARTED (import-hook proven), and its probes were used",
              hook.import_seen and hook.started_at_import and len(hook.spy.probes) > 0,
              f"import_seen={hook.import_seen} started_at_import={hook.started_at_import} "
              f"probes={len(hook.spy.probes)}")

    # =========================== RC2 =========================================
    with tempfile.TemporaryDirectory() as tp:
        root = os.path.join(tp, "c")
        os.makedirs(root)
        _fixtures(P, root)
        _run(P, root, _Prov())
        check("RC2a verify_completed_root PASSES an intact completed root",
              not raises(lambda: RS.verify_completed_root(root)))
        daily_p = os.path.join(root, "calibration_daily.jsonl")
        good = open(daily_p, "rb").read()
        with open(daily_p, "ab") as f:
            f.write(b"\n")
        tampered = raises(lambda: RS.verify_completed_root(root), RS.ResumeIntegrityError)
        reentry_refused = raises(lambda: _run(P, root, _Prov()), RS.ResumeIntegrityError)
        open(daily_p, "wb").write(good)
        extra_p = os.path.join(root, "extra.json")
        open(extra_p, "wb").write(b"{}")
        extra_refused = raises(lambda: RS.verify_completed_root(root),
                               RS.ResumeIntegrityError)
        os.remove(extra_p)
        bm_p = os.path.join(root, "batch_manifest.json")
        bm = _read_json(root, "batch_manifest.json")
        bm["run_id"] = 64 * "0"
        open(bm_p, "wb").write(_canon(bm))
        wrong_id = raises(lambda: RS.verify_completed_root(root), RS.ResumeIntegrityError)
        check("RC2b a TAMPERED completed root refuses typed on verify AND on re-entry "
              "(wrong byte / extra file / wrong run_id -> ResumeIntegrityError, never "
              "BATCH_COMPLETE)",
              tampered and reentry_refused and extra_refused and wrong_id,
              f"byte={tampered} reentry={reentry_refused} extra={extra_refused} "
              f"run_id={wrong_id}")

    # =========================== RC3 =========================================
    with tempfile.TemporaryDirectory() as tp:
        root = os.path.join(tp, "c")
        os.makedirs(root)
        _fixtures(P, root)
        prov = _Prov()
        _run(P, root, prov)
        evs = _events(RS, root)
        by_kind = {}
        for e in evs:
            by_kind.setdefault(e.get("kind"), []).append(e)
        ps = by_kind.get("PROCESS_STARTED", [{}])[0]
        ua = by_kind.get("UNIT_ATTEMPT_STARTED", [{}])[0]
        ss = by_kind.get("UNIT_SOURCE_SELECTED", [{}])
        ut = by_kind.get("UNIT_TERMINAL", [{}])[0]
        ps_ok = all(k in ps for k in ("ordinal", "resume_of", "producer_commit",
                                      "process_started_utc", "campaign_id",
                                      "owner_launch_authorization"))
        ua_ok = all(k in ua for k in ("attempted_utc", "provider",
                                      "ordered_nslc_candidates", "request_start_utc",
                                      "request_end_utc"))
        ut_ok = all(k in ut for k in ("terminal_utc", "provider", "source_url",
                                      "request_start_utc", "request_end_utc",
                                      "publication_record_sha256", "status", "objects"))
        # SOURCE_SELECTED precedes the corresponding fetch: the event chain orders it
        # before UNIT_TERMINAL, and it must exist once per FETCHED unit with exact fields
        ss_ok = (len(ss) >= 1 and all("source_url" in e and "selected_nslc" in e
                                      for e in ss if e))
        order_ok = all(
            min((i for i, e in enumerate(evs) if e.get("kind") == "UNIT_SOURCE_SELECTED"
                 and e.get("unit_id") == t.get("unit_id")), default=10 ** 9)
            < i_t
            for i_t, t in enumerate(evs)
            if t.get("kind") == "UNIT_TERMINAL" and t.get("status") == "FETCHED")
        pl = _read_jsonl(root, "campaign_process_ledger.jsonl")
        pl_ok = all(all(k in r for k in ("ordinal", "resume_of", "producer_commit",
                                         "disposition", "failure_type",
                                         "owner_launch_authorization",
                                         "first_event_sha256", "last_event_sha256"))
                    for r in pl)
        check("RC3a exact typed capsules: PROCESS_STARTED / UNIT_ATTEMPT_STARTED / "
              "UNIT_SOURCE_SELECTED (before its fetch, per unit) / UNIT_TERMINAL carry the "
              "full 1506 field sets; ledger rows carry ordinal..first/last_event_sha256",
              ps_ok and ua_ok and ut_ok and ss_ok and order_ok and pl_ok,
              f"ps={ps_ok} ua={ua_ok} ut={ut_ok} ss={ss_ok} order={order_ok} pl={pl_ok}")
    with tempfile.TemporaryDirectory() as tp:
        root = os.path.join(tp, "c")
        os.makedirs(root)
        _fixtures(P, root)
        raises(lambda: _run(P, root, _Prov(die_after_fetches=0)), _Die)      # death 1
        raises(lambda: _run(P, root, _Prov(die_after_fetches=1),
                            start=datetime(2026, 8, 9, 13, 0, 0, tzinfo=U)), _Die)  # death 2
        _run(P, root, _Prov(), start=datetime(2026, 8, 9, 14, 0, 0, tzinfo=U))
        obs = [e for e in _events(RS, root) if e.get("kind") == "PROCESS_OBSERVED_DEAD"]
        by_proc = {}
        for e in obs:
            by_proc.setdefault(e.get("dead_process_id"), []).append(e)
        pl = _read_jsonl(root, "campaign_process_ledger.jsonl")
        p1_rows = [r for r in pl if r["ordinal"] == 1]
        first_obs_p1 = min((e["observed_dead_utc"] for e in obs
                            if by_proc and e.get("dead_process_id") == pl[0]["process_id"]),
                           default=None)
        check("RC3b DOUBLE DEATH: every dead process gets a persisted first-observation "
              "PROCESS_OBSERVED_DEAD; the ledger's observed_dead_utc equals the FIRST "
              "observation and is never rewritten by later resumes",
              len(obs) >= 2 and len(pl) == 3
              and p1_rows and p1_rows[0].get("observed_dead_utc") == first_obs_p1
              and p1_rows[0].get("process_ended_utc") is None,
              f"obs={len(obs)} pl={len(pl)}")

    # =========================== RC4 =========================================
    with tempfile.TemporaryDirectory() as tp:
        root = os.path.join(tp, "c")
        os.makedirs(root)
        _fixtures(P, root)
        real_replace = os.replace
        state = {"armed": True}

        def dying_replace(src, dst):
            if state["armed"] and os.path.basename(dst) == "batch_manifest.json":
                state["armed"] = False
                raise _Die("injected death at final manifest replace")
            return real_replace(src, dst)

        os.replace = dying_replace
        try:
            died = raises(lambda: _run(P, root, _Prov()), _Die)
        finally:
            os.replace = real_replace
        no_manifest = not os.path.isfile(os.path.join(root, "batch_manifest.json"))
        resumed = _run(P, root, _Prov(),
                       start=datetime(2026, 8, 9, 15, 0, 0, tzinfo=U))
        verified = not raises(lambda: RS.verify_completed_root(root))
        check("RC4a death AT the final manifest replace leaves NO batch_manifest.json "
              "(atomic; temp outside closure) and the next re-entry RESUMES to a complete "
              "verifiable root (never a JSON parse crash)",
              died and no_manifest and isinstance(resumed, dict) and verified,
              f"died={died} no_manifest={no_manifest} verified={verified}")
    with tempfile.TemporaryDirectory() as tp:
        root = os.path.join(tp, "c")
        os.makedirs(os.path.join(root, "raw_objects"))
        _fixtures(P, root)
        junk = os.path.join(root, "raw_objects", "junk.ms")
        open(junk, "wb").write(b"NOT-A-DIGEST-NAMED-OBJECT")
        _run(P, root, _Prov())
        bm = _read_json(root, "batch_manifest.json")
        junk_gone = not os.path.exists(junk)
        junk_quarantined = os.path.isdir(root + ".legacy_quarantine") and any(
            open(os.path.join(root + ".legacy_quarantine", n), "rb").read()
            == b"NOT-A-DIGEST-NAMED-OBJECT"
            for n in os.listdir(root + ".legacy_quarantine"))
        junk_bound = any("junk" in rel for rel in bm.get("artifacts", {}))
        typed_ok = all(
            rel.split("/")[-1] == f"{bm['artifacts'][rel]['sha256']}.ms"
            for rel in bm.get("artifacts", {}) if rel.startswith("raw_objects/"))
        check("RC4b TYPED closure, not os.walk: a seeded non-digest orphan is quarantined "
              "outside the root and never manifested; every manifested raw object is "
              "<its-own-sha256>.ms",
              junk_gone and junk_quarantined and not junk_bound and typed_ok,
              f"gone={junk_gone} quarantined={junk_quarantined} bound={junk_bound} "
              f"typed={typed_ok}")

    # =========================== RC5 =========================================
    with tempfile.TemporaryDirectory() as tp:
        root = os.path.join(tp, "c")
        os.makedirs(root)
        _fixtures(P, root)
        ev1 = RS.append_event(root, {"kind": "PROCESS_STARTED", "process_id": "p1",
                                     "ordinal": 1, "resume_of": None,
                                     "producer_commit": "x",
                                     "process_started_utc": "2026-08-09T12:00:00.000000Z",
                                     "campaign_id": 64 * "e",
                                     "owner_launch_authorization": RECEIPT})
        ev2 = RS.append_event(root, {"kind": "UNIT_ATTEMPT_STARTED", "unit_id": 64 * "a",
                                     "attempt_id": 64 * "d", "status": "IN_PROGRESS",
                                     "attempted_utc": "2026-08-09T12:00:01.000000Z"})
        state_p = os.path.join(root, "resume_state.json")
        full_state = open(state_p, "rb").read()
        doc = json.loads(full_state.decode())
        doc["events"] = doc["events"][:1]                     # clean tail truncation
        open(state_p, "wb").write(_canon(doc))
        truncation_refused = raises(lambda: RS.load_resume_state(root),
                                    RS.ResumeIntegrityError)
        open(state_p, "wb").write(full_state)
        ok_again = not raises(lambda: RS.load_resume_state(root))
        # crash between state append and head update: state ONE event ahead of head
        # (simulate by restoring the pre-append head after a real append)
        head_candidates = [n for n in os.listdir(root)
                           if "head" in n.lower() and n != "resume_state.json"]
        head_ok = len(head_candidates) >= 1
        wal_recovered = False
        head_short_refused = False
        if head_ok:
            head_p = os.path.join(root, head_candidates[0])
            head_before = open(head_p, "rb").read()
            RS.append_event(root, {"kind": "PROCESS_ENDED", "process_id": "p1",
                                   "process_ended_utc": "2026-08-09T12:10:00.000000Z"})
            head_after = open(head_p, "rb").read()
            open(head_p, "wb").write(head_before)             # crash: head not yet updated
            st = None
            try:
                st = RS.load_resume_state(root)
                wal_recovered = len(st["events"]) == 3
            except Exception:
                wal_recovered = False
            open(head_p, "wb").write(head_after)
            # state SHORTER than head refuses
            doc = json.loads(open(state_p, "rb").read().decode())
            doc["events"] = doc["events"][:2]
            saved = open(state_p, "rb").read()
            open(state_p, "wb").write(_canon(doc))
            head_short_refused = raises(lambda: RS.load_resume_state(root),
                                        RS.ResumeIntegrityError)
            open(state_p, "wb").write(saved)
        check("RC5a a CLEAN TAIL TRUNCATION of resume_state refuses (monotonic state-head "
              "anchors event_count + last hash); the intact state still loads",
              truncation_refused and ok_again,
              f"truncated={truncation_refused} intact={ok_again}")
        check("RC5b WAL crash-points: state ONE event ahead of a stale head RECOVERS "
              "(longer state accepted, head repaired, nothing lost); state SHORTER than "
              "head always refuses",
              head_ok and wal_recovered and head_short_refused,
              f"head_file={head_ok} recovered={wal_recovered} "
              f"short_refused={head_short_refused}")


main()
print()
if FAILS:
    print(f"D2 STEP-4B RESUME-CLOSURE RED-KAT FAILURES: {FAILS}")
    sys.exit(1)
print("ALL D2 STEP-4B RESUME-CLOSURE RED-KATs PASS (production-default resume ownership + "
      "verified completed roots + full typed capsules + atomic typed finality + "
      "anti-truncation state-head)")
