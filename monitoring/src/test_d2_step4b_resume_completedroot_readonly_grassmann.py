#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RC5d/RC2e regression KAT (grassmann, codex 1725 HIGH-1) — a COMPLETED root is an immutable,
READ-ONLY verification target: `verify_completed_root` must never mutate it. Counterexample:
after a root is finalized, install its VALID one-event-behind state-head predecessor and repair
only that artifact's SHA/size in batch_manifest.json so ordinary byte-integrity passes. The stale
head is a valid WAL prefix, so the pre-fix `load_resume_state` would repair it AFTER the artifact
loop — accepting once while making the root disagree with the manifest it just accepted. This KAT
locks the fix (read-only completed-root validation): both `verify_completed_root` and production
re-entry must raise `ResumeIntegrityError`, the head bytes must remain UNCHANGED, and there must be
ZERO provider calls. HERMETIC; authorises no reboot, provider I/O, or re-fire.

Companion to the frozen cayley RC bar (`d753440`); grassmann-owned (does not edit any frozen bar).
codex verifies this delta independently per the 1725 handoff.
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
    return (json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha(b):
    return hashlib.sha256(b).hexdigest()


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


class _Prov:
    class ProviderUnavailable(Exception):
        pass

    def __init__(self):
        self.fetches = []
        self.probes = []

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
        body = _body_for(nslc, e)
        path = os.path.join(stage_dir, _sha(body)[:24] + ".ms.part")
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
            "activation_reference_day": "2026-08-09", "incident_reference_day": "2026-07-29",
            "carriers": ["istanbul_marmara"],
            "providers": {"istanbul_marmara": {"provider": "KOERI",
                                               "endpoint": "eida.koeri.boun.edu.tr"}},
            "station_registry": {"istanbul_marmara": [
                {"segment_name": "s0", "station_id": "KO.A", "ordered_nslc_candidates": ["KO.A..HHZ"]},
                {"segment_name": "s0", "station_id": "KO.B", "ordered_nslc_candidates": ["KO.B..HHZ"]},
                {"segment_name": "s1", "station_id": "KO.C", "ordered_nslc_candidates": ["KO.C..HHZ"]},
                {"segment_name": "s1", "station_id": "KO.D", "ordered_nslc_candidates": ["KO.D..HHZ"]}]},
            "incident_days": incident_days, "activation_days": activation_days,
            "scheduled_days": union, "acquisition_order": ["KOERI", "SCEDC"],
            "free_sources_only": True, "outcomes_inspected_before_schedule": False}
    rows = []
    for day in union:
        end = f"{day}T07:00:13.094647Z"
        start = (datetime.strptime(end, "%Y-%m-%dT%H:%M:%S.%fZ")
                 - timedelta(seconds=86400)).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
        rows.append({"carrier_key": "istanbul_marmara", "scored_day": day, "status": "REGISTERED",
                     "record_sha256": 64 * "c", "request_start_utc": start, "request_end_utc": end,
                     "publication_commit": 40 * "a", "publication_repo_path": "docs/ensemble_latest.json",
                     "publication_record_artifact": "publication_records/x.json",
                     "record_git_blob": 40 * "b", "reason_codes": []})
    ledger = {"schema": "geospec-d2-published-phase-ledger-v1", "rows": rows}
    with open(os.path.join(root, "campaign_plan.json"), "wb") as f:
        f.write(_canon(plan))
    with open(os.path.join(root, "published_phase_ledger.json"), "wb") as f:
        f.write(_canon(ledger))


def _run(P, root, prov, start=None):
    return P.run_campaign(plan=None, launch_authorization=RECEIPT, root=root,
                          clock=_mk_clock(start or datetime(2026, 8, 9, 12, 0, 0, tzinfo=timezone.utc), 1),
                          providers=prov)


def main():
    _install_stubs()
    import d2_step4b_producer as P
    import d2_step4b_resume as RS

    with tempfile.TemporaryDirectory() as tp:
        root = os.path.join(tp, "c")
        os.makedirs(root)
        _fixtures(P, root)
        _run(P, root, _Prov())                                  # finalize a completed root
        check("setup: completed root verifies clean", not raises(lambda: RS.verify_completed_root(root)))

        state_p = os.path.join(root, "resume_state.json")
        head_p = os.path.join(root, "resume_state.head.json")
        bm_p = os.path.join(root, "batch_manifest.json")
        events = json.loads(open(state_p, "rb").read().decode())["events"]
        cur_head = json.loads(open(head_p, "rb").read().decode())
        n = len(events)

        # install the VALID one-event-behind predecessor head (event_count n-1, prefix hash exact)
        stale_head = {"generation": cur_head["generation"], "event_count": n - 1,
                      "last_event_sha256": events[n - 2]["event_sha256"]}
        stale_bytes = _canon(stale_head)
        with open(head_p, "wb") as f:
            f.write(stale_bytes)
        # repair ONLY that artifact's byte-integrity in the manifest so a naive rehash passes
        bm = json.loads(open(bm_p, "rb").read().decode())
        bm["artifacts"]["resume_state.head.json"] = {"sha256": _sha(stale_bytes), "size": len(stale_bytes)}
        with open(bm_p, "wb") as f:
            f.write(_canon(bm))

        verify_refused = raises(lambda: RS.verify_completed_root(root), RS.ResumeIntegrityError)
        head_after_verify = open(head_p, "rb").read()
        prov = _Prov()
        reentry_refused = raises(lambda: _run(P, root, prov), RS.ResumeIntegrityError)
        head_after_reentry = open(head_p, "rb").read()

        check("RC5d/RC2e a stale (valid one-behind) state-head with byte-integrity repaired in the "
              "manifest REFUSES on verify_completed_root AND on production re-entry — the verifier "
              "never repairs an immutable completed root, and re-entry makes zero provider calls",
              verify_refused and reentry_refused
              and head_after_verify == stale_bytes and head_after_reentry == stale_bytes
              and prov.fetches == [] and prov.probes == [],
              f"verify={verify_refused} reentry={reentry_refused} "
              f"head_unchanged={head_after_verify == stale_bytes == head_after_reentry} "
              f"provider_calls={len(prov.fetches) + len(prov.probes)}")


main()
print()
if FAILS:
    print(f"D2 STEP-4B COMPLETED-ROOT READ-ONLY KAT FAILURES: {FAILS}")
    sys.exit(1)
print("COMPLETED-ROOT READ-ONLY KAT PASS (verify_completed_root never mutates an immutable root; "
      "stale-head + repaired-manifest refuses on verify and re-entry; zero provider calls)")
