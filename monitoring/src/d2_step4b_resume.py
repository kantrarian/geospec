#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""d2_step4b_resume.py — the D2 step-4b RESUMABILITY layer (grassmann, implements the frozen
cayley bar `test_d2_step4b_resume_redkats_cayley.py` @ GeoSpec `3a829cd`, which freezes codex
extension `codex-d2-step4b-resume-2026-08-09-v1` (1303) + the 1309 pre-I/O causal-order rule).

Additive to `0f3df30`: the production driver (`run_campaign(..., providers=None)`) is byte-for-
byte unchanged — this module is reached ONLY when an injected `providers` object is supplied
(the resume harness seam). Nothing here reboots, performs real provider I/O, re-fires, mints,
lifts a freeze, or makes a claim; the actual resume still waits on asylum's reboot timing +
fresh direct go. Integrity violations raise `ResumeIntegrityError` and NEVER degrade to data-
unavailability or mint a manifest.

Seams (grassmann UNEDITED to the bar):
  * ResumeIntegrityError(RuntimeError)
  * campaign_id_of(plan_bytes, ledger_bytes) -> 64-hex        [R0.1; binary digests]
  * stage_raw_atomic(raw, dest_dir) -> {relative_path, sha256, size_bytes}   [R1.2]
  * append_event(root, event) -> committed hash-chained event  [R1.3]
  * load_resume_state(root) -> {"events": [...], "campaign_id": ...}
  * run_resume_campaign(...) — the resume-aware executor (RK1..RK5), dispatched from the
    producer's single receipt-gated `run_campaign` when providers are injected.
"""
import hashlib
import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timedelta, timezone

RESUME_CAMPAIGN_ID_PREFIX = b"geospec-d2-step4b-resume-v1\0"
_GENESIS = "0" * 64
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_LEGACY_QUARANTINE_SUFFIX = ".legacy_quarantine"
_FETCH_SCRATCH_SUFFIX = ".fetch_scratch"


class ResumeIntegrityError(RuntimeError):
    """An integrity violation (identity drift, tamper, corruption, escaping path, size/sha
    mismatch). It NEVER degrades to data-unavailability and NEVER mints/relabels a manifest."""


# ---- primitives ------------------------------------------------------------
def _sha_hex(raw):
    return hashlib.sha256(raw).hexdigest()


def _canon_bytes(value):
    """Canonical JSON bytes (sorted keys, compact separators, no NaN); one logical value."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def campaign_id_of(plan_bytes, ledger_bytes):
    """R0.1 exact construction over the RAW staged plan/ledger BYTES (binary digests):
        sha256(b"geospec-d2-step4b-resume-v1\\0" + sha256(plan).digest() + sha256(ledger).digest())
    A non-bytes input refuses — the campaign identity is bound to the exact staged bytes, so a
    single tampered byte in campaign_plan.json / published_phase_ledger.json changes the id."""
    if not isinstance(plan_bytes, (bytes, bytearray)) or not isinstance(ledger_bytes, (bytes, bytearray)):
        raise ResumeIntegrityError("campaign_id_of requires plan/ledger BYTES")
    return hashlib.sha256(RESUME_CAMPAIGN_ID_PREFIX
                          + hashlib.sha256(bytes(plan_bytes)).digest()
                          + hashlib.sha256(bytes(ledger_bytes)).digest()).hexdigest()


def _atomic_write_bytes(path, data):
    """Temp-in-same-dir + flush/fsync + atomic os.replace. On any failure the temp is removed and
    the exception re-raised (a partial resume_state / object is never installed)."""
    dest_dir = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=dest_dir, prefix=".rs-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        finally:
            raise


# ---- referenced-path safety ------------------------------------------------
def _within_root(root, rel):
    """True iff `rel` is a relative path that resolves inside `root` (no escape, not absolute)."""
    if not isinstance(rel, str) or rel == "" or os.path.isabs(rel):
        return False
    root_abs = os.path.abspath(root)
    target = os.path.abspath(os.path.normpath(os.path.join(root_abs, rel)))
    return target == root_abs or target.startswith(root_abs + os.sep)


def _check_object_paths(root, event):
    for obj in (event.get("objects") or []):
        rel = obj.get("relative_path") if isinstance(obj, dict) else None
        if rel is not None and not _within_root(root, rel):
            raise ResumeIntegrityError(f"event references escaping/unresolvable path {rel!r}")


# ---- hash-chained event log (resume_state.json) ----------------------------
def _event_canon(event):
    body = {k: v for k, v in event.items() if k != "event_sha256"}
    return _canon_bytes(body)


def _event_digest(event):
    return _sha_hex(_event_canon(event))


def _resume_state_path(root):
    return os.path.join(root, "resume_state.json")


def _read_events(root):
    path = _resume_state_path(root)
    if not os.path.exists(path):
        return []
    with open(path, "rb") as fh:
        raw = fh.read()
    try:
        doc = json.loads(raw.decode("utf-8"))
    except Exception as exc:                                       # noqa: BLE001
        raise ResumeIntegrityError(f"resume_state.json is not valid JSON: {exc}")
    if not isinstance(doc, dict) or not isinstance(doc.get("events"), list):
        raise ResumeIntegrityError("resume_state.json missing an events list")
    return doc["events"]


def _validate_chain(root, events):
    """Full re-derivation: each event's prev-hash links the prior event_sha256; each event_sha256
    equals sha256(canonical event sans its own hash); referenced paths stay inside root. ANY
    break/tamper/truncation raises ResumeIntegrityError."""
    prev = _GENESIS
    for i, ev in enumerate(events):
        if not isinstance(ev, dict) or "event_sha256" not in ev or "prev_event_sha256" not in ev:
            raise ResumeIntegrityError(f"event {i} missing chain fields")
        if ev["prev_event_sha256"] != prev:
            raise ResumeIntegrityError(f"event {i} prev-hash break")
        if _event_digest(ev) != ev["event_sha256"]:
            raise ResumeIntegrityError(f"event {i} content tampered")
        _check_object_paths(root, ev)
        prev = ev["event_sha256"]
    return events


def load_resume_state(root):
    """Validated resume state: full chain re-derivation (tamper/truncation/escape -> refusal),
    surfacing the bound campaign_id (from the first PROCESS_STARTED) for drift checks."""
    events = _validate_chain(root, _read_events(root))
    campaign_id = None
    for ev in events:
        if ev.get("kind") == "PROCESS_STARTED":
            campaign_id = ev.get("campaign_id")
            break
    return {"events": events, "campaign_id": campaign_id}


def append_event(root, event):
    """Commit one event: chain it (prev = last event_sha256 or genesis), validate any referenced
    path (escape -> refusal), stamp its own event_sha256, and atomically replace + fsync
    resume_state.json. Re-validates the existing chain first so an appended event can never sit
    atop a tampered/truncated log."""
    if not isinstance(event, dict):
        raise ResumeIntegrityError("event must be a dict")
    events = _validate_chain(root, _read_events(root))
    prev = events[-1]["event_sha256"] if events else _GENESIS
    new_ev = {k: v for k, v in event.items() if k != "event_sha256"}
    new_ev["prev_event_sha256"] = prev
    _check_object_paths(root, new_ev)
    new_ev["event_sha256"] = _event_digest(new_ev)
    events = list(events) + [new_ev]
    _atomic_write_bytes(_resume_state_path(root),
                        _canon_bytes({"events": events}))
    return new_ev


# ---- durable raw-object staging -------------------------------------------
def stage_raw_atomic(raw, dest_dir):
    """R1.2: durable install of a raw object at `dest_dir/<sha256>.ms` via unique sibling temp +
    flush/fsync + atomic rename. An EXISTING destination is verified byte-for-byte (its bytes must
    hash to the digest that names it) and REUSED WITHOUT opening for write — mismatch (corruption
    at the digest name, or a legacy byte that no longer matches) raises ResumeIntegrityError and
    is NEVER silently overwritten. Returns {relative_path, sha256, size_bytes}; relative_path is
    `<basename(dest_dir)>/<sha>.ms` (root-relative when dest_dir is `<root>/raw_objects`)."""
    if not isinstance(raw, (bytes, bytearray)):
        raise ResumeIntegrityError("stage_raw_atomic requires bytes")
    raw = bytes(raw)
    sha = _sha_hex(raw)
    name = sha + ".ms"
    final = os.path.join(dest_dir, name)
    rel = os.path.basename(os.path.normpath(dest_dir)) + "/" + name
    if os.path.exists(final):
        with open(final, "rb") as fh:
            existing = fh.read()
        if _sha_hex(existing) != sha:
            raise ResumeIntegrityError(f"staged object {name} bytes do not match its digest name")
        return {"relative_path": rel, "sha256": sha, "size_bytes": len(raw)}
    os.makedirs(dest_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest_dir, prefix=".stage-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, final)
    except BaseException:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        finally:
            raise
    return {"relative_path": rel, "sha256": sha, "size_bytes": len(raw)}


def _unit_id(carrier, day, segment, station_id):
    return _sha_hex(_canon_bytes([carrier, day, segment, station_id]))


def _new_attempt_id():
    return uuid.uuid4().hex + uuid.uuid4().hex          # 64 hex, unique per attempt


# ============================================================================
# The resume-aware executor (dispatched from producer.run_campaign when a
# providers object is injected). Reuses the pinned scoring/admission machinery
# of d2_step4b_campaign_run (GeoSpec 3950a2c) so an uninterrupted run and an
# interrupted+resumed run are scientifically identical.
# ============================================================================
def run_resume_campaign(plan, root, *, providers, receipt, clock=None):
    import d2_step4b_campaign_run as CR
    import d2_step4b_producer as PROD

    # -- R0.3: a COMPLETED root is immutable (verify + return; zero provider I/O) ----
    bm_path = os.path.join(root, "batch_manifest.json")
    if os.path.exists(bm_path):
        with open(bm_path, "rb") as fh:
            bm = json.loads(fh.read().decode("utf-8"))
        return {"status": "BATCH_COMPLETE", "root": root, "run_id": bm.get("run_id"),
                "resumed": True, "candidates": bm.get("candidates", [])}

    # -- campaign identity from the RAW staged bytes (R0.1) --------------------------
    with open(os.path.join(root, "campaign_plan.json"), "rb") as fh:
        plan_bytes = fh.read()
    with open(os.path.join(root, "published_phase_ledger.json"), "rb") as fh:
        ledger_bytes = fh.read()
    campaign_id = campaign_id_of(plan_bytes, ledger_bytes)

    # -- single-writer lock (H6); everything below runs under it ---------------------
    writer_fd = PROD._acquire_writer_lock(root)
    try:
        return _execute(plan, root, providers, receipt, clock, campaign_id,
                        plan_bytes, ledger_bytes, CR)
    finally:
        PROD._release_writer_lock(writer_fd)


def _execute(plan, root, providers, receipt, clock, campaign_id, plan_bytes, ledger_bytes, CR):
    import seismic_data as SD
    import fault_correlation as FC

    tick = clock if clock is not None else CR._default_clock()

    # -- prior state + identity-drift refusal BEFORE any provider I/O (R0.2) ---------
    prior_events = load_resume_state(root)["events"]
    prior_started = {ev["process_id"]: ev for ev in prior_events
                     if ev.get("kind") == "PROCESS_STARTED"}
    prior_ended = {ev["process_id"] for ev in prior_events if ev.get("kind") == "PROCESS_ENDED"}
    if prior_events:
        bound = None
        for ev in prior_events:
            if ev.get("kind") == "PROCESS_STARTED":
                bound = ev.get("campaign_id")
                break
        if bound != campaign_id:
            raise ResumeIntegrityError(
                "plan/ledger drift on a resumable root: the staged bytes no longer match the "
                "campaign identity bound in resume_state — no provider I/O, no manifest.")

    # crashed prior processes are observed dead now (recorded in the process ledger) -
    observed_dead = {}
    for pid in prior_started:
        if pid not in prior_ended:
            observed_dead[pid] = _iso(tick())

    # per-unit prior outcomes -------------------------------------------------------
    terminal_by_unit, attempts_by_unit = {}, {}
    for ev in prior_events:
        kind = ev.get("kind")
        if kind == "UNIT_ATTEMPT_STARTED":
            attempts_by_unit.setdefault(ev["unit_id"], []).append(ev)
        elif kind == "UNIT_TERMINAL":
            terminal_by_unit[ev["unit_id"]] = ev

    # -- PROCESS_STARTED under the lock, committed+reopened BEFORE provider I/O (1309) -
    proc_id = uuid.uuid4().hex
    proc_started_utc = _iso(tick())
    append_event(root, {"kind": "PROCESS_STARTED", "process_id": proc_id,
                        "process_started_utc": proc_started_utc, "campaign_id": campaign_id,
                        "plan_sha256": _sha_hex(plan_bytes), "ledger_sha256": _sha_hex(ledger_bytes)})
    load_resume_state(root)                         # reopen: the start event is durable

    # -- UNATTESTED_LEGACY inventory (digest-named objects with no committed event) --
    raw_dir = os.path.join(root, "raw_objects")
    os.makedirs(raw_dir, exist_ok=True)
    attested_by_events = set()
    for ev in prior_events:
        for obj in (ev.get("objects") or []):
            attested_by_events.add(obj["sha256"])
    legacy_shas = set()
    for name in os.listdir(raw_dir):
        if name.endswith(".ms") and _HEX64.match(name[:-3]) and name[:-3] not in attested_by_events:
            legacy_shas.add(name[:-3])
    reproduced_legacy = set()

    scratch_dir = os.fspath(root) + _FETCH_SCRATCH_SUFFIX
    os.makedirs(scratch_dir, exist_ok=True)

    ledger = _load_ledger_dict(root)
    ledger_index = {(r["carrier_key"], r["scored_day"]): r for r in ledger["rows"]}
    station_registry = plan["station_registry"]
    scheduled_days = list(plan["scheduled_days"])
    carriers_ordered = ([c for c in CR.ELIGIBLE if CR.PROVIDERS[c]["provider"] == "KOERI"]
                        + [c for c in CR.ELIGIBLE if CR.PROVIDERS[c]["provider"] == "SCEDC"])

    attempts_summary, input_objects, day_result = [], [], {}

    def _summary(carrier, day, segment, station_id, provider, row, *, status, selected_nslc,
                 refs, reason, terminal_attempt_id, indeterminate, attempted_utc):
        return {"carrier_key": carrier, "scored_day": day, "segment_name": segment,
                "station_id": station_id, "provider": provider,
                "request_start_utc": row["request_start_utc"],
                "request_end_utc": row["request_end_utc"], "selected_nslc": selected_nslc,
                "status": status, "input_object_sha256s": refs, "reason_codes": reason,
                "attempted_utc": attempted_utc, "terminal_attempt_id": terminal_attempt_id,
                "indeterminate_attempt_count": indeterminate}

    for carrier in carriers_ordered:
        if carrier not in station_registry:            # carrier absent from this plan's registry
            continue
        provider = CR.PROVIDERS[carrier]["provider"]
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
                    station_id = srow["station_id"]
                    uid = _unit_id(carrier, day, segment, station_id)
                    prior_attempts = attempts_by_unit.get(uid, [])
                    term = terminal_by_unit.get(uid)

                    if term is not None:
                        # -- R2: terminal unit reused, never re-probed / re-fetched ----------
                        t_attempt = term.get("attempt_id")
                        indeterminate = sum(1 for a in prior_attempts
                                            if a.get("attempt_id") != t_attempt)
                        if term["status"] == "FETCHED":
                            nslc = term.get("selected_nslc")
                            refs = []
                            for obj in term["objects"]:
                                durable = os.path.join(root, obj["relative_path"])
                                if not os.path.exists(durable):
                                    raise ResumeIntegrityError(
                                        f"reused FETCHED object missing: {obj['relative_path']}")
                                with open(durable, "rb") as fh:
                                    body = fh.read()
                                if _sha_hex(body) != obj["sha256"] or len(body) != obj["size_bytes"]:
                                    raise ResumeIntegrityError(
                                        "reused FETCHED object size/sha mismatch (H3)")
                                prov = CR._object_provenance(providers, durable)   # H3 reparse
                                input_objects.append(_input_object(
                                    obj, carrier, day, segment, nslc, row, record_sha, prov,
                                    reuse="REUSED_VERIFIED",
                                    acquired=term.get("process_id"), verified=proc_id))
                                refs.append(obj["sha256"])
                            day_refs.extend(refs)
                            stream = providers.parse_staged(
                                os.path.join(root, term["objects"][0]["relative_path"]))
                            es = CR._station_series(SD, stream, nslc, start)
                            seg_station_es[segment].append((nslc, es))
                            attempts_summary.append(_summary(
                                carrier, day, segment, station_id, provider, row, status="FETCHED",
                                selected_nslc=nslc, refs=refs, reason=[],
                                terminal_attempt_id=t_attempt, indeterminate=indeterminate,
                                attempted_utc=term.get("attempted_utc")))
                        else:
                            attempts_summary.append(_summary(
                                carrier, day, segment, station_id, provider, row,
                                status=term["status"], selected_nslc=term.get("selected_nslc"),
                                refs=[], reason=list(term.get("reason_codes") or []),
                                terminal_attempt_id=t_attempt, indeterminate=indeterminate,
                                attempted_utc=term.get("attempted_utc")))
                        continue

                    # -- nonterminal unit: a fresh attempt (new ordinal, prior danglers kept) -
                    indeterminate = len(prior_attempts)
                    attempt_id = _new_attempt_id()
                    ordinal = indeterminate + 1
                    append_event(root, {"kind": "UNIT_ATTEMPT_STARTED", "unit_id": uid,
                                        "attempt_id": attempt_id, "ordinal": ordinal,
                                        "status": "IN_PROGRESS", "process_id": proc_id,
                                        "carrier_key": carrier, "scored_day": day,
                                        "segment_name": segment, "station_id": station_id})
                    attempted_utc = _iso(tick())
                    candidates = list(srow["ordered_nslc_candidates"])
                    net = candidates[0].split(".")[0]
                    stas = [c.split(".")[1] for c in candidates]
                    chas = [c.split(".")[3] for c in candidates]
                    avail = (providers.koeri_available(net, stas, chas, start, end)
                             if provider == "KOERI"
                             else providers.scedc_available(net, stas, chas, start, end))
                    nslc = None
                    for cand in candidates:
                        if cand in avail:
                            nslc = cand
                            break
                    if nslc is None:
                        append_event(root, {"kind": "UNIT_TERMINAL", "unit_id": uid,
                                            "attempt_id": attempt_id, "status": "UNAVAILABLE",
                                            "process_id": proc_id, "selected_nslc": None,
                                            "objects": [], "reason_codes": ["NO_AVAILABLE_CANDIDATE"],
                                            "attempted_utc": attempted_utc})
                        attempts_summary.append(_summary(
                            carrier, day, segment, station_id, provider, row, status="UNAVAILABLE",
                            selected_nslc=None, refs=[], reason=["NO_AVAILABLE_CANDIDATE"],
                            terminal_attempt_id=attempt_id, indeterminate=indeterminate,
                            attempted_utc=attempted_utc))
                        continue
                    try:
                        res = providers.fetch(provider, nslc, start, end, stage_dir=scratch_dir)
                    except providers.ProviderUnavailable:
                        append_event(root, {"kind": "UNIT_TERMINAL", "unit_id": uid,
                                            "attempt_id": attempt_id, "status": "UNAVAILABLE",
                                            "process_id": proc_id, "selected_nslc": nslc,
                                            "objects": [], "reason_codes": ["PROVIDER_UNAVAILABLE"],
                                            "attempted_utc": attempted_utc})
                        attempts_summary.append(_summary(
                            carrier, day, segment, station_id, provider, row, status="UNAVAILABLE",
                            selected_nslc=nslc, refs=[], reason=["PROVIDER_UNAVAILABLE"],
                            terminal_attempt_id=attempt_id, indeterminate=indeterminate,
                            attempted_utc=attempted_utc))
                        continue
                    except Exception as exc:                              # noqa: BLE001
                        append_event(root, {"kind": "UNIT_TERMINAL", "unit_id": uid,
                                            "attempt_id": attempt_id, "status": "ERROR",
                                            "process_id": proc_id, "selected_nslc": nslc,
                                            "objects": [],
                                            "reason_codes": [f"FETCH_ERROR:{type(exc).__name__}"],
                                            "attempted_utc": attempted_utc})
                        attempts_summary.append(_summary(
                            carrier, day, segment, station_id, provider, row, status="ERROR",
                            selected_nslc=nslc, refs=[],
                            reason=[f"FETCH_ERROR:{type(exc).__name__}"],
                            terminal_attempt_id=attempt_id, indeterminate=indeterminate,
                            attempted_utc=attempted_utc))
                        continue
                    # -- fetch success: durably stage each raw object into the closure -------
                    stream, raw_objects = res["stream"], res["raw_objects"]
                    obj_refs, refs = [], []
                    for ro in raw_objects:
                        with open(ro["staged_path"], "rb") as fh:
                            body = fh.read()
                        staged = stage_raw_atomic(body, raw_dir)
                        if staged["sha256"] != ro["sha256"] or staged["size_bytes"] != ro["size_bytes"]:
                            raise ResumeIntegrityError(
                                "provider-declared size/sha does not match the served bytes")
                        obj_refs.append({"relative_path": staged["relative_path"],
                                         "sha256": staged["sha256"],
                                         "size_bytes": staged["size_bytes"]})
                        refs.append(staged["sha256"])
                    append_event(root, {"kind": "UNIT_TERMINAL", "unit_id": uid,
                                        "attempt_id": attempt_id, "status": "FETCHED",
                                        "process_id": proc_id, "selected_nslc": nslc,
                                        "objects": obj_refs, "reason_codes": [],
                                        "attempted_utc": attempted_utc})
                    for obj in obj_refs:
                        durable = os.path.join(root, obj["relative_path"])
                        prov = CR._object_provenance(providers, durable)
                        if obj["sha256"] in legacy_shas:
                            reproduced_legacy.add(obj["sha256"])
                            disposition = "LEGACY_REUSED_AFTER_REFETCH_ATTESTATION"
                        else:
                            disposition = "FETCHED_NEW"
                        input_objects.append(_input_object(
                            obj, carrier, day, segment, nslc, row, record_sha, prov,
                            reuse=disposition, acquired=proc_id, verified=proc_id))
                    day_refs.extend(refs)
                    es = CR._station_series(SD, stream, nslc, start)
                    seg_station_es[segment].append((nslc, es))
                    attempts_summary.append(_summary(
                        carrier, day, segment, station_id, provider, row, status="FETCHED",
                        selected_nslc=nslc, refs=refs, reason=[],
                        terminal_attempt_id=attempt_id, indeterminate=indeterminate,
                        attempted_utc=attempted_utc))
            base = CR._score_day(SD, FC, seg_station_es, start)
            base["publication_record_sha256"] = record_sha
            base["input_object_sha256s"] = day_refs
            day_result[(carrier, day)] = base

    # -- PROCESS_ENDED: this segment closed gracefully -------------------------------
    append_event(root, {"kind": "PROCESS_ENDED", "process_id": proc_id,
                        "process_ended_utc": _iso(tick())})

    # -- R4: un-reproduced UNATTESTED_LEGACY -> sibling quarantine (out of closure) ---
    unreproduced = legacy_shas - reproduced_legacy
    if unreproduced:
        quarantine = os.fspath(root) + _LEGACY_QUARANTINE_SUFFIX
        os.makedirs(quarantine, exist_ok=True)
        for sha in unreproduced:
            src = os.path.join(raw_dir, sha + ".ms")
            if os.path.exists(src):
                os.replace(src, os.path.join(quarantine, sha + ".ms"))

    result = _assemble_and_finalize(root, plan, ledger, receipt, tick, campaign_id, plan_bytes,
                                    ledger_bytes, attempts_summary, input_objects, day_result,
                                    prior_started, prior_ended, observed_dead, CR)
    _cleanup_scratch(scratch_dir)
    return result


def _load_ledger_dict(root):
    with open(os.path.join(root, "published_phase_ledger.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _input_object(obj, carrier, day, segment, nslc, row, record_sha, prov, *,
                  reuse, acquired, verified):
    return {"sha256": obj["sha256"], "size": obj["size_bytes"],
            "relative_path": obj["relative_path"],
            "kind": "archive-seismic-miniseed-fragments-v1",
            "carrier_key": carrier, "scored_day": day, "segment_name": segment,
            "source_id": nslc, "start_utc": row["request_start_utc"],
            "end_utc": row["request_end_utc"], "native_rate_hz": prov["native_rate_hz"],
            "npts": prov["npts"], "fragment_count": prov["fragment_count"],
            "support_sha256": prov["support_sha256"], "publication_record_sha256": record_sha,
            "reuse_disposition": reuse, "acquired_process_id": acquired,
            "verified_process_id": verified}


def _cleanup_scratch(scratch_dir):
    try:
        for name in os.listdir(scratch_dir):
            try:
                os.remove(os.path.join(scratch_dir, name))
            except OSError:
                pass
        os.rmdir(scratch_dir)
    except OSError:
        pass


def _assemble_and_finalize(root, plan, ledger, receipt, tick, campaign_id, plan_bytes,
                           ledger_bytes, attempts, input_objects, day_result,
                           prior_started, prior_ended, observed_dead, CR):
    """Deterministic assembly: calibration_daily + admission derive ONLY from day_result / the
    accepted sealed diagnostic bytes, so an uninterrupted and an interrupted+resumed run are
    byte-identical here (RK4a). The process ledger + resume_state carry the honest multi-process
    provenance; batch_manifest is written LAST and binds every root file (RK5)."""
    import platform
    import numpy as np

    incident_days = list(plan["incident_days"])
    activation_days = list(plan["activation_days"])

    # -- process ledger (one row per PROCESS_STARTED ever committed) -----------------
    final_events = load_resume_state(root)["events"]
    ended_map = {ev["process_id"]: ev for ev in final_events if ev.get("kind") == "PROCESS_ENDED"}
    process_rows = []
    for ev in final_events:
        if ev.get("kind") != "PROCESS_STARTED":
            continue
        pid = ev["process_id"]
        end_ev = ended_map.get(pid)
        process_rows.append({"process_id": pid, "process_started_utc": ev["process_started_utc"],
                             "campaign_id": ev.get("campaign_id"),
                             "process_ended_utc": end_ev["process_ended_utc"] if end_ev else None,
                             "observed_dead_utc": observed_dead.get(pid)})
    campaign_started_utc = min(r["process_started_utc"] for r in process_rows)
    CR._write_jsonl(os.path.join(root, "campaign_process_ledger.jsonl"), process_rows)

    # -- calibration_daily (2 arms x 3 carriers x arm_days) + operation ledger -------
    rates = {obj["sha256"]: obj["native_rate_hz"] for obj in input_objects}
    daily_rows, operations = [], []
    for arm, arm_days in (("incident", incident_days), ("activation", activation_days)):
        for carrier in CR.ELIGIBLE:
            for day in arm_days:
                base = day_result.get((carrier, day))
                row = {"arm": arm, "carrier_key": carrier, "day": day}
                row.update(base if base is not None else CR._no_record_daily())
                daily_rows.append(row)
                if row["status"] == "ADMITTED":
                    operations.extend(CR._operation_rows(arm, carrier, day, row, rates))

    # -- prior evidence + replay (from the accepted sealed diagnostic bytes) ----------
    with open(CR.DIAGNOSTIC_FIXTURE, "rb") as fh:
        diag_bytes = fh.read()
    diagnostic = json.loads(diag_bytes.decode("utf-8"))
    diag_rel = "d2_diagnostic_result.json"
    with open(os.path.join(root, diag_rel), "wb") as fh:
        fh.write(diag_bytes)
    prior_evidence = {"schema": "geospec-d2-prior-evidence-v1", "diagnostic_result_path": diag_rel,
                      "non_promotional": True,
                      "lane_a_manifest_sha256": CR.PRIOR["lane_a_manifest_sha256"],
                      "diagnostic_manifest_sha256": CR.PRIOR["diagnostic_manifest_sha256"],
                      "diagnostic_result_sha256": CR.PRIOR["diagnostic_result_sha256"]}

    def _phase_record(carrier, phase, scored_day):
        src = diagnostic.get("results", {}).get(carrier, {}).get(phase, {})
        if src.get("status") == "OK":
            return {"scored_day": scored_day, "status": "ADMITTED",
                    "ratio": src.get("lambda2_lambda1"),
                    "common_support_count": src.get("common_support_count"),
                    "correlation_matrix_sha256": CR._matrix_sha(src.get("correlation_matrix")),
                    "qc_reasons": []}
        return {"scored_day": scored_day, "status": "UNAVAILABLE", "ratio": None,
                "common_support_count": None, "correlation_matrix_sha256": None,
                "qc_reasons": ["NO_ACCEPTED_DIAGNOSTIC_FOR_CARRIER"]}

    replay_regions = []
    for target in CR.TARGETS:
        carrier = target["carrier_key"]
        replay_regions.append({"runner_key": target["runner_key"], "carrier_key": carrier,
                               "incident": _phase_record(carrier, "incident", "2026-07-29"),
                               "control": _phase_record(carrier, "control", "2026-07-28")})
    replay_metrics = {"schema": "geospec-d2-segmented-replay-v1",
                      "implementation_commit": CR.IMPLEMENTATION_COMMIT,
                      "prior_evidence_sha256": CR._sha256_bytes(CR._canon(prior_evidence)),
                      "incident_scored_day": "2026-07-29", "control_scored_day": "2026-07-28",
                      "regions": replay_regions}
    replay_by_runner = {r["runner_key"]: r for r in replay_regions}

    # -- thresholds + admission (mirrors codex 0123 deterministically) ---------------
    def _threshold(arm, carrier):
        vals = sorted(float(r["ratio"]) for r in daily_rows
                      if r["arm"] == arm and r["carrier_key"] == carrier
                      and r["status"] == "ADMITTED" and isinstance(r.get("ratio"), (int, float)))
        if len(vals) < CR.POLICY["min_admitted_days"]:
            return None, len(vals)
        import math
        return vals[max(0, math.ceil(CR.POLICY["lower_quantile"] * len(vals)) - 1)], len(vals)

    incident_ref = datetime.strptime("2026-07-29", "%Y-%m-%d").date()
    activation_ref = datetime.strptime(plan["activation_reference_day"], "%Y-%m-%d").date()
    incident_window = {"start": incident_days[0],
                       "end": (incident_ref - timedelta(days=30)).isoformat()}
    activation_window = {"start": activation_days[0],
                         "end": (activation_ref - timedelta(days=30)).isoformat()}
    valid_through_date = activation_ref + timedelta(days=CR.POLICY["candidate_valid_days"])
    expiry_utc = datetime(valid_through_date.year, valid_through_date.month,
                          valid_through_date.day, tzinfo=timezone.utc) + timedelta(days=1)
    admissions, registry = [], {}
    for target in CR.TARGETS:
        runner = target["runner_key"]
        carrier = target["carrier_key"]
        base_row = {"runner_key": runner, "carrier_key": carrier, "topology_ok": True,
                    "topology_reasons": [], "incident_calibration_window": incident_window,
                    "activation_calibration_window": activation_window, "incident_n": 0,
                    "activation_n": 0, "incident_threshold": None, "activation_threshold": None,
                    "artifact_removed": False, "control_clear": False, "capsule_path": None,
                    "capsule_sha256": None, "reason_codes": []}
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
            base_row["_capsule"] = {
                "schema": "geospec-d2-calibration-v1", "region": carrier, "band_tag": CR.BAND_TAG,
                "processing_version": CR.PROCESSING_VERSION, "topology_version": CR.TOPOLOGY_VERSION,
                "threshold": act_thr, "calibration_window": activation_window,
                "source_commit": CR.IMPLEMENTATION_COMMIT, "input_manifest_sha256": None,
                "replay_output_sha256": CR._sha256_bytes(CR._canon(replay_metrics)),
                "valid_through": valid_through_date.isoformat()}
            base_row["_capsule_carrier"] = carrier
        admissions.append(base_row)

    # -- input_manifest (resume schema: reuse_disposition + acquired/verified process) -
    producer_commit = CR._git_head()
    clean_tree = CR._git_clean()
    input_manifest = {"schema": "geospec-d2-step4b-input-manifest-v2-resume",
                      "producer_commit": producer_commit,
                      "implementation_commit": CR.IMPLEMENTATION_COMMIT, "objects": input_objects}
    CR._write_json(os.path.join(root, "input_manifest.json"), input_manifest)
    input_manifest_sha = CR._sha256_file(os.path.join(root, "input_manifest.json"))

    for row in admissions:
        capsule = row.pop("_capsule", None)
        carrier = row.pop("_capsule_carrier", None)
        if capsule is None:
            continue
        issued_utc = tick()
        if issued_utc >= expiry_utc:
            row.update({"status": "BLOCKED_CANDIDATE_WINDOW_EXPIRED",
                        "reason_codes": ["CANDIDATE_WINDOW_EXPIRED"], "capsule_path": None,
                        "capsule_sha256": None})
            continue
        capsule["issued_utc"] = _iso(issued_utc)
        capsule["input_manifest_sha256"] = input_manifest_sha
        rel = f"capsules/{carrier}_calibration.json"
        os.makedirs(os.path.join(root, "capsules"), exist_ok=True)
        CR._write_json(os.path.join(root, rel), capsule)
        capsule_sha = CR._sha256_file(os.path.join(root, rel))
        row["capsule_path"] = rel
        row["capsule_sha256"] = capsule_sha
        registry[carrier] = {"capsule_path": rel, "expected_sha256": capsule_sha,
                             "topology_version": CR.TOPOLOGY_VERSION}

    admission_results = {"schema": "geospec-d2-step4b-admission-results-v1",
                         "implementation_commit": CR.IMPLEMENTATION_COMMIT, "regions": admissions}

    CR._write_jsonl(os.path.join(root, "acquisition_attempts.jsonl"), attempts)
    CR._write_jsonl(os.path.join(root, "calibration_daily.jsonl"), daily_rows)
    CR._write_jsonl(os.path.join(root, "operation_ledger.jsonl"), operations)
    CR._write_json(os.path.join(root, "prior_evidence.json"), prior_evidence)
    CR._write_json(os.path.join(root, "replay_metrics.json"), replay_metrics)
    CR._write_json(os.path.join(root, "admission_results.json"), admission_results)
    CR._write_json(os.path.join(root, "registry_candidate.json"), registry)

    # -- batch_manifest: written LAST, binds every root file (resume_state + ledger) --
    campaign_plan_sha = CR._sha256_file(os.path.join(root, "campaign_plan.json"))
    artifacts = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if rel == "batch_manifest.json":
                continue
            artifacts[rel] = {"sha256": CR._sha256_file(full), "size": os.path.getsize(full)}
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "numpy": np.__version__, "scipy": __import__("scipy").__version__,
                   "obspy": __import__("obspy").__version__}
    auth = {"status": receipt["status"],
            "in_session_timestamp_utc": receipt["in_session_timestamp_utc"],
            "owner_quote_sha256": receipt["owner_quote_sha256"]}
    created_utc = tick()
    candidates = [r["carrier_key"] for r in admissions if r["status"] == "ADMITTED_CANDIDATE"]
    batch_manifest = {"schema": "geospec-d2-step4b-batch-v1", "contract_id": CR.CONTRACT_ID,
                      "run_id": campaign_id, "campaign_id": campaign_id,
                      "producer_commit": producer_commit,
                      "implementation_commit": CR.IMPLEMENTATION_COMMIT, "clean_tree": clean_tree,
                      "band_tag": CR.BAND_TAG, "processing_version": CR.PROCESSING_VERSION,
                      "topology_version": CR.TOPOLOGY_VERSION,
                      "activation_reference_day": plan["activation_reference_day"],
                      "incident_reference_day": "2026-07-29", "calibration_policy": CR.POLICY,
                      "targets": CR.TARGETS, "environment": environment,
                      "implementation_blobs": CR.CORE_FILES, "artifacts": artifacts,
                      "campaign_started_utc": campaign_started_utc, "created_utc": _iso(created_utc),
                      "campaign_plan_sha256": campaign_plan_sha, "owner_launch_authorization": auth,
                      "resumable": True, "process_count": len(process_rows),
                      "production_registry_modified": False, "production_freezes_modified": False,
                      "non_claims": CR.NON_CLAIMS}
    CR._write_json(os.path.join(root, "batch_manifest.json"), batch_manifest)
    return {"status": "BATCH_STAGED", "root": root, "candidates": candidates,
            "attempts": len(attempts), "daily_rows": len(daily_rows), "resumed": bool(prior_started),
            "process_count": len(process_rows), "run_id": campaign_id,
            "clean_tree": clean_tree, "producer_commit": producer_commit}
