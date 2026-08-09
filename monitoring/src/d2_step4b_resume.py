#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""d2_step4b_resume.py — the D2 step-4b RESUMABILITY + CLOSURE layer (grassmann).

Implements the frozen cayley RK bar (`test_d2_step4b_resume_redkats_cayley.py` @ `3a829cd`,
RK1–RK5) AND the frozen resume-CLOSURE bar (`test_d2_step4b_resume_closure_redkats_cayley.py`
@ `55ae770`, RC1–RC5 = codex `1506` findings, schema pin A). Contract
`codex-d2-step4b-resume-2026-08-09-v1` + 1309.

SCHEMA PIN A (codex 1601 / cayley 1557): the ONLY standing D2 step-4b batch is the resume
executor's completed root. `run_campaign` with NO providers argument (production default) is
OWNED by this executor: the real provider module is imported LAZILY, strictly AFTER the durable
reopened PROCESS_STARTED; the completed batch carries `run_id == campaign_id_of(plan, ledger)`
and the resume artifacts. `campaign_run.acquire` remains the inner scientific scoring/assembly
step. Injected providers are the hermetic TEST seam only.

Integrity violations raise `ResumeIntegrityError` and NEVER degrade to data-unavailability or
mint/relabel a manifest. Nothing here reboots, performs real provider I/O, re-fires, mints,
lifts, or claims; the actual resume still waits on asylum's reboot timing + fresh direct go.

Seams (grassmann UNEDITED to the two bars):
  * ResumeIntegrityError(RuntimeError)
  * campaign_id_of / stage_raw_atomic / append_event / load_resume_state
  * verify_completed_root(root)                              [RC2]
  * run_resume_campaign(...) — the resume-aware executor (RK1..RK5 + RC1..RC5)
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
_QUARANTINE_SUFFIX = ".legacy_quarantine"
_FETCH_SCRATCH_SUFFIX = ".fetch_scratch"
_TEMP_PREFIXES = (".rs-", ".stage-", ".head-")
_TEMP_SUFFIXES = (".tmp", ".part")


class ResumeIntegrityError(RuntimeError):
    """An integrity violation (identity drift, tamper, corruption, escaping path, size/sha
    mismatch, truncation). NEVER degrades to data-unavailability, NEVER mints a manifest."""


# ---- primitives ------------------------------------------------------------
def _sha_hex(raw):
    return hashlib.sha256(raw).hexdigest()


def _canon_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def campaign_id_of(plan_bytes, ledger_bytes):
    """R0.1 exact construction over the RAW staged plan/ledger BYTES (binary digests)."""
    if not isinstance(plan_bytes, (bytes, bytearray)) or not isinstance(ledger_bytes, (bytes, bytearray)):
        raise ResumeIntegrityError("campaign_id_of requires plan/ledger BYTES")
    return hashlib.sha256(RESUME_CAMPAIGN_ID_PREFIX
                          + hashlib.sha256(bytes(plan_bytes)).digest()
                          + hashlib.sha256(bytes(ledger_bytes)).digest()).hexdigest()


def _atomic_write_bytes(path, data):
    """Same-dir temp + flush/fsync + atomic os.replace + best-effort directory fsync. On any
    failure the temp is removed and the exception re-raised (no partial install)."""
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
    _fsync_dir(dest_dir)


def _fsync_dir(path):
    try:
        dfd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dfd)
    except OSError:
        pass
    finally:
        os.close(dfd)


def _atomic_write_json(path, value):
    _atomic_write_bytes(path, _canon_bytes(value) + b"\n")


def _atomic_write_jsonl(path, rows):
    buf = b"".join(_canon_bytes(r) + b"\n" for r in rows)
    _atomic_write_bytes(path, buf)


# ---- referenced-path safety ------------------------------------------------
def _within_root(root, rel):
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


# ---- hash-chained event log + monotonic state-head (RC5) -------------------
def _event_canon(event):
    return _canon_bytes({k: v for k, v in event.items() if k != "event_sha256"})


def _event_digest(event):
    return _sha_hex(_event_canon(event))


def _resume_state_path(root):
    return os.path.join(root, "resume_state.json")


def _state_head_path(root):
    return os.path.join(root, "resume_state.head.json")


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


def _read_head(root):
    path = _state_head_path(root)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        raw = fh.read()
    try:
        head = json.loads(raw.decode("utf-8"))
    except Exception as exc:                                       # noqa: BLE001
        raise ResumeIntegrityError(f"resume_state.head.json is not valid JSON: {exc}")
    if not (isinstance(head, dict) and isinstance(head.get("event_count"), int)
            and isinstance(head.get("generation"), int)
            and isinstance(head.get("last_event_sha256"), str)):
        raise ResumeIntegrityError("resume_state.head.json malformed")
    return head


def _write_head(root, *, generation, event_count, last_event_sha256):
    _atomic_write_bytes(_state_head_path(root), _canon_bytes(
        {"schema": "geospec-d2-resume-state-head-v1", "generation": generation,
         "event_count": event_count, "last_event_sha256": last_event_sha256}) + b"\n")


def _validate_chain(root, events):
    """Full re-derivation: prev-hash linkage + recomputed event_sha256 + referenced-path safety.
    ANY break/tamper raises ResumeIntegrityError."""
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
    """Validated resume state. Full chain re-derivation PLUS monotonic state-head reconciliation:
      * head.event_count == len(events)      -> require last hash match (consistent);
      * head.event_count == len(events) - 1  -> WAL crash window (state one event ahead of head):
        recover — accept the longer state, repair the head, lose nothing;
      * len(events) < head.event_count       -> tail truncation -> ResumeIntegrityError.
    Surfaces the bound campaign_id (first PROCESS_STARTED) for drift checks."""
    events = _validate_chain(root, _read_events(root))
    head = _read_head(root)
    n = len(events)
    if head is None:
        if n == 0:
            return {"events": [], "campaign_id": None}
        raise ResumeIntegrityError("resume_state present without a state-head anchor")
    hc = head["event_count"]
    if n == hc:
        if n > 0 and events[-1]["event_sha256"] != head["last_event_sha256"]:
            raise ResumeIntegrityError("state-head last-hash mismatch")
    elif n == hc + 1:
        if hc > 0 and events[hc - 1]["event_sha256"] != head["last_event_sha256"]:
            raise ResumeIntegrityError("state-head WAL prefix mismatch")
        _write_head(root, generation=head["generation"] + 1, event_count=n,
                    last_event_sha256=events[-1]["event_sha256"])       # repair head
    else:
        raise ResumeIntegrityError(
            f"state-head/event-count mismatch (truncation?): events={n} head={hc}")
    campaign_id = None
    for ev in events:
        if ev.get("kind") == "PROCESS_STARTED":
            campaign_id = ev.get("campaign_id")
            break
    return {"events": events, "campaign_id": campaign_id}


def append_event(root, event):
    """Commit one event under the write-ahead protocol: validate the existing chain+head (WAL-
    recovering if needed), chain the new event, atomically replace resume_state.json FIRST, then
    advance the monotonic state-head. An escaping referenced path refuses."""
    if not isinstance(event, dict):
        raise ResumeIntegrityError("event must be a dict")
    state = load_resume_state(root)
    events = state["events"]
    prev = events[-1]["event_sha256"] if events else _GENESIS
    new_ev = {k: v for k, v in event.items() if k != "event_sha256"}
    new_ev["prev_event_sha256"] = prev
    _check_object_paths(root, new_ev)
    new_ev["event_sha256"] = _event_digest(new_ev)
    new_events = list(events) + [new_ev]
    head = _read_head(root)
    generation = (head["generation"] + 1) if head else 1
    _atomic_write_bytes(_resume_state_path(root), _canon_bytes({"events": new_events}))
    _write_head(root, generation=generation, event_count=len(new_events),
                last_event_sha256=new_ev["event_sha256"])
    return new_ev


# ---- durable raw-object staging -------------------------------------------
def stage_raw_atomic(raw, dest_dir):
    """R1.2 durable install at `dest_dir/<sha256>.ms` via unique temp + fsync + atomic rename. An
    EXISTING destination is verified byte-for-byte (must hash to its digest name) and REUSED
    WITHOUT write; mismatch -> ResumeIntegrityError. Returns {relative_path, sha256, size_bytes}."""
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
    return uuid.uuid4().hex + uuid.uuid4().hex


def _source_url(plan, carrier, nslc, row):
    prov = (plan.get("providers", {}) or {}).get(carrier, {}) or {}
    endpoint = prov.get("endpoint", "")
    provider = prov.get("provider", "")
    return (f"{str(provider).lower()}://{endpoint}/fdsnws/dataselect/1/query"
            f"?nslc={nslc}&start={row['request_start_utc']}&end={row['request_end_utc']}")


# ---- RC2: verified completed-root re-entry --------------------------------
def verify_completed_root(root):
    """A completed root is standing iff (all -> else ResumeIntegrityError):
      * run_id == campaign_id == campaign_id_of(reopened plan+ledger bytes) == batch.campaign_id;
      * the on-disk file set (excluding batch_manifest.json) EQUALS batch.artifacts exactly;
      * every artifact independently rehashes + re-sizes to its manifest entry;
      * resume_state + state-head + process ledger reopen and cross-link to the same campaign_id.
    Manifest PRESENCE is never acceptance."""
    bm_path = os.path.join(root, "batch_manifest.json")
    if not os.path.isfile(bm_path):
        raise ResumeIntegrityError("no batch_manifest.json to verify")
    with open(bm_path, "rb") as fh:
        try:
            bm = json.loads(fh.read().decode("utf-8"))
        except Exception as exc:                                   # noqa: BLE001
            raise ResumeIntegrityError(f"batch_manifest.json unreadable: {exc}")
    plan_path = os.path.join(root, "campaign_plan.json")
    ledger_path = os.path.join(root, "published_phase_ledger.json")
    if not (os.path.isfile(plan_path) and os.path.isfile(ledger_path)):
        raise ResumeIntegrityError("completed root missing plan/ledger")
    with open(plan_path, "rb") as fh:
        plan_bytes = fh.read()
    with open(ledger_path, "rb") as fh:
        ledger_bytes = fh.read()
    cid = campaign_id_of(plan_bytes, ledger_bytes)
    if bm.get("run_id") != cid or bm.get("campaign_id") != cid:
        raise ResumeIntegrityError("completed root run_id/campaign_id != recomputed campaign_id")
    artifacts = bm.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ResumeIntegrityError("batch_manifest has no artifacts map")
    on_disk = set()
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            rel = os.path.relpath(os.path.join(dirpath, name), root).replace(os.sep, "/")
            if rel != "batch_manifest.json":
                on_disk.add(rel)
    if on_disk != set(artifacts):
        missing = sorted(set(artifacts) - on_disk)[:4]
        extra = sorted(on_disk - set(artifacts))[:4]
        raise ResumeIntegrityError(f"artifact set != on-disk set (missing={missing} extra={extra})")
    for rel, meta in artifacts.items():
        full = os.path.join(root, rel)
        if not os.path.isfile(full):
            raise ResumeIntegrityError(f"manifest artifact missing on disk: {rel}")
        if os.path.getsize(full) != meta.get("size"):
            raise ResumeIntegrityError(f"artifact size mismatch: {rel}")
        with open(full, "rb") as fh:
            if _sha_hex(fh.read()) != meta.get("sha256"):
                raise ResumeIntegrityError(f"artifact hash mismatch: {rel}")
    if "resume_state.json" not in artifacts or "campaign_process_ledger.jsonl" not in artifacts \
            or "resume_state.head.json" not in artifacts:
        raise ResumeIntegrityError("completed root missing bound resume artifacts")
    load_resume_state(root)                                     # chain + head integrity
    ledger_rows = _read_jsonl(root, "campaign_process_ledger.jsonl")
    if not ledger_rows or bm.get("process_count") != len(ledger_rows):
        raise ResumeIntegrityError("process_count != process-ledger row count")
    for r in ledger_rows:
        if r.get("campaign_id") != cid:
            raise ResumeIntegrityError("process-ledger campaign_id cross-link mismatch")
    return True


def _read_jsonl(root, rel):
    with open(os.path.join(root, rel), "rb") as fh:
        return [json.loads(x) for x in fh.read().decode("utf-8").splitlines() if x.strip()]


# ============================================================================
# The resume-aware executor. Reached ONLY through the producer's receipt-gated
# run_campaign, under its writer lock, via _acquire.
# ============================================================================
def run_resume_campaign(plan, root, *, providers, receipt, clock=None):
    import d2_step4b_campaign_run as CR

    bm_path = os.path.join(root, "batch_manifest.json")
    if os.path.isfile(bm_path):
        verify_completed_root(root)                            # RC2: tampered root -> refusal
        with open(bm_path, "rb") as fh:
            bm = json.loads(fh.read().decode("utf-8"))
        return {"status": "BATCH_COMPLETE", "root": root, "run_id": bm.get("run_id"),
                "resumed": True, "candidates": bm.get("candidates", [])}

    with open(os.path.join(root, "campaign_plan.json"), "rb") as fh:
        plan_bytes = fh.read()
    with open(os.path.join(root, "published_phase_ledger.json"), "rb") as fh:
        ledger_bytes = fh.read()
    campaign_id = campaign_id_of(plan_bytes, ledger_bytes)
    return _execute(plan, root, providers, receipt, clock, campaign_id,
                    plan_bytes, ledger_bytes, CR)


def _execute(plan, root, providers, receipt, clock, campaign_id, plan_bytes, ledger_bytes, CR):
    import seismic_data as SD
    import fault_correlation as FC

    tick = clock if clock is not None else CR._default_clock()

    prior_events = load_resume_state(root)["events"]
    prior_ps = [e for e in prior_events if e.get("kind") == "PROCESS_STARTED"]
    prior_ended = {e["process_id"] for e in prior_events if e.get("kind") == "PROCESS_ENDED"}
    prior_observed = {e.get("dead_process_id") for e in prior_events
                      if e.get("kind") == "PROCESS_OBSERVED_DEAD"}
    if prior_events:
        bound = next((e.get("campaign_id") for e in prior_events
                      if e.get("kind") == "PROCESS_STARTED"), None)
        if bound != campaign_id:
            raise ResumeIntegrityError(
                "plan/ledger drift on a resumable root — no provider I/O, no manifest.")

    terminal_by_unit, attempts_by_unit = {}, {}
    for ev in prior_events:
        kind = ev.get("kind")
        if kind == "UNIT_ATTEMPT_STARTED":
            attempts_by_unit.setdefault(ev["unit_id"], []).append(ev)
        elif kind == "UNIT_TERMINAL":
            terminal_by_unit[ev["unit_id"]] = ev

    # -- PROCESS_STARTED under the (producer-held) writer lock, BEFORE provider import (1309) ----
    proc_id = uuid.uuid4().hex
    ordinal = len(prior_ps) + 1
    resume_of = prior_ps[-1]["process_id"] if prior_ps else None
    producer_commit = CR._git_head()
    append_event(root, {"kind": "PROCESS_STARTED", "process_id": proc_id, "ordinal": ordinal,
                        "resume_of": resume_of, "producer_commit": producer_commit,
                        "process_started_utc": _iso(tick()), "campaign_id": campaign_id,
                        "plan_sha256": _sha_hex(plan_bytes), "ledger_sha256": _sha_hex(ledger_bytes),
                        "owner_launch_authorization": receipt})
    load_resume_state(root)                                    # reopen: the start event is durable

    # -- first-observation death capsules for prior crashed processes (immutable) ---------------
    for ps in prior_ps:
        pid = ps["process_id"]
        if pid not in prior_ended and pid not in prior_observed:
            append_event(root, {"kind": "PROCESS_OBSERVED_DEAD", "dead_process_id": pid,
                                "observed_dead_utc": _iso(tick()), "process_id": proc_id,
                                "dead_ordinal": ps.get("ordinal")})

    # -- RC1: the REAL provider module imports LAZILY, strictly after the durable PROCESS_STARTED -
    if providers is None:
        import d2_step4b_providers as providers          # noqa: F811  (production default)

    raw_dir = os.path.join(root, "raw_objects")
    os.makedirs(raw_dir, exist_ok=True)
    # R4 legacy inventory: digest-named raw objects present WITHOUT any committed event are
    # UNATTESTED_LEGACY. A fresh authorized fetch whose bytes reproduce one attests it
    # (LEGACY_REUSED_AFTER_REFETCH_ATTESTATION); an un-reproduced one is quarantined at closure.
    attested_by_events = set()
    for ev in prior_events:
        for obj in (ev.get("objects") or []):
            attested_by_events.add(obj["sha256"])
    legacy_shas = {name[:-3] for name in os.listdir(raw_dir)
                   if name.endswith(".ms") and _HEX64.match(name[:-3] or "")
                   and name[:-3] not in attested_by_events}
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
        if carrier not in station_registry:
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
                    candidates = list(srow["ordered_nslc_candidates"])

                    if term is not None:
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
                                    raise ResumeIntegrityError("reused object size/sha mismatch (H3)")
                                prov = CR._object_provenance(providers, durable)
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

                    # -- nonterminal unit: a fresh attempt (new ordinal; prior danglers kept) ----
                    indeterminate = len(prior_attempts)
                    attempt_id = _new_attempt_id()
                    attempt_ordinal = indeterminate + 1
                    attempted_utc = _iso(tick())
                    append_event(root, {"kind": "UNIT_ATTEMPT_STARTED", "unit_id": uid,
                                        "attempt_id": attempt_id, "ordinal": attempt_ordinal,
                                        "status": "IN_PROGRESS", "process_id": proc_id,
                                        "carrier_key": carrier, "scored_day": day,
                                        "segment_name": segment, "station_id": station_id,
                                        "provider": provider,
                                        "ordered_nslc_candidates": candidates,
                                        "request_start_utc": row["request_start_utc"],
                                        "request_end_utc": row["request_end_utc"],
                                        "attempted_utc": attempted_utc})
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
                    base_evt = {"unit_id": uid, "attempt_id": attempt_id, "process_id": proc_id,
                                "carrier_key": carrier, "scored_day": day, "segment_name": segment,
                                "station_id": station_id, "provider": provider,
                                "request_start_utc": row["request_start_utc"],
                                "request_end_utc": row["request_end_utc"],
                                "publication_record_sha256": record_sha}
                    if nslc is None:
                        append_event(root, dict(base_evt, kind="UNIT_TERMINAL", status="UNAVAILABLE",
                                                selected_nslc=None, source_url=None, objects=[],
                                                reason_codes=["NO_AVAILABLE_CANDIDATE"],
                                                terminal_utc=_iso(tick())))
                        attempts_summary.append(_summary(
                            carrier, day, segment, station_id, provider, row, status="UNAVAILABLE",
                            selected_nslc=None, refs=[], reason=["NO_AVAILABLE_CANDIDATE"],
                            terminal_attempt_id=attempt_id, indeterminate=indeterminate,
                            attempted_utc=attempted_utc))
                        continue
                    source_url = _source_url(plan, carrier, nslc, row)
                    # UNIT_SOURCE_SELECTED committed BEFORE the corresponding fetch (RC3) ----------
                    append_event(root, {"kind": "UNIT_SOURCE_SELECTED", "unit_id": uid,
                                        "attempt_id": attempt_id, "process_id": proc_id,
                                        "selected_nslc": nslc, "source_url": source_url,
                                        "provider": provider, "carrier_key": carrier,
                                        "scored_day": day, "segment_name": segment,
                                        "station_id": station_id,
                                        "selected_utc": _iso(tick())})
                    try:
                        res = providers.fetch(provider, nslc, start, end, stage_dir=scratch_dir)
                    except providers.ProviderUnavailable:
                        append_event(root, dict(base_evt, kind="UNIT_TERMINAL", status="UNAVAILABLE",
                                                selected_nslc=nslc, source_url=source_url, objects=[],
                                                reason_codes=["PROVIDER_UNAVAILABLE"],
                                                terminal_utc=_iso(tick())))
                        attempts_summary.append(_summary(
                            carrier, day, segment, station_id, provider, row, status="UNAVAILABLE",
                            selected_nslc=nslc, refs=[], reason=["PROVIDER_UNAVAILABLE"],
                            terminal_attempt_id=attempt_id, indeterminate=indeterminate,
                            attempted_utc=attempted_utc))
                        continue
                    except Exception as exc:                          # noqa: BLE001
                        append_event(root, dict(base_evt, kind="UNIT_TERMINAL", status="ERROR",
                                                selected_nslc=nslc, source_url=source_url, objects=[],
                                                reason_codes=[f"FETCH_ERROR:{type(exc).__name__}"],
                                                terminal_utc=_iso(tick())))
                        attempts_summary.append(_summary(
                            carrier, day, segment, station_id, provider, row, status="ERROR",
                            selected_nslc=nslc, refs=[],
                            reason=[f"FETCH_ERROR:{type(exc).__name__}"],
                            terminal_attempt_id=attempt_id, indeterminate=indeterminate,
                            attempted_utc=attempted_utc))
                        continue
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
                    append_event(root, dict(base_evt, kind="UNIT_TERMINAL", status="FETCHED",
                                            selected_nslc=nslc, source_url=source_url,
                                            objects=obj_refs, reason_codes=[],
                                            terminal_utc=_iso(tick())))
                    for obj in obj_refs:
                        durable = os.path.join(root, obj["relative_path"])
                        prov = CR._object_provenance(providers, durable)
                        disposition = ("LEGACY_REUSED_AFTER_REFETCH_ATTESTATION"
                                       if obj["sha256"] in legacy_shas else "FETCHED_NEW")
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

    append_event(root, {"kind": "PROCESS_ENDED", "process_id": proc_id,
                        "process_ended_utc": _iso(tick())})

    result = _assemble_and_finalize(root, plan, ledger, receipt, tick, campaign_id, plan_bytes,
                                    ledger_bytes, attempts_summary, input_objects, day_result,
                                    bool(prior_ps), CR)
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


def _clean_stray_temps(root):
    """Remove any leftover atomic-write temps in root + raw_objects (e.g. a manifest temp left by
    a death AT the final replace) so the typed closure has no non-artifact residue."""
    for d in (root, os.path.join(root, "raw_objects")):
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if name.startswith(_TEMP_PREFIXES) or name.endswith(_TEMP_SUFFIXES):
                try:
                    os.remove(os.path.join(d, name))
                except OSError:
                    pass


def _quarantine_untyped_raw(root, valid_shas):
    """RC4: build the raw-object closure from TYPED receipts, not os.walk. A raw file is retained
    iff its name is `<sha256>.ms` AND that sha is cross-linked to an input receipt; every other
    file (a non-digest orphan like junk.ms, or an un-reproduced legacy byte) is moved to the
    SIBLING `root + '.legacy_quarantine/'`, outside the closure, never manifested."""
    raw_dir = os.path.join(root, "raw_objects")
    if not os.path.isdir(raw_dir):
        return
    quarantine = os.fspath(root) + _QUARANTINE_SUFFIX
    for name in os.listdir(raw_dir):
        full = os.path.join(raw_dir, name)
        if not os.path.isfile(full):
            continue
        keep = (name.endswith(".ms") and _HEX64.match(name[:-3] or "")
                and name[:-3] in valid_shas)
        if not keep:
            os.makedirs(quarantine, exist_ok=True)
            os.replace(full, os.path.join(quarantine, name))


def _assemble_and_finalize(root, plan, ledger, receipt, tick, campaign_id, plan_bytes,
                           ledger_bytes, attempts, input_objects, day_result, resumed, CR):
    """Deterministic assembly (calibration_daily + admission derive ONLY from day_result / the
    sealed diagnostic bytes -> uninterrupted and resumed runs are byte-identical; RK4a). Every
    artifact is written atomically, closure is built from TYPED outputs, batch_manifest is written
    LAST and binds every root file including resume_state + state-head + process ledger (RC4/RC5)."""
    import platform
    import numpy as np

    incident_days = list(plan["incident_days"])
    activation_days = list(plan["activation_days"])

    # -- honest multi-process ledger from the reopened chain (RC3) -------------------
    final_events = load_resume_state(root)["events"]
    ps_events = [e for e in final_events if e.get("kind") == "PROCESS_STARTED"]
    ended_map = {e["process_id"]: e for e in final_events if e.get("kind") == "PROCESS_ENDED"}
    obs_map = {}
    for e in final_events:
        if e.get("kind") == "PROCESS_OBSERVED_DEAD":
            obs_map.setdefault(e.get("dead_process_id"), e)      # first observation only
    events_by_proc = {}
    for e in final_events:
        events_by_proc.setdefault(e.get("process_id"), []).append(e)
    process_rows = []
    for ps in ps_events:
        pid = ps["process_id"]
        ended = ended_map.get(pid)
        obs = obs_map.get(pid)
        proc_events = events_by_proc.get(pid, [])
        process_rows.append({
            "process_id": pid, "ordinal": ps.get("ordinal"), "resume_of": ps.get("resume_of"),
            "producer_commit": ps.get("producer_commit"), "campaign_id": ps.get("campaign_id"),
            "owner_launch_authorization": ps.get("owner_launch_authorization"),
            "process_started_utc": ps.get("process_started_utc"),
            "process_ended_utc": ended["process_ended_utc"] if ended else None,
            "observed_dead_utc": obs["observed_dead_utc"] if obs else None,
            "disposition": "COMPLETED" if ended else "CRASHED",
            "failure_type": None if ended else "PROCESS_DEATH",
            "first_event_sha256": proc_events[0]["event_sha256"] if proc_events else None,
            "last_event_sha256": proc_events[-1]["event_sha256"] if proc_events else None})
    campaign_started_utc = min(r["process_started_utc"] for r in process_rows)
    _atomic_write_jsonl(os.path.join(root, "campaign_process_ledger.jsonl"), process_rows)

    # -- calibration_daily + operation ledger ----------------------------------------
    rates = {obj["sha256"]: obj["native_rate_hz"] for obj in input_objects}
    daily_rows, operations = [], []
    for arm, arm_days in (("incident", incident_days), ("activation", activation_days)):
        for carrier in CR.ELIGIBLE:
            for day in arm_days:
                base = day_result.get((carrier, day))
                dr = {"arm": arm, "carrier_key": carrier, "day": day}
                dr.update(base if base is not None else CR._no_record_daily())
                daily_rows.append(dr)
                if dr["status"] == "ADMITTED":
                    operations.extend(CR._operation_rows(arm, carrier, day, dr, rates))

    # -- prior evidence + replay (from the accepted sealed diagnostic bytes) ----------
    with open(CR.DIAGNOSTIC_FIXTURE, "rb") as fh:
        diag_bytes = fh.read()
    diagnostic = json.loads(diag_bytes.decode("utf-8"))
    _atomic_write_bytes(os.path.join(root, "d2_diagnostic_result.json"), diag_bytes)
    prior_evidence = {"schema": "geospec-d2-prior-evidence-v1",
                      "diagnostic_result_path": "d2_diagnostic_result.json", "non_promotional": True,
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
        runner, carrier = target["runner_key"], target["carrier_key"]
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
                                and inc_rec.get("ratio") is not None and inc_rec["ratio"] >= inc_thr)
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

    # -- typed closure: quarantine untyped raw + clean stray temps BEFORE binding -----
    _clean_stray_temps(root)
    _quarantine_untyped_raw(root, {obj["sha256"] for obj in input_objects})

    producer_commit = CR._git_head()
    clean_tree = CR._git_clean()
    input_manifest = {"schema": "geospec-d2-step4b-input-manifest-v2-resume",
                      "producer_commit": producer_commit,
                      "implementation_commit": CR.IMPLEMENTATION_COMMIT, "objects": input_objects}
    _atomic_write_json(os.path.join(root, "input_manifest.json"), input_manifest)
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
        _atomic_write_json(os.path.join(root, rel), capsule)
        capsule_sha = CR._sha256_file(os.path.join(root, rel))
        row["capsule_path"] = rel
        row["capsule_sha256"] = capsule_sha
        registry[carrier] = {"capsule_path": rel, "expected_sha256": capsule_sha,
                             "topology_version": CR.TOPOLOGY_VERSION}

    admission_results = {"schema": "geospec-d2-step4b-admission-results-v1",
                         "implementation_commit": CR.IMPLEMENTATION_COMMIT, "regions": admissions}

    _atomic_write_jsonl(os.path.join(root, "acquisition_attempts.jsonl"), attempts)
    _atomic_write_jsonl(os.path.join(root, "calibration_daily.jsonl"), daily_rows)
    _atomic_write_jsonl(os.path.join(root, "operation_ledger.jsonl"), operations)
    _atomic_write_json(os.path.join(root, "prior_evidence.json"), prior_evidence)
    _atomic_write_json(os.path.join(root, "replay_metrics.json"), replay_metrics)
    _atomic_write_json(os.path.join(root, "admission_results.json"), admission_results)
    _atomic_write_json(os.path.join(root, "registry_candidate.json"), registry)

    # -- batch_manifest: written LAST (atomic), binds every root file -----------------
    _clean_stray_temps(root)
    campaign_plan_sha = CR._sha256_file(os.path.join(root, "campaign_plan.json"))
    head = _read_head(root)
    artifacts = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if rel == "batch_manifest.json":
                continue
            if rel.startswith("raw_objects/"):
                base_name = os.path.basename(rel)
                sha = base_name[:-3] if base_name.endswith(".ms") else ""
                if not (_HEX64.match(sha or "") and CR._sha256_file(full) == sha):
                    raise ResumeIntegrityError(f"untyped raw object survived closure: {rel}")
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
                      "state_head_generation": head["generation"] if head else 0,
                      "production_registry_modified": False, "production_freezes_modified": False,
                      "non_claims": CR.NON_CLAIMS}
    _atomic_write_json(os.path.join(root, "batch_manifest.json"), batch_manifest)
    return {"status": "BATCH_STAGED", "root": root, "candidates": candidates,
            "attempts": len(attempts), "daily_rows": len(daily_rows), "resumed": resumed,
            "process_count": len(process_rows), "run_id": campaign_id,
            "clean_tree": clean_tree, "producer_commit": producer_commit}
