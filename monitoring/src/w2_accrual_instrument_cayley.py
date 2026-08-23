#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 PRODUCTION ACCRUAL INSTRUMENT, core (cayley) -- the live
wrapper around the w2_barrier ledger. This module is the ONLY place
window-2 production code reads a clock; every barrier operation flows
through the file-backed, hash-chained, append-only ledger; and nothing
executes until the RUNTIME ALLOWLIST walk passes (every BOUND
execution-manifest pin's on-disk bytes must equal its pinned blob,
CRLF->LF normalized -- the same executed-bytes discipline as the
execution verifier, applied to the DISK the instrument runs from).

Pin-independent core (built while grassmann's batch REV ratifies the
engine interpretation pins): persistence, chain verification, state
reconstruction FROM EVENTS (single source of truth), allowlist walk,
live-clock injection. The seam-binding calls (selection execution,
adapter runs, per-lane accrual) layer on top once the bar REV + codex
seam close land.

Persistence model: JSON file {meta: {lease, used_leases}, events: [...]}.
On load the chain is re-verified and ALL ledger state (window dates,
admitted lanes, predictions, seals, terminals, verifier passes,
first-fired, state) is REBUILT by replaying events -- a doctored file
either breaks the chain (LEDGER_CHAIN_BROKEN) or is faithfully
reflected, never silently merged. Refusal semantics live entirely in
w2_barrier; this wrapper adds no policy of its own.

No live PRESTART is performed by importing or KAT-ing this module;
PRESTART requires the full sequence (bars green, manifest CLOSED,
codex round, owner authorization). This module opens no window-2 value.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import w2_barrier as WB

EXEC_MANIFEST_PATH = "docs/f2g_window2_execution/execution_manifest.json"


class InstrumentRefusal(ValueError):
    """Typed refusal; the code leads the message."""


def now_utc_day():
    """THE clock read (live UTC day). Nothing else in window-2
    production reads a clock."""
    return time.strftime("%Y-%m-%d", time.gmtime())


def now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _norm_py(raw):
    return raw.replace(b"\r\n", b"\n")


def _git(repo, args, binary=False):
    p = subprocess.run(["git", "-C", repo] + args, capture_output=True)
    if p.returncode != 0:
        raise InstrumentRefusal(
            f"GIT_READ_FAILED: {' '.join(args)[:80]}")
    return p.stdout if binary else p.stdout.decode("utf-8",
                                                   "replace").strip()


def runtime_allowlist_check(repo, manifest_commit):
    """Every BOUND pin's ON-DISK bytes must equal the pinned blob
    (CRLF->LF normalized). Returns the checked-pin report; raises
    RUNTIME_ALLOWLIST_VIOLATION naming every divergent path. The
    manifest itself is read from the git object at the stated commit,
    never from disk."""
    raw = _git(repo, ["cat-file", "blob",
                      f"{manifest_commit}:{EXEC_MANIFEST_PATH}"],
               binary=True)
    manifest = json.loads(raw.decode("utf-8"))
    if manifest.get("schema") != "f2g-window2-execution-manifest-v1.2":
        raise InstrumentRefusal("RUNTIME_ALLOWLIST_VIOLATION: "
                                "manifest schema mismatch")
    checked = []
    divergent = []
    for slot_name in sorted(manifest["slots"]):
        slot = manifest["slots"][slot_name]
        if slot["status"] != "BOUND":
            continue
        for pin in slot["pins"]:
            disk = os.path.join(repo, pin["path"].replace("/", os.sep))
            if not os.path.exists(disk):
                divergent.append((slot_name, pin["path"], "ABSENT"))
                continue
            with open(disk, "rb") as f:
                got = hashlib.sha256(_norm_py(f.read())).hexdigest()
            if got != pin["blob_sha256"]:
                divergent.append((slot_name, pin["path"],
                                  f"{got[:12]}!={pin['blob_sha256'][:12]}"))
            else:
                checked.append((slot_name, pin["path"]))
    if divergent:
        raise InstrumentRefusal(
            f"RUNTIME_ALLOWLIST_VIOLATION: {divergent}")
    return {"manifest_commit": manifest_commit,
            "manifest_state": manifest["manifest_state"],
            "pins_checked": len(checked), "pins": checked}


STAGED_INVENTORY_BASENAME = "staged_body_inventory.json"
STORE_DESCRIPTOR_BASENAME = "store_descriptor.json"


def _pinned_json(slot, basename, blob_reader):
    pins = [p for p in slot.get("pins", ())
            if isinstance(p, dict) and str(p.get("path", ""))
            .endswith(basename)]
    if len(pins) != 1:
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: producer_boundary is BOUND "
            f"without exactly one {basename} pin (found {len(pins)})")
    pin = pins[0]
    raw = blob_reader(pin["commit"], pin["path"])
    if hashlib.sha256(raw).hexdigest() != pin["blob_sha256"]:
        raise InstrumentRefusal(
            f"PRESTART_ADMISSION_REFUSED: {basename} bytes diverge "
            "from the manifest pin")
    return json.loads(raw.decode("utf-8"))


def verify_staged_store(repo, manifest, *, blob_reader=None,
                        inventory_verifier=None):
    """producer-boundary amendment v1.1 appendix (codex 1843Z item 4
    + 2015Z item 1): when the producer_boundary slot is BOUND, BOTH
    the staged_body_inventory AND the registered store DESCRIPTOR
    reopen from their manifest pins, and grassmann's REV 7 NAMED-STORE
    verifier reopens EVERY object -- the physical root comes only from
    the pinned descriptor mapping (there is no caller path); an
    inventory hash is never mistaken for completed-build availability;
    an unavailable or wrong store is a TYPED refusal, never PASS.
    Returns None while the slot is honestly OPEN (the zero-OPEN
    prestart gate refuses upstream). blob_reader/inventory_verifier
    are injectable for KATs only."""
    slot = manifest["slots"].get("producer_boundary")
    if not isinstance(slot, dict) or slot.get("status") != "BOUND":
        return None
    if blob_reader is None:
        def blob_reader(commit, path):
            return _git(repo, ["cat-file", "blob",
                               f"{commit}:{path}"], binary=True)
    inventory = _pinned_json(slot, STAGED_INVENTORY_BASENAME,
                             blob_reader)
    descriptor = _pinned_json(slot, STORE_DESCRIPTOR_BASENAME,
                              blob_reader)
    if inventory_verifier is None:
        import w2_acquisition_capture_grassmann as CAP

        def inventory_verifier(inv, desc):
            return CAP.verify_staged_body_inventory(inv, desc)
    try:
        report = inventory_verifier(inventory, descriptor)
    except Exception as e:
        raise InstrumentRefusal(
            f"PRESTART_ADMISSION_REFUSED: staged store reopen "
            f"failed: {e}")
    return {"store_id": inventory.get("store_id"),
            "objects": len(inventory.get("objects", {})),
            "report": report}


def assemble_prestart_admission(repo, manifest_commit, bindings,
                                owner_authorization):
    """codex 1815Z item 1: builds the CLOSED admission capsule with
    LIVE verification. Refuses typed PRESTART_ADMISSION_REFUSED when:
    the execution verifier's --prestart walk is not a zero-OPEN PASS
    (today's OPEN manifests refuse here); the runtime allowlist fails;
    or the owner authorization is not a closed binding record whose
    binds match the EXACT manifest commit/blob, lanes, lease, and
    window uuid."""
    import f2g_execution_manifest_verifier_cayley as EMV

    def refuse(detail):
        raise InstrumentRefusal(f"PRESTART_ADMISSION_REFUSED: {detail}")
    verdict = EMV.verify(repo, manifest_commit, prestart=True)
    if verdict.get("verdict") != "PASS" or \
            verdict.get("slots_open", -1) != 0:
        refuse(f"execution verifier --prestart is not a zero-OPEN "
               f"PASS at {manifest_commit}: "
               f"{[t['reason'] for t in verdict.get('typed_reasons', [])][:4]}")
    allowlist = runtime_allowlist_check(repo, manifest_commit)
    raw = _git(repo, ["cat-file", "blob",
                      f"{verdict['manifest_commit']}:"
                      f"{EXEC_MANIFEST_PATH}"], binary=True)
    # v1.1 appendix gate (codex 1843Z item 4): the external staged
    # store must REOPEN at admission time -- runs whenever the
    # producer_boundary slot is BOUND (always true at a zero-OPEN
    # prestart PASS)
    verify_staged_store(repo, json.loads(raw.decode("utf-8")))
    blob_sha = hashlib.sha256(raw).hexdigest()
    if not isinstance(owner_authorization, dict) or \
            set(owner_authorization) != {"quote", "quote_sha256",
                                         "binds"}:
        refuse("owner authorization is not a closed binding record "
               "(bare strings refuse)")
    oa = owner_authorization
    if hashlib.sha256(str(oa["quote"]).encode()).hexdigest() != \
            oa["quote_sha256"]:
        refuse("owner quote digest mismatch")
    binds = oa["binds"]
    if binds.get("manifest_blob_sha256") != blob_sha or \
            str(binds.get("manifest_commit")) != str(manifest_commit):
        refuse("owner binding does not match the exact manifest "
               "commit/blob (manifest changed after the binding?)")
    if sorted(binds.get("lanes", ())) != \
            sorted(bindings.get("lane_uuids", ())) or \
            binds.get("lease") != bindings.get("remote_lease") or \
            binds.get("window_uuid") != \
            bindings.get("global_window_uuid"):
        refuse("owner binding diverges from the bindings' "
               "lanes/lease/window")
    admission = {"schema": WB.ADMISSION_SCHEMA,
                 "manifest_commit": str(manifest_commit),
                 "manifest_blob_sha256": blob_sha,
                 "prestart_verifier": {
                     "verdict": verdict["verdict"],
                     "mode": verdict["mode"],
                     "slots_open": verdict["slots_open"],
                     "manifest_commit": verdict["manifest_commit"]},
                 "allowlist": {"pins_checked":
                               allowlist["pins_checked"]},
                 "owner": {"quote": oa["quote"],
                           "quote_sha256": oa["quote_sha256"],
                           "binds": dict(binds)},
                 "lanes": list(bindings["lane_uuids"]),
                 "lease": bindings["remote_lease"],
                 "window_uuid": bindings["global_window_uuid"]}
    admission["admission_digest"] = WB.admission_digest(admission)
    return admission


class PersistentLedger:
    """File-backed w2_barrier.BarrierLedger: save-after-every-append,
    replay-on-load, chain verified both ways."""

    def __init__(self, path, used_leases=(), clock=now_utc_day):
        # `clock` is injectable FOR KATS ONLY; production always uses
        # the default live-UTC read
        self.path = os.path.abspath(path)
        self._clock = clock
        self.ledger = WB.BarrierLedger(used_leases=used_leases)
        if os.path.exists(self.path):
            self._load()

    # -------- persistence --------
    def save(self):
        doc = {"meta": {"lease": self.ledger._lease,
                        "used_leases":
                            sorted(self.ledger._used_leases)},
               "events": self.ledger.events}
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(doc, f, indent=1, sort_keys=True)
            f.write("\n")
        os.replace(tmp, self.path)

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        led = WB.BarrierLedger(
            used_leases=doc["meta"].get("used_leases", ()))
        led.events = doc["events"]
        led.verify_chain()          # tamper check BEFORE any replay
        self._replay(led)
        led._lease = doc["meta"].get("lease")
        self.ledger = led

    @staticmethod
    def _replay(led):
        """Rebuild ALL derived state from the (verified) event list --
        events are the single source of truth."""
        from datetime import date
        for ev in led.events:
            k, p = ev["kind"], ev["payload"]
            if k == "PRESTART":
                led.state = "ACCRUAL"
                led.admitted_lanes = frozenset(p["lanes"])
                led.evaluation_start = date.fromisoformat(
                    p["evaluation_start"])
                led.evaluation_end = date.fromisoformat(
                    p["evaluation_end"])
                led.maturity_tail_end = date.fromisoformat(
                    p["maturity_tail_end"])
                led._bindings_digest = p["bindings_digest"]
            elif k == "PREDICTION":
                led._predictions[(p["region"], p["issue_day"])] = \
                    p["row_digest"]
            elif k == "SUPPORT_BARRIER_CLOSED":
                led.state = "SUPPORT_BARRIER"
            elif k == "OWNER_SEAL":
                led._sealed_lanes.add(p["lane"])
            elif k == "FINAL_FIRE":
                led._first_fired = True
                led._terminal_lanes.add(p["lane"])
            elif k == "VERIFIER_PASS":
                led._verified_lanes.add(p["lane"])
            elif k == "RELEASE":
                led.state = "RELEASED"
            elif k == "WINDOW3_TERMINAL":
                led.state = "WINDOW3_TERMINAL"
            elif k == "SELECTOR_COMMITTED":
                led._selector = frozenset(p["S"])
                led._selector_power = p["power"]

    # -------- wrapped operations (live clock injected here) --------
    def prestart(self, bindings, admission):
        """Low-level pass-through: the ADMISSION CAPSULE is required
        (the barrier refuses bare bindings). Production callers use
        prestart_production, which BUILDS the capsule with live
        verification."""
        self.ledger.prestart(bindings, self._clock(), admission)
        self.save()

    def prestart_production(self, repo, manifest_commit, bindings,
                            owner_authorization):
        """THE production prestart entry point (codex 1815Z item 1):
        constructs and verifies the closed admission capsule LIVE --
        execution verifier in --prestart mode (must be a zero-OPEN
        PASS), runtime allowlist over those bytes, and the owner
        authorization independently verified against the exact
        manifest blob / lanes / lease / window -- then drives the
        barrier."""
        admission = assemble_prestart_admission(
            repo, manifest_commit, bindings, owner_authorization)
        self.prestart(bindings, admission)
        return admission

    def accrue_prediction(self, lease, region, issue_day, row_digest):
        self.ledger.accrue_prediction(lease, region, issue_day,
                                      row_digest, self._clock())
        self.save()

    def producer_receipt(self, lease, receipt_digest):
        self.ledger.producer_receipt(lease, receipt_digest)
        self.save()

    def close_support_barrier(self, lease, role):
        self.ledger.close_support_barrier(lease, self._clock(), role)
        self.save()

    def record_owner_seal(self, lease, lane, seal_digest):
        self.ledger.record_owner_seal(lease, lane, seal_digest)
        self.save()

    def final_fire(self, lease, lane, result_digest):
        self.ledger.final_fire(lease, lane, result_digest)
        self.save()

    def record_verifier_pass(self, lease, lane, verifier_digest):
        self.ledger.record_verifier_pass(lease, lane, verifier_digest)
        self.save()

    def release(self, lease):
        self.ledger.release(lease)
        self.save()

    def commit_selector(self, power_results):
        s = self.ledger.commit_selector(power_results)
        self.save()
        return s


# ===================================================================
# SEAM LAYER 1 (built on the RATIFIED barrier pins + codex-repaired
# selection REV 2; grassmann 0610Z: "accrual seam layers unblocked"):
# PRESTART binding assembly + selection execution. Later layers
# (per-lane accrual runners, adapter panel builds) follow their pins'
# formal ratification in the remaining bar cycles.
# ===================================================================
import w2_selection as WS

DESIGN_MANIFEST_PATH = "docs/f2g_window2_freeze/byte_pin_manifest.json"
W2_CARRIERS = ("istanbul_marmara", "socal_coachella",
               "turkey_kahramanmaras", "cascadia")


def assemble_prestart_bindings(repo, *, execution_manifest_commit,
                               mf4_model_scaler_digest,
                               power_envelope_digest,
                               global_window_uuid, remote_lease,
                               lane_uuids, owner_authorization,
                               hypothesis_registries_digest,
                               calibration_fits_digest,
                               adapters_digest):
    """Builds the eleven-class PRESTART bindings dict. What is
    resolvable from git is RESOLVED here (execution-manifest blob sha =
    code_manifest; the models class carries the design-manifest linkage
    blob sha); everything else is passed through and must be non-empty
    (the barrier re-validates on prestart -- this assembler adds no
    policy, only resolution)."""
    exec_blob = _git(repo, ["cat-file", "blob",
                            f"{execution_manifest_commit}:"
                            f"{EXEC_MANIFEST_PATH}"], binary=True)
    exec_obj = json.loads(exec_blob.decode("utf-8"))
    dm_commit = exec_obj["design_manifest_commit"]
    dm_blob = _git(repo, ["cat-file", "blob",
                          f"{dm_commit}:{DESIGN_MANIFEST_PATH}"],
                   binary=True)
    return {
        "code_manifest": {
            "execution_manifest_commit": execution_manifest_commit,
            "execution_manifest_blob_sha256":
                hashlib.sha256(exec_blob).hexdigest()},
        "models": {
            "design_manifest_commit": dm_commit,
            "design_manifest_blob_sha256":
                hashlib.sha256(dm_blob).hexdigest()},
        "calibration_fits": calibration_fits_digest,
        "hypothesis_registries": hypothesis_registries_digest,
        "adapters": adapters_digest,
        "mf4_model_scaler": mf4_model_scaler_digest,
        "power_envelope": power_envelope_digest,
        "global_window_uuid": global_window_uuid,
        "remote_lease": remote_lease,
        "lane_uuids": list(lane_uuids),
        "owner_authorization": owner_authorization,
    }


def execute_selection(day_records_by_carrier, cutoff):
    """The cutoff-stable selection execution: w2_selection PRODUCTION
    path per carrier (frozen caps; 90-day frame + presence derivation +
    churn all engine-enforced). A typed carrier refusal
    (INSUFFICIENT_POOL) is RECORDED, never a crash and never a silent
    drop; frame/input violations (SelectionInputInvalid) propagate --
    they are instrument-feed defects, not carrier outcomes. Returns the
    registry record with a digest for the PRESTART hypothesis-registry
    binding."""
    registries = {}
    for carrier in W2_CARRIERS:
        if carrier not in day_records_by_carrier:
            raise InstrumentRefusal(
                f"SELECTION_FEED_MISSING: {carrier}")
        try:
            r = WS.select(carrier, day_records_by_carrier[carrier],
                          cutoff)
            registries[carrier] = {
                "selected": r["selected"], "churn": r["churn"],
                "typing": r["typing"]}
        except WS.InsufficientPool as e:
            registries[carrier] = {"selected": None, "churn": None,
                                   "typing": str(e)}
    record = {"schema": "f2g-w2-selection-registry-v1",
              "cutoff": str(cutoff), "registries": registries}
    record["registry_digest"] = hashlib.sha256(json.dumps(
        record, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    return record


# ===================================================================
# SEAM LAYER 2 (on grassmann's 0710Z W-B1B green + formal B1B pin
# ratification; B2B pins ratified at REV 2): the ADAPTER -- window-2
# family panel assembly from producer day capsules. The MF4 per-lane
# runner holds one more cycle for W-MF4's formal ratification.
# ===================================================================
import f2g_sealed_run_instrument_cayley as SRI  # pinned digest formula


def build_family_panel(calendar, registry_record, producer_days):
    """Assembles the graph-family panel (w2_b2b/w2_b1b shape) from
    producer day capsules. Content-auth BEFORE use: every capsule's
    station_index_digest is recomputed via the PINNED producer formula
    (imported from the sealed instrument, never reimplemented) and must
    match -- typed STATION_INDEX_DIGEST_MISMATCH. The adapter assembles
    SHAPE only; family policy (measured/edge consistency, gates,
    floors) lives in the engines. Carriers the frozen selection rule
    typed out (INSUFFICIENT_POOL) are carried as typed_exclusions in
    panel metadata -- recorded, never silently absent."""
    panel = {"calendar": sorted(str(d) for d in calendar),
             "carriers": {}, "typed_exclusions": {}}
    for carrier in sorted(registry_record["registries"]):
        reg = registry_record["registries"][carrier]
        if reg["selected"] is None:
            panel["typed_exclusions"][carrier] = reg["typing"]
            continue
        days_data = producer_days.get(carrier)
        if days_data is None:
            raise InstrumentRefusal(f"PRODUCER_FEED_MISSING: {carrier}")
        measured = {}
        r = {}
        registered = []
        for day in sorted(days_data):
            cap = days_data[day]
            got = SRI.station_index_digest(cap["measured"])
            if got != cap["station_index_digest"]:
                raise InstrumentRefusal(
                    f"STATION_INDEX_DIGEST_MISMATCH: {carrier} {day} "
                    f"got={got[:12]} recorded="
                    f"{str(cap['station_index_digest'])[:12]}")
            registered.append(day)
            measured[day] = sorted(cap["measured"])
            for e, v in cap.get("edges", {}).items():
                r.setdefault(e, {})[day] = float(v)
        panel["carriers"][carrier] = {
            "registry": list(reg["selected"]),
            "registered_days": registered,
            "measured": measured, "r": r}
    return panel


# ===================================================================
# SEAM LAYER 3 (on grassmann's 0811Z W-MF4 green + formal MF4 pin
# ratification): the MF4 per-lane accrual runner -- the last held
# layer. MAG-dependent duties (calibration-ledger production at the
# cutoff) remain, but no layer now waits on W-MAG's wiring.
# ===================================================================
import w2_mf4 as MF4


def emit_mf4_predictions(pl, mf4_ledger, lease, feeds, regions,
                         bboxes, issue_day):
    """One accrual tick: for every admitted region, build the sealed
    prediction row and record it. Order of operations: engine row ->
    verify -> BARRIER ACCRUE (the chain is the authority) -> return.
    The instrument's only data touch is MECHANICAL: the risk series is
    sliced to rows dated <= issue_day (canonicalization, not policy);
    the engine still enforces ISSUE_TIME_VIOLATION fail-closed behind
    it. Typed no-prediction days emit typing rows and are ACCRUED like
    any prediction-of-record -- never silent. Emission timing and
    duplicates are refused by the barrier (LATE_OR_REVISED_PREDICTION);
    embargo means nothing here reads a row back."""
    rows = []
    for region in sorted(regions):
        feed = feeds.get(region)
        if feed is None:
            raise InstrumentRefusal(f"MF4_FEED_MISSING: {region}")
        risk = {d: v for d, v in feed["risk_series"].items()
                if str(d) <= str(issue_day)}
        row = MF4.predict_row(mf4_ledger, risk, feed["events_view"],
                              bboxes[region], region, issue_day,
                              now_utc())
        MF4.verify_row(row)
        pl.accrue_prediction(lease, region, str(issue_day),
                             row["row_digest"])
        rows.append(row)
    return rows


def append_rows_store(store_path, rows):
    """Append-only embargoed JSONL row store. Duplicate (region,
    issue_day) refuses (PREDICTION_ROW_DUPLICATE) -- second guard
    behind the barrier's; every stored row re-verifies its digest on
    the way in. Nothing here reads rows back out; release-time access
    goes through the barrier's embargo gate."""
    seen = set()
    if os.path.exists(store_path):
        with open(store_path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                seen.add((r["region"], r["issue_day"]))
    with open(store_path, "a", encoding="utf-8", newline="\n") as f:
        for row in rows:
            MF4.verify_row(row)
            key = (row["region"], row["issue_day"])
            if key in seen:
                raise MF4.Mf4Refusal(
                    f"PREDICTION_ROW_DUPLICATE: {key} (store)")
            f.write(json.dumps(row, sort_keys=True,
                               separators=(",", ":")) + "\n")
            seen.add(key)
    return len(rows)


# ---------------------------------------------------------------- selftest
def _selftest():
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="w2_accrual_kat_")
    path = os.path.join(tmpdir, "ledger.json")

    def bindings(lease, lanes=("graph", "mag1")):
        return {k: f"digest-{k}" for k in WB.REQUIRED_BINDINGS
                if k not in ("remote_lease", "lane_uuids",
                             "owner_authorization",
                             "code_manifest")} | {
            "code_manifest": {"execution_manifest_commit": "kat-mc",
                              "execution_manifest_blob_sha256":
                                  "kat-mb"},
            "remote_lease": lease,
            "lane_uuids": list(lanes),
            "global_window_uuid": "kat-window",
            "owner_authorization": "kat-owner-quote"}

    # lifecycle with persistence: prestart + predictions, then RELOAD
    fake = ["2026-09-01"]
    pl = PersistentLedger(path, clock=lambda: fake[0])
    b1_ = bindings("kat-lease-1")
    pl.prestart(b1_, WB._admission(b1_))
    day = pl.ledger.evaluation_start.isoformat()
    fake[0] = day                    # clock advances into the window
    pl.accrue_prediction("kat-lease-1", "ra", day, "row-1")
    pl.producer_receipt("kat-lease-1", "acq-1")

    pl2 = PersistentLedger(path, clock=lambda: fake[0])  # replay
    assert pl2.ledger.state == "ACCRUAL"
    assert pl2.ledger._lease == "kat-lease-1"
    assert pl2.ledger.admitted_lanes == frozenset({"graph", "mag1"})
    assert pl2.ledger.evaluation_start == pl.ledger.evaluation_start
    # duplicate prediction still refuses AFTER reload (state rebuilt)
    try:
        pl2.accrue_prediction("kat-lease-1", "ra", day, "row-1b")
        raise AssertionError("duplicate after reload must refuse")
    except WB.BarrierRefusal as e:
        assert "LATE_OR_REVISED_PREDICTION" in str(e)
    # reused-lease protection survives persistence
    try:
        pl2.ledger.prestart(bindings("kat-lease-1"), "2026-09-01")
        raise AssertionError("state must forbid second prestart")
    except WB.BarrierRefusal as e:
        assert "STATE_INVALID" in str(e)

    # codex 1815Z item-1 production-path doctors: the CURRENT OPEN
    # manifest can NEVER cross into ACCRUAL through the production
    # entry point; forged owner records refuse
    repo = os.path.abspath(os.path.join(_HERE, "..", ".."))
    try:
        assemble_prestart_admission(repo, "205e912", b1_,
                                    "not-a-seal")
        raise AssertionError("OPEN manifest must refuse admission")
    except InstrumentRefusal as e:
        assert "PRESTART_ADMISSION_REFUSED" in str(e) \
            and "SLOT_OPEN" in str(e)
    q = "the owner quote"
    oa_good_shape = {"quote": q, "quote_sha256":
                     hashlib.sha256(q.encode()).hexdigest(),
                     "binds": {"manifest_commit": "205e912",
                               "manifest_blob_sha256": "wrong",
                               "lanes": ["graph", "mag1"],
                               "lease": "kat-lease-1",
                               "window_uuid": "kat-window"}}
    try:
        assemble_prestart_admission(repo, "205e912", b1_,
                                    oa_good_shape)
        raise AssertionError("must refuse before owner checks too")
    except InstrumentRefusal as e:
        assert "PRESTART_ADMISSION_REFUSED" in str(e)

    # v1.1 appendix gate KATs (codex 1843Z item 4 + 2015Z item 1):
    # the named-store reopen -- unit level, injectable carriers
    inv_fix = {"schema": "f2g-w2-staged-body-inventory-v1",
               "store_id": "s4t", "store_root": "s4t://window2",
               "objects": {"MAG_FEED/izn/2026-01-01": {
                   "path": "ab" * 32 + ".body", "sha256": "ab" * 32,
                   "bytes": 4}}}
    desc_fix = {"schema": "f2g-w2-store-descriptor-v1",
                "store_id": "s4t", "store_root": "s4t://window2",
                "physical_root": os.path.join(tmpdir,
                                              "no-such-store")}
    inv_raw = json.dumps(inv_fix).encode()
    desc_raw = json.dumps(desc_fix).encode()
    inv_pin = {"path": "docs/f2g_window2_execution/staged_envelopes/"
                       "staged_body_inventory.json",
               "commit": "c" * 40,
               "blob_sha256": hashlib.sha256(inv_raw).hexdigest()}
    desc_pin = {"path": "docs/f2g_window2_execution/staged_envelopes/"
                        "store_descriptor.json",
                "commit": "c" * 40,
                "blob_sha256": hashlib.sha256(desc_raw).hexdigest()}

    def reader(c, path):
        return desc_raw if path.endswith("store_descriptor.json")             else inv_raw

    def man_with(status, pins):
        return {"slots": {"producer_boundary": {
            "status": status, "pins": pins}}}
    # OPEN slot -> no-op (the zero-OPEN gate owns that refusal)
    assert verify_staged_store(".", man_with("OPEN", [])) is None
    # BOUND without the inventory pin -> typed refusal
    try:
        verify_staged_store(".", man_with("BOUND", [desc_pin]))
        raise AssertionError("missing inventory pin must refuse")
    except InstrumentRefusal as e:
        assert "staged_body_inventory" in str(e)
    # BOUND without the DESCRIPTOR pin -> typed refusal (2015Z: the
    # physical root comes only from the registered descriptor)
    try:
        verify_staged_store(".", man_with("BOUND", [inv_pin]),
                            blob_reader=reader)
        raise AssertionError("missing descriptor pin must refuse")
    except InstrumentRefusal as e:
        assert "store_descriptor" in str(e)
    # pinned bytes diverging -> refusal
    try:
        verify_staged_store(
            ".", man_with("BOUND", [inv_pin, desc_pin]),
            blob_reader=lambda c, p: b"{}",
            inventory_verifier=lambda i, d: True)
        raise AssertionError("divergent inventory bytes must refuse")
    except InstrumentRefusal as e:
        assert "diverge from the manifest pin" in str(e)
    # wrong/unavailable store (verifier raises) -> TYPED refusal
    def _boom_store(inv, desc):
        raise ValueError("CAPTURE_STORE_IDENTITY_MISMATCH: kat")
    try:
        verify_staged_store(
            ".", man_with("BOUND", [inv_pin, desc_pin]),
            blob_reader=reader, inventory_verifier=_boom_store)
        raise AssertionError("wrong store must refuse")
    except InstrumentRefusal as e:
        assert "staged store reopen failed" in str(e)             and "CAPTURE_STORE_IDENTITY_MISMATCH" in str(e)
    # the DEFAULT path consumes the REAL REV 7 named-store verifier:
    # a descriptor whose physical root does not exist refuses
    # CAPTURE_STORE_UNAVAILABLE through the real API (never a PASS)
    try:
        verify_staged_store(".", man_with("BOUND",
                                          [inv_pin, desc_pin]),
                            blob_reader=reader)
        raise AssertionError("unavailable named store must refuse")
    except InstrumentRefusal as e:
        assert "CAPTURE_STORE_UNAVAILABLE" in str(e)
    # a passing reopen returns the report
    rep = verify_staged_store(
        ".", man_with("BOUND", [inv_pin, desc_pin]),
        blob_reader=reader,
        inventory_verifier=lambda i, d: {"reopened":
                                         len(i["objects"])})
    assert rep["objects"] == 1 and rep["store_id"] == "s4t"

    # barrier-level: bare bindings refuse even with valid-shape state
    try:
        PersistentLedger(os.path.join(tmpdir, "bare.json"),
                         clock=lambda: "2026-09-01").prestart(
            bindings("kat-lease-bare"), None)
        raise AssertionError("bare bindings must refuse")
    except WB.BarrierRefusal as e:
        assert "PRESTART_ADMISSION_REFUSED" in str(e)

    # tampered file -> chain broken on load
    doc = json.load(open(path))
    doc["events"][1]["payload"]["region"] = "tampered"
    with open(path, "w") as f:
        json.dump(doc, f)
    try:
        PersistentLedger(path)
        raise AssertionError("tampered file must refuse")
    except WB.BarrierRefusal as e:
        assert "LEDGER_CHAIN_BROKEN" in str(e)

    # runtime allowlist: clean walk over the real BOUND pins, then a
    # doctored on-disk module must be NAMED in the violation
    repo = os.path.abspath(os.path.join(_HERE, "..", ".."))
    # resolve the CURRENT manifest commit dynamically -- slots flip
    # BOUND over the manifest's life and the allowlist must track it
    man_commit = _git(repo, ["log", "-1", "--format=%h", "HEAD", "--",
                             EXEC_MANIFEST_PATH])
    rep = runtime_allowlist_check(repo, man_commit)
    assert rep["pins_checked"] >= 3 and rep["manifest_state"] in (
        "OPEN", "CLOSED")
    target = os.path.join(
        repo, "monitoring", "src", "f2g_design_pin_verifier_cayley.py")
    saved = open(target, "rb").read()
    try:
        with open(target, "ab") as f:
            f.write(b"\n# doctored\n")
        try:
            runtime_allowlist_check(repo, man_commit)
            raise AssertionError("doctored disk module must refuse")
        except InstrumentRefusal as e:
            assert "RUNTIME_ALLOWLIST_VIOLATION" in str(e) \
                and "f2g_design_pin_verifier_cayley.py" in str(e)
    finally:
        with open(target, "wb") as f:
            f.write(saved)
    rep = runtime_allowlist_check(repo, man_commit)   # restored clean
    assert rep["pins_checked"] >= 3

    # --- seam layer 1 KATs ---
    from datetime import date as _date, timedelta as _td
    cut = "2026-08-20"
    days90 = [(_date(2026, 8, 20) - _td(days=89 - i)).isoformat()
              for i in range(90)]
    sts = [f"S{i:02d}" for i in range(18)]
    full = {d: list(sts) for d in days90}
    thin = {d: ["T0", "T1"] for d in days90}      # pool 2 < min 8
    feeds = {"istanbul_marmara": full, "socal_coachella": full,
             "turkey_kahramanmaras": thin, "cascadia": full}
    rec = execute_selection(feeds, cut)
    assert len(rec["registries"]["cascadia"]["selected"]) == 16
    assert len(rec["registries"]["socal_coachella"]["selected"]) == 18
    tk = rec["registries"]["turkey_kahramanmaras"]
    assert tk["selected"] is None \
        and "INSUFFICIENT_POOL" in tk["typing"]
    assert rec["registry_digest"] == execute_selection(
        feeds, cut)["registry_digest"]            # deterministic
    try:
        execute_selection({k: v for k, v in feeds.items()
                           if k != "cascadia"}, cut)
        raise AssertionError("missing carrier feed must refuse")
    except InstrumentRefusal as e:
        assert "SELECTION_FEED_MISSING" in str(e)
    try:
        execute_selection(dict(feeds, cascadia={days90[0]: sts}), cut)
        raise AssertionError("bad frame must propagate")
    except WS.SelectionInputInvalid as e:
        assert "LOOKBACK_FRAME_INVALID" in str(e)

    b = assemble_prestart_bindings(
        repo, execution_manifest_commit=man_commit,
        mf4_model_scaler_digest="kat-mf4", power_envelope_digest="kat-env",
        global_window_uuid="kat-uuid", remote_lease="kat-lease-b",
        lane_uuids=["graph", "mag1", "mf4"],
        owner_authorization="kat-owner",
        hypothesis_registries_digest="kat-reg",
        calibration_fits_digest="kat-cal", adapters_digest="kat-adp")
    assert set(b) == set(WB.REQUIRED_BINDINGS)
    assert b["models"]["design_manifest_commit"].startswith("5fba544")
    pl3 = PersistentLedger(os.path.join(tmpdir, "l3.json"),
                           clock=lambda: "2026-09-01")
    pl3.prestart(b, WB._admission(b))  # assembled dict + admission
    assert pl3.ledger.state == "ACCRUAL"

    # --- seam layer 2 KATs: adapter -> REAL engine end-to-end ---
    import w2_b2b as W2B
    A = [f"A{i}" for i in range(5)]
    Bs = [f"B{i}" for i in range(5)]
    reg10 = sorted(A + Bs)
    cal6 = [f"2026-10-{i:02d}" for i in range(1, 7)]

    def mk_edges(cluster_a, cluster_b, strong=5.0, weak=0.1):
        ew = {}
        nodes = list(cluster_a) + list(cluster_b)
        for i, x in enumerate(nodes):
            for y in nodes[i + 1:]:
                same = (x in cluster_a) == (y in cluster_a)
                ew["|".join(sorted((x, y)))] = strong if same else weak
        return ew

    prod = {"cascadia": {
        d: {"measured": reg10,
            "station_index_digest": SRI.station_index_digest(reg10),
            "edges": mk_edges(A, Bs)} for d in cal6}}
    reg_rec = {"registries": {
        "cascadia": {"selected": reg10, "churn": 1.0, "typing": None},
        "turkey_kahramanmaras": {"selected": None, "churn": None,
                                 "typing": "INSUFFICIENT_POOL: kat"}}}
    panel = build_family_panel(cal6, reg_rec, prod)
    assert panel["typed_exclusions"]["turkey_kahramanmaras"] \
        .startswith("INSUFFICIENT_POOL")
    res = W2B.w2_b2b_family(panel, doc_sha256="ab" * 32, n_draws=49)
    assert res["runs_by_carrier"]["cascadia"] == 1 \
        and res["p_value"] == 1.0, res      # adapter -> engine e2e

    bad = json.loads(json.dumps(prod))
    bad["cascadia"][cal6[2]]["station_index_digest"] = "0" * 64
    try:
        build_family_panel(cal6, reg_rec, bad)
        raise AssertionError("doctored digest must refuse")
    except InstrumentRefusal as e:
        assert "STATION_INDEX_DIGEST_MISMATCH" in str(e)
    try:
        build_family_panel(cal6, {"registries": {
            "cascadia": reg_rec["registries"]["cascadia"]}}, {})
        raise AssertionError("missing producer feed must refuse")
    except InstrumentRefusal as e:
        assert "PRODUCER_FEED_MISSING" in str(e)

    # --- seam layer 3 KATs: MF4 runner through the REAL engine +
    # REAL barrier chain ---
    import numpy as _np
    bbox_k = {"min_lat": 30.0, "max_lat": 40.0,
              "min_lon": -125.0, "max_lon": -115.0}
    bboxes_k = {"ra": bbox_k, "rb": bbox_k}
    rngk = _np.random.Generator(_np.random.PCG64(13))
    cal_days = [(_date(2025, 10, 10) + _td(days=i)).isoformat()
                for i in range(120)]
    risk_k = {r: {d: float(rngk.uniform(0, 1)) for d in cal_days}
              for r in ("ra", "rb")}
    ev_k = [{"day": (_date(2025, 11, 1) + _td(days=7 * i)).isoformat(),
             "lat": 35.0, "lon": -120.0, "mag": 4.5} for i in range(8)]
    mf4_led = MF4.calibrate(risk_k, ev_k, bboxes_k, ["ra", "rb"],
                            "2026-02-10", "2026-02-08")

    fake2 = ["2026-09-01"]
    pl4 = PersistentLedger(os.path.join(tmpdir, "l4.json"),
                           clock=lambda: fake2[0])
    b4_ = bindings("kat-lease-4", lanes=("mf4",))
    pl4.prestart(b4_, WB._admission(b4_))
    d0 = pl4.ledger.evaluation_start.isoformat()
    fake2[0] = d0

    # feeds carry FULL history incl the issue day + one FUTURE-dated
    # row: the mechanical slice removes the future row (the engine
    # would refuse it fail-closed otherwise)
    d_future = (pl4.ledger.evaluation_start + _td(days=1)).isoformat()
    hist_days = [(_date(2026, 8, 20) + _td(days=i)).isoformat()
                 for i in range(14)]
    hist_days = [d for d in hist_days if d <= d0] + [d0, d_future]
    feeds = {r: {"risk_series": {d: 0.5 for d in dict.fromkeys(
        hist_days)}, "events_view": ev_k} for r in ("ra", "rb")}
    rows = emit_mf4_predictions(pl4, mf4_led, "kat-lease-4", feeds,
                                ["ra", "rb"], bboxes_k, d0)
    assert len(rows) == 2 and all("p_model" in r for r in rows)
    assert sum(1 for ev in pl4.ledger.events
               if ev["kind"] == "PREDICTION") == 2
    # duplicate emission refuses AT THE BARRIER
    try:
        emit_mf4_predictions(pl4, mf4_led, "kat-lease-4", feeds,
                             ["ra"], bboxes_k, d0)
        raise AssertionError("duplicate emission must refuse")
    except WB.BarrierRefusal as e:
        assert "LATE_OR_REVISED_PREDICTION" in str(e)
    # typed no-prediction day (no prior-day risk) emits + accrues
    d1 = (pl4.ledger.evaluation_start + _td(days=3)).isoformat()
    fake2[0] = d1
    feeds_thin = {"ra": {"risk_series": {d1: 0.5},
                         "events_view": ev_k}}
    rows2 = emit_mf4_predictions(pl4, mf4_led, "kat-lease-4",
                                 feeds_thin, ["ra"], bboxes_k, d1)
    assert "typing" in rows2[0] \
        and "NO_PREDICTION" in rows2[0]["typing"]
    assert sum(1 for ev in pl4.ledger.events
               if ev["kind"] == "PREDICTION") == 3
    try:
        emit_mf4_predictions(pl4, mf4_led, "kat-lease-4", {},
                             ["ra"], bboxes_k, d1)
        raise AssertionError("missing feed must refuse")
    except InstrumentRefusal as e:
        assert "MF4_FEED_MISSING" in str(e)
    # row store: append + duplicate guard + digest verify on the way in
    store = os.path.join(tmpdir, "mf4_rows.jsonl")
    assert append_rows_store(store, rows) == 2
    try:
        append_rows_store(store, rows[:1])
        raise AssertionError("store duplicate must refuse")
    except MF4.Mf4Refusal as e:
        assert "PREDICTION_ROW_DUPLICATE" in str(e)

    print("w2_accrual_instrument selftest: ALL PASS")


if __name__ == "__main__":
    _selftest()
