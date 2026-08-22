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
    if manifest.get("schema") != "f2g-window2-execution-manifest-v1":
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
    def prestart(self, bindings):
        self.ledger.prestart(bindings, self._clock())
        self.save()

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


# ---------------------------------------------------------------- selftest
def _selftest():
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="w2_accrual_kat_")
    path = os.path.join(tmpdir, "ledger.json")

    def bindings(lease):
        return {k: f"digest-{k}" for k in WB.REQUIRED_BINDINGS
                if k not in ("remote_lease", "lane_uuids",
                             "owner_authorization")} | {
            "remote_lease": lease,
            "lane_uuids": ["graph", "mag1"],
            "owner_authorization": "kat-owner-quote"}

    # lifecycle with persistence: prestart + predictions, then RELOAD
    fake = ["2026-09-01"]
    pl = PersistentLedger(path, clock=lambda: fake[0])
    pl.prestart(bindings("kat-lease-1"))
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
    rep = runtime_allowlist_check(repo, "9d2f034")
    assert rep["pins_checked"] == 3 and rep["manifest_state"] == "OPEN"
    target = os.path.join(
        repo, "monitoring", "src", "f2g_design_pin_verifier_cayley.py")
    saved = open(target, "rb").read()
    try:
        with open(target, "ab") as f:
            f.write(b"\n# doctored\n")
        try:
            runtime_allowlist_check(repo, "9d2f034")
            raise AssertionError("doctored disk module must refuse")
        except InstrumentRefusal as e:
            assert "RUNTIME_ALLOWLIST_VIOLATION" in str(e) \
                and "f2g_design_pin_verifier_cayley.py" in str(e)
    finally:
        with open(target, "wb") as f:
            f.write(saved)
    rep = runtime_allowlist_check(repo, "9d2f034")   # restored clean
    assert rep["pins_checked"] == 3

    print("w2_accrual_instrument selftest: ALL PASS")


if __name__ == "__main__":
    _selftest()
