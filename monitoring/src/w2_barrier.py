#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 BARRIER state machine (cayley) -- prereg v0.3 sec 2 (codex
R2 fix 1, two-stage barrier) + sec 5 (non-circular Holm selector, codex
R2 fix 2), design freeze CLOSED @ 12161f6/5fba544; grassmann's bar seam
pin ("w2_barrier state machine"). Seam FIXED as `w2_barrier`.

An append-only, hash-chained event ledger drives the stages:

  DESIGN_CLOSED -> (prestart) -> ACCRUAL -> (close_support_barrier)
  -> SUPPORT_BARRIER -> (final_fire per lane) -> (release) -> RELEASED
  plus the typed terminal WINDOW3_TERMINAL (post-first-fire failure).

PRESTART binds EVERY lane (code, models, calibration fits, hypothesis
registries, adapters, the M-F4 model+scaler, the anticipated-mask power
envelope, global window UUID, remote lease, lane UUIDs, owner
authorization) and fixes `evaluation_start` = the first UTC day AFTER
the barrier completes, `evaluation_end` = start + 131 d, maturity tail
= end + H_max (7 d).

The NINE sec-2 typed refusals (all KAT'd in the selftest):
  LATE_OR_REVISED_PREDICTION, EARLY_LABEL_ACCESS,
  SEMANTIC_SUPPORT_INSPECTION, MISSING_MATURITY_TAIL,
  CROSS_LANE_RELEASE_BEFORE_TERMINALS, MISSING_LANE_AUTHORIZATION,
  LATE_LANE_ADDITION, REUSED_GLOBAL_LEASE / GLOBAL_LEASE_INCORRECT,
  POST_FIRST_FIRE_SOURCE_CHANGE (-> WINDOW3_TERMINAL).
Plus: PRE_BARRIER_VERDICT_ROW (verdict-bearing row predating the
barrier), EMBARGO_VIOLATION (result access before release),
VALUE_FIRE_SEAL_MISSING, NON_ANALYST_REQUIRED,
BINDING_IMMUTABLE_AFTER_PRESTART (fail-closed extra: rebinding between
PRESTART and first fire refuses WITHOUT the window-3 terminal),
PRESTART_INCOMPLETE, STATE_INVALID, LEDGER_CHAIN_BROKEN.

Non-circular selector (sec 5): S = {h in the FULL FROZEN four-member
graph {B2A,B2B,B1B,B3A} : CP_LCB >= 0.80 at h's registered MDE},
computed ONCE and immutable (SELECTOR_ALREADY_COMMITTED); any
recertification attempt -- including the mandated 4->3 relaxation
counterexample -- refuses (SELECTOR_RECERTIFICATION_REFUSED /
SELECTOR_GRAPH_NOT_FULL); production Holm at family alpha 0.05 runs
over the immutable S only; members outside S are typed
CANNOT_DETERMINE_NO_POWER and never enter Holm.

Interpretation pins (disclosed, R1.2-able):
- Prediction timing: a sealed row must be emitted in
  [issue_day, issue_day + H_max) -- "before its label matures";
  anything else (early, late, or a second row for the same
  (region, issue_day)) is LATE_OR_REVISED_PREDICTION.
- All times are INJECTED as ISO UTC days (no clock reads inside), so
  every boundary is deterministic and bar-KATable; the accrual
  instrument supplies live clock reads at its own layer.
- The ledger is tamper-evident: each event's digest chains over the
  previous digest; verify_chain() re-walks it (LEDGER_CHAIN_BROKEN).

This module opens no window-2 value; it is the machinery that REFUSES
until the registered sequence is satisfied.
"""
import hashlib
import json
from datetime import date, timedelta

H_MAX_DAYS = 7
EVALUATION_SPAN_DAYS = 131          # end = start + 131 (132-day span)
GRAPH_MEMBERS = ("B1B", "B2A", "B2B", "B3A")
CP_FLOOR = 0.80
ALPHA_FAMILY = 0.05
REQUIRED_BINDINGS = (
    "code_manifest", "models", "calibration_fits",
    "hypothesis_registries", "adapters", "mf4_model_scaler",
    "power_envelope", "global_window_uuid", "remote_lease",
    "lane_uuids", "owner_authorization")
ADMISSION_SCHEMA = "f2g-w2-prestart-admission-v1"
ADMISSION_FIELDS = {"schema", "manifest_commit",
                    "manifest_blob_sha256", "prestart_verifier",
                    "allowlist", "owner", "lanes", "lease",
                    "window_uuid", "staged_boundary_sha256",
                    "admission_digest"}


def admission_digest(admission):
    body = {k: admission[k] for k in sorted(admission)
            if k != "admission_digest"}
    return hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class BarrierRefusal(ValueError):
    """Typed refusal; the code leads the message."""


def _d(s):
    return date.fromisoformat(str(s))


class BarrierLedger:
    """Append-only hash-chained barrier ledger."""

    def __init__(self, used_leases=()):
        self.events = []
        self.state = "DESIGN_CLOSED"
        self._used_leases = set(used_leases)
        self._lease = None
        self._bindings_digest = None
        self.admitted_lanes = None
        self.evaluation_start = None
        self.evaluation_end = None
        self.maturity_tail_end = None
        self._predictions = {}
        self._first_fired = False
        self._terminal_lanes = set()
        self._sealed_lanes = set()
        self._verified_lanes = set()
        self._selector = None
        self._selector_power = None

    # ---------------- chain ----------------
    def _append(self, kind, payload):
        prev = self.events[-1]["digest"] if self.events else "0" * 64
        body = json.dumps({"kind": kind, "payload": payload,
                           "prev": prev, "seq": len(self.events)},
                          sort_keys=True, separators=(",", ":"))
        ev = {"seq": len(self.events), "kind": kind,
              "payload": payload, "prev": prev,
              "digest": hashlib.sha256(body.encode()).hexdigest()}
        self.events.append(ev)
        return ev

    def verify_chain(self):
        prev = "0" * 64
        for ev in self.events:
            body = json.dumps({"kind": ev["kind"],
                               "payload": ev["payload"],
                               "prev": ev["prev"], "seq": ev["seq"]},
                              sort_keys=True, separators=(",", ":"))
            if ev["prev"] != prev or \
                    hashlib.sha256(body.encode()).hexdigest() != \
                    ev["digest"]:
                raise BarrierRefusal(
                    f"LEDGER_CHAIN_BROKEN: seq={ev['seq']}")
            prev = ev["digest"]
        return True

    def _check_lease(self, lease):
        if lease != self._lease:
            raise BarrierRefusal(f"GLOBAL_LEASE_INCORRECT: {lease!r}")

    def _require_state(self, *states):
        if self.state not in states:
            raise BarrierRefusal(
                f"STATE_INVALID: {self.state} not in {states}")

    # ---------------- stage 1: PRESTART ----------------
    @staticmethod
    def _validate_admission(admission, bindings):
        """codex 1815Z item 1: the barrier REFUSES bare truthy binding
        dictionaries. PRESTART requires a closed ADMISSION CAPSULE
        whose internal consistency is validated here (the LIVE
        re-verification -- execution verifier --prestart + runtime
        allowlist -- runs in the production instrument that BUILT the
        capsule; this layer is the pure state machine)."""
        def refuse(detail):
            raise BarrierRefusal(f"PRESTART_ADMISSION_REFUSED: {detail}")
        if not isinstance(admission, dict):
            refuse("bare bindings refused -- a closed admission "
                   "capsule is required")
        if set(admission) != ADMISSION_FIELDS or \
                admission.get("schema") != ADMISSION_SCHEMA:
            refuse("admission schema not closed")
        body = {k: admission[k] for k in sorted(admission)
                if k != "admission_digest"}
        got = hashlib.sha256(json.dumps(
            body, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
        if got != admission["admission_digest"]:
            refuse("admission digest mismatch")
        pv = admission["prestart_verifier"]
        if pv.get("verdict") != "PASS" or pv.get("mode") != \
                "prestart" or pv.get("slots_open") != 0:
            refuse(f"prestart verifier verdict not a zero-OPEN PASS: "
                   f"{pv.get('verdict')}/{pv.get('slots_open')}")
        if not str(pv.get("manifest_commit", "")).startswith(
                str(admission["manifest_commit"])[:12]) and \
                str(admission["manifest_commit"]) != \
                str(pv.get("manifest_commit", "")):
            refuse("stale verifier receipt: verdict manifest differs "
                   "from the admission manifest")
        al = admission["allowlist"]
        if not al.get("pins_checked", 0) > 0:
            refuse("allowlist report absent or empty")
        # codex 2235Z item 1: the staged-boundary report digest is a
        # REQUIRED binding (at a zero-OPEN PASS the producer boundary
        # is BOUND, so the S/T/E join must have run)
        sb = admission["staged_boundary_sha256"]
        if not (isinstance(sb, str) and len(sb) == 64 and
                all(c in "0123456789abcdef" for c in sb)):
            refuse("staged-boundary report digest absent or untyped")
        owner = admission["owner"]
        if not isinstance(owner, dict) or \
                set(owner) != {"quote", "quote_sha256", "binds"}:
            refuse("owner authorization is not a closed binding "
                   "record")
        if hashlib.sha256(str(owner["quote"]).encode()).hexdigest() \
                != owner["quote_sha256"]:
            refuse("owner quote digest mismatch")
        b = owner["binds"]
        if set(b) != {"manifest_commit", "manifest_blob_sha256",
                      "lanes", "lease", "window_uuid"}:
            refuse("owner binding fields not closed")
        if b["manifest_blob_sha256"] != \
                admission["manifest_blob_sha256"] or \
                str(b["manifest_commit"]) != \
                str(admission["manifest_commit"]):
            refuse("manifest changed after the owner binding")
        if sorted(b["lanes"]) != sorted(admission["lanes"]) or \
                b["lease"] != admission["lease"] or \
                b["window_uuid"] != admission["window_uuid"]:
            refuse("owner binding diverges from the admission")
        # cross-check the bindings the barrier is about to accept
        if bindings.get("remote_lease") != admission["lease"]:
            refuse("bindings lease differs from the admitted lease")
        if sorted(bindings.get("lane_uuids", ())) != \
                sorted(admission["lanes"]):
            refuse("bindings lanes differ from the admitted lanes")
        if bindings.get("global_window_uuid") != \
                admission["window_uuid"]:
            refuse("bindings window uuid differs from the admission")
        cm = bindings.get("code_manifest")
        if not (isinstance(cm, dict) and
                cm.get("execution_manifest_blob_sha256") ==
                admission["manifest_blob_sha256"]):
            refuse("bindings code_manifest does not carry the "
                   "admitted manifest blob sha")

    def prestart(self, bindings, now_utc_day, admission=None):
        self._require_state("DESIGN_CLOSED")
        self._validate_admission(admission, bindings)
        missing = [k for k in REQUIRED_BINDINGS
                   if not bindings.get(k)]
        if "owner_authorization" in missing or "lane_uuids" in missing:
            raise BarrierRefusal(
                f"MISSING_LANE_AUTHORIZATION: {missing}")
        if missing:
            raise BarrierRefusal(f"PRESTART_INCOMPLETE: {missing}")
        lease = bindings["remote_lease"]
        if lease in self._used_leases:
            raise BarrierRefusal(f"REUSED_GLOBAL_LEASE: {lease!r}")
        self._lease = lease
        self._used_leases.add(lease)
        self.admitted_lanes = frozenset(bindings["lane_uuids"])
        self.evaluation_start = _d(now_utc_day) + timedelta(days=1)
        self.evaluation_end = self.evaluation_start + \
            timedelta(days=EVALUATION_SPAN_DAYS)
        self.maturity_tail_end = self.evaluation_end + \
            timedelta(days=H_MAX_DAYS)
        self._bindings_digest = hashlib.sha256(json.dumps(
            bindings, sort_keys=True, default=str).encode()).hexdigest()
        self._append("PRESTART", {
            "bindings_digest": self._bindings_digest,
            "evaluation_start": self.evaluation_start.isoformat(),
            "evaluation_end": self.evaluation_end.isoformat(),
            "maturity_tail_end": self.maturity_tail_end.isoformat(),
            "lanes": sorted(self.admitted_lanes)})
        self.state = "ACCRUAL"

    def add_lane(self, lease, lane):
        if self.state != "DESIGN_CLOSED":
            raise BarrierRefusal(f"LATE_LANE_ADDITION: {lane!r}")

    def rebind_source(self, lease, what):
        """Any binding change after PRESTART refuses; after the FIRST
        final fire it is the window-3 typed terminal."""
        if self.state == "DESIGN_CLOSED":
            return  # design revisions live in the codex loop
        self._check_lease(lease)
        if self._first_fired:
            self._append("WINDOW3_TERMINAL",
                         {"cause": "POST_FIRST_FIRE_SOURCE_CHANGE",
                          "what": what})
            self.state = "WINDOW3_TERMINAL"
            raise BarrierRefusal(
                f"POST_FIRST_FIRE_SOURCE_CHANGE: {what!r} -> typed "
                "terminal for window 3")
        raise BarrierRefusal(
            f"BINDING_IMMUTABLE_AFTER_PRESTART: {what!r}")

    # ---------------- stage 2: ACCRUAL ----------------
    def accrue_prediction(self, lease, region, issue_day, row_digest,
                          emitted_utc_day):
        self._require_state("ACCRUAL")
        self._check_lease(lease)
        d0 = _d(issue_day)
        if not (self.evaluation_start <= d0 <= self.evaluation_end):
            raise BarrierRefusal(
                f"PRE_BARRIER_VERDICT_ROW: issue day {issue_day} "
                "outside the evaluation window")
        em = _d(emitted_utc_day)
        if not (d0 <= em < d0 + timedelta(days=H_MAX_DAYS)):
            raise BarrierRefusal(
                f"LATE_OR_REVISED_PREDICTION: emitted {emitted_utc_day}"
                f" outside [{issue_day}, +{H_MAX_DAYS}d)")
        key = (str(region), str(issue_day))
        if key in self._predictions:
            raise BarrierRefusal(
                f"LATE_OR_REVISED_PREDICTION: duplicate {key}")
        self._predictions[key] = row_digest
        self._append("PREDICTION", {"region": str(region),
                                    "issue_day": str(issue_day),
                                    "row_digest": row_digest,
                                    "emitted": str(emitted_utc_day)})

    def producer_receipt(self, lease, receipt_digest):
        """Mechanical acquisition only, with access receipts."""
        self._require_state("ACCRUAL")
        self._check_lease(lease)
        self._append("PRODUCER_RECEIPT", {"digest": receipt_digest})

    def read_labels(self, role):
        if self.state in ("DESIGN_CLOSED", "ACCRUAL"):
            raise BarrierRefusal(
                f"EARLY_LABEL_ACCESS: state={self.state}")
        if role != "non_analyst":
            raise BarrierRefusal(f"NON_ANALYST_REQUIRED: {role!r}")
        return True

    def inspect_support(self, semantic):
        """Mechanical support counts/hashes are fine during accrual;
        SEMANTIC inspection is not."""
        if semantic and self.state in ("DESIGN_CLOSED", "ACCRUAL"):
            raise BarrierRefusal("SEMANTIC_SUPPORT_INSPECTION")
        return True

    def record_verdict_row(self, lease, day):
        self._check_lease(lease)
        if self.evaluation_start is None or \
                _d(day) < self.evaluation_start:
            raise BarrierRefusal(f"PRE_BARRIER_VERDICT_ROW: {day}")
        self._append("VERDICT_ROW", {"day": str(day)})

    # ---------------- stage 3: SUPPORT BARRIER ----------------
    def close_support_barrier(self, lease, now_utc_day, role):
        self._require_state("ACCRUAL")
        self._check_lease(lease)
        if role != "non_analyst":
            raise BarrierRefusal(f"NON_ANALYST_REQUIRED: {role!r}")
        if _d(now_utc_day) <= self.maturity_tail_end:
            raise BarrierRefusal(
                f"MISSING_MATURITY_TAIL: {now_utc_day} <= "
                f"{self.maturity_tail_end.isoformat()}")
        self._append("SUPPORT_BARRIER_CLOSED",
                     {"now": str(now_utc_day)})
        self.state = "SUPPORT_BARRIER"

    def record_owner_seal(self, lease, lane, seal_digest):
        self._require_state("SUPPORT_BARRIER")
        self._check_lease(lease)
        if lane not in self.admitted_lanes:
            raise BarrierRefusal(f"MISSING_LANE_AUTHORIZATION: {lane!r}")
        self._sealed_lanes.add(lane)
        self._append("OWNER_SEAL", {"lane": lane,
                                    "seal": seal_digest})

    # ---------------- stage 4: FIRE / RELEASE ----------------
    def final_fire(self, lease, lane, result_digest):
        self._require_state("SUPPORT_BARRIER")
        self._check_lease(lease)
        if lane not in self.admitted_lanes:
            raise BarrierRefusal(f"MISSING_LANE_AUTHORIZATION: {lane!r}")
        if lane not in self._sealed_lanes:
            raise BarrierRefusal(f"VALUE_FIRE_SEAL_MISSING: {lane!r}")
        if lane in self._terminal_lanes:
            raise BarrierRefusal(f"STATE_INVALID: {lane!r} already "
                                 "terminal")
        self._first_fired = True
        self._terminal_lanes.add(lane)
        self._append("FINAL_FIRE", {"lane": lane,
                                    "result_digest": result_digest})

    def record_verifier_pass(self, lease, lane, verifier_digest):
        self._check_lease(lease)
        if lane not in self._terminal_lanes:
            raise BarrierRefusal(f"STATE_INVALID: verifier before "
                                 f"fire for {lane!r}")
        self._verified_lanes.add(lane)
        self._append("VERIFIER_PASS", {"lane": lane,
                                       "verifier": verifier_digest})

    def read_result(self, lease, lane):
        self._check_lease(lease)
        if self.state != "RELEASED":
            raise BarrierRefusal(f"EMBARGO_VIOLATION: state="
                                 f"{self.state}")
        return next(ev["payload"] for ev in reversed(self.events)
                    if ev["kind"] == "FINAL_FIRE"
                    and ev["payload"]["lane"] == lane)

    def release(self, lease):
        self._require_state("SUPPORT_BARRIER")
        self._check_lease(lease)
        missing = (self.admitted_lanes - self._terminal_lanes) | \
                  (self._terminal_lanes - self._verified_lanes)
        if missing:
            raise BarrierRefusal(
                f"CROSS_LANE_RELEASE_BEFORE_TERMINALS: "
                f"{sorted(missing)}")
        self._append("RELEASE", {"lanes": sorted(self.admitted_lanes)})
        self.state = "RELEASED"

    # ---------------- sec-5 non-circular selector ----------------
    def commit_selector(self, power_results):
        """S computed ONCE from the complete four-member synthetic
        power result; immutable thereafter."""
        if self._selector is not None:
            raise BarrierRefusal("SELECTOR_ALREADY_COMMITTED")
        if set(power_results) != set(GRAPH_MEMBERS):
            raise BarrierRefusal(
                f"SELECTOR_GRAPH_NOT_FULL: {sorted(power_results)}")
        for h, res in power_results.items():
            if tuple(sorted(res.get("graph", ()))) != GRAPH_MEMBERS:
                raise BarrierRefusal(
                    f"SELECTOR_GRAPH_NOT_FULL: {h} certified under "
                    f"{res.get('graph')}")
        self._selector = frozenset(
            h for h, res in power_results.items()
            if res["cp_lcb"] >= CP_FLOOR)
        self._selector_power = {h: float(power_results[h]["cp_lcb"])
                                for h in GRAPH_MEMBERS}
        self._append("SELECTOR_COMMITTED",
                     {"S": sorted(self._selector),
                      "power": self._selector_power})
        return self._selector

    def recertify_selector(self, power_results):
        """The mandated 4->3 relaxation counterexample: ANY
        recertification attempt refuses."""
        raise BarrierRefusal(
            "SELECTOR_RECERTIFICATION_REFUSED: certification is never "
            "iterated after removing a member (prereg sec 5)")

    def holm_graph_lane(self, p_values):
        """Holm at family alpha 0.05 over the immutable S; members
        outside S are typed CANNOT_DETERMINE_NO_POWER and never enter."""
        if self._selector is None:
            raise BarrierRefusal("SELECTOR_NOT_COMMITTED")
        if set(p_values) != set(GRAPH_MEMBERS):
            raise BarrierRefusal(
                f"HOLM_INPUT_INCOMPLETE: {sorted(p_values)}")
        out = {}
        s_members = sorted(self._selector)
        for h in GRAPH_MEMBERS:
            if h not in self._selector:
                out[h] = "CANNOT_DETERMINE_NO_POWER"
        order = sorted(s_members, key=lambda h: p_values[h])
        still = True
        for i, h in enumerate(order):
            thresh = ALPHA_FAMILY / (len(s_members) - i)
            if still and p_values[h] <= thresh:
                out[h] = "REJECT"
            else:
                still = False
                out[h] = "NO_REJECT"
        return {"S": s_members, "alpha": ALPHA_FAMILY,
                "verdicts": out}


# ---------------------------------------------------------------- selftest
def _expect(fn, code):
    try:
        fn()
    except BarrierRefusal as e:
        assert code in str(e), (code, str(e))
        return
    raise AssertionError(f"expected {code}")


def _bindings(lease="lease-1"):
    return {k: f"digest-{k}" for k in REQUIRED_BINDINGS
            if k not in ("remote_lease", "lane_uuids",
                         "owner_authorization", "code_manifest")} | {
        "code_manifest": {"execution_manifest_commit": "kat-mc",
                          "execution_manifest_blob_sha256": "kat-mb"},
        "remote_lease": lease,
        "lane_uuids": ["graph", "mag1", "mf4"],
        "global_window_uuid": "kat-window",
        "owner_authorization": "asylum-seal-digest"}


def _admission(bindings, **mut):
    """Internally-consistent KAT admission capsule (the barrier layer
    validates consistency; LIVE verification lives in the production
    instrument). `mut` doctors individual fields AFTER digesting
    unless it doctors the digest itself."""
    owner_quote = "kat-owner-quote"
    adm = {"schema": ADMISSION_SCHEMA, "manifest_commit": "kat-mc",
           "manifest_blob_sha256":
               bindings["code_manifest"]
               ["execution_manifest_blob_sha256"],
           "prestart_verifier": {"verdict": "PASS",
                                 "mode": "prestart",
                                 "slots_open": 0,
                                 "manifest_commit": "kat-mc"},
           "allowlist": {"pins_checked": 3},
           "staged_boundary_sha256": "e" * 64,
           "owner": {"quote": owner_quote,
                     "quote_sha256": hashlib.sha256(
                         owner_quote.encode()).hexdigest(),
                     "binds": {"manifest_commit": "kat-mc",
                               "manifest_blob_sha256":
                                   bindings["code_manifest"]
                                   ["execution_manifest_blob_sha256"],
                               "lanes": list(bindings["lane_uuids"]),
                               "lease": bindings["remote_lease"],
                               "window_uuid":
                                   bindings["global_window_uuid"]}},
           "lanes": list(bindings["lane_uuids"]),
           "lease": bindings["remote_lease"],
           "window_uuid": bindings["global_window_uuid"]}
    adm.update({k: v for k, v in mut.items()
                if k != "admission_digest"})
    adm["admission_digest"] = admission_digest(adm)
    if "admission_digest" in mut:
        adm["admission_digest"] = mut["admission_digest"]
    return adm


def _selftest():
    # PRESTART refusals: missing authorization; incomplete; reused lease
    led = BarrierLedger()
    b = _bindings()
    adm = _admission(b)
    b_no_auth = dict(b, owner_authorization="")
    _expect(lambda: BarrierLedger().prestart(
        b_no_auth, "2026-09-01", _admission(b_no_auth)),
        "MISSING_LANE_AUTHORIZATION")
    b_no_env = dict(b, power_envelope="")
    _expect(lambda: BarrierLedger().prestart(
        b_no_env, "2026-09-01", _admission(b_no_env)),
        "PRESTART_INCOMPLETE")
    _expect(lambda: BarrierLedger(used_leases={"lease-1"})
            .prestart(b, "2026-09-01", adm), "REUSED_GLOBAL_LEASE")

    # codex 1815Z item 1 doctors: bare bindings; stale verifier
    # receipt; manifest changed after the owner binding; non-PASS /
    # OPEN-slot verdicts; doctored digest
    _expect(lambda: BarrierLedger().prestart(b, "2026-09-01"),
            "PRESTART_ADMISSION_REFUSED")
    _expect(lambda: BarrierLedger().prestart(b, "2026-09-01",
                                             "not-a-capsule"),
            "PRESTART_ADMISSION_REFUSED")
    _expect(lambda: BarrierLedger().prestart(
        b, "2026-09-01", _admission(b, prestart_verifier={
            "verdict": "PASS", "mode": "prestart", "slots_open": 0,
            "manifest_commit": "OTHER-COMMIT"})),
        "PRESTART_ADMISSION_REFUSED")     # stale receipt
    _expect(lambda: BarrierLedger().prestart(
        b, "2026-09-01", _admission(b, manifest_blob_sha256="drifted")),
        "PRESTART_ADMISSION_REFUSED")     # manifest changed post-binding
    _expect(lambda: BarrierLedger().prestart(
        b, "2026-09-01", _admission(b, prestart_verifier={
            "verdict": "REFUSE", "mode": "prestart", "slots_open": 2,
            "manifest_commit": "kat-mc"})),
        "PRESTART_ADMISSION_REFUSED")     # OPEN manifest class
    _expect(lambda: BarrierLedger().prestart(
        b, "2026-09-01", _admission(b, admission_digest="0" * 64)),
        "PRESTART_ADMISSION_REFUSED")

    # happy prestart fixes the window
    led.prestart(b, "2026-09-01", adm)
    assert led.state == "ACCRUAL"
    assert led.evaluation_start.isoformat() == "2026-09-02"
    assert led.evaluation_end.isoformat() == "2027-01-11"
    assert led.maturity_tail_end.isoformat() == "2027-01-18"

    # late lane addition; wrong lease; pre-barrier verdict row
    _expect(lambda: led.add_lane("lease-1", "extra"),
            "LATE_LANE_ADDITION")
    _expect(lambda: led.accrue_prediction("wrong", "ra", "2026-09-10",
                                          "d", "2026-09-10"),
            "GLOBAL_LEASE_INCORRECT")
    _expect(lambda: led.record_verdict_row("lease-1", "2026-09-01"),
            "PRE_BARRIER_VERDICT_ROW")
    _expect(lambda: led.accrue_prediction("lease-1", "ra",
                                          "2026-08-30", "d",
                                          "2026-08-30"),
            "PRE_BARRIER_VERDICT_ROW")

    # accrual: ok row; late; early; revised
    led.accrue_prediction("lease-1", "ra", "2026-09-10", "row1",
                          "2026-09-10")
    _expect(lambda: led.accrue_prediction("lease-1", "ra",
                                          "2026-09-11", "row2",
                                          "2026-09-18"),
            "LATE_OR_REVISED_PREDICTION")     # late (>= +7d)
    _expect(lambda: led.accrue_prediction("lease-1", "ra",
                                          "2026-09-12", "row3",
                                          "2026-09-11"),
            "LATE_OR_REVISED_PREDICTION")     # pre-dated
    _expect(lambda: led.accrue_prediction("lease-1", "ra",
                                          "2026-09-10", "row4",
                                          "2026-09-10"),
            "LATE_OR_REVISED_PREDICTION")     # revision/duplicate

    # embargo-side refusals during accrual
    _expect(lambda: led.read_labels("non_analyst"),
            "EARLY_LABEL_ACCESS")
    _expect(lambda: led.inspect_support(semantic=True),
            "SEMANTIC_SUPPORT_INSPECTION")
    assert led.inspect_support(semantic=False)
    led.producer_receipt("lease-1", "acq-digest")

    # support barrier: early close; wrong role; then close
    _expect(lambda: led.close_support_barrier("lease-1", "2027-01-18",
                                              "non_analyst"),
            "MISSING_MATURITY_TAIL")
    _expect(lambda: led.close_support_barrier("lease-1", "2027-01-19",
                                              "analyst"),
            "NON_ANALYST_REQUIRED")
    led.close_support_barrier("lease-1", "2027-01-19", "non_analyst")
    assert led.read_labels("non_analyst")
    _expect(lambda: led.read_labels("analyst"), "NON_ANALYST_REQUIRED")

    # fire gates: unadmitted lane; missing seal; then seal + fire
    _expect(lambda: led.final_fire("lease-1", "rogue", "r"),
            "MISSING_LANE_AUTHORIZATION")
    _expect(lambda: led.final_fire("lease-1", "graph", "r"),
            "VALUE_FIRE_SEAL_MISSING")
    led.record_owner_seal("lease-1", "graph", "seal-g")
    led.final_fire("lease-1", "graph", "graph-result")

    # post-first-fire source change -> typed window-3 terminal
    led2 = BarrierLedger()
    b2_ = _bindings("lease-2")
    led2.prestart(b2_, "2026-09-01", _admission(b2_))
    _expect(lambda: led2.rebind_source("lease-2", "engine blob"),
            "BINDING_IMMUTABLE_AFTER_PRESTART")
    assert led2.state == "ACCRUAL"          # no terminal pre-fire
    _expect(lambda: led.rebind_source("lease-1", "engine blob"),
            "POST_FIRST_FIRE_SOURCE_CHANGE")
    assert led.state == "WINDOW3_TERMINAL"

    # cross-lane release + embargo on a fresh full lifecycle
    led3 = BarrierLedger()
    b3_ = _bindings("lease-3")
    led3.prestart(b3_, "2026-09-01", _admission(b3_))
    led3.close_support_barrier("lease-3", "2027-01-19", "non_analyst")
    for lane in ("graph", "mag1", "mf4"):
        led3.record_owner_seal("lease-3", lane, f"seal-{lane}")
    led3.final_fire("lease-3", "graph", "g-result")
    _expect(lambda: led3.release("lease-3"),
            "CROSS_LANE_RELEASE_BEFORE_TERMINALS")
    _expect(lambda: led3.read_result("lease-3", "graph"),
            "EMBARGO_VIOLATION")
    led3.final_fire("lease-3", "mag1", "m-result")
    led3.final_fire("lease-3", "mf4", "f-result")
    for lane in ("graph", "mag1", "mf4"):
        led3.record_verifier_pass("lease-3", lane, f"v-{lane}")
    led3.release("lease-3")
    assert led3.read_result("lease-3", "graph")["result_digest"] \
        == "g-result"
    led3.verify_chain()
    led3.events[1]["payload"] = {"tampered": True}
    _expect(led3.verify_chain, "LEDGER_CHAIN_BROKEN")

    # sec-5 selector: commit once, immutable; 4->3 refuses; Holm over S
    sel = BarrierLedger()
    full = {h: {"cp_lcb": c, "graph": GRAPH_MEMBERS}
            for h, c in (("B2A", 0.92), ("B2B", 0.85), ("B1B", 0.81),
                         ("B3A", 0.55))}
    _expect(lambda: sel.commit_selector(
        {h: full[h] for h in ("B2A", "B2B", "B1B")}),
        "SELECTOR_GRAPH_NOT_FULL")
    bad_graph = dict(full, B2B={"cp_lcb": 0.85,
                                "graph": ("B2A", "B2B", "B1B")})
    _expect(lambda: sel.commit_selector(bad_graph),
            "SELECTOR_GRAPH_NOT_FULL")
    S = sel.commit_selector(full)
    assert S == frozenset({"B2A", "B2B", "B1B"})
    _expect(lambda: sel.commit_selector(full),
            "SELECTOR_ALREADY_COMMITTED")
    _expect(lambda: sel.recertify_selector(full),
            "SELECTOR_RECERTIFICATION_REFUSED")
    h = sel.holm_graph_lane({"B2A": 0.001, "B2B": 0.03, "B1B": 0.04,
                             "B3A": 0.0001})
    assert h["verdicts"] == {"B2A": "REJECT", "B2B": "NO_REJECT",
                             "B1B": "NO_REJECT",
                             "B3A": "CANNOT_DETERMINE_NO_POWER"}, h
    _expect(lambda: sel.holm_graph_lane({"B2A": 0.01}),
            "HOLM_INPUT_INCOMPLETE")
    _expect(lambda: BarrierLedger().holm_graph_lane(
        {h_: 0.01 for h_ in GRAPH_MEMBERS}), "SELECTOR_NOT_COMMITTED")

    print("w2_barrier selftest: ALL PASS")


if __name__ == "__main__":
    _selftest()
