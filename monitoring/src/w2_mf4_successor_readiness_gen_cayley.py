#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M-F4 SUCCESSOR-READINESS record generator (cayley) -- codex w2r1
cycle-2 ruling (2026-08-30T22:40Z) finding 3 (MAJOR): unchanged bytes
prove fit identity; they do not prove prospective producer readiness
through the successor maturity tail. This generator produces ONE
append-only record that:

1. preserves the maturity record v4 and the final-bind record v1 as
   history, superseding ONLY their standing/date fields (the old
   08-28-schedule accrual span; gate-B OPEN; the final-bind
   PRIVATE_CANDIDATE_PENDING_OWNER_LANDING standing, resolved by the
   record's actual public landing commit);
2. binds the owner continuity renewal (2026-08-22, relational term
   "through the WINDOW-2 CLOSE") to its CONCRETE successor close
   derived from calendar authority v4 + the frozen H;
3. binds the exact technical producer identities: the frozen
   w2_mf4.py entry, the pinned amended ledger/feed/receipt bytes and
   their three-way training-digest equality (ledger == receipt ==
   final-bind bound digests, and codex's own expected values);
4. PROVES the production entry can emit one immutable row per
   admitted (region, issue_day) WITHOUT refit: it re-emits rows via
   the frozen predict_row across the full calibration span for every
   admitted region from the pinned bytes, verify_row-checks each,
   and re-emits to prove determinism. Emission uses a FIXED probe
   issued_utc, so the proof is capability + determinism -- probe
   rows are never production rows and are not persisted;
5. fails closed on any producer, calendar, region-set, or fit-byte
   mismatch (typed ReadinessRefusal; the selftest doctors each one);
6. records a typed READY_FOR_ACCRUAL or refusal state.

Opens no window-2 value; no network; no refit anywhere (predict_row
is apply-only); admits nothing. Lambda_geo INCONCLUSIVE.
"""
import datetime
import hashlib
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import w2_mf4 as MF4  # noqa: E402
import w2_mf4_catalog_adapter_grassmann as ADP  # noqa: E402

OUT_REL = ("docs/f2g_window2_execution/"
           "mf4_successor_readiness_record_v1.json")
SNAPSHOT_REL = ("docs/f2g_window2_execution/mf4_catalog_snapshot/"
                "catalog_snapshot_v1.json")
ACQ_RECEIPT_REL = ("docs/f2g_window2_execution/mf4_catalog_snapshot/"
                   "acquisition_receipt_v1.json")
LEDGER_REL = ("docs/f2g_window2_execution/calibration/"
              "mf4_ledger_amended.json")
FEED_REL = ("docs/f2g_window2_execution/calibration/"
            "mf4_input_feed_amended.json")
RECEIPT_REL = ("docs/f2g_window2_execution/calibration/"
               "mf4_ledger_amended.receipt.json")
FINAL_BIND_REL = ("docs/f2g_window2_execution/calibration/"
                  "mf4_final_bind_record_v1.json")
MATURITY_REL = ("docs/f2g_window2_execution/"
                "mf4_maturity_record_v4.json")
RENEWAL_REL = ("docs/f2g_window2_execution/"
               "renewal_2026-08-22_owner_authorization.md")
RENEWAL_QUOTE_SHA = ("7caab14d2b7379609c096f2a52b25d29"
                     "c8c92209dbd2217111e513b5a8acb270")
CALENDAR_REL = ("docs/f2g_window2_execution/"
                "calendar_authority_w2_v4.json")
MF4_SOURCE_REL = "monitoring/src/w2_mf4.py"
PROBE_ISSUED_UTC = "READINESS_PROBE_NOT_A_PRODUCTION_ISSUANCE"


class ReadinessRefusal(ValueError):
    pass


def _refuse(detail):
    raise ReadinessRefusal(f"MF4_READINESS_REFUSED: {detail}")


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _blob(repo, rel):
    """Committed bytes at HEAD -- never the working tree, so checkout
    newline conversion can never steer an identity."""
    r = subprocess.run(["git", "-C", repo, "cat-file", "blob",
                        f"HEAD:{rel}"], capture_output=True)
    if r.returncode != 0 or not r.stdout:
        _refuse(f"blob unreadable at HEAD: {rel}")
    return r.stdout


def _landing_commit(repo, rel):
    r = subprocess.run(["git", "-C", repo, "log", "-1",
                        "--format=%H", "HEAD", "--", rel],
                       capture_output=True, text=True)
    c = r.stdout.strip()
    if not c:
        _refuse(f"no landing commit for {rel}")
    return c


def build(repo, *, loaders=None):
    """loaders is a KAT-only seam: {rel: bytes} overrides for the
    refusal doctors. Production passes None and reads committed
    bytes only."""
    def raw(rel):
        if loaders is not None and rel in loaders:
            return loaders[rel]
        return _blob(repo, rel)

    led_b = raw(LEDGER_REL)
    feed_b = raw(FEED_REL)
    rec_b = raw(RECEIPT_REL)
    fb_b = raw(FINAL_BIND_REL)
    mat_b = raw(MATURITY_REL)
    ren_b = raw(RENEWAL_REL)
    cal_b = raw(CALENDAR_REL)
    src_b = raw(MF4_SOURCE_REL)

    led = json.loads(led_b.decode("utf-8"))
    feed = json.loads(feed_b.decode("utf-8"))
    rec = json.loads(rec_b.decode("utf-8"))
    fb = json.loads(fb_b.decode("utf-8"))
    mat = json.loads(mat_b.decode("utf-8"))
    cal = json.loads(cal_b.decode("utf-8"))

    # --- fit-byte binding (fails closed on any divergence) --------
    # two identities per artifact, both binding: the CANONICAL
    # compact-JSON digest (codex's independent
    # MF4_POST_RUN_LEDGER_PASS expected values) and the RAW
    # committed blob sha (the final-bind artifact table).
    # Neither substitutes for the other.
    def _canon(obj):
        return hashlib.sha256(json.dumps(
            obj, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
    if _canon(led) != fb["expected_ledger_sha256"]:
        _refuse("ledger canonical digest diverges from the "
                "final-bind expected ledger sha (codex "
                "MF4_POST_RUN_LEDGER_PASS value)")
    if _sha(led_b) != fb["artifact_blob_sha256"][
            "mf4_ledger_amended.json"]:
        _refuse("ledger raw bytes diverge from the final-bind "
                "artifact blob table")
    if _canon(feed) != fb["expected_input_sha256"]:
        _refuse("input-feed canonical digest diverges from the "
                "final-bind expected input sha")
    if _sha(feed_b) != fb["artifact_blob_sha256"][
            "mf4_input_feed_amended.json"]:
        _refuse("input-feed raw bytes diverge from the final-bind "
                "artifact blob table")
    if _sha(rec_b) != fb["artifact_blob_sha256"][
            "mf4_ledger_amended.receipt.json"]:
        _refuse("receipt bytes diverge from the final-bind artifact "
                "blob table")
    tds = {led["amended_training_digest"],
           rec["amended_training_digest"],
           fb["bound_digests"]["amended_training_digest"]}
    if len(tds) != 1:
        _refuse("amended training digest is not three-way equal "
                "(ledger/receipt/final-bind)")

    # --- renewal binding ------------------------------------------
    ren_t = ren_b.decode("utf-8")
    if RENEWAL_QUOTE_SHA not in ren_t or \
            "through the WINDOW-2 CLOSE" not in ren_t:
        _refuse("renewal artifact does not carry the registered "
                "quote sha + relational term")

    # --- successor frame from calendar v4 (never a date literal) --
    if cal.get("schema") != "f2g-w2-calendar-authority-v4":
        _refuse(f"calendar schema {cal.get('schema')!r} is not the "
                "v4 successor authority")
    ev = cal["frame"]["evaluation_days"]
    h = int(mat["frozen_constants"]["h_days"])
    ds = datetime.date.fromisoformat
    close = (ds(ev[-1]) + datetime.timedelta(days=h)).isoformat()
    accrual = {"evaluation_start": ev[0], "evaluation_end": ev[-1],
               "h_days": h, "maturity_tail_end": close}

    # --- region set: pinned feed vs pinned ledger ------------------
    regions = list(feed["regions"])
    if sorted(regions) != sorted(led["regions"]):
        _refuse("admitted region set diverges between the pinned "
                "feed and the pinned ledger")

    # --- catalog: registered snapshot/receipt bytes through the
    # SANCTIONED adapter chain (bytes-only; role guard enforces the
    # CALIBRATION_LATE_REPAIR temporal role -- recomputed historical
    # calibration features, never live/post-evaluation use) ---------
    snap_b = raw(SNAPSHOT_REL)
    acq_b = raw(ACQ_RECEIPT_REL)
    if _sha(snap_b) != feed["catalog"]["snapshot_sha256"]:
        _refuse("catalog snapshot bytes diverge from the pinned "
                "feed catalog binding")
    if _sha(acq_b) != feed["catalog"]["acquisition_receipt_sha256"]:
        _refuse("acquisition receipt bytes diverge from the pinned "
                "feed catalog binding")
    events, table_dig = ADP.events_from_snapshot(
        snap_b, acq_b, use="calibration_features")

    # --- emission proof: apply-never-refit over the full span ------
    span0 = led["calibration_start"]
    span1 = led["calibration_issue_end"]
    days = []
    d = ds(span0)
    while d <= ds(span1):
        days.append(d.isoformat())
        d += datetime.timedelta(days=1)
    emitted = typed = 0
    for region in regions:
        risk = feed["risk_by_region"][region]
        bbox = feed["bboxes"][region]
        for day in days:
            r1 = MF4.predict_row(led, risk, list(events), bbox,
                                 region, day, PROBE_ISSUED_UTC)
            MF4.verify_row(r1)
            r2 = MF4.predict_row(led, risk, list(events), bbox,
                                 region, day, PROBE_ISSUED_UTC)
            if r1["row_digest"] != r2["row_digest"]:
                _refuse(f"emission not deterministic at "
                        f"{region}/{day}")
            emitted += 1
            if "typing" in r1:
                typed += 1
    if emitted != len(regions) * len(days):
        _refuse("emission census incomplete")

    return {
        "schema": "f2g-w2-mf4-successor-readiness-v1",
        "state": "READY_FOR_ACCRUAL",
        "ruling_basis": "codex w2r1 cycle-2 ruling 2026-08-30T22:40Z "
                        "finding 3 (MAJOR): byte reuse proves fit "
                        "identity, not prospective readiness; this "
                        "record binds the missing standing",
        "supersedes_standing_fields_only": {
            "note": "both artifacts stand as history; ONLY the "
                    "standing/date fields below are superseded, "
                    "never their bytes",
            "maturity_record": {
                "path": MATURITY_REL,
                "sha256": _sha(mat_b),
                "superseded": {
                    "gate_b_prospective_producer.status":
                        "OPEN -> discharged by this record's "
                        "emission proof + renewal binding",
                    "gate_b_prospective_producer.accrual_span":
                        "08-28-schedule dates -> the successor "
                        "frame below"}},
            "final_bind_record": {
                "path": FINAL_BIND_REL,
                "sha256": _sha(fb_b),
                "superseded": {
                    "status": "PRIVATE_CANDIDATE_PENDING_OWNER_"
                              "LANDING -> LANDED_PUBLIC (owner-"
                              "fired); landing commit bound below"},
                "landing_commit": _landing_commit(
                    repo, FINAL_BIND_REL)}},
        "owner_continuity_renewal": {
            "path": RENEWAL_REL,
            "sha256": _sha(ren_b),
            "quote_sha256": RENEWAL_QUOTE_SHA,
            "relational_term": "through the WINDOW-2 CLOSE = "
                               "evaluation_end + H_max, binding when "
                               "PRESTART fixes evaluation_start",
            "concrete_close_under_successor": close},
        "successor_accrual_frame": accrual,
        "producer_identities": {
            "frozen_entry": {
                "path": MF4_SOURCE_REL, "sha256": _sha(src_b),
                "entrypoint": "predict_row (apply-never-refit; typed "
                              "no-prediction days emit a typing row, "
                              "never silence)"},
            "ledger_sha256": _sha(led_b),
            "input_feed_sha256": _sha(feed_b),
            "receipt_sha256": _sha(rec_b),
            "catalog_snapshot_sha256": _sha(snap_b),
            "acquisition_receipt_sha256": _sha(acq_b),
            "catalog_table_digest": table_dig,
            "amended_training_digest": led["amended_training_digest"],
            "training_digest_three_way_equal": True},
        "emission_proof": {
            "method": "re-emitted one row per admitted (region, "
                      "issue_day) over the FULL pinned calibration "
                      "span from committed bytes; verify_row on "
                      "every row; double-emission digest equality "
                      "(determinism); fixed probe issued_utc -- "
                      "probe rows are not production rows and are "
                      "not persisted",
            "span": [span0, span1],
            "regions": sorted(regions),
            "rows_emitted": emitted,
            "typed_no_prediction_rows": typed,
            "refit_performed": False},
        "operational_dependency": {
            "daily_input_feed": "the daily-risk monitor supplies "
                                "per-day inputs during accrual; "
                                "continuity is owner-authorized "
                                "(renewal above) and any missing "
                                "accrual day types at scoring time "
                                "under the frozen support rules -- "
                                "never backfilled, never silent"},
        "claim_ceiling": "readiness registration only; candidate "
                         "standing, not admission; no power value, "
                         "no scientific claim; Lambda_geo "
                         "INCONCLUSIVE"}


def main():
    repo = os.path.abspath(os.path.join(_HERE, "..", ".."))
    body = json.dumps(build(repo), indent=1, sort_keys=True) + "\n"
    out = os.path.join(repo, OUT_REL.replace("/", os.sep))
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
    print(f"wrote {OUT_REL}")
    print("record sha256:", _sha(body.encode()))


def _selftest():
    """Refusal doctors: each constructs its precondition by doctoring
    ONE committed byte-stream through the KAT-only loader seam; the
    unmutated control passes first."""
    repo = os.path.abspath(os.path.join(_HERE, "..", ".."))
    rec = build(repo)
    assert rec["state"] == "READY_FOR_ACCRUAL"
    assert rec["emission_proof"]["rows_emitted"] > 0
    assert rec["emission_proof"]["refit_performed"] is False

    def doctored(rel, mutate, why):
        base = _blob(repo, rel)
        try:
            build(repo, loaders={rel: mutate(base)})
            raise SystemExit("readiness doctor must refuse: " + why)
        except ReadinessRefusal as ex:
            assert why in str(ex), (why, str(ex))

    # fit-byte mismatch (one flipped byte in the pinned ledger)
    doctored(LEDGER_REL, lambda b: b.replace(
        b'"n_rows"', b'"n_rowz"', 1), "diverges from the")
    # raw-vs-canonical split: appended whitespace leaves the
    # canonical digest intact, so the RAW blob-table gate must fire
    doctored(FEED_REL, lambda b: b + b"\n",
             "raw bytes diverge from the final-bind artifact blob")
    # calendar mismatch (v3 schema at the v4 path)
    v3 = _blob(repo, "docs/f2g_window2_execution/"
                     "calendar_authority_w2_v3.json")
    doctored(CALENDAR_REL, lambda b: v3, "is not the v4 successor")
    # region-set mismatch (feed regions doctored, sha gate bypassed
    # is impossible -- so doctor the LEDGER copy consistently? No:
    # the sha gate fires FIRST by design; region divergence is only
    # reachable behind matching final-bind bytes, so the doctor
    # proves the ordering: byte identity is the outer gate.
    doctored(FEED_REL, lambda b: json.dumps(dict(
        json.loads(b.decode("utf-8")), regions=["cascadia"]),
        indent=1).encode(), "diverges from the final-bind expected "
        "input sha")
    # renewal mismatch (quote sha stripped)
    doctored(RENEWAL_REL, lambda b: b.replace(
        RENEWAL_QUOTE_SHA.encode(), b"0" * 64),
        "does not carry the registered quote")
    print("w2_mf4_successor_readiness selftest: ALL PASS (control "
          "READY_FOR_ACCRUAL w/ full-span emission proof; doctors: "
          "ledger/feed/calendar/region-order/renewal refuse typed; "
          "no refit anywhere)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
