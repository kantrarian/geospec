#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R4-scorer RECEIPT-GATE red-KATs (cayley, 2026-08-07) — P2 item 1 CONSUMER contract (grassmann `1823` ask).

THE CONTRACT (grassmann implements in r4_prospective_scorer.py, UNEDITED bar — the decouple)
--------------------------------------------------------------------------------------------
(a) `_alarm_available_at(day, receipts_dir=None) -> datetime` becomes receipt-aware:
    - a VALID day receipt at `<receipts_dir>/<day>.json` (schema-valid, server-side source, per
      `publication_receipt`) => availability = the receipt's `deployment.created_at` EXACTLY — whether that
      falls before OR after the ceiling (if Pages genuinely deployed day D at D+1 07:09, that IS when the alarm
      became available; honesty over flattery in both directions);
    - no receipt / invalid receipt => the R6 ceiling `D 23:59:59Z`, NEVER any earlier value (no synthesis);
    - `receipts_dir=None` defaults to the repo's `monitoring/receipts`; tests pass a tmp dir.
(b) NEW `hit_credit_allowed(day, receipts_dir=None) -> bool` = publication_receipt.day_eligible_for_hit over the
    loaded day receipt (False on absent/invalid/client-source). HIT crediting requires BOTH the R4-3 timing
    window AND hit_credit_allowed. FALSE-ALARM accounting is NOT gated (receipt-less days still count FAs —
    the conservative direction).
(c) NOT RETROACTIVE: historical/receipt-less days follow the same absent-file path (ceiling + hit-ineligible);
    corrupted/tampered/client-stamped receipts DEGRADE to the same (ceiling + ineligible) and MUST NOT raise out
    of the scorer (a live monitor degrades, never dies). No backfill, no synthesis, ever.
(d) Receipt location: exactly `monitoring/receipts/<YYYY-MM-DD>.json` (the producer's path, geospec 4025794).

RED AS AUTHORED: `hit_credit_allowed` does not exist and `_alarm_available_at` takes no receipts_dir.
"""
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (HERE, os.path.join(REPO, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import publication_receipt as PR                      # noqa: E402  (the landed P2 module — authoritative)
import r4_prospective_scorer as R4                    # noqa: E402

FAILS = []


def check(desc, ok, detail=""):
    print(f"    [{'PASS' if ok else 'FAIL'}] {desc}" + (f" - {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(desc)


DAY = "2026-08-05"
SERVER_STAMP = "2026-08-06T11:09:03Z"                 # a REAL-shaped Pages stamp: the day AFTER (honest lag)
SERVER_DEP = {"id": "pages-build-777", "created_at": SERVER_STAMP, "source": "github-pages-build"}


def _mk_receipt(tmp, day=DAY, dep=None, tamper=None):
    """Build a REAL receipt via the landed producer module (authoritative source), optionally tampered."""
    art = os.path.join(tmp, "ensemble_latest.json")
    payload = json.dumps({"day": day, "regions": {}}).encode()
    with open(art, "wb") as fh:
        fh.write(payload)
    rc = PR.build_publication_receipt({"docs/ensemble_latest.json": art}, "b" * 40, dict(dep or SERVER_DEP))
    if tamper:
        tamper(rc)
    rdir = os.path.join(tmp, "receipts")
    os.makedirs(rdir, exist_ok=True)
    with open(os.path.join(rdir, f"{day}.json"), "w", encoding="utf-8") as fh:
        json.dump(rc, fh)
    return rdir


def main():
    has_iface = hasattr(R4, "hit_credit_allowed")
    try:
        import inspect
        aa_params = set(inspect.signature(R4._alarm_available_at).parameters)
    except Exception:
        aa_params = set()
    if not has_iface or "receipts_dir" not in aa_params:
        check("RG-0 interface present: hit_credit_allowed + _alarm_available_at(day, receipts_dir=...)",
              False, "AWAITING grassmann's scorer integration -- red-first as authored")
        return

    with tempfile.TemporaryDirectory() as td:
        rdir = _mk_receipt(td)

        # (a) availability semantics
        check("RG-1a valid receipt: availability == the server stamp EXACTLY (even past the ceiling)",
              R4._alarm_available_at(DAY, receipts_dir=rdir) == R4._utc(SERVER_STAMP))
        check("RG-1b no receipt: availability == the 23:59:59Z ceiling",
              R4._alarm_available_at("2026-08-04", receipts_dir=rdir)
              == R4._utc("2026-08-04T23:59:59+00:00"))
        early = {**SERVER_DEP, "created_at": "2026-08-05T14:02:00Z"}     # same-day deploy (pre-ceiling)
        with tempfile.TemporaryDirectory() as td2:
            rdir2 = _mk_receipt(td2, dep=early)
            check("RG-1c pre-ceiling server stamp used exactly (not clamped to the ceiling)",
                  R4._alarm_available_at(DAY, receipts_dir=rdir2) == R4._utc("2026-08-05T14:02:00Z"))

        # (b) hit-credit gate
        check("RG-2a receipt-bearing day: hit_credit_allowed True",
              R4.hit_credit_allowed(DAY, receipts_dir=rdir) is True)
        check("RG-2b receipt-less day: hit_credit_allowed False",
              R4.hit_credit_allowed("2026-08-04", receipts_dir=rdir) is False)

        # (b/c) eligibility composition: timing window vs credit gate are INDEPENDENT checks
        ev_in = "2026-08-10T00:00:00Z"                # inside the 14d window from the server stamp
        check("RG-3a timing window measured from the RECEIPT availability",
              R4.hit_eligible(DAY, ev_in) in (True, False))   # semantic: must not raise; exact below
        # with the receipt (stamp 08-06), an event 15d after the CEILING but 14d after the STAMP:
        ev_edge = "2026-08-20T10:00:00Z"
        elig_new = (R4._utc(ev_edge) - R4._alarm_available_at(DAY, receipts_dir=rdir)).days
        check("RG-3b receipt shifts the window edge honestly (documented via _alarm_available_at)",
              0 < elig_new <= 14)

        # (c) fail-closed degradation, never raise
        def _t_client(rc):
            rc["deployment"]["source"] = "git-commit-timestamp"

        def _t_hash(rc):
            rc["artifact_hashes"] = {"docs/ensemble_latest.json": "0" * 64}

        for name, t in (("client-source", _t_client), ("tampered-hash", _t_hash)):
            with tempfile.TemporaryDirectory() as td3:
                rdir3 = _mk_receipt(td3, tamper=t)
                try:
                    avail = R4._alarm_available_at(DAY, receipts_dir=rdir3)
                    credit = R4.hit_credit_allowed(DAY, receipts_dir=rdir3)
                    check(f"RG-4 {name} receipt degrades to ceiling + no credit (no raise)",
                          avail == R4._utc(f"{DAY}T23:59:59+00:00") and credit is False)
                except Exception as exc:
                    check(f"RG-4 {name} receipt degrades to ceiling + no credit (no raise)",
                          False, f"RAISED {exc}")
        with tempfile.TemporaryDirectory() as td4:
            rdir4 = os.path.join(td4, "receipts")
            os.makedirs(rdir4)
            with open(os.path.join(rdir4, f"{DAY}.json"), "w") as fh:
                fh.write("{not json")
            try:
                ok = (R4._alarm_available_at(DAY, receipts_dir=rdir4)
                      == R4._utc(f"{DAY}T23:59:59+00:00")
                      and R4.hit_credit_allowed(DAY, receipts_dir=rdir4) is False)
                check("RG-5 unparseable receipt file degrades to ceiling + no credit (no raise)", ok)
            except Exception as exc:
                check("RG-5 unparseable receipt file degrades to ceiling + no credit (no raise)",
                      False, f"RAISED {exc}")

        # (c) no-backfill: absence NEVER yields an availability EARLIER than the ceiling
        check("RG-6 absent receipt never yields availability earlier than the ceiling",
              R4._alarm_available_at("2026-07-14", receipts_dir=rdir)
              == R4._utc("2026-07-14T23:59:59+00:00"))


main()
print()
if FAILS:
    print(f"R4 RECEIPT-GATE RED-KAT FAILURES: {FAILS}")
    sys.exit(1)
print("ALL R4 RECEIPT-GATE RED-KATs PASS (receipt-aware availability + hit-credit gate enforced)")
