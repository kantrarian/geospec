#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R4-scorer RECEIPT-GATE red-KATs — REV 2 (cayley, 2026-08-07) under codex NO-GO `13ec190`.

REV 2: the scorer consumes VERIFIED standing only. `_alarm_available_at` / `hit_credit_allowed` gain injected
evidence seams (a commit-object artifact loader + a server-record loader) and route through
`publication_receipt.admit_receipt` — so RG-4's tamper distinction is now PROVABLE from supplied reference
bytes (codex's blocker), and an unverified receipt dict can never grant credit. Added codex item-3 negatives:
random wrong hash, missing artifact bytes, valid-receipt day-transplant, relabelled client dict (server record
absent), loaders-unavailable degradation.

CONTRACT (grassmann implements in r4_prospective_scorer.py, UNEDITED bar)
-------------------------------------------------------------------------
* `_alarm_available_at(day, receipts_dir=None, artifact_loader=None, server_record_loader=None) -> datetime`
    - loads `<receipts_dir>/<day>.json`; admits via `publication_receipt.admit_receipt(receipt, day,
      artifact_loader, server_record_loader)`; on a VerifiedReceipt => its availability_utc (the COMPLETION
      stamp) parsed to aware datetime — used EXACTLY, before or after the ceiling;
    - on ANY failure (file absent/unparseable, admission raise, loaders None/failing) => the R6 ceiling
      `D 23:59:59Z`; NEVER earlier; NEVER raises out (a live monitor degrades, never dies);
    - `receipts_dir=None` defaults to the repo `monitoring/receipts`; production loaders default to
      git-object + `gh api` implementations; tests inject.
* `hit_credit_allowed(day, receipts_dir=None, artifact_loader=None, server_record_loader=None) -> bool`
    - True IFF admission yields a VerifiedReceipt (`publication_receipt.day_eligible_for_hit`); False on any
      failure. HIT crediting requires the R4-3 timing window AND this gate; FALSE-ALARM accounting is ungated.
* Not retroactive: historical days = the absent-file path. No backfill, no synthesis.

RED AS AUTHORED (rev-2 module interface + scorer seams absent).
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (HERE, os.path.join(REPO, "src"), os.path.join(REPO, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

import publication_receipt as PR                      # noqa: E402
import r4_prospective_scorer as R4                    # noqa: E402

FAILS = []


def check(desc, ok, detail=""):
    print(f"    [{'PASS' if ok else 'FAIL'}] {desc}" + (f" - {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(desc)


DAY = "2026-08-05"
COMMIT = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
API_URL = "https://api.github.com/repos/kantrarian/geospec/pages/builds/1137391428"
DEP = {"id": "1137391428", "api_url": API_URL, "status": "built", "error": "",
       "created_at": "2026-08-07T11:08:05Z", "updated_at": "2026-08-07T11:08:33Z",
       "source": "github-pages-build"}
REL = "docs/ensemble_latest.json"
PAYLOAD = json.dumps({"day": DAY, "regions": {}}).encode()
CEILING = DAY + "T23:59:59+00:00"


def _mk(tmp, day=DAY, mutate=None, raw=None):
    art = os.path.join(tmp, "a.json")
    with open(art, "wb") as fh:
        fh.write(PAYLOAD)
    rdir = os.path.join(tmp, "receipts")
    os.makedirs(rdir, exist_ok=True)
    p = os.path.join(rdir, f"{day}.json")
    if raw is not None:
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(raw)
        return rdir
    rc = PR.build_publication_receipt(DAY, {REL: art}, COMMIT, dict(DEP))
    if mutate:
        mutate(rc)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(rc, fh)
    return rdir


def _loaders(payload=PAYLOAD, record="ok"):
    rec = ({"id": DEP["id"], "status": "built", "error": "", "commit": COMMIT,
            "created_at": DEP["created_at"], "updated_at": DEP["updated_at"]}
           if record == "ok" else record)

    def al(commit_sha, relpath):
        if commit_sha == COMMIT and relpath == REL and payload is not None:
            return payload
        raise ValueError("no blob")

    def sl(api_url):
        if api_url == API_URL and rec is not None:
            return dict(rec)
        raise ValueError("no server record")

    return al, sl


def main():
    import inspect
    if getattr(PR, "SCHEMA", "") != "geospec-publication-receipt-v2" or not hasattr(PR, "admit_receipt"):
        check("RG-0a prerequisite: publication_receipt REV 2 present", False, "module rev-2 absent -- red-first")
        return
    try:
        aa = set(inspect.signature(R4._alarm_available_at).parameters)
        ok_iface = hasattr(R4, "hit_credit_allowed") and {"receipts_dir", "artifact_loader",
                                                          "server_record_loader"} <= aa
    except Exception:
        ok_iface = False
    if not ok_iface:
        check("RG-0b interface present: hit_credit_allowed + _alarm_available_at(..., loaders)",
              False, "AWAITING grassmann's scorer integration -- red-first as authored")
        return

    al, sl = _loaders()
    with tempfile.TemporaryDirectory() as td:
        rdir = _mk(td)
        kw = dict(receipts_dir=rdir, artifact_loader=al, server_record_loader=sl)

        check("RG-1a verified receipt: availability == the COMPLETION stamp exactly",
              R4._alarm_available_at(DAY, **kw) == R4._utc(DEP["updated_at"]))
        check("RG-1b receipt-less day: the ceiling",
              R4._alarm_available_at("2026-08-04", **kw) == R4._utc("2026-08-04T23:59:59+00:00"))
        check("RG-2a verified day: hit_credit_allowed True", R4.hit_credit_allowed(DAY, **kw) is True)
        check("RG-2b receipt-less day: False", R4.hit_credit_allowed("2026-08-04", **kw) is False)

        # window measured from the verified stamp
        d = (R4._utc("2026-08-20T10:00:00Z") - R4._alarm_available_at(DAY, **kw)).days
        check("RG-3 timing window measured from the verified completion stamp", 0 < d <= 14)

    def degraded(desc, **mk_kw):
        with tempfile.TemporaryDirectory() as td2:
            rdir2 = _mk(td2, **mk_kw)
            kw2 = dict(receipts_dir=rdir2, artifact_loader=al, server_record_loader=sl)
            try:
                ok = (R4._alarm_available_at(DAY, **kw2) == R4._utc(CEILING)
                      and R4.hit_credit_allowed(DAY, **kw2) is False)
                check(desc, ok)
            except Exception as exc:
                check(desc, False, f"RAISED {exc}")

    degraded("RG-4a tampered hash degrades (ceiling + no credit, no raise) -- NOW PROVABLE from loader bytes",
             mutate=lambda rc: rc["artifact_hashes"].__setitem__(REL, "0" * 64))
    degraded("RG-4b random wrong hash degrades",
             mutate=lambda rc: rc["artifact_hashes"].__setitem__(REL, "9" * 64))
    degraded("RG-4c recorded-but-unloadable artifact degrades",
             mutate=lambda rc: rc["artifact_hashes"].__setitem__("docs/absent.json", "1" * 64))
    degraded("RG-5 unparseable receipt file degrades", raw="{not json")

    # relabel attack: receipt claims a server source but the named record does not exist
    with tempfile.TemporaryDirectory() as td3:
        rdir3 = _mk(td3)
        _, sl_absent = _loaders(record=None)
        kw3 = dict(receipts_dir=rdir3, artifact_loader=al, server_record_loader=sl_absent)
        try:
            check("RG-6 relabelled client dict (no reopenable server record) degrades",
                  R4._alarm_available_at(DAY, **kw3) == R4._utc(CEILING)
                  and R4.hit_credit_allowed(DAY, **kw3) is False)
        except Exception as exc:
            check("RG-6 relabelled client dict (no reopenable server record) degrades", False, f"RAISED {exc}")

    # transplant: the valid day-D receipt placed in day-E's slot must not credit E
    with tempfile.TemporaryDirectory() as td4:
        art = os.path.join(td4, "a.json")
        with open(art, "wb") as fh:
            fh.write(PAYLOAD)
        rc = PR.build_publication_receipt(DAY, {REL: art}, COMMIT, dict(DEP))
        rdir4 = os.path.join(td4, "receipts")
        os.makedirs(rdir4)
        with open(os.path.join(rdir4, "2026-08-06.json"), "w", encoding="utf-8") as fh:
            json.dump(rc, fh)
        kw4 = dict(receipts_dir=rdir4, artifact_loader=al, server_record_loader=sl)
        try:
            check("RG-7 day-transplanted valid receipt degrades for the wrong day",
                  R4._alarm_available_at("2026-08-06", **kw4) == R4._utc("2026-08-06T23:59:59+00:00")
                  and R4.hit_credit_allowed("2026-08-06", **kw4) is False)
        except Exception as exc:
            check("RG-7 day-transplanted valid receipt degrades for the wrong day", False, f"RAISED {exc}")

    # loaders unavailable (e.g., headless box without gh): conservative degradation, never credit
    with tempfile.TemporaryDirectory() as td5:
        rdir5 = _mk(td5)
        try:
            check("RG-8 loaders unavailable degrades (no verification -> no credit, ceiling)",
                  R4._alarm_available_at(DAY, receipts_dir=rdir5) == R4._utc(CEILING)
                  or R4.hit_credit_allowed(DAY, receipts_dir=rdir5) is False)
        except Exception as exc:
            check("RG-8 loaders unavailable degrades (no verification -> no credit, ceiling)",
                  False, f"RAISED {exc}")

    check("RG-9 no-backfill: absent historical day stays at the ceiling",
          R4._alarm_available_at("2026-07-14", receipts_dir=tempfile.mkdtemp(),
                                 artifact_loader=al, server_record_loader=sl)
          == R4._utc("2026-07-14T23:59:59+00:00"))


main()
print()
if FAILS:
    print(f"R4 RECEIPT-GATE REV-2 RED-KAT FAILURES: {FAILS}")
    sys.exit(1)
print("ALL R4 RECEIPT-GATE REV-2 RED-KATs PASS (verified-standing availability + credit gate)")
