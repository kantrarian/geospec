#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R4-scorer RECEIPT-GATE red-KATs — REV 3 (cayley, 2026-08-07) under codex WORKS-WITH-FIX `d76842a`.

REV 3 repairs (codex findings #2 + #5-RG8):
  #2 BLOCKER  the bar now freezes the STANDING-BEARING SCORING PATH, not only helpers: `score()` (and
      `run()`) consume ONE injected standing resolver whose DEFAULT IS FAIL-CLOSED; `eligible_episode`
      requires, FOR THE SAME alarm date d, BOTH the timing window measured from the resolver's
      availability AND the resolver's credit gate (no any-date composition). End-to-end RG-10 cases run
      a real Episode + Event through `score()` + `molchan()`: valid evidence yields exactly one hit;
      hash-invalid / missing receipt / wrong day / absent server record / unavailable loaders yield a
      miss and DO NOT increment molchan.pooled.hits. RG-12 pins the explicit deterministic resolver
      that the pure 23-KAT battery injects (episode/exclusion logic tests keep their subjects; the KAT
      file's score() call sites gain the explicit resolver — production NEVER gets an allow-all default).
  RG-8        `or` -> `and`: unavailable loaders must give the ceiling AND no credit (an unsafe early
      timestamp can no longer hide behind a false credit).

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
* `receipt_standing_resolver(receipts_dir=None, artifact_loader=None, server_record_loader=None) -> callable`
    - returns `resolver(day) -> (availability: aware datetime, credit_allowed: bool)`, routing through the
      two helpers above (i.e., through admit_receipt — the ONLY standing path).
* `score(episodes, mainshocks, exclusions=None, today=None, window=..., standing_resolver=None) -> dict`
    - `standing_resolver=None` (the DEFAULT) is FAIL-CLOSED: every day resolves to (ceiling, False) —
      never an implicit allow. Passing a resolver is the ONLY way any day earns credit.
    - `eligible_episode` credits an episode ONLY via an alarm date d whose resolver availability makes the
      event timing-eligible (0 < origin - availability(d) <= window) AND whose credit gate is True — BOTH
      on the SAME d. FALSE-ALARM/terminal accounting is unchanged (ungated).
* `run(end_date=None, standing_resolver=None)` — threads the resolver into score(); when None it constructs
    `receipt_standing_resolver()` with PRODUCTION defaults (repo receipts dir + git/gh loaders), which
    fail closed per-day. Historical receipt-less days = the ceiling path; no backfill, no synthesis.
* The existing 23-KAT battery (test_r4_prospective_scorer.py) stays green by INJECTING the explicit
    deterministic resolver `lambda d: (_utc(d + "T23:59:59+00:00"), True)` at its score() call sites
    (its subjects are episode/exclusion logic, not standing) — the resolver appears in the test file,
    NEVER as a production default.

RED AS AUTHORED (rev-3 module interface + scorer integration seams absent).
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
REL_ENS = "docs/ensemble_latest.json"
REL_CSV = "docs/data.csv"
PAYLOAD_ENS = json.dumps({"date": DAY, "regions": {}}).encode()
PAYLOAD_CSV = b"date,region,tier,risk,confidence,methods,agreement\n2026-08-05,r1,2,0.61,0.8,4,0.75\n"
CEILING = DAY + "T23:59:59+00:00"

_DEFAULT_RECORD = object()      # codex fix #1 discipline, mirrored here


def _mk(tmp, day=DAY, mutate=None, raw=None):
    paths = {}
    for rel, data in ((REL_ENS, PAYLOAD_ENS), (REL_CSV, PAYLOAD_CSV)):
        f = os.path.join(tmp, rel.replace("/", "__"))
        with open(f, "wb") as fh:
            fh.write(data)
        paths[rel] = f
    rdir = os.path.join(tmp, "receipts")
    os.makedirs(rdir, exist_ok=True)
    p = os.path.join(rdir, f"{day}.json")
    if raw is not None:
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(raw)
        return rdir
    rc = PR.build_publication_receipt(DAY, paths, COMMIT, dict(DEP))
    if mutate:
        mutate(rc)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(rc, fh)
    return rdir


def _loaders(ens=PAYLOAD_ENS, csvb=PAYLOAD_CSV, record=_DEFAULT_RECORD):
    rec = ({"id": DEP["id"], "status": "built", "error": "", "commit": COMMIT,
            "created_at": DEP["created_at"], "updated_at": DEP["updated_at"]}
           if record is _DEFAULT_RECORD else record)
    blobs = {REL_ENS: ens, REL_CSV: csvb}

    def al(commit_sha, relpath):
        if commit_sha == COMMIT and blobs.get(relpath) is not None:
            return blobs[relpath]
        raise ValueError("no blob")

    def sl(api_url):
        if api_url == API_URL and rec is not None:
            return dict(rec)
        raise ValueError("no server record")

    return al, sl


def main():
    import inspect
    if getattr(PR, "SCHEMA", "") != "geospec-publication-receipt-v2" or not hasattr(PR, "admit_receipt") \
            or not hasattr(PR, "MANDATORY_ARTIFACTS"):
        check("RG-0a prerequisite: publication_receipt REV 3 present", False, "module rev-3 absent -- red-first")
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
    try:
        ok_seams = (hasattr(R4, "receipt_standing_resolver")
                    and "standing_resolver" in inspect.signature(R4.score).parameters
                    and "standing_resolver" in inspect.signature(R4.run).parameters)
    except Exception:
        ok_seams = False
    if not ok_seams:
        check("RG-0c integration seams present: receipt_standing_resolver + score()/run(standing_resolver)",
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

        # the resolver routes through the SAME admission path
        res = R4.receipt_standing_resolver(**kw)
        avail, credit = res(DAY)
        avail0, credit0 = res("2026-08-04")
        check("RG-3b receipt_standing_resolver: (completion stamp, True) verified / (ceiling, False) absent",
              avail == R4._utc(DEP["updated_at"]) and credit is True
              and avail0 == R4._utc("2026-08-04T23:59:59+00:00") and credit0 is False)

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

    degraded("RG-4a tampered hash degrades (ceiling + no credit, no raise) -- PROVABLE from loader bytes",
             mutate=lambda rc: rc["artifact_hashes"].__setitem__(REL_ENS, "0" * 64))
    degraded("RG-4b random wrong hash degrades",
             mutate=lambda rc: rc["artifact_hashes"].__setitem__(REL_ENS, "9" * 64))
    degraded("RG-4c recorded-but-unloadable artifact degrades",
             mutate=lambda rc: rc["artifact_hashes"].__setitem__("docs/r5_daily.json", "1" * 64))
    degraded("RG-4d receipt.day edited AWAY in the correct slot degrades (field mismatch)",
             mutate=lambda rc: rc.__setitem__("day", "2026-08-06"))
    # (the stronger cross-day forgery -- field edited to MATCH the wrong slot -- is RG-7b)
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

    # transplants: the valid day-D receipt in day-E's slot; and the edited-day forgery in E's slot
    with tempfile.TemporaryDirectory() as td4:
        rdir4 = _mk(td4, day="2026-08-06")                       # valid day-D receipt, E's slot
        kw4 = dict(receipts_dir=rdir4, artifact_loader=al, server_record_loader=sl)
        try:
            check("RG-7 day-transplanted valid receipt degrades for the wrong day",
                  R4._alarm_available_at("2026-08-06", **kw4) == R4._utc("2026-08-06T23:59:59+00:00")
                  and R4.hit_credit_allowed("2026-08-06", **kw4) is False)
        except Exception as exc:
            check("RG-7 day-transplanted valid receipt degrades for the wrong day", False, f"RAISED {exc}")
    with tempfile.TemporaryDirectory() as td4b:
        rdir4b = _mk(td4b, day="2026-08-06", mutate=lambda rc: rc.__setitem__("day", "2026-08-06"))
        kw4b = dict(receipts_dir=rdir4b, artifact_loader=al, server_record_loader=sl)
        try:
            check("RG-7b edited-day forgery in the wrong slot degrades (reopened carrier date wins)",
                  R4._alarm_available_at("2026-08-06", **kw4b) == R4._utc("2026-08-06T23:59:59+00:00")
                  and R4.hit_credit_allowed("2026-08-06", **kw4b) is False)
        except Exception as exc:
            check("RG-7b edited-day forgery in the wrong slot degrades (reopened carrier date wins)",
                  False, f"RAISED {exc}")

    # loaders unavailable (e.g., headless box without gh): conservative degradation, never credit
    with tempfile.TemporaryDirectory() as td5:
        rdir5 = _mk(td5)
        try:
            check("RG-8 loaders unavailable degrades (ceiling AND no credit -- codex #5: 'or' was unsafe)",
                  R4._alarm_available_at(DAY, receipts_dir=rdir5) == R4._utc(CEILING)
                  and R4.hit_credit_allowed(DAY, receipts_dir=rdir5) is False)
        except Exception as exc:
            check("RG-8 loaders unavailable degrades (ceiling AND no credit -- codex #5: 'or' was unsafe)",
                  False, f"RAISED {exc}")

    check("RG-9 no-backfill: absent historical day stays at the ceiling",
          R4._alarm_available_at("2026-07-14", receipts_dir=tempfile.mkdtemp(),
                                 artifact_loader=al, server_record_loader=sl)
          == R4._utc("2026-07-14T23:59:59+00:00"))

    # ==================================================================================
    # RG-10 -- END-TO-END: real Episode + Event through score() + molchan() (codex #2)
    # ==================================================================================
    EVT = R4.Event("evt_e2e", "2026-08-08", 32.8, 130.7, 5.8, region="r1",
                   origin_utc="2026-08-08T12:00:00Z")
    SERIES = {"r1": [("2026-08-04", 0), (DAY, 2), ("2026-08-06", 0), ("2026-08-07", 0),
                     ("2026-08-08", 0)]}

    def e2e(resolver):
        """Fresh fixtures each call (score mutates episode state)."""
        ep = R4.Episode("r1", DAY, DAY, 1, False, alarm_dates=(DAY,))
        sc = R4.score([ep], [EVT], [], today="2026-08-20", standing_resolver=resolver)
        mol = R4.molchan(SERIES, [ep], sc["outcomes"], [])
        out = sc["outcomes"][0]["outcome"] if sc["outcomes"] else "NO-OUTCOME"
        return out, mol["pooled"]["hits"]

    def e2e_receipts(desc, want_hit, **mk_kw):
        with tempfile.TemporaryDirectory() as td6:
            rdir6 = _mk(td6, **mk_kw)
            resolver = R4.receipt_standing_resolver(receipts_dir=rdir6, artifact_loader=al,
                                                    server_record_loader=sl)
            try:
                out, hits = e2e(resolver)
                ok = (out == "hit" and hits == 1) if want_hit else (out == "miss" and hits == 0)
                check(desc, ok, f"outcome={out} pooled.hits={hits}")
            except Exception as exc:
                check(desc, False, f"RAISED {exc}")

    e2e_receipts("RG-10a END-TO-END valid evidence: score() credits exactly one hit; molchan.pooled.hits == 1",
                 want_hit=True)
    e2e_receipts("RG-10b END-TO-END hash-invalid receipt: miss; molchan.pooled.hits == 0",
                 want_hit=False, mutate=lambda rc: rc["artifact_hashes"].__setitem__(REL_ENS, "0" * 64))
    e2e_receipts("RG-10d END-TO-END wrong-day receipt in the slot: miss; no credit",
                 want_hit=False, mutate=lambda rc: rc.__setitem__("day", "2026-08-04"))
    with tempfile.TemporaryDirectory() as td7:            # RG-10c missing receipt entirely
        resolver = R4.receipt_standing_resolver(receipts_dir=os.path.join(td7, "empty"),
                                                artifact_loader=al, server_record_loader=sl)
        try:
            out, hits = e2e(resolver)
            check("RG-10c END-TO-END missing receipt: miss; molchan.pooled.hits == 0",
                  out == "miss" and hits == 0, f"outcome={out} pooled.hits={hits}")
        except Exception as exc:
            check("RG-10c END-TO-END missing receipt: miss; molchan.pooled.hits == 0", False, f"RAISED {exc}")
    with tempfile.TemporaryDirectory() as td8:            # RG-10e absent server record
        rdir8 = _mk(td8)
        _, sl_none = _loaders(record=None)
        resolver = R4.receipt_standing_resolver(receipts_dir=rdir8, artifact_loader=al,
                                                server_record_loader=sl_none)
        try:
            out, hits = e2e(resolver)
            check("RG-10e END-TO-END absent server record: miss; no credit",
                  out == "miss" and hits == 0, f"outcome={out} pooled.hits={hits}")
        except Exception as exc:
            check("RG-10e END-TO-END absent server record: miss; no credit", False, f"RAISED {exc}")
    with tempfile.TemporaryDirectory() as td9:            # RG-10f loaders unavailable
        rdir9 = _mk(td9)
        resolver = R4.receipt_standing_resolver(receipts_dir=rdir9)
        try:
            out, hits = e2e(resolver)
            check("RG-10f END-TO-END loaders unavailable: miss; no credit",
                  out == "miss" and hits == 0, f"outcome={out} pooled.hits={hits}")
        except Exception as exc:
            check("RG-10f END-TO-END loaders unavailable: miss; no credit", False, f"RAISED {exc}")

    # per-date binding: timing-eligible date without credit + credited date without timing != hit
    with tempfile.TemporaryDirectory() as td10:
        rdir10 = _mk(td10)                                 # receipt exists ONLY for DAY (2026-08-05)
        resolver = R4.receipt_standing_resolver(receipts_dir=rdir10, artifact_loader=al,
                                                server_record_loader=sl)
        early_event = R4.Event("evt_early", "2026-08-06", 32.8, 130.7, 5.8, region="r1",
                               origin_utc="2026-08-06T00:30:00Z")
        # DAY's verified stamp is 2026-08-07T11:08:33Z -> AFTER this event (timing-INeligible on the
        # credited date); 2026-08-04 is receipt-less (credit-INeligible) but its ceiling makes the event
        # timing-eligible. Any cross-date composition of (some date timing) + (some date credit) would hit.
        ep = R4.Episode("r1", "2026-08-04", DAY, 2, False, alarm_dates=("2026-08-04", DAY))
        try:
            sc = R4.score([ep], [early_event], [], today="2026-08-20", standing_resolver=resolver)
            out = sc["outcomes"][0]["outcome"] if sc["outcomes"] else "NO-OUTCOME"
            check("RG-10g per-date binding: timing and credit must hold on the SAME alarm date (no "
                  "cross-date composition)", out == "miss", f"outcome={out}")
        except Exception as exc:
            check("RG-10g per-date binding: timing and credit must hold on the SAME alarm date (no "
                  "cross-date composition)", False, f"RAISED {exc}")

    # default + explicit-resolver semantics
    try:
        ep = R4.Episode("r1", DAY, DAY, 1, False, alarm_dates=(DAY,))
        sc = R4.score([ep], [EVT], [], today="2026-08-20")           # NO resolver passed
        out = sc["outcomes"][0]["outcome"] if sc["outcomes"] else "NO-OUTCOME"
        check("RG-11 DEFAULT IS FAIL-CLOSED: score() without a resolver credits nothing (no implicit allow)",
              out == "miss", f"outcome={out}")
    except Exception as exc:
        check("RG-11 DEFAULT IS FAIL-CLOSED: score() without a resolver credits nothing (no implicit allow)",
              False, f"RAISED {exc}")
    try:
        explicit = lambda d0: (R4._utc(d0 + "T23:59:59+00:00"), True)   # noqa: E731 -- the KAT-battery seam
        ep = R4.Episode("r1", DAY, DAY, 1, False, alarm_dates=(DAY,))
        sc = R4.score([ep], [EVT], [], today="2026-08-20", standing_resolver=explicit)
        out = sc["outcomes"][0]["outcome"] if sc["outcomes"] else "NO-OUTCOME"
        check("RG-12 explicit deterministic resolver (the 23-KAT injection seam) credits normally",
              out == "hit", f"outcome={out}")
    except Exception as exc:
        check("RG-12 explicit deterministic resolver (the 23-KAT injection seam) credits normally",
              False, f"RAISED {exc}")


main()
print()
if FAILS:
    print(f"R4 RECEIPT-GATE REV-3 RED-KAT FAILURES: {FAILS}")
    sys.exit(1)
print("ALL R4 RECEIPT-GATE REV-3 RED-KATs PASS (fail-closed scoring-path standing enforced end-to-end)")
