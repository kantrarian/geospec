#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RED-first KATs -- fault2graph WINDOW-2 bar (grassmann).

Cross-authored (V-D governance, the Phase-B pattern): grassmann authors,
cayley implements the window-2 surfaces BAR-UNEDITED, codex verifies.
Disagreements with any pin go through inbox R1.2, never through this file.

FROZEN AUTHORITIES (codex DESIGN CLOSE 2026-08-22T0145Z):
  design_target  = 12161f6cf2000b2d4aa86fe659f834839ac0d27b
  manifest commit= 5fba5446cd96722fb86021df5065a46b8a1a78f5
  manifest blob  = docs/f2g_window2_freeze/byte_pin_manifest.json
                   (f2g-window2-design-manifest-v2.1, 27 pins)
  design-pin verifier = monitoring/src/f2g_design_pin_verifier_cayley.py
                   @ b755ce1, blob sha a010044f... (KAT 19/19; typed vocab
                   published for this bar's W-PIN classes)
  freeze surfaces read verbatim for the specs below: selection_constants.md,
  cascadia_carrier_capsule.md, annex_b2b.md, annex_b1b.md, annex_mf4.md,
  mag1_instantiation.md (+ mag capsules and band-B SOS json).

STATUS: GREEN-NOW classes = W-PIN (verifier wrap + independent pin
recompute), W-CAS-a (cascadia receipt/envelope conformance), W-SEL-a
(in-bar reference selection oracle self-check). ENGINE-GATED classes are
typed red with EXACT fixture specifications pinned below -- cayley's
window-2 implementations do not exist yet; each seam group turns its
classes live in the established red-first rhythm.

PINNED SEAMS (module names R1.2-open until cayley's first implementation
commit; behavior pins are FROZEN by the annex texts):
  w2_selection.select(carrier_key, presence_by_station, day_sets, cap)
      -> {"selected": [ids], "churn": float, "typing": None|"BELOW_FLOOR"}
      or typed INSUFFICIENT_POOL raise. Constants must ALSO exist as module
      literals equal to the frozen table (0.85, 0.80, caps 16/20/14/16,
      min 8).
  w2_cascadia.registry_for_day(utc_day) -> per-day epoch/location-resolved
      NET.STA rows (epoch-active-first THEN blank->00->lexicographic).
  w2_b2b_family(panel, ...), w2_b1b_family(panel, ...) -- calendar-frame
      engines per annex; same result-shape conventions as the Phase-B bar.
  w2_mf4 accrual/scoring seams; w2_mag1 seams; w2_barrier state machine.

ENGINE-GATED FIXTURE SPECS (frozen; implemented in bar REVs as seams land):
  W-SEL-b : engine select == in-bar oracle on: nominal; INSUFFICIENT_POOL
            at pool=7; BELOW_FLOOR disclosure (floor unreachable at min);
            exact presence-tie lexicographic ordering; drop-worst path.
  W-CAS-b : TOUT epoch transition (blank ends 2026-07-16 exactly as 00
            opens -- naive blank-first must NOT select the dead epoch);
            RER three adjacent epochs; simultaneously-active blank/00
            (blank wins); envelope-vs-body recomputation equality.
  W-B2B   : annex KAT list verbatim -- variable-support planted runs
            surviving registered churn; adversarial alternating dropout
            refuses/terminates typed (never mis-counts); intersection
            floor boundary (floor-1 refuses INTERSECTION_BELOW_FLOOR,
            floor passes); label-permutation invariance (relabeled
            identical partition is NOT a switch); segment minimum
            (PARTITION_DEGENERATE_SIDE) boundary.
  W-B1B   : endpoint-order invariance; the exact (z=8, q_A=2, q_B=3) ->
            8/3 fixture through observed/null/LOCO/injected; ONE
            under-support endpoint -> whole-carrier ZERO_SCALE_REFUSAL on
            ALL four paths (never edge deletion/shrinkage); winsorization
            c=8 four-leg identity (skipping any one leg must change T and
            refuse); single-station gain-step x{3,10} artifact must NOT
            certify (specificity <= 0.05 alongside CP-LB >= 0.80);
            HEALTH_ADMISSION_VIOLATION on evaluation-window influence.
  W-MF4   : label-maturity lock (appending a post-snapshot event inside
            the 7-day tail leaves training digest + coefficients
            byte-identical); CALIBRATION_LABEL_NOT_MATURE typing;
            REGION_UNSCORABLE_ZERO_CLASS; >1/3-unscorable ->
            ENDPOINT_UNSCORABLE (no-drop); immutable signed prediction
            rows (post-issue mutation refuses); persistence baseline
            presence; 14d synchronized blocks B=999 constants.
  W-MAG   : apply-never-refit incl. the M3 reference regression; VIC XYZS
            frame (S excluded from the horizontal vector) + the four
            capsule frame-refusal cases; band-B SOS byte-pinned
            coefficient equality + FILTER_SUPPORT_INSUFFICIENT;
            kahramanmaras MAG-UNTESTABLE typing; internal Holm structure
            (3 primaries after the VIC+NEW cascadia admission).
  W-BARRIER: the v0.3 sec-2 state machine's nine typed refusals;
            pre-barrier verdict row refusal; post-first-fire failure
            typed to window 3; embargo integrity; non-circular Holm S
            computed ONCE (mandated 4->3 relaxation attempt REFUSES);
            codex's five cross-lane KATs.

No window-2 measurement, prediction, residual, or alignment value is
opened by this bar. Lambda_geo INCONCLUSIVE. grassmann 2026-08-22.
"""
import hashlib
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))

DESIGN_TARGET = "12161f6cf2000b2d4aa86fe659f834839ac0d27b"
MANIFEST_COMMIT = "5fba5446cd96722fb86021df5065a46b8a1a78f5"
MANIFEST_PATH = "docs/f2g_window2_freeze/byte_pin_manifest.json"
RECEIPT_BODY_SHA = ("d4256792bf85edf855a4dbaf7841982824a020cd5e075c103d8322"
                    "48c513a847")
PIN_VERIFIER_BLOB = ("a010044f4365983e439605c7605d608963a81a1ee389439c90a78"
                     "64ca9a21ac6")

SEL = {"presence_floor": 0.85, "churn_floor": 0.80,
       "caps": {"istanbul_marmara": 16, "socal_coachella": 20,
                "turkey_kahramanmaras": 14, "cascadia": 16},
       "minimum": 8}

FAILS = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"    [{tag}] W2-{name}" + (f" - {detail}" if detail and not ok
                                      else ""))
    if not ok:
        FAILS.append(name)


def _blob(ref):
    return subprocess.run(["git", "-C", _REPO, "cat-file", "blob", ref],
                          capture_output=True).stdout


# ---- W-PIN: design-pin integrity -------------------------------------------
def w_pin():
    # a: cayley's verifier executed as a subprocess against the frozen
    #    manifest commit -- must report PASS 27/27 (the verifier reads git
    #    objects only, so this is host- and checkout-independent)
    try:
        vsrc = _blob(f"b755ce1:monitoring/src/"
                     f"f2g_design_pin_verifier_cayley.py")
        ok_id = hashlib.sha256(vsrc).hexdigest() == PIN_VERIFIER_BLOB
        r = subprocess.run(
            [sys.executable,
             os.path.join(_HERE, "f2g_design_pin_verifier_cayley.py"),
             _REPO, MANIFEST_COMMIT],
            capture_output=True, text=True, timeout=300)
        out = (r.stdout + r.stderr)
        ok_run = r.returncode == 0 and ('"verdict": "PASS"' in out
                                        or "PASS" in out)
        check("PIN-a verifier PASS at the frozen manifest commit",
              ok_id and ok_run,
              f"blob_id={ok_id} rc={r.returncode} tail={out[-120:]}")
    except Exception as exc:
        check("PIN-a verifier PASS at the frozen manifest commit", False,
              f"{type(exc).__name__}: {exc}")

    # b: INDEPENDENT spot recompute (no self-consistency with the verifier):
    #    reopen the manifest from its commit, recompute every pin's blob
    #    sha256 directly from git objects, and verify the design target +
    #    pin count from the bytes
    try:
        man = json.loads(_blob(f"{MANIFEST_COMMIT}:{MANIFEST_PATH}")
                         .decode("utf-8"))
        pins = man.get("pins")
        pin_items = list(pins.items()) if isinstance(pins, dict) \
            else [(p.get("path"), p) for p in pins]
        ok_b = (man.get("design_target_commit") == DESIGN_TARGET
                and man.get("pin_count") == len(pin_items)
                and man.get("pin_count") == 27)
        bad = []
        for name, pin in pin_items:
            raw = _blob(f"{pin['commit']}:{pin['path']}")
            if hashlib.sha256(raw).hexdigest() != pin["blob_sha256"]:
                bad.append(name)
                ok_b = False
        check("PIN-b independent 27-pin blob recompute", ok_b,
              f"bad={bad[:3]} count={man.get('pin_count')}")
    except Exception as exc:
        check("PIN-b independent 27-pin blob recompute", False,
              f"{type(exc).__name__}: {exc}")


# ---- W-CAS-a: cascadia receipt/envelope conformance ------------------------
def w_cas_a():
    try:
        body = _blob(f"{MANIFEST_COMMIT}:docs/f2g_window2_freeze/receipts/"
                     f"cascadia_UW_CC_CN_HHZ.txt")
        ok_sha = hashlib.sha256(body).hexdigest() == RECEIPT_BODY_SHA
        rows = [l for l in body.decode("utf-8", "replace").splitlines()
                if l and not l.startswith("#")]
        idents = {}
        for l in rows:
            p = l.split("|")
            if len(p) > 2:
                idents.setdefault(p[0], set()).add(p[1])
        uniq = {f"{n}.{s}" for n, ss in idents.items() for s in ss}
        by_net = {n: len(ss) for n, ss in idents.items()}
        ok_counts = (len(rows) == 203 and len(uniq) == 198
                     and by_net.get("UW") == 118 and by_net.get("CC") == 43
                     and by_net.get("CN") == 37)
        env = json.loads(_blob(
            f"{MANIFEST_COMMIT}:docs/f2g_window2_freeze/receipts/"
            f"cascadia_UW_CC_CN_HHZ.envelope.json").decode("utf-8"))
        env_leaves = json.dumps(env)
        ok_env = RECEIPT_BODY_SHA in env_leaves \
            and "service.earthscope.org" in env_leaves
        # the TOUT epoch-transition facts the W-CAS-b fixtures rely on
        tout = [l for l in rows if "|TOUT|" in l]
        ok_tout = len(tout) >= 2 and any("|00|" in l for l in tout) \
            and any(l.split("|")[2] == "" for l in tout)
        check("CAS-a receipt/envelope conformance (203/198, 118/43/37, "
              "TOUT epochs present)",
              ok_sha and ok_counts and ok_env and ok_tout,
              f"sha={ok_sha} counts={ok_counts} ({len(rows)}/{len(uniq)}/"
              f"{by_net}) env={ok_env} tout={ok_tout}")
    except Exception as exc:
        check("CAS-a receipt/envelope conformance", False,
              f"{type(exc).__name__}: {exc}")


# ---- W-SEL-a: in-bar reference selection oracle (self-check) ---------------
def _ref_select(presence, day_sets, cap, minimum=8,
                pfloor=0.85, cfloor=0.80):
    """The frozen algorithm, implemented independently as this bar's oracle:
    greedy by (presence DESC, station_id ASC) into the cap over stations
    meeting the presence floor; then drop-worst by presence (ties: drop the
    lexicographically LAST, preserving ASC preference) until carrier-set
    churn >= floor or the minimum is reached; below minimum -> raise
    INSUFFICIENT_POOL; floor unreachable at minimum -> BELOW_FLOOR typed."""
    eligible = sorted((s for s, p in presence.items() if p >= pfloor),
                      key=lambda s: (-presence[s], s))
    if len(eligible) < minimum:            # pool test precedes the cap
        raise ValueError("INSUFFICIENT_POOL")
    pool = eligible[:cap]

    def churn(sel):
        sel_set = set(sel)
        sims = []
        for a, b in zip(day_sets, day_sets[1:]):
            aa, bb = set(a) & sel_set, set(b) & sel_set
            u = aa | bb
            sims.append(1.0 if not u else len(aa & bb) / len(u))
        return sum(sims) / len(sims) if sims else 1.0

    sel = list(pool)
    while churn(sel) < cfloor and len(sel) > minimum:
        # drop the LEAST-preferred member: lowest presence, then the
        # lexicographically LAST id (cayley R1.2-a repair adopted verbatim
        # -- the prior negative-ord list key inverted prefix-related ids)
        worst = max(sel, key=lambda s: (-presence[s], s))
        sel.remove(worst)
    c = churn(sel)
    return {"selected": sorted(sel), "churn": c,
            "typing": None if c >= cfloor else "BELOW_FLOOR"}


def w_sel_a():
    try:
        # hand-computed fixture: 12 stations, 3 below the presence floor,
        # 9 eligible in [0.86, 0.98] (cayley R1.2-b: presence is a fraction,
        # never > 1.0); cap 8 -> greedy takes top-8 eligible
        pres = {"S00": 0.70, "S01": 0.75, "S02": 0.80}
        pres.update({f"S{i + 3:02d}": 0.86 + 0.015 * i for i in range(9)})
        days = [sorted(pres)] * 5                                  # stable
        r = _ref_select(pres, days, cap=8)
        elig = {s: p for s, p in pres.items() if p >= 0.85}
        expect = sorted(sorted(elig, key=lambda s: (-elig[s], s))[:8])
        ok1 = r["selected"] == expect and r["typing"] is None \
            and abs(r["churn"] - 1.0) < 1e-12
        # presence ties broken lexicographically ASC
        pres2 = {s: 0.90 for s in ("B", "A", "C", "E", "D", "F", "G", "H",
                                   "I")}
        r2 = _ref_select(pres2, [list(pres2)] * 3, cap=8)
        ok2 = r2["selected"] == sorted(list(pres2))[:8]
        # INSUFFICIENT_POOL at pool==7
        pres3 = {f"T{i}": 0.90 for i in range(7)}
        try:
            _ref_select(pres3, [list(pres3)] * 3, cap=16)
            ok3 = False
        except ValueError as exc:
            ok3 = "INSUFFICIENT_POOL" in str(exc)
        check("SEL-a reference oracle self-check (greedy order, tie-break, "
              "INSUFFICIENT_POOL)", ok1 and ok2 and ok3,
              f"greedy={ok1} ties={ok2} pool={ok3}")
    except Exception as exc:
        check("SEL-a reference oracle self-check", False,
              f"{type(exc).__name__}: {exc}")


# ---- W-SEL-b: engine == oracle (wired vs w2_selection @ a9bb8ae) -----------
def w_sel_b():
    try:
        import w2_selection as WS
        ok_const = (WS.PRESENCE_FLOOR == 0.85 and WS.CHURN_FLOOR == 0.80
                    and WS.CAPS == SEL["caps"] and WS.MINIMUM == 8)
        # nominal + ties: engine must equal the oracle exactly
        p1 = {"S00": 0.70, "S01": 0.75, "S02": 0.80}
        p1.update({f"S{i + 3:02d}": 0.86 + 0.015 * i for i in range(9)})
        d1 = [sorted(p1)] * 5
        e1 = WS.select_fixture(p1, d1, cap=8)      # REV-3 migration: fixture
        o1 = _ref_select(p1, d1, cap=8)            # seam per cap policy P1-3
        ok_nom = e1 == o1
        # INSUFFICIENT_POOL at pool == 7
        p3 = {f"T{i}": 0.90 for i in range(7)}
        try:
            WS.select_fixture(p3, [sorted(p3)] * 3, cap=16)
            ok_pool = False
        except ValueError as exc:
            ok_pool = "INSUFFICIENT_POOL" in str(exc)
        # drop-worst + prefix corner (cayley's fixture, expected values
        # from the ratified rule): 7 stable @0.99 + {"A","AB"} tied @0.90
        # flapping on alternate days -> one drop -> "AB" out, "A" survives
        stable = [f"Z{i}" for i in range(7)]
        p4 = {s: 0.99 for s in stable}
        p4.update({"A": 0.90, "AB": 0.90})
        d4 = [sorted(stable + ["A", "AB"]) if i % 2 == 0 else sorted(stable)
              for i in range(6)]
        e4 = WS.select_fixture(p4, d4, cap=9)
        o4 = _ref_select(p4, d4, cap=9)
        ok_corner = e4 == o4 and "AB" not in e4["selected"] \
            and "A" in e4["selected"]
        # production-path locks (codex P1-2/P1-3, cayley REV-2 KAT list):
        from datetime import date as _date, timedelta as _td
        cutoff = "2026-07-10"
        days90 = [( _date(2026, 7, 10) - _td(days=i)).isoformat()
                  for i in range(89, -1, -1)]
        sts = [f"P{i:02d}" for i in range(10)]
        recs = {d: list(sts) for d in days90}
        ep = WS.select("cascadia", recs, cutoff)          # cap=None resolves 16
        ok_prod = sorted(ep["selected"]) == sts and ep["typing"] is None
        try:
            WS.select("cascadia", {d: recs[d] for d in days90[1:]}, cutoff)
            ok_frame = False
        except Exception as exc:
            ok_frame = "LOOKBACK_FRAME_INVALID" in str(exc)
        try:
            WS.select("cascadia", recs, cutoff,
                      presence_declared={s: 89 for s in sts})
            ok_decl = False
        except Exception as exc:
            ok_decl = "PRESENCE_LOOKBACK_MISMATCH" in str(exc)
        try:
            WS.select("cascadia", recs, cutoff, cap=15)
            ok_cap = False
        except Exception as exc:
            ok_cap = "CAP_OVERRIDE_REFUSED" in str(exc)
        ok_cap16 = WS.select("cascadia", recs, cutoff, cap=16) == ep
        # exact-arithmetic churn lock (cayley REV-2 KAT #1, expected values
        # frozen): 8 stable + 3 flappy alternating -> J exactly 8/11 -> one
        # drop (lex-LAST flappy) -> J exactly 4/5 -> exact stop, 10 survive
        st8 = [f"K{i}" for i in range(8)]
        fl3 = ["ZA", "ZB", "ZC"]        # sort AFTER the stable set so the
        p5 = {s: 0.99 for s in st8 + fl3}   # lex-LAST tie drop hits a flapper
        d5 = [sorted(st8 + fl3) if i % 2 == 0 else sorted(st8)
              for i in range(6)]
        e5 = WS.select_fixture(p5, d5, cap=11)
        o5 = _ref_select(p5, d5, cap=11)
        ok_exact = e5 == o5 and sorted(e5["selected"]) == sorted(
            st8 + ["ZA", "ZB"]) and e5["typing"] is None
        check("SEL-b engine == oracle (constants, fixture path, production "
              "path, frame/decl/cap refusals, exact-churn lock)",
              ok_const and ok_nom and ok_pool and ok_corner and ok_prod
              and ok_frame and ok_decl and ok_cap and ok_cap16 and ok_exact,
              f"const={ok_const} nom={ok_nom} pool={ok_pool} "
              f"corner={ok_corner} prod={ok_prod} frame={ok_frame} "
              f"decl={ok_decl} cap={ok_cap}/{ok_cap16} exact={ok_exact}")
    except ImportError:
        check("SEL-b engine == oracle", False, "W2_ENGINE_ABSENT")
    except Exception as exc:
        check("SEL-b engine == oracle", False, f"{type(exc).__name__}: {exc}")


# ---- W-CAS-b: epoch-first precedence vs an independent in-bar resolver -----
def _parse_epochs(body_text):
    eps = []
    for l in body_text.splitlines():
        if not l or l.startswith("#"):
            continue
        p = l.split("|")
        if len(p) < 2:
            continue
        eps.append({"net": p[0], "sta": p[1], "loc": p[2],
                    "start": p[-2], "end": p[-1]})
    return eps


def _resolve_day(eps, day):
    """Independent oracle: ACTIVE at day-start 00:00:00Z (half-open
    [start, end)) first, THEN blank -> 00 -> lexicographic."""
    t = day + "T00:00:00"
    out = {}
    for key in {(e["net"], e["sta"]) for e in eps}:
        act = [e for e in eps if (e["net"], e["sta"]) == key
               and e["start"][:19] <= t and (not e["end"].strip()
                                             or e["end"][:19] > t)]
        if not act:
            continue
        act.sort(key=lambda e: (e["loc"] != "", e["loc"]))
        out[f"{key[0]}.{key[1]}"] = act[0]["loc"]
    return out


def w_cas_b():
    try:
        import w2_cascadia as WC
        body = _blob(f"{MANIFEST_COMMIT}:docs/f2g_window2_freeze/receipts/"
                     f"cascadia_UW_CC_CN_HHZ.txt").decode("utf-8", "replace")
        eps = _parse_epochs(body)
        ok_days, bad = True, []
        for day in ("2026-07-11", "2026-07-15", "2026-07-16", "2026-07-30",
                    "2026-07-31", "2026-11-30"):
            oracle = _resolve_day(eps, day)
            rows = WC.registry_for_day(day, repo=_REPO)
            got = {r["id"]: r["location"] for r in rows}
            if got != oracle:
                ok_days = False
                diff = {k: (oracle.get(k), got.get(k))
                        for k in set(oracle) | set(got)
                        if oracle.get(k) != got.get(k)}
                bad.append((day, dict(list(diff.items())[:2])))
        # TOUT ruled facts asserted explicitly on both paths
        o15 = _resolve_day(eps, "2026-07-15").get("UW.TOUT")
        o16 = _resolve_day(eps, "2026-07-16").get("UW.TOUT")
        ok_tout = o15 == "" and o16 == "00"
        # synthetic: simultaneous blank/00 -> blank; same-loc overlap refuses
        from datetime import datetime as _dt
        synth = [{"network": "XX", "station": "AAA", "location": "",
                  "channel": "HHZ", "latitude": 0.0, "longitude": 0.0,
                  "epoch_start": _dt(2026, 7, 1), "epoch_end":
                  _dt(2026, 12, 31)},
                 {"network": "XX", "station": "AAA", "location": "00",
                  "channel": "HHZ", "latitude": 0.0, "longitude": 0.0,
                  "epoch_start": _dt(2026, 7, 1), "epoch_end":
                  _dt(2026, 12, 31)}]
        r_syn = WC.registry_for_day("2026-08-01", epochs=synth)
        got_syn = {r["id"]: r["location"] for r in r_syn}
        ok_syn = got_syn.get("XX.AAA") == ""
        dup = [dict(synth[0]), dict(synth[0])]
        try:
            WC.registry_for_day("2026-08-01", epochs=dup)
            ok_dup = False
        except Exception as exc:
            ok_dup = "EPOCH_OVERLAP_SAME_LOCATION" in str(exc)
        try:
            WC.registry_for_day("08/01/2026", epochs=synth)
            ok_fmt = False
        except Exception as exc:
            ok_fmt = "BAD_DAY_FORMAT" in str(exc)
        s = WC.receipt_summary(repo=_REPO)
        ok_env = (s.get("rows") == 203 and s.get("unique_net_sta") == 198
                  and s.get("by_network") == {"UW": 118, "CC": 43, "CN": 37})
        check("CAS-b epoch-first precedence (6-day full-registry oracle "
              "sweep, TOUT, blank-wins, overlap+format refusals, envelope)",
              ok_days and ok_tout and ok_syn and ok_dup and ok_fmt and ok_env,
              f"days={ok_days} {bad[:2]} tout={ok_tout} syn={ok_syn} "
              f"dup={ok_dup} fmt={ok_fmt} env={ok_env}")
    except ImportError:
        check("CAS-b epoch-first precedence", False, "W2_ENGINE_ABSENT")
    except Exception as exc:
        check("CAS-b epoch-first precedence", False,
              f"{type(exc).__name__}: {exc}")


# ---- engine-gated classes: typed red until cayley's surfaces land ----------
_GATED = (
    "B2B annex KATs (planted churn survival, adversarial dropout, "
    "floor boundary, label invariance, segment minimum) [engine landed "
    "7b4d097; fixtures wire in the next REV with the three ratified pins]",
    "B1B annex KATs (endpoint invariance, 8/(max(2,3)) fixture x4 paths, "
    "ZERO_SCALE_REFUSAL never-shrink, winsor 4-leg identity, gain-step "
    "specificity, health admission)",
    "MF4 annex KATs (label maturity byte-lock, zero-class/no-drop, "
    "immutable rows, persistence baseline, block constants)",
    "MAG annex KATs (apply-never-refit, VIC XYZS/S-exclusion + 4 frame "
    "refusals, SOS byte equality, MAG-UNTESTABLE, 3-primary Holm)",
    "BARRIER state machine (9 typed refusals, pre-barrier verdict row, "
    "window-3 typing, embargo integrity, Holm-S once + 4->3 refusal, "
    "5 cross-lane KATs)",
)


def main():
    w_pin()
    w_cas_a()
    w_sel_a()
    w_sel_b()
    w_cas_b()
    for nm in _GATED:
        check(nm, False, "W2_ENGINE_ABSENT (expected red; fixture spec "
                         "frozen in the bar header)")


main()
print()
if FAILS:
    print(f"WINDOW-2 RED-KAT FAILURES ({len(FAILS)}): "
          f"{[f.split(' ')[0] for f in FAILS]}")
    sys.exit(1)
print("ALL WINDOW-2 RED-KATs PASS")
