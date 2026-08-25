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


# ---- W-B2B: annex KATs wired vs w2_b2b (pins ratified in REV 2) ------------
def _b2b_day(ca, cb, strong=5.0, weak=0.1):
    """In-bar edge builder (independent of the engine's selftest helper):
    strong intra-cluster / weak inter-cluster -> deterministic Fiedler
    bipartition {ca, cb}."""
    ew = {}
    nodes = list(ca) + list(cb)
    for i, x in enumerate(nodes):
        for y in nodes[i + 1:]:
            ew["|".join(sorted((x, y)))] = strong if (x in ca) == (y in ca) \
                else weak
    return ew


def _b2b_carrier(days, day_edges, measured, registry):
    r = {}
    for d, ew, m in zip(days, day_edges, measured):
        ms = set(m)
        for e, w in ew.items():
            a, b = e.split("|")
            if a in ms and b in ms:      # finite edges only on measured
                r.setdefault(e, {})[d] = w   # endpoints (ratified pin 3)
    return {"registry": list(registry), "registered_days": list(days),
            "measured": {d: sorted(m) for d, m in zip(days, measured)},
            "r": r}


def w_b2b():
    try:
        import w2_b2b as WB
        days = [f"2026-09-{i:02d}" for i in range(1, 7)]
        A = [f"A{i}" for i in range(5)]
        B = [f"B{i}" for i in range(5)]
        reg = A + B                               # floor ceil(20/3) = 7

        def fam(car, nd=199, cal=None):
            return WB.w2_b2b_family({"calendar": cal or days,
                                     "carriers": {"x": car}},
                                    doc_sha256="cd" * 32, n_draws=nd)

        ok_floorfn = WB.overlap_floor(10) == 7 and WB.overlap_floor(9) == 6
        # (1) variable-support planted: alternate days drop 3 stations ->
        # I_d exactly 7 == floor -> comparisons proceed, ONE run survives
        m_alt = [reg if i % 2 == 0 else reg[:7] for i in range(6)]
        r1 = fam(_b2b_carrier(days, [_b2b_day(A, B)] * 6, m_alt, reg))
        ok1 = r1["runs_by_carrier"]["x"] == 1 and not r1.get("day_refusals")
        # (2) adversarial alternating dropout: overlaps below floor on
        # every pair -> runs == candidates, every pair typed (ratified pin)
        m_adv = [reg[:7] if i % 2 == 0 else reg[3:] for i in range(6)]
        r2 = fam(_b2b_carrier(days, [_b2b_day(A, B)] * 6, m_adv, reg))
        n_typed2 = sum(1 for x in r2.get("day_refusals", [])
                       if "INTERSECTION_BELOW_FLOOR" in str(x))
        ok2 = r2["runs_by_carrier"]["x"] == 6 and n_typed2 == 5
        # (3) floor boundary at registry 9 (floor 6): shared 6 passes
        # (1 run over two days); shared 5 refuses typed (2 runs)
        reg9 = A + B[:4]                          # floor 6
        d2 = days[:2]
        shared6 = A[:3] + B[:3]                   # 3/3, both sides >= 2
        r3a = fam(_b2b_carrier(d2, [_b2b_day(A, B[:4])] * 2,
                               [reg9, shared6], reg9), cal=d2)
        shared5 = A[:3] + B[:2]                   # shared 5 < floor 6
        r3b = fam(_b2b_carrier(d2, [_b2b_day(A, B[:4])] * 2,
                               [reg9, shared5], reg9), cal=d2)
        ok3 = (r3a["runs_by_carrier"]["x"] == 1
               and not r3a.get("day_refusals")
               and r3b["runs_by_carrier"]["x"] == 2
               and any("INTERSECTION_BELOW_FLOOR" in str(x)
                       for x in r3b.get("day_refusals", [])))
        # (4) label-permutation invariance: swapped cluster arguments ->
        # identical unordered bipartition -> ONE run
        r4 = fam(_b2b_carrier(days, [_b2b_day(A, B)] * 3
                              + [_b2b_day(B, A)] * 3, [reg] * 6, reg))
        ok4 = r4["runs_by_carrier"]["x"] == 1
        # (5) segment minimum: a 9/1 partition on the induced set ->
        # PARTITION_DEGENERATE_SIDE typed
        lone = ["L0"]
        r5 = fam(_b2b_carrier(d2, [_b2b_day(reg[:9], lone)] * 2,
                              [reg[:9] + lone] * 2, reg[:9] + lone))
        ok5 = any("PARTITION_DEGENERATE_SIDE" in str(x)
                  for x in r5.get("day_refusals", []))
        check("B2B annex KATs (planted survival, adversarial dropout, "
              "floor boundary, label invariance, degenerate side)",
              ok_floorfn and ok1 and ok2 and ok3 and ok4 and ok5,
              f"floorfn={ok_floorfn} planted={ok1} adv={ok2} "
              f"(r={r2['runs_by_carrier']['x']},typed={n_typed2}) "
              f"boundary={ok3} label={ok4} degen={ok5}")
    except ImportError:
        check("B2B annex KATs", False, "W2_ENGINE_ABSENT")
    except Exception as exc:
        check("B2B annex KATs", False, f"{type(exc).__name__}: {exc}")


# ---- W-BARRIER: the sec-2 state machine + sec-5 selector -------------------
def w_barrier():
    try:
        import w2_barrier as WBAR

        def expect(fn, code):
            try:
                fn()
                return False
            except Exception as exc:
                return code in str(exc)

        def bindings(lease="LEASE-1"):
            return {k: f"sha-{k}" for k in WBAR.REQUIRED_BINDINGS
                    if k not in ("remote_lease", "lane_uuids",
                                 "code_manifest")} | {
                "remote_lease": lease,
                "lane_uuids": ["seismic", "mf4", "mag1"],
                "code_manifest": {
                    "execution_manifest_commit": "bar-mc",
                    "execution_manifest_blob_sha256": "bar-mb"}}

        def fresh():
            led = WBAR.BarrierLedger()
            b = bindings()
            led.prestart(b, "2026-08-25", WBAR._admission(b))
            return led

        # lifecycle happy path with independent boundary expectations
        L = fresh()
        ok_dates = (L.evaluation_start.isoformat() == "2026-08-26"
                    and L.evaluation_end.isoformat() == "2027-01-04"
                    and L.maturity_tail_end.isoformat() == "2027-01-11")
        L.accrue_prediction("LEASE-1", "cascadia", "2026-08-26", "d1",
                            "2026-08-26")
        L.producer_receipt("LEASE-1", "r1")
        L.close_support_barrier("LEASE-1", "2027-01-12", "non_analyst")
        for lane in ("seismic", "mf4", "mag1"):
            L.record_owner_seal("LEASE-1", lane, f"seal-{lane}")
            L.final_fire("LEASE-1", lane, f"res-{lane}")
            L.record_verifier_pass("LEASE-1", lane, f"ver-{lane}")
        L.release("LEASE-1")
        ok_happy = L.state == "RELEASED" and L.verify_chain() \
            and L.read_result("LEASE-1", "mf4")["result_digest"] == "res-mf4"

        # the nine sec-2 refusals + the additional typed classes
        R = []
        R.append(expect(lambda: fresh().accrue_prediction(
            "LEASE-1", "r", "2026-08-26", "d", "2026-09-05"),
            "LATE_OR_REVISED_PREDICTION"))               # late emit
        led2 = fresh()
        led2.accrue_prediction("LEASE-1", "r", "2026-08-26", "d",
                               "2026-08-26")
        R.append(expect(lambda: led2.accrue_prediction(
            "LEASE-1", "r", "2026-08-26", "d2", "2026-08-27"),
            "LATE_OR_REVISED_PREDICTION"))               # duplicate
        R.append(expect(lambda: fresh().read_labels("non_analyst"),
                        "EARLY_LABEL_ACCESS"))
        R.append(expect(lambda: fresh().inspect_support(True),
                        "SEMANTIC_SUPPORT_INSPECTION"))
        R.append(expect(lambda: fresh().close_support_barrier(
            "LEASE-1", "2027-01-11", "non_analyst"),
            "MISSING_MATURITY_TAIL"))                    # == tail refuses
        led3 = fresh()
        led3.close_support_barrier("LEASE-1", "2027-01-12", "non_analyst")
        led3.record_owner_seal("LEASE-1", "seismic", "s")
        led3.final_fire("LEASE-1", "seismic", "res")
        led3.record_verifier_pass("LEASE-1", "seismic", "v")
        R.append(expect(lambda: led3.release("LEASE-1"),
                        "CROSS_LANE_RELEASE_BEFORE_TERMINALS"))
        b_noauth = {k: v for k, v in bindings().items()
                    if k != "owner_authorization"}
        R.append(expect(lambda: WBAR.BarrierLedger().prestart(
            b_noauth, "2026-08-25", WBAR._admission(b_noauth)),
            "MISSING_LANE_AUTHORIZATION"))
        led4 = fresh()
        led4.close_support_barrier("LEASE-1", "2027-01-12", "non_analyst")
        R.append(expect(lambda: led4.final_fire("LEASE-1", "mf4", "r"),
                        "VALUE_FIRE_SEAL_MISSING"))      # unsealed fire
        R.append(expect(lambda: fresh().add_lane("LEASE-1", "extra"),
                        "LATE_LANE_ADDITION"))
        b_reuse = bindings()
        R.append(expect(lambda: WBAR.BarrierLedger(
            used_leases=("LEASE-1",)).prestart(
                b_reuse, "2026-08-25", WBAR._admission(b_reuse)),
            "REUSED_GLOBAL_LEASE"))
        # 1815Z item-1 admission doctors (the composed PRESTART gate):
        # bare bindings, non-capsule, doctored digest, stale verifier
        # receipt, OPEN-manifest class, post-binding manifest drift
        b_adm = bindings()
        R.append(expect(lambda: WBAR.BarrierLedger().prestart(
            b_adm, "2026-08-25"), "PRESTART_ADMISSION_REFUSED"))
        R.append(expect(lambda: WBAR.BarrierLedger().prestart(
            b_adm, "2026-08-25", "not-a-capsule"),
            "PRESTART_ADMISSION_REFUSED"))
        R.append(expect(lambda: WBAR.BarrierLedger().prestart(
            b_adm, "2026-08-25",
            WBAR._admission(b_adm, admission_digest="0" * 64)),
            "PRESTART_ADMISSION_REFUSED"))
        R.append(expect(lambda: WBAR.BarrierLedger().prestart(
            b_adm, "2026-08-25", WBAR._admission(
                b_adm, prestart_verifier={
                    "verdict": "PASS", "mode": "prestart",
                    "slots_open": 0,
                    "manifest_commit": "SOME-OTHER"})),
            "PRESTART_ADMISSION_REFUSED"))     # stale receipt
        R.append(expect(lambda: WBAR.BarrierLedger().prestart(
            b_adm, "2026-08-25", WBAR._admission(
                b_adm, prestart_verifier={
                    "verdict": "REFUSE", "mode": "prestart",
                    "slots_open": 2, "manifest_commit": "kat-mc"})),
            "PRESTART_ADMISSION_REFUSED"))     # OPEN-manifest class
        R.append(expect(lambda: WBAR.BarrierLedger().prestart(
            b_adm, "2026-08-25", WBAR._admission(
                b_adm, manifest_blob_sha256="drifted")),
            "PRESTART_ADMISSION_REFUSED"))     # post-binding drift
        R.append(expect(lambda: fresh().producer_receipt("WRONG", "r"),
                        "GLOBAL_LEASE_INCORRECT"))
        # rebind BEFORE any fire refuses WITHOUT the terminal
        led5 = fresh()
        ok_rebind_pre = expect(lambda: led5.rebind_source(
            "LEASE-1", "adapter"), "BINDING_IMMUTABLE_AFTER_PRESTART") \
            and led5.state == "ACCRUAL"
        # rebind AFTER first fire -> WINDOW3_TERMINAL
        R.append(expect(lambda: led3.rebind_source("LEASE-1", "engine"),
                        "POST_FIRST_FIRE_SOURCE_CHANGE"))
        ok_w3 = led3.state == "WINDOW3_TERMINAL"
        R.append(expect(lambda: fresh().record_verdict_row(
            "LEASE-1", "2026-08-20"), "PRE_BARRIER_VERDICT_ROW"))
        R.append(expect(lambda: fresh().read_result("LEASE-1", "mf4"),
                        "EMBARGO_VIOLATION"))
        # chain tamper-evidence
        led6 = fresh()
        led6.events[0]["payload"]["lanes"] = ["doctored"]
        ok_chain = expect(led6.verify_chain, "LEDGER_CHAIN_BROKEN")

        # sec-5 selector + Holm with a hand-derived expectation
        led7 = fresh()
        power = {h: {"cp_lcb": v, "graph": list(WBAR.GRAPH_MEMBERS)}
                 for h, v in (("B2A", 0.861), ("B2B", 0.83),
                              ("B1B", 0.42), ("B3A", 0.10))}
        S = led7.commit_selector(power)
        ok_sel = sorted(S) == ["B2A", "B2B"]
        ok_once = expect(lambda: led7.commit_selector(power),
                         "SELECTOR_ALREADY_COMMITTED")
        ok_relax = expect(lambda: led7.recertify_selector(
            {h: power[h] for h in ("B2A", "B2B", "B1B")}),
            "SELECTOR_RECERTIFICATION_REFUSED")
        part = {h: dict(power[h], graph=["B2A", "B2B", "B1B"])
                for h in power}
        ok_full = expect(lambda: WBAR.BarrierLedger().commit_selector(part),
                         "SELECTOR_GRAPH_NOT_FULL")
        # Holm hand-check: S={B2A,B2B}; p={B2A:.020,B2B:.030}; order B2A
        # first at alpha/2=.025 -> REJECT; then B2B at alpha/1=.05 ->
        # REJECT; B3A p=.0001 stays typed OUTSIDE
        hv = led7.holm_graph_lane({"B2A": 0.020, "B2B": 0.030,
                                   "B1B": 0.90, "B3A": 0.0001})
        ok_holm = (hv["verdicts"] == {"B2A": "REJECT", "B2B": "REJECT",
                                      "B1B": "CANNOT_DETERMINE_NO_POWER",
                                      "B3A": "CANNOT_DETERMINE_NO_POWER"})
        # and the step-down stop: p={B2A:.030,B2B:.030} -> B2A at .025
        # fails -> BOTH NO_REJECT
        hv2 = led7.holm_graph_lane({"B2A": 0.030, "B2B": 0.030,
                                    "B1B": 0.9, "B3A": 0.9})
        ok_holm2 = hv2["verdicts"]["B2A"] == "NO_REJECT" \
            and hv2["verdicts"]["B2B"] == "NO_REJECT"

        ok_all = (ok_dates and ok_happy and all(R) and ok_rebind_pre
                  and ok_w3 and ok_chain and ok_sel and ok_once
                  and ok_relax and ok_full and ok_holm and ok_holm2)
        check("BARRIER state machine + selector (lifecycle, 14 typed "
              "refusals, window-3 terminal, chain tamper, Holm-S-once + "
              "hand-derived Holm)", ok_all,
              f"dates={ok_dates} happy={ok_happy} refusals={R} "
              f"rebind_pre={ok_rebind_pre} w3={ok_w3} chain={ok_chain} "
              f"sel={ok_sel}/{ok_once}/{ok_relax}/{ok_full} "
              f"holm={ok_holm}/{ok_holm2}")
    except ImportError:
        check("BARRIER state machine + selector", False, "W2_ENGINE_ABSENT")
    except Exception as exc:
        check("BARRIER state machine + selector", False,
              f"{type(exc).__name__}: {exc}")


# ---- W-B1B: annex KATs wired vs w2_b1b -------------------------------------
def w_b1b():
    try:
        import w2_b1b as WB1
        import numpy as _np
        days40 = [f"2026-{7 + i // 28:02d}-{1 + i % 28:02d}"
                  for i in range(40)]
        GEO = dict(n_blocks=4, block_len=10, baseline_positions=20,
                   testable_min=10)

        def carrier(rng, sts, spike_edge=None, spike_pos=(),
                    spike_val=100.0, sparse=None, gain=None,
                    gain_from=None):
            r = {}
            for i, a in enumerate(sts):
                for b in sts[i + 1:]:
                    e = f"{a}|{b}"
                    ser = {}
                    for j, d in enumerate(days40):
                        v = float(rng.normal(0.0, 1.0))
                        if spike_edge == e and j in spike_pos:
                            v = spike_val
                        if gain and gain in (a, b) and j >= gain_from:
                            v *= 10.0            # single-station gain step
                        if sparse in (a, b) and j >= 5:
                            continue             # under-support endpoint
                        ser[d] = v
                    r[e] = ser
            return {"registry": list(sts), "registered_days": list(days40),
                    "r": r}

        def fam(car, nd=199, **kw):
            return WB1.w2_b1b_family(
                {"calendar": days40, "carriers": {"x": car}},
                doc_sha256="ef" * 32, n_draws=nd, **GEO, **kw)

        sts = [f"N{i}" for i in range(5)]
        # (1) the exact annex unit fixture + endpoint-order invariance
        ok1 = abs(WB1.edge_scale(8.0, 2.0, 3.0) - 8.0 / 3.0) < 1e-15 \
            and WB1.edge_scale(5.5, 2.0, 3.0) == WB1.edge_scale(5.5, 3.0,
                                                                2.0) \
            and WB1.RENORM_MIN_SUPPORT == 20 and WB1.WINSOR_C == 8.0
        # (2) determinism + four-path routing identity: identical calls
        # bit-identical; loco fold routes the SAME function/panel
        r2a = fam(carrier(_np.random.default_rng(21), sts),
                  return_null=True)
        r2b = fam(carrier(_np.random.default_rng(21), sts),
                  return_null=True)
        ok2 = r2a == r2b and fam(carrier(_np.random.default_rng(21), sts),
                                 fold="loco:N0")["T_obs"] == r2a["T_obs"]
        # (3) ZERO_SCALE_REFUSAL never-shrink: ONE under-support endpoint
        # types the WHOLE carrier/family; p None; no partial answer
        r3 = fam(carrier(_np.random.default_rng(22), sts, sparse="N4"))
        ok3 = "ZERO_SCALE_REFUSAL" in str(r3.get("verdict")) \
            and r3.get("p_value") is None and r3.get("T_obs") is None
        # (4) winsorization binds BEFORE window means: an eval spike of
        # raw magnitude ~100 caps at 8 -> T_obs <= 8 exactly, yet far
        # above the noise-only T
        spike_e = f"{sts[0]}|{sts[1]}"
        r4 = fam(carrier(_np.random.default_rng(23), sts,
                         spike_edge=spike_e,
                         spike_pos=(25, 26, 27, 28, 29, 30, 31)),
                 nd=99)
        r4n = fam(carrier(_np.random.default_rng(23), sts), nd=9)
        ok4 = r4["T_obs"] is not None and r4["T_obs"] <= 8.0 + 1e-12 \
            and r4["T_obs"] > 6.0 > (r4n["T_obs"] or 0)
        # (5) gain-step artifact (the KOZT class): x10 from an eval onset
        # on ONE station's incident edges -- capped ceiling + block
        # relocation keep it from producing a small p
        r5 = fam(carrier(_np.random.default_rng(24), sts, gain="N2",
                         gain_from=25), nd=199)
        ok5 = r5["p_value"] is not None and r5["p_value"] > 0.05
        # (6) geometry refusals (typed; PRESTART-fixed geometry contract)
        try:
            WB1.w2_b1b_family({"calendar": days40, "carriers": {}},
                              doc_sha256="ef" * 32, n_draws=9)
            ok6a = False
        except Exception as exc:
            ok6a = "GEOMETRY_ABSENT" in str(exc)
        try:
            WB1.w2_b1b_family({"calendar": days40, "carriers": {"x": {}}},
                              doc_sha256="ef" * 32, n_draws=9, n_blocks=4,
                              block_len=10, baseline_positions=25)
            ok6b = False
        except Exception as exc:
            ok6b = "GEOMETRY_NOT_BLOCK_ALIGNED" in str(exc)
        check("B1B annex KATs (edge_scale 8/3 + invariance, determinism + "
              "four-path routing, ZERO_SCALE never-shrink, winsor cap, "
              "gain-step artifact, geometry refusals)",
              ok1 and ok2 and ok3 and ok4 and ok5 and ok6a and ok6b,
              f"unit={ok1} det={ok2} zeroscale={ok3} winsor={ok4} "
              f"(T={r4.get('T_obs')}) gain={ok5} (p={r5.get('p_value')}) "
              f"geo={ok6a}/{ok6b}")
    except ImportError:
        check("B1B annex KATs", False, "W2_ENGINE_ABSENT")
    except Exception as exc:
        check("B1B annex KATs", False, f"{type(exc).__name__}: {exc}")


# ---- W-MF4: annex KATs wired vs w2_mf4 -------------------------------------
def w_mf4():
    try:
        import w2_mf4 as WM
        from datetime import date as _date, timedelta as _td

        ok_const = (WM.H_DAYS == 7 and WM.BLOCK_LEN == 14
                    and WM.B_REPLICATES == 999
                    and WM.CAL_START == "2025-10-18" and WM.MAG_MIN == 4.0
                    and WM.ROLL_MIN_PRIOR == 4 and WM.ROLL_WINDOW == 7)

        def days(a, b):
            d0, d1 = _date.fromisoformat(a), _date.fromisoformat(b)
            out = []
            while d0 <= d1:
                out.append(d0.isoformat())
                d0 += _td(days=1)
            return out

        span = days("2025-10-01", "2026-02-10")
        regions = ["R0", "R1", "R2"]
        bboxes = {r: {"min_lat": 10.0 * i, "max_lat": 10.0 * i + 5,
                      "min_lon": 10.0 * i, "max_lon": 10.0 * i + 5}
                  for i, r in enumerate(regions)}
        # deterministic arithmetic, never language hash() (salted per
        # process -- a nondeterministic fixture and the banned class)
        risk = {r: {d: 0.1 + 0.05 * (((ri * 31 + di * 7) % 7) / 7.0)
                    for di, d in enumerate(span)}
                for ri, r in enumerate(regions)}
        events = []
        for i, r in enumerate(regions):
            for dd in ("2025-11-05", "2025-12-01", "2025-12-20",
                       "2026-01-10"):
                events.append({"day": dd, "mag": 4.5,
                               "lat": 10.0 * i + 2, "lon": 10.0 * i + 2})

        led = WM.calibrate(risk, events, bboxes, regions,
                           freeze_day="2026-02-05",
                           snapshot_end="2026-02-05")
        # (1) label-maturity BYTE-LOCK: an event AFTER the snapshot end
        # cannot touch training rows -> digest + coefficients byte-equal
        led2 = WM.calibrate(risk, events + [{"day": "2026-02-07",
                                             "mag": 5.5, "lat": 2.0,
                                             "lon": 2.0}],
                            bboxes, regions, freeze_day="2026-02-05",
                            snapshot_end="2026-02-05")
        ok_lock = (led["training_digest"] == led2["training_digest"]
                   and led["coef"] == led2["coef"]
                   and led["intercept"] == led2["intercept"])
        # (2) CALIBRATION_LABEL_NOT_MATURE: past the matured bound
        try:
            WM.calibrate(risk, events, bboxes, regions,
                         freeze_day="2026-02-05",
                         snapshot_end="2026-02-05",
                         requested_issue_end="2026-01-30")
            ok_mature = False
        except Exception as exc:
            ok_mature = "CALIBRATION_LABEL_NOT_MATURE" in str(exc)
        # (3) issue-time violation + typed no-prediction rows
        try:
            WM.features({"2026-02-01": 0.1, "2026-02-02": 0.1},
                        events, bboxes["R0"], "2026-02-01")
            ok_issue = False
        except Exception as exc:
            ok_issue = "ISSUE_TIME_VIOLATION" in str(exc)
        # (4) immutable signed rows + duplicate + persistence baseline
        r0series = {d: risk["R0"][d] for d in days("2026-01-20",
                                                   "2026-02-01")}
        row = WM.predict_row(led, r0series, events, bboxes["R0"], "R0",
                             "2026-02-01", "2026-02-01T00:05:00Z")
        ok_row = WM.verify_row(dict(row)) and "p_persistence" in row \
            and "p_model" in row
        bad = dict(row, p_model=0.999)
        try:
            WM.verify_row(bad)
            ok_mut = False
        except Exception as exc:
            ok_mut = "PREDICTION_ROW_MUTATED" in str(exc)
        rows = WM.append_row([], row)
        try:
            WM.append_row(rows, row)
            ok_dup = False
        except Exception as exc:
            ok_dup = "PREDICTION_ROW_DUPLICATE" in str(exc)
        # typed no-prediction (missing prior day) emits a typing row
        row_t = WM.predict_row(led, {"2026-02-01": 0.1}, events,
                               bboxes["R0"], "R0", "2026-02-01", "t")
        ok_typed = "typing" in row_t and "NO_PREDICTION" in row_t["typing"]
        # (5) endpoint: zero-class + the >1/3 no-drop rule
        eval_days = days("2026-02-01", "2026-02-04")
        pred_rows = []
        for r in regions:
            for d in eval_days:
                s = {k: risk[r][k] for k in days("2026-01-20", d)}
                pr = WM.predict_row(led, s, events, bboxes[r], r, d, "t")
                if "typing" not in pr:
                    pred_rows = WM.append_row(pred_rows, pr)
        ev_eval = events + [{"day": "2026-02-03", "mag": 4.4,
                             "lat": 2.0, "lon": 2.0}]   # R0 only has class 1
        res = WM.score_endpoint(pred_rows, ev_eval, bboxes, regions,
                                eval_days, "ab" * 32, b=99)
        txt = json.dumps(res, default=str)
        ok_nodrop = "ENDPOINT_UNSCORABLE" in txt \
            and "R1" in txt and "R2" in txt   # 2/3 zero-class -> no-drop
        check("MF4 annex KATs (constants, label-maturity byte-lock, "
              "not-mature refusal, issue-time, immutable/dup rows + typed "
              "no-prediction, persistence baseline, zero-class no-drop)",
              ok_const and ok_lock and ok_mature and ok_issue and ok_row
              and ok_mut and ok_dup and ok_typed and ok_nodrop,
              f"const={ok_const} lock={ok_lock} mature={ok_mature} "
              f"issue={ok_issue} row={ok_row} mut={ok_mut} dup={ok_dup} "
              f"typed={ok_typed} nodrop={ok_nodrop}")
    except ImportError:
        check("MF4 annex KATs", False, "W2_ENGINE_ABSENT")
    except Exception as exc:
        check("MF4 annex KATs", False, f"{type(exc).__name__}: {exc}")


# ---- W-MAG: instantiation KATs wired vs w2_mag1 ----------------------------
def w_mag():
    try:
        import w2_mag1 as WMG
        import numpy as _np
        ok_const = (WMG.PADLEN == 27 and WMG.CAUSAL_SPAN == 266
                    and WMG.EDGE_EXCLUSION == 532 and WMG.DAY_FLOOR == 1296
                    and WMG.SPAN_THRESHOLD == 1e-12)
        # (1) SOS byte authority from git objects; disclosed scipy path
        sos, rec = WMG.load_sos(repo=_REPO)
        ok_sos = sos.shape == (4, 6) and rec["serialized_sha256"] == \
            WMG.SOS_SERIALIZED_SHA and rec["scipy_local"] is not None \
            and (rec["regenerated"] is (rec["scipy_local"]
                                        == WMG.PINNED_SCIPY))
        # (2) filter chain: segments split on NaN, never interpolate;
        # edge exclusion exact; short segment -> excluded wholesale
        ok_seg = (WMG.segment_usable_n(1064) == 0
                  and WMG.segment_usable_n(1065) == 1)
        v = _np.sin(_np.arange(4320) / 40.0)
        v[4200:] = _np.nan
        f = WMG.band_b_series(v, sos)
        fin_idx = _np.where(_np.isfinite(f))[0]
        ok_edge = fin_idx.size > 0 and fin_idx[0] == 532 \
            and fin_idx[-1] == 4200 - 532 - 1
        short = _np.ones(1000)
        ok_short = not _np.any(_np.isfinite(WMG.band_b_series(short, sos)))
        de = WMG.daily_energy(f, {"d0": (0, 1440), "d1": (1440, 2880),
                                  "d2": (2880, 4320)})
        ok_floor = (de["d0"]["typing"] == "DAY_BELOW_FLOOR"
                    and de["d1"]["typing"] is None
                    and de["d1"]["surviving"] == 1440
                    and de["d2"]["typing"] == "DAY_BELOW_FLOOR")
        # (3) real capsules load with sha-verified bodies; frame doctors
        cap_v, body_v = WMG.load_capsule("vic", repo=_REPO)
        cap_n, body_n = WMG.load_capsule("new", repo=_REPO)
        ok_caps = cap_v.get("sensor_orientation") == "XYZS" \
            and "probe_body_sha256" in cap_n

        def refuses(fn, code):
            try:
                fn()
                return False
            except Exception as exc:
                return code in str(exc)

        arrs = {"X": [1.0, 2.0], "Y": [3.0, 4.0], "Z": [5.0, 6.0],
                "S": [7.0, 8.0]}
        cmap = {"geographic_X_north": "X", "geographic_Y_east": "Y",
                "geographic_Z_down": "Z"}
        syn = {"sensor_orientation": "XYZS", "component_map": dict(cmap)}
        x, y, z = WMG.convert_frame(syn, arrs, "XYZS")
        ok_conv = list(x) == [1.0, 2.0] and list(y) == [3.0, 4.0]
        # map-less REPORTED path (the NEW defect-fix class)
        xr, yr, zr = WMG.convert_frame(
            {"reported_orientation": "XYZF"}, arrs, "XYZF")
        ok_conv = ok_conv and list(xr) == [1.0, 2.0]
        # IZN angular path: the pinned hand fixture (H=100, D=30deg)
        xa, ya, za = WMG.convert_frame(
            {"sensor_orientation": "HDZS",
             "component_map": {"present": True},
             "declination_units": "degrees"},
            {"H": [100.0], "D": [30.0], "Z": [7.0]}, "HDZS")
        ok_ang = abs(xa[0] - 86.60254037844388) < 1e-9 \
            and abs(ya[0] - 50.0) < 1e-9
        ok_d1 = refuses(lambda: WMG.convert_frame(
            dict(syn, sensor_orientation="XYZQ"), arrs, "XYZQ"),
            "FRAME_NOT_CLOSED")
        ok_d2 = refuses(lambda: WMG.convert_frame(
            {"sensor_orientation": "XYZS", "component_map":
             dict(cmap, geographic_X_north="S")}, arrs, "XYZS"),
            "EXCLUDED_CHANNEL_IN_HORIZONTAL")
        ok_d3 = refuses(lambda: WMG.convert_frame(
            syn, {k: v_ for k, v_ in arrs.items() if k != "Y"}, "XYZS"),
            "FRAME_NOT_CLOSED")
        # (4) endpoint typing incl the untestable carrier
        ok_ep = WMG.endpoints_for("cascadia") == ("M1", "M2", "M3") \
            and refuses(lambda: WMG.endpoints_for("turkey_kahramanmaras"),
                        "MAG_UNTESTABLE")
        # (5) internal 3-primary Holm, hand-derived both directions
        h1 = WMG.holm_internal({("istanbul_marmara", "M2"): 0.010,
                                ("socal_coachella", "M3"): 0.020,
                                ("cascadia", "M3"): 0.040})
        h2 = WMG.holm_internal({("istanbul_marmara", "M2"): 0.020,
                                ("socal_coachella", "M3"): 0.020,
                                ("cascadia", "M3"): 0.040})
        # hand-derived: m=3, alpha .05 -> thresholds .05/3, .05/2, .05/1
        # h1: .010<=.016667, .020<=.025, .040<=.05 -> all reject
        # h2: smallest .020 > .016667 -> step-down stops -> none reject
        ok_h1 = h1["alpha"] == 0.05 \
            and h1["order"][0] == ["istanbul_marmara", "M2"] \
            and h1["rejected"] == {"istanbul_marmara:M2": True,
                                   "socal_coachella:M3": True,
                                   "cascadia:M3": True}
        ok_h2 = h2["rejected"] == {"istanbul_marmara:M2": False,
                                   "socal_coachella:M3": False,
                                   "cascadia:M3": False} \
            and refuses(lambda: WMG.holm_internal(
                {("istanbul_marmara", "M2"): 0.01}),
                "HOLM_STRUCTURE_MISMATCH")
        check("MAG instantiation KATs (constants, SOS authority + scipy "
              "disclosure, filter chain + day floor, capsules + frame "
              "doctors, MAG_UNTESTABLE, 3-primary Holm both ways)",
              ok_const and ok_sos and ok_seg and ok_edge and ok_short
              and ok_floor and ok_caps and ok_conv and ok_ang and ok_d1
              and ok_d2 and ok_d3 and ok_ep and ok_h1 and ok_h2,
              f"const={ok_const} sos={ok_sos} seg={ok_seg} edge={ok_edge} "
              f"short={ok_short} floor={ok_floor} caps={ok_caps} "
              f"conv={ok_conv} ang={ok_ang} "
              f"doctors={ok_d1}/{ok_d2}/{ok_d3} ep={ok_ep} "
              f"holm={ok_h1}/{ok_h2}")
    except ImportError:
        check("MAG instantiation KATs", False, "W2_ENGINE_ABSENT")
    except Exception as exc:
        check("MAG instantiation KATs", False,
              f"{type(exc).__name__}: {exc}")


# ---- REV 9 (the convergence close packet; codex 1335Z/1358Z item 5 +
# 1721Z sequencing + 1423Z/1519Z null guards) -------------------------------
def w_mag_exec():
    """REV 9 group 1: the REAL execution-bound MAG capsule set through
    the execution-manifest authority -- all three capsules + envelopes
    + bodies, independent body-sha recompute, real parsers, end-to-end
    frame conversion (IZN degrees + scalar-S exclusion), moved-path and
    unknown-name doctors."""
    try:
        import math as _m
        import w2_mag1 as WMG

        def refuses(fn, code):
            try:
                fn()
                return False
            except WMG.Mag1Refusal as e:
                return str(e).startswith(code)

        mc = subprocess.run(
            ["git", "-C", _REPO, "log", "-1", "--format=%H", "--",
             WMG.EXEC_MANIFEST_PATH],
            capture_output=True, text=True).stdout.strip()
        man = json.loads(_blob(
            f"{mc}:{WMG.EXEC_MANIFEST_PATH}").decode("utf-8"))
        slot = man["slots"]["mag_capsules"]["status"]
        expect_mode = "pin_checked" if slot == "BOUND" else "pre_bind"
        target = man["execution_target_commit"]

        # IZN -- GIN angular HDZS, declination in DEGREES
        cap_i, body_i, rec_i = WMG.load_execution_capsule("izn", mc)
        arrs_i = WMG.gin_arrays(body_i)
        raw_i = _blob(f"{target}:{rec_i['body_path']}")
        ok_izn = rec_i["mode"] == expect_mode \
            and {"H", "D", "Z", "S"} <= set(arrs_i) \
            and hashlib.sha256(raw_i).hexdigest() \
            == cap_i["probe_body_sha256"] \
            and len(arrs_i["H"]) == cap_i["recomputed_coverage_samples"]
        xi, yi, zi = WMG.convert_frame(cap_i, arrs_i, "HDZS")
        k = next(j for j, (h, d) in enumerate(zip(arrs_i["H"],
                                                  arrs_i["D"]))
                 if h is not None and d is not None)
        h_k, d_k = float(arrs_i["H"][k]), float(arrs_i["D"][k])
        ok_izn = ok_izn \
            and abs(xi[k] - h_k * _m.cos(_m.radians(d_k))) < 1e-9 \
            and abs(yi[k] - h_k * _m.sin(_m.radians(d_k))) < 1e-9
        # scalar-S exclusion is structural: the angular path needs only
        # H/D/Z -- removing S entirely must still convert
        xs2, _, _ = WMG.convert_frame(
            cap_i, {q: v for q, v in arrs_i.items() if q != "S"},
            "HDZS")
        ok_izn = ok_izn and abs(xs2[k] - xi[k]) < 1e-15

        # FRN + TUC -- USGS map-less reported XYZF (identity elements)
        ok_us = True
        for name in ("frn", "tuc"):
            cap_u, body_u, rec_u = WMG.load_execution_capsule(name, mc)
            arrs_u = WMG.usgs_arrays(body_u)
            raw_u = _blob(f"{target}:{rec_u['body_path']}")
            ok_us = ok_us and rec_u["mode"] == expect_mode \
                and cap_u["reported_orientation"] == "XYZF" \
                and hashlib.sha256(raw_u).hexdigest() \
                == cap_u["probe_body_sha256"] \
                and {"X", "Y", "Z"} <= set(arrs_u) \
                and len(arrs_u["X"]) \
                == cap_u["recomputed_coverage_samples"]
            xu, yu, zu = WMG.convert_frame(cap_u, arrs_u, "XYZF")
            j = next(i for i, v in enumerate(arrs_u["X"])
                     if v is not None)
            ok_us = ok_us and abs(xu[j] - float(arrs_u["X"][j])) \
                < 1e-15 and len(xu) == len(arrs_u["X"])

        # doctors: unknown name; a PRE-RELOCATION manifest commit (the
        # moved-path class -- its target lacks the execution tree)
        ok_d1 = refuses(lambda: WMG.load_execution_capsule("vic", mc),
                        "CAPSULE_UNKNOWN")
        first_mc = subprocess.run(
            ["git", "-C", _REPO, "log", "--format=%H", "--",
             WMG.EXEC_MANIFEST_PATH],
            capture_output=True, text=True).stdout.split()[-1]
        ok_d2 = refuses(
            lambda: WMG.load_execution_capsule("izn", first_mc),
            "ARTIFACT_UNREADABLE")

        check("MAG-EXEC real execution capsules (manifest-authority "
              "loads x3, independent body-sha recompute, GIN/USGS "
              "parsers, IZN degree conversion + scalar-S exclusion, "
              "XYZF identity, unknown-name + moved-path doctors)",
              ok_izn and ok_us and ok_d1 and ok_d2,
              f"izn={ok_izn} usgs={ok_us} doctors={ok_d1}/{ok_d2} "
              f"mode_expected={expect_mode}")
    except ImportError:
        check("MAG-EXEC real execution capsules", False,
              "W2_ENGINE_ABSENT")
    except Exception as exc:
        check("MAG-EXEC real execution capsules", False,
              f"{type(exc).__name__}: {exc}")


def w_mag_b():
    """REV 9 group 2: part-B statistic surfaces -- the codex hand
    fixtures verbatim (Spearman tie 7/9, window_energy 6.5, m1_p
    add-one endpoints), monotone/zero-variance behavior, and the
    subtraction/M3 ledger discipline."""
    try:
        import numpy as np
        import w2_mag1 as WMG

        def refuses(fn, code):
            try:
                fn()
                return False
            except WMG.Mag1Refusal as e:
                return str(e).startswith(code)

        sp = WMG._spearman([1, 1, 2, 3], [10, 10, 30, 20])
        ok_sp = abs(sp - 7.0 / 9.0) < 1e-15 \
            and WMG._spearman([2, 2, 5, 7], [1000, 1000, 27000, 8000]) \
            == sp \
            and WMG._spearman([1, 1, 1, 1], [1, 2, 3, 4]) == 0.0
        ok_we = WMG.window_energy([1.0, -2.0, float("nan"), 3.0, 4.0],
                                  0, 5) == 6.5 \
            and refuses(lambda: WMG.window_energy(
                [float("nan")] * 3, 0, 3, min_support=1),
                "M1_WINDOW_SUPPORT_INSUFFICIENT")
        ok_p = WMG.m1_p(10.0, [1.0] * 4) == 1.0 / 5.0 \
            and WMG.m1_p(0.0, [1.0] * 4) == 1.0

        # subtraction ledger discipline on a small planted fixture
        from datetime import datetime as _dtb, timedelta as _tdb
        n = 3000
        rng = np.random.Generator(np.random.PCG64(77))
        times = [(_dtb(2026, 1, 1) + _tdb(minutes=i)).isoformat()
                 for i in range(n)]
        wx = {"symh": rng.normal(size=n).tolist()}
        vals = (20000 + 0.5 * np.asarray(wx["symh"])
                + rng.normal(scale=0.1, size=n)).tolist()
        led = WMG.fit_subtraction(times, vals, -120.0, wx)
        resid = WMG.apply_subtraction(led, times, vals, wx)
        ok_fit = bool(np.isfinite(resid).all()) \
            and abs(float(np.mean(resid))) < 1.0
        bad = json.loads(json.dumps(led))
        bad["coef"][0] += 1.0
        ok_led = refuses(
            lambda: WMG.apply_subtraction(bad, times, vals, wx),
            "LEDGER_MUTATED")
        ok_sup = refuses(
            lambda: WMG.fit_subtraction(times[:10], vals[:10], -120.0,
                                        {"symh": wx["symh"][:10]}),
            "SUBTRACTION_INSUFFICIENT_SUPPORT")
        ok_rank = refuses(
            lambda: WMG.fit_subtraction(
                times, vals, -120.0,
                {"symh": wx["symh"], "symh2": list(wx["symh"])}),
            "SUBTRACTION_DESIGN_RANK_DEFICIENT")
        # 3 design cols (intercept + reference + 1 weather): floor is
        # 6 rows, so 5 rows -> support; 6 constant rows -> rank
        ok_m3 = refuses(
            lambda: WMG.fit_m3_reference([1.0] * 5, [1.0] * 5,
                                         {"symh": [0.0] * 5}),
            "M3_INSUFFICIENT_SUPPORT") \
            and refuses(
                lambda: WMG.fit_m3_reference([1.0] * 6, [1.0] * 6,
                                             {"symh": [0.0] * 6}),
                "M3_DESIGN_RANK_DEFICIENT")

        check("MAG-B part-B statistics (Spearman midrank tie 7/9 + "
              "monotone invariance + zero-variance 0, window_energy "
              "6.5 + support refusal, m1_p add-one endpoints, "
              "subtraction fit/apply ledger discipline + typed "
              "refusals, M3 support refusal)",
              ok_sp and ok_we and ok_p and ok_fit and ok_led
              and ok_sup and ok_rank and ok_m3,
              f"sp={ok_sp} we={ok_we} p={ok_p} fit={ok_fit} "
              f"led={ok_led} sup={ok_sup} rank={ok_rank} m3={ok_m3}")
    except ImportError:
        check("MAG-B part-B statistics", False, "W2_ENGINE_ABSENT")
    except Exception as exc:
        check("MAG-B part-B statistics", False,
              f"{type(exc).__name__}: {exc}")


def w_mag_null():
    """REV 9 groups 3+4: the registered feature-capsule null -- raw
    recomputation non-equivalence reproduced, full in-bar independent
    oracle (own midrank Spearman + rotation census) matched exactly,
    operation-record binding recomputed, capsule digest recomputed,
    non-finite boundary doctors, finite-pair floor boundary."""
    try:
        import numpy as np
        import w2_mag1 as WMG
        from datetime import date as _dn, timedelta as _tdn

        def refuses(fn, code):
            try:
                fn()
                return False
            except WMG.Mag1Refusal as e:
                return str(e).startswith(code)

        ok_const = WMG.M2_LAGS == (0, 1, -1, 2, -2, 3, -3) \
            and WMG.M2_MIN_OVERLAP == 60 and WMG.M2_EXCLUDED_OFFSET == 3

        # (3a) raw-recomputation NON-equivalence: byte-identical target
        # day, neighbor day zeros vs strong noise -> different median
        # energy (sosfiltfilt support crosses day boundaries)
        sos, _ = WMG.load_sos(repo=_REPO)
        rng = np.random.Generator(np.random.PCG64(31))
        dayvecs = [rng.normal(size=1440) for _ in range(7)]
        va = np.concatenate([np.zeros(1440) if i == 2 else dayvecs[i]
                             for i in range(7)])
        vb = np.concatenate([5.0 * rng.normal(size=1440) if i == 2
                             else dayvecs[i] for i in range(7)])
        assert (va[3 * 1440:4 * 1440]
                == vb[3 * 1440:4 * 1440]).all()
        fa = WMG.band_b_series(va, sos)
        fb = WMG.band_b_series(vb, sos)
        ea = WMG.window_energy(fa, 3 * 1440, 4 * 1440)
        eb = WMG.window_energy(fb, 3 * 1440, 4 * 1440)
        ok_nonlocal = ea != eb

        # (3b) in-bar independent oracle over a 70-day fixture with
        # typed absences: T_obs, eligible-offset CENSUS, n_null and p
        # must match EXACTLY (no offset may vanish silently)
        days70 = [(_dn(2026, 3, 1) + _tdn(days=i)).isoformat()
                  for i in range(70)]
        mag_d = {d: float(rng.normal()) for d in days70}
        mag_d[days70[10]] = None
        mag_d[days70[45]] = None
        gr_d = {d: float(rng.normal()) for d in days70}

        def oracle_ranks(x):
            order = np.argsort(x, kind="mergesort")
            rk = np.empty(len(x), dtype=float)
            sx = np.asarray(x)[order]
            i = 0
            while i < len(x):
                j = i
                while j + 1 < len(x) and sx[j + 1] == sx[i]:
                    j += 1
                rk[order[i:j + 1]] = (i + j) / 2.0 + 1.0
                i = j + 1
            return rk

        def oracle_spearman(a, b):
            ra, rb = oracle_ranks(np.asarray(a, float)), \
                oracle_ranks(np.asarray(b, float))
            ra = ra - ra.mean()
            rb = rb - rb.mean()
            den = (float((ra ** 2).sum())
                   * float((rb ** 2).sum())) ** 0.5
            if den == 0.0:
                return 0.0
            return float((ra * rb).sum() / den)

        def oracle_stat(m_by_day, g_by_day, days):
            pos = {d: i for i, d in enumerate(days)}
            best = None
            for lag in (0, 1, -1, 2, -2, 3, -3):
                pairs = []
                for d in days:
                    jj = pos[d] + lag
                    if 0 <= jj < len(days):
                        mv = m_by_day.get(days[jj])
                        gv = g_by_day.get(d)
                        if mv is not None and gv is not None \
                                and np.isfinite(mv) \
                                and np.isfinite(gv):
                            pairs.append((mv, gv))
                if len(pairs) < 60:
                    continue
                rho = oracle_spearman([p[0] for p in pairs],
                                      [p[1] for p in pairs])
                if best is None or rho > best:
                    best = rho
            return best        # None == typed insufficient overlap

        obs_o = oracle_stat(mag_d, gr_d, days70)
        nulls_o, elig_o = [], []
        for off in range(70):
            if min(off, 70 - off) <= 3:
                continue
            rot = {days70[(i + off) % 70]: mag_d.get(days70[i])
                   for i in range(70)}
            s = oracle_stat(rot, gr_d, days70)
            if s is not None:
                nulls_o.append(s)
                elig_o.append(off)
        p_o = (1 + sum(1 for s in nulls_o if s >= obs_o)) \
            / (len(nulls_o) + 1)

        res = WMG.m2_pairing(mag_d, gr_d, days70,
                             subtraction_ledger_digest="ab" * 32,
                             sos_digest="cd" * 32,
                             source_input_digest="ef" * 32)
        ok_oracle = res["T_obs"] == obs_o \
            and res["eligible_offsets"] == elig_o \
            and res["n_null"] == len(nulls_o) \
            and res["p_value"] == float(p_o) \
            and res["eligible_offsets"] == list(range(4, 67))
        ok_support = res["capsule"]["surviving_support"] == 68

        # (3c) binding recomputation: capsule digest, graph day-index
        # digest, implementation sha, frozen parameters
        cc = dict(res["capsule"])
        cd = cc.pop("capsule_digest")
        ok_capd = hashlib.sha256(json.dumps(
            cc, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest() == cd
        opr = res["operation_record"]
        gd = hashlib.sha256(json.dumps(
            {"days": days70, "values": {d: gr_d.get(d)
                                        for d in days70}},
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        with open(os.path.join(_HERE, "w2_mag1.py"), "rb") as fsrc:
            impl = hashlib.sha256(
                fsrc.read().replace(b"\r\n", b"\n")).hexdigest()
        ok_bind = opr["capsule_digest"] == cd \
            and opr["graph_day_index_digest"] == gd \
            and opr["implementation_sha256_normalized"] == impl \
            and opr["lags"] == [0, 1, -1, 2, -2, 3, -3] \
            and opr["min_overlap"] == 60 \
            and opr["excluded_offsets_rule"] == "|offset| <= 3" \
            and "capsule" in opr["null"]

        # (4) non-finite boundary + finite-pair floor
        days60 = days70[:60]
        m60 = {d: float(rng.normal()) for d in days60}
        g60 = {d: float(rng.normal()) for d in days60}
        m_nan = dict(m60)
        m_nan[days60[5]] = float("nan")
        m_inf = dict(m60)
        m_inf[days60[6]] = float("inf")
        g_nan = dict(g60)
        g_nan[days60[7]] = float("nan")
        kw = dict(subtraction_ledger_digest="ab" * 32,
                  sos_digest="cd" * 32, source_input_digest="ef" * 32)
        ok_nf = refuses(lambda: WMG.m2_pairing(m_nan, g60, days60,
                                               **kw),
                        "M2_NONFINITE_INPUT") \
            and refuses(lambda: WMG.m2_pairing(m_inf, g60, days60,
                                               **kw),
                        "M2_NONFINITE_INPUT") \
            and refuses(lambda: WMG.m2_pairing(m60, g_nan, days60,
                                               **kw),
                        "M2_NONFINITE_INPUT")
        # 61 days, one typed None -> exactly 60 finite lag-0 pairs:
        # the floor passes and every eligible offset survives (census)
        days61 = days70[:61]
        m61 = {d: float(rng.normal()) for d in days61}
        m61[days61[30]] = None
        g61 = {d: float(rng.normal()) for d in days61}
        res61 = WMG.m2_pairing(m61, g61, days61, **kw)
        ok_floor = res61["capsule"]["surviving_support"] == 60 \
            and res61["eligible_offsets"] == list(range(4, 58)) \
            and res61["n_null"] == 54 \
            and np.isfinite(res61["T_obs"])

        # (5) 1815Z item-5 temporal-carrier doctors: the codex
        # duplicate-inflation repro (59 unique + 1 repeat), reordered,
        # gapped, invalid, extra -- all refuse BEFORE the statistic
        dup = days60[:59] + [days60[10]]
        ok_didx = refuses(lambda: WMG.m2_pairing(m60, g60, dup, **kw),
                          "M2_DAY_INDEX_INVALID") \
            and refuses(lambda: WMG.m2_pairing(
                m60, g60, list(reversed(days60)), **kw),
                "M2_DAY_INDEX_INVALID") \
            and refuses(lambda: WMG.m2_pairing(
                m60, g60, days60[:30] + days60[31:], **kw),
                "M2_DAY_INDEX_INVALID") \
            and refuses(lambda: WMG.m2_pairing(
                m60, g60, ["not-a-day"] + days60[1:], **kw),
                "M2_DAY_INDEX_INVALID")

        # (2-adjacent) the certification data gate is unmintable from
        # the bar side: a forged caller geometry dict must refuse
        # before any replicate
        try:
            import w2_power_harness_cayley as WPH

            def _forge(ref, code):
                try:
                    WPH.run_point_certification(_REPO, ref, "B2A",
                                                {"m": 3})
                    return False
                except Exception as exc:
                    return code in str(exc)
            mc9 = subprocess.run(
                ["git", "-C", _REPO, "log", "-1", "--format=%H", "--",
                 "docs/f2g_window2_execution/execution_manifest.json"],
                capture_output=True, text=True).stdout.strip()
            ok_forge = _forge(
                {"bound": True, "schema": "forged", "registries": {},
                 "segments": {}}, "POWER_GEOMETRY_REF_INVALID") \
                and _forge({"manifest_commit": mc9,
                            "path": "docs/never/pinned.json"},
                           "POWER_GEOMETRY_NOT_MANIFEST_PINNED")
        except ImportError:
            ok_forge = False

        check("MAG-NULL feature-capsule null (raw non-equivalence "
              "reproduced, full in-bar oracle EXACT incl offset "
              "census, capsule+graph+impl binding recomputed, "
              "non-finite boundary doctors, 61-day finite-floor "
              "boundary, temporal-carrier doctors, forged-geometry "
              "gate)",
              ok_const and ok_nonlocal and ok_oracle and ok_support
              and ok_capd and ok_bind and ok_nf and ok_floor
              and ok_didx and ok_forge,
              f"const={ok_const} nonlocal={ok_nonlocal} "
              f"oracle={ok_oracle} support={ok_support} "
              f"capd={ok_capd} bind={ok_bind} nf={ok_nf} "
              f"floor={ok_floor} didx={ok_didx} forge={ok_forge}")
    except ImportError:
        check("MAG-NULL feature-capsule null", False,
              "W2_ENGINE_ABSENT")
    except Exception as exc:
        check("MAG-NULL feature-capsule null", False,
              f"{type(exc).__name__}: {exc}")


# ---- REV 11: the LOCO-composition amendment's five locking KATs -----------
def w_loco():
    """codex 1933Z contract, grassmann-ratified 2328Z, cayley-bound
    1a9ba2f. Five groups: (1) Holm substitution-not-marginal; (2) exact
    partial recomputation vs an in-bar reference that holds the other
    three coordinates byte-identical; (3) projection/provenance
    doctors; (4) typed-fold vs missing-fold state split; (5) the
    specificity anti-rescue, structural + semantic."""
    try:
        import inspect
        import w2_power_harness_cayley as WPH
        import w2_b1b as WB1

        def refuses(fn, code):
            try:
                fn()
                return False
            except Exception as exc:
                return code in str(exc)

        # in-bar independent Holm-4 step-down (my own, not the engine)
        def my_holm(pv, alpha=0.05):
            order = sorted(pv, key=lambda k: pv[k])
            out = set()
            for i, k in enumerate(order):
                if pv[k] <= alpha / (len(order) - i):
                    out.add(k)
                else:
                    break
            return out

        # (1) Holm substitution, not p <= .05 -- the codex hand
        # fixture verbatim, engine vs my own step-down
        pv1 = {"B2A": .001, "B2B": .010, "B1B": .024, "B3A": .8}
        pv1s = dict(pv1, B1B=.030)
        ok_k1 = "B1B" in WPH.holm_rejects(pv1) \
            and "B1B" not in WPH.holm_rejects(pv1s) \
            and .030 <= .05 \
            and my_holm(pv1) == set(WPH.holm_rejects(pv1)) \
            and my_holm(pv1s) == set(WPH.holm_rejects(pv1s))

        # fixture: the v2 FRAME (REV 12 rebase, calendar ruling 1) --
        # views over the committed 192-day authority calendar, B1B
        # geometry = the AUTHORITY fields (16 x 12, baseline 60), no
        # geometry knob anywhere. The capped signal run = last 3 days
        # of block 6 + first 4 days of block 7 (positions 81..87, all
        # inside evaluation), so a 7-day all-capped window exists ONLY
        # under the ordered PAIR (6,7) adjacency -- a 7-run can span
        # at most two 12-day blocks, no single block carries >= 7
        # caps, and the reversed order leaves the runs discontiguous.
        # Spikes VARY with j (no permuted baseline mix zeroes a MAD).
        # ND=499 draws: every fold p lands <= .05 deterministically at
        # this seed authority (fold A at exactly .05 -- the boundary).
        FRAME = WPH.w2_calendar_frame()
        ENG = FRAME["engine_days"]
        CAPS = set(range(81, 88))
        REG4 = ["A", "B", "C", "D"]
        ND = 499

        def mk_view(signal_stations):
            r = {}
            for i, a in enumerate(REG4):
                for b in REG4[i + 1:]:
                    e = f"{a}|{b}"
                    hot = a in signal_stations or b in signal_stations
                    ser = {}
                    for j, d in enumerate(ENG):
                        base = 0.1 + 0.01 * ((j * 7 + i) % 5)
                        ser[d] = (10.0 + 0.05 * j) \
                            if (hot and j in CAPS) else base
                    r[e] = ser
            return {"calendar": list(ENG),
                    "carriers": {"c1": {
                        "registry": list(REG4),
                        "registered_days": list(ENG), "r": r}}}

        def capsule(reg=REG4):
            return {"loco_registry_carrier": "c1",
                    "registries": {"c1": list(reg)},
                    "calendar_frame": json.loads(
                        json.dumps(FRAME))}

        DS = "ab" * 32

        def oracle_recovery(view, pv_full, reg):
            gb = FRAME["b1b"]
            if "B1B" not in my_holm(pv_full):
                return None                    # early-exit class
            ok = True
            for s in sorted(reg):
                proj = WPH.b1b_loco_project(view, s, carrier="c1")
                try:
                    p_s = WB1.w2_b1b_family(
                        proj, doc_sha256=DS, n_draws=ND,
                        fold=f"loco:{s}", n_blocks=gb["n_blocks"],
                        block_len=gb["block_len"],
                        baseline_positions=gb["baseline_positions"]
                    )["p_value"]
                except WB1.PanelInvalid:
                    p_s = None
                if p_s is None or "B1B" not in my_holm(
                        dict(pv_full, B1B=p_s)):
                    ok = False
            return ok

        # (2) exact partial recomputation: engine == my reference on
        # BOTH the all-pairs (recovery-True) and the single-station
        # (recovery-False) constructions; the other three coordinates
        # are the SAME pv_full objects (byte-identical reuse); after a
        # full positive the fold count is exactly |R_NEW|
        pv_ok = {"B2A": .001, "B2B": .002, "B3A": .003, "B1B": .020}
        v_all = mk_view(set(REG4))
        fc = []
        eng_all = WPH._b1b_loco_recovery({"b1b": v_all}, pv_ok,
                                         capsule(), ND, DS,
                                         fold_counter=fc)
        ora_all = oracle_recovery(v_all, pv_ok, REG4)
        v_gain = mk_view({"D"})
        fc2 = []
        eng_gain = WPH._b1b_loco_recovery({"b1b": v_gain}, pv_ok,
                                          capsule(), ND, DS,
                                          fold_counter=fc2)
        ora_gain = oracle_recovery(v_gain, pv_ok, REG4)
        ok_k2 = eng_all is True and eng_all == ora_all \
            and eng_gain is False and eng_gain == ora_gain \
            and fc == sorted(REG4) and fc2 == sorted(REG4) \
            and "b2a" not in inspect.getsource(
                WPH._b1b_loco_recovery).lower()
        # early-exit without folds on full-Holm non-rejection
        pv_no = {"B2A": .001, "B2B": .002, "B3A": .003, "B1B": .8}
        fc3 = []
        ok_k2 = ok_k2 and WPH._b1b_loco_recovery(
            {"b1b": v_all}, pv_no, capsule(), ND, DS,
            fold_counter=fc3) is False and fc3 == []

        # (3) projection exactness + carrier scoping + provenance
        # doctors (8ae9c22 repair 2 cross-authored)
        proj_b = WPH.b1b_loco_project(v_all, "B", carrier="c1")
        pc = proj_b["carriers"]["c1"]
        vc = v_all["carriers"]["c1"]
        ok_k3 = pc["registry"] == ["A", "C", "D"] \
            and proj_b["calendar"] == v_all["calendar"] \
            and pc["registered_days"] == vc["registered_days"] \
            and set(pc["r"]) == {e for e in vc["r"]
                                 if "B" not in e.split("|")} \
            and all(pc["r"][e] == vc["r"][e] for e in pc["r"])
        # carrier collision: shared station across two carriers --
        # unqualified is AMBIGUOUS and refuses; scoped projection
        # changes ONLY the target carrier (c2 byte-identical)
        v_two = json.loads(json.dumps(v_all))
        v_two["carriers"]["c2"] = {
            "registry": ["A", "Q"],
            "registered_days": list(ENG),
            "r": {"A|Q": {d: 0.2 for d in ENG}}}
        ok_k3 = ok_k3 and refuses(
            lambda: WPH.b1b_loco_project(v_two, "A"),
            "POWER_LOCO_FOLD_SET_INVALID") \
            and refuses(
                lambda: WPH.b1b_loco_project(v_two, "Q",
                                             carrier="c1"),
                "POWER_LOCO_FOLD_SET_INVALID")
        proj_a1 = WPH.b1b_loco_project(v_two, "A", carrier="c1")
        ok_k3 = ok_k3 \
            and proj_a1["carriers"]["c2"] == v_two["carriers"]["c2"] \
            and "A" not in proj_a1["carriers"]["c1"]["registry"]
        for bad in (["A"], ["A", "B", "C"], ["A", "A", "B"],
                    ["A", "Z"]):
            ok_k3 = ok_k3 and refuses(
                lambda b=bad: WPH.verify_fold_set(b, ["A", "B"]),
                "POWER_LOCO_FOLD_SET_INVALID")
        # the fold token routes the null substream: same projection,
        # different fold -> different null vector; token recorded
        gb = FRAME["b1b"]
        r_f = WB1.w2_b1b_family(proj_b, doc_sha256=DS, n_draws=25,
                                fold="loco:B", return_null=True,
                                n_blocks=gb["n_blocks"],
                                block_len=gb["block_len"],
                                baseline_positions=gb[
                                    "baseline_positions"])
        r_g = WB1.w2_b1b_family(proj_b, doc_sha256=DS, n_draws=25,
                                fold="full", return_null=True,
                                n_blocks=gb["n_blocks"],
                                block_len=gb["block_len"],
                                baseline_positions=gb[
                                    "baseline_positions"])
        ok_k3 = ok_k3 and r_f["fold"] == "loco:B" \
            and r_f["null_T"] != r_g["null_T"]

        # (4) typed fold vs missing fold vs PROVENANCE (8ae9c22 repair
        # 3 cross-authored): a 2-station registry degenerates every
        # fold (projection-induced EDGE_SET_EMPTY -> typed no-p) ->
        # recovery False WITHOUT a raise, fold set exact; a missing
        # fold REFUSES; a calendar defect in the fold view raises
        # POWER_LOCO_PROVENANCE_INVALID -- three states, never
        # collapsed
        v2 = {"calendar": list(ENG),
              "carriers": {"c1": {
                  "registry": ["A", "B"],
                  "registered_days": list(ENG),
                  "r": {"A|B": {d: ((10.0 + 0.05 * j) if j in CAPS
                                    else 0.1 + 0.01 * (j % 5))
                                for j, d in enumerate(ENG)}}}}}
        fc4 = []
        typed = WPH._b1b_loco_recovery({"b1b": v2}, pv_ok,
                                       capsule(["A", "B"]), 49, DS,
                                       fold_counter=fc4)
        ok_k4 = typed is False and fc4 == ["A", "B"] \
            and refuses(lambda: WPH.verify_fold_set(
                ["A"], ["A", "B"]), "POWER_LOCO_FOLD_SET_INVALID")
        v_badcal = json.loads(json.dumps(v2))
        v_badcal["calendar"] = list(reversed(ENG))
        ok_k4 = ok_k4 and refuses(
            lambda: WPH._b1b_loco_recovery(
                {"b1b": v_badcal}, pv_ok, capsule(["A", "B"]), 49,
                DS), "POWER_LOCO_PROVENANCE_INVALID")

        # (5) specificity anti-rescue (8ae9c22 repair 1
        # cross-authored): the detection entry REFUSES gain points
        # (exercised directly -- the guard fires before any capsule
        # load); the artifact class has no LOCO path; the counting
        # rule's boundary arithmetic is the annex ceiling (2/40 = .05
        # PASS, 3/40 FAIL); semantically the single-station
        # construction is a pre-LOCO full-Holm positive while LOCO
        # recovery is False -- exactly what anti-rescue preserves
        src_art = inspect.getsource(WPH.run_artifact_class).lower()
        src_spec = inspect.getsource(
            WPH.run_b1b_specificity_certification)
        ok_k5 = "loco" not in src_art \
            and refuses(
                lambda: WPH.run_point_certification(
                    _REPO, {"manifest_commit": "x", "path": "y"},
                    "B1B", {"gain": 10.0}),
                "POWER_SPECIFICITY_ENTRYPOINT_REQUIRED") \
            and WPH.ARTIFACT_MAX_RATE == 0.05 \
            and (0 / 40) <= WPH.ARTIFACT_MAX_RATE \
            and (2 / 40) <= WPH.ARTIFACT_MAX_RATE \
            and not ((3 / 40) <= WPH.ARTIFACT_MAX_RATE) \
            and "holm_rejects" in src_spec \
            and "B1B" in my_holm(pv_ok) \
            and eng_gain is False

        check("LOCO amendment locking KATs (Holm substitution not "
              "marginal, exact partial recompute vs in-bar reference "
              "+ fold census + early-exit, projection exactness + "
              "fold-set/substream provenance, typed-vs-missing fold "
              "split, specificity anti-rescue structural + semantic)",
              ok_k1 and ok_k2 and ok_k3 and ok_k4 and ok_k5,
              f"k1={ok_k1} k2={ok_k2} k3={ok_k3} k4={ok_k4} "
              f"k5={ok_k5}")
    except ImportError:
        check("LOCO amendment locking KATs", False, "W2_ENGINE_ABSENT")
    except Exception as exc:
        check("LOCO amendment locking KATs", False,
              f"{type(exc).__name__}: {exc}")


# ---- REV 12: the calendar-authority locking KATs (codex 1400Z r.1) --------
def w_cal():
    """Bar-side calendar locks: (1) THREE-way frame identity -- my own
    in-bar derivation from the ruling text == the engine derivation ==
    the COMMITTED authority artifact; (2) the compression trap
    (compacted registered_days refuses -- availability is a mask,
    never a deletion); (3) 08-28 exclusion + shifted-date refusals;
    (4) the certification path invokes only the pinned NON-cal
    entrypoints."""
    try:
        import inspect
        from datetime import date as _dc, timedelta as _tdc
        import w2_power_harness_cayley as WPH

        def refuses(fn, code):
            try:
                fn()
                return False
            except Exception as exc:
                return code in str(exc)

        def _span(a, b):
            d0, d1 = _dc.fromisoformat(a), _dc.fromisoformat(b)
            return [(d0 + _tdc(days=i)).isoformat()
                    for i in range((d1 - d0).days + 1)]

        # (1) my own derivation from the SUCCESSOR schedule (owner
        # redate to PRESTART 2026-08-28, quote sha c2fdcf76...):
        # cutoff = 08-27 the last complete day, baseline 60 ending at
        # cutoff, 08-28 the excluded PRESTART day, evaluation 132
        # from 08-29
        my_base = _span("2026-06-29", "2026-08-27")
        my_eval = _span("2026-08-29", "2027-01-07")
        my_frame = {"baseline_days": my_base,
                    "excluded_days": ["2026-08-28"],
                    "evaluation_days": my_eval,
                    "engine_days": my_base + my_eval}
        eng_frame = WPH.w2_calendar_frame()
        ok_id = len(my_base) == 60 and len(my_eval) == 132 \
            and len(my_frame["engine_days"]) == 192 \
            and "2026-08-28" not in my_frame["engine_days"] \
            and all(eng_frame[k] == my_frame[k] for k in my_frame)
        # ... == the COMMITTED authority artifact bytes
        with open(os.path.join(
                _REPO, "docs", "f2g_window2_execution",
                "calendar_authority_w2_v3.json"),
                encoding="utf-8") as f:
            auth = json.load(f)
        af = auth.get("frame", auth)
        ok_id = ok_id and all(af[k] == my_frame[k] for k in my_frame) \
            and af.get("b1b") == {"n_blocks": 16, "block_len": 12,
                                  "baseline_positions": 60}

        # (2) the compression trap: a compacted registered_days list
        # refuses CALENDAR_MASK_COMPRESSION; the honest form (full
        # grid + availability mask) passes
        eng = eng_frame["engine_days"]
        compacted = [d for d in eng if d != "2026-07-01"]
        ok_cmp = refuses(
            lambda: WPH._validate_carrier_mask(
                "c1", {"registered_days": compacted,
                       "available_days": compacted}, eng_frame),
            "CALENDAR_MASK_COMPRESSION") \
            and WPH._validate_carrier_mask(
                "c1", {"registered_days": list(eng),
                       "available_days":
                       [d for d in eng if d != "2026-07-01"]},
                eng_frame) is None

        # (3) 08-28 (PRESTART) refusals + shifted/extra authority
        # dates
        ok_x = refuses(
            lambda: WPH._validate_carrier_mask(
                "c1", {"registered_days": list(eng),
                       "available_days": ["2026-08-28"]}, eng_frame),
            "CALENDAR_EXCLUDED_DAY")
        shifted = json.loads(json.dumps(eng_frame))
        shifted["baseline_days"] = ["2026-06-28"] \
            + shifted["baseline_days"][1:]
        ok_x = ok_x and refuses(
            lambda: WPH._validate_calendar_frame(shifted),
            "CALENDAR_AUTHORITY_MISMATCH")
        extra = json.loads(json.dumps(eng_frame))
        extra["engine_days"] = extra["engine_days"] + ["2027-01-08"]
        ok_x = ok_x and refuses(
            lambda: WPH._validate_calendar_frame(extra),
            "CALENDAR_AUTHORITY_MISMATCH")

        # (4) the bound replicate path routes ONLY through the pinned
        # NON-cal entrypoints (no _family_cal call survives)
        src = inspect.getsource(WPH.replicate_pvalues_bound)
        ok_nc = "family_cal" not in src \
            and "b2a_family" in src and "b3a_family" in src

        check("CAL calendar-authority locks (three-way frame identity "
              "vs my own derivation + committed artifact, compression "
              "trap, 08-28 exclusion + shifted/extra-date refusals, "
              "non-cal-only certification path)",
              ok_id and ok_cmp and ok_x and ok_nc,
              f"id={ok_id} cmp={ok_cmp} x={ok_x} nc={ok_nc}")
    except ImportError:
        check("CAL calendar-authority locks", False,
              "W2_ENGINE_ABSENT")
    except Exception as exc:
        check("CAL calendar-authority locks", False,
              f"{type(exc).__name__}: {exc}")


# ---- REV 13: candidate-selector + cert-runner locks (codex 1909Z) ---------
def w_selrun():
    """Bar-side locks over cayley's repair revision: (1) the
    registered two-stage selector vs a full IN-BAR oracle (own sorts,
    engineered tie + stage-2 reordering + B2A select-all + gain
    VALUE-ordering) with digest recomputation; (2) selector quality/
    coverage/stage-2-scope doctors; (3) runner fire-input, points,
    selector-digest, stale-outdir, invocation-write-once and
    manifest-resolution refusals -- the TOCTOU family's typed
    boundaries."""
    try:
        import tempfile
        import w2_tier_selector_cayley as WTS
        import w2_cert_runner_cayley as WCR

        def refuses(fn, code):
            try:
                fn()
                return False
            except Exception as exc:
                return code in str(exc)

        def canon_sha(obj):
            return hashlib.sha256(json.dumps(
                obj, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest()

        def outs(c):
            return [True] * c + [False] * (50 - c)

        # grids: gains placed at indices 0 (gain 10) and 11 (gain 3)
        # to prove VALUE-ordered specificity append, never index-order
        g_b1b = [{"gain": 10.0}] + [{"k": i} for i in range(10)] \
            + [{"gain": 3.0}]
        grids = {"B2A": [{"m": 1}, {"m": 2}, {"m": 3}],
                 "B2B": [{"m": m, "dropout": d}
                         for m, d in ((1, 10), (2, 10), (3, 10),
                                      (1, 25), (2, 25))],
                 "B1B": g_b1b,
                 "B3A": [{"k": i} for i in range(4)]}
        pre = {"B2A": [50, 50, 49],          # tie -> grid-index order
               "B2B": [10, 50, 10, 50, 20],  # tie at 50 -> idx 1, 3
               "B1B": [50] * 8 + [10, 10],   # top-8 = first eight
               "B3A": [1, 2, 3, 4]}
        post_b1b = [10, 20, 30, 40, 50, 45, 35, 25]  # reorders st.2
        fams = {}
        for fam in ("B2A", "B2B", "B1B", "B3A"):
            det = [(i, p) for i, p in enumerate(grids[fam])
                   if "gain" not in p]
            entries = []
            for k, (gi, gp) in enumerate(det):
                e = {"point": gp, "outcomes": outs(pre[fam][k])}
                if fam == "B1B":
                    e["post_loco_outcomes"] = (
                        outs(post_b1b[k]) if k < 8 else None)
                entries.append(e)
            fams[fam] = entries
        smoke = {"quality": {"R": 50, "n_draws": 999},
                 "geometry_capsule_digest": "ab" * 32,
                 "families": fams}
        REF_S = {"commit": "kat-commit", "path": "kat/smoke.json"}
        REF_G = {"commit": "kat-commit", "path": "kat/grids.json"}
        art = WTS.select_candidates(smoke, grids, smoke_ref=REF_S,
                                    effect_grids_ref=REF_G)

        def kat_reader(commit, path):
            if commit != "kat-commit":
                raise WTS.SelectorRefusal(
                    f"SELECTOR_ARTIFACT_INVALID: {path} unreadable "
                    f"at {commit} (uncommitted carriers never bind)")
            if path == "kat/smoke.json":
                return json.dumps(smoke).encode()
            if path == "kat/grids.json":
                return json.dumps({"grids": grids}).encode()
            raise WTS.SelectorRefusal(
                f"SELECTOR_ARTIFACT_INVALID: {path} unreadable")

        # IN-BAR oracle: my own two-stage derivation
        def oracle(fam):
            det = [(i, p) for i, p in enumerate(grids[fam])
                   if "gain" not in p]
            order1 = sorted(range(len(det)), key=lambda k: (
                -pre[fam][k], det[k][0]))
            keep = order1[:min(8, len(det))]
            if fam == "B1B":
                pick = sorted(keep, key=lambda k: (
                    -post_b1b[k], det[k][0]))[:3]
            else:
                pick = keep[:3]
            return ([det[k][0] for k in keep],
                    [det[k][0] for k in pick])
        ok_sel = True
        for fam in ("B2A", "B2B", "B1B", "B3A"):
            t8, sel = oracle(fam)
            ok_sel = ok_sel \
                and art["top8_grid_indices"][fam] == t8 \
                and art["selected_grid_indices"][fam] == sel
        # hand expectations pinned: the engineered outcomes
        ok_sel = ok_sel \
            and art["selected_grid_indices"]["B2A"] == [0, 1, 2] \
            and art["selected_grid_indices"]["B2B"] == [1, 3, 4] \
            and art["selected_grid_indices"]["B1B"] == [5, 6, 4] \
            and art["selected_grid_indices"]["B3A"] == [3, 2, 1]
        ordered = art["ordered_points"]
        ok_ord = len(ordered) == 14 \
            and [o["family"] for o in ordered] == \
            ["B2A"] * 3 + ["B2B"] * 3 + ["B1B"] * 3 + ["B3A"] * 3 \
            + ["B1B"] * 2 \
            and ordered[12]["point"] == {"gain": 3.0} \
            and ordered[13]["point"] == {"gain": 10.0} \
            and ordered[12]["entry"] == "specificity" \
            and art["ordered_points_sha256"] == canon_sha(ordered) \
            and art["tier_s_label"].startswith("PRELIMINARY_SMOKE")
        # determinism
        ok_ord = ok_ord and WTS.select_candidates(
            smoke, grids, smoke_ref=REF_S,
            effect_grids_ref=REF_G)["ordered_points_sha256"] == \
            art["ordered_points_sha256"]

        # selector doctors
        import json as _j
        bad = _j.loads(_j.dumps(smoke))
        bad["families"]["B2A"][0]["outcomes"] = outs(50)[:49]
        ok_doc = refuses(lambda: WTS.select_candidates(bad, grids),
                         "SELECTOR_QUALITY_INVALID")
        bad2 = _j.loads(_j.dumps(smoke))
        bad2["families"]["B2A"][0]["outcomes"] = [1] * 50
        ok_doc = ok_doc and refuses(
            lambda: WTS.select_candidates(bad2, grids),
            "SELECTOR_QUALITY_INVALID")
        bad3 = _j.loads(_j.dumps(smoke))
        bad3["families"]["B1B"][9]["post_loco_outcomes"] = outs(10)
        ok_doc = ok_doc and refuses(
            lambda: WTS.select_candidates(bad3, grids),
            "SELECTOR_STAGE2_INVALID")
        bad4 = _j.loads(_j.dumps(smoke))
        bad4["families"]["B1B"][0]["post_loco_outcomes"] = None
        ok_doc = ok_doc and refuses(
            lambda: WTS.select_candidates(bad4, grids),
            "SELECTOR_STAGE2_INVALID")
        bad5 = _j.loads(_j.dumps(smoke))
        bad5["families"]["B3A"] = list(
            reversed(bad5["families"]["B3A"]))
        ok_doc = ok_doc and refuses(
            lambda: WTS.select_candidates(bad5, grids),
            "SELECTOR_COVERAGE_INVALID")
        bad6 = _j.loads(_j.dumps(smoke))
        bad6["quality"] = {"R": 20, "n_draws": 999}
        ok_doc = ok_doc and refuses(
            lambda: WTS.select_candidates(bad6, grids),
            "SELECTOR_QUALITY_INVALID")

        # runner typed boundaries (the TOCTOU family)
        pts = art["ordered_points"]
        ok_run = True
        for np_bad in (0, -1, "2", len(pts) + 1, True):
            ok_run = ok_run and refuses(
                lambda n=np_bad: WCR._validate_fire_inputs(
                    _REPO, "HEAD", n, pts,
                    os.path.join(tempfile.gettempdir(),
                                 "w2_bar_no_such_dir")),
                "RUNNER_PROCESS_COUNT_INVALID"), np_bad
        stale_dir = tempfile.mkdtemp(prefix="w2_bar_stale_")
        with open(os.path.join(stale_dir, "invocation_record.json"),
                  "w") as f:
            f.write("{}")
        ok_run = ok_run and refuses(
            lambda: WCR._validate_fire_inputs(_REPO, "HEAD", 1, pts,
                                              stale_dir),
            "RUNNER_OUTDIR_STALE")
        ok_run = ok_run and refuses(
            lambda: WCR.resolve_manifest_commit(
                _REPO, "no-such-ref-w2-bar"),
            "RUNNER_MANIFEST_UNRESOLVABLE")
        full = WCR.resolve_manifest_commit(_REPO, "HEAD")
        ok_run = ok_run and len(full) == 40
        dup = pts + [pts[0]]
        ok_run = ok_run and refuses(
            lambda: WCR.validate_points(dup),
            "RUNNER_POINTS_INVALID")
        ok_run = ok_run and refuses(
            lambda: WCR.validate_points(
                [{"family": "B2A", "point": {"m": 1},
                  "entry": "specificity"}]),
            "RUNNER_POINTS_INVALID")
        # --- codex 2235Z item 2 (in-bar per the 0130Z ask): the
        # selector fires ONLY as a verified COMMITTED artifact ---
        # positive: the engine artifact + its bound carriers verify
        ok_run = ok_run and WTS.verify_selector_artifact(
            _REPO, art, blob_reader=kat_reader) is True
        # a fabricated minimal artifact (self-consistent digest, no
        # bindings) refuses -- integrity is not correctness
        fab = {"schema": "f2g-w2-tier-selector-v1",
               "ordered_points": [{"family": "B2A",
                                   "point": {"m": 999},
                                   "entry": "detection"}],
               "ordered_points_sha256": canon_sha(
                   [{"family": "B2A", "point": {"m": 999},
                     "entry": "detection"}])}
        ok_run = ok_run and refuses(
            lambda: WTS.verify_selector_artifact(
                _REPO, fab, blob_reader=kat_reader),
            "SELECTOR_ARTIFACT_INVALID")
        # uncommitted carriers never bind (real git reader, bogus ref)
        art_bogus = _j.loads(_j.dumps(art))
        art_bogus["smoke_ref"] = {"commit": "0" * 40,
                                  "path": "no/such.json"}
        ok_run = ok_run and refuses(
            lambda: WTS.verify_selector_artifact(_REPO, art_bogus),
            "SELECTOR_ARTIFACT_INVALID")
        # altered points/gains diverge from the independent rerun
        art_gain = _j.loads(_j.dumps(art))
        art_gain["ordered_points"][13]["point"] = {"gain": 99.0}
        ok_run = ok_run and refuses(
            lambda: WTS.verify_selector_artifact(
                _REPO, art_gain, blob_reader=kat_reader),
            "SELECTOR_ARTIFACT_INVALID")
        # the runner load path: committed-object semantics via the
        # injected reader; a fabricated artifact refuses at load
        sel_blob = _j.dumps(art).encode()

        def sel_reader(commit, path):
            if (commit, path) == ("kat-commit", "kat/selector.json"):
                return sel_blob
            return kat_reader(commit, path)
        art2, pts2, sel_sha = WCR.load_selector_committed(
            _REPO, "kat-commit", "kat/selector.json",
            blob_reader=sel_reader)
        ok_run = ok_run and pts2 == pts

        def fab_reader(commit, path):
            if (commit, path) == ("kat-commit", "kat/selector.json"):
                return _j.dumps(fab).encode()
            return kat_reader(commit, path)
        ok_run = ok_run and refuses(
            lambda: WCR.load_selector_committed(
                _REPO, "kat-commit", "kat/selector.json",
                blob_reader=fab_reader),
            "RUNNER_SELECTOR_INVALID")

        # --- codex 2235Z item 3 (in-bar): workers authenticate the
        # COMPLETE invocation core, and publication is create-once ---
        inv_dir = tempfile.mkdtemp(prefix="w2_bar_inv_")
        rec_inv = WCR.write_invocation_record(
            inv_dir, pts, full, "geom.json", 2, ["kat"],
            "kat-commit", "kat/selector.json", sel_sha)
        isha = rec_inv["invocation_sha256"]
        inv2, ipts2 = WCR._load_invocation(inv_dir, isha)
        ok_run = ok_run and ipts2 == pts
        ok_run = ok_run and refuses(
            lambda: WCR.write_invocation_record(
                inv_dir, pts, full, "geom.json", 2, ["kat"],
                "kat-commit", "kat/selector.json", sel_sha),
            "RUNNER_PUBLISH_EXISTS")
        # manifest-only and geometry-only post-write mutations refuse
        # at the worker (the points digest alone no longer
        # authenticates)
        inv_path = os.path.join(inv_dir, "invocation_record.json")
        for fld, val in (("manifest_commit", "c" * 40),
                         ("geometry_path", "geometry-B.json")):
            with open(inv_path, encoding="utf-8") as f:
                doct = _j.load(f)
            doct[fld] = val
            with open(inv_path, "w", encoding="utf-8") as f:
                _j.dump(doct, f)
            ok_run = ok_run and refuses(
                lambda: WCR._load_invocation(inv_dir, isha),
                "RUNNER_INVOCATION_DIGEST_MISMATCH"), fld
            with open(inv_path, "w", encoding="utf-8",
                      newline="\n") as f:
                _j.dump(rec_inv, f, indent=1, sort_keys=True)
                f.write("\n")
        inv3, _ = WCR._load_invocation(inv_dir, isha)
        ok_run = ok_run and inv3["invocation_sha256"] == isha
        ok_run = ok_run and refuses(
            lambda: WCR._load_invocation(inv_dir, "0" * 64),
            "RUNNER_INVOCATION_DIGEST_MISMATCH")

        # --- REV 15 doctor 3 (codex 0245Z): three COMMITTED,
        # internally consistent substitute carriers refuse as
        # UNADMITTED -- the artifact verifies (integrity), the
        # manifest simply does not pin these grids (authority)
        def adm_reader(commit, path):
            if path.endswith("execution_manifest.json"):
                return _j.dumps({"slots": {"power_harness": {
                    "pins": [{"path": "docs/other.json",
                              "blob_sha256": "0" * 64,
                              "commit": "b" * 40}]}}}).encode()
            return sel_reader(commit, path)
        ok_run = ok_run and WTS.verify_selector_artifact(
            _REPO, art, blob_reader=adm_reader) is True \
            and refuses(
                lambda: WTS.verify_selector_admission(
                    _REPO, art, "kat-commit",
                    blob_reader=adm_reader,
                    git_resolve=lambda c: "a" * 40),
                "SELECTOR_UNADMITTED")

        # --- REV 17 (codex 0320Z item 2 + 0432Z chain): the ADMITTED
        # grid + fabricated smoke + minimal pre-invocation, MY OWN
        # construction over the FULL chronological chain
        # manifest -> pre-invocation -> results/completion -> smoke ---
        # REV 18: STRICT stage ancestry -- a real staged commit chain
        # MC < PRE_C < RC(results+completion shared) < SC(smoke) with
        # an order-based ancestry helper (strict_edge refuses a == b
        # itself; reflexive chains are structurally dead)
        MC = "a" * 40
        PRE_C = "b" * 40
        RC = "c" * 40
        SC = "d" * 40
        CHAIN = [MC, PRE_C, RC, SC]
        GEOMD = "ab" * 32
        BAR_GEOM = {"capsule_digest": GEOMD,
                    "loco_registry_carrier": "cascadia",
                    "registries": {"cascadia": ["S0", "S1"]},
                    "seed_authority_sha256": "b" * 64}

        def bar_geom_loader(mc, path):
            return BAR_GEOM

        def bar_anc(a, b):
            return a in CHAIN and b in CHAIN and \
                CHAIN.index(a) < CHAIN.index(b)
        grids_raw = _j.dumps({"grids": grids}).encode()
        impl_raw = b"# bar-pinned impl"
        geom_raw = b"{}"
        adm_pins2 = [
            {"path": "kat/grids.json", "commit": MC,
             "blob_sha256": hashlib.sha256(grids_raw).hexdigest()},
            {"path": "kat/impl.py", "commit": MC,
             "blob_sha256": hashlib.sha256(impl_raw).hexdigest()},
            {"path": "kat/geom.json", "commit": MC,
             "blob_sha256": hashlib.sha256(geom_raw).hexdigest()}]
        store2 = {(MC, "kat/grids.json"): grids_raw,
                  (MC, "kat/impl.py"): impl_raw,
                  (MC, "kat/geom.json"): geom_raw}

        def rdr2(commit, path):
            if path.endswith("execution_manifest.json"):
                return _j.dumps({"slots": {"power_harness": {
                    "pins": adm_pins2}}}).encode()
            try:
                return store2[(commit, path)]
            except KeyError:
                raise WTS.SelectorRefusal(
                    f"SELECTOR_UNADMITTED: {path} unreadable at "
                    f"{commit}")
        # derivational results v2 from MY true smoke: per-replicate
        # FOUR-family p-vectors (Holm replays them) + B1B fold p-maps
        # over the bound registry (the substitution rule replays them)
        FAMS4 = ("B1B", "B2A", "B2B", "B3A")
        REG_G = BAR_GEOM["registries"]["cascadia"]
        impl_id = {"commit": MC, "path": "kat/impl.py",
                   "blob_sha256": hashlib.sha256(impl_raw)
                   .hexdigest()}

        def my_reps(outcomes, fam):
            return [{"p_values": {f: (0.001 if (o and f == fam)
                                      else 0.9) for f in FAMS4}}
                    for o in outcomes]

        def my_folds(post):
            if post is None:
                return None
            out = []
            for p in post:
                m = {st: 0.001 for st in REG_G}
                if not p:
                    m[REG_G[0]] = 0.9
                out.append(m)
            return out
        results = {"schema": "f2g-w2-tier-s-results-v2",
                   "quality": {"R": 50, "n_draws": 999},
                   "seed_authority_sha256": "b" * 64,
                   "geometry_capsule_digest": GEOMD,
                   "implementation": dict(impl_id),
                   "families": {}}
        for f in fams:
            det_pts = [p for p in grids[f] if "gain" not in p]
            results["families"][f] = [
                {"point": dict(e["point"]),
                 "grid_index": det_pts.index(e["point"]),
                 "replicates": my_reps(e["outcomes"], f),
                 "loco_folds": (my_folds(e.get("post_loco_outcomes"))
                                if f == "B1B" else None)}
                for e in fams[f]]
        r_raw = _j.dumps(results).encode()
        r_sha = hashlib.sha256(r_raw).hexdigest()
        det_order = {f: [p for p in grids[f] if "gain" not in p]
                     for f in ("B2A", "B2B", "B1B", "B3A")}
        pre = {"schema": "f2g-w2-tier-s-pre-invocation-v1",
               "manifest_commit": MC,
               "effect_grids": {"commit": MC,
                                "path": "kat/grids.json",
                                "blob_sha256": hashlib.sha256(
                                    grids_raw).hexdigest()},
               "effect_grids_content_sha256": canon_sha(grids),
               "geometry": {"commit": MC, "path": "kat/geom.json",
                            "capsule_digest": GEOMD},
               "quality": {"R": 50, "n_draws": 999},
               "seed_authority_sha256": "b" * 64,
               "implementation": dict(impl_id),
               "grid_order_sha256": canon_sha(det_order),
               "output_root": "kat", "argv": ["kat"],
               "host": "kat", "interpreter": {"executable": "kat"},
               "fired_utc": "2026-08-25T00:00:00Z"}
        pre["invocation_sha256"] = canon_sha(
            {k: v for k, v in pre.items()
             if k != "invocation_sha256"})
        store2[(PRE_C, "kat/ts_pre.json")] = _j.dumps(pre).encode()
        comp = {"schema": "f2g-w2-tier-s-completion-v1",
                "pre_invocation_sha256": pre["invocation_sha256"],
                "results_blob_sha256": r_sha,
                "fired_utc": "2026-08-25T00:00:00Z",
                "completed_utc": "2026-08-25T11:00:00Z"}
        store2[(RC, "kat/ts_comp.json")] = _j.dumps(comp).encode()
        store2[(RC, "kat/ts_results.json")] = r_raw
        chain_fields = dict(
            pre_invocation_ref={"commit": PRE_C,
                                "path": "kat/ts_pre.json"},
            pre_invocation_sha256=pre["invocation_sha256"],
            completion_ref={"commit": RC, "path": "kat/ts_comp.json"},
            results_ref={"commit": RC, "path": "kat/ts_results.json",
                         "blob_sha256": r_sha})
        smoke_adm = dict(smoke, schema="f2g-w2-tier-s-smoke-v1",
                         effect_grids_sha256=canon_sha(grids),
                         **chain_fields)
        store2[(SC, "kat/smoke2.json")] = _j.dumps(
            smoke_adm).encode()
        refs_adm = dict(smoke_ref={"commit": SC,
                                   "path": "kat/smoke2.json"},
                        effect_grids_ref={"commit": MC,
                                          "path": "kat/grids.json"})
        art_adm = WTS.select_candidates(smoke_adm, grids, **refs_adm)
        adm_ok = WTS.verify_selector_admission(
            _REPO, art_adm, MC, blob_reader=rdr2,
            git_resolve=lambda c: c,
            geometry_loader=bar_geom_loader, is_ancestor=bar_anc)
        ok_run = ok_run and adm_ok["pre_invocation"][
            "invocation_sha256"] == pre["invocation_sha256"]
        # LOCK A: a minimal self-hashed dict is not the closed
        # pre-invocation capsule
        min_inv = {"schema": "f2g-w2-tier-s-invocation-v1",
                   "purpose": "attests no execution"}
        min_inv["invocation_sha256"] = canon_sha(
            {k: v for k, v in min_inv.items()
             if k != "invocation_sha256"})
        store2[(PRE_C, "kat/min_inv.json")] = _j.dumps(
            min_inv).encode()
        fab_f = {}
        for f, entries in fams.items():
            fab_f[f] = [dict(e, outcomes=[True] * 50)
                        for e in entries]
            if f == "B1B":
                for k, e in enumerate(fab_f[f]):
                    e["post_loco_outcomes"] = ([True] * 50 if k < 8
                                               else None)
        fab_smoke = dict(smoke, families=fab_f,
                         schema="f2g-w2-tier-s-smoke-v1",
                         effect_grids_sha256=canon_sha(grids),
                         **dict(chain_fields,
                                pre_invocation_ref={
                                    "commit": PRE_C,
                                    "path": "kat/min_inv.json"},
                                pre_invocation_sha256=min_inv[
                                    "invocation_sha256"]))
        store2[(SC, "kat/fab_smoke.json")] = _j.dumps(
            fab_smoke).encode()
        art_fab = WTS.select_candidates(
            fab_smoke, grids,
            smoke_ref={"commit": SC, "path": "kat/fab_smoke.json"},
            effect_grids_ref=refs_adm["effect_grids_ref"])
        ok_run = ok_run and refuses(
            lambda: WTS.verify_selector_admission(
                _REPO, art_fab, MC, blob_reader=rdr2,
                git_resolve=lambda c: c,
                geometry_loader=bar_geom_loader,
                is_ancestor=bar_anc),
            "pre-invocation is not")
        # LOCK B: a WELL-FORMED chain whose fabricated smoke does not
        # replay from the results p-vectors refuses at the Holm replay
        fab2 = dict(fab_smoke, **chain_fields)
        store2[(SC, "kat/fab2_smoke.json")] = _j.dumps(fab2).encode()
        art_fab2 = WTS.select_candidates(
            fab2, grids,
            smoke_ref={"commit": SC, "path": "kat/fab2_smoke.json"},
            effect_grids_ref=refs_adm["effect_grids_ref"])
        ok_run = ok_run and refuses(
            lambda: WTS.verify_selector_admission(
                _REPO, art_fab2, MC, blob_reader=rdr2,
                git_resolve=lambda c: c,
                geometry_loader=bar_geom_loader,
                is_ancestor=bar_anc),
            "registered Holm rule")
        # LOCK C (1328Z item 2): STRICT stage ancestry -- the same-
        # commit flat store (the post-hoc combined capsule) refuses,
        # and a reflexive smoke->selector edge refuses
        flat = {}
        for (c, p), v in store2.items():
            flat[(MC, p)] = v

        def flat_reader(commit, path):
            if path.endswith("execution_manifest.json"):
                return rdr2(commit, path)
            try:
                return flat[(MC, path)]
            except KeyError:
                raise WTS.SelectorRefusal(
                    f"SELECTOR_UNADMITTED: {path} unreadable")
        flat_fields = dict(
            pre_invocation_ref={"commit": MC,
                                "path": "kat/ts_pre.json"},
            pre_invocation_sha256=pre["invocation_sha256"],
            completion_ref={"commit": MC, "path": "kat/ts_comp.json"},
            results_ref={"commit": MC, "path": "kat/ts_results.json",
                         "blob_sha256": r_sha})
        flat_smoke = dict(smoke_adm, **flat_fields)
        flat[(MC, "kat/flat_smoke.json")] = _j.dumps(
            flat_smoke).encode()
        art_flat = WTS.select_candidates(
            flat_smoke, grids,
            smoke_ref={"commit": MC, "path": "kat/flat_smoke.json"},
            effect_grids_ref=refs_adm["effect_grids_ref"])
        ok_run = ok_run and refuses(
            lambda: WTS.verify_selector_admission(
                _REPO, art_flat, MC, blob_reader=flat_reader,
                git_resolve=lambda c: c,
                geometry_loader=bar_geom_loader,
                is_ancestor=bar_anc),
            "STRICT stage ancestry")
        ok_run = ok_run and refuses(
            lambda: WTS.verify_selector_admission(
                _REPO, art_adm, MC, blob_reader=rdr2,
                git_resolve=lambda c: c,
                geometry_loader=bar_geom_loader,
                is_ancestor=bar_anc,
                selector_identity={"commit": SC,
                                   "path": "kat/selector.json"}),
            "STRICT stage ancestry")
        # zero-network pre-freeze doctor (0349Z item 1 + 1328Z item
        # 3, cross-authored in-bar): the production capture path
        # refuses a committed-but-UNPINNED authority AND an
        # unregistered key, opener counter provably unmoved
        import w2_acquisition_capture_grassmann as CAPB
        net = {"n": 0}

        def counting_opener(url):
            net["n"] += 1
            raise AssertionError("must never be reached")
        bar_keys = {"SELECTION_RECORDS": {"cascadia": ["2026-08-20"]},
                    "MAG_FEED": {"frn": ["2026-08-20"]},
                    "MAG_WEATHER_FEED": {"mf4drv": ["2026-08-20"]}}
        bar_auth = {"schema": "f2g-w2-expected-contracts-v3",
                    "template_token_vocabulary":
                        ["{day}", "{day_next}"],
                    "prestart_expected_keys": bar_keys,
                    "prestart_expected_keys_sha256": canon_sha(
                        bar_keys),
                    "static_layer": {
                        lane: {"carriers": {ck: {
                            "static_contract_template": {
                                "source": {"kind": "kat",
                                           "ref": "kat://src"},
                                "endpoint": "https://kat.example/x",
                                "request_params": {},
                                "operation_params": {
                                    "carrier": ck, "day": "{day}"}},
                            "cutoff": "2026-08-25"}}}
                        for lane, cks in bar_keys.items()
                        for ck in cks},
                    "dynamic_layer": {}, "digests": {},
                    "provenance": {}}
        bar_auth_raw = _j.dumps(bar_auth).encode()
        bar_pins = {"status": "BOUND", "pins": [
            {"path": "kat/auth.json", "commit": "kat-auth",
             "blob_sha256": hashlib.sha256(bar_auth_raw)
             .hexdigest()}]}
        # freeze finding 1 + end-to-end finding 3: the authority pin
        # is admitted ONLY from a BOUND AUTHORITY_SLOT (accrual_impl)
        bar_man = {"slots": {CAPB.AUTHORITY_SLOT: bar_pins}}
        bar_man_raw = _j.dumps(bar_man).encode()
        wrong_man_raw = _j.dumps(
            {"slots": {"producer_boundary": bar_pins}}).encode()
        open_man_raw = _j.dumps(
            {"slots": {CAPB.AUTHORITY_SLOT:
                       dict(bar_pins, status="OPEN")}}).encode()

        def cap_reader(c, p):
            if p == CAPB.EXEC_MANIFEST_PATH:
                return bar_man_raw
            if p == "kat/auth.json":
                return bar_auth_raw
            raise CAPB.CaptureRefusal(
                f"CAPTURE_AUTHORITY_INVALID: {p} unreadable")

        def cap_call(path="kat/auth.json", day="2026-09-30",
                     reader=None):
            return CAPB.capture_authorized(
                _REPO, "kat-man", path, "SELECTION_RECORDS",
                "cascadia", day, "x", "x", "x", lambda b: {},
                opener=counting_opener, clock=lambda: "x",
                blob_reader=reader or cap_reader,
                git_resolve=lambda c: "b" * 40,
                authority_reproducer=lambda: _j.loads(
                    bar_auth_raw.decode()))
        try:
            cap_call(path="kat/unpinned_auth.json")
            ok_run = False
        except CAPB.CaptureRefusal as exc:
            ok_run = ok_run and "CAPTURE_AUTHORITY_UNADMITTED" in \
                str(exc)
        # a same-path pin in the WRONG slot refuses UNADMITTED
        try:
            cap_call(reader=lambda c, p: wrong_man_raw
                     if p == CAPB.EXEC_MANIFEST_PATH
                     else cap_reader(c, p))
            ok_run = False
        except CAPB.CaptureRefusal as exc:
            ok_run = ok_run and "CAPTURE_AUTHORITY_UNADMITTED" in \
                str(exc)
        # end-to-end finding 3: an OPEN slot STILL carrying the
        # reviewed pin refuses UNADMITTED before any pin read
        try:
            cap_call(reader=lambda c, p: open_man_raw
                     if p == CAPB.EXEC_MANIFEST_PATH
                     else cap_reader(c, p))
            ok_run = False
        except CAPB.CaptureRefusal as exc:
            ok_run = ok_run and "CAPTURE_AUTHORITY_UNADMITTED" in \
                str(exc)
        try:
            cap_call()                    # unregistered day
            ok_run = False
        except CAPB.CaptureRefusal as exc:
            ok_run = ok_run and "CAPTURE_AUTHORITY_INVALID" in \
                str(exc)
        ok_run = ok_run and net["n"] == 0

        check("SELRUN candidate-selector + cert-runner locks "
              "(two-stage selector == in-bar oracle w/ tie + stage-2 "
              "reorder + B2A select-all + gain value-order + digest "
              "determinism, quality/coverage/stage-2 doctors, "
              "fire-input/points/selector-digest/stale-outdir/"
              "write-once/manifest-resolution refusals)",
              ok_sel and ok_ord and ok_doc and ok_run,
              f"sel={ok_sel} ord={ok_ord} doc={ok_doc} run={ok_run}")
    except ImportError:
        check("SELRUN candidate-selector + cert-runner locks", False,
              "W2_ENGINE_ABSENT")
    except Exception as exc:
        check("SELRUN candidate-selector + cert-runner locks", False,
              f"{type(exc).__name__}: {exc}")


# ---- REV 15: the admission-boundary locks (codex 0238Z items 1-2) ---------
def w_admit():
    """The admission consumer's coordinated-substitution boundaries,
    cross-authored: a full six-key positive through the REAL
    verify_staged_boundary (real capture_day carriers, real store,
    real five-map join), then: whole-day omission refuses against the
    INDEPENDENT authority; a coordinated artifact+output_sha256
    forgery refuses at the transform recomputation; the fail-closed
    no-dispatcher default; the wrong-prefix pin."""
    try:
        import tempfile
        import w2_acquisition_capture_grassmann as CAP
        import w2_producer_grassmann as PRODM
        import w2_accrual_instrument_cayley as ACC

        # DISCLOSED TRANSITIONAL SEAM (codex 1304Z bridge finding 4 +
        # 1424Z ruling 4). The v4 split retires MF4_FEED, which named
        # two carrier spaces at once, into MAG_WEATHER_FEED +
        # MF4_MONITOR_FEED with NO compatibility alias. Until the
        # registered PRESTART_LANES constant is rebased onto the v4
        # vocabulary, this group cannot construct a valid authority:
        # the lane the instrument REQUIRES is one production now
        # REFUSES. That is the cascade cayley flagged in advance, not
        # a defect in either surface -- so it is reported RED with an
        # exact reason rather than softened into a pass.
        _retired = [ln for ln in ACC.PRESTART_LANES
                    if ln not in PRODM.RECORD_LANES]
        if _retired:
            check("ADMIT admission-boundary locks",
                  False,
                  "DISCLOSED SEAM (not a defect): registered "
                  f"PRESTART_LANES still names {_retired}, retired by "
                  "the v4 lane split; the boundary requires the "
                  "authority lane set to EQUAL PRESTART_LANES while "
                  "production refuses the retired name. Owner: "
                  "cayley's v4 rebase (PRESTART_LANES -> "
                  "MAG_WEATHER_FEED + MF4_MONITOR_FEED). This group "
                  "goes green on that rebase with no change here.")
            return

        def canon(o):
            return PRODM._canon_digest(o)

        root = tempfile.mkdtemp(prefix="w2_bar_admit_")
        store = os.path.join(root, "store")
        rdir = os.path.join(root, "records")
        tdir = os.path.join(root, "transcripts")
        # the fixture lane set is DERIVED from the registered
        # PRESTART_LANES constant, so this group tracks the v3 -> v4
        # rename (MF4_FEED split into MAG_WEATHER_FEED +
        # MF4_MONITOR_FEED, codex 1304Z bridge finding 4) across the
        # cross-module seam instead of pinning either vocabulary
        _CARRIER = {"SELECTION_RECORDS": "cascadia",
                    "MAG_FEED": "frn"}
        LANES = tuple((ln, _CARRIER.get(ln, "katc"))
                      for ln in ACC.PRESTART_LANES)
        DAYS = ("2026-08-20", "2026-08-21")
        bodies_by_url = {}

        def opener(url):
            return (200, {"content-type": "text/plain"},
                    bodies_by_url[url], url)

        def clock():
            return "2026-08-20T12:00:00Z"

        def disp(lane, raw_body, static_contract):
            return {"n_bytes": len(raw_body), "lane": lane}

        pins = []
        blobmap = {}
        entries = {}
        held = {}
        for lane, ck in LANES:
            for day in DAYS:
                body = f"{lane}|{ck}|{day}".encode()
                ep = (f"https://kat.example/"
                      f"{lane.lower()}/{ck}/{day}")
                bodies_by_url[ep] = body
                spec = {"lane": lane, "carrier": ck, "utc_day": day,
                        "endpoint": ep, "request_params": {},
                        "source": {"kind": "kat",
                                   "ref": "synthetic://fixture"},
                        "cutoff": "2026-08-25",
                        "operation_params": {"carrier": ck,
                                             "day": day},
                        "expected_keys": [day]}
                _, _, rec, tr = CAP.capture_day(
                    spec, store, rdir, tdir,
                    lambda b, L=lane: {"n_bytes": len(b), "lane": L},
                    opener=opener, clock=clock)
                s = CAP.static_contract_of(spec)
                art_j = {"n_bytes": len(body), "lane": lane}
                stem = f"{lane.lower()}_{ck}_{day}"
                for cls, obj in (("record", rec), ("transcript", tr),
                                 ("contract", s),
                                 ("artifact", art_j)):
                    path = ACC.STAGED_PREFIX + stem + \
                        ACC.STAGED_CLASS_SUFFIX[cls]
                    raw = (json.dumps(obj, indent=1, sort_keys=True)
                           + "\n").encode()
                    blobmap[("kat", path)] = raw
                    pins.append({"commit": "kat", "path": path,
                                 "blob_sha256": hashlib.sha256(raw)
                                 .hexdigest()})
                entries[f"{lane}/{ck}/{day}"] = {
                    "sha256": rec["raw_body_sha256"],
                    "bytes": rec["raw_body_bytes"]}
                held[(lane, ck, day)] = (spec, body, tr)
        inv = CAP.build_staged_body_inventory("s4t-kat", "kat://s",
                                              entries)
        desc = {"schema": CAP.STORE_DESCRIPTOR_SCHEMA,
                "store_id": "s4t-kat", "store_root": "kat://s",
                "physical_root": store}
        auth_keys = {lane: {ck: list(DAYS)}
                     for lane, ck in LANES}

        def keys_sha(k):
            return hashlib.sha256(json.dumps(
                k, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest()

        def mk_template(lane, ck):
            return {"source": {"kind": "kat",
                               "ref": "synthetic://fixture"},
                    "endpoint": (f"https://kat.example/"
                                 f"{lane.lower()}/{ck}/{{day}}"),
                    "request_params": {},
                    "operation_params": {"carrier": ck,
                                         "day": "{day}"}}
        # REV 16: the CLOSED self-verifying authority capsule (codex
        # 0320Z item 3) with the per-key static templates (item 1)
        auth = {"schema": "f2g-w2-expected-contracts-v3",
                "template_token_vocabulary": ["{day}", "{day_next}", "{day_compact}"],
                "prestart_expected_keys": auth_keys,
                "prestart_expected_keys_sha256": keys_sha(auth_keys),
                "static_layer": {
                    lane: {"carriers": {ck: {
                        "static_contract_template":
                            mk_template(lane, ck),
                        "cutoff": "2026-08-25"}}}
                    for lane, ck in LANES},
                "dynamic_layer": {}, "digests": {},
                "provenance": {"generator": "bar-kat"}}
        for basename, obj in (
                (ACC.STORE_DESCRIPTOR_BASENAME, desc),
                (ACC.STAGED_INVENTORY_BASENAME, inv),
                (ACC.EXPECTED_KEYS_BASENAME, auth)):
            path = ACC.STAGED_PREFIX + basename
            raw = (json.dumps(obj, indent=1, sort_keys=True)
                   + "\n").encode()
            blobmap[("kat", path)] = raw
            pins.append({"commit": "kat", "path": path,
                         "blob_sha256": hashlib.sha256(raw)
                         .hexdigest()})

        def man_of(pin_list):
            return {"slots": {"producer_boundary": {
                "status": "BOUND", "pins": pin_list}}}

        def reader(commit, path):
            return blobmap[(commit, path)]

        def boundary(pin_list, dispatcher=disp, reproducer=None):
            return ACC.verify_staged_boundary(
                _REPO, man_of(pin_list), blob_reader=reader,
                transform_dispatcher=dispatcher,
                authority_reproducer=(
                    reproducer or (lambda: json.loads(
                        json.dumps(auth)))))

        def refuses_detail(fn, needle):
            try:
                fn()
                return False
            except Exception as exc:
                return "PRESTART_ADMISSION_REFUSED" in str(exc) \
                    and needle in str(exc)

        # POSITIVE: the full six-key boundary through everything real
        res = boundary(pins)
        ok_pos = isinstance(res, dict) \
            and set(res["report"]["lanes"]) == {
                f"{ln}/{ck}" for ln, ck in LANES} \
            and all(v["days"] == 2
                    for v in res["report"]["lanes"].values()) \
            and len(res["staged_boundary_sha256"]) == 64

        # doctor 1: a WHOLE authorized day consistently omitted from
        # every class refuses against the independent authority
        drop_stem = "selection_records_cascadia_2026-08-21"
        omit = [p for p in pins if drop_stem not in p["path"]]
        ok_omit = refuses_detail(lambda: boundary(omit),
                                 "omission never shrinks")

        # doctor 2: coordinated artifact + output_sha256 forgery --
        # E agrees with the forged artifact, the S/T/E join would
        # pass, and the TRANSFORM RECOMPUTATION refuses
        spec_f, body_f, tr_f = held[("MAG_FEED", "frn", DAYS[0])]
        forged_art = {"n_bytes": 999, "fabricated": True}
        forged_rec = PRODM.build_envelope_record(
            lane="MAG_FEED", carrier="frn", utc_day=DAYS[0],
            raw_body=body_f,
            source={"kind": "kat", "ref": "synthetic://fixture"},
            endpoint=spec_f["endpoint"], request_params={},
            transcript=tr_f, cutoff="2026-08-25",
            operation_params=spec_f["operation_params"],
            expected_keys=[DAYS[0]], artifact=forged_art)
        forged_pins = []
        for p in pins:
            q = dict(p)
            stem_f = f"mag_feed_frn_{DAYS[0]}"
            if q["path"] == ACC.STAGED_PREFIX + stem_f + \
                    ".artifact.json":
                raw = (json.dumps(forged_art, indent=1,
                                  sort_keys=True) + "\n").encode()
                blobmap[("katf", q["path"])] = raw
                q.update(commit="katf",
                         blob_sha256=hashlib.sha256(raw).hexdigest())
            elif q["path"] == ACC.STAGED_PREFIX + stem_f + \
                    ".record.json":
                raw = (json.dumps(forged_rec, indent=1,
                                  sort_keys=True) + "\n").encode()
                blobmap[("katf", q["path"])] = raw
                q.update(commit="katf",
                         blob_sha256=hashlib.sha256(raw).hexdigest())
            forged_pins.append(q)
        ok_forge = refuses_detail(lambda: boundary(forged_pins),
                                  "never derivation")

        # fail-closed (freeze finding 3 successor form): the
        # boundary's DEFAULT dispatcher is now the REGISTERED
        # production transform, and the synthetic 'kat' source kind
        # is unregistered there -- the default path refuses typed
        # instead of admitting digest-only. (Before the transform
        # landed, the same call refused 'never admitted digest-only';
        # both spellings prove no digest-only admission exists.)
        ok_fc = False
        try:
            boundary(pins, None)
        except Exception as exc:
            ok_fc = ("ADMISSION_TRANSFORM_REFUSED" in str(exc)
                     and "unregistered" in str(exc)) \
                or "never admitted digest-only" in str(exc)

        # wrong prefix: a staged-class basename outside the exact
        # staged_envelopes prefix refuses (never enters or vanishes)
        stray = dict(pins[0])
        stray["path"] = "docs/elsewhere/" + os.path.basename(
            pins[0]["path"])
        ok_pre = refuses_detail(
            lambda: boundary(pins + [stray]),
            "outside the exact prefix")

        # --- REV 16 (codex 0320Z): the authority-capsule locks + the
        # coordinated evil-endpoint reproduction ---
        def with_auth(a2, tag):
            raw = (json.dumps(a2, indent=1, sort_keys=True)
                   + "\n").encode()
            path = ACC.STAGED_PREFIX + ACC.EXPECTED_KEYS_BASENAME
            blobmap[(tag, path)] = raw
            out = []
            for p in pins:
                q = dict(p)
                if q["path"] == path:
                    q["commit"] = tag
                    q["blob_sha256"] = hashlib.sha256(raw).hexdigest()
                out.append(q)
            return out

        def mut_auth(**over):
            a2 = json.loads(json.dumps(auth))
            a2.update(over)
            return a2
        # forged digest text merely repeated -> refuses
        a_forged = mut_auth(
            prestart_expected_keys_sha256="attested-key-digest")
        ok_auth = refuses_detail(
            lambda: boundary(with_auth(a_forged, "k1"),
                             reproducer=lambda: a_forged),
            "does not recompute")
        # empty carrier map for a named lane
        ek = json.loads(json.dumps(auth_keys))
        ek[LANES[-1][0]] = {}
        a_empty = mut_auth(prestart_expected_keys=ek,
                           prestart_expected_keys_sha256=keys_sha(ek))
        ok_auth = ok_auth and refuses_detail(
            lambda: boundary(with_auth(a_empty, "k2"),
                             reproducer=lambda: a_empty),
            "carrier map is empty")
        # duplicate day
        dk = json.loads(json.dumps(auth_keys))
        dk["MAG_FEED"]["frn"] = [DAYS[0], DAYS[0]]
        a_dup = mut_auth(prestart_expected_keys=dk,
                         prestart_expected_keys_sha256=keys_sha(dk))
        ok_auth = ok_auth and refuses_detail(
            lambda: boundary(with_auth(a_dup, "k3"),
                             reproducer=lambda: a_dup),
            "not unique ascending")
        # shifted / non-canonical day spelling (ascending-preserving
        # so the ordering check does not fire first)
        sk = json.loads(json.dumps(auth_keys))
        sk["MAG_FEED"]["frn"] = [DAYS[0], "2026-08-3"]
        a_shift = mut_auth(prestart_expected_keys=sk,
                           prestart_expected_keys_sha256=keys_sha(sk))
        ok_auth = ok_auth and refuses_detail(
            lambda: boundary(with_auth(a_shift, "k4"),
                             reproducer=lambda: a_shift),
            "non-canonical day")
        # an OPEN token in a CONSUMED template refuses (the two-phase
        # v3 freeze precedes any capture)
        a_open = json.loads(json.dumps(auth))
        a_open["static_layer"]["MAG_FEED"]["carriers"]["frn"][
            "static_contract_template"]["endpoint"] = \
            "OPEN_REVIEW_ROUND"
        ok_auth = ok_auth and refuses_detail(
            lambda: boundary(with_auth(a_open, "k5"),
                             reproducer=lambda: a_open),
            "OPEN tokens")
        # reproduction failure: the pinned artifact must REPRODUCE
        # from the registered generator
        a_neq = mut_auth(provenance={"generator": "someone-else"})
        ok_auth = ok_auth and refuses_detail(
            lambda: boundary(pins,
                             reproducer=lambda: a_neq),
            "does not REPRODUCE")

        # the coordinated EVIL-ENDPOINT reproduction (codex item 1):
        # genuine bytes, internally consistent S/T/E + artifact built
        # against an unregistered endpoint -- refuses at the
        # S-admission equality, never reaches the join
        e_lane, e_ck, e_day = "MAG_FEED", "frn", DAYS[0]
        e_body = f"{e_lane}|{e_ck}|{e_day}".encode()
        e_ep = "https://evil.example/data"
        bodies_by_url[e_ep] = e_body
        e_spec = {"lane": e_lane, "carrier": e_ck, "utc_day": e_day,
                  "endpoint": e_ep, "request_params": {},
                  "source": {"kind": "kat",
                             "ref": "synthetic://fixture"},
                  "cutoff": "2026-08-25",
                  "operation_params": {"carrier": e_ck,
                                       "day": e_day},
                  "expected_keys": [e_day]}
        _, _, e_rec, e_tr = CAP.capture_day(
            e_spec, store, os.path.join(root, "evil_r"),
            os.path.join(root, "evil_t"),
            lambda b, L=e_lane: {"n_bytes": len(b), "lane": L},
            opener=opener, clock=clock)
        e_s = CAP.static_contract_of(e_spec)
        e_art = {"n_bytes": len(e_body), "lane": e_lane}
        e_stem = f"{e_lane.lower()}_{e_ck}_{e_day}"
        evil_pins = []
        for p in pins:
            q = dict(p)
            for cls, obj in (("record", e_rec), ("transcript", e_tr),
                             ("contract", e_s), ("artifact", e_art)):
                want_p = ACC.STAGED_PREFIX + e_stem + \
                    ACC.STAGED_CLASS_SUFFIX[cls]
                if q["path"] == want_p:
                    raw = (json.dumps(obj, indent=1, sort_keys=True)
                           + "\n").encode()
                    blobmap[("kate", want_p)] = raw
                    q.update(commit="kate",
                             blob_sha256=hashlib.sha256(raw)
                             .hexdigest())
            evil_pins.append(q)
        ok_evil = refuses_detail(
            lambda: boundary(evil_pins),
            "S is admitted, never submitted")

        check("ADMIT admission-boundary locks (six-key REAL positive "
              "through capture_day + store + five-map join, "
              "whole-day-omission vs the independent authority, "
              "coordinated artifact+digest forgery vs transform "
              "recomputation, fail-closed no-dispatcher, "
              "wrong-prefix pin, closed authority capsule locks, "
              "evil-endpoint S-admission)",
              ok_pos and ok_omit and ok_forge and ok_fc and ok_pre
              and ok_auth and ok_evil,
              f"pos={ok_pos} omit={ok_omit} forge={ok_forge} "
              f"fc={ok_fc} prefix={ok_pre} auth={ok_auth} "
              f"evil={ok_evil}")
    except ImportError:
        check("ADMIT admission-boundary locks", False,
              "W2_ENGINE_ABSENT")
    except Exception as exc:
        check("ADMIT admission-boundary locks", False,
              f"{type(exc).__name__}: {exc}")


# ---- REV 20: the registered admission-transform + canonical-URL
# locks (codex freeze-review findings 2+3) ----
def w_xform():
    """Freeze findings 2+3, cross-authored in-bar: (2) the canonical
    URL builder reproduces the probe-confirmed OMNIWeb repeated-
    parameter grammar EXACTLY and can never emit the stringified
    container; the frozen socal template derives the attempt-4
    requested query verbatim (parse-equality against the pinned
    envelope). (3) the REGISTERED production admission transform
    recomputes SoCal attempt-4's REAL committed 59-row body to
    exactly the 12 registered stations -- the 47 outside-bbox rows
    provably cannot enter -- plus the registered-set narrowing
    doctor."""
    try:
        import w2_producer_grassmann as PRODX
        import w2_acquisition_capture_grassmann as CAPX
        from urllib.parse import urlsplit, parse_qs

        # (2) EXACT OMNI URL: repeated keys in registered order,
        # sorted key spelling, never the stringified list
        url = PRODX.requested_url_of(
            "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi",
            {"activity": "retrieve", "res": "min",
             "spacecraft": "omni_min", "start_date": "20251115",
             "end_date": "20251115", "vars": ["17", "21", "25"]})
        ok_url = url == (
            "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi?"
            "activity=retrieve&end_date=20251115&res=min&"
            "spacecraft=omni_min&start_date=20251115&"
            "vars=17&vars=21&vars=25")
        ok_url = ok_url and "%5B" not in url and "%27" not in url \
            and parse_qs(urlsplit(url).query)["vars"] == \
            ["17", "21", "25"]

        def refuses_p(fn):
            try:
                fn()
                return False
            except PRODX.ProducerRefusal as e:
                return "PRODUCER_URL_PARAM_INVALID" in str(e)
        ok_neg = all(refuses_p(
            lambda b=b: PRODX.requested_url_of(
                "https://e.example/x", b))
            for b in ({"vars": []}, {"vars": [["17"]]},
                      {"vars": {"17": 1}}))

        # (3) the FROZEN socal template + the REAL attempt-4 body
        auth_p = os.path.join(_REPO, "docs", "f2g_window2_execution",
                              "staged_expected_contracts_v3.json")
        with open(auth_p, encoding="utf-8") as f:
            authx = json.load(f)
        csoc = authx["static_layer"]["SELECTION_RECORDS"][
            "carriers"]["socal_coachella"]
        tpl = csoc["static_contract_template"]

        def sub(v):
            if isinstance(v, str):
                return (v.replace("{day_next}", "2025-11-16")
                        .replace("{day}", "2025-11-15")
                        .replace("{day_compact}", "20251115"))
            if isinstance(v, dict):
                return {k: sub(x) for k, x in v.items()}
            if isinstance(v, list):
                return [sub(x) for x in v]
            return v

        def s_soc(op_override=None):
            return {"lane": "SELECTION_RECORDS",
                    "carrier": "socal_coachella",
                    "utc_day": "2025-11-15",
                    "endpoint": tpl["endpoint"],
                    "request_params": sub(tpl["request_params"]),
                    "source": dict(tpl["source"]),
                    "cutoff": csoc["cutoff"],
                    "operation_params": (
                        op_override
                        or sub(tpl["operation_params"])),
                    "expected_keys": ["2025-11-15"]}
        ev_dir = os.path.join(_REPO, "docs", "f2g_window2_execution",
                              "probe_evidence")
        with open(os.path.join(
                ev_dir, "socal_coachella_attempt4.body"), "rb") as f:
            raw4 = f.read()
        with open(os.path.join(
                ev_dir, "socal_coachella_attempt4.envelope.json"),
                encoding="utf-8") as f:
            env4 = json.load(f)
        # the frozen template derives the attempt-4 query verbatim
        # (parse-equality; the envelope preserved fire-time order)
        durl = PRODX.requested_url_of(tpl["endpoint"],
                                      sub(tpl["request_params"]))
        ok_derive = (urlsplit(durl)[:3] ==
                     urlsplit(env4["requested_url"])[:3]
                     and parse_qs(urlsplit(durl).query) ==
                     parse_qs(urlsplit(
                         env4["requested_url"]).query))
        regs = sorted(sub(tpl["operation_params"])[
            "registered_station_filter"].split(","))
        art = CAPX.admission_transform("SELECTION_RECORDS", raw4,
                                       s_soc())
        ok_soc = (art["present_stations"] == regs
                  and art["absent_stations"] == []
                  and art["registered_stations"] == regs
                  and art["data_rows"] == 59
                  and art["outside_station_rows_excluded"] == 47
                  and set(art["present_stations"]) <= set(regs))
        # narrowing doctor: dropping one station from the registered
        # set removes EXACTLY it -- outside rows can never leak in
        op_n = sub(tpl["operation_params"])
        op_n["registered_station_filter"] = ",".join(
            s for s in regs if s != "ACP")
        art_n = CAPX.admission_transform("SELECTION_RECORDS", raw4,
                                         s_soc(op_n))
        ok_soc = ok_soc and art_n["present_stations"] == \
            [s for s in regs if s != "ACP"] \
            and art_n["outside_station_rows_excluded"] == 48

        # end-to-end finding 1: the REAL frozen cascadia receipt
        # swept across ALL 90 selection days -- the registered
        # transform must equal w2_cascadia.registry_for_day EXACTLY
        # (one semantics, no fork); 2026-07-14 locked at the frozen
        # registry's 169 NET.STA identities (the day whose nine
        # later-that-day starters exposed the fork)
        import w2_cascadia as CASC
        import subprocess
        from datetime import date as _date, timedelta as _td
        casc_raw = subprocess.run(
            ["git", "-C", _REPO, "cat-file", "blob",
             f"{CASC.MANIFEST_COMMIT}:{CASC.RECEIPT_PATH}"],
            capture_output=True).stdout
        cc = authx["static_layer"]["SELECTION_RECORDS"][
            "carriers"]["cascadia"]
        ctpl = cc["static_contract_template"]
        days90 = authx["prestart_expected_keys"][
            "SELECTION_RECORDS"]["cascadia"]
        ok_casc = len(days90) == 90 and len(casc_raw) > 0
        for d in days90:
            dn = (_date.fromisoformat(d) + _td(days=1)).isoformat()

            def subd(v):
                return (v.replace("{day_next}", dn)
                        .replace("{day}", d)) \
                    if isinstance(v, str) else v
            sc = {"lane": "SELECTION_RECORDS",
                  "carrier": "cascadia", "utc_day": d,
                  "endpoint": ctpl["endpoint"],
                  "request_params": {k: subd(v) for k, v in
                                     ctpl["request_params"].items()},
                  "source": dict(ctpl["source"]),
                  "cutoff": cc["cutoff"],
                  "operation_params": {k: subd(v) for k, v in
                                       ctpl["operation_params"]
                                       .items()},
                  "expected_keys": [d]}
            art_cd = CAPX.admission_transform("SELECTION_RECORDS",
                                              casc_raw, sc)
            frozen = sorted(r["id"] for r in
                            CASC.registry_for_day(d, repo=_REPO))
            if art_cd["present_stations"] != frozen:
                ok_casc = False
                break
            if d == "2026-07-14" and len(frozen) != 169:
                ok_casc = False
                break

        check("XFORM registered admission transform + canonical URL "
              "(exact OMNI repeated-parameter URL + stringification "
              "negatives, frozen-socal-template derivation "
              "parse-equality vs the pinned attempt-4 envelope, REAL "
              "59-row body -> 12/12 registered w/ 47 outside "
              "excluded, registered-set narrowing doctor, REAL "
              "cascadia receipt 90-day sweep == frozen "
              "registry_for_day incl the 169-identity 2026-07-14 "
              "lock)",
              ok_url and ok_neg and ok_derive and ok_soc and ok_casc,
              f"url={ok_url} neg={ok_neg} derive={ok_derive} "
              f"soc={ok_soc} casc={ok_casc}")
    except ImportError:
        check("XFORM registered admission transform + canonical URL",
              False, "W2_ENGINE_ABSENT")
    except Exception as exc:
        check("XFORM registered admission transform + canonical URL",
              False, f"{type(exc).__name__}: {exc}")


_GATED = ()


def main():
    w_pin()
    w_cas_a()
    w_sel_a()
    w_sel_b()
    w_cas_b()
    w_b2b()
    w_barrier()
    w_b1b()
    w_mf4()
    w_mag()
    w_mag_exec()
    w_mag_b()
    w_mag_null()
    w_loco()
    w_cal()
    w_selrun()
    w_admit()
    w_xform()


main()
print()
if FAILS:
    print(f"WINDOW-2 RED-KAT FAILURES ({len(FAILS)}): "
          f"{[f.split(' ')[0] for f in FAILS]}")
    sys.exit(1)
print("ALL WINDOW-2 RED-KATs PASS")
