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
                    if k not in ("remote_lease", "lane_uuids")} | {
                "remote_lease": lease,
                "lane_uuids": ["seismic", "mf4", "mag1"]}

        def fresh():
            led = WBAR.BarrierLedger()
            led.prestart(bindings(), "2026-08-25")
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
        R.append(expect(lambda: WBAR.BarrierLedger().prestart(
            {k: v for k, v in bindings().items()
             if k != "owner_authorization"}, "2026-08-25"),
            "MISSING_LANE_AUTHORIZATION"))
        led4 = fresh()
        led4.close_support_barrier("LEASE-1", "2027-01-12", "non_analyst")
        R.append(expect(lambda: led4.final_fire("LEASE-1", "mf4", "r"),
                        "VALUE_FIRE_SEAL_MISSING"))      # unsealed fire
        R.append(expect(lambda: fresh().add_lane("LEASE-1", "extra"),
                        "LATE_LANE_ADDITION"))
        R.append(expect(lambda: WBAR.BarrierLedger(
            used_leases=("LEASE-1",)).prestart(bindings(), "2026-08-25"),
            "REUSED_GLOBAL_LEASE"))
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


# ---- engine-gated classes: typed red until cayley's surfaces land ----------
_GATED = (
    "B1B annex KATs (endpoint invariance, 8/(max(2,3)) fixture x4 paths, "
    "ZERO_SCALE_REFUSAL never-shrink, winsor 4-leg identity, gain-step "
    "specificity, health admission)",
    "MF4 annex KATs (label maturity byte-lock, zero-class/no-drop, "
    "immutable rows, persistence baseline, block constants)",
    "MAG annex KATs (apply-never-refit, VIC XYZS/S-exclusion + 4 frame "
    "refusals, SOS byte equality, MAG-UNTESTABLE, 3-primary Holm)",
)


def main():
    w_pin()
    w_cas_a()
    w_sel_a()
    w_sel_b()
    w_cas_b()
    w_b2b()
    w_barrier()
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
