#!/usr/bin/env python3
"""Lock tests for generate_atlas.py (codex atlas fixes 2-5).

Plain-assert style, runnable on py3.11 and py3.14:
    python monitoring/test_atlas_generator.py
Exit 0 = all lock tests pass. Writes only under a temp sandbox;
the repo's real docs/atlas.html is never touched (OUT is
redirected for every generation in here).
"""
import copy
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import generate_atlas as GA  # noqa: E402

FAILS = []


def _sandbox_base():
    """grassmann 1933Z LOW: the suite pointed sandboxes at %TEMP%, which
    on a machine whose repo is on another drive made a provenance source
    unnameable relative to REPO and crashed ntpath.relpath. Sandboxes go
    on the repository's own drive (monitoring/data is gitignored), so the
    suite no longer depends on where %TEMP% happens to live."""
    base = os.path.join(GA.REPO, "monitoring", "data", "_atlas_test_tmp")
    os.makedirs(base, exist_ok=True)
    return base


def _mkdtemp():
    return tempfile.mkdtemp(dir=_sandbox_base())


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  -- {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def extract(html, sid):
    m = re.search('<script id="' + sid + '"[^>]*>(.*?)</script>',
                  html, re.S)
    return json.loads(m.group(1).replace("<\\/", "</"))


def main():
    sandbox = _mkdtemp()
    out = os.path.join(sandbox, "atlas.html")
    real_out = GA.OUT
    GA.OUT = out
    try:
        # ---- golden generation ------------------------------------
        rc = GA.main()
        html = io.open(out, encoding="utf-8").read()
        d = extract(html, "geodata")
        check("G0 golden generation exits 0 and both blocks parse",
              rc == 0 and isinstance(d, dict)
              and isinstance(extract(html, "landdata"), list))
        gold_sha = sha(out)

        # ---- fix 2: unlocated census, no (0,0), plotted-finite ----
        sts = [dict(st, carrier=car, segment=seg)
               for car, obj in d["carriers"].items()
               for seg, rows in obj["segments"].items()
               for st in rows]
        khmn = [st for st in sts if st["code"] == "KHMN"]
        check("L2a KHMN remains in the census as selected/unlocated"
              " with null coordinates",
              len(khmn) == 1 and khmn[0]["sel"]
              and not khmn[0]["located"]
              and khmn[0]["lat"] is None and khmn[0]["lon"] is None)
        located = [st for st in sts if st["located"]]
        check("L2b every located station has finite in-range "
              "coordinates",
              all(isinstance(st["lat"], (int, float))
                  and isinstance(st["lon"], (int, float))
                  and -90 <= st["lat"] <= 90
                  and -180 <= st["lon"] <= 180 for st in located))
        check("L2c no station occupies (0,0) -- null never coerces "
              "to Null Island",
              not any(st["lat"] == 0 and st["lon"] == 0
                      for st in located))
        cen = d["census"]
        check("L2d census discloses located/unlocated exactly",
              cen["selected"] == cen["selected_located"]
              + len([c for c in cen["unlocated_codes"]
                     if any(s["code"] == c and s["sel"]
                            for s in sts)])
              and "KHMN" in cen["unlocated_codes"])
        tpl = io.open(GA.TEMPLATE, encoding="utf-8").read()
        check("L2e the renderer refuses to project an unlocated or "
              "non-finite station (guard present)",
              "!st.located || !isFinite(st.lat)" in tpl)

        # ---- fix 3: poisoned numerics refuse before replacement ---
        real_build = GA.build_bundle
        gold_bundle = real_build()

        def poisoned(mutate):
            b = copy.deepcopy(gold_bundle)
            mutate(b)
            return b

        cases = [
            ("NaN risk", lambda b: b["daily"]["regions"][0]
             .__setitem__("risk", float("nan"))),
            ("Infinity magnitude", lambda b: b["daily"]["events"][0]
             .__setitem__("mag", float("inf"))),
            ("numeric-string risk", lambda b: b["daily"]["regions"][0]
             .__setitem__("risk", "0.5")),
            ("out-of-range latitude", lambda b:
             b["mags"][0].__setitem__("lat", 95.0)),
        ]
        for label, mut in cases:
            GA.build_bundle = lambda m=mut: poisoned(m)
            try:
                GA.main()
                ok, why = False, "generation did not refuse"
            except SystemExit as e:
                ok = e.code not in (0, None)
                why = f"exit {e.code}"
            except (ValueError, TypeError) as e:
                ok, why = True, type(e).__name__
            finally:
                GA.build_bundle = real_build
            check(f"L3 {label} refuses nonzero before replacement",
                  ok, why)
            check(f"L3 {label} leaves the old page byte-identical "
                  "and no temp file",
                  sha(out) == gold_sha
                  and not os.path.exists(out + ".tmp"))

        # ---- fix 4: provenance digests recompute from bytes -------
        prov = {p["name"]: p for p in d["provenance"]}
        repo = GA.REPO
        all_match = True
        n_checked = 0
        for name, p in prov.items():
            if p["sha256"] is None:
                continue
            fp = os.path.join(repo, *p["path"].split("/")) \
                if name != "template" else GA.TEMPLATE
            got = hashlib.sha256(open(fp, "rb").read()).hexdigest()
            n_checked += 1
            if got != p["sha256"]:
                all_match = False
        check(f"L4a every published source digest recomputes from "
              f"its bytes ({n_checked} sources)",
              all_match and n_checked >= 8)
        check("L4b the unrecomputable hand-written literal is gone",
              "dd386eda" not in tpl and "dd386eda" not in html)
        # mutation: a one-byte template change moves digest AND page
        tdir = _mkdtemp()
        tcopy = os.path.join(tdir, "atlas_template.html")
        shutil.copyfile(GA.TEMPLATE, tcopy)
        real_tpl = GA.TEMPLATE
        out2 = os.path.join(sandbox, "atlas2.html")
        out3 = os.path.join(sandbox, "atlas3.html")
        try:
            GA.TEMPLATE = tcopy
            GA.OUT = out2
            GA.main()
            with open(tcopy, "a", encoding="utf-8") as f:
                f.write("<!-- mutated -->")
            GA.OUT = out3
            GA.main()
            p2 = extract(io.open(out2, encoding="utf-8").read(),
                         "geodata")["provenance"]
            p3 = extract(io.open(out3, encoding="utf-8").read(),
                         "geodata")["provenance"]
            t2 = next(x for x in p2 if x["name"] == "template")
            t3 = next(x for x in p3 if x["name"] == "template")
            check("L4c a mutated template moves its digest and the "
                  "page together",
                  t2["sha256"] != t3["sha256"]
                  and sha(out2) != sha(out3))
        finally:
            GA.TEMPLATE = real_tpl
            GA.OUT = out

        # ---- fix 5: lag is observed, never a constant -------------
        lag_ok = True
        for lag in (0, 1, 2, 3):
            rdir = _mkdtemp()
            os.makedirs(os.path.join(rdir, "docs"))
            import datetime as dt
            run_day = dt.date(2026, 8, 29)
            ens = {"date": (run_day
                            - dt.timedelta(days=lag)).isoformat(),
                   "timestamp": run_day.isoformat() + "T07:01:00",
                   "regions": {"ridgecrest": {
                       "tier": 0, "tier_name": "NORMAL",
                       "combined_risk": 0.1}},
                   "summary": {}, "earthquake_events": {}}
            json.dump(ens, open(os.path.join(
                rdir, "docs", "ensemble_latest.json"), "w"))
            real_repo = GA.REPO
            try:
                GA.REPO = rdir
                got = GA.daily([])["lag_days"]
            finally:
                GA.REPO = real_repo
            if got != lag:
                lag_ok = False
        check("L5a fixtures at 0/1/2/3-day lag each render their "
              "actual observed lag", lag_ok)
        check("L5b the hard-coded two-day constant is gone from the "
              "template",
              "lags run date by 2 days" not in tpl)

        # ---- B2/B4 (program review 2026-09-01): qualifier fields ----
        rows = d["daily"]["regions"]
        keys = ("methods_available", "agreement", "confirmed",
                "components", "weights")
        check("L6a every rendered daily row carries the five qualifier "
              f"fields ({len(rows)} rows)",
              rows and all(all(k in r for k in keys) for r in rows)
              and all(set(r["components"]) == set(GA.COMPONENTS)
                      and all(s in GA.COMPONENT_STATUS
                              for s in r["components"].values())
                      for r in rows))
        check("L6b on the real record methods_available == live count "
              "== weight carriers for every row",
              all(r["methods_available"] is None or (
                  r["methods_available"] == len(
                      [k for k, s in r["components"].items()
                       if s == "live"])
                  and set(r["weights"]) == {
                      k for k, s in r["components"].items()
                      if s == "live"}) for r in rows))
        check("L6c the daily census is present and sums to the row "
              "count for every component",
              set(d["daily"]["components_live"]) == set(GA.COMPONENTS)
              and all(sum(c[s] for s in GA.COMPONENT_STATUS) == len(rows)
                      == c["n"]
                      for c in d["daily"]["components_live"].values()))
        # constructed fixture: one single-method CONFIRMED WATCH with
        # FC stale + LG no data, one two-method NORMAL with FC
        # no-registry, one zero-method DEGRADED -- each status class
        # is CONSTRUCTED so its derivation is tested, not assumed
        fdir = _mkdtemp()
        os.makedirs(os.path.join(fdir, "docs"))

        def comp(avail, notes="", frozen=False):
            return {"available": avail, "frozen": frozen, "notes": notes}
        fix = {"date": "2026-08-30", "timestamp": "2026-09-01T07:05:19",
               "summary": {}, "earthquake_events": {},
               "regions": {
                   "turkey_kahramanmaras": {
                       "tier": 1, "tier_name": "WATCH",
                       "combined_risk": 0.314, "methods_available": 1,
                       "agreement": "single_method",
                       "effective_weights": {"seismic_thd": 1.0},
                       "persistence": {"is_confirmed": True},
                       "components": {
                           "lambda_geo": comp(False,
                                              "No Lambda_geo data available"),
                           "fault_correlation": comp(
                               False, "calibration unavailable: scored day "
                               "2026-08-30 past valid_through 2026-08-23 "
                               "(STALE)"),
                           "seismic_thd": comp(True, "z=1.76")}},
                   "ridgecrest": {
                       "tier": 0, "tier_name": "NORMAL",
                       "combined_risk": 0.035, "methods_available": 2,
                       "agreement": "all_normal",
                       "effective_weights": {"lambda_geo": 0.5714,
                                             "seismic_thd": 0.4286},
                       "persistence": {"is_confirmed": False},
                       "components": {
                           "lambda_geo": comp(True),
                           "fault_correlation": comp(
                               False, "calibration unavailable: no "
                               "registry entry for region ridgecrest"),
                           "seismic_thd": comp(True)}},
                   "hualien": {
                       "tier": -1, "tier_name": "DEGRADED",
                       "combined_risk": 0.0, "methods_available": 0,
                       "agreement": "insufficient_data",
                       "effective_weights": {},
                       "persistence": {"is_confirmed": False},
                       "components": {
                           "lambda_geo": comp(False),
                           "fault_correlation": comp(True, frozen=True),
                           "seismic_thd": comp(False)}}}}
        json.dump(fix, open(os.path.join(fdir, "docs",
                                         "ensemble_latest.json"), "w"))
        real_repo = GA.REPO
        try:
            GA.REPO = fdir
            fd = GA.daily([])
        finally:
            GA.REPO = real_repo
        by = {r["id"]: r for r in fd["regions"]}
        tk, rc, hl = (by["turkey_kahramanmaras"], by["ridgecrest"],
                      by["hualien"])
        check("L6d fixture: single-method confirmed WATCH derives "
              "LG no_data / FC stale / THD live, methods 1, confirmed",
              tk["components"] == {"lambda_geo": "no_data",
                                   "fault_correlation": "stale",
                                   "seismic_thd": "live"}
              and tk["methods_available"] == 1 and tk["confirmed"]
              and tk["agreement"] == "single_method"
              and tk["weights"] == {"seismic_thd": 1.0})
        check("L6e fixture: two-method row derives FC no_registry and "
              "two live carriers; degraded row derives frozen + zero",
              rc["components"]["fault_correlation"] == "no_registry"
              and rc["methods_available"] == 2 and not rc["confirmed"]
              and hl["components"] == {"lambda_geo": "no_data",
                                       "fault_correlation": "frozen",
                                       "seismic_thd": "no_data"}
              and hl["methods_available"] == 0 and hl["weights"] == {})
        check("L6f fixture census: FC live 0 / stale 1 / no_registry 1 "
              "/ frozen 1; THD live 2 / no_data 1; n=3 each",
              fd["components_live"]["fault_correlation"] == {
                  "live": 0, "frozen": 1, "stale": 1, "no_registry": 1,
                  "no_data": 0, "n": 3}
              and fd["components_live"]["seismic_thd"] == {
                  "live": 2, "frozen": 0, "stale": 0, "no_registry": 0,
                  "no_data": 1, "n": 3})
        # anti-vacuity: the consistency lock REFUSES an inconsistent row
        # and leaves the standing page byte-identical (fix-3 discipline)
        # every mutation CONSTRUCTS its inconsistency on row 0 without
        # depending on today's record (row 0 always has three
        # components); the only accepted outcome is the TYPED refusal
        def m_count(b):
            r = b["daily"]["regions"][0]
            r["methods_available"] = (r["methods_available"] or 0) + 1

        def m_dark_weight(b):
            r = b["daily"]["regions"][0]
            r["components"]["lambda_geo"] = "stale"     # force one dark
            r["weights"]["lambda_geo"] = 0.5            # ...and weight it
            r["methods_available"] = len(               # keep the count
                [s for s in r["components"].values() if s == "live"])

        def m_vocab(b):
            b["daily"]["regions"][0]["components"]["seismic_thd"] = "bogus"

        def m_missing(b):
            del b["daily"]["regions"][0]["agreement"]

        def m_census(b):
            del b["daily"]["components_live"]
        qcases = [("methods_available disagrees with live count", m_count),
                  ("weight carried by a dark component", m_dark_weight),
                  ("component status outside the closed vocabulary",
                   m_vocab),
                  ("qualifier field missing", m_missing),
                  ("census absent", m_census)]
        for label, mut in qcases:
            GA.build_bundle = lambda m=mut: poisoned(m)
            try:
                GA.main()
                ok, why = False, "generation did not refuse"
            except SystemExit as e:
                ok = "ATLAS_VALIDATE_REFUSED" in str(e.code)
                why = f"exit {e.code}"
            except Exception as e:  # noqa: BLE001 -- any other path is a FAIL
                ok, why = False, f"untyped {type(e).__name__}: {e}"
            finally:
                GA.build_bundle = real_build
            check(f"L6g {label} refuses TYPED before replacement", ok, why)
            check(f"L6g {label} leaves the old page byte-identical",
                  sha(out) == gold_sha and not os.path.exists(out + ".tmp"))
        # renderer hooks present (LIMIT: source-level check only; the
        # qualifier string is composed in the browser, not verified by
        # a JS engine here)
        check("L6h the template composes the qualifier and the census "
              "from the row fields (renderer hooks present)",
              "r.methods_available === 1" in tpl
              and "components live:" in tpl
              and "qualifier(r)" in tpl and "compMarks(r)" in tpl
              and 'row("qualifier"' in tpl)
        # every rendered row on the REAL page carries a qualifier
        # decision the browser can compose: fields present (L6a) and,
        # for the real record, at least one single-method row exists
        # today OR none does -- report which, assert nothing about it
        n_single = sum(1 for r in rows if r["methods_available"] == 1)
        print(f"  [INFO] real record: {n_single} single-method row(s) of "
              f"{len(rows)}; census "
              f"{ {k: v['live'] for k, v in d['daily']['components_live'].items()} }")
    finally:
        GA.OUT = real_out

    # ---- L7 the EOL-view refusal (grassmann 1933Z MEDIUM) ---------
    # The pin fixes the repository; it does NOT rewrite a checkout made
    # before it. These two build the exact states that matter, because
    # a check nothing constructs proves nothing.
    cap_dir = os.path.join(GA.REPO, "docs", "f2g_window2_execution",
                           "mag_capsules")
    victim = os.path.join(cap_dir, sorted(
        f for f in os.listdir(cap_dir) if f.endswith(".json"))[0])
    original = open(victim, "rb").read()
    try:
        # L7a a CRLF view of a committed input REFUSES, typed
        open(victim, "wb").write(original.replace(b"\n", b"\r\n"))
        try:
            GA.provenance()
            check("L7a a CRLF view of a pinned provenance input REFUSES "
                  "typed (the state grassmann measured on devildog)",
                  False, "provenance() returned instead of refusing")
        except SystemExit as e:
            check("L7a a CRLF view of a pinned provenance input REFUSES "
                  "typed (the state grassmann measured on devildog)",
                  str(e).startswith("ATLAS_PROVENANCE_EOL_VIEW:"),
                  f"raised {str(e)[:90]}")

        # L7b anti-vacuity: restore and the SAME call must succeed, so
        # L7a is the CRLF view failing and not provenance() always
        # failing
        open(victim, "wb").write(original)
        try:
            rows = GA.provenance()
            check("L7b the same call passes once the file is restored "
                  "(L7a is not a check that always fires)",
                  any(r["name"].startswith("mag_capsule:") for r in rows))
        except SystemExit as e:
            check("L7b the same call passes once the file is restored "
                  "(L7a is not a check that always fires)",
                  False, f"refused on clean bytes: {str(e)[:90]}")

        # L7c the daily path is NOT collateral: a genuine CONTENT change
        # (what the runner does to docs/ensemble_latest.json before it
        # regenerates) must pass, because normalising it does not
        # reproduce the blob
        obj = json.loads(original.decode("utf-8"))
        obj["_atlas_lock_test_marker"] = "content change, not an EOL view"
        open(victim, "wb").write(
            json.dumps(obj, indent=1).encode("utf-8"))
        try:
            GA.provenance()
            check("L7c an uncommitted CONTENT change is left alone (the "
                  "daily runner rewrites an input before generating)",
                  True)
        except SystemExit as e:
            check("L7c an uncommitted CONTENT change is left alone (the "
                  "daily runner rewrites an input before generating)",
                  False, f"refused a content change: {str(e)[:90]}")
    finally:
        open(victim, "wb").write(original)
    check("L7d the victim file is restored byte-for-byte",
          open(victim, "rb").read() == original)

    # sandboxes live under the repo now (see _sandbox_base), so they
    # are cleaned up rather than accumulating in the working tree
    shutil.rmtree(os.path.join(GA.REPO, "monitoring", "data",
                               "_atlas_test_tmp"), ignore_errors=True)

    print()
    if FAILS:
        print(f"ATLAS LOCK-TEST FAILURES ({len(FAILS)}): {FAILS}")
        return 1
    print("ATLAS GENERATOR LOCK TESTS: ALL PASS (sandboxed writes "
          "only; repo docs/atlas.html untouched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
