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
    sandbox = tempfile.mkdtemp()
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
        tdir = tempfile.mkdtemp()
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
            rdir = tempfile.mkdtemp()
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
    finally:
        GA.OUT = real_out
    print()
    if FAILS:
        print(f"ATLAS LOCK-TEST FAILURES ({len(FAILS)}): {FAILS}")
        return 1
    print("ATLAS GENERATOR LOCK TESTS: ALL PASS (sandboxed writes "
          "only; repo docs/atlas.html untouched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
