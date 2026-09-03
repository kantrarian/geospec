#!/usr/bin/env python3
"""docs/ PUBLIC-SURFACE BAR (cayley, 2026-09-01) -- grassmann program
review 2053Z, items B1 / B2 / B4 and cayley finding C.

`docs/` is the GitHub Pages root of a PUBLIC repository: every path
under it is served, and every tracked path anywhere in the repository
is world-readable. This bar reads the tree AT A COMMIT (never the
working tree -- `git ls-tree` / `git show <rev>:<path>`) and locks:

  S1  no path under docs/ has a component containing "private"
      (a name that promises non-publication on a published surface is
      a false invariant whatever the bytes are)
  S2  docs/atlas.html embeds a geodata block whose EVERY daily region
      row carries the qualifier fields (methods_available, agreement,
      confirmed, components, weights) with the closed component-status
      vocabulary, plus the components_live census -- a tier is never
      published without how many methods carried it
  S3  docs/index.html's CSV-only fallback is MARKED (`fallback: true`)
      and its construction carries none of the previously synthesised
      literals (confidence 0.5 / methods_available 1 /
      agreement 'single_method' / amp_72h / 20).
      LIMIT: S3 is a MARKER check on source text, not a semantic proof
      that nothing is fabricated; it defends the specific defect found.

Exit 0 GREEN, 1 RED (each failing lock printed), 2 REFUSED (revision
unresolvable / files absent). `--selftest` constructs temp repositories
that violate each lock and one that satisfies all three, so every RED
path is exercised by construction.

Usage:  python monitoring/docs_surface_bar_cayley.py <repo> [rev]
        python monitoring/docs_surface_bar_cayley.py --selftest
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

COMPONENTS = ("lambda_geo", "fault_correlation", "seismic_thd")
COMPONENT_STATUS = ("live", "frozen", "stale", "no_registry", "no_data")
QUALIFIER_KEYS = ("methods_available", "agreement", "confirmed",
                  "components", "weights")
FALLBACK_FORBIDDEN = ("confidence: 0.5", "methods_available: 1",
                      "agreement: 'single_method'", "amp_72h) / 20")


class Refused(Exception):
    pass


def _git(repo, *args, binary=False):
    try:
        out = subprocess.check_output(["git", "-C", repo] + list(args),
                                      stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise Refused(f"git {' '.join(args)}: "
                      f"{e.output.decode('utf-8', 'replace').strip()}")
    return out if binary else out.decode("utf-8", "replace")


def resolve(repo, rev):
    out = _git(repo, "rev-parse", "--verify", "--quiet", rev + "^{commit}")
    sha = out.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise Refused(f"REVISION_UNRESOLVABLE: {rev!r}")
    return sha


def tree_paths(repo, sha, prefix):
    out = _git(repo, "ls-tree", "-r", "--name-only", sha, "--", prefix)
    return [p for p in out.splitlines() if p.strip()]


def blob(repo, sha, path):
    try:
        return _git(repo, "show", f"{sha}:{path}")
    except Refused:
        return None


def s1_no_private_under_docs(paths):
    bad = [p for p in paths
           if any("private" in comp.lower() for comp in p.split("/"))]
    return (not bad), bad


def _geodata(html):
    m = re.search(r'<script id="geodata"[^>]*>(.*?)</script>', html, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1).replace("<\\/", "</"))
    except ValueError:
        return None


def s2_atlas_rows_qualified(html):
    if html is None:
        return False, ["docs/atlas.html absent"]
    d = _geodata(html)
    if not isinstance(d, dict):
        return False, ["geodata block absent or unparsable"]
    daily = d.get("daily") or {}
    rows = daily.get("regions")
    if not isinstance(rows, list) or not rows:
        return False, ["daily.regions absent or empty"]
    bad = []
    for r in rows:
        rid = r.get("id", "?")
        for k in QUALIFIER_KEYS:
            if k not in r:
                bad.append(f"{rid}: lacks {k}")
        comps = r.get("components")
        if isinstance(comps, dict):
            if set(comps) != set(COMPONENTS):
                bad.append(f"{rid}: components keys {sorted(comps)}")
            for k, s in comps.items():
                if s not in COMPONENT_STATUS:
                    bad.append(f"{rid}: component {k} status {s!r}")
        elif "components" in r:
            bad.append(f"{rid}: components not a dict")
    cl = daily.get("components_live")
    if not isinstance(cl, dict) or set(cl) != set(COMPONENTS):
        bad.append("components_live census absent or mis-keyed")
    return (not bad), bad


def s3_index_fallback_marked(html):
    if html is None:
        return False, ["docs/index.html absent"]
    m = re.search(r"const fakeEnsemble = \{(.*?)\n\s*\};", html, re.S)
    if not m:
        return False, ["fakeEnsemble construction not found"]
    body = m.group(1)
    bad = []
    if "fallback: true" not in body:
        bad.append("fallback marker absent from fakeEnsemble")
    for tok in FALLBACK_FORBIDDEN:
        if tok in body:
            bad.append(f"synthesised literal present: {tok}")
    return (not bad), bad


def run(repo, rev="HEAD", quiet=False):
    sha = resolve(repo, rev)
    paths = tree_paths(repo, sha, "docs")
    if not paths:
        raise Refused("docs/ absent at " + sha)
    results = [
        ("S1 no 'private' path under docs/",
         *s1_no_private_under_docs(paths)),
        ("S2 every atlas daily row carries the qualifier fields + census",
         *s2_atlas_rows_qualified(blob(repo, sha, "docs/atlas.html"))),
        ("S3 index.html CSV fallback marked, synthesised literals absent "
         "(MARKER CHECK)",
         *s3_index_fallback_marked(blob(repo, sha, "docs/index.html"))),
    ]
    red = [r for r in results if not r[1]]
    if not quiet:
        print(f"docs-surface bar at {sha} ({len(paths)} paths under docs/)")
        for name, ok, detail in results:
            print(f"  [{'GREEN' if ok else 'RED'}] {name}")
            if not ok:
                for line in detail[:12]:
                    print(f"         - {line}")
                if len(detail) > 12:
                    print(f"         - ... {len(detail) - 12} more")
        print("DOCS_SURFACE_BAR:", "GREEN" if not red else
              f"RED ({len(red)} lock(s))")
    return 0 if not red else 1


# ---------------------------------------------------------------- selftest
def _mk_repo(files):
    d = tempfile.mkdtemp(prefix="dsb_")
    subprocess.check_call(["git", "init", "-q", d])
    for rel, content in files.items():
        p = os.path.join(d, *rel.split("/"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
    subprocess.check_call(["git", "-C", d, "add", "-A"])
    subprocess.check_call(["git", "-C", d, "-c", "user.name=dsb",
                           "-c", "user.email=dsb@selftest", "commit",
                           "-q", "-m", "fixture"])
    return d


def _atlas(rows, census=True):
    daily = {"regions": rows}
    if census:
        daily["components_live"] = {k: {"live": 1, "n": 1}
                                    for k in COMPONENTS}
    return ('<html><script id="geodata" type="application/json">'
            + json.dumps({"daily": daily}) + "</script></html>\n")


GOOD_ROW = {"id": "r", "tier": 1, "methods_available": 1,
            "agreement": "single_method", "confirmed": True,
            "components": {"lambda_geo": "no_data",
                           "fault_correlation": "stale",
                           "seismic_thd": "live"},
            "weights": {"seismic_thd": 1.0}}
GOOD_INDEX = ("<script>\n const fakeEnsemble = {\n   fallback: true,\n"
              "   regions: {}\n };\n</script>\n")
BAD_INDEX = ("<script>\n const fakeEnsemble = {\n   regions: { x: {"
             "\n     combined_risk: parseFloat(d.amp_72h) / 20 || 0,\n"
             "     confidence: 0.5,\n     methods_available: 1,\n"
             "     agreement: 'single_method' } }\n };\n</script>\n")


def selftest():
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f"  -- {detail}" if detail and not ok else ""))
        if not ok:
            fails.append(name)
    good = _mk_repo({"docs/atlas.html": _atlas([GOOD_ROW]),
                     "docs/index.html": GOOD_INDEX,
                     "docs/data.csv": "region,tier\n"})
    check("T1 fully satisfying fixture is GREEN",
          run(good, quiet=True) == 0)
    bad_all = _mk_repo({"docs/geo2graph_map_private/x.json": "{}\n",
                        "docs/atlas.html": _atlas([{"id": "r", "tier": 1}]),
                        "docs/index.html": BAD_INDEX})
    check("T2 fixture violating all three locks is RED",
          run(bad_all, quiet=True) == 1)
    check("T2a S1 names the private path",
          s1_no_private_under_docs(
              ["docs/geo2graph_map_private/x.json", "docs/ok.html"])
          == (False, ["docs/geo2graph_map_private/x.json"]))
    check("T2b S1 is case-insensitive and matches directory components",
          not s1_no_private_under_docs(["docs/Private_Notes/a.md"])[0]
          and s1_no_private_under_docs(["docs/privately_ok.md"])[0] is False
          and s1_no_private_under_docs(["docs/public/a.md"])[0])
    ok, det = s2_atlas_rows_qualified(_atlas([{"id": "r", "tier": 1}]))
    check("T2c S2 lists every missing qualifier key",
          not ok and sum(1 for x in det if "lacks" in x) == 5)
    ok, det = s2_atlas_rows_qualified(_atlas(
        [dict(GOOD_ROW, components=dict(GOOD_ROW["components"],
                                        seismic_thd="bogus"))]))
    check("T2d S2 refuses a status outside the closed vocabulary",
          not ok and any("bogus" in x for x in det))
    ok, det = s2_atlas_rows_qualified(_atlas([GOOD_ROW], census=False))
    check("T2e S2 refuses a page without the components_live census",
          not ok and any("census" in x for x in det))
    check("T2f S2 refuses a page with no geodata block",
          not s2_atlas_rows_qualified("<html></html>")[0])
    ok, det = s3_index_fallback_marked(BAD_INDEX)
    check("T2g S3 names the marker AND all four synthesised literals",
          not ok and len(det) == 5)
    check("T2h S3 accepts the marked, literal-free fallback",
          s3_index_fallback_marked(GOOD_INDEX)[0])
    check("T2i S3 refuses a page with no fakeEnsemble at all",
          not s3_index_fallback_marked("<html></html>")[0])
    one = _mk_repo({"docs/atlas.html": _atlas([GOOD_ROW]),
                    "docs/index.html": GOOD_INDEX,
                    "docs/sub/PRIVATE.json": "{}\n"})
    check("T3 a single S1 violation alone is RED",
          run(one, quiet=True) == 1)
    try:
        run(good, "deadbeefdeadbeef", quiet=True)
        check("T4 unresolvable revision REFUSES (exit 2 path)", False,
              "did not refuse")
    except Refused as e:
        check("T4 unresolvable revision REFUSES (exit 2 path)",
              "REVISION_UNRESOLVABLE" in str(e) or "rev-parse" in str(e))
    nodocs = _mk_repo({"README.md": "x\n"})
    try:
        run(nodocs, quiet=True)
        check("T5 repository without docs/ REFUSES", False, "did not refuse")
    except Refused:
        check("T5 repository without docs/ REFUSES", True)
    # the bar reads the COMMIT, not the working tree: dirty a good repo
    with open(os.path.join(good, "docs", "index.html"), "w",
              encoding="utf-8") as f:
        f.write(BAD_INDEX)
    check("T6 a dirty working tree does not change the verdict at HEAD",
          run(good, quiet=True) == 0)
    for d in (good, bad_all, one, nodocs):
        shutil.rmtree(d, ignore_errors=True)
    print()
    if fails:
        print(f"DOCS_SURFACE_BAR SELFTEST FAILURES ({len(fails)}): {fails}")
        return 1
    print("DOCS_SURFACE_BAR SELFTEST: ALL PASS")
    return 0


def main(argv):
    if argv[1:2] == ["--selftest"]:
        return selftest()
    if not argv[1:]:
        print(__doc__)
        return 2
    repo = argv[1]
    rev = argv[2] if len(argv) > 2 else "HEAD"
    try:
        return run(repo, rev)
    except Refused as e:
        print(f"DOCS_SURFACE_BAR: REFUSED -- {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
