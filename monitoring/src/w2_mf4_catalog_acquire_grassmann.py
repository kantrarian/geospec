#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MF4 late-calibration-catalog acquisition (grassmann) -- the ONE
receipted ComCat snapshot under the codex 2026-08-29T17:58Z pinned
contract, the 1901Z pre-HTTP repair ruling, and the append-only repair
amendment (+ corrections 1-2)
docs/f2g_window2_execution/amendment_mf4_late_catalog_repair_20260829.md.
Lane AMENDED_AFTER_FREEZE (disclosed late repair, never the original
untouched preregistration).

Contract (codex 1758Z, verbatim bounds):
- provider: USGS ComCat FDSN event query; one query per each of the 13
  registered polygon-union bboxes (FAULT_SEGMENTS vertices; alias
  socal_saf_coachella->socal_coachella; tokyo_kanto typed-excluded);
- origin superset [2025-10-11T00:00:00Z, 2026-08-28T00:00:00Z), local
  admitted filter identical; minmagnitude=4.0; format=geojson;
  orderby=time-asc; limit=20000 (count==limit refuses);
- kept fields: id, exact UTC origin time (ms), lat, lon, magnitude.

Codex 1901Z repairs implemented here:
1. --fire requires --fire-authorization <json> (schema
   geospec-mf4-fire-authorization-v1) binding the exact public
   amendment/correction commit, module Git-blob sha256 (LF bytes),
   module runtime sha256, query-contract digest, the codex-pass inbox
   commit, the owner fire-go quote/time/scope, and
   output_target_must_be_absent. The module recomputes its own runtime
   bytes AND their LF-normalized (Git-blob) form and refuses
   MF4_FIRE_AUTH_* on any mismatch BEFORE any HTTP. The authority file
   + digest bind into every receipt. A self-reported hash is never the
   pin.
2. Transactional staging: all local invariants + authority preflighted
   before HTTP; a unique exclusive staging directory (refusing
   links/reparse escapes and pre-existing targets); every response's
   raw bytes + attempt metadata sealed with EXCLUSIVE creation BEFORE
   parsing; any failure writes a terminal typed refusal manifest
   binding every attempt and preserves staging evidence; the success
   path atomically publishes the whole directory; a second fire
   refuses target/staging reuse (continuation = new owner decision,
   reusing sealed bytes, never re-querying).
3. Closed parser: strict JSON (non-finite constants refused);
   top-level FeatureCollection; integer non-bool metadata.count ==
   len(features); each item a Feature with nonempty string id and
   Point geometry; finite numeric non-bool lat/lon/mag; integer
   non-bool epoch ms; mag >= 4.0; registered bbox/time membership;
   nondecreasing (time_ms, id) order; allowed JSON/GeoJSON content
   type; effective URL must equal the requested URL (redirects
   refuse).
4. Every attempt receipted (requested + effective URL, status,
   response headers, exception/refusal code, byte length/digest,
   parser result). After all validations: ONE canonical global event
   table sorted by (time_ms, id), identical cross-region duplicates
   deduplicated, region-membership retained as a side channel, table
   digest bound for the downstream adapter.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
REPO = os.path.dirname(os.path.dirname(_HERE))

PROVIDER_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
T_START = "2025-10-11T00:00:00"       # superset start (UTC, inclusive)
T_END = "2026-08-28T00:00:00"         # superset end (UTC, exclusive)
MINMAG = 4.0
LIMIT = 20000
ALLOWED_CONTENT_TYPES = ("application/json", "application/geo+json",
                         "text/json")
ALIAS = {"socal_saf_coachella": "socal_coachella"}
TYPED_EXCLUSIONS = {"tokyo_kanto": "MF4_BBOX_UNREGISTERED"}
ADMITTED = [
    "anchorage", "campi_flegrei", "cascadia", "hualien",
    "istanbul_marmara", "kaikoura", "kumamoto", "mexico_guerrero",
    "norcal_hayward", "ridgecrest", "socal_saf_coachella",
    "socal_saf_mojave", "turkey_kahramanmaras",
]
PINNED_BBOXES = {
    "anchorage":            {"min_lat": 60.0,  "max_lat": 62.0,  "min_lon": -151.0, "max_lon": -148.0},
    "campi_flegrei":        {"min_lat": 40.8,  "max_lat": 40.85, "min_lon": 14.05,  "max_lon": 14.2},
    "cascadia":             {"min_lat": 45.0,  "max_lat": 51.0,  "min_lon": -128.0, "max_lon": -121.5},
    "hualien":              {"min_lat": 22.0,  "max_lat": 25.5,  "min_lon": 120.0,  "max_lon": 122.5},
    "istanbul_marmara":     {"min_lat": 40.3,  "max_lat": 41.1,  "min_lon": 27.0,   "max_lon": 31.0},
    "kaikoura":             {"min_lat": -43.0, "max_lat": -41.5, "min_lon": 171.5,  "max_lon": 174.5},
    "kumamoto":             {"min_lat": 32.0,  "max_lat": 34.0,  "min_lon": 130.0,  "max_lon": 132.0},
    "mexico_guerrero":      {"min_lat": 15.5,  "max_lat": 18.5,  "min_lon": -101.0, "max_lon": -97.0},
    "norcal_hayward":       {"min_lat": 37.0,  "max_lat": 38.5,  "min_lon": -123.0, "max_lon": -121.2},
    "ridgecrest":           {"min_lat": 35.3,  "max_lat": 36.0,  "min_lon": -117.9, "max_lon": -117.45},
    "socal_saf_coachella":  {"min_lat": 32.8,  "max_lat": 34.0,  "min_lon": -116.8, "max_lon": -115.2},
    "socal_saf_mojave":     {"min_lat": 33.8,  "max_lat": 36.0,  "min_lon": -118.5, "max_lon": -116.0},
    "turkey_kahramanmaras": {"min_lat": 36.0,  "max_lat": 39.0,  "min_lon": 35.0,   "max_lon": 40.0},
}
FINAL_DIR = os.path.join(REPO, "docs", "f2g_window2_execution",
                         "mf4_catalog_snapshot")
COMPLETENESS_POLICY = ("ComCat as-is at snapshot time; per-region "
                       "completeness caveat disclosed; no post-snapshot "
                       "revision enters this calibration snapshot")
AUTH_SCHEMA = "geospec-mf4-fire-authorization-v3"
SCOPE_LITERAL = ("MF4 late-calibration-catalog acquisition: exactly one "
                 "fire of the 13 registered ComCat queries under "
                 "amendment_mf4_late_catalog_repair_20260829 and its "
                 "corrections; nothing else")
FRAMEWORK_REPO = r"C:\agent-framework"
MODULE_REL = "monitoring/src/w2_mf4_catalog_acquire_grassmann.py"
CORRECTION3_REL = ("docs/f2g_window2_execution/"
                   "amendment_mf4_late_catalog_repair_20260829_"
                   "correction4.md")


class Refusal(SystemExit):
    def __init__(self, code, detail):
        self.code = code
        super().__init__(f"REFUSED {code}: {detail}")


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _utcnow():
    return dt.datetime.now(dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ")


def _strict_loads(raw):
    def _no(const):
        raise Refusal("MF4_CATALOG_NONFINITE_JSON", const)
    return json.loads(raw.decode("utf-8"), parse_constant=_no)


def module_identities():
    """Runtime bytes + their LF-normalized (Git-blob-frame) form,
    bound SEPARATELY (codex fix 1: the checkout frame may differ)."""
    raw = open(os.path.abspath(__file__), "rb").read()
    lf = raw.replace(b"\r\n", b"\n")
    return {"path": "monitoring/src/w2_mf4_catalog_acquire_grassmann.py",
            "runtime_sha256": _sha(raw), "runtime_bytes": len(raw),
            "git_blob_sha256": _sha(lf), "git_blob_bytes": len(lf)}


def query_contract():
    c = {"provider": PROVIDER_URL, "t_start": T_START, "t_end": T_END,
         "minmagnitude": MINMAG, "limit": LIMIT,
         "orderby": "time-asc", "format": "geojson",
         "admitted": ADMITTED, "alias": ALIAS,
         "typed_exclusions": TYPED_EXCLUSIONS,
         "pinned_bboxes": PINNED_BBOXES,
         "completeness_policy": COMPLETENESS_POLICY}
    canon = json.dumps(c, sort_keys=True).encode("utf-8")
    return c, _sha(canon)


def _git(repo, *args):
    r = subprocess.run(["git", "-C", repo] + list(args),
                       capture_output=True)
    if r.returncode != 0:
        raise Refusal("MF4_FIRE_AUTH_GIT",
                      f"git {' '.join(args)}: "
                      f"{r.stderr.decode(errors='replace')[:200]}")
    return r.stdout


def _is_ancestor(repo, anc, desc):
    r = subprocess.run(["git", "-C", repo, "merge-base",
                        "--is-ancestor", anc, desc],
                       capture_output=True)
    return r.returncode == 0


def verify_fire_authorization(path):
    """Codex 1942Z item 1: the authority chain must be EXTERNALLY
    recomputable -- clean public worktree head/tree, module blob from
    the COMMITTED tree, correction doc in the same tree pinning the
    same identities, a codex PASS record reachable from
    agent-framework main binding base/patch/tree/module/query, and an
    owner-go record identifying that PASS with a strict UTC and the
    exact scope literal. Every check refuses MF4_FIRE_AUTH_* BEFORE
    any HTTP; a self-reported hash is never the pin."""
    if not path or not os.path.isfile(path):
        raise Refusal("MF4_FIRE_AUTH_MISSING", repr(path))
    raw = open(path, "rb").read()
    auth = _strict_loads(raw)
    if auth.get("schema") != AUTH_SCHEMA:
        raise Refusal("MF4_FIRE_AUTH_SCHEMA", str(auth.get("schema")))
    for k in ("public_head_commit", "public_head_tree",
              "module_git_blob_sha256", "query_contract_sha256",
              "codex_pass", "owner_fire_go",
              "output_target_must_be_absent"):
        if k not in auth:
            raise Refusal("MF4_FIRE_AUTH_INCOMPLETE", k)

    # 1. clean public worktree at the exact authorized head/tree,
    #    and that head must BE the published public master
    if _git(REPO, "status", "--porcelain").strip():
        raise Refusal("MF4_FIRE_AUTH_DIRTY_WORKTREE", REPO)
    head = _git(REPO, "rev-parse", "HEAD").decode().strip()
    tree = _git(REPO, "rev-parse", "HEAD^{tree}").decode().strip()
    if head != auth["public_head_commit"]:
        raise Refusal("MF4_FIRE_AUTH_HEAD_PIN", head[:16])
    if tree != auth["public_head_tree"]:
        raise Refusal("MF4_FIRE_AUTH_TREE_PIN", tree[:16])
    remote = _git(REPO, "rev-parse", "origin/master").decode().strip()
    if remote != head:
        raise Refusal("MF4_FIRE_AUTH_NOT_PUBLIC",
                      f"HEAD {head[:12]} != origin/master {remote[:12]}")

    # 2. module blob from the COMMITTED tree (never a runtime copy);
    #    runtime bytes must LF-normalize to that exact blob; the
    #    correction-3 doc in the SAME tree must pin the identities
    blob = _git(REPO, "show", f"HEAD:{MODULE_REL}")
    if _sha(blob) != auth["module_git_blob_sha256"]:
        raise Refusal("MF4_FIRE_AUTH_BLOB_PIN", _sha(blob)[:16])
    runtime = open(os.path.abspath(__file__), "rb").read()
    if _sha(runtime.replace(b"\r\n", b"\n")) != _sha(blob):
        raise Refusal("MF4_FIRE_AUTH_RUNTIME_PIN",
                      "runtime bytes diverge from the committed blob")
    _, cdig = query_contract()
    if auth["query_contract_sha256"] != cdig:
        raise Refusal("MF4_FIRE_AUTH_CONTRACT_PIN", cdig[:16])
    corr = _git(REPO, "show", f"HEAD:{CORRECTION3_REL}") \
        .decode("utf-8", errors="replace")
    if _sha(blob) not in corr or cdig not in corr:
        raise Refusal("MF4_FIRE_AUTH_CORRECTION_PIN",
                      "correction 3 in the authorized tree does not "
                      "pin the module/query identities")

    # 3. codex PASS (codex 2014Z item 2): a STRUCTURED JSON record --
    #    substring matching is banned; the verdict is an exact enum;
    #    reachability is from origin/main (never mutable local main);
    #    the reviewed result tree must equal the CURRENT tree and the
    #    reviewed base must be an ancestor of HEAD
    cp = auth["codex_pass"]
    for k in ("framework_commit", "file"):
        if not isinstance(cp.get(k), str) or not cp.get(k):
            raise Refusal("MF4_FIRE_AUTH_INCOMPLETE", f"codex_pass.{k}")
    def _reachable_from_origin_main(commit):
        got = _git(FRAMEWORK_REPO, "merge-base", commit,
                   "origin/main").decode().strip()
        return got == _git(FRAMEWORK_REPO, "rev-parse",
                           commit).decode().strip()
    if not _reachable_from_origin_main(cp["framework_commit"]):
        raise Refusal("MF4_FIRE_AUTH_PASS_UNREACHABLE",
                      cp["framework_commit"][:12])
    pass_bytes = _git(FRAMEWORK_REPO, "show",
                      f"{cp['framework_commit']}:{cp['file']}")
    try:
        pr = _strict_loads(pass_bytes)
    except Exception:                                   # noqa: BLE001
        raise Refusal("MF4_FIRE_AUTH_PASS_UNBOUND",
                      "pass record is not strict JSON")
    if pr.get("verdict") != "PRE_HTTP_PASS":
        raise Refusal("MF4_FIRE_AUTH_PASS_VERDICT",
                      repr(pr.get("verdict"))[:60])
    for k in ("base_commit", "bundle_sha256", "result_tree",
              "module_git_blob_sha256", "query_contract_sha256"):
        if not isinstance(pr.get(k), str) or not pr.get(k):
            raise Refusal("MF4_FIRE_AUTH_PASS_UNBOUND", f"missing {k}")
    if pr["result_tree"] != tree:
        raise Refusal("MF4_FIRE_AUTH_PASS_TREE",
                      f"reviewed tree {pr['result_tree'][:12]} != "
                      f"current {tree[:12]}")
    if pr["module_git_blob_sha256"] != _sha(blob) \
            or pr["query_contract_sha256"] != cdig:
        raise Refusal("MF4_FIRE_AUTH_PASS_UNBOUND",
                      "pass identities diverge from recomputed")
    if not _is_ancestor(REPO, pr["base_commit"], "HEAD"):
        raise Refusal("MF4_FIRE_AUTH_PASS_BASE",
                      f"reviewed base {pr['base_commit'][:12]} is not "
                      "an ancestor of HEAD")

    # 4. owner go: a LATER record descending from the PASS commit,
    #    binding the pass commit, current public HEAD/tree, exact
    #    scope literal, strict UTC, and the quote
    go = auth["owner_fire_go"]
    for k in ("quote", "utc", "scope", "pass_framework_commit",
              "source_framework_commit", "source_file"):
        if not isinstance(go.get(k), str) or not go.get(k):
            raise Refusal("MF4_FIRE_AUTH_INCOMPLETE",
                          f"owner_fire_go.{k}")
    if go["pass_framework_commit"] != cp["framework_commit"]:
        raise Refusal("MF4_FIRE_AUTH_GO_PASS_MISMATCH",
                      go["pass_framework_commit"][:12])
    try:
        dt.datetime.strptime(go["utc"], "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise Refusal("MF4_FIRE_AUTH_GO_UTC", repr(go["utc"]))
    if go["scope"] != SCOPE_LITERAL:
        raise Refusal("MF4_FIRE_AUTH_GO_SCOPE", repr(go["scope"])[:80])
    if not _reachable_from_origin_main(go["source_framework_commit"]):
        raise Refusal("MF4_FIRE_AUTH_GO_UNREACHABLE",
                      go["source_framework_commit"][:12])
    if not _is_ancestor(FRAMEWORK_REPO, cp["framework_commit"],
                        go["source_framework_commit"]):
        raise Refusal("MF4_FIRE_AUTH_GO_NOT_AFTER_PASS",
                      "go source does not descend from the pass commit")
    go_bytes = _git(FRAMEWORK_REPO, "show",
                    f"{go['source_framework_commit']}:"
                    f"{go['source_file']}")
    go_text = go_bytes.decode("utf-8", errors="replace")
    for token in (cp["framework_commit"], head, tree, SCOPE_LITERAL,
                  go["utc"], go["quote"]):
        if token not in go_text:
            raise Refusal("MF4_FIRE_AUTH_GO_UNBOUND",
                          f"go source lacks {token[:24]}")
    if auth["output_target_must_be_absent"] is not True:
        raise Refusal("MF4_FIRE_AUTH_INCOMPLETE",
                      "output_target_must_be_absent must be true")
    return auth, {"file": path, "sha256": _sha(raw), "bytes": len(raw),
                  "go_source_sha256": _sha(go_bytes)}


def calibration_snapshot_role_guard(snapshot, use):
    """Codex 1942Z item 4: the late-repair snapshot serves exactly two
    AMENDED roles -- calibration labels and RECOMPUTED historical
    calibration features. Any live/post-2026-08-29 prediction use must
    refuse; live features require a separately receipted issue-time
    catalog view."""
    role = (snapshot or {}).get("temporal_role")
    if role != "CALIBRATION_LATE_REPAIR":
        raise Refusal("MF4_CATALOG_ROLE_UNBOUND", repr(role))
    if use not in ("calibration_labels", "calibration_features"):
        raise Refusal("MF4_CATALOG_ROLE_VIOLATION",
                      f"late-repair snapshot offered for {use!r}")


def build_bboxes():
    import fault_segments as FS
    src_raw = open(FS.__file__, "rb").read()
    out = {}
    for region in ADMITTED:
        carrier = ALIAS.get(region, region)
        if carrier not in FS.FAULT_SEGMENTS:
            raise Refusal("MF4_BBOX_UNREGISTERED",
                          f"{region}: carrier {carrier} absent")
        if region not in PINNED_BBOXES:
            raise Refusal("MF4_BBOX_UNREGISTERED",
                          f"{region}: no pinned bbox (a region outside "
                          "the registered 13 must refuse typed, never "
                          "resolve through a neighbouring carrier)")
        lats, lons = [], []
        for seg in FS.FAULT_SEGMENTS[carrier]:
            for (lat, lon) in seg.polygon:
                lats.append(float(lat)), lons.append(float(lon))
        bbox = {"min_lat": min(lats), "max_lat": max(lats),
                "min_lon": min(lons), "max_lon": max(lons)}
        if bbox != PINNED_BBOXES[region]:
            raise Refusal("MF4_BBOX_PIN_MISMATCH",
                          f"{region}: recomputed {bbox} != pinned")
        out[region] = {"carrier": carrier, "bbox": bbox}
    return out, {"path": "monitoring/src/fault_segments.py",
                 "sha256": _sha(src_raw), "bytes": len(src_raw)}


def query_url(bbox):
    params = [
        ("format", "geojson"), ("starttime", T_START), ("endtime", T_END),
        ("minmagnitude", repr(MINMAG)),
        ("minlatitude", repr(bbox["min_lat"])),
        ("maxlatitude", repr(bbox["max_lat"])),
        ("minlongitude", repr(bbox["min_lon"])),
        ("maxlongitude", repr(bbox["max_lon"])),
        ("orderby", "time-asc"), ("limit", str(LIMIT)),
    ]
    return PROVIDER_URL + "?" + urllib.parse.urlencode(params), dict(params)


def _num(v):
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and v == v and v not in (float("inf"), float("-inf")))


def validate_events(region, bbox, raw):
    """Codex fix 3: the closed parser."""
    doc = _strict_loads(raw)
    if not isinstance(doc, dict) or doc.get("type") != "FeatureCollection":
        raise Refusal("MF4_CATALOG_MALFORMED",
                      f"{region}: not a FeatureCollection")
    feats = doc.get("features")
    if not isinstance(feats, list):
        raise Refusal("MF4_CATALOG_MALFORMED", f"{region}: no features")
    meta = doc.get("metadata") or {}
    count = meta.get("count")
    if (not isinstance(count, int) or isinstance(count, bool)
            or count != len(feats)):
        raise Refusal("MF4_CATALOG_COUNT_MISMATCH",
                      f"{region}: metadata.count {count!r} != "
                      f"{len(feats)} features")
    if len(feats) >= LIMIT:
        raise Refusal("MF4_CATALOG_QUERY_LIMIT",
                      f"{region}: count {len(feats)} hit limit {LIMIT}")
    t_lo = dt.datetime(2025, 10, 11, tzinfo=dt.timezone.utc)
    t_hi = dt.datetime(2026, 8, 28, tzinfo=dt.timezone.utc)
    events, seen, prev = [], set(), None
    for f in feats:
        if not isinstance(f, dict) or f.get("type") != "Feature":
            raise Refusal("MF4_CATALOG_MALFORMED",
                          f"{region}: non-Feature item")
        eid = f.get("id")
        if not isinstance(eid, str) or not eid:
            raise Refusal("MF4_CATALOG_MISSING_ID", f"{region}")
        p = f.get("properties") or {}
        geom = f.get("geometry") or {}
        if geom.get("type") != "Point":
            raise Refusal("MF4_CATALOG_MALFORMED",
                          f"{region}: {eid} non-Point geometry")
        g = geom.get("coordinates")
        mag, tms = p.get("mag"), p.get("time")
        if (not isinstance(g, list) or len(g) < 2 or not _num(g[0])
                or not _num(g[1]) or not _num(mag)
                or not isinstance(tms, int) or isinstance(tms, bool)):
            raise Refusal("MF4_CATALOG_MALFORMED",
                          f"{region}: {eid} null/malformed field")
        if float(mag) < MINMAG:
            raise Refusal("MF4_CATALOG_MAG_BELOW_THRESHOLD",
                          f"{region}: {eid} mag {mag}")
        t = dt.datetime.fromtimestamp(tms / 1000.0, dt.timezone.utc)
        if not (t_lo <= t < t_hi):
            raise Refusal("MF4_CATALOG_TEMPORAL_FILTER",
                          f"{region}: {eid} at {t.isoformat()}")
        lon, lat = float(g[0]), float(g[1])
        if not (bbox["min_lat"] <= lat <= bbox["max_lat"]
                and bbox["min_lon"] <= lon <= bbox["max_lon"]):
            raise Refusal("MF4_CATALOG_SPATIAL_FILTER",
                          f"{region}: {eid} at ({lat},{lon})")
        if eid in seen:
            raise Refusal("MF4_CATALOG_DUPLICATE_ID", f"{region}: {eid}")
        seen.add(eid)
        if prev is not None and (tms, eid) < prev:
            raise Refusal("MF4_CATALOG_ORDER_VIOLATION",
                          f"{region}: {eid} out of time-asc order")
        prev = (tms, eid)
        events.append({"id": eid, "time_ms": tms,
                       "time_utc": t.strftime(
                           "%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                       "lat": lat, "lon": lon, "mag": float(mag)})
    return events


def canonical_event_table(per_region):
    """Codex fix 4: ONE canonical global table sorted (time_ms, id),
    identical cross-region duplicates deduplicated, region membership
    kept as a side channel; inconsistent shared IDs refuse."""
    canon_by_id, membership = {}, {}
    for region, evs in per_region.items():
        for ev in evs:
            canon = json.dumps(ev, sort_keys=True)
            if ev["id"] in canon_by_id:
                if canon_by_id[ev["id"]] != canon:
                    raise Refusal("MF4_CATALOG_INCONSISTENT_ID",
                                  f"{ev['id']}")
            else:
                canon_by_id[ev["id"]] = canon
            membership.setdefault(ev["id"], []).append(region)
    table = sorted((json.loads(c) for c in canon_by_id.values()),
                   key=lambda e: (e["time_ms"], e["id"]))
    for k in membership:
        membership[k] = sorted(membership[k])
    raw = json.dumps(table, sort_keys=True).encode("utf-8")
    return table, membership, _sha(raw)


def _assert_safe_dir(path):
    if os.path.islink(path):
        raise Refusal("MF4_FIRE_TARGET_UNSAFE", f"link: {path}")
    rp, ap = os.path.realpath(path), os.path.abspath(path)
    if os.path.normcase(rp) != os.path.normcase(ap):
        raise Refusal("MF4_FIRE_TARGET_UNSAFE",
                      f"reparse escape: {path} -> {rp}")


def _seal(staging, name, data):
    """Exclusive-create seal with reopen-verify + fsync: EVERY sealed
    evidence file (raw responses and attempt records included, codex
    2014Z item 4) is made durable before the directory publishes."""
    p = os.path.join(staging, name)
    with open(p, "xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    with open(p, "rb") as f:
        if _sha(f.read()) != _sha(data):
            raise Refusal("MF4_FIRE_FINALIZATION_EXCEPTION",
                          f"seal reopen mismatch: {name}")
    return p


def fire(auth_path, opener=urllib.request.urlopen):
    auth, auth_ident = verify_fire_authorization(auth_path)
    bboxes, fs_ident = build_bboxes()
    ident = module_identities()
    contract, contract_sha = query_contract()

    if os.path.exists(FINAL_DIR):
        raise Refusal("MF4_FIRE_TARGET_EXISTS", FINAL_DIR)
    staging = FINAL_DIR + ".staging"
    if os.path.exists(staging):
        raise Refusal("MF4_FIRE_STAGING_EXISTS",
                      staging + " (continuation requires a NEW owner "
                      "decision and must reuse sealed responses, never "
                      "re-query)")
    parent = os.path.dirname(FINAL_DIR)
    os.makedirs(parent, exist_ok=True)
    _assert_safe_dir(parent)
    os.mkdir(staging)                       # exclusive: raced -> OSError
    _assert_safe_dir(staging)

    attempts, per_region = {}, {}
    common = {"authorization": auth_ident, "authorization_content": auth,
              "acquisition_code_identity": ident,
              "fault_segments_identity": fs_ident,
              "query_contract_sha256": contract_sha}

    def refusal_manifest(code, detail):
        man = dict(common)
        man.update({"schema": "geospec-mf4-catalog-refusal-manifest-v1",
                    "refusal_code": code, "refusal_detail": detail,
                    "utc": _utcnow(), "attempts": attempts,
                    "note": ("staging evidence preserved; every sealed "
                             "response is immutable; continuation "
                             "requires a new owner decision and reuses "
                             "sealed bytes")})
        with open(os.path.join(staging, "REFUSAL_MANIFEST.json"), "w",
                  encoding="utf-8", newline="\n") as f:
            json.dump(man, f, indent=1, sort_keys=True)
            f.write("\n")

    for region in ADMITTED:
        url, params = query_url(bboxes[region]["bbox"])
        att = {"region": region, "requested_url": url, "params": params,
               "bbox": bboxes[region]["bbox"],
               "carrier": bboxes[region]["carrier"],
               "request_utc": _utcnow()}
        attempts[region] = att
        try:
            with opener(url, timeout=120) as resp:
                att["effective_url"] = resp.geturl()
                att["http_status"] = resp.status
                att["response_headers"] = dict(resp.headers.items())
                raw = resp.read()
            att["response_utc"] = _utcnow()
            att["raw_bytes"] = len(raw)
            att["raw_sha256"] = _sha(raw)
            # seal BEFORE parsing (codex fix 2)
            att["raw_file"] = f"raw_{region}.geojson"
            _seal(staging, att["raw_file"], raw)
            _seal(staging, f"attempt_{region}.json",
                  (json.dumps(att, sort_keys=True, indent=1) + "\n")
                  .encode("utf-8"))
            if att["http_status"] != 200:
                raise Refusal("MF4_CATALOG_HTTP_STATUS",
                              f"{region}: {att['http_status']}")
            ctype = att["response_headers"].get("Content-Type", "")
            if ctype.split(";")[0].strip().lower() \
                    not in ALLOWED_CONTENT_TYPES:
                raise Refusal("MF4_CATALOG_CONTENT_TYPE",
                              f"{region}: {ctype!r}")
            if att["effective_url"] != url:
                raise Refusal("MF4_CATALOG_REDIRECT",
                              f"{region}: {att['effective_url']}")
            events = validate_events(region, bboxes[region]["bbox"], raw)
            att["event_count"] = len(events)
            att["parse_result"] = "OK"
            per_region[region] = events
        except Refusal as r:
            att["parse_result"] = r.code
            refusal_manifest(r.code, str(r))
            raise
        except Exception as e:                          # noqa: BLE001
            att["parse_result"] = f"EXCEPTION:{type(e).__name__}"
            refusal_manifest("MF4_CATALOG_ATTEMPT_EXCEPTION",
                             f"{region}: {type(e).__name__}: {e}")
            raise Refusal("MF4_CATALOG_ATTEMPT_EXCEPTION",
                          f"{region}: {type(e).__name__}: {e}")

    try:
        table, membership, table_sha = canonical_event_table(per_region)
    except Refusal as r:
        refusal_manifest(r.code, str(r))
        raise

    snap = {"schema": "geospec-mf4-calibration-catalog-snapshot-v1",
            "temporal_role": "CALIBRATION_LATE_REPAIR",
            "temporal_role_policy": (
                "historical calibration recent_event is RECOMPUTED "
                "from this single late-repair snapshot, superseding "
                "the annex issue-time feature clause as an "
                "AMENDED_AFTER_FREEZE feature-provenance change "
                "(correction 3); live predictions must refuse this "
                "snapshot"),
            "amendment": ("docs/f2g_window2_execution/"
                          "amendment_mf4_late_catalog_repair_20260829.md"),
            "lane_status": "AMENDED_AFTER_FREEZE",
            "query_contract": contract,
            "query_contract_sha256": contract_sha,
            "canonical_event_table": table,
            "canonical_event_table_sha256": table_sha,
            "region_membership": membership,
            "events_by_region_counts":
                {r: len(v) for r, v in per_region.items()},
            **common}
    # codex 1942Z item 2: finalization inside ONE terminal guard --
    # canonical bytes, exclusive seal, reopen/hash/validate, fsync,
    # atomic publish; any exception preserves staging and emits a
    # typed refusal manifest where the filesystem still permits.
    try:
        snap_bytes = (json.dumps(snap, indent=1, sort_keys=True)
                      + "\n").encode("utf-8")
        sp = _seal(staging, "catalog_snapshot_v1.json", snap_bytes)
        with open(sp, "rb") as f:
            reread = f.read()
        if _sha(reread) != _sha(snap_bytes):
            raise Refusal("MF4_FIRE_FINALIZATION_EXCEPTION",
                          "snapshot reopen digest mismatch")
        _strict_loads(reread)
        rec = {"schema": "geospec-mf4-catalog-acquisition-receipt-v1",
               "fired_utc": _utcnow(), "attempts": attempts,
               "snapshot_file": "catalog_snapshot_v1.json",
               "snapshot_sha256": _sha(snap_bytes),
               "canonical_event_table_sha256": table_sha, **common}
        rec_bytes = (json.dumps(rec, indent=1, sort_keys=True)
                     + "\n").encode("utf-8")
        rp = _seal(staging, "acquisition_receipt_v1.json", rec_bytes)
        with open(rp, "rb") as f:
            if _sha(f.read()) != _sha(rec_bytes):
                raise Refusal("MF4_FIRE_FINALIZATION_EXCEPTION",
                              "receipt reopen digest mismatch")
        # every file (raw responses, attempt records, snapshot,
        # receipt) is already fsynced inside _seal with its WRITE
        # handle -- an O_RDONLY fsync is EBADF on Windows (found by
        # lock D7, disclosed). Directory-level fsync is unavailable on
        # Windows; per-file fsync + atomic rename is the strongest
        # local guarantee (disclosed).
        os.rename(staging, FINAL_DIR)      # atomic directory publish
        return rec
    except Refusal as r:
        refusal_manifest(r.code, str(r))
        raise
    except Exception as e:                              # noqa: BLE001
        refusal_manifest("MF4_FIRE_FINALIZATION_EXCEPTION",
                         f"{type(e).__name__}: {e}")
        raise Refusal("MF4_FIRE_FINALIZATION_EXCEPTION",
                      f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--fire", action="store_true")
    ap.add_argument("--fire-authorization", default=None)
    args = ap.parse_args()

    bboxes, fs_ident = build_bboxes()
    print(f"bbox pin check: 13/13 exact (fault_segments "
          f"{fs_ident['sha256'][:16]})")
    if args.plan:
        _, cdig = query_contract()
        ident = module_identities()
        print(f"query-contract sha256: {cdig}")
        print(f"module runtime sha256: {ident['runtime_sha256']}")
        print(f"module git-blob sha256: {ident['git_blob_sha256']}")
        for region in ADMITTED:
            url, _ = query_url(bboxes[region]["bbox"])
            print(f"{region}: {url}")
        print("PLAN ONLY -- zero HTTP performed")
        return
    rec = fire(args.fire_authorization)
    print("SNAPSHOT PUBLISHED:", FINAL_DIR)
    print("  snapshot sha256:", rec["snapshot_sha256"])
    print("  canonical event table sha256:",
          rec["canonical_event_table_sha256"])


if __name__ == "__main__":
    main()
