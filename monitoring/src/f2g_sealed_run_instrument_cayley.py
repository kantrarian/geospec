#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SEALED-RUN INSTRUMENT (cayley) -- fault2graph Phase B, calendar frame.

Authorized by the FRESH owner seal (option a, quote sha bb94a28b...,
manifest owner_seal_cal_fresh @ geospec 931ec65): at most ONE real-data
run, B2A verdict-bearing at certified power, B1A/B3A typed non-answers.

This module is the INSTRUMENT, not the fire: it (1) builds the hash-
sealed INPUT manifest -- every input bound by digest, snapshot content
NEVER parsed here (byte hashing only); (2) provides the snapshot->panel
ADAPTER whose only parse path is guarded: passing the real snapshot
directory without allow_real=True refuses typed SEALED_INPUT_GUARD; KATs
run on synthetic snapshot trees carrying the REAL calendar authority
(bound mode) with hash-derived values. The FIRE lives in the separate
driver and is gated on codex's instrument check.

Adapter contract (derived from the Phase-A builder CODE, never its data):
snapshot = {kind: daily_snapshot, carrier_key, day, station_index_digest,
station_states, coheres_with: [{station_a<min>, station_b<max>, r, ...}]}.
Panel: bound-mode calendar panel exactly as the admitted engine's
_cal_load consumes -- REAL 132-day shared calendar + exact per-carrier
masks (registered days = the calendar authority, byte-verified by the
engine), stations = the 35-station selected registry (12/12/11; includes
KO.KHMN, which is geometry-excluded but data-selected), segments = the
real Phase-A segment tables, r values copied verbatim, absences stay
absent (never synthesized).
"""
import hashlib
import json
import os
import sys
import time

ARTIFACT_ROOT = "data/phase_a_builder_artifact_v1"
ANCHOR_SHA = ("0850cf3d24602ab0ba420412f5b292c9d33464852fbb9de8d7363a019c"
              "7886ad")
ANCHOR_REPO_PATH = "docs/evidence_phase_a_result_anchor.json"
CAL_AUTH_COMMIT = "8111805"
CAL_AUTH_PATH = "docs/f2g_phase_b_shared_calendar_v1.json"
AMENDMENT2_SHA = ("58b513b6c30b70c8014510788da9d7d819ce8971ca59b7dfdc11c57a"
                  "1664586f")
ENGINE_COMMIT = "24b0d8f"
ENGINE_PATH = "monitoring/src/d2_f2g_phase_b_stats.py"
RESULTS_COMMIT = "9387d5c"
B2A_RESULTS_PATH = "docs/f2g_phase_b_power_annex_b2a_cal_results.json"
CARRIERS = ("istanbul_marmara", "socal_coachella", "turkey_kahramanmaras")
# DISCLOSED: the admitted engine (frozen 24b0d8f) accepts exactly one panel
# FORMAT identifier, "fixture-panel-cal-v1" -- a format name from the power
# lane, not a provenance claim. Real/sealed provenance rides the sealed-run
# input manifest and result artifact, never this string. Changing the frozen
# engine over a name would force a bar re-run + full repin for zero
# behavioral gain.
PANEL_SCHEMA = "fixture-panel-cal-v1"
MANIFEST_OUT = "docs/f2g_sealed_run_input_manifest.json"


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _sha_file(p):
    return _sha_bytes(open(p, "rb").read())


def _blob(repo, commit, path):
    import subprocess
    return subprocess.check_output(
        ["git", "cat-file", "blob", f"{commit}:{path}"], cwd=repo)


def load_registry(repo):
    """Selected 35-station registry + segment map from the Phase-A tables
    (geometry files; no measurement values)."""
    root = os.path.join(repo, ARTIFACT_ROOT, "tables")
    mo = [json.loads(l) for l in open(os.path.join(root, "member_of.jsonl"),
                                      encoding="utf-8") if l.strip()]
    reg = {ck: {"stations": [], "segments": {}} for ck in CARRIERS}
    for m in sorted(mo, key=lambda x: x["station_id"]):
        reg[m["carrier_key"]]["stations"].append(m["station_id"])
        reg[m["carrier_key"]]["segments"][m["station_id"]] = \
            m["segment_name"]
    counts = {ck: len(reg[ck]["stations"]) for ck in CARRIERS}
    assert counts == {"istanbul_marmara": 12, "socal_coachella": 12,
                      "turkey_kahramanmaras": 11}, counts
    return reg


def load_calendar(repo):
    auth = json.loads(_blob(repo, CAL_AUTH_COMMIT, CAL_AUTH_PATH))
    cal = list(auth["shared_calendar_days"])
    masks = {ck: list(auth["carrier_masks"][ck]["registered_days"])
             for ck in CARRIERS}
    assert len(cal) == 132
    assert {ck: len(masks[ck]) for ck in CARRIERS} == {
        "istanbul_marmara": 111, "socal_coachella": 108,
        "turkey_kahramanmaras": 111}
    return cal, masks


def build_input_manifest(repo, write=True):
    """Hash-seal EVERY input before any value is read. Byte hashing only:
    no snapshot content is parsed by this function."""
    anchor_artifact = os.path.join(repo, ARTIFACT_ROOT,
                                   "phase_a_result.json")
    ab = open(anchor_artifact, "rb").read()
    if _sha_bytes(ab) != ANCHOR_SHA:
        raise RuntimeError("ANCHOR_SHA_MISMATCH (artifact copy)")
    if _sha_bytes(_blob(repo, "HEAD", ANCHOR_REPO_PATH)) != ANCHOR_SHA:
        raise RuntimeError("ANCHOR_SHA_MISMATCH (committed docs copy)")
    anchor = json.loads(ab)
    od = anchor["output_digests"]
    n_snap = 0
    per_carrier = {ck: [] for ck in CARRIERS}
    for rel, want in sorted(od.items()):
        got = _sha_file(os.path.join(repo, ARTIFACT_ROOT, rel))
        if got != want:
            raise RuntimeError(f"OUTPUT_DIGEST_MISMATCH: {rel}")
        if rel.startswith("snapshots/"):
            n_snap += 1
            ck, day = rel.split("/")[1], rel.split("/")[2][:-5]
            per_carrier[ck].append(day)
    cal, masks = load_calendar(repo)
    for ck in CARRIERS:
        if sorted(per_carrier[ck]) != sorted(masks[ck]):
            raise RuntimeError(f"MASK_SNAPSHOT_SET_MISMATCH: {ck}")
    engine_disk = open(os.path.join(repo, ENGINE_PATH), "rb").read() \
        .replace(b"\r\n", b"\n")
    engine_blob = _blob(repo, ENGINE_COMMIT, ENGINE_PATH)
    if _sha_bytes(engine_disk) != _sha_bytes(engine_blob):
        raise RuntimeError("ENGINE_UNATTESTED")
    b2a_res = json.loads(_blob(repo, RESULTS_COMMIT, B2A_RESULTS_PATH))
    if b2a_res["terminal_type"] != "CERTIFIED":
        raise RuntimeError("B2A_POWER_CONTRACT_NOT_CERTIFIED")
    manifest = {
        "schema": "f2g-sealed-run-input-manifest-v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                       time.gmtime()),
        "seal": {"manifest_block": "owner_seal_cal_fresh",
                 "geospec_commit": "931ec65",
                 "quote_sha256": "bb94a28bec0060d7d45b799f17536c499539f324"
                                 "da0f58ac3a1edcfdf594a7e4"},
        "anchor": {"sha256": ANCHOR_SHA,
                   "artifact_path": f"{ARTIFACT_ROOT}/phase_a_result.json",
                   "repo_path": ANCHOR_REPO_PATH},
        "snapshots": {"count": n_snap,
                      "digest_authority": "anchor.output_digests "
                                          "(byte-verified per file)",
                      "per_carrier_days": {ck: len(per_carrier[ck])
                                           for ck in CARRIERS}},
        "calendar_authority": {"commit": CAL_AUTH_COMMIT,
                               "path": CAL_AUTH_PATH,
                               "sha256": _sha_bytes(_blob(
                                   repo, CAL_AUTH_COMMIT, CAL_AUTH_PATH))},
        "amendment2_sha256": AMENDMENT2_SHA,
        "engine": {"commit": ENGINE_COMMIT, "path": ENGINE_PATH,
                   "lf_sha256": _sha_bytes(engine_blob)},
        "registry": {"selected_counts": {"istanbul_marmara": 12,
                                         "socal_coachella": 12,
                                         "turkey_kahramanmaras": 11},
                     "source": f"{ARTIFACT_ROOT}/tables/member_of.jsonl",
                     "note": "KO.KHMN geometry-excluded but data-selected;"
                             " it is a LOCO fold like every other station"},
        "power_contracts": {
            "B2A": {"certified": True, "results_commit": RESULTS_COMMIT,
                    "results_path": B2A_RESULTS_PATH,
                    "results_blob_sha256": _sha_bytes(_blob(
                        repo, RESULTS_COMMIT, B2A_RESULTS_PATH)),
                    "certified_points": [{"m": 1}, {"m": 2}, {"m": 3}],
                    "lb95": 0.8608916593317347},
            "B1A": None, "B3A": None},
        "verdict_typing": "B2A verdict-bearing (certified contract); "
                          "B1A/B3A power_contract=None -> nonpositives "
                          "type CANNOT_DETERMINE_NO_POWER (frozen sec-5)",
        "read_disclosure": "this manifest was built with BYTE HASHING "
                           "only; no snapshot value was parsed",
        "one_shot": "at most ONE real-data run; the driver refuses if the "
                    "sealed result artifact already exists",
    }
    if write:
        out = os.path.join(repo, MANIFEST_OUT)
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(manifest, f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"wrote {MANIFEST_OUT} ({n_snap} snapshots bound)")
    return manifest


def build_panel(repo, snapshots_dir, allow_real=False):
    """Snapshots -> bound calendar panel. THE ONLY parse path for
    snapshot content. Passing the real artifact directory without
    allow_real=True refuses: the fire is gated on codex's instrument
    check, and KATs use synthetic trees."""
    real_root = os.path.normcase(os.path.abspath(
        os.path.join(repo, ARTIFACT_ROOT)))
    if os.path.normcase(os.path.abspath(snapshots_dir)).startswith(
            real_root) and not allow_real:
        raise RuntimeError("SEALED_INPUT_GUARD: real snapshot directory "
                           "requires allow_real=True (fire-time only, "
                           "after the codex instrument check)")
    cal, masks = load_calendar(repo)
    reg = load_registry(repo)
    carriers = {}
    for ck in CARRIERS:
        stations = set(reg[ck]["stations"])
        r = {}
        for day in masks[ck]:
            p = os.path.join(snapshots_dir, ck, f"{day}.json")
            if not os.path.exists(p):
                raise RuntimeError(f"MISSING_REGISTERED_SNAPSHOT: "
                                   f"{ck}/{day}")
            snap = json.loads(open(p, encoding="utf-8").read())
            if snap.get("kind") != "daily_snapshot" or \
                    snap.get("carrier_key") != ck or \
                    snap.get("day") != day:
                raise RuntimeError(f"SNAPSHOT_FIELD_MISMATCH: {ck}/{day}")
            seen = set()
            for e in snap.get("coheres_with", []):
                a, b = e["station_a"], e["station_b"]
                if a >= b:
                    raise RuntimeError(f"PAIR_ORDER: {ck}/{day} {a}|{b}")
                if a not in stations or b not in stations:
                    raise RuntimeError(f"UNSELECTED_STATION_EDGE: "
                                       f"{ck}/{day} {a}|{b}")
                if (a, b) in seen:
                    raise RuntimeError(f"DUPLICATE_PAIR: {ck}/{day} "
                                       f"{a}|{b}")
                seen.add((a, b))
                v = e["r"]
                if type(v) not in (int, float) or v != v or \
                        abs(v) > 1.0:
                    raise RuntimeError(f"NONFINITE_OR_RANGE_VALUE: "
                                       f"{ck}/{day} {a}|{b}")
                r.setdefault(f"{a}|{b}", {})[day] = float(v)
        extra = [d for d in os.listdir(os.path.join(snapshots_dir, ck))
                 if d.endswith(".json") and d[:-5] not in set(masks[ck])]
        if extra:
            raise RuntimeError(f"UNREGISTERED_SNAPSHOT_DAY: {ck} {extra[:2]}")
        carriers[ck] = {"registered_days": list(masks[ck]),
                        "stations": list(reg[ck]["stations"]),
                        "segments": dict(reg[ck]["segments"]),
                        "r": r}
    return {"schema": PANEL_SCHEMA,
            "calendar_authority_mode": "bound",
            "shared_calendar_days": list(cal),
            "carriers": carriers}


def panel_digest(panel):
    return _sha_bytes(json.dumps(panel, sort_keys=True,
                                 separators=(",", ":")).encode("utf-8"))


def drop_station(panel, st):
    out = {k: panel[k] for k in ("schema", "calendar_authority_mode",
                                 "shared_calendar_days")}
    cs = {}
    for ck, c in panel["carriers"].items():
        if st not in c["stations"]:
            cs[ck] = c
            continue
        cs[ck] = {"registered_days": c["registered_days"],
                  "stations": [x for x in c["stations"] if x != st],
                  "segments": {k: v for k, v in c["segments"].items()
                               if k != st},
                  "r": {e: s for e, s in c["r"].items()
                        if st not in e.split("|")}}
    out["carriers"] = cs
    return out


def all_stations(repo):
    reg = load_registry(repo)
    out = []
    for ck in CARRIERS:
        out.extend(reg[ck]["stations"])
    return out


# ---------------- KATs (synthetic snapshot trees only) ----------------

def _hval(*parts):
    """Deterministic hash-derived value in (-0.95, 0.95)."""
    h = int.from_bytes(hashlib.sha256("|".join(parts).encode())
                       .digest()[:8], "big")
    return (h / 2 ** 64) * 1.9 - 0.95


def make_synthetic_tree(repo, root):
    cal, masks = load_calendar(repo)
    reg = load_registry(repo)
    for ck in CARRIERS:
        os.makedirs(os.path.join(root, ck), exist_ok=True)
        sts = sorted(reg[ck]["stations"])
        pairs = [(a, b) for i, a in enumerate(sts) for b in sts[i + 1:]]
        for day in masks[ck]:
            rows = [{"type": "coheres_with", "campaign_id": "kat",
                     "carrier_key": ck, "day": day, "algorithm_id": "kat",
                     "station_index_digest": "kat0", "station_a": a,
                     "station_b": b, "r": _hval(ck, day, a, b),
                     "unit": 1, "n_overlap": 100}
                    for a, b in pairs]
            snap = {"schema": "kat", "kind": "daily_snapshot",
                    "campaign_id": "kat", "carrier_key": ck, "day": day,
                    "station_index_digest": "kat0", "station_states": {},
                    "coheres_with": rows}
            with open(os.path.join(root, ck, f"{day}.json"), "w",
                      encoding="utf-8", newline="\n") as f:
                json.dump(snap, f, sort_keys=True)
    return root


def run_kats(repo, scratch):
    t0 = time.time()
    sys.path.insert(0, os.path.join(repo, "monitoring", "src"))
    os.chdir(repo)
    import d2_f2g_phase_b_stats as E
    tree = make_synthetic_tree(repo, os.path.join(scratch,
                                                  "sealed_kat_tree"))
    # K0: real-dir guard refuses
    try:
        build_panel(repo, os.path.join(repo, ARTIFACT_ROOT, "snapshots"))
        raise SystemExit("K0 DEFECT: real dir accepted without allow_real")
    except RuntimeError as e:
        assert "SEALED_INPUT_GUARD" in str(e)
    print("K0 PASS: real-directory guard refuses")
    # K1: synthetic tree -> panel -> engine accepts bound mode, all fams
    panel = build_panel(repo, tree)
    dig = panel_digest(panel)
    for fam, fn in (("B1A", E.b1a_family_cal), ("B2A", E.b2a_family_cal),
                    ("B3A", E.b3a_family_cal)):
        res = fn(panel, doc_sha256=AMENDMENT2_SHA, n_draws=49,
                 power_contract=None)
        assert res.get("verdict") is not None, (fam, res)
        print(f"K1 {fam}: engine ACCEPTED bound real-shaped panel "
              f"(verdict {res['verdict']!r}, p={res.get('p_value')})")
    # K2: drop_station removes station/edges/segment entry
    st = panel["carriers"][CARRIERS[0]]["stations"][0]
    d2 = drop_station(panel, st)
    c2 = d2["carriers"][CARRIERS[0]]
    assert st not in c2["stations"] and st not in c2["segments"]
    assert all(st not in e.split("|") for e in c2["r"])
    print(f"K2 PASS: drop_station({st}) exact")
    # K3: typed refusals
    def doctor(name, mutate, expect):
        import shutil
        broken = os.path.join(scratch, f"sealed_kat_{name}")
        if os.path.exists(broken):
            shutil.rmtree(broken)
        shutil.copytree(tree, broken)
        mutate(broken)
        try:
            build_panel(repo, broken)
            raise SystemExit(f"K3 DEFECT: {name} accepted")
        except RuntimeError as e:
            assert expect in str(e), (name, str(e))
        print(f"K3 PASS: {name} -> {expect}")

    cal, masks = load_calendar(repo)
    ck0 = CARRIERS[0]
    d0 = masks[ck0][0]
    absent = sorted(set(cal) - set(masks[ck0]))[0]

    def m_missing(root):
        os.unlink(os.path.join(root, ck0, f"{d0}.json"))

    def m_extra(root):
        import shutil
        shutil.copy(os.path.join(root, ck0, f"{d0}.json"),
                    os.path.join(root, ck0, f"{absent}.json"))

    def _rewrite(root, fn):
        p = os.path.join(root, ck0, f"{d0}.json")
        s = json.loads(open(p, encoding="utf-8").read())
        fn(s)
        json.dump(s, open(p, "w", encoding="utf-8", newline="\n"))

    doctor("missing-registered-day", m_missing,
           "MISSING_REGISTERED_SNAPSHOT")
    doctor("absent-day-synthesis", m_extra, "UNREGISTERED_SNAPSHOT_DAY")
    doctor("unselected-station-edge", lambda r: _rewrite(
        r, lambda s: s["coheres_with"][0].update(station_a="AA.FAKE")),
        "UNSELECTED_STATION_EDGE")
    doctor("duplicate-pair", lambda r: _rewrite(
        r, lambda s: s["coheres_with"].append(dict(s["coheres_with"][0]))),
        "DUPLICATE_PAIR")
    doctor("nonfinite-value", lambda r: _rewrite(
        r, lambda s: s["coheres_with"][0].update(r=float("nan"))),
        "NONFINITE_OR_RANGE_VALUE")
    doctor("day-field-mismatch", lambda r: _rewrite(
        r, lambda s: s.update(day="1999-01-01")),
        "SNAPSHOT_FIELD_MISMATCH")
    print(f"ALL SEALED-INSTRUMENT KATS PASS ({time.time()-t0:.0f}s, "
          f"synthetic panel digest {dig[:12]}...)")


if __name__ == "__main__":
    repo = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
    if len(sys.argv) > 2 and sys.argv[2] == "--kats":
        run_kats(repo, sys.argv[3])
    else:
        build_input_manifest(repo)
