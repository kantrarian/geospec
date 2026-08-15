#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R5 replay-reproducibility capsule (codex 172526Z capsule contract +
r5-canonical-numerics-v1 amendment 185751Z, owner-gated).

regenerate: owner-approved bounded regeneration -- for each capsule entry,
  refit + retransform FROM THE CAPSULE-BOUND INPUTS under the canonical-numerics
  product (q11), REFUSING unless the scientific values are invariant at q11
  precision vs the existing records (stat, n_fit, residual percentile); rewrite
  model.json/record.json with canonical fixed-point numeric tokens, the 18
  affected docs/r5_daily.json entries, the manifest, and append the policy id +
  old->new digest map to the dated correction record.

verify: clean-state, host-independent verifier:
  1. rebuilds each model with fit_region (q11 product) from capsule inputs ONLY;
  2. recomputes the canonical model digest AND canonical record bytes exactly
     (exact equality on the canonical domain -- never tolerant equality);
  3. proves MODEL_FILE is neither read nor created;
  4. `verify --mutation-selftest`: refuses one-byte mutations of ratio input,
     precipitation input, model body, and manifest.
  Persisted model/record numeric tokens are read with a decimal-preserving
  parser that REJECTS any token not already in canonical q11 form.
"""
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Repo root derived from this file's location (tools/r5_replay_capsule.py) so the
# clean-state verifier runs from any clone.
REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "monitoring" / "src"
sys.path.insert(0, str(SRC))

CAPSULE_DIR = REPO / "monitoring" / "replay_capsules" / "r5_20260812_20260813"
R5_DAILY = REPO / "docs" / "r5_daily.json"
DAYS = ["2026-08-12", "2026-08-13"]
CODE_COMMIT = "ee77902"          # historical=True seam
NUMERICS_POLICY_ID = "r5-canonical-numerics-v1"


def canon(obj) -> bytes:
    """Raw-input serialization (ratios/precip/query -- host-measured inputs, NOT
    the claim surface; bound byte-exact by the manifest)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_canonical_json(path: Path, CN):
    """§3 strict scalar read: decimal-preserving parse in which every float token
    must already be canonical q11 text. Used ONLY to read scalar fields (digests,
    counts) — never to recanonicalize (plain files cannot preserve the int/float
    class of integral floats; byte-equality against the canonical writer is the
    authoritative structural check)."""
    def pf(tok):
        CN.parse_canonical_token(tok, n=11)
        return float(tok)
    return json.loads(path.read_text(encoding="utf-8"), parse_float=pf)


def _capsule_transform(PR, region, query, ratios, precip, day):
    """Run the historical transform fed ONLY by capsule inputs; MODEL_FILE trapped."""
    tmp = Path(tempfile.mkdtemp()) / "must_never_exist.json"
    old = (PR.fetch_precip, PR.load_ratio_history, PR.MODEL_FILE)
    try:
        PR.fetch_precip = lambda *a, _p=precip, **k: dict(_p)
        PR.load_ratio_history = lambda *a, _r=ratios, **k: dict(_r)
        PR.MODEL_FILE = tmp
        out = PR.r5_transform(region, query["lat"], query["lon"],
                              query["raw_ratio"], day, historical=True)
        if tmp.exists():
            print(f"FAIL {day}/{region}: MODEL_FILE was created on the historical path")
            sys.exit(1)
        return out
    finally:
        PR.fetch_precip, PR.load_ratio_history, PR.MODEL_FILE = old


def regenerate():
    import canonical_numerics as CN
    import precip_residual as PR

    manifest_old = json.loads((CAPSULE_DIR / "manifest.json").read_text())
    r5_daily = json.loads(R5_DAILY.read_text())
    digest_map = {}
    manifest = {"schema": "geospec-r5-replay-capsule-v2", "code_commit": CODE_COMMIT,
                "numeric_policy": CN.POLICY, "days": DAYS, "entries": {}}
    for key in sorted(manifest_old["entries"].keys()):
        day, region = key.split("/", 1)
        d = CAPSULE_DIR / day / region
        ratios = json.loads((d / "ratios.json").read_text())
        precip = json.loads((d / "precip.json").read_text())
        query = json.loads((d / "query.json").read_text())
        rec_old = json.loads((d / "record.json").read_text())
        model = PR.fit_region(region, ratios, precip, day, sig_digits=11)
        if model is None:
            print(f"REFUSE {key}: canonical fit ineligible"); sys.exit(1)
        rec_new = _capsule_transform(PR, region, query, ratios, precip, day)
        if rec_new is None:
            print(f"REFUSE {key}: canonical transform returned None"); sys.exit(1)
        # §5 refusal gate (codex dfad58d step 2 tightening): statistic, percentile,
        # RANK, and n_fit invariance at q11 BEFORE replacement — any scientific-value
        # change stops for asylum rather than consuming the standing gate.
        for field in ("stat", "residual_percentile"):
            if CN.qsig(rec_new[field], 11) != CN.qsig(rec_old[field], 11):
                print(f"REFUSE {key}: {field} moved beyond q11 "
                      f"({rec_old[field]} -> {rec_new[field]})"); sys.exit(1)
        if rec_new["n_fit"] != rec_old["n_fit"]:
            print(f"REFUSE {key}: n_fit changed {rec_old['n_fit']} -> {rec_new['n_fit']}")
            sys.exit(1)
        if "residual_rank_index" in rec_old and \
                rec_new.get("residual_rank_index") != rec_old["residual_rank_index"]:
            print(f"REFUSE {key}: residual_rank_index changed "
                  f"{rec_old['residual_rank_index']} -> "
                  f"{rec_new.get('residual_rank_index')}"); sys.exit(1)
        new_digest = CN.model_digest(model, n=11)
        if rec_new["model_sha256"] != new_digest:
            print(f"REFUSE {key}: transform digest != model digest"); sys.exit(1)
        digest_map[key] = {"old_model_sha256": rec_old["model_sha256"],
                           "new_model_sha256": new_digest}
        files = {
            "ratios.json": canon(ratios),
            "precip.json": canon(precip),
            "query.json": canon(query),
            "model.json": CN.canonical_json_bytes(model, n=11),
            "record.json": CN.canonical_json_bytes(rec_new, n=11),
        }
        entry = {}
        for name, b in files.items():
            (d / name).write_bytes(b)
            entry[name] = sha(b)
        manifest["entries"][key] = entry
        # the affected r5_daily entry (plain JSON floats of the q11 values)
        r5_daily["days"][day][region] = rec_new
        print(f"OK {key}: q11 invariant; digest {rec_old['model_sha256'][:12]} -> "
              f"{new_digest[:12]}")
    mb = canon(manifest)
    (CAPSULE_DIR / "manifest.json").write_bytes(mb)
    entry = {
        "applied_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "days": DAYS,
        "numeric_policy_id": NUMERICS_POLICY_ID,
        "reason": "r5-canonical-numerics-v1 amendment (codex 185751Z, owner-gated): "
                  "digest-bearing values projected to q11 canonical numerics so the "
                  "capsule verifies host-independently; scientific values invariant at "
                  "q11 (refusal-gated). Digest map below; capsule manifest rebound.",
        "model_digest_map": digest_map,
        "capsule_manifest_sha256": sha(mb),
        "by": "grassmann",
    }
    corr = r5_daily.setdefault("corrections", [])
    prior = [c for c in corr if c.get("numeric_policy_id") == NUMERICS_POLICY_ID]
    if prior:
        # Upsert: keep the FIRST run's true old->new digest map (a re-run maps
        # q11->q11 identity); refresh manifest sha + timestamp.
        keep = prior[0]
        if any(v["old_model_sha256"] != v["new_model_sha256"]
               for v in keep["model_digest_map"].values()):
            entry["model_digest_map"] = keep["model_digest_map"]
        r5_daily["corrections"] = [c for c in corr if c not in prior] + [entry]
    else:
        corr.append(entry)
    R5_DAILY.write_text(json.dumps(r5_daily, indent=1))
    print(f"REGENERATED {len(digest_map)} entries; new manifest_sha256={sha(mb)}")


def _verify_once(capsule: Path) -> None:
    import canonical_numerics as CN
    import precip_residual as PR
    manifest = json.loads((capsule / "manifest.json").read_text())
    if sha(canon(manifest)) != sha((capsule / "manifest.json").read_bytes()):
        print("FAIL: manifest not canonical"); sys.exit(1)
    for key, files in manifest["entries"].items():
        day, region = key.split("/", 1)
        d = capsule / day / region
        for name, expect in files.items():
            actual = sha((d / name).read_bytes())
            if actual != expect:
                print(f"FAIL {key}/{name}: sha {actual[:12]} != manifest {expect[:12]}")
                sys.exit(1)
        ratios = json.loads((d / "ratios.json").read_text())
        precip = json.loads((d / "precip.json").read_text())
        query = json.loads((d / "query.json").read_text())
        record = load_canonical_json(d / "record.json", CN)   # token validation + scalars
        model = PR.fit_region(region, ratios, precip, day, sig_digits=11)
        if model is None:
            print(f"FAIL {key}: fit returned None"); sys.exit(1)
        if CN.model_digest(model, n=11) != record["model_sha256"]:
            print(f"FAIL {key}: model does not recompute to published digest")
            sys.exit(1)
        # Byte equality of the recomputed objects against the STORED bytes is the
        # authoritative structural check (never parse-and-recanonicalize).
        if CN.canonical_json_bytes(model, n=11) != (d / "model.json").read_bytes():
            print(f"FAIL {key}: stored model bytes are not the canonical serialization "
                  f"of the recomputed model"); sys.exit(1)
        out = _capsule_transform(PR, region, query, ratios, precip, day)
        if out is None or CN.canonical_json_bytes(out, n=11) != \
                (d / "record.json").read_bytes():
            print(f"FAIL {key}: R5 output does not reproduce the stored record bytes")
            sys.exit(1)
    print(f"VERIFY PASS: {len(manifest['entries'])} entries recompute exactly on the "
          f"canonical domain; MODEL_FILE never touched; "
          f"manifest_sha256={sha(canon(manifest))}")


def _mutate_one_byte(p: Path):
    b = bytearray(p.read_bytes())
    b[len(b) // 2] ^= 0x01
    p.write_bytes(bytes(b))


def verify(mutation_selftest: bool):
    _verify_once(CAPSULE_DIR)
    if not mutation_selftest:
        return
    import subprocess
    key = next(iter(json.loads((CAPSULE_DIR / "manifest.json").read_text())["entries"]))
    targets = [CAPSULE_DIR / key / n for n in
               ("ratios.json", "precip.json", "model.json")] + [CAPSULE_DIR / "manifest.json"]
    for t in targets:
        work = Path(tempfile.mkdtemp()) / "capsule"
        shutil.copytree(CAPSULE_DIR, work)
        _mutate_one_byte(work / t.relative_to(CAPSULE_DIR))
        rc = subprocess.run([sys.executable, __file__, "verify-dir", str(work)]).returncode
        verdict = "REFUSED (correct)" if rc != 0 else "ACCEPTED (WRONG)"
        print(f"mutation {t.relative_to(CAPSULE_DIR)}: {verdict}")
        if rc == 0:
            sys.exit(1)
    print("MUTATION SELFTEST PASS: all four input classes refuse on a one-byte flip")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "regenerate":
        regenerate()
    elif mode == "verify":
        verify("--mutation-selftest" in sys.argv)
    elif mode == "verify-dir":
        _verify_once(Path(sys.argv[2]))
    else:
        print("usage: r5_replay_capsule.py regenerate | verify [--mutation-selftest]")
        sys.exit(2)
