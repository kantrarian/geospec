#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R5 replay-reproducibility capsule (codex 172526Z WORKS-WITH-FIX repair).

generate: from the producer host, preserve the EXACT bytes consumed for each
  regenerated day/region of 859f1db -- full ratio-history dict as returned by
  load_ratio_history, full precipitation dict as returned by (cache-backed)
  fetch_precip plus query/source metadata incl. the target raw ratio, the
  canonical fitted-model JSON, and the emitted R5 record -- and a manifest
  binding every file SHA-256 to the code commit, day, and region. Refuses to
  write a capsule whose recomputed model_sha256 differs from the published one.

verify: clean-state verifier per the codex contract:
  1. rebuilds each model with fit_region from capsule inputs ONLY
     (fetch_precip/load_ratio_history are replaced by capsule readers);
  2. recomputes the canonical model digest AND the full R5 output byte-for-byte
     against the capsule record;
  3. proves MODEL_FILE is neither read nor created (redirected to a tempdir,
     asserted absent after every call);
  4. self-test: refuses one-byte mutations of the ratio input, precipitation
     input, model body, and manifest (run with `verify --mutation-selftest`).
"""
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

# Repo root derived from this file's location (tools/r5_replay_capsule.py) so the
# clean-state verifier runs from any clone.
REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "monitoring" / "src"
sys.path.insert(0, str(SRC))

CAPSULE_DIR = REPO / "monitoring" / "replay_capsules" / "r5_20260812_20260813"
DAYS = ["2026-08-12", "2026-08-13"]
CODE_COMMIT = "ee77902"


def canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def generate():
    import precip_residual as PR
    from validate_predictions import REGION_DEFINITIONS

    r5_daily = json.loads((REPO / "docs" / "r5_daily.json").read_text())
    if CAPSULE_DIR.exists():
        shutil.rmtree(CAPSULE_DIR)
    manifest = {"schema": "geospec-r5-replay-capsule-v1", "code_commit": CODE_COMMIT,
                "days": DAYS, "entries": {}}
    for day in DAYS:
        for region, rec in r5_daily["days"][day].items():
            if not (isinstance(rec, dict) and rec.get("r5_computed")):
                continue
            center = REGION_DEFINITIONS[region]["center"]
            ratios = PR.load_ratio_history(region)
            precip = PR.fetch_precip(region, center[0], center[1], day)
            model = PR.fit_region(region, ratios, precip, day)
            if model is None:
                print(f"REFUSE {day}/{region}: fit ineligible on current inputs"); sys.exit(1)
            model_sha = sha(canon(model))
            if model_sha != rec["model_sha256"]:
                print(f"REFUSE {day}/{region}: recomputed model_sha {model_sha[:12]} != "
                      f"published {rec['model_sha256'][:12]} -- inputs drifted, "
                      f"value-change disclosure required"); sys.exit(1)
            d = CAPSULE_DIR / day / region
            d.mkdir(parents=True)
            files = {
                "ratios.json": canon(ratios),
                "precip.json": canon(precip),
                "query.json": canon({"region": region, "lat": center[0], "lon": center[1],
                                     "end": day, "raw_ratio": rec["raw_ratio"],
                                     "source": "open-meteo archive via cache-backed "
                                               "fetch_precip (monitoring/data/precip_cache)"}),
                "model.json": canon(model),
                "record.json": canon(rec),
            }
            entry = {}
            for name, b in files.items():
                (d / name).write_bytes(b)
                entry[name] = sha(b)
            manifest["entries"][f"{day}/{region}"] = entry
            print(f"OK {day}/{region}: model_sha MATCHES published ({model_sha[:12]})")
    mb = canon(manifest)
    (CAPSULE_DIR / "manifest.json").write_bytes(mb)
    print(f"CAPSULE {len(manifest['entries'])} entries; manifest_sha256={sha(mb)}")


def _verify_once(capsule: Path) -> None:
    """Raises SystemExit(1) on ANY mismatch."""
    import precip_residual as PR
    manifest = json.loads((capsule / "manifest.json").read_text())
    if sha(canon(manifest)) != sha((capsule / "manifest.json").read_bytes()):
        print("FAIL: manifest not canonical"); sys.exit(1)
    tmp = Path(tempfile.mkdtemp())
    old_model_file = PR.MODEL_FILE
    PR.MODEL_FILE = tmp / "must_never_exist.json"
    old_fetch, old_hist = PR.fetch_precip, PR.load_ratio_history
    try:
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
            record = json.loads((d / "record.json").read_text())
            model = PR.fit_region(region, ratios, precip, day)
            if model is None or sha(canon(model)) != record["model_sha256"]:
                print(f"FAIL {key}: model does not recompute to published digest")
                sys.exit(1)
            PR.fetch_precip = lambda *a, _p=precip, **k: dict(_p)
            PR.load_ratio_history = lambda *a, _r=ratios, **k: dict(_r)
            out = PR.r5_transform(region, query["lat"], query["lon"],
                                  query["raw_ratio"], day, historical=True)
            if out is None or canon(out) != canon(record):
                print(f"FAIL {key}: R5 output does not reproduce byte-for-byte")
                sys.exit(1)
            if PR.MODEL_FILE.exists():
                print(f"FAIL {key}: MODEL_FILE was created on the historical path")
                sys.exit(1)
        print(f"VERIFY PASS: {len(manifest['entries'])} entries recompute byte-for-byte; "
              f"MODEL_FILE never touched; manifest_sha256={sha(canon(manifest))}")
    finally:
        PR.MODEL_FILE = old_model_file
        PR.fetch_precip, PR.load_ratio_history = old_fetch, old_hist


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
    if mode == "generate":
        generate()
    elif mode == "verify":
        verify("--mutation-selftest" in sys.argv)
    elif mode == "verify-dir":
        _verify_once(Path(sys.argv[2]))
    else:
        print("usage: r5_replay_capsule.py generate | verify [--mutation-selftest]")
        sys.exit(2)
