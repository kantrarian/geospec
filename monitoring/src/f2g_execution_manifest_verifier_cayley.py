#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 EXECUTION-MANIFEST v1 executable verifier (cayley).
Schema contract: docs/f2g_window2_execution/execution_manifest_schema_v1.md.

Reads the execution manifest FROM GIT OBJECTS ONLY at the stated
commit; verifies top-field closure, target/ancestor relations, repo
identity, the DESIGN LINKAGE (design manifest blob sha + target echo +
a full design-pin walk via the pinned design verifier, whose executed
bytes are attested and compared CRLF->LF-normalized against its BOUND
pin), the closed 10-slot set, per-slot schema/status coherence, every
BOUND pin (ancestor + last-touch-at-target + blob sha), the bars family
set, and manifest_state consistency. `--prestart` additionally refuses
every OPEN slot. Typed refusals only; a refusal without its typed
reason is a DEFECT (KAT-enforced).

A PASS authorizes NOTHING by itself: no PRESTART (codex's round over
the bound bytes is separate), no fire, no prospective-value access.

Usage:
  f2g_execution_manifest_verifier_cayley.py <repo> <manifest-commit>
      [--prestart] [--kat]
"""
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

MANIFEST_PATH = "docs/f2g_window2_execution/execution_manifest.json"
DESIGN_MANIFEST_PATH = "docs/f2g_window2_freeze/byte_pin_manifest.json"
SCHEMA = "f2g-window2-execution-manifest-v1.2"
TOP_FIELDS = {"schema", "generated_utc", "repository_url",
              "execution_target_commit", "target_ref",
              "design_manifest_commit", "design_manifest_blob_sha256",
              "design_target_commit", "manifest_state", "slots"}
SLOT_SET = {"execution_generator", "execution_verifier",
            "design_pin_verifier", "selection_impl", "adapter_impl",
            "accrual_impl", "mag_capsules", "calibration_ledgers",
            "bars",
            # v1.2 (codex 1400Z ruling 2): producer_code RENAMED to
            # the staged-envelope trust boundary
            "producer_boundary",
            # v1.1: the repaired execution tools (codex 1358Z)
            "power_harness", "calibration_runner"}
SLOT_FIELDS = {"status", "owner", "note", "pins"}
PIN_FIELDS = {"path", "commit", "blob_sha256"}
REQUIRED_BAR_FAMILIES = {"W-SEL", "W-CAS", "W-B2B", "W-B1B", "W-MF4",
                         "W-MAG", "W-BARRIER", "W-PIN"}
# v1.2 producer boundary (codex 1400Z ruling 2 + amendment v1): the
# only registered mode, and the pin classes a BOUND slot MUST cover
# ("a note string or empty pin set can never turn the slot BOUND")
PRODUCER_BOUNDARY_MODE = "staged_envelope"
PRODUCER_AMENDMENT_PATH = ("docs/f2g_window2_execution/"
                           "producer_boundary_amendment_v1.md")
PRODUCER_ENVELOPE_PREFIX = ("docs/f2g_window2_execution/"
                            "staged_envelopes/")
DPV_PATH = "monitoring/src/f2g_design_pin_verifier_cayley.py"
REPO_IDENTITY = "kantrarian/geospec"

_DESIGN_WALK_MEMO = {}


def _run(repo, args, binary=False):
    p = subprocess.run(["git", "-C", repo] + args, capture_output=True)
    out = p.stdout if binary else p.stdout.decode("utf-8",
                                                  "replace").strip()
    return p.returncode, out


def _is_ancestor(repo, a, b):
    rc, _ = _run(repo, ["merge-base", "--is-ancestor", a, b])
    return rc == 0


def _norm_repo_identity(url):
    u = url.strip().lower()
    if u.endswith(".git"):
        u = u[:-4]
    u = u.replace(":", "/")
    parts = [p for p in u.split("/") if p]
    return "/".join(parts[-2:]) if len(parts) >= 2 else u


def _norm_py(raw):
    return raw.replace(b"\r\n", b"\n")


def _refuse(res, reason, where, detail):
    res["typed_reasons"].append({"reason": reason, "where": where,
                                 "detail": detail})


def check_executed_design_verifier(repo, pinned_sha, res):
    """Compare the on-disk design verifier (the bytes we are about to
    EXECUTE) against its BOUND pin, CRLF->LF normalized; attest the
    executed sha. Returns the disk path or None on refusal."""
    disk = os.path.join(repo, DPV_PATH.replace("/", os.sep))
    if not os.path.exists(disk):
        _refuse(res, "EXECUTED_BYTES_MISMATCH", "design_pin_verifier",
                "design verifier absent on disk")
        return None
    with open(disk, "rb") as f:
        got = hashlib.sha256(_norm_py(f.read())).hexdigest()
    res["executed_design_verifier_sha256"] = got
    if got != pinned_sha:
        _refuse(res, "EXECUTED_BYTES_MISMATCH", "design_pin_verifier",
                f"disk(norm)={got[:12]} pinned={pinned_sha[:12]}")
        return None
    return disk


def run_design_walk(repo, disk_path, design_manifest_commit):
    memo_key = design_manifest_commit
    if memo_key in _DESIGN_WALK_MEMO:
        return _DESIGN_WALK_MEMO[memo_key]
    spec = importlib.util.spec_from_file_location("f2g_dpv", disk_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.verify(repo, design_manifest_commit)
    _DESIGN_WALK_MEMO[memo_key] = out
    return out


def _verify_obj(repo, manifest_commit, obj, prestart=False):
    res = {"verdict": None, "typed_reasons": [], "mode":
           "prestart" if prestart else "default",
           "manifest_commit": manifest_commit, "slots_bound": 0,
           "slots_open": 0, "pins_checked": 0}

    if not isinstance(obj, dict) or obj.get("schema") != SCHEMA:
        _refuse(res, "MANIFEST_SCHEMA_MISMATCH", None,
                f"schema={obj.get('schema') if isinstance(obj, dict) else type(obj).__name__}")
        res["verdict"] = "REFUSE"
        return res
    missing = TOP_FIELDS - set(obj)
    if missing:
        _refuse(res, "TOP_FIELD_MISSING", None, sorted(missing))
        res["verdict"] = "REFUSE"
        return res

    target = obj["execution_target_commit"]
    rc, target_full = _run(repo, ["rev-parse", f"{target}^{{commit}}"])
    if rc != 0 or not target_full:
        _refuse(res, "TARGET_UNRESOLVABLE", None, target)
        res["verdict"] = "REFUSE"
        return res
    if not _is_ancestor(repo, target_full, manifest_commit):
        _refuse(res, "TARGET_NOT_ANCESTOR_OF_MANIFEST_COMMIT", None,
                f"{target_full[:12]} !~> {manifest_commit[:12]}")

    rc, origin = _run(repo, ["config", "--get", "remote.origin.url"])
    declared = _norm_repo_identity(obj["repository_url"])
    if declared != REPO_IDENTITY or (
            rc == 0 and origin
            and _norm_repo_identity(origin) != declared):
        _refuse(res, "REPO_IDENTITY_MISMATCH", None,
                f"declared={declared}")

    # --- design linkage ---
    dm = obj["design_manifest_commit"]
    rc, dm_full = _run(repo, ["rev-parse", f"{dm}^{{commit}}"])
    dm_obj = None
    if rc != 0 or not dm_full:
        _refuse(res, "DESIGN_COMMIT_UNRESOLVABLE", None, dm)
    else:
        if not _is_ancestor(repo, dm_full, target_full):
            _refuse(res, "DESIGN_NOT_ANCESTOR", None,
                    f"{dm_full[:12]} !~> {target_full[:12]}")
        rc, dm_blob = _run(repo, ["cat-file", "blob",
                                  f"{dm_full}:{DESIGN_MANIFEST_PATH}"],
                           binary=True)
        if rc != 0:
            _refuse(res, "DESIGN_BLOB_SHA_MISMATCH", None,
                    "design manifest blob absent at commit")
        else:
            got = hashlib.sha256(dm_blob).hexdigest()
            if got != obj["design_manifest_blob_sha256"]:
                _refuse(res, "DESIGN_BLOB_SHA_MISMATCH", None,
                        f"recorded={obj['design_manifest_blob_sha256'][:12]} got={got[:12]}")
            else:
                dm_obj = json.loads(dm_blob.decode("utf-8"))
                if dm_obj.get("design_target_commit") != \
                        obj["design_target_commit"]:
                    _refuse(res, "DESIGN_TARGET_INCONSISTENT", None,
                            f"declared={obj['design_target_commit'][:12]} in-manifest={str(dm_obj.get('design_target_commit'))[:12]}")

    # --- slots ---
    slots = obj["slots"]
    if not isinstance(slots, dict) or set(slots) != SLOT_SET:
        delta = (set(slots) ^ SLOT_SET) if isinstance(slots, dict) \
            else {"NOT_DICT"}
        _refuse(res, "SLOT_SET_NOT_CLOSED", None, sorted(delta))
        res["verdict"] = "REFUSE"
        return res

    dpv_pin_sha = None
    for name in sorted(slots):
        slot = slots[name]
        extra = {"families"} if name == "bars" else set()
        if name == "producer_boundary":
            extra = {"boundary_mode"}
        fields = SLOT_FIELDS | extra
        if not isinstance(slot, dict) or not \
                SLOT_FIELDS <= set(slot) or not set(slot) <= fields:
            _refuse(res, "SLOT_SCHEMA_NOT_CLOSED", name,
                    sorted(set(slot) ^ SLOT_FIELDS)
                    if isinstance(slot, dict) else ["NOT_DICT"])
            continue
        if name == "producer_boundary" and \
                slot.get("boundary_mode") != PRODUCER_BOUNDARY_MODE:
            _refuse(res, "PRODUCER_BOUNDARY_MODE_UNREGISTERED", name,
                    f"boundary_mode={slot.get('boundary_mode')!r} "
                    f"!= {PRODUCER_BOUNDARY_MODE!r}")
            continue
        status = slot["status"]
        if status == "BOUND":
            res["slots_bound"] += 1
            if not slot["pins"]:
                _refuse(res, "SLOT_BOUND_WITHOUT_PINS", name, "")
                continue
            if name == "bars":
                fams = set(slot.get("families", []))
                if fams != REQUIRED_BAR_FAMILIES:
                    _refuse(res, "BARS_FAMILY_SET_MISMATCH", name,
                            sorted(fams ^ REQUIRED_BAR_FAMILIES))
            if name == "producer_boundary":
                paths = [p.get("path") for p in slot["pins"]
                         if isinstance(p, dict)]
                have_amend = PRODUCER_AMENDMENT_PATH in paths
                have_code = any(str(p).startswith("monitoring/src/")
                                for p in paths)
                # codex 2235Z item 1: an inventory/descriptor under
                # staged_envelopes/ is NEVER an envelope record --
                # only actual .record.json envelopes satisfy the
                # class
                have_env = any(str(p).startswith(
                    PRODUCER_ENVELOPE_PREFIX)
                    and str(p).endswith(".record.json")
                    for p in paths)
                # codex 0238Z item 1: the REGISTERED expected-keys
                # authority is a required pin class
                have_auth = any(str(p).endswith(
                    "staged_expected_contracts_v2.json")
                    for p in paths)
                if not (have_amend and have_code and have_env
                        and have_auth):
                    _refuse(res, "PRODUCER_BOUNDARY_PINS_INCOMPLETE",
                            name,
                            f"amendment={have_amend} code={have_code} "
                            f"envelopes={have_env} "
                            f"authority={have_auth}")
            for p in slot["pins"]:
                res["pins_checked"] += 1
                if not isinstance(p, dict) or set(p) != PIN_FIELDS:
                    _refuse(res, "PIN_SCHEMA_NOT_CLOSED", name,
                            sorted(set(p) ^ PIN_FIELDS)
                            if isinstance(p, dict) else ["NOT_DICT"])
                    continue
                rc, last = _run(repo, ["log", "-1", "--format=%H",
                                       target_full, "--", p["path"]])
                if rc != 0 or not last:
                    _refuse(res, "PATH_NOT_AT_TARGET", name, p["path"])
                    continue
                if not _is_ancestor(repo, p["commit"], target_full):
                    _refuse(res, "NON_ANCESTOR_PIN", name,
                            f"{p['commit'][:12]} !~> {target_full[:12]}")
                if last != p["commit"]:
                    _refuse(res, "LAST_TOUCH_MISMATCH", name,
                            f"recorded={p['commit'][:12]} at-target={last[:12]}")
                rc, blob = _run(repo, ["cat-file", "blob",
                                       f"{p['commit']}:{p['path']}"],
                                binary=True)
                if rc != 0:
                    _refuse(res, "BLOB_MISSING", name,
                            f"{p['commit'][:12]}:{p['path']}")
                    continue
                got = hashlib.sha256(blob).hexdigest()
                if got != p["blob_sha256"]:
                    _refuse(res, "BLOB_SHA_MISMATCH", name,
                            f"recorded={p['blob_sha256'][:12]} got={got[:12]}")
                elif name == "design_pin_verifier" \
                        and p["path"] == DPV_PATH:
                    dpv_pin_sha = got
        elif status == "OPEN":
            res["slots_open"] += 1
            if slot["pins"]:
                _refuse(res, "SLOT_OPEN_WITH_PINS", name,
                        f"{len(slot['pins'])} pins")
            if prestart:
                _refuse(res, "SLOT_OPEN", name,
                        f"owner={slot['owner']}")
        else:
            _refuse(res, "SLOT_SCHEMA_NOT_CLOSED", name,
                    f"status={status}")

    want = "CLOSED" if res["slots_open"] == 0 else "OPEN"
    if obj["manifest_state"] != want:
        _refuse(res, "MANIFEST_STATE_WRONG", None,
                f"declared={obj['manifest_state']} computed={want}")

    # --- design walk (only meaningful once linkage + dpv pin are ok) ---
    if dm_obj is not None and dpv_pin_sha is not None:
        disk = check_executed_design_verifier(repo, dpv_pin_sha, res)
        if disk is not None:
            walk = run_design_walk(repo, disk, dm_full)
            res["design_walk"] = {"verdict": walk["verdict"],
                                  "pins_checked":
                                      walk.get("pins_checked", 0)}
            if walk["verdict"] != "PASS":
                _refuse(res, "DESIGN_WALK_FAILED", None,
                        json.dumps(walk["typed_reasons"])[:300])

    res["verdict"] = "PASS" if not res["typed_reasons"] else "REFUSE"
    return res


def verify(repo, manifest_commit, prestart=False):
    rc, full = _run(repo, ["rev-parse", f"{manifest_commit}^{{commit}}"])
    if rc != 0 or not full:
        return {"verdict": "REFUSE", "manifest_commit": manifest_commit,
                "typed_reasons": [{"reason":
                                   "MANIFEST_COMMIT_UNRESOLVABLE",
                                   "where": None,
                                   "detail": manifest_commit}]}
    rc, raw = _run(repo, ["cat-file", "blob", f"{full}:{MANIFEST_PATH}"],
                   binary=True)
    if rc != 0:
        return {"verdict": "REFUSE", "manifest_commit": full,
                "typed_reasons": [{"reason": "MANIFEST_NOT_IN_COMMIT",
                                   "where": None,
                                   "detail": f"{full[:12]}:{MANIFEST_PATH}"}]}
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        return {"verdict": "REFUSE", "manifest_commit": full,
                "typed_reasons": [{"reason": "MANIFEST_UNPARSEABLE",
                                   "where": None, "detail": str(e)}]}
    res = _verify_obj(repo, full, obj, prestart=prestart)
    res["manifest_blob_sha256"] = hashlib.sha256(raw).hexdigest()
    return res


def _has(res, reason):
    return any(r["reason"] == reason for r in res["typed_reasons"])


def kat(repo, manifest_commit):
    rc, full = _run(repo, ["rev-parse", f"{manifest_commit}^{{commit}}"])
    assert rc == 0
    rc, raw = _run(repo, ["cat-file", "blob", f"{full}:{MANIFEST_PATH}"],
                   binary=True)
    assert rc == 0, "real execution manifest missing at commit"
    base = json.loads(raw.decode("utf-8"))
    failures = []

    def case(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'DEFECT'}] {name}"
              + (f" -- {detail}" if detail and not ok else ""))
        if not ok:
            failures.append(name)

    # 1/2. positive default PASS; prestart refuses the OPEN slots
    r = verify(repo, full)
    case("positive-default",
         r["verdict"] == "PASS" and not r["typed_reasons"],
         json.dumps(r["typed_reasons"]))
    r = verify(repo, full, prestart=True)
    open_n = sum(1 for s in base["slots"].values()
                 if s["status"] == "OPEN")
    case("prestart-refuses-open",
         r["verdict"] == "REFUSE" and sum(
             1 for t in r["typed_reasons"]
             if t["reason"] == "SLOT_OPEN") == open_n,
         json.dumps(r["typed_reasons"])[:200])

    # 3. schema doctored
    d = copy.deepcopy(base)
    d["schema"] = "wrong"
    case("schema-mismatch",
         _has(_verify_obj(repo, full, d), "MANIFEST_SCHEMA_MISMATCH"))

    # 4. top field missing
    d = copy.deepcopy(base)
    del d["design_manifest_commit"]
    case("top-field-missing",
         _has(_verify_obj(repo, full, d), "TOP_FIELD_MISSING"))

    # 5. target not ancestor (real child sha vs parent as anchor)
    rc, parent = _run(repo, ["rev-parse", f"{full}^"])
    d = copy.deepcopy(base)
    d["execution_target_commit"] = full
    case("target-not-ancestor",
         rc == 0 and _has(_verify_obj(repo, parent, d),
                          "TARGET_NOT_ANCESTOR_OF_MANIFEST_COMMIT"))

    # 6. design blob sha doctored
    d = copy.deepcopy(base)
    s = d["design_manifest_blob_sha256"]
    d["design_manifest_blob_sha256"] = \
        ("0" if s[0] != "0" else "1") + s[1:]
    case("design-blob-sha-mismatch",
         _has(_verify_obj(repo, full, d), "DESIGN_BLOB_SHA_MISMATCH"))

    # 7. design target echo doctored
    d = copy.deepcopy(base)
    d["design_target_commit"] = "f" * 40
    case("design-target-inconsistent",
         _has(_verify_obj(repo, full, d), "DESIGN_TARGET_INCONSISTENT"))

    # 8. slot set not closed
    d = copy.deepcopy(base)
    del d["slots"]["bars"]
    case("slot-set-not-closed",
         _has(_verify_obj(repo, full, d), "SLOT_SET_NOT_CLOSED"))

    # 9. bound without pins
    d = copy.deepcopy(base)
    d["slots"]["design_pin_verifier"]["pins"] = []
    case("slot-bound-without-pins",
         _has(_verify_obj(repo, full, d), "SLOT_BOUND_WITHOUT_PINS"))

    # 10. open with pins (pick an OPEN slot dynamically -- slots flip
    # BOUND over the manifest's life)
    open_name = next(n for n in sorted(base["slots"])
                     if base["slots"][n]["status"] == "OPEN")
    d = copy.deepcopy(base)
    d["slots"][open_name]["pins"] = \
        copy.deepcopy(base["slots"]["design_pin_verifier"]["pins"])
    case("slot-open-with-pins",
         _has(_verify_obj(repo, full, d), "SLOT_OPEN_WITH_PINS"))

    # 11. bars family set mismatch (BOUND with a real pin, one family
    # missing)
    d = copy.deepcopy(base)
    d["slots"]["bars"] = {
        "status": "BOUND", "owner": "grassmann", "note": "kat",
        "pins": copy.deepcopy(
            base["slots"]["design_pin_verifier"]["pins"]),
        "families": sorted(REQUIRED_BAR_FAMILIES - {"W-PIN"})}
    case("bars-family-mismatch",
         _has(_verify_obj(repo, full, d), "BARS_FAMILY_SET_MISMATCH"))

    # 12. pin blob sha doctored
    d = copy.deepcopy(base)
    p = d["slots"]["design_pin_verifier"]["pins"][0]
    p["blob_sha256"] = ("0" if p["blob_sha256"][0] != "0" else "1") \
        + p["blob_sha256"][1:]
    case("pin-blob-sha-mismatch",
         _has(_verify_obj(repo, full, d), "BLOB_SHA_MISMATCH"))

    # 13. non-ancestor pin (manifest commit is a descendant of target)
    d = copy.deepcopy(base)
    d["slots"]["execution_generator"]["pins"][0]["commit"] = full
    case("non-ancestor-pin",
         _has(_verify_obj(repo, full, d), "NON_ANCESTOR_PIN"))

    # 14. manifest_state doctored
    d = copy.deepcopy(base)
    d["manifest_state"] = "CLOSED"
    case("manifest-state-wrong",
         _has(_verify_obj(repo, full, d), "MANIFEST_STATE_WRONG"))

    # 15. executed-bytes mismatch (unit: wrong pinned sha for the disk
    # design verifier)
    r = {"typed_reasons": []}
    check_executed_design_verifier(repo, "0" * 64, r)
    case("executed-bytes-mismatch",
         any(t["reason"] == "EXECUTED_BYTES_MISMATCH"
             for t in r["typed_reasons"]))

    # 16. dirty-disk independence (doctored working-tree manifest;
    # git-object verdict unchanged; tree restored byte-exactly)
    disk = os.path.join(repo, MANIFEST_PATH.replace("/", os.sep))
    saved = None
    if os.path.exists(disk):
        with open(disk, "rb") as f:
            saved = f.read()
    try:
        with open(disk, "w", encoding="utf-8") as f:
            f.write('{"schema": "DOCTORED-ON-DISK"}\n')
        r = verify(repo, full)
        case("dirty-disk-ignored",
             r["verdict"] == "PASS" and not r["typed_reasons"],
             json.dumps(r["typed_reasons"]))
    finally:
        if saved is not None:
            with open(disk, "wb") as f:
                f.write(saved)

    # 17. producer boundary_mode absent -> refuse (v1.2)
    d = copy.deepcopy(base)
    del d["slots"]["producer_boundary"]["boundary_mode"]
    case("producer-mode-absent",
         _has(_verify_obj(repo, full, d),
              "PRODUCER_BOUNDARY_MODE_UNREGISTERED"))

    # 18. producer boundary_mode divergent -> refuse
    d = copy.deepcopy(base)
    d["slots"]["producer_boundary"]["boundary_mode"] = \
        "acquisition_code"
    case("producer-mode-divergent",
         _has(_verify_obj(repo, full, d),
              "PRODUCER_BOUNDARY_MODE_UNREGISTERED"))

    # 19. producer BOUND with pins that miss the registered classes
    # (a real pin, but neither amendment nor envelope records) ->
    # refuse; a note string or pin count can never bind the boundary
    d = copy.deepcopy(base)
    d["slots"]["producer_boundary"] = {
        "status": "BOUND", "owner": "grassmann", "note": "kat",
        "boundary_mode": PRODUCER_BOUNDARY_MODE,
        "pins": copy.deepcopy(
            base["slots"]["design_pin_verifier"]["pins"])}
    case("producer-pins-incomplete",
         _has(_verify_obj(repo, full, d),
              "PRODUCER_BOUNDARY_PINS_INCOMPLETE"))

    # 20. codex 2235Z item 1 lock: inventory + descriptor under
    # staged_envelopes/ (plus amendment + code) can NEVER satisfy the
    # envelope-record class -- only .record.json envelopes count
    d = copy.deepcopy(base)
    real_pin = copy.deepcopy(
        base["slots"]["design_pin_verifier"]["pins"][0])
    pins = []
    for path in (PRODUCER_AMENDMENT_PATH,
                 "monitoring/src/w2_producer_grassmann.py",
                 PRODUCER_ENVELOPE_PREFIX
                 + "staged_body_inventory.json",
                 PRODUCER_ENVELOPE_PREFIX + "store_descriptor.json"):
        p = copy.deepcopy(real_pin)
        p["path"] = path
        pins.append(p)
    d["slots"]["producer_boundary"] = {
        "status": "BOUND", "owner": "grassmann", "note": "kat",
        "boundary_mode": PRODUCER_BOUNDARY_MODE, "pins": pins}
    case("producer-inventory-not-envelope",
         _has(_verify_obj(repo, full, d),
              "PRODUCER_BOUNDARY_PINS_INCOMPLETE"))

    print(f"KAT: {20 - len(failures)}/20 pass"
          + (f"; DEFECTS: {failures}" if failures else ""))
    return not failures


def main():
    repo = os.path.abspath(sys.argv[1])
    manifest_commit = sys.argv[2]
    flags = sys.argv[3:]
    with open(os.path.abspath(__file__), "rb") as f:
        self_sha = hashlib.sha256(_norm_py(f.read())).hexdigest()
    if "--kat" in flags:
        if not kat(repo, manifest_commit):
            print("KAT_DEFECT: matrix failed; verdict withheld")
            sys.exit(2)
    res = verify(repo, manifest_commit, prestart="--prestart" in flags)
    res["verifier_source_sha256_normalized"] = self_sha
    res["verified_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                        time.gmtime())
    res["authorizes"] = ("NOTHING by itself: no PRESTART (codex round "
                         "over bound bytes is separate), no fire, no "
                         "prospective-value access")
    print(json.dumps(res, indent=1, sort_keys=True))
    sys.exit(0 if res["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
