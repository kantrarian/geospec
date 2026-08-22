#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 DESIGN-PIN executable verifier (cayley; PRESTART deliverable
declared in design-manifest-v2.1 manifest_class).

Reopens EVERY pin of the committed byte_pin_manifest.json FROM GIT
OBJECTS ONLY (never the working tree), against the manifest's own
declared design_target_commit, and refuses on absence, byte mismatch,
non-ancestor pins, last-touch retarget, open pin schema, unknown
section rule, repository-identity mismatch, and unlisted freeze
artifacts. Typed refusals only; a refusal without its typed reason is
a DEFECT (KAT-enforced).

This closes the DESIGN-pin integrity walk as an executable check. It
does NOT close execution-manifest-v1 (selection/adapter/accrual
implementations, bars, producer code, calibration ledgers) -- that
remains a separate REQUIRED PRESTART deliverable, and no PASS here
authorizes PRESTART, fire, or any prospective-value access.

Usage:
  f2g_design_pin_verifier_cayley.py <repo> <manifest-commit>
  f2g_design_pin_verifier_cayley.py <repo> <manifest-commit> --kat
"""
import copy
import hashlib
import json
import os
import subprocess
import sys
import time

MANIFEST_PATH = "docs/f2g_window2_freeze/byte_pin_manifest.json"
SCHEMA = "f2g-window2-design-manifest-v2.1"
FREEZE_DIR = "docs/f2g_window2_freeze"
TOP_FIELDS = {"schema", "generated_utc", "repository_url",
              "design_target_commit", "target_ref", "pin_count",
              "manifest_class", "pins"}
PIN_FIELDS = {"path", "commit", "blob_sha256", "object_id",
              "imported_section", "imported_section_sha256"}
OBJECT_ID = {".md": "markdown-contract", ".json": "json-object",
             ".py": "python-source", ".txt": "fdsn-station-text"}
REPO_IDENTITY = "kantrarian/geospec"


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


def load_manifest_bytes(repo, manifest_commit):
    rc, raw = _run(repo, ["cat-file", "blob",
                          f"{manifest_commit}:{MANIFEST_PATH}"],
                   binary=True)
    if rc != 0:
        return None
    return raw


def _refuse(res, reason, pin, detail):
    res["typed_reasons"].append({"reason": reason, "pin": pin,
                                 "detail": detail})


def _verify_obj(repo, manifest_commit, obj):
    """Verify a parsed manifest object. All reads via git objects; the
    working tree is never consulted. Collects ALL typed reasons rather
    than stopping at the first."""
    res = {"verdict": None, "typed_reasons": [], "pins_checked": 0,
           "manifest_commit": manifest_commit}

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

    target = obj["design_target_commit"]
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
                f"declared={declared} origin={_norm_repo_identity(origin) if origin else 'ABSENT'}")

    pins = obj["pins"]
    if not isinstance(pins, dict) or obj["pin_count"] != len(pins):
        _refuse(res, "PIN_COUNT_MISMATCH", None,
                f"pin_count={obj['pin_count']} len={len(pins) if isinstance(pins, dict) else 'NOT_DICT'}")

    pinned_paths = set()
    for key in sorted(pins):
        pin = pins[key]
        res["pins_checked"] += 1
        if not isinstance(pin, dict) or set(pin) != PIN_FIELDS:
            delta = (set(pin) ^ PIN_FIELDS) if isinstance(pin, dict) \
                else {"NOT_DICT"}
            _refuse(res, "PIN_SCHEMA_NOT_CLOSED", key, sorted(delta))
            continue
        path = pin["path"]
        pinned_paths.add(path)

        rc, last = _run(repo, ["log", "-1", "--format=%H",
                               target_full, "--", path])
        if rc != 0 or not last:
            _refuse(res, "PATH_NOT_AT_TARGET", key, path)
            continue
        if not _is_ancestor(repo, pin["commit"], target_full):
            _refuse(res, "NON_ANCESTOR_PIN", key,
                    f"{pin['commit'][:12]} !~> {target_full[:12]}")
        if last != pin["commit"]:
            _refuse(res, "LAST_TOUCH_MISMATCH", key,
                    f"recorded={pin['commit'][:12]} at-target={last[:12]}")

        rc, blob = _run(repo, ["cat-file", "blob",
                               f"{pin['commit']}:{path}"], binary=True)
        if rc != 0:
            _refuse(res, "BLOB_MISSING", key,
                    f"{pin['commit'][:12]}:{path}")
            continue
        got = hashlib.sha256(blob).hexdigest()
        if got != pin["blob_sha256"]:
            _refuse(res, "BLOB_SHA_MISMATCH", key,
                    f"recorded={pin['blob_sha256'][:12]} got={got[:12]}")

        ext = os.path.splitext(path)[1]
        if pin["object_id"] != OBJECT_ID.get(ext, "bytes"):
            _refuse(res, "OBJECT_ID_MISMATCH", key,
                    f"recorded={pin['object_id']} rule={OBJECT_ID.get(ext, 'bytes')}")
        if pin["imported_section"] != "whole-file":
            _refuse(res, "SECTION_RULE_UNKNOWN", key,
                    pin["imported_section"])
        elif pin["imported_section_sha256"] != got:
            _refuse(res, "SECTION_DIGEST_MISMATCH", key,
                    f"recorded={pin['imported_section_sha256'][:12]} got={got[:12]}")

    # completeness: every freeze-dir artifact at the target must be
    # pinned (no unlisted dependency can hide inside the freeze tree)
    rc, tree = _run(repo, ["ls-tree", "-r", "--name-only",
                           target_full, "--", FREEZE_DIR])
    if rc == 0:
        for path in tree.splitlines():
            if path and path != MANIFEST_PATH \
                    and path not in pinned_paths:
                _refuse(res, "UNPINNED_FREEZE_ARTIFACT", None, path)

    res["verdict"] = "PASS" if not res["typed_reasons"] else "REFUSE"
    return res


def verify(repo, manifest_commit):
    """The real path: manifest bytes come from the stated commit's git
    object, NEVER from disk; a doctored working tree cannot alter the
    result (KAT-enforced)."""
    rc, full = _run(repo, ["rev-parse", f"{manifest_commit}^{{commit}}"])
    if rc != 0 or not full:
        return {"verdict": "REFUSE", "pins_checked": 0,
                "manifest_commit": manifest_commit,
                "typed_reasons": [{"reason": "MANIFEST_COMMIT_UNRESOLVABLE",
                                   "pin": None,
                                   "detail": manifest_commit}]}
    raw = load_manifest_bytes(repo, full)
    if raw is None:
        return {"verdict": "REFUSE", "pins_checked": 0,
                "manifest_commit": full,
                "typed_reasons": [{"reason": "MANIFEST_NOT_IN_COMMIT",
                                   "pin": None,
                                   "detail": f"{full[:12]}:{MANIFEST_PATH}"}]}
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        return {"verdict": "REFUSE", "pins_checked": 0,
                "manifest_commit": full,
                "typed_reasons": [{"reason": "MANIFEST_UNPARSEABLE",
                                   "pin": None, "detail": str(e)}]}
    res = _verify_obj(repo, full, obj)
    res["manifest_blob_sha256"] = hashlib.sha256(raw).hexdigest()
    return res


def _has(res, reason):
    return any(r["reason"] == reason for r in res["typed_reasons"])


def kat(repo, manifest_commit):
    """KAT matrix. Fixtures are derived from the REAL committed
    manifest, then doctored (self-consistent-synthetic lesson). Every
    negative case asserts its EXACT typed reason; a refusal without the
    typed reason is a DEFECT."""
    rc, full = _run(repo, ["rev-parse", f"{manifest_commit}^{{commit}}"])
    assert rc == 0, "manifest commit unresolvable"
    raw = load_manifest_bytes(repo, full)
    assert raw is not None, "real manifest missing"
    base = json.loads(raw.decode("utf-8"))
    target = base["design_target_commit"]
    failures = []

    def case(name, ok, detail=""):
        line = f"  [{'PASS' if ok else 'DEFECT'}] {name}" + \
               (f" -- {detail}" if detail and not ok else "")
        print(line)
        if not ok:
            failures.append(name)

    # 1. positive: real manifest verifies clean
    r = verify(repo, full)
    case("positive-real-manifest",
         r["verdict"] == "PASS" and not r["typed_reasons"]
         and r["pins_checked"] == base["pin_count"],
         json.dumps(r["typed_reasons"]))

    # 2. schema doctored
    d = copy.deepcopy(base)
    d["schema"] = "wrong-schema"
    case("schema-mismatch",
         _has(_verify_obj(repo, full, d), "MANIFEST_SCHEMA_MISMATCH"))

    # 3. top field missing
    d = copy.deepcopy(base)
    del d["design_target_commit"]
    case("top-field-missing",
         _has(_verify_obj(repo, full, d), "TOP_FIELD_MISSING"))

    # 4. target unresolvable
    d = copy.deepcopy(base)
    d["design_target_commit"] = "f" * 40
    case("target-unresolvable",
         _has(_verify_obj(repo, full, d), "TARGET_UNRESOLVABLE"))

    # 5. target not an ancestor of the manifest commit: real child sha
    # as target, verified against its own parent
    rc, parent = _run(repo, ["rev-parse", f"{full}^"])
    d = copy.deepcopy(base)
    d["design_target_commit"] = full
    case("target-not-ancestor",
         rc == 0 and _has(_verify_obj(repo, parent, d),
                          "TARGET_NOT_ANCESTOR_OF_MANIFEST_COMMIT"))

    # 6. repository identity doctored
    d = copy.deepcopy(base)
    d["repository_url"] = "https://github.com/evil/geospec"
    case("repo-identity-mismatch",
         _has(_verify_obj(repo, full, d), "REPO_IDENTITY_MISMATCH"))

    # 7. pin count doctored
    d = copy.deepcopy(base)
    d["pin_count"] = d["pin_count"] + 1
    case("pin-count-mismatch",
         _has(_verify_obj(repo, full, d), "PIN_COUNT_MISMATCH"))

    # 8/9. pin schema not closed (extra field; missing field)
    k0 = sorted(base["pins"])[0]
    d = copy.deepcopy(base)
    d["pins"][k0]["extra"] = 1
    case("pin-extra-field",
         _has(_verify_obj(repo, full, d), "PIN_SCHEMA_NOT_CLOSED"))
    d = copy.deepcopy(base)
    del d["pins"][k0]["blob_sha256"]
    case("pin-missing-field",
         _has(_verify_obj(repo, full, d), "PIN_SCHEMA_NOT_CLOSED"))

    # 10. non-ancestor pin: the manifest commit itself is a DESCENDANT
    # of the target, so it can never be a valid pin commit
    d = copy.deepcopy(base)
    d["pins"][k0]["commit"] = full
    case("non-ancestor-pin",
         _has(_verify_obj(repo, full, d), "NON_ANCESTOR_PIN"))

    # 11. last-touch retarget: another pin's (ancestor) commit
    others = [k for k in sorted(base["pins"])
              if base["pins"][k]["commit"] != base["pins"][k0]["commit"]]
    d = copy.deepcopy(base)
    d["pins"][k0]["commit"] = base["pins"][others[0]]["commit"]
    case("last-touch-mismatch",
         _has(_verify_obj(repo, full, d), "LAST_TOUCH_MISMATCH"))

    # 12. blob sha doctored
    d = copy.deepcopy(base)
    s = d["pins"][k0]["blob_sha256"]
    d["pins"][k0]["blob_sha256"] = ("0" if s[0] != "0" else "1") + s[1:]
    case("blob-sha-mismatch",
         _has(_verify_obj(repo, full, d), "BLOB_SHA_MISMATCH"))

    # 13. object id doctored
    d = copy.deepcopy(base)
    d["pins"][k0]["object_id"] = "wrong-object"
    case("object-id-mismatch",
         _has(_verify_obj(repo, full, d), "OBJECT_ID_MISMATCH"))

    # 14. section digest doctored
    d = copy.deepcopy(base)
    s = d["pins"][k0]["imported_section_sha256"]
    d["pins"][k0]["imported_section_sha256"] = \
        ("0" if s[0] != "0" else "1") + s[1:]
    case("section-digest-mismatch",
         _has(_verify_obj(repo, full, d), "SECTION_DIGEST_MISMATCH"))

    # 15. unknown section rule
    d = copy.deepcopy(base)
    d["pins"][k0]["imported_section"] = "lines 1-10"
    case("section-rule-unknown",
         _has(_verify_obj(repo, full, d), "SECTION_RULE_UNKNOWN"))

    # 16. path unknown at target
    d = copy.deepcopy(base)
    d["pins"][k0]["path"] = "docs/DOES_NOT_EXIST.md"
    case("path-not-at-target",
         _has(_verify_obj(repo, full, d), "PATH_NOT_AT_TARGET"))

    # 17. unlisted freeze artifact: drop a freeze-dir pin (count fixed
    # up so ONLY the completeness check can catch it)
    kf = next(k for k in sorted(base["pins"])
              if base["pins"][k]["path"].startswith(FREEZE_DIR + "/"))
    d = copy.deepcopy(base)
    del d["pins"][kf]
    d["pin_count"] = len(d["pins"])
    case("unpinned-freeze-artifact",
         _has(_verify_obj(repo, full, d), "UNPINNED_FREEZE_ARTIFACT"))

    # 18. dirty-disk independence: doctor the WORKING-TREE manifest,
    # rerun the real git-object path, assert an identical PASS
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

    # 19. manifest absent at commit: repo root commit
    rc, roots = _run(repo, ["rev-list", "--max-parents=0", full])
    root = roots.splitlines()[0]
    case("manifest-not-in-commit",
         _has(verify(repo, root), "MANIFEST_NOT_IN_COMMIT"))

    print(f"KAT: {19 - len(failures)}/19 pass"
          + (f"; DEFECTS: {failures}" if failures else ""))
    return not failures


def main():
    repo = os.path.abspath(sys.argv[1])
    manifest_commit = sys.argv[2]
    run_kat = "--kat" in sys.argv[3:]
    with open(os.path.abspath(__file__), "rb") as f:
        self_sha = hashlib.sha256(f.read()).hexdigest()
    if run_kat:
        if not kat(repo, manifest_commit):
            print("KAT_DEFECT: matrix failed; verdict withheld")
            sys.exit(2)
    res = verify(repo, manifest_commit)
    res["verifier_source_sha256"] = self_sha
    res["verified_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                        time.gmtime())
    res["authorizes"] = ("NOTHING beyond design-pin integrity: no "
                         "PRESTART, no fire, no prospective-value "
                         "access; execution-manifest-v1 is separate "
                         "and still open")
    print(json.dumps(res, indent=1, sort_keys=True))
    sys.exit(0 if res["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
