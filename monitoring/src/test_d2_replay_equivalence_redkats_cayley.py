"""RED-first KATs -- D2 sealed-replay EXACT SCIENTIFIC EQUIVALENCE bar (cayley).

Codex ruling `f2f24b6` (2026-08-17, both remint findings sustained): a replacement
root produced by a sealed zero-provider-I/O replay is acceptable ONLY as a
provenance-preserving correction -- it must be SCIENTIFICALLY BYTE-EQUIVALENT to the
held acquisition root it supersedes. Grassmann's finding 2 (`2b27621`) is the
counterexample this bar exists to refuse: 596/720 daily rows differed, the socal
admitted set shifted 96->91, and every threshold moved, because the failed launcher's
SealedProvider skipped the live pipeline's merge+trim. Admission-set or threshold
equality alone is INSUFFICIENT (drift can cancel at those projections); the bar
compares the full typed daily-row carrier before any downstream scalarization.

REQUIRED EQUAL (codex f2f24b6 items 1-4):
  1. canonical digest equality for EVERY daily scientific row, in the same
     carrier/arm/day index order (full typed row -- matrices, eigenvalues, supports,
     ratios, qc, derivations -- no projection; the row schema carries no identity);
  2. exact admitted/refused sets and reason codes (explicit set-level check on top of
     the row digests, for diagnostics);
  3. exact matrix / eigenvalue / replay-output / ratio / sample-count / threshold
     values (rows + admission_results + replay_metrics + prior_evidence);
  4. exact provider-object identities (sha256+size set equality; per-object typed
     equality after dropping ONLY process-linkage/issuance keys) and byte-equal held
     raw objects.
ONLY THE EXPLICIT ALLOWLIST MAY DIFFER (item 5): repaired source/core-blob
attestations (capsule source_commit, manifest producer/implementation commits),
digests INDUCED by those attestations (input_manifest_sha256, capsule_sha256,
registry expected_sha256), live issuance time (issued_utc), replacement/supersession
identity (WAL, process/attempt ids and timestamps, batch manifest/root digest).
ANY file outside the classification -- including a file present in only one root --
REFUSES: unknown surface is not silently tolerated.

Verification sequence this bar pins (codex f2f24b6 authority section):
  RED (production): run on (held d2_renewal_campaign_20260816, no-standing remint) --
      MUST FAIL, reproducing finding 2 (~596/720 differing rows).
  GREEN (acceptance): run on (held, corrected-SealedProvider replacement) -- MUST
      PASS before any lane closes. Set D2_HELD_ROOT + D2_REPLACEMENT_ROOT.
No provider I/O in this bar. No lift, no claim, no replay authorization.
"""

import hashlib
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FAILS = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    suffix = f" - {detail}" if detail and not ok else ""
    print(f"    [{tag}] {name}{suffix}")
    if not ok:
        FAILS.append(name)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


# ---- classification -----------------------------------------------------------------
BYTE_EQUAL = {
    "campaign_plan.json", "published_phase_ledger.json", "replay_metrics.json",
    "prior_evidence.json", "d2_diagnostic_result.json",
}
TYPED = {
    "admission_results.json", "input_manifest.json", "registry_candidate.json",
}
IDENTITY_TOLERANT = {
    "acquisition_attempts.jsonl", "operation_ledger.jsonl",
    "campaign_process_ledger.jsonl",
}
REPLACEMENT_IDENTITY = {           # wholesale-allowlisted (may differ / be one-sided)
    "batch_manifest.json", "resume_state.json", "resume_state.head.json",
}
DAILY = "calibration_daily.jsonl"

# per-file typed allowlists (attestations + induced digests + issuance ONLY)
CAPSULE_ALLOW = {"source_commit", "input_manifest_sha256", "issued_utc"}
ADMISSION_ALLOW_TOP = {"implementation_commit"}
ADMISSION_ALLOW_ROW = {"capsule_sha256"}
MANIFEST_ALLOW_TOP = {"producer_commit", "implementation_commit"}
REGISTRY_ALLOW = {"expected_sha256"}
# generic identity-key predicate for object rows / ledger rows
def _is_identity_key(k):
    return (k.endswith("_utc") or k.endswith("_id") or k.startswith("attempt")
            or k.startswith("process") or "process" in k or k.startswith("reuse")
            or k == "owner_launch_authorization")


def _drop_identity(obj):
    if isinstance(obj, dict):
        return {k: _drop_identity(v) for k, v in obj.items() if not _is_identity_key(k)}
    if isinstance(obj, list):
        return [_drop_identity(v) for v in obj]
    return obj


def _read_json(root, rel):
    with open(os.path.join(root, rel), "rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


def _read_jsonl(root, rel):
    rows = []
    with open(os.path.join(root, rel), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _walk_rel(root):
    out = set()
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            out.add(os.path.relpath(os.path.join(dirpath, name), root)
                    .replace(os.sep, "/"))
    return out


def validate_replay_equivalence(held, repl):
    """Returns (ok, detail, report). REFUSES on the first violated class; report
    always carries the daily-row tally when rows were reached."""
    report = {}
    held_files, repl_files = _walk_rel(held), _walk_rel(repl)

    def classify(rel):
        base = rel.split("/", 1)[0]
        if rel in BYTE_EQUAL or rel in TYPED or rel in IDENTITY_TOLERANT \
                or rel in REPLACEMENT_IDENTITY or rel == DAILY:
            return "known"
        if base == "raw_objects":
            return "raw"
        if base == "capsules":
            return "capsule"
        return "UNKNOWN"

    for rel in sorted(held_files | repl_files):
        cls = classify(rel)
        if cls == "UNKNOWN":
            return False, f"unclassified file (out of allowlist): {rel}", report
        if rel not in held_files and cls not in ("known",) or \
           rel not in repl_files and cls not in ("known",):
            if cls in ("raw", "capsule"):
                return False, f"one-sided {cls} file: {rel}", report
        if (rel in held_files) != (rel in repl_files) and rel not in REPLACEMENT_IDENTITY:
            return False, f"file present in only one root: {rel}", report

    # 0) byte-equal class
    for rel in sorted(BYTE_EQUAL):
        with open(os.path.join(held, rel), "rb") as fh:
            hb = fh.read()
        with open(os.path.join(repl, rel), "rb") as fh:
            rb = fh.read()
        if hb != rb:
            return False, f"byte-equal class differs: {rel}", report

    # 1) daily rows: identical ordered index + full-row canonical digest equality
    hrows = _read_jsonl(held, DAILY)
    rrows = _read_jsonl(repl, DAILY)
    report["daily_total"] = len(hrows)
    if len(hrows) != len(rrows):
        return False, f"daily row count {len(hrows)} != {len(rrows)}", report
    hidx = [(r.get("carrier_key"), r.get("arm"), r.get("day")) for r in hrows]
    ridx = [(r.get("carrier_key"), r.get("arm"), r.get("day")) for r in rrows]
    if hidx != ridx:
        first = next(i for i, (a, b) in enumerate(zip(hidx, ridx)) if a != b)
        return False, f"daily index order diverges at {first}: {hidx[first]} vs {ridx[first]}", report
    diffs = [i for i, (a, b) in enumerate(zip(hrows, rrows))
             if sha(canon(a)) != sha(canon(b))]
    report["daily_differing"] = len(diffs)
    report["first_diff"] = (hidx[diffs[0]] if diffs else None)
    if diffs:
        return False, (f"daily rows differ: {len(diffs)}/{len(hrows)} "
                       f"(first {hidx[diffs[0]]})"), report

    # 2) explicit admitted/refused sets + reason codes (diagnostic redundancy)
    def _sets(rows):
        adm = {(r["carrier_key"], r["arm"], r["day"]) for r in rows
               if r.get("status") == "ADMITTED"}
        rsn = {(r["carrier_key"], r["arm"], r["day"], tuple(r.get("qc_reasons") or []))
               for r in rows}
        return adm, rsn
    if _sets(hrows) != _sets(rrows):
        return False, "admitted/refused sets or reason codes differ", report

    # 3) admission_results typed compare
    ha = _read_json(held, "admission_results.json")
    ra = _read_json(repl, "admission_results.json")
    ha2 = {k: v for k, v in ha.items() if k not in ADMISSION_ALLOW_TOP and k != "regions"}
    ra2 = {k: v for k, v in ra.items() if k not in ADMISSION_ALLOW_TOP and k != "regions"}
    if ha2 != ra2:
        return False, "admission_results top-level differs beyond allowlist", report
    hr, rr = ha.get("regions"), ra.get("regions")
    if not isinstance(hr, list) or not isinstance(rr, list) or len(hr) != len(rr):
        return False, "admission_results.regions shape/count differs", report
    for i, (a, b) in enumerate(zip(hr, rr)):
        a2 = {k: v for k, v in a.items() if k not in ADMISSION_ALLOW_ROW}
        b2 = {k: v for k, v in b.items() if k not in ADMISSION_ALLOW_ROW}
        if a2 != b2:
            keys = sorted(k for k in set(a2) | set(b2) if a2.get(k) != b2.get(k))
            return False, f"admission region[{i}] differs beyond allowlist: {keys}", report

    # 4) capsules typed compare (thresholds/windows/replay sha EXACT)
    hcaps = sorted(f for f in held_files if f.startswith("capsules/"))
    for rel in hcaps:
        a = _read_json(held, rel)
        b = _read_json(repl, rel)
        a2 = {k: v for k, v in a.items() if k not in CAPSULE_ALLOW}
        b2 = {k: v for k, v in b.items() if k not in CAPSULE_ALLOW}
        if a2 != b2:
            keys = sorted(k for k in set(a2) | set(b2) if a2.get(k) != b2.get(k))
            return False, f"capsule {rel} differs beyond allowlist: {keys}", report

    # 5) input_manifest: object identity exact (sha+size set + projected typed rows)
    hm = _read_json(held, "input_manifest.json")
    rm = _read_json(repl, "input_manifest.json")
    hm2 = {k: v for k, v in hm.items() if k not in MANIFEST_ALLOW_TOP and k != "objects"}
    rm2 = {k: v for k, v in rm.items() if k not in MANIFEST_ALLOW_TOP and k != "objects"}
    if hm2 != rm2:
        return False, "input_manifest top-level differs beyond allowlist", report
    ho, ro = hm.get("objects") or [], rm.get("objects") or []
    hset = {(o.get("sha256"), o.get("size_bytes")) for o in ho}
    rset = {(o.get("sha256"), o.get("size_bytes")) for o in ro}
    report["objects_held"] = len(hset)
    if hset != rset:
        return False, (f"provider-object sha/size sets differ "
                       f"(held {len(hset)} repl {len(rset)})"), report
    hproj = sorted((canon(_drop_identity(o)) for o in ho))
    rproj = sorted((canon(_drop_identity(o)) for o in ro))
    if hproj != rproj:
        return False, "provider-object typed rows differ beyond identity keys", report

    # 6) raw objects byte-equal by relative path
    hraw = {f for f in held_files if f.startswith("raw_objects/")}
    for rel in sorted(hraw):
        with open(os.path.join(held, rel), "rb") as fh:
            hb = fh.read()
        with open(os.path.join(repl, rel), "rb") as fh:
            rb = fh.read()
        if hb != rb:
            return False, f"raw object bytes differ: {rel}", report

    # 7) registry candidate (expected_sha256 induced -> allowlisted)
    hg = _read_json(held, "registry_candidate.json")
    rg = _read_json(repl, "registry_candidate.json")
    if set(hg) != set(rg):
        return False, "registry carriers differ", report
    for carrier in hg:
        a2 = {k: v for k, v in hg[carrier].items() if k not in REGISTRY_ALLOW}
        b2 = {k: v for k, v in rg[carrier].items() if k not in REGISTRY_ALLOW}
        if a2 != b2:
            return False, f"registry[{carrier}] differs beyond allowlist", report

    # 8) identity-tolerant ledgers: count + ordered label projection exact
    for rel in sorted(IDENTITY_TOLERANT):
        a = _read_jsonl(held, rel)
        b = _read_jsonl(repl, rel)
        if len(a) != len(b):
            return False, f"{rel} row count {len(a)} != {len(b)}", report
        pa = [canon(_drop_identity(r)) for r in a]
        pb = [canon(_drop_identity(r)) for r in b]
        if pa != pb:
            first = next(i for i, (x, y) in enumerate(zip(pa, pb)) if x != y)
            return False, f"{rel} label projection differs at row {first}", report

    return True, (f"EQUIVALENT: {report['daily_total']} daily rows digest-equal, "
                  f"{report.get('objects_held')} objects identity-equal"), report


# ---- synthetic lock fixtures --------------------------------------------------------
def _mk_root(td):
    os.makedirs(os.path.join(td, "capsules"), exist_ok=True)
    os.makedirs(os.path.join(td, "raw_objects"), exist_ok=True)
    plan = {"contract_id": "codex-d2-campaign-v2-renewal-2026-08-16-v1", "carriers": ["c1"]}
    ledger = {"rows": [{"carrier_key": "c1", "scored_day": "2026-03-02"}]}
    days = ["2026-03-02", "2026-03-03", "2026-03-04"]
    rows = []
    for arm in ("incident", "activation"):
        for d in days:
            admitted = d != "2026-03-04"
            rows.append({"arm": arm, "carrier_key": "c1", "day": d,
                         "status": "ADMITTED" if admitted else "REJECTED",
                         "ratio": 0.31 + 0.01 * days.index(d) if admitted else None,
                         "qc_reasons": [] if admitted else ["NO_PUBLISHED_DAILY_RECORD"],
                         "common_support_count": 78524 if admitted else None,
                         "correlation_matrix": [[1.0, 0.4], [0.4, 1.0]] if admitted else None,
                         "ordered_eigenvalues": [1.4, 0.6] if admitted else None,
                         "input_object_sha256s": ["aa" * 32] if admitted else []})
    raw = b"held-raw-object-bytes-v1"
    obj = {"sha256": sha(raw), "size_bytes": len(raw), "carrier_key": "c1",
           "scored_day": "2026-03-02", "segment_name": "seg_a", "nslc": "KO.S00..BHZ",
           "record_sha256": "c" * 64, "acquired_by_process": "p-held",
           "verified_by_process": "p-held", "reuse": "FRESH", "acquired_utc": "t1"}
    files = {
        "campaign_plan.json": canon(plan),
        "published_phase_ledger.json": canon(ledger),
        "replay_metrics.json": canon({"replay": "metrics", "sha": "d" * 64}),
        "prior_evidence.json": canon({"prior": True}),
        "d2_diagnostic_result.json": canon({"diag": 1}),
        "admission_results.json": canon(
            {"schema": "adm-v1", "implementation_commit": "292b1069" + "0" * 32,
             "regions": [{"runner_key": "c1", "carrier_key": "c1",
                          "status": "ADMITTED_CANDIDATE", "incident_threshold": 0.31,
                          "activation_threshold": 0.31, "incident_n": 2,
                          "activation_n": 2, "reason_codes": [],
                          "capsule_path": "capsules/c1_calibration.json",
                          "capsule_sha256": "e" * 64}]}),
        "capsules/c1_calibration.json": canon(
            {"schema": "geospec-d2-calibration-v1", "region": "c1", "threshold": 0.31,
             "calibration_window": {"start": "2026-03-02", "end": "2026-07-17"},
             "replay_output_sha256": "d" * 64, "valid_through": "2026-08-23",
             "source_commit": "held-commit", "input_manifest_sha256": "f" * 64,
             "issued_utc": "2026-08-16T02:00:00.000000Z"}),
        "input_manifest.json": canon(
            {"schema": "im-v2-resume", "producer_commit": "held-commit",
             "implementation_commit": "292b1069" + "0" * 32, "objects": [obj]}),
        "registry_candidate.json": canon(
            {"c1": {"capsule_path": "capsules/c1_calibration.json",
                    "expected_sha256": "e" * 64, "topology_version": "t1"}}),
        "acquisition_attempts.jsonl": b"\n".join(canon(r) for r in [
            {"carrier_key": "c1", "scored_day": "2026-03-02", "segment_name": "seg_a",
             "station_id": "KO.S00", "provider": "KOERI", "status": "FETCHED",
             "selected_nslc": "KO.S00..BHZ", "attempt_id": "a-held",
             "process_id": "p-held", "attempted_utc": "t1"},
            {"carrier_key": "c1", "scored_day": "2026-03-03", "segment_name": "seg_a",
             "station_id": "KO.S00", "provider": "KOERI", "status": "FETCHED",
             "selected_nslc": "KO.S00..BHZ", "attempt_id": "b-held",
             "process_id": "p-held", "attempted_utc": "t2"}]),
        "operation_ledger.jsonl": canon(
            {"arm": "incident", "carrier_key": "c1", "day": "2026-03-02",
             "op": "SCORE", "operation_id": "o-held"}),
        "campaign_process_ledger.jsonl": canon(
            {"process_id": "p-held", "ordinal": 1, "disposition": "COMPLETED",
             "producer_commit": "held-commit", "owner_launch_authorization": "recpt",
             "process_started_utc": "t0", "process_ended_utc": "t9"}),
        "resume_state.json": canon({"events": ["held-wal"]}),
        "resume_state.head.json": canon({"generation": 1}),
        "batch_manifest.json": canon({"run_id": "held-run", "root_sha": "1" * 64}),
        f"raw_objects/{sha(raw)}.ms": raw,
        "calibration_daily.jsonl": b"\n".join(canon(r) for r in rows),
    }
    for rel, body in files.items():
        path = os.path.join(td, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(body)
    return files


def _mk_pair(mutate=None):
    """held + replacement with ALLOWLIST-ONLY differences; mutate(repl_dir) optional."""
    th = tempfile.mkdtemp()
    tr = tempfile.mkdtemp()
    _mk_root(th)
    _mk_root(tr)

    def rewrite(rel, fn):
        p = os.path.join(tr, rel)
        doc = json.loads(open(p, "rb").read().decode("utf-8"))
        fn(doc)
        with open(p, "wb") as fh:
            fh.write(canon(doc))

    # allowlisted replacement-identity differences (nominal MUST still pass)
    rewrite("capsules/c1_calibration.json", lambda d: d.update(
        source_commit="repl-commit", input_manifest_sha256="9" * 64,
        issued_utc="2026-08-17T18:00:00.000000Z"))
    rewrite("input_manifest.json", lambda d: (
        d.update(producer_commit="repl-commit", implementation_commit="repl-impl"),
        d["objects"][0].update(acquired_by_process="p-repl",
                               verified_by_process="p-repl", reuse="REUSED_VERIFIED",
                               acquired_utc="t7")))
    rewrite("admission_results.json", lambda d: (
        d.update(implementation_commit="repl-impl"),
        d["regions"][0].update(capsule_sha256="9" * 64)))
    rewrite("registry_candidate.json", lambda d: d["c1"].update(expected_sha256="9" * 64))
    rewrite("batch_manifest.json", lambda d: d.update(run_id="repl-run", root_sha="2" * 64))
    rewrite("resume_state.json", lambda d: d.update(events=["repl-wal", "x"]))
    rewrite("resume_state.head.json", lambda d: d.update(generation=3))
    # attempts/process/operation ledgers: flip identity fields only
    for rel in ("acquisition_attempts.jsonl", "operation_ledger.jsonl",
                "campaign_process_ledger.jsonl"):
        p = os.path.join(tr, rel)
        rows = [json.loads(x) for x in open(p, "rb").read().decode("utf-8").splitlines() if x]
        for r in rows:
            for k in list(r):
                if k.endswith("_id"):
                    r[k] = r[k].replace("held", "repl") if isinstance(r[k], str) else r[k]
                if k.endswith("_utc"):
                    r[k] = "t-repl"
        with open(p, "wb") as fh:
            fh.write(b"\n".join(canon(r) for r in rows))
    if mutate:
        mutate(tr)
    return th, tr


def _edit_jsonl(root, rel, fn):
    p = os.path.join(root, rel)
    rows = [json.loads(x) for x in open(p, "rb").read().decode("utf-8").splitlines() if x]
    fn(rows)
    with open(p, "wb") as fh:
        fh.write(b"\n".join(canon(r) for r in rows))


def _edit_json(root, rel, fn):
    p = os.path.join(root, rel)
    doc = json.loads(open(p, "rb").read().decode("utf-8"))
    fn(doc)
    with open(p, "wb") as fh:
        fh.write(canon(doc))


def main():
    # RE-1 nominal: allowlist-only pair PASSES (proves the allowlist admits exactly
    # the attestation/issuance/identity surface and nothing else blocks)
    th, tr = _mk_pair()
    ok_nom, det_nom, rep_nom = validate_replay_equivalence(th, tr)
    check("RE-1a nominal replacement (allowlist-only differences: attestations, "
          "induced digests, issuance, WAL/batch/process identity) is EQUIVALENT",
          ok_nom, det_nom)

    # RE-1b..: violation battery -- each class must REFUSE
    def refused(mut):
        h, r = _mk_pair(mutate=mut)
        ok, det, _ = validate_replay_equivalence(h, r)
        return (not ok), det

    battery = []
    v, d = refused(lambda tr: _edit_jsonl(tr, DAILY, lambda rows: rows[0].update(
        ratio=rows[0]["ratio"] + 1e-9)))
    battery.append(("ratio nudged 1e-9", v, d))
    v, d = refused(lambda tr: _edit_jsonl(tr, DAILY, lambda rows: rows.reverse()))
    battery.append(("row order reversed", v, d))
    v, d = refused(lambda tr: _edit_jsonl(tr, DAILY, lambda rows: rows.pop()))
    battery.append(("row dropped", v, d))
    v, d = refused(lambda tr: _edit_jsonl(tr, DAILY, lambda rows: rows[0].update(
        status="REJECTED")))
    battery.append(("status flip", v, d))
    v, d = refused(lambda tr: _edit_jsonl(tr, DAILY, lambda rows: rows[-1].update(
        qc_reasons=["OTHER"])))
    battery.append(("reason-code change", v, d))
    v, d = refused(lambda tr: _edit_jsonl(tr, DAILY, lambda rows: rows[0].update(
        ordered_eigenvalues=[1.4000001, 0.5999999])))
    battery.append(("eigenvalue drift", v, d))
    v, d = refused(lambda tr: _edit_jsonl(tr, DAILY, lambda rows: rows[0].update(
        common_support_count=78528)))
    battery.append(("support-count shift", v, d))
    v, d = refused(lambda tr: _edit_json(tr, "admission_results.json",
                                         lambda doc: doc["regions"][0].update(
                                             activation_threshold=0.315)))
    battery.append(("threshold moved", v, d))
    v, d = refused(lambda tr: _edit_json(tr, "capsules/c1_calibration.json",
                                         lambda doc: doc.update(threshold=0.315)))
    battery.append(("capsule threshold changed (non-allowlisted)", v, d))
    v, d = refused(lambda tr: open(os.path.join(tr, "raw_objects",
                                                os.listdir(os.path.join(tr, "raw_objects"))[0]),
                                   "wb").write(b"tampered"))
    battery.append(("raw object bytes differ", v, d))
    v, d = refused(lambda tr: _edit_json(tr, "campaign_plan.json",
                                         lambda doc: doc.update(carriers=["c1", "c2"])))
    battery.append(("plan bytes differ", v, d))
    v, d = refused(lambda tr: open(os.path.join(tr, "extra_report.json"), "wb")
                   .write(b"{}"))
    battery.append(("out-of-allowlist file added", v, d))
    v, d = refused(lambda tr: _edit_jsonl(tr, "acquisition_attempts.jsonl",
                                          lambda rows: rows[0].update(status="REFUSED")))
    battery.append(("attempt label flip", v, d))
    v, d = refused(lambda tr: _edit_json(tr, "input_manifest.json", lambda doc:
                                         doc["objects"][0].update(sha256="b" * 64)))
    battery.append(("provider-object sha changed", v, d))

    all_refused = all(v for _, v, _ in battery)
    check("RE-1b violation battery: every non-allowlisted difference REFUSES "
          "(14 classes: row nudge/order/count/status/reasons/eigenvalues/supports, "
          "thresholds in admission+capsule, raw bytes, plan bytes, unknown file, "
          "attempt labels, object identity)", all_refused,
          "; ".join(f"{n}: {'ok' if v else 'NOT REFUSED'}" for n, v, _ in battery))

    # RE-2 real pair (verification lanes)
    held_env = os.environ.get("D2_HELD_ROOT")
    repl_env = os.environ.get("D2_REPLACEMENT_ROOT")
    if held_env and repl_env and os.path.isdir(held_env) and os.path.isdir(repl_env):
        ok, det, rep = validate_replay_equivalence(held_env, repl_env)
        check("RE-2 real-pair exact scientific equivalence (held vs replacement): "
              "720 typed daily rows digest-equal in index order + sets/reasons + "
              "thresholds + objects + raw bytes; only the attestation/issuance/"
              "identity allowlist differs", ok, f"{det} report={rep}")
    else:
        check("RE-2 real-pair exact scientific equivalence (held vs replacement)",
              False, "PENDING PAIR (set D2_HELD_ROOT + D2_REPLACEMENT_ROOT; the "
                     "RED run is (held, no-standing remint) which MUST FAIL ~596/720; "
                     "the GREEN run is (held, corrected replacement) which MUST PASS)")


main()
print()
if FAILS:
    print(f"D2 REPLAY-EQUIVALENCE RED-KAT FAILURES: {FAILS}")
    sys.exit(1)
print("ALL D2 REPLAY-EQUIVALENCE RED-KATs PASS")
