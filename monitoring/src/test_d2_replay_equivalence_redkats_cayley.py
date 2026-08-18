"""RED-first KATs -- D2 sealed-replay EXACT SCIENTIFIC EQUIVALENCE bar (cayley).

REV 2.4 (codex ruling `69b801f` OPTION A, superseding the REV 2.3 class-wide
allowance): the projection is NAMED -- `sealed-unavailable-quotient-v1` -- and is
NOT a class allowance: divergent pairs are accepted ONLY under a DECLARED quotient
key set, and the projected key set must equal the declaration EXACTLY. An eighth
key, a missing key, a duplicate key, or any pair under an empty/undeclared
quotient REFUSES. The production RE-2 lane declares the SEVEN predeclared remint3
keys (grassmann `d483f37` exhaustive disclosure; carrier keys per
d2_renewal_plan._TARGET_ORDER). The held candidate must bind to the row's station
(changed-candidate refusal). BOTH original and replacement labels are recorded in
report["projected_attempts"] -- neither is overwritten nor reported as literal
equality -- and the acceptance receipt states science_daily_differing=N/M,
attempt_projection=<name>, projected_attempts=N. Disclosure: REV 2.3 was authored
concurrently with (and landed 4 minutes after) codex's ruling -- concordant on
option (a) and the rejection of (b), but its class-wide allowance lacked the named
quotient, the exact predeclared-key pin, and the both-labels receipt; REV 2.4
folds the ruling in verbatim. REV 2.3's pair conditions and lock battery are
retained unchanged underneath the quotient gate.

REV 2.3 (grassmann remint3 packet `d483f37`, disposition (a) -- cayley ruling, codex
ratified as OPTION A in `69b801f`): the SOLE admissible attempts-label divergence is the sealed
provider's truthful rendering of a live transient provider failure. The live run
experienced availability-yes-then-transient-fetch-error (terminal UNAVAILABLE,
selected_nslc carries the candidate, reason PROVIDER_UNAVAILABLE, zero input
objects); the sealed provider holds no object for that unit and honestly reports
no-candidate (selected_nslc None, reason NO_AVAILABLE_CANDIDATE, zero input
objects). Forcing label equality would make the sealed replay ASSERT an
availability outcome it never derived from bytes -- option (b) ledger-consulting
label reproduction is REJECTED as retrofitted provenance whose equality is vacuous
(labels copied from the very ledger they are compared against verify nothing).
Allowed EXACTLY when, at the same ordinal row of acquisition_attempts.jsonl ONLY,
with the same key schema and every field outside ATT_ALLOW | {selected_nslc,
reason_codes} byte-equal: held {selected_nslc: <non-empty str>, reason_codes:
["PROVIDER_UNAVAILABLE"]} vs sealed {selected_nslc: None, reason_codes:
["NO_AVAILABLE_CANDIDATE"]}, status == "UNAVAILABLE" AND input_object_sha256s == []
on BOTH sides. Every allowed pair is enumerated in report["transient_label_rows"]
and disclosed in the PASS detail; verification lanes must confirm the enumerated
set equals the disclosed row list (remint3: grassmann's known 7). Locks: wrong
reason-code pair either side, extra reason code, reversed direction, data-bearing
row either side, non-UNAVAILABLE status, held selected_nslc None, station identity
mismatch, and the same pair in operation_ledger.jsonl -- all REFUSE.

REV 2.2 (codex `e127a7f` WORKS-WITH-FIX -- P1/P2 semantics CONCURRED, two root-local
integrity holes repaired; four accepted counterexamples locked):
  2.2-1 P1 no longer drops core_blobs blind: both plans must share the complete
        top-level schema; core_blobs must be a dict whose keyset equals the FIVE
        production CORE_FILES with 40-lower-hex git blob values -- validated BEFORE
        projection; only the VALUES stay cross-root tolerant (RN-5g binds them to the
        producing commit per root). Missing map / wrong keyset / extra key / malformed
        blob each REFUSES.
  2.2-2 P2 root-local closure is now two-way: within each root, every ordered WAL
        PROCESS_STARTED receipt verifies, process ids are nonempty and unique, the
        ordered WAL pid list equals the ledger's, and batch_manifest.process_count ==
        len(process_rows). Orphan-invalid WAL starts and duplicate ledger rows REFUSE;
        cross-root row counts stay free.

REV 2 (codex `76d93f7` WORKS-WITH-FIX -- three classification defects repaired -- plus
grassmann `b947a1e` classification surprise):
  C1 CRITICAL receipt projection: receipts are now VALIDATED AND CROSS-BOUND FIRST
     (real `d2_step4b_producer.verify_launch_authorization`, the production seam), and
     only their cross-root VALUE is omitted from equality. Both roots independently:
     batch receipt + every process-row receipt + every WAL PROCESS_STARTED receipt must
     verify; each process row must bind to its WAL PROCESS_STARTED receipt; the batch
     receipt must equal a COMPLETED process row's receipt. Missing / RELAYED / bad-UTC /
     bad-hash / unbound each REFUSES (always-running locks).
  C2 HIGH generic identity predicate ERASED scientific identity (source_id,
     station_id dropped for ending in _id): replaced with PER-FILE EXACT ALLOWLISTS
     pinned to the production schemas (_input_object, _summary, process rows).
     source_id / station_id counterexamples locked as REFUSE.
  C3 HIGH asymmetric classification: batch_manifest.json + resume_state.json +
     resume_state.head.json REQUIRED IN BOTH ROOTS; batch_manifest compared TYPED
     (named replacement-identity/issuance/attestation/induced fields dropped; all
     scientific policy/targets/versions/non-claims fields EXACT; artifacts sub-map for
     byte-equal-class files EXACT); producer_commit allowlisted in the process-ledger
     and batch attestation surface. Missing-manifest REFUSE + attestation-only PASS
     locked.
  G1 (grassmann): publication_records/** classified BYTE-EQUAL-REQUIRED (sealed
     upstream inputs staged verbatim; any difference is a real violation).
  P1 (surfaced for codex ratification): campaign_plan.json moved BYTE_EQUAL -> TYPED
     with allowlist {core_blobs, created_utc, registered_utc} ONLY -- the authorized
     replay rebuilds the plan at the repaired HEAD (supersession by design, grassmann
     c879f29), so core-blob attestations and plan issuance may differ while carriers/
     days/windows/registry/providers/contract stay EXACT. A scheduled_days change is
     locked as REFUSE.
  P2 (surfaced for codex ratification): campaign_process_ledger.jsonl is REPLACEMENT
     IDENTITY (row-count-free) -- the held root carries its crash/resume process
     history, the replacement carries its own -- but EVERY row's receipt must verify
     and WAL-bind (C1). Science lives in daily rows / objects / attempts, all pinned.

Codex ruling `f2f24b6` (both remint findings sustained): a replacement root is
acceptable ONLY as a provenance-preserving correction -- SCIENTIFICALLY
BYTE-EQUIVALENT to the held acquisition root. Finding 2 (596/720 daily rows drifted,
socal 96->91, thresholds moved) is the counterexample this bar refuses. Admission-set
or threshold equality alone is INSUFFICIENT; the full typed daily-row carrier is
compared before any scalarization.

Verification sequence (f2f24b6 authority section):
  RED (production): (held d2_renewal_campaign_20260816, no-standing remint) MUST FAIL
      on the drift signature (~596/720). GREEN: (held, corrected replacement) MUST
      PASS. Set D2_HELD_ROOT + D2_REPLACEMENT_ROOT.
No provider I/O. No lift, no claim, no replay authorization.
"""

import hashlib
import json
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
# the plan core_blobs keyset authority (== renewal bar CORE_FILES; RN-5g holds the
# independent per-root producer_commit -> git ls-tree -> blob binding)
PLAN_CORE_FILES = ["monitoring/src/seismic_data.py",
                   "monitoring/src/fault_correlation.py",
                   "monitoring/src/ensemble.py",
                   "monitoring/src/d2_step4b_campaign_run.py",
                   "monitoring/src/d2_renewal_plan.py"]
sys.path.insert(0, HERE)
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


# ---- classification (REV 2) ---------------------------------------------------------
BYTE_EQUAL = {
    "published_phase_ledger.json", "replay_metrics.json",
    "prior_evidence.json", "d2_diagnostic_result.json",
}
BYTE_EQUAL_DIRS = {"raw_objects", "publication_records"}      # G1
TYPED = {
    "admission_results.json", "input_manifest.json", "registry_candidate.json",
    "campaign_plan.json", "batch_manifest.json",              # P1 + C3
}
IDENTITY_TOLERANT = {"acquisition_attempts.jsonl", "operation_ledger.jsonl"}
REPLACEMENT_IDENTITY = {                                      # P2 + WAL (receipt-bound)
    "resume_state.json", "resume_state.head.json", "campaign_process_ledger.jsonl",
}
REQUIRED_BOTH = {                                             # C3: presence mandatory
    "batch_manifest.json", "resume_state.json", "resume_state.head.json",
    "campaign_process_ledger.jsonl",
}
DAILY = "calibration_daily.jsonl"

# per-file EXACT allowlists (C2) -- pinned to production schemas
CAPSULE_ALLOW = {"source_commit", "input_manifest_sha256", "issued_utc"}
ADMISSION_ALLOW_TOP = {"implementation_commit"}
ADMISSION_ALLOW_ROW = {"capsule_sha256"}
MANIFEST_ALLOW_TOP = {"producer_commit", "implementation_commit"}
OBJ_ALLOW = {"reuse_disposition", "acquired_process_id", "verified_process_id"}
ATT_ALLOW = {"attempt_id", "terminal_attempt_id", "attempted_utc",
             "indeterminate_attempt_count", "process_id"}
# REV 2.3: the ONLY attempts fields that may diverge beyond ATT_ALLOW, and only as
# the exact transient pair (held live-transient -> sealed truthful no-candidate)
TRANSIENT_ATT_PAIR = {"selected_nslc", "reason_codes"}
# REV 2.4 (codex 69b801f): sealed-unavailable-quotient-v1 -- the SEVEN predeclared
# remint3 attempt keys (carrier_key, scored_day, station_id), frozen from grassmann
# d483f37's exhaustive disclosure + d2_renewal_plan._TARGET_ORDER carrier keys.
# The RE-2 real-pair lane declares EXACTLY this set; any deviation refuses.
SEALED_UNAVAILABLE_QUOTIENT_V1 = (
    ("istanbul_marmara", "2026-04-22", "KO.GEML"),
    ("istanbul_marmara", "2026-06-15", "KO.GEML"),
    ("turkey_kahramanmaras", "2026-05-25", "KO.KBAN"),
    ("turkey_kahramanmaras", "2026-06-07", "KO.KOZT"),
    ("turkey_kahramanmaras", "2026-06-15", "KO.KARA"),
    ("turkey_kahramanmaras", "2026-07-01", "KO.KBAN"),
    ("turkey_kahramanmaras", "2026-07-02", "KO.KOZT"),
)
OPS_ALLOW = {"operation_id", "process_id", "operation_utc", "created_utc",
             "recorded_utc"}
REGISTRY_ALLOW = {"expected_sha256"}
PLAN_ALLOW = {"core_blobs", "created_utc", "registered_utc"}                 # P1
BM_ALLOW = {"run_id", "campaign_id", "producer_commit", "implementation_commit",
            "clean_tree", "campaign_started_utc", "created_utc",
            "campaign_plan_sha256", "owner_launch_authorization", "process_count",
            "state_head_generation", "artifacts"}                            # C3


def _project(row, allow):
    return {k: v for k, v in row.items() if k not in allow}


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


def _classify(rel):
    base = rel.split("/", 1)[0]
    if rel in BYTE_EQUAL or rel == DAILY:
        return "byte"
    if base in BYTE_EQUAL_DIRS and "/" in rel:
        return "byte"
    if rel in TYPED:
        return "typed"
    if rel in IDENTITY_TOLERANT:
        return "tolerant"
    if rel in REPLACEMENT_IDENTITY:
        return "replacement"
    if base == "capsules" and "/" in rel:
        return "typed"
    return "UNKNOWN"


def _validate_receipts(root, label):
    """C1: validate + cross-bind receipts for ONE root. Returns error string or None."""
    from d2_step4b_producer import verify_launch_authorization as verify
    bm = _read_json(root, "batch_manifest.json")
    auth = bm.get("owner_launch_authorization")
    if not verify(auth):
        return f"{label}: batch_manifest receipt fails verify_launch_authorization"
    proc_rows = _read_jsonl(root, "campaign_process_ledger.jsonl")
    if not proc_rows:
        return f"{label}: empty campaign_process_ledger"
    events = _read_json(root, "resume_state.json").get("events")
    if not isinstance(events, list):
        return f"{label}: resume_state.json missing events list"
    # REV 2.2 (codex e127a7f #2): root-local WAL/ledger CLOSURE, two-way.
    # Cross-root row counts stay free (P2); within the root: every ordered
    # PROCESS_STARTED receipt verifies, process ids are nonempty and unique, the
    # ordered WAL pid list equals the ledger's (orphan starts AND duplicate/missing
    # ledger rows both refuse), and batch_manifest.process_count matches.
    starts = [e for e in events if isinstance(e, dict)
              and e.get("kind") == "PROCESS_STARTED"]
    for e in starts:
        if not verify(e.get("owner_launch_authorization")):
            return (f"{label}: WAL PROCESS_STARTED {e.get('process_id')} receipt "
                    f"fails verify_launch_authorization")
    wal_pids = [e.get("process_id") for e in starts]
    if not wal_pids or any(not p for p in wal_pids) \
            or len(set(wal_pids)) != len(wal_pids):
        return f"{label}: WAL PROCESS_STARTED process ids empty or non-unique"
    ledger_pids = [r.get("process_id") for r in proc_rows]
    if wal_pids != ledger_pids:
        return (f"{label}: ordered WAL PROCESS_STARTED pids != ledger pids "
                f"({len(wal_pids)} vs {len(ledger_pids)})")
    if bm.get("process_count") != len(proc_rows):
        return (f"{label}: batch_manifest.process_count {bm.get('process_count')} "
                f"!= {len(proc_rows)} ledger rows")
    wal_receipts = {e.get("process_id"): e.get("owner_launch_authorization")
                    for e in starts}
    completed_receipts = []
    for r in proc_rows:
        rc = r.get("owner_launch_authorization")
        pid = r.get("process_id")
        if not verify(rc):
            return f"{label}: process {pid} receipt fails verify_launch_authorization"
        if wal_receipts[pid] != rc:
            return f"{label}: process {pid} receipt != its WAL PROCESS_STARTED receipt"
        if r.get("disposition") == "COMPLETED":
            completed_receipts.append(rc)
    if auth not in completed_receipts:
        return f"{label}: batch receipt is not any COMPLETED process row's receipt"
    return None


def _transient_label_pair(hrow, rrow):
    """REV 2.3 sole admissible attempts-label divergence: a live transient provider
    failure (availability-yes then fetch-error) rendered by the sealed provider as
    no-candidate. Same key schema; every field outside ATT_ALLOW | TRANSIENT_ATT_PAIR
    byte-equal (binds carrier/day/segment/station/provider/request window); terminal
    UNAVAILABLE + zero input objects on BOTH sides; direction fixed held->sealed."""
    if set(hrow.keys()) != set(rrow.keys()):
        return False
    for k in hrow:
        if k in ATT_ALLOW or k in TRANSIENT_ATT_PAIR:
            continue
        if hrow[k] != rrow[k]:
            return False
    sel = hrow.get("selected_nslc")
    return (hrow.get("status") == "UNAVAILABLE" and rrow.get("status") == "UNAVAILABLE"
            and hrow.get("input_object_sha256s") == []
            and rrow.get("input_object_sha256s") == []
            and isinstance(sel, str) and sel != ""
            # REV 2.4 changed-candidate refusal: the recorded candidate must bind
            # to the row's own station identity (NET.STA prefix of the NSLC)
            and isinstance(hrow.get("station_id"), str)
            and sel.startswith(hrow["station_id"] + ".")
            and rrow.get("selected_nslc") is None
            and hrow.get("reason_codes") == ["PROVIDER_UNAVAILABLE"]
            and rrow.get("reason_codes") == ["NO_AVAILABLE_CANDIDATE"])


def validate_replay_equivalence(held, repl, quotient_keys=()):
    """Returns (ok, detail, report). quotient_keys: the DECLARED
    sealed-unavailable-quotient key set (carrier_key, scored_day, station_id);
    the projected pair set must equal it EXACTLY (REV 2.4, codex 69b801f).
    Empty declaration = no attempts-label divergence is admissible at all."""
    report = {}
    held_files, repl_files = _walk_rel(held), _walk_rel(repl)

    for rel in sorted(held_files | repl_files):
        cls = _classify(rel)
        if cls == "UNKNOWN":
            return False, f"unclassified file (out of allowlist): {rel}", report
        one_sided = (rel in held_files) != (rel in repl_files)
        if one_sided and (rel in REQUIRED_BOTH or cls != "replacement"):
            return False, f"file present in only one root: {rel}", report
    for rel in REQUIRED_BOTH:                                   # C3 presence
        if rel not in held_files or rel not in repl_files:
            return False, f"required file missing: {rel}", report

    # C1: receipts validated + cross-bound in BOTH roots before any projection
    for root, label in ((held, "held"), (repl, "replacement")):
        err = _validate_receipts(root, label)
        if err:
            return False, err, report

    # byte-equal class (incl. publication_records/**, raw_objects/**)
    for rel in sorted(f for f in held_files if _classify(f) == "byte" and f != DAILY):
        with open(os.path.join(held, rel), "rb") as fh:
            hb = fh.read()
        with open(os.path.join(repl, rel), "rb") as fh:
            rb = fh.read()
        if hb != rb:
            return False, f"byte-equal class differs: {rel}", report

    # daily rows: identical ordered index + full-row canonical digest equality
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

    # explicit admitted/refused sets + reason codes (diagnostic redundancy)
    def _sets(rows):
        adm = {(r["carrier_key"], r["arm"], r["day"]) for r in rows
               if r.get("status") == "ADMITTED"}
        rsn = {(r["carrier_key"], r["arm"], r["day"], tuple(r.get("qc_reasons") or []))
               for r in rows}
        return adm, rsn
    if _sets(hrows) != _sets(rrows):
        return False, "admitted/refused sets or reason codes differ", report

    # campaign_plan typed (P1) -- REV 2.2 (codex e127a7f #1): schema + core_blobs
    # presence/keyset/blob-shape validated BEFORE projection; values stay cross-root
    # tolerant (RN-5g binds them to the producing commit per root)
    hp = _read_json(held, "campaign_plan.json")
    rp = _read_json(repl, "campaign_plan.json")
    if set(hp.keys()) != set(rp.keys()):
        keys = sorted(set(hp.keys()) ^ set(rp.keys()))
        return False, f"campaign_plan top-level schema differs: {keys}", report
    for label, plan in (("held", hp), ("replacement", rp)):
        cb = plan.get("core_blobs")
        if not isinstance(cb, dict) or set(cb.keys()) != set(PLAN_CORE_FILES):
            return False, (f"{label} plan core_blobs missing or keyset != the five "
                           f"production CORE_FILES"), report
        if not all(isinstance(v, str) and _HEX40.match(v) for v in cb.values()):
            return False, f"{label} plan core_blobs has a malformed git blob id", report
    if _project(hp, PLAN_ALLOW) != _project(rp, PLAN_ALLOW):
        keys = sorted(k for k in set(hp) | set(rp)
                      if k not in PLAN_ALLOW and hp.get(k) != rp.get(k))
        return False, f"campaign_plan differs beyond allowlist: {keys}", report

    # admission_results typed
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
        if _project(a, ADMISSION_ALLOW_ROW) != _project(b, ADMISSION_ALLOW_ROW):
            keys = sorted(k for k in set(a) | set(b) if k not in ADMISSION_ALLOW_ROW
                          and a.get(k) != b.get(k))
            return False, f"admission region[{i}] differs beyond allowlist: {keys}", report

    # capsules typed (thresholds/windows/replay sha EXACT)
    for rel in sorted(f for f in held_files if f.startswith("capsules/")):
        a = _read_json(held, rel)
        b = _read_json(repl, rel)
        if _project(a, CAPSULE_ALLOW) != _project(b, CAPSULE_ALLOW):
            keys = sorted(k for k in set(a) | set(b) if k not in CAPSULE_ALLOW
                          and a.get(k) != b.get(k))
            return False, f"capsule {rel} differs beyond allowlist: {keys}", report

    # input_manifest: sha+size set + per-object EXACT allowlist projection (C2)
    hm = _read_json(held, "input_manifest.json")
    rm = _read_json(repl, "input_manifest.json")
    hm2 = {k: v for k, v in hm.items() if k not in MANIFEST_ALLOW_TOP and k != "objects"}
    rm2 = {k: v for k, v in rm.items() if k not in MANIFEST_ALLOW_TOP and k != "objects"}
    if hm2 != rm2:
        return False, "input_manifest top-level differs beyond allowlist", report
    ho, ro = hm.get("objects") or [], rm.get("objects") or []
    hset = {(o.get("sha256"), o.get("size")) for o in ho}
    rset = {(o.get("sha256"), o.get("size")) for o in ro}
    report["objects_held"] = len(hset)
    if hset != rset:
        return False, (f"provider-object sha/size sets differ "
                       f"(held {len(hset)} repl {len(rset)})"), report
    hproj = sorted(canon(_project(o, OBJ_ALLOW)) for o in ho)
    rproj = sorted(canon(_project(o, OBJ_ALLOW)) for o in ro)
    if hproj != rproj:
        first = next(i for i, (x, y) in enumerate(zip(hproj, rproj)) if x != y)
        return False, f"provider-object typed rows differ (sorted idx {first})", report

    # registry candidate
    hg = _read_json(held, "registry_candidate.json")
    rg = _read_json(repl, "registry_candidate.json")
    if set(hg) != set(rg):
        return False, "registry carriers differ", report
    for carrier in hg:
        if _project(hg[carrier], REGISTRY_ALLOW) != _project(rg[carrier], REGISTRY_ALLOW):
            return False, f"registry[{carrier}] differs beyond allowlist", report

    # identity-tolerant ledgers: count + ordered EXACT-allowlist projection (C2);
    # REV 2.3: attempts rows may ADDITIONALLY diverge only as the exact transient
    # pair, each allowance enumerated in the report (never silent)
    for rel, allow in (("acquisition_attempts.jsonl", ATT_ALLOW),
                       ("operation_ledger.jsonl", OPS_ALLOW)):
        a = _read_jsonl(held, rel)
        b = _read_jsonl(repl, rel)
        if len(a) != len(b):
            return False, f"{rel} row count {len(a)} != {len(b)}", report
        transient = []
        for i, (ra, rb) in enumerate(zip(a, b)):
            if canon(_project(ra, allow)) == canon(_project(rb, allow)):
                continue
            if rel == "acquisition_attempts.jsonl" and _transient_label_pair(ra, rb):
                # REV 2.4: BOTH labels recorded; neither overwritten nor reported
                # as literal equality (codex condition 6)
                transient.append({
                    "key": (ra.get("carrier_key"), ra.get("scored_day"),
                            ra.get("station_id")),
                    "held": {"selected_nslc": ra.get("selected_nslc"),
                             "reason_codes": ra.get("reason_codes")},
                    "sealed": {"selected_nslc": rb.get("selected_nslc"),
                               "reason_codes": rb.get("reason_codes")}})
                continue
            return False, f"{rel} label projection differs at row {i}", report
        if rel == "acquisition_attempts.jsonl":
            report["projected_attempts"] = transient
            report["quotient_declared"] = [tuple(k) for k in quotient_keys]
            # REV 2.4 exact-set gate: projected keys == declared quotient, no
            # duplicates, no eighth key, no missing key (codex 69b801f)
            pkeys = [t["key"] for t in transient]
            if len(pkeys) != len(set(pkeys)):
                return False, "duplicate projected quotient key", report
            declared = {tuple(k) for k in quotient_keys}
            extra = sorted(set(pkeys) - declared)
            missing = sorted(declared - set(pkeys))
            if extra:
                return False, (f"projected attempt key(s) outside the declared "
                               f"quotient: {extra}"), report
            if missing:
                return False, (f"declared quotient key(s) missing from the "
                               f"projection: {missing}"), report

    # batch_manifest typed (C3): scientific surface EXACT, artifacts sub-map for
    # byte-equal-class files EXACT, named identity/attestation/induced fields dropped
    hb = _read_json(held, "batch_manifest.json")
    rb = _read_json(repl, "batch_manifest.json")
    if _project(hb, BM_ALLOW) != _project(rb, BM_ALLOW):
        keys = sorted(k for k in set(hb) | set(rb) if k not in BM_ALLOW
                      and hb.get(k) != rb.get(k))
        return False, f"batch_manifest differs beyond allowlist: {keys}", report
    hart = {k: v for k, v in (hb.get("artifacts") or {}).items()
            if _classify(k) == "byte"}
    rart = {k: v for k, v in (rb.get("artifacts") or {}).items()
            if _classify(k) == "byte"}
    if hart != rart:
        keys = sorted(k for k in set(hart) | set(rart) if hart.get(k) != rart.get(k))
        return False, f"batch_manifest artifacts differ on byte-equal files: {keys[:3]}", report

    proj = report.get("projected_attempts") or []
    pname = "sealed-unavailable-quotient-v1" if quotient_keys else "none"
    return True, (f"EQUIVALENT: science_daily_differing="
                  f"{report['daily_differing']}/{report['daily_total']}, "
                  f"{report.get('objects_held')} objects identity-equal, "
                  f"attempt_projection={pname}, projected_attempts={len(proj)}: "
                  f"{[t['key'] for t in proj]}"), report


# ---- synthetic lock fixtures --------------------------------------------------------
def _receipt(ts, quote_seed):
    return {"status": "VERIFIED_DIRECT", "in_session_timestamp_utc": ts,
            "owner_quote_sha256": sha(quote_seed.encode())}


def _mk_root(td, *, pid, receipt, commit, blobs_tag):
    os.makedirs(os.path.join(td, "capsules"), exist_ok=True)
    os.makedirs(os.path.join(td, "raw_objects"), exist_ok=True)
    os.makedirs(os.path.join(td, "publication_records"), exist_ok=True)
    days = ["2026-03-02", "2026-03-03", "2026-03-04"]
    plan = {"contract_id": "codex-d2-campaign-v2-renewal-2026-08-16-v1",
            "carriers": ["c1"], "scheduled_days": days,
            "activation_reference_day": "2026-08-16",
            "core_blobs": {f: blobs_tag * 40 for f in PLAN_CORE_FILES},
            "created_utc": f"t-{pid}", "registered_utc": f"t-{pid}"}
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
    obj = {"sha256": sha(raw), "size": len(raw), "relative_path": f"raw_objects/{sha(raw)}.ms",
           "kind": "archive-seismic-miniseed-fragments-v1", "carrier_key": "c1",
           "scored_day": "2026-03-02", "segment_name": "seg_a", "source_id": "KO.S00..BHZ",
           "start_utc": "2026-03-01T07:00:13.094647Z", "end_utc": "2026-03-02T07:00:13.094647Z",
           "native_rate_hz": 25.0, "npts": 1170000, "fragment_count": 1,
           "support_sha256": "b" * 64, "publication_record_sha256": "c" * 64,
           "reuse_disposition": "FRESH" if pid == "p-held" else "REUSED_VERIFIED",
           "acquired_process_id": pid, "verified_process_id": pid}
    files = {
        "campaign_plan.json": canon(plan),
        "published_phase_ledger.json": canon(
            {"rows": [{"carrier_key": "c1", "scored_day": "2026-03-02"}]}),
        "replay_metrics.json": canon({"replay": "metrics", "sha": "d" * 64}),
        "prior_evidence.json": canon({"prior": True}),
        "d2_diagnostic_result.json": canon({"diag": 1}),
        "publication_records/2026-03-02.json": canon({"pub": "2026-03-02", "sha": "c" * 64}),
        "admission_results.json": canon(
            {"schema": "adm-v1", "implementation_commit": "292b1069" + "0" * 32,
             "regions": [{"runner_key": "c1", "carrier_key": "c1",
                          "status": "ADMITTED_CANDIDATE", "incident_threshold": 0.31,
                          "activation_threshold": 0.31, "incident_n": 2,
                          "activation_n": 2, "reason_codes": [],
                          "capsule_path": "capsules/c1_calibration.json",
                          "capsule_sha256": sha(commit.encode())}]}),
        "capsules/c1_calibration.json": canon(
            {"schema": "geospec-d2-calibration-v1", "region": "c1", "threshold": 0.31,
             "calibration_window": {"start": "2026-03-02", "end": "2026-07-17"},
             "replay_output_sha256": "d" * 64, "valid_through": "2026-08-23",
             "source_commit": commit, "input_manifest_sha256": sha(pid.encode()),
             "issued_utc": f"t-{pid}"}),
        "input_manifest.json": canon(
            {"schema": "im-v2-resume", "producer_commit": commit,
             "implementation_commit": commit, "objects": [obj]}),
        "registry_candidate.json": canon(
            {"c1": {"capsule_path": "capsules/c1_calibration.json",
                    "expected_sha256": sha(commit.encode()), "topology_version": "t1"}}),
        "acquisition_attempts.jsonl": b"\n".join(canon(r) for r in [
            {"carrier_key": "c1", "scored_day": d, "segment_name": "seg_a",
             "station_id": "KO.S00", "provider": "KOERI",
             "request_start_utc": "2026-03-01T07:00:13.094647Z",
             "request_end_utc": f"{d}T07:00:13.094647Z",
             "selected_nslc": "KO.S00..BHZ", "status": "FETCHED",
             "input_object_sha256s": [sha(raw)], "reason_codes": [],
             "attempted_utc": f"t-{pid}", "attempt_id": f"a-{pid}-{d}",
             "terminal_attempt_id": f"a-{pid}-{d}", "indeterminate_attempt_count": 0,
             "process_id": pid} for d in days[:2]]),
        "operation_ledger.jsonl": canon(
            {"arm": "incident", "carrier_key": "c1", "day": "2026-03-02",
             "op": "SCORE", "operation_id": f"o-{pid}"}),
        "campaign_process_ledger.jsonl": canon(
            {"process_id": pid, "ordinal": 1, "resume_of": None,
             "producer_commit": commit, "campaign_id": sha(canon(plan)),
             "owner_launch_authorization": receipt, "disposition": "COMPLETED",
             "failure_type": None, "process_started_utc": f"t0-{pid}",
             "process_ended_utc": f"t9-{pid}", "observed_dead_utc": None,
             "first_event_sha256": "1" * 64, "last_event_sha256": "2" * 64}),
        "resume_state.json": canon(
            {"events": [{"kind": "PROCESS_STARTED", "process_id": pid,
                         "owner_launch_authorization": receipt,
                         "campaign_id": sha(canon(plan))},
                        {"kind": "PROCESS_ENDED", "process_id": pid}]}),
        "resume_state.head.json": canon(
            {"generation": 1, "event_count": 2, "last_event_sha256": "2" * 64}),
        f"raw_objects/{sha(raw)}.ms": raw,
        "calibration_daily.jsonl": b"\n".join(canon(r) for r in rows),
    }
    for rel, body in files.items():
        path = os.path.join(td, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(body)
    # batch_manifest LAST: real artifacts walk of what was just written
    artifacts = {}
    for rel in _walk_rel(td):
        p = os.path.join(td, rel)
        with open(p, "rb") as fh:
            body = fh.read()
        artifacts[rel] = {"sha256": sha(body), "size": len(body)}
    bm = {"schema": "geospec-d2-step4b-batch-v1",
          "contract_id": "codex-d2-campaign-v2-renewal-2026-08-16-v1",
          "run_id": sha(canon(plan)), "campaign_id": sha(canon(plan)),
          "producer_commit": commit, "implementation_commit": commit,
          "clean_tree": True, "band_tag": "bp_1_10_env1hz",
          "processing_version": "pv", "topology_version": "t1",
          "activation_reference_day": "2026-08-16", "incident_reference_day": "2026-07-29",
          "calibration_policy": {"min_admitted_days": 60}, "targets": [{"runner_key": "c1"}],
          "environment": {"python": "3.14.0"}, "implementation_blobs": ["monitoring/src/x.py"],
          "artifacts": artifacts, "campaign_started_utc": f"t0-{pid}",
          "created_utc": f"t9-{pid}", "campaign_plan_sha256": sha(canon(plan)),
          "owner_launch_authorization": receipt, "resumable": True,
          "process_count": 1, "state_head_generation": 1,
          "production_registry_modified": False, "production_freezes_modified": False,
          "non_claims": ["registry status only"]}
    with open(os.path.join(td, "batch_manifest.json"), "wb") as fh:
        fh.write(canon(bm))


def _mk_pair(mutate=None):
    """held + replacement differing ONLY on the allowlist surface: fresh valid receipt,
    new process identity, repaired producer commit -> new core_blobs/attestations/
    induced digests. Nominal MUST pass; mutate(repl_dir) applies a violation."""
    th = tempfile.mkdtemp()
    tr = tempfile.mkdtemp()
    _mk_root(th, pid="p-held", receipt=_receipt("2026-08-16T01:00:00+00:00", "held-go"),
             commit="held-commit", blobs_tag="a")
    _mk_root(tr, pid="p-repl", receipt=_receipt("2026-08-17T19:00:00+00:00", "fresh-go"),
             commit="repl-commit", blobs_tag="b")
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


def _att_pair_row(pid, *, selected, reasons, status="UNAVAILABLE", shas=(),
                  station="KO.T01", day="2026-03-04"):
    """REV 2.3 fixture: an appended attempts row whose non-identity fields are
    byte-equal across roots except the mutated pair surface under test."""
    return {"carrier_key": "c1", "scored_day": day, "segment_name": "seg_a",
            "station_id": station, "provider": "KOERI",
            "request_start_utc": "2026-03-03T07:00:13.094647Z",
            "request_end_utc": f"{day}T07:00:13.094647Z",
            "selected_nslc": selected, "status": status,
            "input_object_sha256s": list(shas), "reason_codes": list(reasons),
            "attempted_utc": f"t-{pid}", "attempt_id": f"tp-{pid}",
            "terminal_attempt_id": f"tp-{pid}", "indeterminate_attempt_count": 0,
            "process_id": pid}


def main():
    th, tr = _mk_pair()
    ok_nom, det_nom, rep_nom = validate_replay_equivalence(th, tr)
    check("RE-1a nominal replacement (allowlist-only: FRESH VALID receipt, new process "
          "identity, repaired producer commit + core_blobs + induced digests, plan "
          "issuance) is EQUIVALENT -- incl. codex C3 attestation-only PASS lock",
          ok_nom, det_nom)

    def refused(mut):
        h, r = _mk_pair(mutate=mut)
        ok, det, _ = validate_replay_equivalence(h, r)
        return (not ok), det

    # REV 2.3/2.4 nominal: exactly ONE transient pair (held live-transient with the
    # selected candidate, sealed truthful no-candidate) under a DECLARED one-key
    # quotient must PASS with BOTH labels recorded in the receipt
    T01_KEY = ("c1", "2026-03-04", "KO.T01")
    th2, tr2 = _mk_pair()
    _edit_jsonl(th2, "acquisition_attempts.jsonl", lambda rows: rows.append(
        _att_pair_row("p-held", selected="KO.T01..BHZ",
                      reasons=["PROVIDER_UNAVAILABLE"])))
    _edit_jsonl(tr2, "acquisition_attempts.jsonl", lambda rows: rows.append(
        _att_pair_row("p-repl", selected=None, reasons=["NO_AVAILABLE_CANDIDATE"])))
    ok_tp, det_tp, rep_tp = validate_replay_equivalence(th2, tr2,
                                                        quotient_keys=(T01_KEY,))
    proj_tp = rep_tp.get("projected_attempts") or []
    check("RE-1c REV 2.4 transient-label pair under a DECLARED one-key quotient "
          "(held PROVIDER_UNAVAILABLE + selected candidate vs sealed "
          "NO_AVAILABLE_CANDIDATE + None; terminal UNAVAILABLE, zero input objects "
          "BOTH) PASSES with BOTH labels recorded in the receipt",
          ok_tp and [t["key"] for t in proj_tp] == [T01_KEY]
          and proj_tp[0]["held"] == {"selected_nslc": "KO.T01..BHZ",
                                     "reason_codes": ["PROVIDER_UNAVAILABLE"]}
          and proj_tp[0]["sealed"] == {"selected_nslc": None,
                                       "reason_codes": ["NO_AVAILABLE_CANDIDATE"]}
          and "attempt_projection=sealed-unavailable-quotient-v1" in det_tp
          and "projected_attempts=1" in det_tp
          and "science_daily_differing=0/6" in det_tp,
          f"{det_tp} report={rep_tp}")

    def refused_att_pair(held_row, repl_row, rel="acquisition_attempts.jsonl",
                         quotient=(T01_KEY,)):
        h, r = _mk_pair()
        _edit_jsonl(h, rel, lambda rows: rows.append(held_row))
        _edit_jsonl(r, rel, lambda rows: rows.append(repl_row))
        ok, det, _ = validate_replay_equivalence(h, r, quotient_keys=quotient)
        return (not ok), det

    battery = []
    # -- original 14 non-allowlisted violation classes
    battery.append(("ratio nudged 1e-9", *refused(lambda t: _edit_jsonl(
        t, DAILY, lambda rows: rows[0].update(ratio=rows[0]["ratio"] + 1e-9)))))
    battery.append(("row order reversed", *refused(lambda t: _edit_jsonl(
        t, DAILY, lambda rows: rows.reverse()))))
    battery.append(("row dropped", *refused(lambda t: _edit_jsonl(
        t, DAILY, lambda rows: rows.pop()))))
    battery.append(("status flip", *refused(lambda t: _edit_jsonl(
        t, DAILY, lambda rows: rows[0].update(status="REJECTED")))))
    battery.append(("reason-code change", *refused(lambda t: _edit_jsonl(
        t, DAILY, lambda rows: rows[-1].update(qc_reasons=["OTHER"])))))
    battery.append(("eigenvalue drift", *refused(lambda t: _edit_jsonl(
        t, DAILY, lambda rows: rows[0].update(ordered_eigenvalues=[1.4000001, 0.5999999])))))
    battery.append(("support-count shift", *refused(lambda t: _edit_jsonl(
        t, DAILY, lambda rows: rows[0].update(common_support_count=78528)))))
    battery.append(("threshold moved (admission)", *refused(lambda t: _edit_json(
        t, "admission_results.json",
        lambda d: d["regions"][0].update(activation_threshold=0.315)))))
    battery.append(("capsule threshold changed", *refused(lambda t: _edit_json(
        t, "capsules/c1_calibration.json", lambda d: d.update(threshold=0.315)))))
    battery.append(("raw object bytes differ", *refused(lambda t: open(
        os.path.join(t, "raw_objects", os.listdir(os.path.join(t, "raw_objects"))[0]),
        "wb").write(b"tampered"))))
    battery.append(("plan scheduled_days differ (non-allowlisted plan field)",
                    *refused(lambda t: _edit_json(
                        t, "campaign_plan.json",
                        lambda d: d.update(scheduled_days=d["scheduled_days"][:2])))))
    battery.append(("out-of-allowlist file added", *refused(
        lambda t: open(os.path.join(t, "extra_report.json"), "wb").write(b"{}"))))
    battery.append(("attempt label flip", *refused(lambda t: _edit_jsonl(
        t, "acquisition_attempts.jsonl", lambda rows: rows[0].update(status="REFUSED")))))
    battery.append(("provider-object sha changed", *refused(lambda t: _edit_json(
        t, "input_manifest.json", lambda d: d["objects"][0].update(sha256="b" * 64)))))
    # -- REV 2 locks: codex C2 counterexamples
    battery.append(("object source_id changed (codex C2)", *refused(lambda t: _edit_json(
        t, "input_manifest.json",
        lambda d: d["objects"][0].update(source_id="XX.OTHER..BHZ")))))
    battery.append(("attempt station_id changed (codex C2)", *refused(lambda t: _edit_jsonl(
        t, "acquisition_attempts.jsonl",
        lambda rows: rows[0].update(station_id="XX.OTHER")))))
    # -- REV 2 locks: codex C3
    battery.append(("batch_manifest deleted (codex C3)", *refused(
        lambda t: os.remove(os.path.join(t, "batch_manifest.json")))))
    battery.append(("batch_manifest policy changed", *refused(lambda t: _edit_json(
        t, "batch_manifest.json",
        lambda d: d.update(calibration_policy={"min_admitted_days": 2})))))
    # -- REV 2 locks: grassmann G1
    battery.append(("publication record tampered (G1)", *refused(lambda t: _edit_json(
        t, "publication_records/2026-03-02.json", lambda d: d.update(pub="tampered")))))
    battery.append(("publication record missing one side (codex 1e4efb7)", *refused(
        lambda t: os.remove(os.path.join(t, "publication_records", "2026-03-02.json")))))
    battery.append(("publication record added one side (codex 1e4efb7)", *refused(
        lambda t: open(os.path.join(t, "publication_records", "2026-03-05.json"),
                       "wb").write(canon({"pub": "extra"})))))
    # -- REV 2 locks: codex C1 receipt battery
    battery.append(("receipt missing (C1)", *refused(lambda t: _edit_jsonl(
        t, "campaign_process_ledger.jsonl",
        lambda rows: rows[0].pop("owner_launch_authorization")))))
    battery.append(("receipt RELAYED (C1)", *refused(lambda t: _edit_jsonl(
        t, "campaign_process_ledger.jsonl",
        lambda rows: rows[0]["owner_launch_authorization"].update(status="RELAYED")))))
    battery.append(("receipt bad-UTC (C1)", *refused(lambda t: _edit_jsonl(
        t, "campaign_process_ledger.jsonl",
        lambda rows: rows[0]["owner_launch_authorization"].update(
            in_session_timestamp_utc="not-utc")))))
    battery.append(("receipt bad-hash (C1)", *refused(lambda t: _edit_jsonl(
        t, "campaign_process_ledger.jsonl",
        lambda rows: rows[0]["owner_launch_authorization"].update(
            owner_quote_sha256="zz")))))
    battery.append(("receipt WAL-binding mismatch (C1)", *refused(lambda t: _edit_json(
        t, "resume_state.json",
        lambda d: d["events"][0]["owner_launch_authorization"].update(
            owner_quote_sha256="e" * 64)))))
    battery.append(("batch receipt unbound to COMPLETED row (C1)", *refused(
        lambda t: _edit_json(
            t, "batch_manifest.json",
            lambda d: d["owner_launch_authorization"].update(
                owner_quote_sha256="f" * 64)))))
    # -- REV 2.2 locks: codex e127a7f #1 (plan core_blobs integrity)
    battery.append(("plan core_blobs deleted (2.2-1)", *refused(lambda t: _edit_json(
        t, "campaign_plan.json", lambda d: d.pop("core_blobs")))))
    battery.append(("plan core_blobs wrong keyset (2.2-1)", *refused(
        lambda t: _edit_json(t, "campaign_plan.json", lambda d: d.update(
            core_blobs={"monitoring/src/unrelated.py": "b" * 40})))))
    battery.append(("plan core_blobs extra key (2.2-1)", *refused(
        lambda t: _edit_json(t, "campaign_plan.json", lambda d: d["core_blobs"].update(
            {"monitoring/src/extra.py": "b" * 40})))))
    battery.append(("plan core_blobs malformed blob id (2.2-1)", *refused(
        lambda t: _edit_json(t, "campaign_plan.json", lambda d: d["core_blobs"].update(
            {PLAN_CORE_FILES[0]: "not-a-git-blob"})))))
    # -- REV 2.2 locks: codex e127a7f #2 (root-local WAL/ledger closure)
    battery.append(("orphan invalid WAL PROCESS_STARTED (2.2-2)", *refused(
        lambda t: _edit_json(t, "resume_state.json", lambda d: d["events"].append(
            {"kind": "PROCESS_STARTED", "process_id": "p-orphan",
             "owner_launch_authorization": {"status": "RELAYED",
                                            "in_session_timestamp_utc": "not-utc",
                                            "owner_quote_sha256": "zz"}})))))
    battery.append(("duplicate ledger row, stale process_count (2.2-2)", *refused(
        lambda t: _edit_jsonl(t, "campaign_process_ledger.jsonl",
                              lambda rows: rows.append(dict(rows[0]))))))
    # -- REV 2.3 locks: the transient pair is the ONLY admissible label divergence
    battery.append(("transient pair wrong live reason FETCH_ERROR (2.3)",
                    *refused_att_pair(
                        _att_pair_row("p-held", selected="KO.T01..BHZ",
                                      reasons=["FETCH_ERROR:ConnectionError"]),
                        _att_pair_row("p-repl", selected=None,
                                      reasons=["NO_AVAILABLE_CANDIDATE"]))))
    battery.append(("transient pair wrong sealed reason (2.3)", *refused_att_pair(
        _att_pair_row("p-held", selected="KO.T01..BHZ",
                      reasons=["PROVIDER_UNAVAILABLE"]),
        _att_pair_row("p-repl", selected=None, reasons=["PROVIDER_UNAVAILABLE"]))))
    battery.append(("transient pair extra sealed reason code (2.3)", *refused_att_pair(
        _att_pair_row("p-held", selected="KO.T01..BHZ",
                      reasons=["PROVIDER_UNAVAILABLE"]),
        _att_pair_row("p-repl", selected=None,
                      reasons=["NO_AVAILABLE_CANDIDATE", "EXTRA"]))))
    battery.append(("transient pair direction reversed (2.3)", *refused_att_pair(
        _att_pair_row("p-held", selected=None, reasons=["NO_AVAILABLE_CANDIDATE"]),
        _att_pair_row("p-repl", selected="KO.T01..BHZ",
                      reasons=["PROVIDER_UNAVAILABLE"]))))
    battery.append(("transient pair data-bearing held row (2.3)", *refused_att_pair(
        _att_pair_row("p-held", selected="KO.T01..BHZ",
                      reasons=["PROVIDER_UNAVAILABLE"], shas=["ee" * 32]),
        _att_pair_row("p-repl", selected=None, reasons=["NO_AVAILABLE_CANDIDATE"]))))
    battery.append(("transient pair data-bearing sealed row (2.3)", *refused_att_pair(
        _att_pair_row("p-held", selected="KO.T01..BHZ",
                      reasons=["PROVIDER_UNAVAILABLE"]),
        _att_pair_row("p-repl", selected=None, reasons=["NO_AVAILABLE_CANDIDATE"],
                      shas=["ee" * 32]))))
    battery.append(("transient pair non-UNAVAILABLE status both sides (2.3)",
                    *refused_att_pair(
                        _att_pair_row("p-held", selected="KO.T01..BHZ",
                                      reasons=["PROVIDER_UNAVAILABLE"],
                                      status="ERROR"),
                        _att_pair_row("p-repl", selected=None,
                                      reasons=["NO_AVAILABLE_CANDIDATE"],
                                      status="ERROR"))))
    battery.append(("transient pair held selected_nslc None (2.3)", *refused_att_pair(
        _att_pair_row("p-held", selected=None, reasons=["PROVIDER_UNAVAILABLE"]),
        _att_pair_row("p-repl", selected=None, reasons=["NO_AVAILABLE_CANDIDATE"]))))
    battery.append(("transient pair station identity mismatch (2.3)",
                    *refused_att_pair(
                        _att_pair_row("p-held", selected="KO.T01..BHZ",
                                      reasons=["PROVIDER_UNAVAILABLE"]),
                        _att_pair_row("p-repl", selected=None,
                                      reasons=["NO_AVAILABLE_CANDIDATE"],
                                      station="KO.T02"))))
    _op_pair = lambda sel, rsn: {"arm": "incident", "carrier_key": "c1",  # noqa: E731
                                 "day": "2026-03-04", "op": "ACQUIRE",
                                 "operation_id": "o-shared",
                                 "status": "UNAVAILABLE",
                                 "input_object_sha256s": [],
                                 "selected_nslc": sel, "reason_codes": rsn}
    battery.append(("transient pair in operation_ledger REFUSES (2.3)",
                    *refused_att_pair(
                        _op_pair("KO.T01..BHZ", ["PROVIDER_UNAVAILABLE"]),
                        _op_pair(None, ["NO_AVAILABLE_CANDIDATE"]),
                        rel="operation_ledger.jsonl", quotient=())))
    # -- REV 2.4 locks: exact-set quotient gate (codex 69b801f)
    battery.append(("well-formed pair under EMPTY/undeclared quotient (2.4)",
                    *refused_att_pair(
                        _att_pair_row("p-held", selected="KO.T01..BHZ",
                                      reasons=["PROVIDER_UNAVAILABLE"]),
                        _att_pair_row("p-repl", selected=None,
                                      reasons=["NO_AVAILABLE_CANDIDATE"]),
                        quotient=())))
    battery.append(("declared quotient key MISSING from projection (2.4)",
                    *refused_att_pair(
                        _att_pair_row("p-held", selected="KO.T01..BHZ",
                                      reasons=["PROVIDER_UNAVAILABLE"]),
                        _att_pair_row("p-repl", selected=None,
                                      reasons=["NO_AVAILABLE_CANDIDATE"]),
                        quotient=(T01_KEY, ("c1", "2026-03-04", "KO.T09")))))

    def refused_att_pairs(held_rows, repl_rows, quotient):
        h, r = _mk_pair()
        _edit_jsonl(h, "acquisition_attempts.jsonl",
                    lambda rows: rows.extend(held_rows))
        _edit_jsonl(r, "acquisition_attempts.jsonl",
                    lambda rows: rows.extend(repl_rows))
        ok, det, _ = validate_replay_equivalence(h, r, quotient_keys=quotient)
        return (not ok), det

    battery.append(("EIGHTH key: second well-formed pair outside declaration (2.4)",
                    *refused_att_pairs(
                        [_att_pair_row("p-held", selected="KO.T01..BHZ",
                                       reasons=["PROVIDER_UNAVAILABLE"]),
                         _att_pair_row("p-held2", selected="KO.T02..BHZ",
                                       reasons=["PROVIDER_UNAVAILABLE"],
                                       station="KO.T02")],
                        [_att_pair_row("p-repl", selected=None,
                                       reasons=["NO_AVAILABLE_CANDIDATE"]),
                         _att_pair_row("p-repl2", selected=None,
                                       reasons=["NO_AVAILABLE_CANDIDATE"],
                                       station="KO.T02")],
                        quotient=(T01_KEY,))))
    battery.append(("DUPLICATE projected quotient key (2.4)",
                    *refused_att_pairs(
                        [_att_pair_row("p-held", selected="KO.T01..BHZ",
                                       reasons=["PROVIDER_UNAVAILABLE"]),
                         _att_pair_row("p-held2", selected="KO.T01..BHZ",
                                       reasons=["PROVIDER_UNAVAILABLE"])],
                        [_att_pair_row("p-repl", selected=None,
                                       reasons=["NO_AVAILABLE_CANDIDATE"]),
                         _att_pair_row("p-repl2", selected=None,
                                       reasons=["NO_AVAILABLE_CANDIDATE"])],
                        quotient=(T01_KEY,))))
    battery.append(("changed candidate: held selected_nslc not bound to its own "
                    "station (2.4)", *refused_att_pair(
                        _att_pair_row("p-held", selected="KO.OTHER..BHZ",
                                      reasons=["PROVIDER_UNAVAILABLE"]),
                        _att_pair_row("p-repl", selected=None,
                                      reasons=["NO_AVAILABLE_CANDIDATE"]))))

    all_refused = all(v for _, v, _ in battery)
    check("RE-1b violation battery: every non-allowlisted difference REFUSES "
          f"({len(battery)} classes incl. codex C1 receipt locks, C2 source_id/"
          "station_id, C3 manifest presence+policy, G1 publication records, "
          "REV 2.3 transient-pair locks, REV 2.4 quotient-gate locks)",
          all_refused,
          "; ".join(f"{n}: {'ok' if v else 'NOT REFUSED'}" for n, v, _ in battery))

    held_env = os.environ.get("D2_HELD_ROOT")
    repl_env = os.environ.get("D2_REPLACEMENT_ROOT")
    if held_env and repl_env and os.path.isdir(held_env) and os.path.isdir(repl_env):
        # REV 2.4: the real-pair lane declares EXACTLY the seven predeclared
        # remint3 keys (sealed-unavailable-quotient-v1, codex 69b801f)
        ok, det, rep = validate_replay_equivalence(
            held_env, repl_env, quotient_keys=SEALED_UNAVAILABLE_QUOTIENT_V1)
        check("RE-2 real-pair exact scientific equivalence (held vs replacement) "
              "under sealed-unavailable-quotient-v1 (the predeclared 7 keys)",
              ok, f"{det} report={rep}")
    else:
        check("RE-2 real-pair exact scientific equivalence (held vs replacement)",
              False, "PENDING PAIR (set D2_HELD_ROOT + D2_REPLACEMENT_ROOT; the "
                     "RED run is (held, no-standing remint) which MUST FAIL on the "
                     "drift signature ~596/720; the GREEN run is (held, corrected "
                     "replacement) which MUST PASS)")


main()
print()
if FAILS:
    print(f"D2 REPLAY-EQUIVALENCE RED-KAT FAILURES: {FAILS}")
    sys.exit(1)
print("ALL D2 REPLAY-EQUIVALENCE RED-KATs PASS")
