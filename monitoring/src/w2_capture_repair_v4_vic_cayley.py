#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V4 OFFLINE VIC REPLAY (cayley) -- codex 0510Z P0-2.

**ZERO NEW HTTP. This driver never opens a socket.**

WHAT HAPPENED, AND WHY THIS IS A REPAIR RATHER THAN A RE-CAPTURE
---------------------------------------------------------------
On 2026-08-27 all 212 `MAG_FEED/vic` keys were fetched successfully and
then refused at the transform:

    ADMISSION_TRANSFORM_REFUSED: GIN MAG: source orientation 'XYZS' is
    not a registered REPORTED convention and no committed frame capsule
    exists at .../mag_capsules/mag_capsule_vic.json

The refusal was correct: `w2_mag1.REPORTED_CONVENTIONS` registers only
`XYZF`, so VIC falls through to the execution-capsule branch of
`_mag_frame_authority`, which found nothing at the resolved path. The
capsule was never unauthored -- it is committed at
`docs/f2g_window2_freeze/mag_capsule_vic.json` and was simply never
copied into the execution tree when codex `0451Z` relocated IZN/FRN/TUC.

Crucially the refusal fired AFTER the bytes were persisted: in
`capture_day` the order is write transcript -> verify_transcript ->
artifact_builder (refuses here) -> build record -> write record. So all
212 raw bodies are in the named store, content-addressed, with their
digests bound in the preserved transcripts.

**Therefore completing VIC costs zero requests.** This driver replays
bytes already held, exactly as the v3 driver
(`w2_capture_repair_grassmann.py`) did for the Kp vocabulary class.
That historical v3 driver is NOT mutated and its v3 restage relation is
NOT used here.

PROOF KIND -- deliberately unchanged
------------------------------------
These stay `NATIVE_V4_CAPTURE`. They are native HTTP exchanges that were
genuinely performed and are being repaired after a transform refusal.
They are NOT `RESTAGED_LINEAGE`, and this driver does not invent a new
proof kind to describe them. A repair that quietly re-labels its
evidence class is how a weaker provenance gets smuggled into a stronger
slot.

CLAIM CEILING
-------------
Completing these 212 keys does NOT close the 2,056-key boundary. Even
after this replay and the predecessor bridge the maximum is 2,055
complete four-class keys, because the one USGS 404 has no body and no
transcript and therefore cannot satisfy a body-backed scientific key.
That is an owner decision (codex 0510Z P0-3), not something this driver
may paper over, and nothing here emits BOUND or claims 635 native keys.
"""
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import w2_acquisition_capture_grassmann as CAP
import w2_accrual_instrument_cayley as ACC
import w2_producer_grassmann as PROD
import w2_capture_run_v4_grassmann as RUN4

REPO = RUN4.REPO
STAGED_DIR = RUN4.STAGED_DIR
STORE = RUN4.STORE_PHYSICAL
LEDGER = RUN4.LEDGER
# derived from the admission consumer's registered path (codex 1547Z
# repair 1: one authority), so the generator's C11d equality KAT now
# checks derivation plumbing rather than two hand-typed literals
REPAIR_LEDGER = os.path.join(REPO,
                             *ACC.VIC_REPAIR_LEDGER_PATH.split("/"))
# codex 2313Z P0-3: per-key CREATE-ONCE repair receipts; the 212-line
# ledger is generated ATOMICALLY from these objects -- never an
# appended mutable tail that a crash can truncate mid-line
RECEIPTS_DIR = os.path.join(REPO, "docs", "f2g_window2_execution",
                            "vic_repair_receipts")

TARGET_LANE = "MAG_FEED"
TARGET_CARRIER = "vic"
# the exact refusal class this driver repairs, and nothing else
TARGET_MARK = "is not a registered REPORTED convention"
EXPECTED_TARGET_COUNT = 212
PROOF_KIND = "NATIVE_V4_CAPTURE"
REPAIR_ID = "vic-frame-capsule-registration-20260827"

# the production-resolved execution capsule path. Imported from the
# production module rather than retyped so this driver cannot drift
# from the path the transform actually reads.
EXEC_CAPSULE_REL = (f"{CAP.EXEC_CAPSULE_DIR}/"
                    f"mag_capsule_{TARGET_CARRIER}.json")


class VicReplayRefusal(Exception):
    """Typed, fail-closed. The code leads the message."""


def _refuse(code, detail):
    raise VicReplayRefusal(f"{code}: {detail}")


def _stem(lane, ck, day):
    """v4 stems come from the PRODUCTION token formula -- unlike the v3
    driver, which had to carry a frozen historical formula because the
    v4 dispatcher retires its lane names."""
    return CAP._path_tokens(lane, ck, day)


def _cls_path(staged_dir, stem, cls):
    return os.path.join(staged_dir, stem + ACC.STAGED_CLASS_SUFFIX[cls])


def default_store_reader(store_dir):
    def read(sha):
        p = os.path.join(store_dir, sha + ".body")
        if not os.path.isfile(p):
            _refuse("VIC_BODY_ABSENT",
                    f"no store body for {sha[:12]} at {store_dir}")
        with open(p, "rb") as f:
            raw = f.read()
        got = hashlib.sha256(raw).hexdigest()
        if got != sha:
            _refuse("VIC_BODY_DIGEST_DIVERGENT",
                    f"store body {sha[:12]} recomputes {got[:12]}")
        return raw
    return read


def load_ledger(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def target_set(plan_keys, ledger_rows, *, expect=EXPECTED_TARGET_COUNT):
    """The EXACT target set, derived from the pinned plan AND the
    terminal ledger, refusing every divergence codex enumerated.

    Neither source is trusted alone: the plan says what was authorized,
    the ledger says what actually refused, and the driver only proceeds
    where they agree exactly. A ledger-only derivation would let a
    refusal for an unauthorized key pull that key into the repair.
    """
    plan_vic = {k for k in plan_keys
                if k.split("/")[0] == TARGET_LANE
                and k.split("/")[1] == TARGET_CARRIER}

    marked, non_vic, mistyped = set(), set(), set()
    for r in ledger_rows:
        if TARGET_MARK not in str(r.get("refusal", "")):
            continue
        key = r["key"]
        if r.get("status") != "REFUSED":
            mistyped.add(key)
            continue
        lane, ck = key.split("/")[0], key.split("/")[1]
        if lane != TARGET_LANE or ck != TARGET_CARRIER:
            non_vic.add(key)
            continue
        marked.add(key)

    if non_vic:
        _refuse("VIC_NON_TARGET_KEY_IN_REFUSAL_CLASS",
                f"{len(non_vic)} non-VIC key(s) carry the frame refusal "
                f"mark: {sorted(non_vic)[:3]}")
    if mistyped:
        _refuse("VIC_REFUSAL_STATUS_MISTYPED",
                f"{len(mistyped)} key(s) carry the mark without "
                f"status REFUSED: {sorted(mistyped)[:3]}")
    missing = plan_vic - marked
    extra = marked - plan_vic
    if missing:
        _refuse("VIC_TARGET_SET_INCOMPLETE",
                f"{len(missing)} planned VIC key(s) have no frame "
                f"refusal in the terminal ledger: {sorted(missing)[:3]}")
    if extra:
        _refuse("VIC_TARGET_SET_UNPLANNED",
                f"{len(extra)} refused VIC key(s) are not in the pinned "
                f"plan: {sorted(extra)[:3]}")
    if expect is not None and len(marked) != expect:
        _refuse("VIC_TARGET_COUNT_DIVERGENT",
                f"expected exactly {expect} target keys, derived "
                f"{len(marked)}")
    return sorted(marked)


def capsule_binding(repo=REPO, rel=EXEC_CAPSULE_REL):
    """The execution capsule must be present at the RESOLVED path and is
    bound by digest into every repair record. Presence at any other path
    satisfies nothing -- that mismatch is the whole defect being
    repaired."""
    full = os.path.join(repo, *rel.split("/"))
    if not os.path.isfile(full):
        _refuse("VIC_EXEC_CAPSULE_ABSENT",
                f"no capsule at the production-resolved path {rel} -- "
                "registration must land before replay")
    with open(full, "rb") as f:
        raw = f.read()
    try:
        obj = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        _refuse("VIC_EXEC_CAPSULE_UNPARSEABLE", f"{rel}: {exc}")
    if not isinstance(obj, dict) or not obj.get("component_map") or \
            not obj.get("sensor_orientation"):
        _refuse("VIC_EXEC_CAPSULE_NOT_A_FRAME",
                f"{rel} lacks component_map/sensor_orientation")
    return {"capsule_path": rel,
            "capsule_sha256": hashlib.sha256(raw).hexdigest(),
            "sensor_orientation": obj.get("sensor_orientation")}


def replay_key(key, row, authority, *, staged_dir, store_read,
               capsule, transform_identity, dry=False):
    """One key, fully verified before and after the transform."""
    lane, ck, day = key.split("/")
    stem = _stem(lane, ck, day)

    rp = _cls_path(staged_dir, stem, "record")
    # codex 0551Z repair 2: NO record-exists shortcut. A lone
    # `.record.json` -- from an interruption between the record write
    # and the receipt append, or from forgery -- previously returned
    # ALREADY_PRESENT with NO transcript read, NO join, NO receipt.
    # Every invocation now reopens the evidence, rebuilds all three
    # outputs, runs the full join, and passes each output through the
    # write-once canonical-identity check: identical pre-existing
    # bytes are verified and reused, divergent bytes refuse typed.
    pre_existing = os.path.exists(rp)

    tp = _cls_path(staged_dir, stem, "transcript")
    if not os.path.isfile(tp):
        _refuse("VIC_TRANSCRIPT_ABSENT",
                f"{key}: no preserved transcript at {tp}")
    with open(tp, encoding="utf-8") as f:
        t = json.load(f)

    # the transcript must name THIS key before its body is trusted
    if t.get("lane") not in (None, lane) or \
            t.get("carrier") not in (None, ck) or \
            t.get("utc_day") not in (None, day):
        _refuse("VIC_TRANSCRIPT_KEY_DIVERGENT",
                f"{key}: transcript names "
                f"{t.get('lane')}/{t.get('carrier')}/{t.get('utc_day')}")

    body = store_read(t["raw_body_sha256"])
    if "raw_body_bytes" in t and len(body) != t["raw_body_bytes"]:
        _refuse("VIC_BODY_LENGTH_DIVERGENT",
                f"{key}: store body is {len(body)} bytes, transcript "
                f"binds {t['raw_body_bytes']}")

    s = ACC.authoritative_static_contract(authority, lane, ck, day)
    # the preserved transcript must still bind S and these exact bytes
    PROD.verify_transcript(t, s, raw_body=body)

    artifact = CAP.admission_transform(lane, body, s)
    record = PROD.build_envelope_record(
        lane=s["lane"], carrier=s["carrier"], utc_day=s["utc_day"],
        raw_body=body, source=dict(s["source"]),
        endpoint=s["endpoint"],
        request_params=dict(s["request_params"]), transcript=t,
        cutoff=s["cutoff"],
        operation_params=dict(s["operation_params"]),
        expected_keys=list(s["expected_keys"]), artifact=artifact)

    # the FULL five-map join through the real gate, exactly as at capture
    PROD.verify_staged_day_set(
        {day: record}, {day: body}, {day: artifact}, {day: s},
        {day: t}, [day], ck, lane)

    if dry:
        # codex 2240Z P0-2 + 2313Z P0-2 plan mode: every verification
        # and join has already run above; the preview binds the exact
        # INPUT identities and every prospective OUTPUT digest,
        # including the repair receipt -- nothing is written
        # codex r2 review: dry mode must also perform the APPLY path's
        # canonical-identity check against every output that already
        # exists. Otherwise a plan can report verified over a
        # divergent record that apply will immediately refuse.
        for cls, path, want in (
                ("contract", _cls_path(staged_dir, stem, "contract"), s),
                ("artifact", _cls_path(staged_dir, stem, "artifact"),
                 artifact),
                ("record", rp, record)):
            if not os.path.exists(path):
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    have = json.load(f)
            except Exception as exc:
                _refuse("VIC_DRY_OUTPUT_DIVERGENT",
                        f"{key}: existing {cls} is not readable JSON "
                        f"({type(exc).__name__}: {str(exc)[:120]})")
            if PROD._canon_digest(have) != PROD._canon_digest(want):
                _refuse("VIC_DRY_OUTPUT_DIVERGENT",
                        f"{key}: existing {cls} differs from the "
                        "prospective create-once output")
        preview_entry = {"key": key,
                         "repair": REPAIR_ID,
                         "proof_kind": PROOF_KIND,
                         "original_seq": row.get("seq"),
                         "original_refusal":
                             str(row.get("refusal", ""))[:300],
                         "raw_body_sha256": t["raw_body_sha256"],
                         "raw_body_bytes": len(body),
                         "transcript_sha256": PROD._canon_digest(t),
                         "output_sha256": record["output_sha256"],
                         "transform_identity": transform_identity,
                         "http_requests": 0}
        preview_entry.update(capsule)
        entry_preview = {
            "key": key,
            "inputs": {"transcript_sha256": PROD._canon_digest(t),
                       "raw_body_sha256": t["raw_body_sha256"],
                       "static_contract_sha256":
                           PROD._canon_digest(s)},
            "would_write": {
                "contract": PROD._canon_digest(s),
                "artifact": PROD._canon_digest(artifact),
                "record": PROD._canon_digest(record),
                "repair_receipt": hashlib.sha256(json.dumps(
                    preview_entry, sort_keys=True,
                    separators=(",", ":")).encode()).hexdigest()}}
        return "DRY_VERIFIED", entry_preview
    CAP._write_once_json(_cls_path(staged_dir, stem, "contract"), s,
                         "CAPTURE_RECORD_DIVERGENT")
    CAP._write_once_json(_cls_path(staged_dir, stem, "artifact"),
                         artifact, "CAPTURE_RECORD_DIVERGENT")
    CAP._write_once_json(rp, record, "CAPTURE_RECORD_DIVERGENT")

    entry = {"key": key,
             "repair": REPAIR_ID,
             "proof_kind": PROOF_KIND,
             "original_seq": row.get("seq"),
             "original_refusal": str(row.get("refusal", ""))[:300],
             "raw_body_sha256": t["raw_body_sha256"],
             "raw_body_bytes": len(body),
             "transcript_sha256": PROD._canon_digest(t),
             "output_sha256": record["output_sha256"],
             "transform_identity": transform_identity,
             "http_requests": 0}
    entry.update(capsule)
    return ("VERIFIED_PRESENT" if pre_existing else "REPAIRED"), entry


def replay(authority, *, plan_keys, ledger_rows, staged_dir=None,
           store_read=None, repair_ledger=None, repo=REPO,
           expect=EXPECTED_TARGET_COUNT, dry=False):
    staged_dir = STAGED_DIR if staged_dir is None else staged_dir
    store_read = (default_store_reader(STORE) if store_read is None
                  else store_read)
    repair_ledger = (REPAIR_LEDGER if repair_ledger is None
                     else repair_ledger)

    capsule = capsule_binding(repo)
    ident = CAP.transform_identity()
    targets = target_set(plan_keys, ledger_rows, expect=expect)
    by_key = {r["key"]: r for r in ledger_rows}

    # codex 1547Z repair 2 + 2313Z P0-3: idempotence now lives in
    # the per-key CREATE-ONCE receipt objects; the ledger is a pure
    # ATOMIC projection of them, generated after the loop
    receipts_dir = os.path.dirname(repair_ledger) \
        if repair_ledger != REPAIR_LEDGER else RECEIPTS_DIR
    if repair_ledger != REPAIR_LEDGER:
        receipts_dir = os.path.join(
            os.path.dirname(repair_ledger), "vic_repair_receipts")
    if not dry:
        os.makedirs(receipts_dir, exist_ok=True)
    repaired = verified = receipts = 0
    previews = {}
    for key in targets:
        state, entry = replay_key(
            key, by_key[key], authority, staged_dir=staged_dir,
            store_read=store_read, capsule=capsule,
            transform_identity=ident, dry=dry)
        if dry:
            repaired += 1
            previews[key] = entry
            continue
        # codex 2313Z P0-3: per-key CREATE-ONCE receipt object --
        # identical reuse, divergent refuse; a crash can never leave
        # a truncated mutable tail
        lane2, ck2, day2 = key.split("/")
        rcpt_path = os.path.join(
            receipts_dir, _stem(lane2, ck2, day2) + ".repair.json")
        prior = os.path.exists(rcpt_path)
        try:
            CAP._write_once_json(rcpt_path, entry,
                                 "VIC_REPAIR_ENTRY_DIVERGENT")
        except CAP.CaptureRefusal as exc:
            _refuse("VIC_REPAIR_ENTRY_DIVERGENT",
                    f"{key}: {str(exc)[:160]}")
        if prior:
            receipts += 1
        if state == "REPAIRED":
            repaired += 1
        else:
            verified += 1
    if not dry:
        # ---- the ATOMIC ledger projection: 212 lines generated from
        # the create-once receipt objects, published via exclusive
        # link; an existing ledger must equal the projection exactly
        lines = []
        for key in targets:
            lane2, ck2, day2 = key.split("/")
            rp2 = os.path.join(receipts_dir,
                               _stem(lane2, ck2, day2)
                               + ".repair.json")
            with open(rp2, encoding="utf-8") as f:
                lines.append(json.dumps(json.load(f),
                                        sort_keys=True))
        want = "\n".join(lines) + "\n"
        if os.path.exists(repair_ledger):
            with open(repair_ledger, encoding="utf-8") as f:
                have = f.read()
            if have != want:
                _refuse("VIC_REPAIR_LEDGER_DIVERGENT",
                        "the published ledger does not equal the "
                        "projection of the create-once receipts "
                        "(truncation or tamper); regenerate is "
                        "refused, audit the receipts")
        else:
            import tempfile as _tf
            os.makedirs(os.path.dirname(repair_ledger) or ".",
                        exist_ok=True)
            fd, tmp = _tf.mkstemp(
                dir=os.path.dirname(repair_ledger) or ".",
                suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8",
                               newline="\n") as f:
                    f.write(want)
                    f.flush()
                    os.fsync(f.fileno())
                try:
                    os.link(tmp, repair_ledger)
                except FileExistsError:
                    with open(repair_ledger,
                              encoding="utf-8") as f:
                        if f.read() != want:
                            _refuse("VIC_REPAIR_LEDGER_DIVERGENT",
                                    "a concurrent ledger publication "
                                    "diverges from the receipt "
                                    "projection")
            finally:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    return {"targets": len(targets),
            ("dry_verified" if dry else "repaired"): repaired,
            "verified_present": verified,
            "receipts_already_present": receipts,
            "http_requests": 0, "capsule": capsule,
            "previews": previews if dry else None}


# ------------------------------------------------------------------ #
# CONTROLS. Fixture-driven so they run anywhere, but the fixture body
# and capsule are the REAL committed VIC bytes, not synthesised ones --
# a fixture invented from the same constants it verifies proves nothing.
# ------------------------------------------------------------------ #
def _selftest():
    import subprocess
    import tempfile
    import w2_no_network_grassmann as NONET

    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f"  -- {detail}" if detail and not ok else ""))
        if not ok:
            fails.append(name)

    def refuses(fn, code):
        try:
            fn()
        except VicReplayRefusal as exc:
            return str(exc).startswith(code)
        except Exception:
            return False
        return False

    def blob(path):
        p = subprocess.run(
            ["git", "-C", REPO, "cat-file", "blob", f"HEAD:{path}"],
            capture_output=True)
        return None if p.returncode else p.stdout

    # ---- exact-set derivation, over a synthetic plan/ledger pair whose
    # SHAPE mirrors the terminal run (212 vic + 1 new 404 + clean omni)
    import datetime
    d0 = datetime.date(2026, 1, 1)
    days = [(d0 + datetime.timedelta(days=n)).isoformat()
            for n in range(EXPECTED_TARGET_COUNT)]
    plan = ([f"MAG_FEED/vic/{d}" for d in days]
            + [f"MAG_FEED/new/{d}" for d in days]
            + [f"MAG_WEATHER_FEED/omni/{d}" for d in days])
    led = [{"key": f"MAG_FEED/vic/{d}", "seq": i, "status": "REFUSED",
            "refusal": f"ADMISSION_TRANSFORM_REFUSED: GIN MAG: source "
                       f"orientation 'XYZS' {TARGET_MARK} and no "
                       f"committed frame capsule exists"}
           for i, d in enumerate(days)]
    led.append({"key": "MAG_FEED/new/2026-03-23", "seq": 999,
                "status": "REFUSED",
                "refusal": "CAPTURE_HTTP_STATUS: 404"})

    got = target_set(plan, led)
    check("C1 exact-set: derives exactly 212 VIC keys and excludes the "
          "404, which is a different refusal class",
          len(got) == 212 and all(k.split("/")[1] == "vic" for k in got),
          f"n={len(got)}")

    check("C2 count divergence refuses (211 planned vs 212 refused)",
          refuses(lambda: target_set(plan[1:], led),
                  "VIC_TARGET_SET_UNPLANNED"))
    check("C3 a planned VIC key with no ledger refusal refuses",
          refuses(lambda: target_set(plan + ["MAG_FEED/vic/2099-01-01"],
                                     led), "VIC_TARGET_SET_INCOMPLETE"))
    check("C4 non-VIC injection into the refusal class refuses",
          refuses(lambda: target_set(
              plan, led + [{"key": "MAG_FEED/new/2026-01-01", "seq": 1,
                            "status": "REFUSED",
                            "refusal": f"x {TARGET_MARK} y"}]),
              "VIC_NON_TARGET_KEY_IN_REFUSAL_CLASS"))
    check("C5 the mark without status REFUSED refuses (mistyped row)",
          refuses(lambda: target_set(
              plan, led + [{"key": "MAG_FEED/vic/2099-02-02", "seq": 2,
                            "status": "CAPTURED",
                            "refusal": f"x {TARGET_MARK} y"}]),
              "VIC_REFUSAL_STATUS_MISTYPED"))
    check("C6 an explicit expected count that disagrees refuses",
          refuses(lambda: target_set(plan, led, expect=211),
                  "VIC_TARGET_COUNT_DIVERGENT"))

    # ---- capsule binding, against the REAL committed bytes
    real_capsule = blob("docs/f2g_window2_freeze/mag_capsule_vic.json")
    check("C7 the real freeze VIC capsule is fetchable for fixtures",
          real_capsule is not None and len(real_capsule) > 100)

    with tempfile.TemporaryDirectory() as td:
        rel_dir = os.path.join(td, *CAP.EXEC_CAPSULE_DIR.split("/"))
        os.makedirs(rel_dir, exist_ok=True)
        check("C8 absent capsule at the resolved path refuses",
              refuses(lambda: capsule_binding(td),
                      "VIC_EXEC_CAPSULE_ABSENT"))

        # the capsule at a NON-resolved path satisfies nothing -- the
        # exact defect being repaired, asserted rather than assumed
        alt = os.path.join(td, "docs", "f2g_window2_freeze")
        os.makedirs(alt, exist_ok=True)
        with open(os.path.join(alt, "mag_capsule_vic.json"), "wb") as f:
            f.write(real_capsule)
        check("C9 a real capsule at the FREEZE path still refuses -- "
              "only the production-resolved path satisfies",
              refuses(lambda: capsule_binding(td),
                      "VIC_EXEC_CAPSULE_ABSENT"))

        target = os.path.join(rel_dir, "mag_capsule_vic.json")
        with open(target, "wb") as f:
            f.write(real_capsule)
        b = capsule_binding(td)
        check("C10 the SAME bytes at the resolved path bind, and the "
              "digest is recomputed not copied",
              b["capsule_sha256"] ==
              hashlib.sha256(real_capsule).hexdigest()
              and b["sensor_orientation"] == "XYZS",
              f"{b}")

        with open(target, "wb") as f:
            f.write(b'{"schema":"x"}')
        check("C11 a mutated capsule missing the frame fields refuses",
              refuses(lambda: capsule_binding(td),
                      "VIC_EXEC_CAPSULE_NOT_A_FRAME"))

    # ---- store reader: absence and digest divergence
    with tempfile.TemporaryDirectory() as td:
        read = default_store_reader(td)
        sha = hashlib.sha256(b"hello").hexdigest()
        check("C12 a missing store body refuses",
              refuses(lambda: read(sha), "VIC_BODY_ABSENT"))
        with open(os.path.join(td, sha + ".body"), "wb") as f:
            f.write(b"TAMPERED")
        check("C13 a store body whose digest diverges refuses",
              refuses(lambda: read(sha), "VIC_BODY_DIGEST_DIVERGENT"))
        with open(os.path.join(td, sha + ".body"), "wb") as f:
            f.write(b"hello")
        check("C14 an intact store body reads back", read(sha) == b"hello")

    # ---- K: INTERRUPTION / IDEMPOTENCE KATs (codex 0551Z repair 2).
    # These drive the REAL replay_key/replay file-state machine --
    # write-once publication, canonical-identity divergence, receipt
    # idempotence -- over a temp staged dir. The DATA-VALIDATION inner
    # calls (static contract, transcript verification, transform,
    # record build, five-map join) are stubbed to deterministic fakes,
    # CLEARLY so: these KATs verify the interruption state machine,
    # not data validation; the real-store data controls remain due on
    # devildog per codex 0551Z. The stubs cannot fake the filesystem
    # behavior under test.
    import tempfile as _tf2
    _kkey = "MAG_FEED/vic/2026-01-05"
    _kbody = b"KAT-BODY-BYTES"
    _ksha = hashlib.sha256(_kbody).hexdigest()
    _kcap = {"capsule_path": "kat", "capsule_sha256": "0" * 64,
             "sensor_orientation": "XYZS"}

    def _k_store(sha):
        if sha != _ksha:
            _refuse("VIC_BODY_ABSENT", sha[:12])
        return _kbody

    def _k_transcript(td, stem):
        t = {"raw_body_sha256": _ksha,
             "raw_body_bytes": len(_kbody),
             "lane": "MAG_FEED", "carrier": "vic",
             "utc_day": "2026-01-05"}
        with open(_cls_path(td, stem, "transcript"), "w",
                  encoding="utf-8") as f:
            json.dump(t, f)
        return t

    _reals = (ACC.authoritative_static_contract, PROD.verify_transcript,
              CAP.admission_transform, PROD.build_envelope_record,
              PROD.verify_staged_day_set)

    def _stub_contract(authority, lane, ck, day):
        return {"lane": lane, "carrier": ck, "utc_day": day,
                "source": {"kind": "kat"}, "endpoint": "kat://e",
                "request_params": {}, "cutoff": "2026-08-01",
                "operation_params": {}, "expected_keys": []}

    def _stub_vt(t, s, raw_body=None):
        # non-vacuous: the stub still requires the reopened body to be
        # the transcript-bound bytes
        assert hashlib.sha256(raw_body).hexdigest() \
            == t["raw_body_sha256"], "stub verify_transcript: body!=T"

    def _stub_transform(lane, body, s):
        return {"outcome": "ADMITTED", "kat": True}

    def _stub_record(**kw):
        return {"lane": kw["lane"], "carrier": kw["carrier"],
                "utc_day": kw["utc_day"],
                "raw_body_sha256":
                    hashlib.sha256(kw["raw_body"]).hexdigest(),
                "output_sha256": PROD._canon_digest(kw["artifact"]),
                "kat_record": True}

    _joins = []

    def _stub_join(*a, **k):
        _joins.append(1)
    ACC.authoritative_static_contract = _stub_contract
    PROD.verify_transcript = _stub_vt
    CAP.admission_transform = _stub_transform
    PROD.build_envelope_record = _stub_record
    PROD.verify_staged_day_set = _stub_join

    def _rk(td):
        return replay_key(_kkey, {"seq": 81, "refusal": "kat"},
                          {}, staged_dir=td, store_read=_k_store,
                          capsule=_kcap, transform_identity="kat-ident")

    def _k_refuses(fn, needle):
        try:
            fn()
        except Exception as exc:
            return needle in str(exc)
        return False
    try:
        stem_k = _stem("MAG_FEED", "vic", "2026-01-05")
        # K1a codex's exact repro shape: ONLY a record, nothing else
        with _tf2.TemporaryDirectory() as td:
            with open(_cls_path(td, stem_k, "record"), "w") as f:
                f.write("{}")
            check("K1a a lone record with no transcript REFUSES "
                  "(was ALREADY_PRESENT)",
                  _k_refuses(lambda: _rk(td), "VIC_TRANSCRIPT_ABSENT"))
        # K1b evidence present but the record is a forged/divergent {}
        with _tf2.TemporaryDirectory() as td:
            _k_transcript(td, stem_k)
            with open(_cls_path(td, stem_k, "record"), "w") as f:
                f.write("{}")
            check("K1b divergent pre-existing record bytes REFUSE via "
                  "canonical identity",
                  _k_refuses(lambda: _rk(td),
                             "CAPTURE_RECORD_DIVERGENT"))
            check("K1c dry planning also refuses a divergent "
                  "pre-existing record",
                  _k_refuses(lambda: replay_key(
                      _kkey, {"seq": 81, "refusal": "kat"}, {},
                      staged_dir=td, store_read=_k_store,
                      capsule=_kcap, transform_identity="kat-ident",
                      dry=True), "VIC_DRY_OUTPUT_DIVERGENT"))
        # K2 fresh dir -> REPAIRED; rerun -> VERIFIED_PRESENT with the
        # IDENTICAL entry (idempotence at key level)
        with _tf2.TemporaryDirectory() as td:
            _k_transcript(td, stem_k)
            st1, e1 = _rk(td)
            st2, e2 = _rk(td)
            check("K2 fresh run REPAIRED then rerun VERIFIED_PRESENT "
                  "with an identical reconstructed entry",
                  st1 == "REPAIRED" and st2 == "VERIFIED_PRESENT"
                  and e1 == e2 and _joins[-2:] == [1, 1],
                  f"{st1}/{st2} equal={e1 == e2}")
            # K3 partial state: artifact+record deleted, contract kept
            os.remove(_cls_path(td, stem_k, "artifact"))
            os.remove(_cls_path(td, stem_k, "record"))
            st3, e3 = _rk(td)
            check("K3 partial three-file interruption resumes to a "
                  "complete set (identical contract reused)",
                  st3 == "REPAIRED" and e3 == e1
                  and all(os.path.exists(_cls_path(td, stem_k, c))
                          for c in ("contract", "artifact", "record")))
            # K3b divergent pre-existing CONTRACT refuses
            os.remove(_cls_path(td, stem_k, "contract"))
            with open(_cls_path(td, stem_k, "contract"), "w") as f:
                f.write('{"forged": true}')
            check("K3b divergent pre-existing contract refuses",
                  _k_refuses(lambda: _rk(td),
                             "CAPTURE_RECORD_DIVERGENT"))
        # K4/K5 receipt idempotence at replay() level
        with _tf2.TemporaryDirectory() as td:
            _k_transcript(td, stem_k)
            rl = os.path.join(td, "repair_ledger.jsonl")
            kplan = [_kkey]
            kled = [{"key": _kkey, "seq": 81, "status": "REFUSED",
                     "refusal": f"x {TARGET_MARK} y"}]
            r1 = replay({}, plan_keys=kplan, ledger_rows=kled,
                        staged_dir=td, store_read=_k_store,
                        repair_ledger=rl, expect=1)
            n1 = sum(1 for _l in open(rl, encoding="utf-8") if _l.strip())
            r2 = replay({}, plan_keys=kplan, ledger_rows=kled,
                        staged_dir=td, store_read=_k_store,
                        repair_ledger=rl, expect=1)
            n2 = sum(1 for _l in open(rl, encoding="utf-8") if _l.strip())
            check("K4 record-without-receipt resume appends the receipt "
                  "exactly once (rerun adds nothing)",
                  r1["repaired"] == 1 and n1 == 1
                  and r2["receipts_already_present"] == 1 and n2 == 1,
                  f"r1={r1} n1={n1} r2={r2} n2={n2}")
            # K5 a DIVERGENT existing entry for the key refuses
            e = json.loads(open(rl, encoding="utf-8").read())
            e["output_sha256"] = "f" * 64
            with open(rl, "w", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(e, sort_keys=True) + "\n")
            check("K5 a tampered published ledger refuses as a "
                  "divergent projection of the create-once receipts",
                  _k_refuses(lambda: replay(
                      {}, plan_keys=kplan, ledger_rows=kled,
                      staged_dir=td, store_read=_k_store,
                      repair_ledger=rl, expect=1),
                      "VIC_REPAIR_LEDGER_DIVERGENT"))
    finally:
        (ACC.authoritative_static_contract, PROD.verify_transcript,
         CAP.admission_transform, PROD.build_envelope_record,
         PROD.verify_staged_day_set) = _reals

    # ---- the standing guarantee: no socket, ever
    attempted_before = NONET.attempts_total()
    with NONET.no_network():
        target_set(plan, led)
        default_store_reader(tempfile.gettempdir())
    check("C15 the whole target/binding path runs with sockets BLOCKED "
          "and attempts zero",
          NONET.attempts_total() == attempted_before,
          f"attempts delta={NONET.attempts_total() - attempted_before}")

    # ---- proof kind must not drift
    check("C16 proof kind stays NATIVE_V4_CAPTURE (not RESTAGED_LINEAGE, "
          "not a new kind)", PROOF_KIND == "NATIVE_V4_CAPTURE")

    print()
    if fails:
        print(f"V4 VIC REPLAY CONTROL FAILURES ({len(fails)}): {fails}")
        return 1
    print("V4 VIC REPLAY: ALL CONTROLS PASS  (no network; writes "
          "confined to selftest temp dirs)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv or len(sys.argv) == 1:
        raise SystemExit(_selftest())
    raise SystemExit("usage: w2_capture_repair_v4_vic_cayley.py "
                     "--selftest   (live replay is driven by the "
                     "frozen candidate, not this entrypoint)")
