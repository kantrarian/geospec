#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 PRODUCTION ACCRUAL INSTRUMENT, core (cayley) -- the live
wrapper around the w2_barrier ledger. This module is the ONLY place
window-2 production code reads a clock; every barrier operation flows
through the file-backed, hash-chained, append-only ledger; and nothing
executes until the RUNTIME ALLOWLIST walk passes (every BOUND
execution-manifest pin's on-disk bytes must equal its pinned blob,
CRLF->LF normalized -- the same executed-bytes discipline as the
execution verifier, applied to the DISK the instrument runs from).

Pin-independent core (built while grassmann's batch REV ratifies the
engine interpretation pins): persistence, chain verification, state
reconstruction FROM EVENTS (single source of truth), allowlist walk,
live-clock injection. The seam-binding calls (selection execution,
adapter runs, per-lane accrual) layer on top once the bar REV + codex
seam close land.

Persistence model: JSON file {meta: {lease, used_leases}, events: [...]}.
On load the chain is re-verified and ALL ledger state (window dates,
admitted lanes, predictions, seals, terminals, verifier passes,
first-fired, state) is REBUILT by replaying events -- a doctored file
either breaks the chain (LEDGER_CHAIN_BROKEN) or is faithfully
reflected, never silently merged. Refusal semantics live entirely in
w2_barrier; this wrapper adds no policy of its own.

No live PRESTART is performed by importing or KAT-ing this module;
PRESTART requires the full sequence (bars green, manifest CLOSED,
codex round, owner authorization). This module opens no window-2 value.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import w2_barrier as WB

EXEC_MANIFEST_PATH = "docs/f2g_window2_execution/execution_manifest.json"


class InstrumentRefusal(ValueError):
    """Typed refusal; the code leads the message."""


def now_utc_day():
    """THE clock read (live UTC day). Nothing else in window-2
    production reads a clock."""
    return time.strftime("%Y-%m-%d", time.gmtime())


def now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _norm_py(raw):
    return raw.replace(b"\r\n", b"\n")


def _git(repo, args, binary=False):
    p = subprocess.run(["git", "-C", repo] + args, capture_output=True)
    if p.returncode != 0:
        raise InstrumentRefusal(
            f"GIT_READ_FAILED: {' '.join(args)[:80]}")
    return p.stdout if binary else p.stdout.decode("utf-8",
                                                   "replace").strip()


def runtime_allowlist_check(repo, manifest_commit):
    """Every BOUND pin's ON-DISK bytes must equal the pinned blob
    (CRLF->LF normalized). Returns the checked-pin report; raises
    RUNTIME_ALLOWLIST_VIOLATION naming every divergent path. The
    manifest itself is read from the git object at the stated commit,
    never from disk."""
    raw = _git(repo, ["cat-file", "blob",
                      f"{manifest_commit}:{EXEC_MANIFEST_PATH}"],
               binary=True)
    manifest = json.loads(raw.decode("utf-8"))
    if manifest.get("schema") != "f2g-window2-execution-manifest-v1.2":
        raise InstrumentRefusal("RUNTIME_ALLOWLIST_VIOLATION: "
                                "manifest schema mismatch")
    checked = []
    divergent = []
    for slot_name in sorted(manifest["slots"]):
        slot = manifest["slots"][slot_name]
        if slot["status"] != "BOUND":
            continue
        for pin in slot["pins"]:
            disk = os.path.join(repo, pin["path"].replace("/", os.sep))
            if not os.path.exists(disk):
                divergent.append((slot_name, pin["path"], "ABSENT"))
                continue
            with open(disk, "rb") as f:
                got = hashlib.sha256(_norm_py(f.read())).hexdigest()
            if got != pin["blob_sha256"]:
                divergent.append((slot_name, pin["path"],
                                  f"{got[:12]}!={pin['blob_sha256'][:12]}"))
            else:
                checked.append((slot_name, pin["path"]))
    if divergent:
        raise InstrumentRefusal(
            f"RUNTIME_ALLOWLIST_VIOLATION: {divergent}")
    return {"manifest_commit": manifest_commit,
            "manifest_state": manifest["manifest_state"],
            "pins_checked": len(checked), "pins": checked}


STAGED_INVENTORY_BASENAME = "staged_body_inventory.json"
STORE_DESCRIPTOR_BASENAME = "store_descriptor.json"


def _pinned_json(slot, basename, blob_reader):
    pins = [p for p in slot.get("pins", ())
            if isinstance(p, dict) and str(p.get("path", ""))
            .endswith(basename)]
    if len(pins) != 1:
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: producer_boundary is BOUND "
            f"without exactly one {basename} pin (found {len(pins)})")
    pin = pins[0]
    raw = blob_reader(pin["commit"], pin["path"])
    if hashlib.sha256(raw).hexdigest() != pin["blob_sha256"]:
        raise InstrumentRefusal(
            f"PRESTART_ADMISSION_REFUSED: {basename} bytes diverge "
            "from the manifest pin")
    return json.loads(raw.decode("utf-8"))


def verify_staged_store(repo, manifest, *, blob_reader=None,
                        inventory_verifier=None):
    """producer-boundary amendment v1.1 appendix (codex 1843Z item 4
    + 2015Z item 1): when the producer_boundary slot is BOUND, BOTH
    the staged_body_inventory AND the registered store DESCRIPTOR
    reopen from their manifest pins, and grassmann's REV 7 NAMED-STORE
    verifier reopens EVERY object -- the physical root comes only from
    the pinned descriptor mapping (there is no caller path); an
    inventory hash is never mistaken for completed-build availability;
    an unavailable or wrong store is a TYPED refusal, never PASS.
    Returns None while the slot is honestly OPEN (the zero-OPEN
    prestart gate refuses upstream). blob_reader/inventory_verifier
    are injectable for KATs only."""
    slot = manifest["slots"].get("producer_boundary")
    if not isinstance(slot, dict) or slot.get("status") != "BOUND":
        return None
    if blob_reader is None:
        def blob_reader(commit, path):
            return _git(repo, ["cat-file", "blob",
                               f"{commit}:{path}"], binary=True)
    inventory = _pinned_json(slot, STAGED_INVENTORY_BASENAME,
                             blob_reader)
    descriptor = _pinned_json(slot, STORE_DESCRIPTOR_BASENAME,
                              blob_reader)
    if inventory_verifier is None:
        import w2_acquisition_capture_grassmann as CAP

        def inventory_verifier(inv, desc):
            return CAP.verify_staged_body_inventory(inv, desc)
    try:
        report = inventory_verifier(inventory, descriptor)
    except Exception as e:
        raise InstrumentRefusal(
            f"PRESTART_ADMISSION_REFUSED: staged store reopen "
            f"failed: {e}")
    return {"store_id": inventory.get("store_id"),
            "objects": len(inventory.get("objects", {})),
            "report": report}


STAGED_CLASS_SUFFIX = {"record": ".record.json",
                       "transcript": ".transcript.json",
                       "contract": ".contract.json",
                       "artifact": ".artifact.json"}
# codex 0655Z item 3: provenance is a tagged XOR. A restaged key's
# provenance carrier is `.restage.json` (the pinned lineage record),
# never a relabelled native envelope; `record` above is the NATIVE
# provenance form only. STAGED_ALL_SUFFIXES is the one recognition
# authority for the staged space -- the generator imports it, so a
# carrier the boundary parses is a carrier the generator pins.
STAGED_RESTAGE_SUFFIX = ".restage.json"
STAGED_ALL_SUFFIXES = tuple(STAGED_CLASS_SUFFIX.values()) + (
    STAGED_RESTAGE_SUFFIX,)
# codex 1547Z repair 1: the ADMISSION CONSUMER is the one authority
# for the staged prefix -- the generator and the execution verifier
# import THESE constants, so a carrier rename can never close in only
# two of three layers. The v3 prefix is RETIRED: a staged-class pin
# under it refuses typed, never parses, never satisfies.
STAGED_PREFIX = "docs/f2g_window2_execution/staged_envelopes_v4/"
STAGED_PREFIX_RETIRED = "docs/f2g_window2_execution/staged_envelopes/"
EXPECTED_KEYS_BASENAME = "staged_expected_contracts_v4.json"
# codex 1547Z repair 2: the three operation-evidence classes the
# boundary derives its ADMITTED partition from. Exact registered
# repo-relative paths; the generator's final-bind contract requires
# the same three (it imports these).
TERMINAL_RECEIPT_PATH = ("docs/f2g_window2_execution/"
                         "capture_terminal_receipt_v4.json")
VIC_REPAIR_LEDGER_PATH = ("docs/f2g_window2_execution/"
                          "capture_repair_ledger_v4_vic.jsonl")
PREDECESSOR_RECORD_PATH = ("docs/f2g_window2_execution/"
                           "predecessor_bridge_record_v4.json")
# codex 2240Z P0-1: the CLOSED retry-operation chain -- six exact
# registered paths, including the mandatory transport receipt. Only a
# fully verified chain may add the former-404 key to the admitted
# partition; the frozen main ledger honestly still marks it REFUSED.
RETRY_STEM = "mag_feed_new_2026-03-23"
RETRY_DIR_REL = "docs/f2g_window2_execution/retry_404_v4/"
RETRY_CHAIN_PATHS = {
    "dispatch": RETRY_DIR_REL + RETRY_STEM + ".dispatch.json",
    "transport_receipt": (RETRY_DIR_REL + "attempt_local/"
                          + RETRY_STEM + ".transport_receipt.json"),
    "prepared": RETRY_DIR_REL + RETRY_STEM + ".prepared.json",
    "result": RETRY_DIR_REL + RETRY_STEM + ".result.json",
    "classes_complete": (RETRY_DIR_REL + RETRY_STEM
                         + ".classes_complete.json"),
    "index": ("docs/f2g_window2_execution/"
              "capture_retry_ledger_v4.jsonl"),
}
CAPTURE_LEDGER_PATH = ("docs/f2g_window2_execution/"
                       "capture_run_ledger_v4.jsonl")
TERMINAL_RECEIPT_SCHEMA = "f2g-w2-capture-terminal-receipt-v1"
# v4 lane split (codex 0527Z finding 3): MF4_FEED named two
# carrier spaces at once and is RETIRED with no alias.
# codex 2119Z closure 4: the three proof kinds. A key's REQUIRED
# proof is DERIVED from its registered disposition in the pinned
# capsule -- never read from a submitted label, or a mislabelled key
# would authenticate itself. Disposition answers "how were the bytes
# obtained"; proof kind answers "what does admitting it establish".
PROOF_KIND_FOR_DISPOSITION = {
    "HTTP_CAPTURE": "NATIVE_V4_CAPTURE",
    "REUSE_OR_BRIDGE": "RESTAGED_LINEAGE",
    "PREDECESSOR": "PREDECESSOR_BRIDGE"}
PROOF_KINDS = tuple(sorted(PROOF_KIND_FOR_DISPOSITION.values()))

def compute_proof_kind_partitions(authorized_keys, capsule):
    """codex 2119Z closure 4. Returns the three EXACT, DISJOINT,
    RECOMPUTED proof-kind partitions over the authority key set.

    A key's required proof is DERIVED from its registered disposition
    in the capsule -- never read from a submitted label, since a
    mislabelled key would otherwise authenticate itself. Refuses
    typed on double disposition, on any authority key left
    undisposed, and on any disposed key outside the authority.
    """
    disposed = {
        "HTTP_CAPTURE": set(capsule["http_capture"]),
        "REUSE_OR_BRIDGE": set(capsule["reuse_or_bridge"]),
        "PREDECESSOR": set(capsule["predecessor"])}
    seen = {}
    for disp, keyset in disposed.items():
        kind = PROOF_KIND_FOR_DISPOSITION[disp]
        for k in keyset:
            if k in seen:
                raise InstrumentRefusal(
                    "PRESTART_ADMISSION_REFUSED: key {} is disposed "
                    "twice ({} and {}) -- proof-kind partitions must "
                    "be DISJOINT".format(k, seen[k], kind))
            seen[k] = kind
    want = set(authorized_keys)
    missing = sorted(want - set(seen))
    extra = sorted(set(seen) - want)
    if missing or extra:
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: proof-kind partitions do not "
            f"cover the authority EXACTLY -- {len(missing)} undisposed "
            f"(first {missing[:2]}), {len(extra)} outside the "
            f"authority (first {extra[:2]})")
    parts = {}
    for kind in PROOF_KINDS:
        ks = sorted(k for k, v in seen.items() if v == kind)
        parts[kind] = {
            "count": len(ks),                       # RECOMPUTED
            "keys_sha256": hashlib.sha256(json.dumps(
                ks, separators=(",", ":")).encode()).hexdigest(),
            "join_result": "ADMITTED"}
    total = sum(p["count"] for p in parts.values())
    if total != len(want):
        raise InstrumentRefusal(
            f"PRESTART_ADMISSION_REFUSED: proof-kind union {total} != "
            f"authority {len(want)}")
    return parts


PRESTART_LANES = ("SELECTION_RECORDS", "MAG_FEED",
                  "MAG_WEATHER_FEED")


def _parse_staged_pin(path):
    """<lane-lower>_<carrier>_<YYYY-MM-DD>.<class>.json under the
    EXACT staged_envelopes prefix (codex 0238Z item 1: full-path
    discipline -- a staged-class basename outside the prefix REFUSES
    rather than entering or vanishing). Returns None only for paths
    that are not staged-class files anywhere."""
    path = str(path)
    base = os.path.basename(path)
    for cls, suf in list(STAGED_CLASS_SUFFIX.items()) + [
            ("restage", STAGED_RESTAGE_SUFFIX)]:
        if base.endswith(suf):
            stem = base[: -len(suf)]
            break
    else:
        return None
    import w2_producer_grassmann as PROD
    for lane in sorted(PROD.RECORD_LANES):
        pre = lane.lower() + "_"
        if stem.startswith(pre):
            carrier, sep, day = stem[len(pre):].rpartition("_")
            if sep and carrier:
                # codex 1547Z repair 1: the RETIRED prefix is named
                # as retired, not misreported as merely misplaced
                if path.startswith(STAGED_PREFIX_RETIRED):
                    raise InstrumentRefusal(
                        "PRESTART_ADMISSION_REFUSED: staged-class "
                        f"pin under the RETIRED v3 prefix: {path} -- "
                        "the registered staged space is "
                        f"{STAGED_PREFIX}")
                if path != STAGED_PREFIX + base:
                    raise InstrumentRefusal(
                        "PRESTART_ADMISSION_REFUSED: staged-class "
                        f"pin outside the exact prefix: {path}")
                return lane, carrier, day, cls
    # codex 0655Z item 3 (v3/v4-key swap): a staged-suffix basename
    # under the registered prefixes whose stem matches NO registered
    # lane REFUSES typed -- a v3-lane stem (e.g. mf4_feed_*) must
    # never vanish from the walk
    if path.startswith(STAGED_PREFIX) or \
            path.startswith(STAGED_PREFIX_RETIRED):
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: staged-class pin with an "
            f"unregistered lane stem under the staged prefix: "
            f"{path}")
    return None


AUTHORITY_TOP_FIELDS = {
    "schema", "prestart_expected_keys",
    "prestart_expected_keys_sha256", "static_layer",
    "dynamic_layer", "digests", "provenance",
    "template_token_vocabulary",
    # v4: the registered predecessor lineage the bridge reopens
    "registered_probe_authority"}
AUTHORITY_CENSUS = 2056          # v4: 5x212 + 3x212 + 4x90
#   MAG 5 obs x 212d (07-31 cutoff) + weather 3 x 212d
#   + selection 4 x 90d (calendar-v4-derived cutoff)
AUTHORITY_SCHEMA = "f2g-w2-expected-contracts-v4"
TEMPLATE_TOKEN_VOCABULARY = ("{day}", "{day_next}", "{day_compact}")


def _validate_expected_keys_authority(repo, authority, *,
                                      reproducer=None):
    """codex 0320Z item 3: the key authority capsule is CLOSED and
    SELF-VERIFYING -- exact top-level schema; the recorded key digest
    RECOMPUTES; every day list unique + ascending canonical; the
    registered census; and the whole artifact REPRODUCES from the
    pinned generator (production default; injectable for KATs
    only). A forged digest or empty carrier map never passes."""
    def refuse(detail):
        raise InstrumentRefusal(
            f"PRESTART_ADMISSION_REFUSED: authority {detail}")
    if not isinstance(authority, dict) or \
            set(authority) != AUTHORITY_TOP_FIELDS or \
            authority.get("schema") != AUTHORITY_SCHEMA:
        refuse("top-level schema not closed")
    if authority["template_token_vocabulary"] != \
            list(TEMPLATE_TOKEN_VOCABULARY):
        refuse("template token vocabulary is not the registered set")
    keys = authority["prestart_expected_keys"]
    got = hashlib.sha256(json.dumps(
        keys, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    if got != authority["prestart_expected_keys_sha256"]:
        refuse("key digest does not recompute")
    if sorted(keys) != sorted(PRESTART_LANES):
        refuse("lane set is not the registered PRESTART lanes")
    census = 0
    for lane in keys:
        if not keys[lane]:
            refuse(f"{lane} carrier map is empty")
        for ck, days in keys[lane].items():
            if not days or days != sorted(days) or \
                    len(set(days)) != len(days):
                refuse(f"{lane}/{ck} days not unique ascending")
            import datetime as _dt
            for d in days:
                try:
                    ok = _dt.date.fromisoformat(
                        str(d)).isoformat() == d
                except ValueError:
                    ok = False
                if not ok:
                    refuse(f"{lane}/{ck} non-canonical day {d!r}")
            census += len(days)
    if reproducer is None and census != AUTHORITY_CENSUS:
        # the census constant binds the PRODUCTION path; injected
        # fixture reproducers carry their own censuses under the
        # same closed schema + reproduction equality
        refuse(f"census {census} != {AUTHORITY_CENSUS}")
    if reproducer is None:
        import w2_expected_contracts_gen_v4_cayley as GEN

        def reproducer():
            return GEN.build(repo)
    if json.dumps(reproducer(), sort_keys=True,
                  separators=(",", ":")) != \
            json.dumps(authority, sort_keys=True,
                       separators=(",", ":")):
        refuse("artifact does not REPRODUCE from the pinned "
               "calendar/probe/schedule generator")
    return authority


def authoritative_static_contract(authority, lane, carrier, day):
    """codex 0320Z item 1: the per-day S entry derives from the
    INDEPENDENT authority template (never the submitted sidecar);
    any OPEN token in the consumed template REFUSES -- the two-phase
    rule is structural: the v3 value freeze precedes the first
    capture."""
    entry = authority["static_layer"][lane]["carriers"][carrier]
    t = entry.get("static_contract_template")
    if not isinstance(t, dict):
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: no static-contract "
            f"template registered for {lane}/{carrier}")
    raw_template = json.dumps(t, sort_keys=True, separators=(",", ":"))
    if "OPEN_REVIEW_ROUND" in raw_template:
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: the consumed authority "
            f"template for {lane}/{carrier} carries OPEN tokens "
            "-- the v3 static freeze precedes any capture")
    import re
    template_tokens = set(re.findall(
        r"\{[A-Za-z][A-Za-z0-9_]*\}", raw_template))
    unknown_tokens = template_tokens - set(TEMPLATE_TOKEN_VOCABULARY)
    if unknown_tokens:
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: the consumed authority "
            f"template for {lane}/{carrier} carries unregistered "
            f"tokens {sorted(unknown_tokens)}")

    # {day_next} = the UTC day AFTER {day} (registered token for
    # half-open [day, day_next) request windows -- USGS/FDSN day
    # forms); substituted BEFORE {day}, whose pattern is a prefix
    # substring of it (grassmann capture-specs freeze condition 1)
    import datetime as _dt
    day_next = (_dt.date.fromisoformat(str(day))
                + _dt.timedelta(days=1)).isoformat()
    day_compact = str(day).replace("-", "")

    def sub(v):
        if isinstance(v, str):
            return v.replace("{day_next}", day_next).replace(
                "{day_compact}", day_compact).replace(
                "{day}", day)
        if isinstance(v, dict):
            return {k: sub(x) for k, x in v.items()}
        if isinstance(v, list):
            return [sub(x) for x in v]
        return v
    import w2_producer_grassmann as PROD
    contract = {"schema": PROD.STATIC_CONTRACT_SCHEMA,
                "lane": str(lane), "carrier": str(carrier),
                "utc_day": str(day),
                "source": sub(dict(t["source"])),
                "endpoint": sub(str(t["endpoint"])),
                "request_params": sub(dict(t["request_params"])),
                "cutoff": str(entry["cutoff"]),
                "operation_params": sub(dict(t["operation_params"])),
                "expected_keys": [str(day)]}
    unresolved = set(re.findall(
        r"\{[A-Za-z][A-Za-z0-9_]*\}",
        json.dumps(contract, sort_keys=True, separators=(",", ":"))))
    if unresolved:
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: the derived static contract "
            f"for {lane}/{carrier} carries unresolved tokens "
            f"{sorted(unresolved)}")
    return contract


# fixture-only chain-validation context (mirrors the retry module's
# _kat_allow_unpinned discipline): production leaves this EMPTY; the
# selftest installs overrides and restores them
_kat_chain_ctx = {}


def _derive_admitted_partition(slot, blob_reader, manifest, groups,
                                _canon):
    """codex 1547Z repair 2: the ADMITTED partition, derived from the
    THREE PINNED OPERATION-EVIDENCE RECORDS -- never from a caller
    argument on the production path, and never from the authority
    alone. Every value is RECOMPUTED against sibling pinned bytes;
    registration by path is membership, this is evidence semantics.

    Sources, each fail-closed:
      terminal receipt   recomputes the frozen ledger digest, line
                         count and CAPTURED/REFUSED key partition,
                         and binds the rebuilt inventory digest
      VIC repair ledger  exactly the registered VIC key set (derived
                         from the PINNED disposition capsule), each
                         entry http_requests=0, uniform transform
                         identity, JOINED to its staged transcript's
                         raw_body_sha256
      predecessor record grassmann's bridge record: bridge_sha256
                         recomputes, names the capsule's single
                         predecessor key, and its evidence/artifact
                         digests equal the staged classes for that key

    Returns the admitted key set as (lane, carrier, day) tuples."""
    def _pin_bytes(path_suffix):
        pins = [q for q in slot.get("pins", ())
                if isinstance(q, dict)
                and str(q.get("path", "")) == path_suffix]
        if len(pins) != 1:
            raise InstrumentRefusal(
                "PRESTART_ADMISSION_REFUSED: no capture-run archive "
                "was supplied and the producer boundary lacks exactly "
                f"one pin at {path_suffix} -- the admitted partition "
                "derives from pinned operation records")
        q = pins[0]
        raw = blob_reader(q["commit"], q["path"])
        if hashlib.sha256(raw).hexdigest() != q["blob_sha256"]:
            raise InstrumentRefusal(
                "PRESTART_ADMISSION_REFUSED: operation-evidence pin "
                f"bytes diverge: {path_suffix}")
        return raw

    # ---- terminal receipt vs the pinned frozen ledger -------------
    receipt = json.loads(_pin_bytes(TERMINAL_RECEIPT_PATH)
                         .decode("utf-8"))
    if not isinstance(receipt, dict) or \
            receipt.get("schema") != TERMINAL_RECEIPT_SCHEMA:
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: terminal receipt schema is "
            f"not {TERMINAL_RECEIPT_SCHEMA}")
    led_raw = _pin_bytes(CAPTURE_LEDGER_PATH)
    if hashlib.sha256(led_raw).hexdigest() != \
            receipt.get("ledger_sha256"):
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: terminal receipt "
            "ledger_sha256 does not recompute from the pinned ledger")
    rows = [json.loads(x) for x in led_raw.decode("utf-8")
            .splitlines() if x.strip()]
    if len(rows) != receipt.get("ledger_lines"):
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: terminal receipt line count "
            f"{receipt.get('ledger_lines')} != ledger {len(rows)}")
    cap_keys = sorted(r["key"] for r in rows
                      if r.get("status") == "CAPTURED")
    ref_keys = sorted(r["key"] for r in rows
                      if r.get("status") == "REFUSED")
    if cap_keys != sorted(receipt.get("admitted_keys", ())) or \
            ref_keys != sorted(receipt.get("refused_keys", ())):
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: terminal receipt partition "
            "does not recompute from the pinned ledger rows")
    # codex 1705Z repair 1: the inventory resolves at its EXACT
    # registered path -- a basename-endswith walk let a decoy at
    # docs/attacker/ be bound by the receipt while the scientific
    # consumer read the real one (an object-identity split between
    # receipt verification and consumption).
    inv_raw = _pin_bytes(STAGED_PREFIX + STAGED_INVENTORY_BASENAME)
    if hashlib.sha256(inv_raw).hexdigest() != \
            receipt.get("inventory_sha256"):
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: terminal receipt "
            "inventory_sha256 does not bind the EXACT registered "
            "inventory pin")

    # ---- VIC repair ledger vs the registered VIC set --------------
    capsule = _registered_disposition_capsule(manifest, blob_reader)
    vic_registered = sorted(
        k for k in capsule.get("http_capture", ())
        if str(k).split("/")[1:2] == ["vic"])
    rep_rows = [json.loads(x) for x in
                _pin_bytes(VIC_REPAIR_LEDGER_PATH).decode("utf-8")
                .splitlines() if x.strip()]
    rep_keys = sorted(r.get("key") for r in rep_rows)
    if rep_keys != vic_registered or \
            len(set(rep_keys)) != len(rep_keys):
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: VIC repair records are not "
            "exactly the registered VIC key set "
            f"({len(rep_keys)} vs {len(vic_registered)} registered)")
    # codex 0655Z item 4: transform_identity is the CLOSED DICT the
    # dispatcher emits (transform_identity_from_source) -- a set
    # comprehension over the real rows raised raw TypeError
    # (unhashable dict) instead of verifying uniformity. Shape first
    # (keyset derived from the producer itself, never a duplicated
    # constant), then exactly one canonical digest across all rows.
    import w2_acquisition_capture_grassmann as _CAPI
    _ident_keys = set(_CAPI.transform_identity_from_source(b""))
    ident_digests = set()
    for r in rep_rows:
        ident = r.get("transform_identity")
        if not isinstance(ident, dict) or set(ident) != _ident_keys:
            raise InstrumentRefusal(
                "PRESTART_ADMISSION_REFUSED: VIC repair entry for "
                f"{r.get('key')} does not carry the closed "
                "transform-identity object")
        ident_digests.add(hashlib.sha256(json.dumps(
            ident, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest())
    if len(ident_digests) != 1:
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: VIC repair transform "
            "identity is non-uniform across the repair rows")
    for r in rep_rows:
        if r.get("http_requests") != 0:
            raise InstrumentRefusal(
                "PRESTART_ADMISSION_REFUSED: VIC repair entry for "
                f"{r.get('key')} claims http_requests="
                f"{r.get('http_requests')}; the replay is zero-HTTP")
        lane, ck, day = str(r["key"]).split("/")
        tr = groups.get((lane, ck), {}).get("transcript", {}).get(day)
        if tr is None or tr.get("raw_body_sha256") != \
                r.get("raw_body_sha256"):
            raise InstrumentRefusal(
                "PRESTART_ADMISSION_REFUSED: VIC repair entry for "
                f"{r['key']} is not joined to its staged "
                "transcript's raw_body_sha256")

    # ---- predecessor bridge record vs its staged classes ----------
    import w2_predecessor_bridge_cayley as PB
    brec = json.loads(_pin_bytes(PREDECESSOR_RECORD_PATH)
                      .decode("utf-8"))
    body = {k: v for k, v in brec.items() if k != "bridge_sha256"}
    if hashlib.sha256(json.dumps(
            body, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest() != brec.get("bridge_sha256"):
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: predecessor bridge record "
            "bridge_sha256 does not recompute")
    if brec.get("schema") != PB.BRIDGE_SCHEMA:
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: predecessor record schema "
            f"{brec.get('schema')!r} != registered bridge schema")
    pred = sorted(capsule.get("predecessor", ()))
    if len(pred) != 1:
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: the capsule registers "
            f"{len(pred)} predecessor keys; exactly one is expected")
    pk = (str(brec.get("lane")) + "/" + str(brec.get("carrier"))
          + "/" + str(brec.get("utc_day")))
    if pk != pred[0]:
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: predecessor record names "
            f"{pk}, capsule registers {pred[0]}")
    plane, pck, pday = pred[0].split("/")
    pcls = groups.get((plane, pck), {})
    ptr = pcls.get("transcript", {}).get(pday)
    part = pcls.get("artifact", {}).get(pday)
    if ptr is None or part is None or \
            brec.get("evidence", {}).get("raw_body_sha256") != \
            ptr.get("raw_body_sha256") or \
            brec.get("evidence", {}).get("transcript_sha256") != \
            _canon(ptr) or \
            brec.get("artifact_sha256") != _canon(part):
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: predecessor record evidence "
            "digests do not equal the staged classes for its key")

    admitted = set()
    for k in cap_keys + rep_keys + [pred[0]]:
        parts = str(k).split("/")
        if len(parts) == 3:
            admitted.add(tuple(parts))

    # ---- codex 0655Z item 3: the REGISTERED lineage set admits the
    # restaged keys. reuse_or_bridge comes from the capsule resolved
    # from its accrual_impl pin above (never from a caller); each
    # registered key must have its staged restage carrier, whose
    # content the boundary's lineage branch verifies against the
    # registered entry through the pinned verifier. A registered key
    # with no carrier refuses -- registration is never presence.
    for k in (capsule.get("reuse_or_bridge") or {}):
        parts = str(k).split("/")
        if len(parts) != 3:
            raise InstrumentRefusal(
                "PRESTART_ADMISSION_REFUSED: registered lineage key "
                f"{k!r} is not lane/carrier/day")
        lane_r, ck_r, day_r = parts
        if day_r not in groups.get((lane_r, ck_r), {}).get(
                "restage", {}):
            raise InstrumentRefusal(
                "PRESTART_ADMISSION_REFUSED: registered lineage key "
                f"{k} has no staged restage carrier")
        admitted.add((lane_r, ck_r, day_r))

    # ---- codex 2240Z P0-1 + 2313Z P0-1: the retry-operation chain
    # is validated by THE ONE shared semantic authority in the retry
    # module (validate_admitted_chain) -- the admission boundary and
    # retry finalization apply identical contracts, so a chain that
    # finalization would refuse can never be admitted here, and vice
    # versa. Pinned bytes in, semantics enforced there.
    import w2_capture_retry_404_v4_cayley as RETRY

    def _chain_json(member):
        return json.loads(_pin_bytes(RETRY_CHAIN_PATHS[member])
                          .decode("utf-8"))

    idx_rows = [json.loads(x) for x in
                _pin_bytes(RETRY_CHAIN_PATHS["index"]).decode("utf-8")
                .splitlines() if x.strip()]
    if len(idx_rows) != 1:
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: retry chain: index carries "
            f"{len(idx_rows)} rows; exactly one")
    rk = RETRY.TARGET_KEY
    rlane, rck, rday = rk.split("/")
    rcls = groups.get((rlane, rck), {})
    inv_obj = json.loads(inv_raw.decode("utf-8"))
    try:
        RETRY.validate_admitted_chain(
            {"dispatch": _chain_json("dispatch"),
             "transport_receipt": _chain_json("transport_receipt"),
             "prepared": _chain_json("prepared"),
             "result": _chain_json("result"),
             "classes_complete": _chain_json("classes_complete"),
             "index": idx_rows[0]},
            {"ledger_raw": led_raw,
             "expect_entry_sha": _kat_chain_ctx.get(
                 "expect_entry_sha", RETRY.ORIGINAL_ENTRY_SHA256),
             "class_objs": {cls: rcls.get(cls, {}).get(rday)
                            for cls in STAGED_CLASS_SUFFIX},
             "inventory_entry":
                 (inv_obj.get("objects") or {}).get(rk),
             "require_inventory": True,
             "allow_unpinned": _kat_chain_ctx.get("allow_unpinned",
                                                  False),
             "resolve_commit": _kat_chain_ctx.get("resolve_commit"),
             "reopen_manifest":
                 _kat_chain_ctx.get("reopen_manifest")})
    except SystemExit as exc:
        raise InstrumentRefusal(
            f"PRESTART_ADMISSION_REFUSED: retry chain: {exc}")
    if rk not in ref_keys:
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: retry chain: the retry key "
            "is not in the terminal receipt's REFUSED partition")

    admitted.add((rlane, rck, rday))
    return admitted


def _boundary_mechanics(repo, manifest, *, blob_reader=None,
                        store_reader=None, day_set_gate=None,
                        transform_dispatcher=None,
                        authority_reproducer=None,
                        capture_archive=None,
                        manifest_commit=None, restage_gate=None):
    """codex 2235Z item 1 + 0238Z items 1-2: the admission-owned
    S/T/E consumer. The (lane, carrier, day) key set comes ONLY from
    the REGISTERED expected-keys authority pin (never from the
    submitted pins -- submitted contracts cannot authorize their own
    completeness); a BIJECTION is required between the authority keys
    and the four staged classes plus the inventory; every produced
    artifact is RECOMPUTED from the reopened raw body through the
    registered lane transform (a coordinated artifact+output_sha256
    forgery diverges here); only then does grassmann's five-map join
    run. Returns {report, staged_boundary_sha256}; None while the
    slot is honestly OPEN. All readers/gates injectable for KATs."""
    slot = manifest["slots"].get("producer_boundary")
    if not isinstance(slot, dict) or slot.get("status") != "BOUND":
        return None
    if blob_reader is None:
        def blob_reader(commit, path):
            return _git(repo, ["cat-file", "blob",
                               f"{commit}:{path}"], binary=True)
    store_rep = verify_staged_store(repo, manifest,
                                    blob_reader=blob_reader)
    descriptor = _pinned_json(slot, STORE_DESCRIPTOR_BASENAME,
                              blob_reader)
    inventory = _pinned_json(slot, STAGED_INVENTORY_BASENAME,
                             blob_reader)
    authority = _pinned_json(slot, EXPECTED_KEYS_BASENAME,
                             blob_reader)
    _validate_expected_keys_authority(
        repo, authority, reproducer=authority_reproducer)
    auth_keys = authority["prestart_expected_keys"]
    if sorted(auth_keys) != sorted(PRESTART_LANES):
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: authority lane set is not "
            "the registered PRESTART lanes")
    authorized = set()
    for lane in auth_keys:
        for ck, days in auth_keys[lane].items():
            for d in days:
                authorized.add((lane, ck, d))
    groups = {}
    seen = set()
    for pin in slot.get("pins", ()):
        parsed = _parse_staged_pin(pin.get("path", ""))
        if parsed is None:
            continue
        lane, carrier, day, cls = parsed
        if (lane, carrier, day, cls) in seen:
            raise InstrumentRefusal(
                "PRESTART_ADMISSION_REFUSED: duplicate staged class "
                f"{lane}/{carrier}/{day}/{cls}")
        seen.add((lane, carrier, day, cls))
        if (lane, carrier, day) not in authorized:
            raise InstrumentRefusal(
                "PRESTART_ADMISSION_REFUSED: staged key "
                f"{lane}/{carrier}/{day} is not in the "
                "registered expected-keys authority (DAY_CAPSULE is "
                "accrual-time)")
        raw = blob_reader(pin["commit"], pin["path"])
        if hashlib.sha256(raw).hexdigest() != pin["blob_sha256"]:
            raise InstrumentRefusal(
                "PRESTART_ADMISSION_REFUSED: staged pin bytes "
                f"diverge: {pin['path']}")
        groups.setdefault((lane, carrier), {}).setdefault(
            cls, {})[day] = json.loads(raw.decode("utf-8"))
    if not groups:
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: producer_boundary is BOUND "
            "with no per-day S/T/E pin classes (an inventory plus "
            "descriptor is never a staged boundary)")
    import w2_restage_lineage_grassmann as _RLIN
    for (lane, carrier, day) in sorted(authorized):
        classes = groups.get((lane, carrier), {})
        for cls in ("transcript", "contract", "artifact"):
            if day not in classes.get(cls, {}):
                raise InstrumentRefusal(
                    "PRESTART_ADMISSION_REFUSED: authority key "
                    f"{lane}/{carrier}/{day} lacks staged "
                    f"class {cls} (omission never shrinks the "
                    "expected set)")
        # codex 0655Z item 3: provenance is a tagged XOR -- exactly
        # one of the native envelope record or the restage lineage
        # record, never zero, never both, never one relabelled as
        # the other
        has_rec = day in classes.get("record", {})
        has_rst = day in classes.get("restage", {})
        if has_rec == has_rst:
            raise InstrumentRefusal(
                "PRESTART_ADMISSION_REFUSED: authority key "
                f"{lane}/{carrier}/{day} carries "
                + ("both provenance forms" if has_rec
                   else "no provenance carrier")
                + " (provenance is record XOR restage)")
        if has_rec and classes["record"][day].get("schema") == \
                _RLIN.RESTAGE_SCHEMA:
            raise InstrumentRefusal(
                "PRESTART_ADMISSION_REFUSED: authority key "
                f"{lane}/{carrier}/{day} carries a restage lineage "
                "record relabelled as the native envelope")
    inv_keys = set(inventory.get("objects", {}))
    want_keys = set()
    for (lane, ck, d) in authorized:
        want_keys.add(f"{lane}/{ck}/{d}")
    if inv_keys != want_keys:
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: inventory key set diverges "
            f"from the authority (extra="
            f"{sorted(inv_keys - want_keys)[:3]}, missing="
            f"{sorted(want_keys - inv_keys)[:3]})")
    if store_reader is None:
        def store_reader(desc, relpath):
            base = os.path.realpath(str(desc["physical_root"]))
            pth = os.path.realpath(os.path.join(base, str(relpath)))
            if not pth.startswith(base + os.sep):
                raise InstrumentRefusal(
                    "PRESTART_ADMISSION_REFUSED: store path escape "
                    f"{relpath}")
            with open(pth, "rb") as f:
                return f.read()
    if transform_dispatcher is None:
        # codex 0238Z item 2: FAIL-CLOSED until grassmann's staging
        # driver registers the production lane-transform dispatcher.
        import w2_acquisition_capture_grassmann as CAP
        transform_dispatcher = getattr(CAP, "admission_transform",
                                       None)
        if transform_dispatcher is None:
            raise InstrumentRefusal(
                "PRESTART_ADMISSION_REFUSED: the registered lane "
                "transform dispatcher is not yet available "
                "(grassmann staging driver) -- artifacts are never "
                "admitted digest-only")
    if day_set_gate is None:
        import w2_producer_grassmann as _PROD
        day_set_gate = _PROD.verify_staged_day_set
    import w2_producer_grassmann as _PRODC
    report = {}
    for (lane, carrier), classes in sorted(groups.items()):
        expected_days = sorted(auth_keys[lane][carrier])
        native_days = [d for d in expected_days
                       if d in classes.get("record", {})]
        restage_days = [d for d in expected_days
                        if d in classes.get("restage", {})]
        bodies = {}
        auth_contracts = {}
        for day in expected_days:
            key = f"{lane}/{carrier}/{day}"
            # codex 0320Z item 1: derive the AUTHORITATIVE S and
            # require the submitted sidecar to EQUAL it -- downstream
            # consumes only the authority's entry
            auth_s = authoritative_static_contract(
                authority, lane, carrier, day)
            if json.dumps(classes["contract"][day], sort_keys=True,
                          separators=(",", ":")) != \
                    json.dumps(auth_s, sort_keys=True,
                               separators=(",", ":")):
                raise InstrumentRefusal(
                    "PRESTART_ADMISSION_REFUSED: submitted static "
                    f"contract for {key} diverges from the "
                    "INDEPENDENT authority entry (S is admitted, "
                    "never submitted)")
            auth_contracts[day] = auth_s
            bodies[day] = store_reader(
                descriptor, inventory["objects"][key]["path"])
            if day not in restage_days:
                # 0238Z item 2: RECOMPUTE the produced artifact from
                # the reopened body through the registered transform
                # -- fed the AUTHORITATIVE S. Restaged keys are
                # recomputed through the PINNED transform inside the
                # lineage verifier below instead.
                recomputed = transform_dispatcher(
                    lane, bodies[day], auth_contracts[day])
                if _PRODC._canon_digest(recomputed) != \
                        _PRODC._canon_digest(
                            classes["artifact"][day]):
                    raise InstrumentRefusal(
                        "PRESTART_ADMISSION_REFUSED: produced "
                        f"artifact for {key} diverges from the "
                        "registered transform recomputation (digest "
                        "agreement with E is never derivation)")
        # codex 0655Z item 3, restage branch: the staged S and E
        # must equal the digests the REGISTERED lineage entry binds,
        # the record must name THIS key, and the pinned lineage
        # verifier (which recomputes through the PINNED transform)
        # must pass -- fail closed when no commit context exists
        import w2_producer_grassmann as _PRODR
        restage_att = {}
        for day in restage_days:
            key = f"{lane}/{carrier}/{day}"
            rst = classes["restage"][day]
            if rst.get("v4_key") != key:
                raise InstrumentRefusal(
                    "PRESTART_ADMISSION_REFUSED: restage record "
                    f"staged at {key} names v4_key "
                    f"{rst.get('v4_key')!r} (placement must equal "
                    "the registered v4 key)")
            if _PRODR._canon_digest(auth_contracts[day]) != \
                    rst.get("s_v4_sha256"):
                raise InstrumentRefusal(
                    "PRESTART_ADMISSION_REFUSED: staged contract "
                    f"for {key} does not equal the restage record's "
                    "registered s_v4_sha256")
            if _PRODR._canon_digest(classes["artifact"][day]) != \
                    rst.get("artifact_sha256"):
                raise InstrumentRefusal(
                    "PRESTART_ADMISSION_REFUSED: staged artifact "
                    f"for {key} does not equal the restage record's "
                    "registered artifact_sha256")
            gate = restage_gate
            if gate is None:
                if manifest_commit is None:
                    raise InstrumentRefusal(
                        "PRESTART_ADMISSION_REFUSED: restaged key "
                        f"{key} requires the pinned manifest commit "
                        "for lineage resolution and none was "
                        "supplied (fail closed, never skipped)")
                import w2_restage_lineage_grassmann as _RLING

                def gate(record, transcript, raw_body):
                    return _RLING.verify_restage_lineage_pinned(
                        repo, manifest_commit, record, transcript,
                        raw_body)
            try:
                gate(rst, classes["transcript"][day], bodies[day])
            except InstrumentRefusal:
                raise
            except Exception as e:
                raise InstrumentRefusal(
                    "PRESTART_ADMISSION_REFUSED: restage lineage "
                    f"verification failed for {key}: "
                    f"{type(e).__name__}: {str(e)[:200]}")
            # codex 1309Z fix 1: the VALIDATED restage bindings enter
            # the boundary digest -- two valid lineages with
            # different verified body/S/E/transcript bindings must
            # never share a report digest
            restage_att[day] = {
                "v4_key": key,
                "raw_body_sha256": hashlib.sha256(
                    bodies[day]).hexdigest(),
                "s_sha256": _PRODR._canon_digest(
                    auth_contracts[day]),
                "artifact_sha256": _PRODR._canon_digest(
                    classes["artifact"][day]),
                "transcript_sha256": _PRODR._canon_digest(
                    classes["transcript"][day]),
                "restage_record_sha256": _PRODR._canon_digest(rst)}
        if native_days:
            try:
                out = day_set_gate(
                    {d: classes["record"][d] for d in native_days},
                    {d: bodies[d] for d in native_days},
                    {d: classes["artifact"][d]
                     for d in native_days},
                    {d: auth_contracts[d] for d in native_days},
                    {d: classes["transcript"][d]
                     for d in native_days},
                    native_days, carrier, lane)
            except InstrumentRefusal:
                raise
            except Exception as e:
                raise InstrumentRefusal(
                    f"PRESTART_ADMISSION_REFUSED: S/T/E join failed "
                    f"for {lane}/{carrier}: {e}")
        else:
            out = {}
        report[f"{lane}/{carrier}"] = {
            "days": len(expected_days),
            "native_days": len(native_days),
            "restaged_days": len(restage_days),
            # codex 1309Z fix 1: TAGGED canonical structure -- the
            # native join output and the restage attestations both
            # bind the per-lane digest
            "day_digests_sha256": hashlib.sha256(json.dumps(
                {"native": out, "restage": restage_att},
                sort_keys=True, separators=(",", ":"))
                .encode()).hexdigest()}
    full = {"schema": "f2g-w2-staged-boundary-report-v3",
            "expected_keys_sha256":
                authority["prestart_expected_keys_sha256"],
            "store": store_rep, "lanes": report}
    digest = hashlib.sha256(json.dumps(
        full, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    # ---- v4 part 7 (codex 0527Z finding 5, boundary side) --------
    # The authority says what is EXPECTED; grassmann's capture-run
    # archive says what was ADMITTED. Without it, a key REFUSED at
    # capture is indistinguishable here from one never attempted --
    # and a REFUSED key must NEVER silently satisfy a scientific key.
    # Fail closed: an absent archive can never mean "skip the check".
    if capture_archive is None:
        # codex 1547Z repair 2: the PRODUCTION path derives the
        # partition from the three pinned operation-evidence records.
        # The caller-supplied archive remains ONLY as the structural
        # KAT's fixture door (BP-1 keeps its signature; the shared
        # bar's archive doctors keep their semantics). When neither
        # pinned records nor an archive exist, the refusal inside the
        # derivation still names the missing partition.
        import w2_producer_grassmann as _PJ
        admitted = _derive_admitted_partition(
            slot, blob_reader, manifest, groups, _PJ._canon_digest)
    else:
        import w2_acquisition_capture_grassmann as CAP
        # verify the archive OURSELVES; trusting a caller-verified
        # one would accept a forgery (the content-auth mistake again)
        # codex 0655Z item 3: the capture archive partitions the
        # NATIVE portion of the authority -- restaged keys were
        # never capture attempts. This caller-archive branch is the
        # STRUCTURAL door only; production (archive=None) admits
        # restaged keys from the REGISTERED lineage set instead.
        _native_auth = {
            lane: {ck: [d for d in days
                        if d not in groups.get(
                            (lane, ck), {}).get("restage", {})]
                   for ck, days in auth_keys[lane].items()}
            for lane in auth_keys}
        _rst_body = set()
        _nat_body = set()
        for (ln, ck), cl in groups.items():
            for d in cl.get("restage", {}):
                _ent = inventory.get("objects", {}).get(
                    f"{ln}/{ck}/{d}") or {}
                if _ent.get("sha256"):
                    _rst_body.add(str(_ent["sha256"]))
            for d in cl.get("record", {}):
                _ent = inventory.get("objects", {}).get(
                    f"{ln}/{ck}/{d}") or {}
                if _ent.get("sha256"):
                    _nat_body.add(str(_ent["sha256"]))
        # codex 1309Z fix 2: a content-addressed store may hold ONE
        # body shared by a native and a restaged key -- remove only
        # the restage-EXCLUSIVE digests, or the native archive would
        # falsely miss its own required body
        _rst_body -= _nat_body
        # STRUCTURAL DOOR ONLY: lineage-restaged bodies are
        # accounted by the INVENTORY, not by the native-capture
        # archive, so the archive verifier sweeps a NATIVE store
        # view with exactly those digests removed. CAP's verifier
        # stays byte-untouched -- its source identity is bound by
        # the registered disposition capsule and may not drift in a
        # cayley packet. Production (archive=None) never copies:
        # it derives restage admission from the registered capsule.
        import shutil as _sh
        import tempfile as _tf
        _sd = os.path.realpath(str(descriptor["physical_root"]))
        with _tf.TemporaryDirectory() as _nat_dir:
            if os.path.isdir(_sd):
                for _f in os.listdir(_sd):
                    if _f.endswith(".body") and                             _f[:-len(".body")] not in _rst_body:
                        _sh.copyfile(os.path.join(_sd, _f),
                                     os.path.join(_nat_dir, _f))
            _nat_desc = dict(descriptor, physical_root=_nat_dir)
            try:
                # the archive verifier takes the AUTHORITY key
                # mapping, so it re-derives the partition from the
                # same registered source this boundary uses
                CAP.verify_capture_run_archive(capture_archive,
                                               _nat_desc,
                                               _native_auth)
            except Exception as e:                        # noqa: BLE001
                raise InstrumentRefusal(
                    "PRESTART_ADMISSION_REFUSED: the capture-run "
                    f"archive failed its own verifier "
                    f"({type(e).__name__}: {str(e)[:110]})")
        admitted = set()
        for k in CAP.admitted_keys(capture_archive):
            parts = str(k).split("/")
            if len(parts) == 3:
                admitted.add(tuple(parts))
        for (ln, ck), cl in groups.items():
            for d in cl.get("restage", {}):
                admitted.add((ln, ck, d))
    unmet = sorted(f"{ln}/{ck}/{d}"
                   for (ln, ck, d) in authorized
                   if (ln, ck, d) not in admitted)
    if unmet:
        raise InstrumentRefusal(
            f"PRESTART_ADMISSION_REFUSED: {len(unmet)} expected "
            "authority key(s) are NOT in the archive's ADMITTED "
            f"partition (first: {unmet[:3]}). A REFUSED key preserves "
            "its evidence but never satisfies a scientific key; the "
            "lawful resolutions are re-admission, a registered "
            "ADMITTED_ABSENCE, or removal by an authority amendment "
            "declared BEFORE the failures were seen. Dropping it "
            "here would make the authority data-dependent.")
    # MECHANICS ONLY. Deliberately returns neither
    # staged_boundary_sha256 nor proof kinds: those are admission
    # facts, and this function is shared with a structural KAT that
    # must be incapable of establishing them.
    return {"full": full, "digest": digest, "authority": authority,
            "descriptor": descriptor, "authorized": authorized}


DISPOSITION_CAPSULE_BASENAME = "key_disposition_capsule_v4.json"


def _registered_disposition_capsule(manifest, blob_reader):
    """Resolve the capsule from the REGISTERED accrual_impl pin.

    My 0110Z finding: build_fixture_capsule can mint a capsule over
    the REAL authority claiming all 2,056 keys are native captures,
    and it passes strict verify_lineage_registry with
    bodies_recomputed=0. Nothing in the capsule's own verifier can
    stop that, because the verifier can only check INTERNAL
    consistency against the authority key set. A capsule\'s authority
    comes from its PROVENANCE -- derived by build() from the real
    archive and store, then PINNED -- so the boundary must resolve it
    from the pin rather than trust what a caller hands it. This is
    the same repair codex required on the predecessor bridge, which I
    applied to the dispatcher and failed to apply here.
    """
    slot = manifest.get("slots", {}).get("accrual_impl")
    if not isinstance(slot, dict):
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: no accrual_impl slot pins "
            "the disposition capsule -- the boundary will not accept "
            "a caller-supplied capsule in its place")
    # codex 0151Z P0-3. Found by turning my own self-check on this
    # helper after codex's P0-1 against grassmann's runner, and
    # mutation-proved by both of us. The object check below is sound
    # -- one pin, blob at the pinned commit, SHA recomputed -- but it
    # never asserted that the slot is BOUND. That was correct only
    # INCIDENTALLY, because the generator happens to emit accrual_impl
    # in BOUND_SLOTS and OPEN slots happen to carry zero pins. The
    # slot map is DATA, edited by people who never read this line, so
    # "correct because of how the generator currently builds slots"
    # is the RG-9 shape: right for the state it was written in,
    # silently wrong when that state moves.
    if slot.get("status") != "BOUND":
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: accrual_impl carries a "
            f"capsule pin but its status is {slot.get('status')!r}, "
            "not BOUND -- a pin in a slot the manifest itself "
            "declares unbound may not authorize an admission")
    pins = [p for p in slot.get("pins", ())
            if isinstance(p, dict)
            and str(p.get("path", "")).endswith(
                DISPOSITION_CAPSULE_BASENAME)]
    if len(pins) != 1:
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: accrual_impl must pin "
            f"exactly one {DISPOSITION_CAPSULE_BASENAME} "
            f"(found {len(pins)})")
    pin = pins[0]
    raw = blob_reader(pin["commit"], pin["path"])
    got = hashlib.sha256(raw).hexdigest()
    if got != pin["blob_sha256"]:
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: the pinned disposition "
            f"capsule bytes diverge from the manifest pin ({got[:12]} "
            f"!= {pin['blob_sha256'][:12]})")
    return json.loads(raw.decode("utf-8"))


def bind_registered_capsule(manifest, blob_reader, supplied=None):
    """Resolve the capsule from the pin and REFUSE any substitute.

    Extracted so the lock can exercise it directly against the REAL
    committed capsule and a REAL forgery, rather than asserting on a
    parameter name -- the weak proxy that already cost me the bridge.
    """
    registered = _registered_disposition_capsule(manifest, blob_reader)
    if supplied is not None:
        a = json.dumps(supplied, sort_keys=True, separators=(",", ":"))
        b = json.dumps(registered, sort_keys=True,
                       separators=(",", ":"))
        if a != b:
            raise InstrumentRefusal(
                "PRESTART_ADMISSION_REFUSED: the supplied disposition "
                "capsule is NOT_THE_REGISTERED_CAPSULE. A capsule that "
                "passes its own verifier can still report a FALSE "
                "provenance -- a fixture-built capsule over the real "
                "authority verifies clean with bodies_recomputed=0 "
                "while claiming every key is a native capture -- so "
                "the boundary binds the pin and never the argument.")
    return registered


def verify_staged_boundary(repo, manifest, *, blob_reader=None,
                           store_reader=None, day_set_gate=None,
                           transform_dispatcher=None,
                           authority_reproducer=None,
                           capture_archive=None,
                           disposition_capsule=None,
                           manifest_commit=None, restage_gate=None):
    """THE PRODUCTION ADMISSION BOUNDARY -- strict, and the only
    entrypoint whose result may be consumed as an admission fact.

    The disposition capsule is RESOLVED FROM THE MANIFEST PIN. A
    supplied capsule is permitted only if it is byte-identical to the
    pinned one; it never substitutes for it. Returns
    {report, staged_boundary_sha256, proof_kinds}; None while the slot
    is honestly OPEN.
    """
    mech = _boundary_mechanics(
        repo, manifest, blob_reader=blob_reader,
        store_reader=store_reader, day_set_gate=day_set_gate,
        transform_dispatcher=transform_dispatcher,
        authority_reproducer=authority_reproducer,
        capture_archive=capture_archive,
        manifest_commit=manifest_commit, restage_gate=restage_gate)
    if mech is None:
        return None
    full, digest = mech["full"], mech["digest"]
    authority, descriptor = mech["authority"], mech["descriptor"]
    if blob_reader is None:
        def blob_reader(commit, path):
            return _git(repo, ["cat-file", "blob",
                               f"{commit}:{path}"], binary=True)
    # ---- closure 4 + 0110Z: the capsule comes from the PIN --------
    registered = bind_registered_capsule(manifest, blob_reader,
                                         supplied=disposition_capsule)
    import w2_disposition_capsule_grassmann as DISP
    try:
        # closure 2 (2303Z): the LINEAGE REGISTRY contract, never
        # verify_ceiling -- a ceiling PASS establishes only request
        # membership and reports lineage_evidence_verified=False.
        # Verified against the store this boundary is actually
        # binding, from the REGISTERED descriptor.
        DISP.verify_lineage_registry(
            registered, authority=authority,
            store_root=descriptor["physical_root"])
    except Exception as e:                                # noqa: BLE001
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: the registered disposition "
            f"capsule failed its own verifier ({type(e).__name__}: "
            f"{str(e)[:110]})")
    authorized_keys = {f"{ln}/{ck}/{d}"
                       for (ln, ck, d) in mech["authorized"]}
    partitions = compute_proof_kind_partitions(authorized_keys,
                                               registered)
    total = sum(p["count"] for p in partitions.values())
    # the typed partitions come FIRST; the aggregate is derived from
    # them rather than reported beside them
    full = dict(full)
    full["proof_kinds"] = partitions
    full["proof_kind_total"] = total
    return {"report": full, "staged_boundary_sha256": digest,
            "proof_kinds": partitions}


STRUCTURAL_KAT_SCHEMA = ("claim_scope", "admission_eligible",
                         "proof_kind_status", "authorizes",
                         "structural_kat_sha256", "structure")


def verify_staged_boundary_structure_kat(
        repo, manifest, *, blob_reader=None, store_reader=None,
        day_set_gate=None, transform_dispatcher=None,
        authority_reproducer=None, capture_archive=None,
        manifest_commit=None, restage_gate=None):
    """codex 0057Z P0: the PORTABLE STRUCTURAL KERNEL.

    Exercises the pin walk, staged store, S/T/E bijection, artifact
    recomputation and the archive ADMITTED|REFUSED partition -- the
    mechanics -- on any host, WITHOUT the lineage registry or the
    source-body store.

    Its result is structurally incapable of satisfying admission: it
    carries no staged_boundary_sha256, no proof kinds, and a closed
    stamp saying so. That is what lets the production entrypoint bind
    the pinned capsule strictly without making the shared bar
    permanently red off the evidence host. Portability and admission
    semantics are different things and must not be selected by a flag
    on one function.
    """
    mech = _boundary_mechanics(
        repo, manifest, blob_reader=blob_reader,
        store_reader=store_reader, day_set_gate=day_set_gate,
        transform_dispatcher=transform_dispatcher,
        authority_reproducer=authority_reproducer,
        capture_archive=capture_archive,
        manifest_commit=manifest_commit, restage_gate=restage_gate)
    if mech is None:
        return None
    lanes = dict(mech["full"].get("lanes", {}))
    body = {"claim_scope": "STRUCTURAL_KAT_ONLY",
            "admission_eligible": False,
            "proof_kind_status": "NOT_EVALUATED",
            "authorizes": "NOTHING",
            "structure": {"lanes": lanes,
                          "authority_keys": len(mech["authorized"])}}
    body["structural_kat_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return body


def consume_as_admission(result):
    """THE ANTI-CONFUSION DOOR (codex 0057Z P0).

    Any consumer treating a boundary result as an admission fact must
    pass it through here. A STRUCTURAL_KAT_ONLY stamp refuses -- so a
    structural result can never be read as a production boundary
    result merely because both are dicts with plausible fields.
    """
    if not isinstance(result, dict):
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: boundary result is not a "
            "closed record")
    if result.get("claim_scope") == "STRUCTURAL_KAT_ONLY" or \
            result.get("admission_eligible") is False or \
            result.get("authorizes") == "NOTHING":
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: a STRUCTURAL_KAT_ONLY result "
            "was offered as an admission fact. It establishes "
            "mechanics only -- no lineage registry, no source bodies, "
            "no proof kinds -- and authorizes NOTHING.")
    if "staged_boundary_sha256" not in result or \
            "proof_kinds" not in result:
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: an admission fact requires "
            "both staged_boundary_sha256 and the three proof kinds")
    return result


def assemble_prestart_admission(repo, manifest_commit, bindings,
                                owner_authorization):
    """codex 1815Z item 1: builds the CLOSED admission capsule with
    LIVE verification. Refuses typed PRESTART_ADMISSION_REFUSED when:
    the execution verifier's --prestart walk is not a zero-OPEN PASS
    (today's OPEN manifests refuse here); the runtime allowlist fails;
    or the owner authorization is not a closed binding record whose
    binds match the EXACT manifest commit/blob, lanes, lease, and
    window uuid."""
    import f2g_execution_manifest_verifier_cayley as EMV

    def refuse(detail):
        raise InstrumentRefusal(f"PRESTART_ADMISSION_REFUSED: {detail}")
    verdict = EMV.verify(repo, manifest_commit, prestart=True)
    if verdict.get("verdict") != "PASS" or \
            verdict.get("slots_open", -1) != 0:
        refuse(f"execution verifier --prestart is not a zero-OPEN "
               f"PASS at {manifest_commit}: "
               f"{[t['reason'] for t in verdict.get('typed_reasons', [])][:4]}")
    allowlist = runtime_allowlist_check(repo, manifest_commit)
    raw = _git(repo, ["cat-file", "blob",
                      f"{verdict['manifest_commit']}:"
                      f"{EXEC_MANIFEST_PATH}"], binary=True)
    # codex 2235Z item 1: the FULL staged boundary (named store +
    # every S/T/E carrier + the five-map join) must verify at
    # admission time -- runs whenever the producer_boundary slot is
    # BOUND (always true at a zero-OPEN prestart PASS); its report
    # digest binds into the admission capsule below
    boundary = verify_staged_boundary(
        repo, json.loads(raw.decode("utf-8")),
        manifest_commit=verdict["manifest_commit"])
    boundary_sha = (boundary or {}).get("staged_boundary_sha256")
    blob_sha = hashlib.sha256(raw).hexdigest()
    if not isinstance(owner_authorization, dict) or \
            set(owner_authorization) != {"quote", "quote_sha256",
                                         "binds"}:
        refuse("owner authorization is not a closed binding record "
               "(bare strings refuse)")
    oa = owner_authorization
    if hashlib.sha256(str(oa["quote"]).encode()).hexdigest() != \
            oa["quote_sha256"]:
        refuse("owner quote digest mismatch")
    binds = oa["binds"]
    if binds.get("manifest_blob_sha256") != blob_sha or \
            str(binds.get("manifest_commit")) != str(manifest_commit):
        refuse("owner binding does not match the exact manifest "
               "commit/blob (manifest changed after the binding?)")
    if sorted(binds.get("lanes", ())) != \
            sorted(bindings.get("lane_uuids", ())) or \
            binds.get("lease") != bindings.get("remote_lease") or \
            binds.get("window_uuid") != \
            bindings.get("global_window_uuid"):
        refuse("owner binding diverges from the bindings' "
               "lanes/lease/window")
    admission = {"schema": WB.ADMISSION_SCHEMA,
                 "manifest_commit": str(manifest_commit),
                 "manifest_blob_sha256": blob_sha,
                 "prestart_verifier": {
                     "verdict": verdict["verdict"],
                     "mode": verdict["mode"],
                     "slots_open": verdict["slots_open"],
                     "manifest_commit": verdict["manifest_commit"]},
                 "allowlist": {"pins_checked":
                               allowlist["pins_checked"]},
                 "owner": {"quote": oa["quote"],
                           "quote_sha256": oa["quote_sha256"],
                           "binds": dict(binds)},
                 "lanes": list(bindings["lane_uuids"]),
                 "lease": bindings["remote_lease"],
                 "window_uuid": bindings["global_window_uuid"],
                 "staged_boundary_sha256": boundary_sha}
    admission["admission_digest"] = WB.admission_digest(admission)
    return admission


class PersistentLedger:
    """File-backed w2_barrier.BarrierLedger: save-after-every-append,
    replay-on-load, chain verified both ways."""

    def __init__(self, path, used_leases=(), clock=now_utc_day):
        # `clock` is injectable FOR KATS ONLY; production always uses
        # the default live-UTC read
        self.path = os.path.abspath(path)
        self._clock = clock
        self.ledger = WB.BarrierLedger(used_leases=used_leases)
        if os.path.exists(self.path):
            self._load()

    # -------- persistence --------
    def save(self):
        doc = {"meta": {"lease": self.ledger._lease,
                        "used_leases":
                            sorted(self.ledger._used_leases)},
               "events": self.ledger.events}
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(doc, f, indent=1, sort_keys=True)
            f.write("\n")
        os.replace(tmp, self.path)

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        led = WB.BarrierLedger(
            used_leases=doc["meta"].get("used_leases", ()))
        led.events = doc["events"]
        led.verify_chain()          # tamper check BEFORE any replay
        self._replay(led)
        led._lease = doc["meta"].get("lease")
        self.ledger = led

    @staticmethod
    def _replay(led):
        """Rebuild ALL derived state from the (verified) event list --
        events are the single source of truth."""
        from datetime import date
        for ev in led.events:
            k, p = ev["kind"], ev["payload"]
            if k == "PRESTART":
                led.state = "ACCRUAL"
                led.admitted_lanes = frozenset(p["lanes"])
                led.evaluation_start = date.fromisoformat(
                    p["evaluation_start"])
                led.evaluation_end = date.fromisoformat(
                    p["evaluation_end"])
                led.maturity_tail_end = date.fromisoformat(
                    p["maturity_tail_end"])
                led._bindings_digest = p["bindings_digest"]
            elif k == "PREDICTION":
                led._predictions[(p["region"], p["issue_day"])] = \
                    p["row_digest"]
            elif k == "SUPPORT_BARRIER_CLOSED":
                led.state = "SUPPORT_BARRIER"
            elif k == "OWNER_SEAL":
                led._sealed_lanes.add(p["lane"])
            elif k == "FINAL_FIRE":
                led._first_fired = True
                led._terminal_lanes.add(p["lane"])
            elif k == "VERIFIER_PASS":
                led._verified_lanes.add(p["lane"])
            elif k == "RELEASE":
                led.state = "RELEASED"
            elif k == "WINDOW3_TERMINAL":
                led.state = "WINDOW3_TERMINAL"
            elif k == "SELECTOR_COMMITTED":
                led._selector = frozenset(p["S"])
                led._selector_power = p["power"]

    # -------- wrapped operations (live clock injected here) --------
    def prestart(self, bindings, admission):
        """Low-level pass-through: the ADMISSION CAPSULE is required
        (the barrier refuses bare bindings). Production callers use
        prestart_production, which BUILDS the capsule with live
        verification."""
        self.ledger.prestart(bindings, self._clock(), admission)
        self.save()

    def prestart_production(self, repo, manifest_commit, bindings,
                            owner_authorization):
        """THE production prestart entry point (codex 1815Z item 1):
        constructs and verifies the closed admission capsule LIVE --
        execution verifier in --prestart mode (must be a zero-OPEN
        PASS), runtime allowlist over those bytes, and the owner
        authorization independently verified against the exact
        manifest blob / lanes / lease / window -- then drives the
        barrier."""
        admission = assemble_prestart_admission(
            repo, manifest_commit, bindings, owner_authorization)
        self.prestart(bindings, admission)
        return admission

    def accrue_prediction(self, lease, region, issue_day, row_digest):
        self.ledger.accrue_prediction(lease, region, issue_day,
                                      row_digest, self._clock())
        self.save()

    def producer_receipt(self, lease, receipt_digest):
        self.ledger.producer_receipt(lease, receipt_digest)
        self.save()

    def close_support_barrier(self, lease, role):
        self.ledger.close_support_barrier(lease, self._clock(), role)
        self.save()

    def record_owner_seal(self, lease, lane, seal_digest):
        self.ledger.record_owner_seal(lease, lane, seal_digest)
        self.save()

    def final_fire(self, lease, lane, result_digest):
        self.ledger.final_fire(lease, lane, result_digest)
        self.save()

    def record_verifier_pass(self, lease, lane, verifier_digest):
        self.ledger.record_verifier_pass(lease, lane, verifier_digest)
        self.save()

    def release(self, lease):
        self.ledger.release(lease)
        self.save()

    def commit_selector(self, power_results):
        s = self.ledger.commit_selector(power_results)
        self.save()
        return s


# ===================================================================
# SEAM LAYER 1 (built on the RATIFIED barrier pins + codex-repaired
# selection REV 2; grassmann 0610Z: "accrual seam layers unblocked"):
# PRESTART binding assembly + selection execution. Later layers
# (per-lane accrual runners, adapter panel builds) follow their pins'
# formal ratification in the remaining bar cycles.
# ===================================================================
import w2_selection as WS

DESIGN_MANIFEST_PATH = "docs/f2g_window2_freeze/byte_pin_manifest.json"
W2_CARRIERS = ("istanbul_marmara", "socal_coachella",
               "turkey_kahramanmaras", "cascadia")


def assemble_prestart_bindings(repo, *, execution_manifest_commit,
                               mf4_model_scaler_digest,
                               power_envelope_digest,
                               global_window_uuid, remote_lease,
                               lane_uuids, owner_authorization,
                               hypothesis_registries_digest,
                               calibration_fits_digest,
                               adapters_digest):
    """Builds the eleven-class PRESTART bindings dict. What is
    resolvable from git is RESOLVED here (execution-manifest blob sha =
    code_manifest; the models class carries the design-manifest linkage
    blob sha); everything else is passed through and must be non-empty
    (the barrier re-validates on prestart -- this assembler adds no
    policy, only resolution)."""
    exec_blob = _git(repo, ["cat-file", "blob",
                            f"{execution_manifest_commit}:"
                            f"{EXEC_MANIFEST_PATH}"], binary=True)
    exec_obj = json.loads(exec_blob.decode("utf-8"))
    dm_commit = exec_obj["design_manifest_commit"]
    dm_blob = _git(repo, ["cat-file", "blob",
                          f"{dm_commit}:{DESIGN_MANIFEST_PATH}"],
                   binary=True)
    return {
        "code_manifest": {
            "execution_manifest_commit": execution_manifest_commit,
            "execution_manifest_blob_sha256":
                hashlib.sha256(exec_blob).hexdigest()},
        "models": {
            "design_manifest_commit": dm_commit,
            "design_manifest_blob_sha256":
                hashlib.sha256(dm_blob).hexdigest()},
        "calibration_fits": calibration_fits_digest,
        "hypothesis_registries": hypothesis_registries_digest,
        "adapters": adapters_digest,
        "mf4_model_scaler": mf4_model_scaler_digest,
        "power_envelope": power_envelope_digest,
        "global_window_uuid": global_window_uuid,
        "remote_lease": remote_lease,
        "lane_uuids": list(lane_uuids),
        "owner_authorization": owner_authorization,
    }


def execute_selection(day_records_by_carrier, cutoff):
    """The cutoff-stable selection execution: w2_selection PRODUCTION
    path per carrier (frozen caps; 90-day frame + presence derivation +
    churn all engine-enforced). A typed carrier refusal
    (INSUFFICIENT_POOL) is RECORDED, never a crash and never a silent
    drop; frame/input violations (SelectionInputInvalid) propagate --
    they are instrument-feed defects, not carrier outcomes. Returns the
    registry record with a digest for the PRESTART hypothesis-registry
    binding."""
    registries = {}
    for carrier in W2_CARRIERS:
        if carrier not in day_records_by_carrier:
            raise InstrumentRefusal(
                f"SELECTION_FEED_MISSING: {carrier}")
        try:
            r = WS.select(carrier, day_records_by_carrier[carrier],
                          cutoff)
            registries[carrier] = {
                "selected": r["selected"], "churn": r["churn"],
                "typing": r["typing"]}
        except WS.InsufficientPool as e:
            registries[carrier] = {"selected": None, "churn": None,
                                   "typing": str(e)}
    record = {"schema": "f2g-w2-selection-registry-v1",
              "cutoff": str(cutoff), "registries": registries}
    record["registry_digest"] = hashlib.sha256(json.dumps(
        record, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    return record


# ===================================================================
# SEAM LAYER 2 (on grassmann's 0710Z W-B1B green + formal B1B pin
# ratification; B2B pins ratified at REV 2): the ADAPTER -- window-2
# family panel assembly from producer day capsules. The MF4 per-lane
# runner holds one more cycle for W-MF4's formal ratification.
# ===================================================================
import f2g_sealed_run_instrument_cayley as SRI  # pinned digest formula


def build_family_panel(calendar, registry_record, producer_days):
    """Assembles the graph-family panel (w2_b2b/w2_b1b shape) from
    producer day capsules. Content-auth BEFORE use: every capsule's
    station_index_digest is recomputed via the PINNED producer formula
    (imported from the sealed instrument, never reimplemented) and must
    match -- typed STATION_INDEX_DIGEST_MISMATCH. The adapter assembles
    SHAPE only; family policy (measured/edge consistency, gates,
    floors) lives in the engines. Carriers the frozen selection rule
    typed out (INSUFFICIENT_POOL) are carried as typed_exclusions in
    panel metadata -- recorded, never silently absent."""
    panel = {"calendar": sorted(str(d) for d in calendar),
             "carriers": {}, "typed_exclusions": {}}
    for carrier in sorted(registry_record["registries"]):
        reg = registry_record["registries"][carrier]
        if reg["selected"] is None:
            panel["typed_exclusions"][carrier] = reg["typing"]
            continue
        days_data = producer_days.get(carrier)
        if days_data is None:
            raise InstrumentRefusal(f"PRODUCER_FEED_MISSING: {carrier}")
        measured = {}
        r = {}
        registered = []
        for day in sorted(days_data):
            cap = days_data[day]
            got = SRI.station_index_digest(cap["measured"])
            if got != cap["station_index_digest"]:
                raise InstrumentRefusal(
                    f"STATION_INDEX_DIGEST_MISMATCH: {carrier} {day} "
                    f"got={got[:12]} recorded="
                    f"{str(cap['station_index_digest'])[:12]}")
            registered.append(day)
            measured[day] = sorted(cap["measured"])
            for e, v in cap.get("edges", {}).items():
                r.setdefault(e, {})[day] = float(v)
        panel["carriers"][carrier] = {
            "registry": list(reg["selected"]),
            "registered_days": registered,
            "measured": measured, "r": r}
    return panel


# ===================================================================
# SEAM LAYER 3 (on grassmann's 0811Z W-MF4 green + formal MF4 pin
# ratification): the MF4 per-lane accrual runner -- the last held
# layer. MAG-dependent duties (calibration-ledger production at the
# cutoff) remain, but no layer now waits on W-MAG's wiring.
# ===================================================================
import w2_mf4 as MF4


def emit_mf4_predictions(pl, mf4_ledger, lease, feeds, regions,
                         bboxes, issue_day):
    """One accrual tick: for every admitted region, build the sealed
    prediction row and record it. Order of operations: engine row ->
    verify -> BARRIER ACCRUE (the chain is the authority) -> return.
    The instrument's only data touch is MECHANICAL: the risk series is
    sliced to rows dated <= issue_day (canonicalization, not policy);
    the engine still enforces ISSUE_TIME_VIOLATION fail-closed behind
    it. Typed no-prediction days emit typing rows and are ACCRUED like
    any prediction-of-record -- never silent. Emission timing and
    duplicates are refused by the barrier (LATE_OR_REVISED_PREDICTION);
    embargo means nothing here reads a row back."""
    rows = []
    for region in sorted(regions):
        feed = feeds.get(region)
        if feed is None:
            raise InstrumentRefusal(f"MF4_FEED_MISSING: {region}")
        risk = {d: v for d, v in feed["risk_series"].items()
                if str(d) <= str(issue_day)}
        row = MF4.predict_row(mf4_ledger, risk, feed["events_view"],
                              bboxes[region], region, issue_day,
                              now_utc())
        MF4.verify_row(row)
        pl.accrue_prediction(lease, region, str(issue_day),
                             row["row_digest"])
        rows.append(row)
    return rows


def append_rows_store(store_path, rows):
    """Append-only embargoed JSONL row store. Duplicate (region,
    issue_day) refuses (PREDICTION_ROW_DUPLICATE) -- second guard
    behind the barrier's; every stored row re-verifies its digest on
    the way in. Nothing here reads rows back out; release-time access
    goes through the barrier's embargo gate."""
    seen = set()
    if os.path.exists(store_path):
        with open(store_path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                seen.add((r["region"], r["issue_day"]))
    with open(store_path, "a", encoding="utf-8", newline="\n") as f:
        for row in rows:
            MF4.verify_row(row)
            key = (row["region"], row["issue_day"])
            if key in seen:
                raise MF4.Mf4Refusal(
                    f"PREDICTION_ROW_DUPLICATE: {key} (store)")
            f.write(json.dumps(row, sort_keys=True,
                               separators=(",", ":")) + "\n")
            seen.add(key)
    return len(rows)


# ---------------------------------------------------------------- selftest
def _selftest():
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="w2_accrual_kat_")
    path = os.path.join(tmpdir, "ledger.json")

    def bindings(lease, lanes=("graph", "mag1")):
        return {k: f"digest-{k}" for k in WB.REQUIRED_BINDINGS
                if k not in ("remote_lease", "lane_uuids",
                             "owner_authorization",
                             "code_manifest")} | {
            "code_manifest": {"execution_manifest_commit": "kat-mc",
                              "execution_manifest_blob_sha256":
                                  "kat-mb"},
            "remote_lease": lease,
            "lane_uuids": list(lanes),
            "global_window_uuid": "kat-window",
            "owner_authorization": "kat-owner-quote"}

    # lifecycle with persistence: prestart + predictions, then RELOAD
    fake = ["2026-09-01"]
    pl = PersistentLedger(path, clock=lambda: fake[0])
    b1_ = bindings("kat-lease-1")
    pl.prestart(b1_, WB._admission(b1_))
    day = pl.ledger.evaluation_start.isoformat()
    fake[0] = day                    # clock advances into the window
    pl.accrue_prediction("kat-lease-1", "ra", day, "row-1")
    pl.producer_receipt("kat-lease-1", "acq-1")

    pl2 = PersistentLedger(path, clock=lambda: fake[0])  # replay
    assert pl2.ledger.state == "ACCRUAL"
    assert pl2.ledger._lease == "kat-lease-1"
    assert pl2.ledger.admitted_lanes == frozenset({"graph", "mag1"})
    assert pl2.ledger.evaluation_start == pl.ledger.evaluation_start
    # duplicate prediction still refuses AFTER reload (state rebuilt)
    try:
        pl2.accrue_prediction("kat-lease-1", "ra", day, "row-1b")
        raise AssertionError("duplicate after reload must refuse")
    except WB.BarrierRefusal as e:
        assert "LATE_OR_REVISED_PREDICTION" in str(e)
    # reused-lease protection survives persistence
    try:
        pl2.ledger.prestart(bindings("kat-lease-1"), "2026-09-01")
        raise AssertionError("state must forbid second prestart")
    except WB.BarrierRefusal as e:
        assert "STATE_INVALID" in str(e)

    # codex 1815Z item-1 production-path doctors: the CURRENT OPEN
    # manifest can NEVER cross into ACCRUAL through the production
    # entry point; forged owner records refuse
    repo = os.path.abspath(os.path.join(_HERE, "..", ".."))
    try:
        assemble_prestart_admission(repo, "205e912", b1_,
                                    "not-a-seal")
        raise AssertionError("OPEN manifest must refuse admission")
    except InstrumentRefusal as e:
        # w2r1 cycle-3 succession: the 205e912-era manifest predates
        # the 13-slot SLOT_SET, so the successor verifier refuses it
        # SLOT_SET_NOT_CLOSED before it can refuse SLOT_OPEN. Either
        # typed reason satisfies the doctor's intent -- an OPEN-era
        # manifest can never cross into ACCRUAL.
        assert "PRESTART_ADMISSION_REFUSED" in str(e) \
            and ("SLOT_OPEN" in str(e)
                 or "SLOT_SET_NOT_CLOSED" in str(e))
    q = "the owner quote"
    oa_good_shape = {"quote": q, "quote_sha256":
                     hashlib.sha256(q.encode()).hexdigest(),
                     "binds": {"manifest_commit": "205e912",
                               "manifest_blob_sha256": "wrong",
                               "lanes": ["graph", "mag1"],
                               "lease": "kat-lease-1",
                               "window_uuid": "kat-window"}}
    try:
        assemble_prestart_admission(repo, "205e912", b1_,
                                    oa_good_shape)
        raise AssertionError("must refuse before owner checks too")
    except InstrumentRefusal as e:
        assert "PRESTART_ADMISSION_REFUSED" in str(e)

    # v1.1 appendix gate KATs (codex 1843Z item 4 + 2015Z item 1):
    # the named-store reopen -- unit level, injectable carriers
    inv_fix = {"schema": "f2g-w2-staged-body-inventory-v1",
               "store_id": "s4t", "store_root": "s4t://window2",
               "objects": {"MAG_FEED/izn/2026-01-01": {
                   "path": "ab" * 32 + ".body", "sha256": "ab" * 32,
                   "bytes": 4}}}
    desc_fix = {"schema": "f2g-w2-store-descriptor-v1",
                "store_id": "s4t", "store_root": "s4t://window2",
                "physical_root": os.path.join(tmpdir,
                                              "no-such-store")}
    inv_raw = json.dumps(inv_fix).encode()
    desc_raw = json.dumps(desc_fix).encode()
    inv_pin = {"path": STAGED_PREFIX + STAGED_INVENTORY_BASENAME,
               "commit": "c" * 40,
               "blob_sha256": hashlib.sha256(inv_raw).hexdigest()}
    desc_pin = {"path": STAGED_PREFIX + STORE_DESCRIPTOR_BASENAME,
                "commit": "c" * 40,
                "blob_sha256": hashlib.sha256(desc_raw).hexdigest()}

    def reader(c, path):
        return desc_raw if path.endswith("store_descriptor.json")             else inv_raw

    def man_with(status, pins):
        return {"slots": {"producer_boundary": {
            "status": status, "pins": pins}}}
    # OPEN slot -> no-op (the zero-OPEN gate owns that refusal)
    assert verify_staged_store(".", man_with("OPEN", [])) is None
    # BOUND without the inventory pin -> typed refusal
    try:
        verify_staged_store(".", man_with("BOUND", [desc_pin]))
        raise AssertionError("missing inventory pin must refuse")
    except InstrumentRefusal as e:
        assert "staged_body_inventory" in str(e)
    # BOUND without the DESCRIPTOR pin -> typed refusal (2015Z: the
    # physical root comes only from the registered descriptor)
    try:
        verify_staged_store(".", man_with("BOUND", [inv_pin]),
                            blob_reader=reader)
        raise AssertionError("missing descriptor pin must refuse")
    except InstrumentRefusal as e:
        assert "store_descriptor" in str(e)
    # pinned bytes diverging -> refusal
    try:
        verify_staged_store(
            ".", man_with("BOUND", [inv_pin, desc_pin]),
            blob_reader=lambda c, p: b"{}",
            inventory_verifier=lambda i, d: True)
        raise AssertionError("divergent inventory bytes must refuse")
    except InstrumentRefusal as e:
        assert "diverge from the manifest pin" in str(e)
    # wrong/unavailable store (verifier raises) -> TYPED refusal
    def _boom_store(inv, desc):
        raise ValueError("CAPTURE_STORE_IDENTITY_MISMATCH: kat")
    try:
        verify_staged_store(
            ".", man_with("BOUND", [inv_pin, desc_pin]),
            blob_reader=reader, inventory_verifier=_boom_store)
        raise AssertionError("wrong store must refuse")
    except InstrumentRefusal as e:
        assert "staged store reopen failed" in str(e)             and "CAPTURE_STORE_IDENTITY_MISMATCH" in str(e)
    # the DEFAULT path consumes the REAL REV 7 named-store verifier:
    # a descriptor whose physical root does not exist refuses
    # CAPTURE_STORE_UNAVAILABLE through the real API (never a PASS)
    try:
        verify_staged_store(".", man_with("BOUND",
                                          [inv_pin, desc_pin]),
                            blob_reader=reader)
        raise AssertionError("unavailable named store must refuse")
    except InstrumentRefusal as e:
        assert "CAPTURE_STORE_UNAVAILABLE" in str(e)
    # a passing reopen returns the report
    rep = verify_staged_store(
        ".", man_with("BOUND", [inv_pin, desc_pin]),
        blob_reader=reader,
        inventory_verifier=lambda i, d: {"reopened":
                                         len(i["objects"])})
    assert rep["objects"] == 1 and rep["store_id"] == "s4t"

    # --- codex 2235Z item 1 + 0238Z items 1-2: the S/T/E boundary
    # consumer against the REGISTERED expected-keys authority ---
    import w2_acquisition_capture_grassmann as CAP
    import w2_producer_grassmann as PROD
    broot = os.path.join(tmpdir, "boundary")
    staging = os.path.join(broot, "staging")
    FIX = {}

    def opener(url):
        return FIX[url]

    def bspec(day, lane="MAG_FEED", carrier="izn"):
        return {"lane": lane, "carrier": carrier,
                "utc_day": day,
                "endpoint": "https://kat.example/gin",
                "request_params": {"obs": carrier, "d": day},
                "source": {"kind": "gin-minute",
                           "ref": "https://kat.example/gin"},
                "cutoff": "2026-08-27",
                "operation_params": {"day": day},
                "expected_keys": [day]}

    def bbuilder(body):
        return {"n_bytes": len(body)}

    def bdispatch(lane, body, contract):
        return bbuilder(body)
    days = ("2026-08-20", "2026-08-21")
    blobs = {}
    inv_entries = {}
    pins2 = []
    pre = STAGED_PREFIX

    def add_pin(path, raw):
        blobs[("f" * 40, path)] = raw
        pins2.append({"path": path, "commit": "f" * 40,
                      "blob_sha256":
                          hashlib.sha256(raw).hexdigest()})
    b_keys = {"SELECTION_RECORDS": {"kat_sel": ["2026-08-20"]},
              "MAG_FEED": {"izn": list(days)},
              "MAG_WEATHER_FEED": {"kat_drv": ["2026-08-20",
                                               "2026-08-21"]}}
    # codex 0655Z item 3: kat_drv/2026-08-21 is RESTAGED -- a mixed
    # native+restaged fixture; a native-only fixture cannot close
    # this seam
    _RST_KEY = ("MAG_WEATHER_FEED", "kat_drv", "2026-08-21")
    _rst_raws = {}
    arch_admitted = {}
    fixture_keys = [(lane, ck, d)
                    for lane in sorted(b_keys)
                    for ck in b_keys[lane]
                    for d in b_keys[lane][ck]]
    for (lane, ck, day) in fixture_keys:
        sp = bspec(day, lane=lane, carrier=ck)
        url = PROD.requested_url_of(sp["endpoint"],
                                    sp["request_params"])
        body = f"body-{lane}-{ck}-{day}".encode()
        FIX[url] = (200, {"content-type": "text/plain"}, body,
                    url + "&final=1")
        rp, tp, rec, tr = CAP.capture_day(
            sp, staging, os.path.join(broot, "records"),
            os.path.join(broot, "transcripts"), bbuilder,
            opener=opener,
            clock=lambda d=day: f"{d}T12:00:01Z")
        stem = f"{lane.lower()}_{ck}_{day}"
        if (lane, ck, day) == _RST_KEY:
            # provenance carrier = the lineage record; the native
            # envelope is deliberately NOT staged, and the key is
            # ABSENT from the capture archive (it was never a
            # capture attempt)
            _rr = {"schema": "f2g-w2-restage-lineage-v1",
                   "v4_key": f"{lane}/{ck}/{day}",
                   "s_v4_sha256": PROD._canon_digest(
                       CAP.static_contract_of(sp)),
                   "artifact_sha256": PROD._canon_digest(
                       bbuilder(body))}
            _rst_raws["restage"] = json.dumps(
                _rr, sort_keys=True, separators=(",", ":")).encode()
            _rst_raws["record"] = open(rp, "rb").read()
            _rst_raws["stem"] = stem
            _rst_raws["body_sha"] = hashlib.sha256(body).hexdigest()
            add_pin(pre + f"{stem}.restage.json",
                    _rst_raws["restage"])
        else:
            add_pin(pre + f"{stem}.record.json",
                    open(rp, "rb").read())
        add_pin(pre + f"{stem}.transcript.json",
                open(tp, "rb").read())
        add_pin(pre + f"{stem}.contract.json", json.dumps(
            CAP.static_contract_of(sp), sort_keys=True,
            separators=(",", ":")).encode())
        add_pin(pre + f"{stem}.artifact.json", json.dumps(
            bbuilder(body), sort_keys=True,
            separators=(",", ":")).encode())
        inv_entries[f"{lane}/{ck}/{day}"] = {
            "sha256": rec["raw_body_sha256"],
            "bytes": rec["raw_body_bytes"]}
        if (lane, ck, day) == _RST_KEY:
            continue
        # v4 part 7: the archive entry for this ADMITTED key
        arch_admitted[f"{lane}/{ck}/{day}"] = {
            "lane": lane, "carrier": ck, "utc_day": day,
            "static_contract_sha256":
                tr["static_contract_sha256"],
            "transcript_sha256": PROD._canon_digest(tr),
            "raw_body_sha256": rec["raw_body_sha256"],
            "raw_body_bytes": rec["raw_body_bytes"],
            "output_sha256": PROD._canon_digest(bbuilder(body))}
    b_inv = CAP.build_staged_body_inventory("s4t", "s4t://window2",
                                            inv_entries)
    b_desc = {"schema": "f2g-w2-store-descriptor-v1",
              "store_id": "s4t", "store_root": "s4t://window2",
              "physical_root": staging}
    # the FIXTURE authority artifact: CLOSED shape, recomputed
    # digest, filled templates for every fixture carrier (the
    # census lock rides the production reproducer path)
    def tmpl_for(carrier):
        return {"source": {"kind": "gin-minute",
                           "ref": "https://kat.example/gin"},
                "endpoint": "https://kat.example/gin",
                "request_params": {"obs": carrier, "d": "{day}"},
                "operation_params": {"day": "{day}"}}

    def carriers_block(lane):
        return {"carriers": {ck: {
            "cutoff": "2026-08-27",
            "static_contract_template": tmpl_for(ck)}
            for ck in b_keys[lane]}}
    b_auth = {"schema": AUTHORITY_SCHEMA,
              "registered_probe_authority": {
                  "path": "docs/f2g_window2_execution/"
                          "omni_probe_authority_v4.json",
                  "commit": "a" * 40, "blob_sha256": "b" * 64,
                  "role": "kat fixture lineage"},
              "template_token_vocabulary":
                  list(TEMPLATE_TOKEN_VOCABULARY),
              "prestart_expected_keys": b_keys,
              "prestart_expected_keys_sha256": hashlib.sha256(
                  json.dumps(b_keys, sort_keys=True,
                             separators=(",", ":")).encode()
                  ).hexdigest(),
              "static_layer": {lane: carriers_block(lane)
                               for lane in b_keys},
              "dynamic_layer": {}, "digests": {}, "provenance": {}}

    def b_repro():
        return b_auth
    add_pin(pre + "staged_body_inventory.json",
            json.dumps(b_inv, sort_keys=True,
                       separators=(",", ":")).encode())
    add_pin(pre + "store_descriptor.json",
            json.dumps(b_desc, sort_keys=True,
                       separators=(",", ":")).encode())
    add_pin(pre + "staged_expected_contracts_v4.json",
            json.dumps(b_auth, sort_keys=True,
                       separators=(",", ":")).encode())

    def breader(c, path):
        return blobs[(c, path)]

    def bman(pins):
        return man_with("BOUND", pins)
    b_auth_id = {"commit": "f" * 40,
                 "path": ("docs/f2g_window2_execution/"
                          + EXPECTED_KEYS_BASENAME),
                 "blob_sha256": hashlib.sha256(
                     json.dumps(b_auth, sort_keys=True,
                                separators=(",", ":")).encode()
                     ).hexdigest(),
                 "keys_sha256": b_auth[
                     "prestart_expected_keys_sha256"]}
    b_archive = CAP.build_capture_run_archive(
        "s4t", "s4t://window2", b_auth_id, arch_admitted, {})
    # codex 0057Z P0: these fixture manifests exercise MECHANICS, so
    # they run the STRUCTURAL kernel. They cannot run production --
    # production resolves the disposition capsule from the manifest
    # pin and a six-key fixture manifest carries none. That is the
    # intended property, not a limitation: portable structural
    # testing must be incapable of producing an admission fact.
    _rst_calls = []

    def brestage_gate(record, transcript, raw_body):
        # KAT door for the lineage verifier (a six-key fixture
        # manifest carries no pinned disposition capsule); the
        # mechanics' own v4_key/s_v4/artifact joins still run REAL
        _rst_calls.append((record["v4_key"],
                           hashlib.sha256(raw_body).hexdigest()))
        return {"ok": True}
    out = verify_staged_boundary_structure_kat(
        ".", bman(pins2), capture_archive=b_archive,
        blob_reader=breader, transform_dispatcher=bdispatch,
        authority_reproducer=b_repro, restage_gate=brestage_gate)
    assert out["structure"]["lanes"]["MAG_FEED/izn"]["days"] == 2
    _mw = out["structure"]["lanes"]["MAG_WEATHER_FEED/kat_drv"]
    assert _mw["native_days"] == 1 and _mw["restaged_days"] == 1, _mw
    assert _rst_calls == [("MAG_WEATHER_FEED/kat_drv/2026-08-21",
                           _rst_raws["body_sha"])], _rst_calls
    assert len(out["structural_kat_sha256"]) == 64
    assert out["claim_scope"] == "STRUCTURAL_KAT_ONLY"
    assert out["admission_eligible"] is False
    assert out["proof_kind_status"] == "NOT_EVALUATED"
    assert "staged_boundary_sha256" not in out
    assert "proof_kinds" not in out
    try:
        consume_as_admission(out)
        raise AssertionError(
            "a STRUCTURAL_KAT_ONLY result must never be consumable "
            "as an admission fact")
    except InstrumentRefusal:
        pass
    assert verify_staged_boundary_structure_kat(
        ".", man_with("OPEN", [])) is None

    def brefuses(pins, needle, **kw):
        try:
            verify_staged_boundary_structure_kat(
                ".", bman(pins), capture_archive=b_archive,
                blob_reader=breader,
                transform_dispatcher=kw.pop(
                    "transform_dispatcher", bdispatch),
                authority_reproducer=kw.pop(
                    "authority_reproducer", b_repro),
                restage_gate=kw.pop("restage_gate",
                                    brestage_gate), **kw)
            return False
        except InstrumentRefusal as e:
            return needle in str(e)
    # WHOLE-DAY OMISSION: drop day 21 consistently from all four
    # classes, the inventory, AND the store -- the AUTHORITY still
    # expects it (its own store dir so the store reopen is clean)
    staging1 = os.path.join(broot, "staging_omit")
    os.makedirs(staging1, exist_ok=True)
    for k, e in inv_entries.items():
        if not k.endswith("2026-08-20"):
            continue
        with open(os.path.join(staging, e["sha256"] + ".body"),
                  "rb") as f:
            bb = f.read()
        with open(os.path.join(staging1, e["sha256"] + ".body"),
                  "wb") as f:
            f.write(bb)
    b_inv1 = CAP.build_staged_body_inventory(
        "s4t", "s4t://window2",
        {k: v for k, v in inv_entries.items()
         if k.endswith("2026-08-20")})
    b_desc1 = dict(b_desc, physical_root=staging1)
    omit = [pn for pn in pins2
            if "2026-08-21" not in pn["path"]
            and not pn["path"].endswith("staged_body_inventory.json")
            and not pn["path"].endswith("store_descriptor.json")]
    for name, obj in (("inv1/staged_body_inventory.json", b_inv1),
                      ("inv1/store_descriptor.json", b_desc1)):
        raw1 = json.dumps(obj, sort_keys=True,
                          separators=(",", ":")).encode()
        blobs[("f" * 40, pre + name)] = raw1
        omit.append({"path": pre + name, "commit": "f" * 40,
                     "blob_sha256":
                         hashlib.sha256(raw1).hexdigest()})
    assert brefuses(omit, "lacks staged class")
    # EXTRA inventory key beyond the authority (its object EXISTS in
    # the store, so the KEY-SET check -- not the store reopen -- is
    # what refuses)
    xbody = b"extra-body"
    xsha = hashlib.sha256(xbody).hexdigest()
    with open(os.path.join(staging, xsha + ".body"), "wb") as f:
        f.write(xbody)
    b_inv2 = CAP.build_staged_body_inventory(
        "s4t", "s4t://window2",
        dict(inv_entries, **{"MAG_FEED/izn/2026-08-22": {
            "sha256": xsha, "bytes": len(xbody)}}))
    raw2 = json.dumps(b_inv2, sort_keys=True,
                      separators=(",", ":")).encode()
    blobs[("f" * 40, pre + "inv2/staged_body_inventory.json")] = raw2
    extra = [pn for pn in pins2
             if not pn["path"].endswith("staged_body_inventory.json")]
    extra.append({"path": pre + "inv2/staged_body_inventory.json",
                  "commit": "f" * 40,
                  "blob_sha256": hashlib.sha256(raw2).hexdigest()})
    assert brefuses(extra, "inventory key set diverges")
    os.unlink(os.path.join(staging, xsha + ".body"))   # no leak
    # DUPLICATE parsed class (same basename, second directory)
    dup = [dict(pn) for pn in pins2]
    first = dict(dup[0])
    first["path"] = pre + "dup/" + os.path.basename(first["path"])
    blobs[("f" * 40, first["path"])] = breader("f" * 40,
                                               dup[0]["path"])
    dup.append(first)
    assert brefuses(dup, "outside the exact prefix") or \
        brefuses(dup, "duplicate staged class")
    # WRONG PREFIX: a staged-class basename outside staged_envelopes/
    wrong = [dict(pn) for pn in pins2]
    w0 = dict(wrong[0])
    w0["path"] = "docs/other/" + os.path.basename(w0["path"])
    blobs[("f" * 40, w0["path"])] = breader("f" * 40,
                                            wrong[0]["path"])
    wrong.append(w0)
    assert brefuses(wrong, "outside the exact prefix")
    # ---- codex 0655Z item 3: mixed-provenance mutations ----------
    _rp = pre + _rst_raws["stem"] + ".restage.json"
    _rec_p = pre + _rst_raws["stem"] + ".record.json"
    _no_rst = [q for q in pins2 if q["path"] != _rp]
    assert brefuses(_no_rst, "no provenance carrier")
    blobs[("e" * 40, _rec_p)] = _rst_raws["record"]
    _dbl = pins2 + [{"path": _rec_p, "commit": "e" * 40,
                     "blob_sha256": hashlib.sha256(
                         _rst_raws["record"]).hexdigest()}]
    assert brefuses(_dbl, "both provenance forms")
    blobs[("d" * 40, _rec_p)] = _rst_raws["restage"]
    _rel = _no_rst + [{"path": _rec_p, "commit": "d" * 40,
                       "blob_sha256": hashlib.sha256(
                           _rst_raws["restage"]).hexdigest()}]
    assert brefuses(_rel, "relabelled as the native envelope")
    _v3p = pre + "mf4_feed_kat_drv_2026-08-21.restage.json"
    blobs[("c" * 40, _v3p)] = _rst_raws["restage"]
    assert brefuses(pins2 + [{"path": _v3p, "commit": "c" * 40,
                              "blob_sha256": hashlib.sha256(
                                  _rst_raws["restage"]).hexdigest()}],
                    "unregistered lane stem")

    def _rst_mut(commit, **field):
        obj = json.loads(_rst_raws["restage"].decode("utf-8"))
        obj.update(field)
        raw = json.dumps(obj, sort_keys=True,
                         separators=(",", ":")).encode()
        blobs[(commit, _rp)] = raw
        return _no_rst + [{"path": _rp, "commit": commit,
                           "blob_sha256": hashlib.sha256(
                               raw).hexdigest()}]
    assert brefuses(_rst_mut("b" * 40,
                             v4_key="MAG_FEED/izn/2026-08-20"),
                    "placement must equal")
    assert brefuses(_rst_mut("9" * 40, s_v4_sha256="9" * 64),
                    "registered s_v4_sha256")
    assert brefuses(_rst_mut("8" * 40, artifact_sha256="8" * 64),
                    "registered artifact_sha256")
    # fail closed: no gate and no commit context refuses, never skips
    assert brefuses(pins2, "requires the pinned manifest commit",
                    restage_gate=None)
    print("  0655Z item 3: mixed provenance XOR + restage joins -- "
          "missing/double/relabelled/v3-stem/key-swap/digest "
          "mutations all refuse typed; no-context fails closed")
    # ---- codex 1309Z fix 1: the VALIDATED restage value binds the
    # digests -- variant B differs ONLY in the restaged body, and
    # both the per-lane digest and the full structural digest move
    _rk2 = "MAG_WEATHER_FEED/kat_drv/2026-08-21"

    def _variant(tag, rst_entry, staging_v):
        pins_v = [pn for pn in pins2
                  if not pn["path"].endswith(
                      "staged_body_inventory.json")
                  and not pn["path"].endswith(
                      "store_descriptor.json")]
        inv_v = CAP.build_staged_body_inventory(
            "s4t", "s4t://window2",
            dict(inv_entries, **{_rk2: rst_entry}))
        desc_v = dict(b_desc, physical_root=staging_v)
        for name_v, obj_v in (
                (tag + "/staged_body_inventory.json", inv_v),
                (tag + "/store_descriptor.json", desc_v)):
            raw_v = json.dumps(obj_v, sort_keys=True,
                               separators=(",", ":")).encode()
            blobs[("f" * 40, pre + name_v)] = raw_v
            pins_v.append({"path": pre + name_v, "commit": "f" * 40,
                           "blob_sha256": hashlib.sha256(
                               raw_v).hexdigest()})
        return pins_v

    def _copy_bodies(dst, skip_rst=True):
        os.makedirs(dst, exist_ok=True)
        for k_c, e_c in inv_entries.items():
            if skip_rst and k_c == _rk2:
                continue
            with open(os.path.join(staging,
                                   e_c["sha256"] + ".body"),
                      "rb") as f:
                bb_c = f.read()
            with open(os.path.join(dst, e_c["sha256"] + ".body"),
                      "wb") as f:
                f.write(bb_c)
    staging2 = os.path.join(broot, "staging_att")
    _copy_bodies(staging2)
    _body_b = b"body-restaged-variant-B"
    _sha_b = hashlib.sha256(_body_b).hexdigest()
    with open(os.path.join(staging2, _sha_b + ".body"), "wb") as f:
        f.write(_body_b)
    out_b = verify_staged_boundary_structure_kat(
        ".", bman(_variant("attb", {"sha256": _sha_b,
                                    "bytes": len(_body_b)},
                           staging2)),
        capture_archive=b_archive, blob_reader=breader,
        transform_dispatcher=bdispatch,
        authority_reproducer=b_repro, restage_gate=brestage_gate)
    _lA = out["structure"]["lanes"]["MAG_WEATHER_FEED/kat_drv"]
    _lB = out_b["structure"]["lanes"]["MAG_WEATHER_FEED/kat_drv"]
    assert _lA["day_digests_sha256"] != _lB["day_digests_sha256"], \
        "restage body change must move the per-lane digest"
    assert out["structural_kat_sha256"] != \
        out_b["structural_kat_sha256"], \
        "restage body change must move the full structural digest"
    # ---- codex 1309Z fix 2: a native and a restaged key may share
    # ONE content-addressed body -- the valid shared fixture passes,
    # and an altered shared body still refuses
    _shared_key = "MAG_FEED/izn/2026-08-20"
    _shared = inv_entries[_shared_key]
    staging3 = os.path.join(broot, "staging_shared")
    _copy_bodies(staging3)
    pins_c = _variant("attc", {"sha256": _shared["sha256"],
                               "bytes": _shared["bytes"]}, staging3)
    out_c = verify_staged_boundary_structure_kat(
        ".", bman(pins_c), capture_archive=b_archive,
        blob_reader=breader, transform_dispatcher=bdispatch,
        authority_reproducer=b_repro, restage_gate=brestage_gate)
    assert out_c["claim_scope"] == "STRUCTURAL_KAT_ONLY", \
        "the deduplicated shared-body fixture must PASS"
    with open(os.path.join(staging3,
                           _shared["sha256"] + ".body"),
              "wb") as f:
        f.write(b"altered-shared")
    assert brefuses(pins_c, "CAPTURE_INVENTORY_OBJECT_MISMATCH")
    print("  1309Z fixes: restage attestations bind the boundary "
          "digests (variant moves both); a shared native/restage "
          "body passes and its alteration still refuses")
    # DAY_CAPSULE at PRESTART refuses as unauthorized
    dc = [dict(pn) for pn in pins2]
    dcr = breader("f" * 40, pre + "mag_feed_izn_2026-08-20"
                  ".record.json")
    dc_path = pre + "day_capsule_cascadia_2026-08-20.record.json"
    blobs[("f" * 40, dc_path)] = dcr
    dc.append({"path": dc_path, "commit": "f" * 40,
               "blob_sha256": hashlib.sha256(dcr).hexdigest()})
    assert brefuses(dc, "not in the "
                    "registered expected-keys authority")
    # COORDINATED artifact + output_sha256 forgery: both agree, the
    # transform recomputation diverges (codex 0238Z item 2)
    forged_art = {"n_bytes": 999, "fabricated": True}
    forged_sha = PROD._canon_digest(forged_art)
    co = [dict(pn) for pn in pins2]
    for pn in co:
        if pn["path"].endswith("_2026-08-20.artifact.json"):
            raw = json.dumps(forged_art, sort_keys=True,
                             separators=(",", ":")).encode()
            blobs[("2" * 40, pn["path"])] = raw
            pn["commit"] = "2" * 40
            pn["blob_sha256"] = hashlib.sha256(raw).hexdigest()
        if pn["path"].endswith("_2026-08-20.record.json"):
            rec0 = json.loads(breader("f" * 40, pn["path"]))
            rec0["output_sha256"] = forged_sha
            raw = json.dumps(rec0, sort_keys=True,
                             separators=(",", ":")).encode()
            blobs[("2" * 40, pn["path"])] = raw
            pn["commit"] = "2" * 40
            pn["blob_sha256"] = hashlib.sha256(raw).hexdigest()
    assert brefuses(co, "diverges from the registered transform "
                    "recomputation")
    # dispatcher ABSENT -> fail-closed (never digest-only admission)
    if not hasattr(CAP, "admission_transform"):
        try:
            verify_staged_boundary_structure_kat(
                ".", bman(pins2), capture_archive=b_archive,
                blob_reader=breader, authority_reproducer=b_repro)
            raise AssertionError("absent dispatcher must refuse")
        except InstrumentRefusal as e:
            assert "transform dispatcher is not yet available" \
                in str(e)
    # TAMPERED pin bytes refuse before any gate
    tam = [dict(pn) for pn in pins2]
    tam[0] = dict(tam[0], blob_sha256="0" * 64)
    assert brefuses(tam, "diverge")

    # --- codex 0320Z item 3: authority-capsule doctors ---
    import copy as _cp

    def auth_pin_for(a):
        raw = json.dumps(a, sort_keys=True,
                         separators=(",", ":")).encode()
        path = pre + "staged_expected_contracts_v4.json"
        blobs[("9" * 40, path)] = raw
        return {"path": path, "commit": "9" * 40,
                "blob_sha256": hashlib.sha256(raw).hexdigest()}

    def with_auth(a):
        pins_a = [pn for pn in pins2 if not pn["path"].endswith(
            "staged_expected_contracts_v4.json")]
        pins_a.append(auth_pin_for(a))
        return pins_a
    # forged key digest
    a_forge = _cp.deepcopy(b_auth)
    a_forge["prestart_expected_keys_sha256"] = "attested-key-digest"
    assert brefuses(with_auth(a_forge), "does not recompute",
                    authority_reproducer=lambda: a_forge)
    # empty carrier map
    a_empty = _cp.deepcopy(b_auth)
    a_empty["prestart_expected_keys"]["MAG_WEATHER_FEED"] = {}
    a_empty["prestart_expected_keys_sha256"] = hashlib.sha256(
        json.dumps(a_empty["prestart_expected_keys"],
                   sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    assert brefuses(with_auth(a_empty), "carrier map is empty",
                    authority_reproducer=lambda: a_empty)
    # duplicate day
    a_dup = _cp.deepcopy(b_auth)
    a_dup["prestart_expected_keys"]["MAG_FEED"]["izn"] = [
        days[0], days[0]]
    a_dup["prestart_expected_keys_sha256"] = hashlib.sha256(
        json.dumps(a_dup["prestart_expected_keys"],
                   sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    assert brefuses(with_auth(a_dup), "not unique ascending",
                    authority_reproducer=lambda: a_dup)
    # shifted / non-canonical day
    a_shift = _cp.deepcopy(b_auth)
    a_shift["prestart_expected_keys"]["MAG_FEED"]["izn"] = [
        days[1], "2026-8-20"]      # sorted; the DAY is non-canonical
    a_shift["prestart_expected_keys_sha256"] = hashlib.sha256(
        json.dumps(a_shift["prestart_expected_keys"],
                   sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    assert brefuses(with_auth(a_shift), "non-canonical day",
                    authority_reproducer=lambda: a_shift)
    # reproduction divergence (an authority that is internally valid
    # but is NOT what the pinned generator produces)
    assert brefuses(pins2, "does not REPRODUCE",
                    authority_reproducer=lambda: dict(
                        b_auth, provenance={"x": 1}))
    # OPEN token in the CONSUMED template -> the two-phase gate
    a_open = _cp.deepcopy(b_auth)
    a_open["static_layer"]["MAG_FEED"]["carriers"]["izn"][
        "static_contract_template"]["request_params"] = \
        "OPEN_REVIEW_ROUND"
    assert brefuses(with_auth(a_open), "OPEN tokens",
                    authority_reproducer=lambda: a_open)

    # --- codex 0320Z item 1: the coordinated evil-S/T/E case -- the
    # authority registers kat.example; internally consistent carriers
    # built against evil.example REFUSE at the S-admission equality
    evil_root = os.path.join(broot, "evil")
    e_day = "2026-08-20"
    e_sp = dict(bspec(e_day), endpoint="https://evil.example/data",
                source={"kind": "gin-minute",
                        "ref": "https://evil.example/data"})
    e_url = PROD.requested_url_of(e_sp["endpoint"],
                                  e_sp["request_params"])
    # the GENUINE body for that key -- content-addressed staging
    # reuses the existing object; only S diverges (the exact codex
    # reproduction: real bytes, unadmitted operation record)
    FIX[e_url] = (200, {"content-type": "text/plain"},
                  f"body-MAG_FEED-izn-{e_day}".encode(),
                  e_url + "&final=1")
    e_rp, e_tp, e_rec, e_tr = CAP.capture_day(
        e_sp, staging, os.path.join(evil_root, "records"),
        os.path.join(evil_root, "transcripts"), bbuilder,
        opener=opener, clock=lambda: f"{e_day}T12:00:01Z")
    evil = [dict(pn) for pn in pins2]
    stem20 = "mag_feed_izn_2026-08-20"
    for pn in evil:
        if pn["path"] == pre + f"{stem20}.record.json":
            raw = open(e_rp, "rb").read()
        elif pn["path"] == pre + f"{stem20}.transcript.json":
            raw = open(e_tp, "rb").read()
        elif pn["path"] == pre + f"{stem20}.contract.json":
            raw = json.dumps(CAP.static_contract_of(e_sp),
                             sort_keys=True,
                             separators=(",", ":")).encode()
        else:
            continue
        blobs[("3" * 40, pn["path"])] = raw
        pn["commit"] = "3" * 40
        pn["blob_sha256"] = hashlib.sha256(raw).hexdigest()
    assert brefuses(evil, "diverges from the "
                    "INDEPENDENT authority entry")

    # barrier-level: bare bindings refuse even with valid-shape state
    try:
        PersistentLedger(os.path.join(tmpdir, "bare.json"),
                         clock=lambda: "2026-09-01").prestart(
            bindings("kat-lease-bare"), None)
        raise AssertionError("bare bindings must refuse")
    except WB.BarrierRefusal as e:
        assert "PRESTART_ADMISSION_REFUSED" in str(e)

    # tampered file -> chain broken on load
    doc = json.load(open(path))
    doc["events"][1]["payload"]["region"] = "tampered"
    with open(path, "w") as f:
        json.dump(doc, f)
    try:
        PersistentLedger(path)
        raise AssertionError("tampered file must refuse")
    except WB.BarrierRefusal as e:
        assert "LEDGER_CHAIN_BROKEN" in str(e)

    repo = os.path.abspath(os.path.join(_HERE, "..", ".."))
    # The committed production authority must pass the exact closed
    # schema/reproducer gate. This catches generator/consumer field or
    # schema drift before the freeze and capture paths are exercised.
    prod_auth_path = os.path.join(
        repo, "docs", "f2g_window2_execution",
        EXPECTED_KEYS_BASENAME)
    with open(prod_auth_path, encoding="utf-8") as f:
        _validate_expected_keys_authority(repo, json.load(f))

    # runtime allowlist: clean walk over the real BOUND pins, then a
    # doctored on-disk module must be NAMED in the violation
    # resolve the CURRENT manifest commit dynamically -- slots flip
    # BOUND over the manifest's life and the allowlist must track it
    man_commit = _git(repo, ["log", "-1", "--format=%h", "HEAD", "--",
                             EXEC_MANIFEST_PATH])
    rep = runtime_allowlist_check(repo, man_commit)
    assert rep["pins_checked"] >= 3 and rep["manifest_state"] in (
        "OPEN", "CLOSED")
    target = os.path.join(
        repo, "monitoring", "src", "f2g_design_pin_verifier_cayley.py")
    saved = open(target, "rb").read()
    try:
        with open(target, "ab") as f:
            f.write(b"\n# doctored\n")
        try:
            runtime_allowlist_check(repo, man_commit)
            raise AssertionError("doctored disk module must refuse")
        except InstrumentRefusal as e:
            assert "RUNTIME_ALLOWLIST_VIOLATION" in str(e) \
                and "f2g_design_pin_verifier_cayley.py" in str(e)
    finally:
        with open(target, "wb") as f:
            f.write(saved)
    rep = runtime_allowlist_check(repo, man_commit)   # restored clean
    assert rep["pins_checked"] >= 3

    # --- seam layer 1 KATs ---
    from datetime import date as _date, timedelta as _td
    cut = "2026-08-20"
    days90 = [(_date(2026, 8, 20) - _td(days=89 - i)).isoformat()
              for i in range(90)]
    sts = [f"S{i:02d}" for i in range(18)]
    full = {d: list(sts) for d in days90}
    thin = {d: ["T0", "T1"] for d in days90}      # pool 2 < min 8
    feeds = {"istanbul_marmara": full, "socal_coachella": full,
             "turkey_kahramanmaras": thin, "cascadia": full}
    rec = execute_selection(feeds, cut)
    assert len(rec["registries"]["cascadia"]["selected"]) == 16
    assert len(rec["registries"]["socal_coachella"]["selected"]) == 18
    tk = rec["registries"]["turkey_kahramanmaras"]
    assert tk["selected"] is None \
        and "INSUFFICIENT_POOL" in tk["typing"]
    assert rec["registry_digest"] == execute_selection(
        feeds, cut)["registry_digest"]            # deterministic
    try:
        execute_selection({k: v for k, v in feeds.items()
                           if k != "cascadia"}, cut)
        raise AssertionError("missing carrier feed must refuse")
    except InstrumentRefusal as e:
        assert "SELECTION_FEED_MISSING" in str(e)
    try:
        execute_selection(dict(feeds, cascadia={days90[0]: sts}), cut)
        raise AssertionError("bad frame must propagate")
    except WS.SelectionInputInvalid as e:
        assert "LOOKBACK_FRAME_INVALID" in str(e)

    b = assemble_prestart_bindings(
        repo, execution_manifest_commit=man_commit,
        mf4_model_scaler_digest="kat-mf4", power_envelope_digest="kat-env",
        global_window_uuid="kat-uuid", remote_lease="kat-lease-b",
        lane_uuids=["graph", "mag1", "mf4"],
        owner_authorization="kat-owner",
        hypothesis_registries_digest="kat-reg",
        calibration_fits_digest="kat-cal", adapters_digest="kat-adp")
    assert set(b) == set(WB.REQUIRED_BINDINGS)
    assert b["models"]["design_manifest_commit"].startswith("5fba544")
    pl3 = PersistentLedger(os.path.join(tmpdir, "l3.json"),
                           clock=lambda: "2026-09-01")
    pl3.prestart(b, WB._admission(b))  # assembled dict + admission
    assert pl3.ledger.state == "ACCRUAL"

    # --- seam layer 2 KATs: adapter -> REAL engine end-to-end ---
    import w2_b2b as W2B
    A = [f"A{i}" for i in range(5)]
    Bs = [f"B{i}" for i in range(5)]
    reg10 = sorted(A + Bs)
    cal6 = [f"2026-10-{i:02d}" for i in range(1, 7)]

    def mk_edges(cluster_a, cluster_b, strong=5.0, weak=0.1):
        ew = {}
        nodes = list(cluster_a) + list(cluster_b)
        for i, x in enumerate(nodes):
            for y in nodes[i + 1:]:
                same = (x in cluster_a) == (y in cluster_a)
                ew["|".join(sorted((x, y)))] = strong if same else weak
        return ew

    prod = {"cascadia": {
        d: {"measured": reg10,
            "station_index_digest": SRI.station_index_digest(reg10),
            "edges": mk_edges(A, Bs)} for d in cal6}}
    reg_rec = {"registries": {
        "cascadia": {"selected": reg10, "churn": 1.0, "typing": None},
        "turkey_kahramanmaras": {"selected": None, "churn": None,
                                 "typing": "INSUFFICIENT_POOL: kat"}}}
    panel = build_family_panel(cal6, reg_rec, prod)
    assert panel["typed_exclusions"]["turkey_kahramanmaras"] \
        .startswith("INSUFFICIENT_POOL")
    res = W2B.w2_b2b_family(panel, doc_sha256="ab" * 32, n_draws=49)
    assert res["runs_by_carrier"]["cascadia"] == 1 \
        and res["p_value"] == 1.0, res      # adapter -> engine e2e

    bad = json.loads(json.dumps(prod))
    bad["cascadia"][cal6[2]]["station_index_digest"] = "0" * 64
    try:
        build_family_panel(cal6, reg_rec, bad)
        raise AssertionError("doctored digest must refuse")
    except InstrumentRefusal as e:
        assert "STATION_INDEX_DIGEST_MISMATCH" in str(e)
    try:
        build_family_panel(cal6, {"registries": {
            "cascadia": reg_rec["registries"]["cascadia"]}}, {})
        raise AssertionError("missing producer feed must refuse")
    except InstrumentRefusal as e:
        assert "PRODUCER_FEED_MISSING" in str(e)

    # --- seam layer 3 KATs: MF4 runner through the REAL engine +
    # REAL barrier chain ---
    import numpy as _np
    bbox_k = {"min_lat": 30.0, "max_lat": 40.0,
              "min_lon": -125.0, "max_lon": -115.0}
    bboxes_k = {"ra": bbox_k, "rb": bbox_k}
    rngk = _np.random.Generator(_np.random.PCG64(13))
    cal_days = [(_date(2025, 10, 10) + _td(days=i)).isoformat()
                for i in range(120)]
    risk_k = {r: {d: float(rngk.uniform(0, 1)) for d in cal_days}
              for r in ("ra", "rb")}
    ev_k = [{"day": (_date(2025, 11, 1) + _td(days=7 * i)).isoformat(),
             "lat": 35.0, "lon": -120.0, "mag": 4.5} for i in range(8)]
    mf4_led = MF4.calibrate(risk_k, ev_k, bboxes_k, ["ra", "rb"],
                            "2026-02-10", "2026-02-08")

    fake2 = ["2026-09-01"]
    pl4 = PersistentLedger(os.path.join(tmpdir, "l4.json"),
                           clock=lambda: fake2[0])
    b4_ = bindings("kat-lease-4", lanes=("mf4",))
    pl4.prestart(b4_, WB._admission(b4_))
    d0 = pl4.ledger.evaluation_start.isoformat()
    fake2[0] = d0

    # feeds carry FULL history incl the issue day + one FUTURE-dated
    # row: the mechanical slice removes the future row (the engine
    # would refuse it fail-closed otherwise)
    d_future = (pl4.ledger.evaluation_start + _td(days=1)).isoformat()
    hist_days = [(_date(2026, 8, 20) + _td(days=i)).isoformat()
                 for i in range(14)]
    hist_days = [d for d in hist_days if d <= d0] + [d0, d_future]
    feeds = {r: {"risk_series": {d: 0.5 for d in dict.fromkeys(
        hist_days)}, "events_view": ev_k} for r in ("ra", "rb")}
    rows = emit_mf4_predictions(pl4, mf4_led, "kat-lease-4", feeds,
                                ["ra", "rb"], bboxes_k, d0)
    assert len(rows) == 2 and all("p_model" in r for r in rows)
    assert sum(1 for ev in pl4.ledger.events
               if ev["kind"] == "PREDICTION") == 2
    # duplicate emission refuses AT THE BARRIER
    try:
        emit_mf4_predictions(pl4, mf4_led, "kat-lease-4", feeds,
                             ["ra"], bboxes_k, d0)
        raise AssertionError("duplicate emission must refuse")
    except WB.BarrierRefusal as e:
        assert "LATE_OR_REVISED_PREDICTION" in str(e)
    # typed no-prediction day (no prior-day risk) emits + accrues
    d1 = (pl4.ledger.evaluation_start + _td(days=3)).isoformat()
    fake2[0] = d1
    feeds_thin = {"ra": {"risk_series": {d1: 0.5},
                         "events_view": ev_k}}
    rows2 = emit_mf4_predictions(pl4, mf4_led, "kat-lease-4",
                                 feeds_thin, ["ra"], bboxes_k, d1)
    assert "typing" in rows2[0] \
        and "NO_PREDICTION" in rows2[0]["typing"]
    assert sum(1 for ev in pl4.ledger.events
               if ev["kind"] == "PREDICTION") == 3
    try:
        emit_mf4_predictions(pl4, mf4_led, "kat-lease-4", {},
                             ["ra"], bboxes_k, d1)
        raise AssertionError("missing feed must refuse")
    except InstrumentRefusal as e:
        assert "MF4_FEED_MISSING" in str(e)
    # row store: append + duplicate guard + digest verify on the way in
    store = os.path.join(tmpdir, "mf4_rows.jsonl")
    assert append_rows_store(store, rows) == 2
    try:
        append_rows_store(store, rows[:1])
        raise AssertionError("store duplicate must refuse")
    except MF4.Mf4Refusal as e:
        assert "PREDICTION_ROW_DUPLICATE" in str(e)

    # {day_next} substitution KAT (grassmann freeze condition 1):
    # Month/year boundaries prove real UTC date arithmetic. Unknown
    # brace tokens refuse HERE, before capture can make a network call.
    auth_dn = {"template_token_vocabulary":
                   list(TEMPLATE_TOKEN_VOCABULARY),
               "static_layer": {"L": {"carriers": {"c": {
        "cutoff": "2026-08-27",
        "static_contract_template": {
            "source": {"kind": "fdsn", "ref": "r"},
            "endpoint": "https://x.example/q",
            "request_params": {"starttime": "{day}T00:00:00",
                               "endtime": "{day_next}T00:00:00"},
            "operation_params": {"window": "[{day}, {day_next})"}
        }}}}}}
    sc = authoritative_static_contract(auth_dn, "L", "c",
                                       "2026-08-31")
    rp = sc["request_params"]
    assert rp["starttime"] == "2026-08-31T00:00:00"
    assert rp["endtime"] == "2026-09-01T00:00:00"
    assert sc["operation_params"]["window"] ==         "[2026-08-31, 2026-09-01)"
    sc2 = authoritative_static_contract(auth_dn, "L", "c",
                                        "2026-12-31")
    assert sc2["request_params"]["endtime"] ==         "2027-01-01T00:00:00"                 # year boundary
    # {day_compact}: OMNIWeb compact-date form
    auth_cp = json.loads(json.dumps(auth_dn))
    auth_cp["static_layer"]["L"]["carriers"]["c"][
        "static_contract_template"]["request_params"] = {
        "start_date": "{day_compact}", "end_date": "{day_compact}"}
    sc3 = authoritative_static_contract(auth_cp, "L", "c",
                                        "2026-08-31")
    assert sc3["request_params"] == {"start_date": "20260831",
                                     "end_date": "20260831"}
    auth_bad = json.loads(json.dumps(auth_dn))
    auth_bad["static_layer"]["L"]["carriers"]["c"][
        "static_contract_template"]["request_params"]["odd"] = \
        "{day_prev}"
    try:
        authoritative_static_contract(auth_bad, "L", "c",
                                      "2026-08-31")
        raise AssertionError("unregistered template token must refuse")
    except InstrumentRefusal as e:
        assert "unregistered tokens" in str(e)

    # ---- codex 1547Z repair 1: prefix controls ON THE CONSUMER ----
    ok = _parse_staged_pin(
        STAGED_PREFIX + "mag_feed_new_2026-01-02.record.json")
    assert ok == ("MAG_FEED", "new", "2026-01-02", "record"), ok
    try:
        _parse_staged_pin(STAGED_PREFIX_RETIRED
                          + "mag_feed_new_2026-01-02.record.json")
        raise AssertionError("retired v3 prefix must refuse")
    except InstrumentRefusal as e:
        assert "RETIRED v3 prefix" in str(e), str(e)[:120]
    try:
        _parse_staged_pin("docs/elsewhere/"
                          "mag_feed_new_2026-01-02.record.json")
        raise AssertionError("mixed/foreign prefix must refuse")
    except InstrumentRefusal as e:
        assert "outside the exact prefix" in str(e)
    print("  1547Z r1: consumer parses v4, refuses RETIRED v3 and "
          "foreign prefixes typed")

    # ---- codex 1547Z repair 2 + 2240Z P0-1: derivation doctors --
    # ONE consolidated fixture: the three operation-evidence records
    # AND the six-member retry chain, because the production
    # derivation requires the chain unconditionally (the retry
    # happened; the frozen ledger honestly still marks its key
    # REFUSED, and only the verified chain may admit it).
    import w2_predecessor_bridge_cayley as _PB
    import w2_capture_retry_404_v4_cayley as _RETRY

    def _dig(b):
        return hashlib.sha256(b).hexdigest()

    def _canon(o):
        return hashlib.sha256(json.dumps(
            o, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()

    _rk = _RETRY.TARGET_KEY
    _rlane, _rck, _rday = _rk.split("/")
    _r_body_sha = "77" * 32
    _r_tr = {"raw_body_sha256": _r_body_sha, "lane": _rlane,
             "carrier": _rck, "utc_day": _rday,
             "requested_url": _RETRY.REGISTERED_REQUEST_URL,
             "effective_url": _RETRY.REGISTERED_REQUEST_URL,
             "http_status": 200, "raw_body_bytes": 5}
    _r_rec = {"raw_body_sha256": _r_body_sha, "kat": "record"}
    _r_ct = {"kat": "contract"}
    _r_art = {"kat": "artifact"}
    _r_map = {"contract": _canon(_r_ct), "artifact": _canon(_r_art),
              "record": _canon(_r_rec), "transcript": _canon(_r_tr)}

    # codex 0655Z item 4: the fixture carries the REAL cardinality
    # (212 VIC rows) and the REAL closed-dict identity shape, derived
    # from the authoritative producer -- a single string-identity row
    # could never construct the uniformity property being tested.
    import datetime as _dt
    import w2_acquisition_capture_grassmann as _CAPK
    _vic_days = [(_dt.date(2026, 1, 1) + _dt.timedelta(days=i)
                  ).isoformat() for i in range(212)]
    _vic_keys = [f"MAG_FEED/vic/{d}" for d in _vic_days]
    _ident = _CAPK.transform_identity_from_source(b"kat-vic-source")
    _led_rows = [{"key": "MAG_FEED/new/2026-01-01", "seq": 0,
                  "status": "CAPTURED"}]
    _seq = 1
    for _vk in _vic_keys:
        if _seq == 81:
            _led_rows.append({"key": _rk, "seq": 81,
                              "status": "REFUSED",
                              "refusal":
                                  "CAPTURE_HTTP_STATUS: 404"})
            _seq += 1
        _led_rows.append({"key": _vk, "seq": _seq,
                          "status": "REFUSED", "refusal": "frame"})
        _seq += 1
    _rk_row = next(r for r in _led_rows if r["key"] == _rk)
    _led_raw = ("\n".join(json.dumps(r, sort_keys=True)
                          for r in _led_rows) + "\n").encode()
    _inv_raw = json.dumps(
        {"objects": {_rk: {"path": _r_body_sha + ".body"}}}).encode()
    _vic_tr = {"raw_body_sha256": "ab" * 32}
    _pred_tr = {"raw_body_sha256": "cd" * 32}
    _pred_art = {"outcome": "ADMITTED"}
    _groups = {("MAG_FEED", "vic"): {"transcript":
                                     {d: dict(_vic_tr)
                                      for d in _vic_days}},
               ("MAG_WEATHER_FEED", "omni"): {
                   "transcript": {"2026-01-01": _pred_tr},
                   "artifact": {"2026-01-01": _pred_art}},
               (_rlane, _rck): {
                   "contract": {_rday: _r_ct},
                   "artifact": {_rday: _r_art},
                   "record": {_rday: _r_rec},
                   "transcript": {_rday: _r_tr}}}
    _receipt = {"schema": TERMINAL_RECEIPT_SCHEMA,
                "ledger_sha256": _dig(_led_raw),
                "ledger_lines": len(_led_rows),
                "admitted_keys": ["MAG_FEED/new/2026-01-01"],
                "refused_keys": sorted(_vic_keys + [_rk]),
                "inventory_sha256": _dig(_inv_raw)}
    _rep_rows = [{"key": _vk,
                  "raw_body_sha256": "ab" * 32,
                  "transform_identity": dict(_ident),
                  "http_requests": 0} for _vk in _vic_keys]
    _brec = {"schema": _PB.BRIDGE_SCHEMA, "lane": "MAG_WEATHER_FEED",
             "carrier": "omni", "utc_day": "2026-01-01",
             "evidence": {"raw_body_sha256": "cd" * 32,
                          "transcript_sha256": _canon(_pred_tr)},
             "artifact_sha256": _canon(_pred_art)}
    _brec["bridge_sha256"] = _canon(
        {k: v for k, v in _brec.items() if k != "bridge_sha256"})
    _capsule = {"http_capture": ["MAG_FEED/new/2026-01-01"]
                + list(_vic_keys) + [_rk],
                "predecessor": ["MAG_WEATHER_FEED/omni/2026-01-01"],
                "reuse_or_bridge": []}
    # FULL valid response evidence: the shared chain authority now
    # applies the retry module's transport value checks, so the
    # fixture must construct genuine semantics, not a shape token
    _ev = {"status": 200,
           "requested_url": _RETRY.REGISTERED_REQUEST_URL,
           "effective_url": _RETRY.REGISTERED_REQUEST_URL,
           "request_start_utc": "2026-08-27T21:51:00Z",
           "response_complete_utc": "2026-08-27T21:51:02Z",
           "headers": {}, "body_bytes_seen": 5}

    def _sd(obj, field):
        obj[field] = _canon({k: v for k, v in obj.items()
                             if k != field})
        return obj

    def _mk_chain(**over):
        d = {"schema": "f2g-w2-retry-404-dispatch-v1", "key": _rk,
             "owner_authorization": "kat", "contract": "kat",
             "original_ledger": {"path": CAPTURE_LEDGER_PATH,
                                 "sha256": _dig(_led_raw),
                                 "seq": 81,
                                 "entry_sha256":
                                     _canon(_rk_row)},
             "manifest_commit": "a" * 40,
             "manifest_blob_sha256": "b" * 64,
             "capsule_pin_commit": "c" * 40,
             "capsule_sha256": "d" * 64,
             "attempt_id": "f" * 32,
             "executed_code": {"path": "kat",
                               "disk_sha256": "0" * 64,
                               "pin_commit": None,
                               "pin_blob_sha256": None,
                               "note": "kat"},
             "store": {"id": "kat", "root": "kat"},
             "expected_classes": ["kat"],
             "registered_request_url":
                 _RETRY.REGISTERED_REQUEST_URL,
             "max_logical_http_operations": 1,
             "vic_http_operations": 0,
             "dispatched_utc": "2026-08-27T21:50:00Z"}
        _sd(d, "dispatch_sha256")
        rcpt = {"schema": "f2g-w2-retry-transport-receipt-v1",
                "kind": "response",
                "dispatch_sha256": d["dispatch_sha256"],
                "attempt_id": d["attempt_id"], "evidence": dict(_ev)}
        _sd(rcpt, "receipt_sha256")
        pr = {"schema": "f2g-w2-retry-prepared-v1", "key": _rk,
              "dispatch_sha256": d["dispatch_sha256"],
              "outcome": "CAPTURED_ADMITTED",
              "class_canon_sha256": dict(_r_map),
              "opener_calls": 1, "transport": dict(_ev),
              "terminal_ledger_sha256_recomputed": _dig(_led_raw),
              "terminal_ledger_unchanged": True,
              "completed_utc": "2026-08-27T21:53:00Z"}
        _sd(pr, "prepared_sha256")
        # the canonical projection ITSELF, via the retry module's
        # projector -- the chain authority requires exact equality
        res = _RETRY._expected_admitted_result(
            {"expect_url": _RETRY.REGISTERED_REQUEST_URL}, d, pr,
            _r_rec)
        mark = {"schema": "f2g-w2-retry-classes-complete-v1",
                "key": _rk, "class_canon_sha256": dict(_r_map)}
        idx = {"key": _rk, "outcome": "CAPTURED_ADMITTED",
               "dispatch_sha256": d["dispatch_sha256"],
               "result_sha256": res["result_sha256"],
               "opener_calls": 1,
               "http_operations_authorized": 1}
        chain = {"dispatch": d, "transport_receipt": rcpt,
                 "prepared": pr, "result": res,
                 "classes_complete": mark, "index": idx}
        chain.update(over)
        return chain

    def _mk_blobs(receipt=None, rep_rows=None, brec=None, led=None,
                  inv=None, chain=None, drop=None):
        receipt = _receipt if receipt is None else receipt
        rep_rows2 = _rep_rows if rep_rows is None else rep_rows
        brec2 = _brec if brec is None else brec
        led2 = _led_raw if led is None else led
        inv2 = _inv_raw if inv is None else inv
        chain = _mk_chain() if chain is None else chain
        rep_raw = ("\n".join(json.dumps(r, sort_keys=True)
                             for r in rep_rows2)
                   + ("\n" if rep_rows2 else "")).encode()
        blobs = {TERMINAL_RECEIPT_PATH:
                 json.dumps(receipt).encode(),
                 CAPTURE_LEDGER_PATH: led2,
                 VIC_REPAIR_LEDGER_PATH: rep_raw,
                 PREDECESSOR_RECORD_PATH:
                 json.dumps(brec2).encode(),
                 STAGED_PREFIX + STAGED_INVENTORY_BASENAME: inv2}
        for m, pth in RETRY_CHAIN_PATHS.items():
            if m == drop:
                continue
            obj = chain[m]
            if m == "index":
                blobs[pth] = (json.dumps(obj, sort_keys=True)
                              + "\n").encode()
            else:
                blobs[pth] = json.dumps(obj).encode()
        slot = {"status": "BOUND", "pins": [
            {"path": pth, "commit": "kat",
             "blob_sha256": _dig(raw)}
            for pth, raw in blobs.items()]}

        def blob(commit, pth):
            return blobs[pth]
        return slot, blob

    _real_cap = globals()["_registered_disposition_capsule"]
    globals()["_registered_disposition_capsule"] = \
        lambda manifest, blob_reader: _capsule
    globals()["_kat_chain_ctx"] = {"allow_unpinned": True,
                                   "resolve_commit": str,
                                   "reopen_manifest": str,
                                   "expect_entry_sha":
                                       _canon(_rk_row)}
    try:
        slot, blob = _mk_blobs()
        adm = _derive_admitted_partition(slot, blob, {}, _groups,
                                         _canon)
        _want_adm = {("MAG_FEED", "new", "2026-01-01"),
                     ("MAG_WEATHER_FEED", "omni", "2026-01-01"),
                     (_rlane, _rck, _rday)}
        _want_adm.update(("MAG_FEED", "vic", d) for d in _vic_days)
        assert adm == _want_adm, (len(adm), len(_want_adm))

        def _must_refuse(needle, **over):
            slot2, blob2 = _mk_blobs(**over)
            try:
                _derive_admitted_partition(slot2, blob2, {}, _groups,
                                           _canon)
                raise AssertionError(f"derivation must refuse "
                                     f"({needle})")
            except InstrumentRefusal as e:
                assert needle in str(e), \
                    f"wanted {needle!r} got {str(e)[:140]!r}"
        # ---- the 1547Z three-record doctors, over the merged fixture
        _must_refuse("partition does not recompute",
                     receipt=dict(_receipt,
                                  admitted_keys=[
                                      "MAG_FEED/new/2026-01-01",
                                      "MAG_FEED/vic/2026-01-01"],
                                  refused_keys=[_rk]))
        _must_refuse("ledger_sha256 does not recompute",
                     led=b'{"key": "x"}\n')
        _must_refuse("not exactly the registered VIC key set",
                     rep_rows=[])
        def _rr():
            return [dict(r) for r in _rep_rows]
        _m = _rr()
        _m[0]["http_requests"] = 1
        _must_refuse("the replay is zero-HTTP", rep_rows=_m)
        _m = _rr()
        _m[0]["raw_body_sha256"] = "ee" * 32
        _must_refuse("not joined to its staged", rep_rows=_m)
        # ---- codex 0655Z item 4: closed-dict identity KATs --------
        # 212 equal dicts is the happy path above; here: non-dict,
        # missing, and one-changed-field -- each refuses TYPED
        _m = _rr()
        _m[0]["transform_identity"] = "ident-1"
        _must_refuse("closed transform-identity", rep_rows=_m)
        _m = _rr()
        del _m[1]["transform_identity"]
        _must_refuse("closed transform-identity", rep_rows=_m)
        _m = _rr()
        _m[2]["transform_identity"] = dict(_ident,
                                           source_sha256="9" * 64)
        _must_refuse("non-uniform", rep_rows=_m)
        # ---- 0655Z item 3: the REGISTERED lineage set admits ------
        _caps_rst = dict(_capsule)
        _caps_rst["reuse_or_bridge"] = {
            "MAG_FEED/rlin/2026-01-01": {"kat": 1}}
        globals()["_registered_disposition_capsule"] = (
            lambda manifest, blob_reader: _caps_rst)
        _groups_rst = dict(_groups)
        _groups_rst[("MAG_FEED", "rlin")] = {
            "restage": {"2026-01-01": {
                "schema": "f2g-w2-restage-lineage-v1"}}}
        _slotR, _blobR = _mk_blobs()
        _admR = _derive_admitted_partition(_slotR, _blobR, {},
                                           _groups_rst, _canon)
        assert ("MAG_FEED", "rlin", "2026-01-01") in _admR
        try:
            _derive_admitted_partition(_slotR, _blobR, {}, _groups,
                                       _canon)
            raise AssertionError("registered lineage key with no "
                                 "staged carrier must refuse")
        except InstrumentRefusal as e:
            assert "no staged restage carrier" in str(e), \
                str(e)[:140]
        globals()["_registered_disposition_capsule"] = (
            lambda manifest, blob_reader: _capsule)
        _must_refuse("bridge_sha256 does not recompute",
                     brec=dict(_brec, artifact_sha256="ff" * 32))
        _wrong = {k: v for k, v in _brec.items()
                  if k != "bridge_sha256"}
        _wrong["utc_day"] = "2026-01-02"
        _wrong["bridge_sha256"] = _canon(_wrong)
        _must_refuse("capsule registers", brec=_wrong)
        # decoy inventory at a same-basename unregistered path
        _decoy_raw = b'{"objects": {"forged": 1}}'
        _slotD, _blobD = _mk_blobs(
            receipt=dict(_receipt,
                         inventory_sha256=_dig(_decoy_raw)))
        _slotD = {"status": "BOUND",
                  "pins": [{"path": "docs/attacker/"
                            + STAGED_INVENTORY_BASENAME,
                            "commit": "kat",
                            "blob_sha256": _dig(_decoy_raw)}]
                  + list(_slotD["pins"])}
        _bD = _blobD

        def _blobD2(commit, pth):
            if pth == "docs/attacker/" + STAGED_INVENTORY_BASENAME:
                return _decoy_raw
            return _bD(commit, pth)
        try:
            _derive_admitted_partition(_slotD, _blobD2, {}, _groups,
                                       _canon)
            raise AssertionError("decoy inventory must refuse")
        except InstrumentRefusal as e:
            assert "EXACT registered" in str(e), str(e)[:140]
        # a missing evidence pin refuses WITH the archive needle
        _slot3, _blob3 = _mk_blobs()
        _slot3 = {"status": "BOUND",
                  "pins": [q for q in _slot3["pins"]
                           if q["path"] != TERMINAL_RECEIPT_PATH]}
        try:
            _derive_admitted_partition(_slot3, _blob3, {}, _groups,
                                       _canon)
            raise AssertionError("missing receipt pin must refuse")
        except InstrumentRefusal as e:
            assert "no capture-run archive was supplied" in str(e)
        print("  1547Z r2: admitted partition derives from pinned "
              "operation records; every doctored record refuses "
              "typed; missing pins carry the archive needle")

        # ---- the 2240Z P0-1 retry-chain doctors -------------------
        for m in RETRY_CHAIN_PATHS:
            _must_refuse("lacks exactly one", drop=m)
        _c = _mk_chain()
        _c["dispatch"] = dict(_c["dispatch"], contract="doctored")
        _must_refuse("RETRY_RECORD_SELF_DIGEST", chain=_c)
        # result-only forgery, in BOTH shapes: an incoherent rewrite
        # dies at the identity join; a COHERENT result+index rewrite
        # must still die at the cross-member map divergence
        _c = _mk_chain()
        _r2 = dict(_c["result"])
        _r2["scientific"] = dict(_r2["scientific"],
                                 classes_published={"x": "y"})
        _sd(_r2, "result_sha256")
        _c["result"] = _r2
        _must_refuse("identities do not join", chain=_c)
        _c["index"] = dict(_c["index"],
                           result_sha256=_r2["result_sha256"])
        # the shared authority catches the COHERENT rewrite at the
        # canonical-projection equality, upstream of the map check
        _must_refuse("canonical projection", chain=_c)
        _c = _mk_chain()
        _d2 = dict(_c["dispatch"], key="MAG_FEED/vic/2026-01-01")
        _sd(_d2, "dispatch_sha256")
        _c["dispatch"] = _d2
        _must_refuse("RETRY_RECORD_WRONG_KEY", chain=_c)
        for n in (0, 2):
            _c = _mk_chain()
            _p2 = dict(_c["prepared"], opener_calls=n)
            _sd(_p2, "prepared_sha256")
            _c["prepared"] = _p2
            _must_refuse("one-opener", chain=_c)
        _c = _mk_chain()
        _p2 = dict(_c["prepared"],
                   terminal_ledger_sha256_recomputed="9" * 64)
        _sd(_p2, "prepared_sha256")
        _c["prepared"] = _p2
        _must_refuse("unchanged-ledger", chain=_c)
        _c = _mk_chain()
        _c["classes_complete"] = {
            "schema": "f2g-w2-retry-classes-complete-v1",
            "key": _rk,
            "class_canon_sha256": dict(_r_map, record="8" * 64)}
        _must_refuse("diverge across", chain=_c)
        # published class divergence (groups-side)
        _g3 = dict(_groups)
        _g3[(_rlane, _rck)] = dict(
            _groups[(_rlane, _rck)],
            artifact={_rday: {"kat": "tampered"}})
        slotX, blobX = _mk_blobs()
        try:
            _derive_admitted_partition(slotX, blobX, {}, _g3, _canon)
            raise AssertionError("class divergence must refuse")
        except InstrumentRefusal as e:
            assert "does not recompute to the chain" in str(e)
        # codex 2313Z P0-1: the coherent ALL-SIX-member 599 + evil
        # URL mutation -- every self-digest recomputed, every
        # cross-member equality preserved; only SEMANTICS can refuse
        _evil_ev = dict(_ev, status=599,
                        requested_url="evil://not-the-dispatched-url",
                        effective_url="evil://not-the-dispatched-url")
        _c = _mk_chain()
        _d6 = {k: v for k, v in _c["dispatch"].items()
               if k != "dispatch_sha256"}
        _sd(_d6, "dispatch_sha256")
        _rc6 = {"schema": "f2g-w2-retry-transport-receipt-v1",
                "kind": "response",
                "dispatch_sha256": _d6["dispatch_sha256"],
                "attempt_id": _d6["attempt_id"],
                "evidence": dict(_evil_ev)}
        _sd(_rc6, "receipt_sha256")
        _p6 = {k: v for k, v in _c["prepared"].items()
               if k != "prepared_sha256"}
        _p6["transport"] = dict(_evil_ev)
        _p6["dispatch_sha256"] = _d6["dispatch_sha256"]
        _sd(_p6, "prepared_sha256")
        _r6 = _RETRY._expected_admitted_result(
            {"expect_url": _RETRY.REGISTERED_REQUEST_URL}, _d6,
            _p6, _r_rec)
        _i6 = _RETRY._index_entry_of(_d6, _r6)
        _c6 = {"dispatch": _d6, "transport_receipt": _rc6,
               "prepared": _p6, "result": _r6,
               "classes_complete": _c["classes_complete"],
               "index": _i6}
        _must_refuse("RETRY_CHAIN_SEMANTICS", chain=_c6)

        # inventory/body divergence
        _inv_bad = json.dumps({"objects": {
            _rk: {"path": "0" * 64 + ".body"}}}).encode()
        _must_refuse("joined through the final inventory",
                     inv=_inv_bad,
                     receipt=dict(_receipt,
                                  inventory_sha256=_dig(_inv_bad)))
    finally:
        globals()["_registered_disposition_capsule"] = _real_cap
        globals()["_kat_chain_ctx"] = {}
    print("  2240Z P0-1: the verified six-member retry chain admits "
          "the former-404 key; every omitted/mutated member, "
          "forgery, wrong key, opener 0/2, ledger change, marker "
          "divergence and inventory divergence refuses typed")

    # cycle-3 succession doctors (codex cycle-2 finding 1 item
    # 4): the superseded v3 authority cannot satisfy the
    # successor gate. Real committed artifacts; PRODUCTION
    # reproducer path (reproducer=None -> pinned v4 generator).
    _repo = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", ".."))
    with open(os.path.join(
            _repo, "docs", "f2g_window2_execution",
            "staged_expected_contracts_v3.json"),
            encoding="utf-8") as _f:
        _auth3 = json.load(_f)
    try:
        _validate_expected_keys_authority(_repo, _auth3)
        raise SystemExit(
            "v3 authority must refuse at the v4 gate")
    except InstrumentRefusal as _ex:
        assert "schema" in str(_ex), str(_ex)
    _auth3r = json.loads(json.dumps(_auth3))
    _auth3r["schema"] = AUTHORITY_SCHEMA
    try:
        _validate_expected_keys_authority(_repo, _auth3r)
        raise SystemExit(
            "relabeled v3 authority must refuse")
    except InstrumentRefusal as _ex:
        assert "REPRODUCE" in str(_ex), str(_ex)
    print("  cycle-3 authority succession: committed v3 authority "
          "refuses (schema); relabeled v3 refuses (does not "
          "REPRODUCE from the pinned v4 generator)")

    print("w2_accrual_instrument selftest: ALL PASS")


if __name__ == "__main__":
    _selftest()
