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
STAGED_PREFIX = "docs/f2g_window2_execution/staged_envelopes/"
EXPECTED_KEYS_BASENAME = "staged_expected_contracts_v3.json"
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
    for cls, suf in STAGED_CLASS_SUFFIX.items():
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
                if path != STAGED_PREFIX + base:
                    raise InstrumentRefusal(
                        "PRESTART_ADMISSION_REFUSED: staged-class "
                        f"pin outside the exact prefix: {path}")
                return lane, carrier, day, cls
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
#   + selection 4 x 90d (08-27 cutoff)
AUTHORITY_SCHEMA = "f2g-w2-expected-contracts-v3"
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
        refuse("template token vocabulary is not the registered v3 set")
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
        import w2_expected_contracts_gen_cayley as GEN

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


def verify_staged_boundary(repo, manifest, *, blob_reader=None,
                           store_reader=None, day_set_gate=None,
                           transform_dispatcher=None,
                           authority_reproducer=None,
                           capture_archive=None,
                           disposition_capsule=None):
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
    for (lane, carrier, day) in sorted(authorized):
        classes = groups.get((lane, carrier), {})
        for cls in STAGED_CLASS_SUFFIX:
            if day not in classes.get(cls, {}):
                raise InstrumentRefusal(
                    "PRESTART_ADMISSION_REFUSED: authority key "
                    f"{lane}/{carrier}/{day} lacks staged "
                    f"class {cls} (omission never shrinks the "
                    "expected set)")
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
            # 0238Z item 2: RECOMPUTE the produced artifact from the
            # reopened body through the registered transform -- fed
            # the AUTHORITATIVE S
            recomputed = transform_dispatcher(
                lane, bodies[day], auth_contracts[day])
            if _PRODC._canon_digest(recomputed) != \
                    _PRODC._canon_digest(classes["artifact"][day]):
                raise InstrumentRefusal(
                    "PRESTART_ADMISSION_REFUSED: produced artifact "
                    f"for {key} diverges from the registered "
                    "transform recomputation (digest agreement with "
                    "E is never derivation)")
        try:
            out = day_set_gate(
                classes["record"], bodies, classes["artifact"],
                auth_contracts, classes["transcript"],
                expected_days, carrier, lane)
        except InstrumentRefusal:
            raise
        except Exception as e:
            raise InstrumentRefusal(
                f"PRESTART_ADMISSION_REFUSED: S/T/E join failed for "
                f"{lane}/{carrier}: {e}")
        report[f"{lane}/{carrier}"] = {
            "days": len(expected_days),
            "day_digests_sha256": hashlib.sha256(json.dumps(
                out, sort_keys=True, separators=(",", ":"))
                .encode()).hexdigest()}
    full = {"schema": "f2g-w2-staged-boundary-report-v2",
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
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: no capture-run archive was "
            "supplied -- the boundary cannot bind without the "
            "registered ADMITTED|REFUSED partition")
    import w2_acquisition_capture_grassmann as CAP
    # verify the archive OURSELVES; trusting a caller-verified one
    # would accept a forgery (the content-auth mistake again)
    try:
        # the archive verifier takes the AUTHORITY key mapping, so
        # it re-derives the partition from the same registered source
        # this boundary uses -- not from a list we flattened
        CAP.verify_capture_run_archive(capture_archive, descriptor,
                                       auth_keys)
    except Exception as e:                                # noqa: BLE001
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: the capture-run archive "
            f"failed its own verifier ({type(e).__name__}: "
            f"{str(e)[:110]})")
    admitted = set()
    for k in CAP.admitted_keys(capture_archive):
        parts = str(k).split("/")
        if len(parts) == 3:
            admitted.add(tuple(parts))
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
    # ---- closure 4: typed proof-kind partitions -----------------
    # Three DIFFERENT claims are being made across the 2,056 keys. A
    # report aggregated by lane/carrier lets them disappear into one
    # total that reads with the strength of its strongest member.
    if disposition_capsule is None:
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: no disposition capsule was "
            "supplied -- the boundary cannot report proof kinds "
            "without the registered dispositions, and an absent "
            "capsule can never mean 'report one undifferentiated "
            "total'")
    import w2_disposition_capsule_grassmann as DISP
    try:
        # closure 2 (2303Z): the boundary must use the LINEAGE
        # REGISTRY contract, never verify_ceiling -- a ceiling PASS
        # only establishes request membership and explicitly reports
        # lineage_evidence_verified=False. It fails CLOSED without
        # the source-body store, which is correct: the boundary can
        # only pass where the evidence actually lives.
        DISP.verify_lineage_registry(disposition_capsule,
                                     authority=authority)
    except Exception as e:                                # noqa: BLE001
        raise InstrumentRefusal(
            "PRESTART_ADMISSION_REFUSED: the disposition capsule "
            f"failed its own verifier ({type(e).__name__}: "
            f"{str(e)[:110]})")
    authorized_keys = {f"{ln}/{ck}/{d}"
                       for (ln, ck, d) in authorized}
    partitions = compute_proof_kind_partitions(authorized_keys,
                                               disposition_capsule)
    total = sum(p["count"] for p in partitions.values())
    # the typed partitions come FIRST; the aggregate is reported only
    # after them, and is derived from them rather than beside them
    full = dict(full)
    full["proof_kinds"] = partitions
    full["proof_kind_total"] = total
    return {"report": full, "staged_boundary_sha256": digest,
            "proof_kinds": partitions}


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
        repo, json.loads(raw.decode("utf-8")))
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
        assert "PRESTART_ADMISSION_REFUSED" in str(e) \
            and "SLOT_OPEN" in str(e)
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
    inv_pin = {"path": "docs/f2g_window2_execution/staged_envelopes/"
                       "staged_body_inventory.json",
               "commit": "c" * 40,
               "blob_sha256": hashlib.sha256(inv_raw).hexdigest()}
    desc_pin = {"path": "docs/f2g_window2_execution/staged_envelopes/"
                        "store_descriptor.json",
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
    pre = "docs/f2g_window2_execution/staged_envelopes/"

    def add_pin(path, raw):
        blobs[("f" * 40, path)] = raw
        pins2.append({"path": path, "commit": "f" * 40,
                      "blob_sha256":
                          hashlib.sha256(raw).hexdigest()})
    b_keys = {"SELECTION_RECORDS": {"kat_sel": ["2026-08-20"]},
              "MAG_FEED": {"izn": list(days)},
              "MAG_WEATHER_FEED": {"kat_drv": ["2026-08-20"]}}
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
        add_pin(pre + f"{stem}.record.json", open(rp, "rb").read())
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
    add_pin(pre + "staged_expected_contracts_v3.json",
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
    out = verify_staged_boundary(".", bman(pins2),
                                 capture_archive=b_archive,
                                 blob_reader=breader,
                                 transform_dispatcher=bdispatch,
                                 authority_reproducer=b_repro)
    assert out["report"]["lanes"]["MAG_FEED/izn"]["days"] == 2
    assert len(out["staged_boundary_sha256"]) == 64
    assert verify_staged_boundary(".", man_with("OPEN", [])) is None

    def brefuses(pins, needle, **kw):
        try:
            verify_staged_boundary(".", bman(pins),
                                   capture_archive=b_archive,
                                   blob_reader=breader,
                                   transform_dispatcher=kw.pop(
                                       "transform_dispatcher",
                                       bdispatch),
                                   authority_reproducer=kw.pop(
                                       "authority_reproducer",
                                       b_repro), **kw)
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
            verify_staged_boundary(".", bman(pins2),
                                   capture_archive=b_archive,
                                   blob_reader=breader,
                                   authority_reproducer=b_repro)
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
        path = pre + "staged_expected_contracts_v3.json"
        blobs[("9" * 40, path)] = raw
        return {"path": path, "commit": "9" * 40,
                "blob_sha256": hashlib.sha256(raw).hexdigest()}

    def with_auth(a):
        pins_a = [pn for pn in pins2 if not pn["path"].endswith(
            "staged_expected_contracts_v3.json")]
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

    print("w2_accrual_instrument selftest: ALL PASS")


if __name__ == "__main__":
    _selftest()
