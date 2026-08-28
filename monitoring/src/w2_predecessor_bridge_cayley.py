#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OMNI PREDECESSOR-EVIDENCE BRIDGE (cayley) -- codex 0544Z step 3,
REPAIRED for codex 1304Z bridge findings 1-3.

WHAT THIS IS FOR
----------------
The corrected-OMNI probe fires under a narrow grammar-only authority
that admits nothing scientifically -- but its day, 2026-01-01, is a
real expected key of the v4 authority. Refetching it would waste a
request against asylum's 636 ceiling and discard the grammar anchor;
relabelling the probe record as a production record would be
provenance fraud. This bridge is the ONLY path by which those bytes
become scientifically admissible.

WHAT THE FIRST VERSION GOT WRONG (owned)
----------------------------------------
I claimed transform-bypass was "structurally impossible" because the
function took no `artifact` argument. codex demonstrated otherwise:
the caller still supplied `transform_dispatcher` -- the function that
MAKES the artifact -- so an attacker-chosen dispatcher produced a
valid bridge record. Removing a parameter NAME is not removing a
CAPABILITY. Two further bypasses: the transcript was never consumed
(a minimal caller-synthesized envelope passed), and both authority
lineages were recorded but not authenticated (mutating the probe
authority's `authorization` block, or the v4 authority's outer
fields, passed).

THE REPAIRED CONTRACT
---------------------
* **No injection.** The production entry takes no dispatcher. It
  reopens the execution manifest, locates the pinned transform, and
  invokes the registered dispatcher itself. If the transform is NOT
  pinned, the bridge REFUSES -- admitting scientific data through an
  unpinned transform is exactly the gap this program keeps paying
  for. Injection lives only in `fixture_only_bridge_with_dispatcher`,
  which the production path never calls.
* **Authorities are reopened, not accepted.** Both are read as exact
  Git blobs from registered refs: the v4 authority from its manifest
  pin (then through the production authority verifier), and the probe
  authority from the pin the v4 authority itself carries. Blob-SHA
  equality authenticates the WHOLE file, so mutating any outer
  section refuses.
* **T is reopened and verified.** The create-once transcript is read
  by content address and run through the production
  `verify_transcript` against the derived contract and the reopened
  body. No caller-synthesized envelope stands in for T.

Opens no window-2 value; makes no network call.
"""
import hashlib
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

BRIDGE_SCHEMA = "f2g-w2-predecessor-bridge-v4"
PROBE_AUTHORITY_SCHEMA = "f2g-w2-omni-probe-authority-v4"
DISPATCHER_PATH = "monitoring/src/w2_acquisition_capture_grassmann.py"
EXEC_MANIFEST_PATH = ("docs/f2g_window2_execution/"
                      "execution_manifest.json")
PROBE_AUTHORITY_SECTIONS = {"schema", "authorization", "probe",
                            "discipline", "claim_ceiling",
                            "producer", "probe_sha256"}


class BridgeRefusal(ValueError):
    """Typed refusal; the code leads the message."""


def _refuse(detail):
    raise BridgeRefusal("PREDECESSOR_BRIDGE_REFUSED: " + str(detail))


def _digest(obj):
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def _git_blob(repo, commit, path):
    r = subprocess.run(["git", "-C", repo, "cat-file", "blob",
                        f"{commit}:{path}"], capture_output=True)
    if r.returncode != 0:
        _refuse(f"{path} is unreadable at {str(commit)[:12]}")
    return r.stdout


def _pin_for(manifest, path):
    for slot in (manifest.get("slots") or {}).values():
        if not isinstance(slot, dict) or slot.get("status") != "BOUND":
            continue
        for pin in slot.get("pins") or []:
            if isinstance(pin, dict) and pin.get("path") == path:
                return pin
    return None


def _pin_for_slot(manifest, slot_name, path):
    """Return an exact-path pin only from its declared BOUND authority.

    Membership somewhere in the manifest is insufficient: operation
    code pinned under a verification or later artifact-boundary slot
    answers to the wrong authority.
    """
    slot = (manifest.get("slots") or {}).get(slot_name)
    if not isinstance(slot, dict) or slot.get("status") != "BOUND":
        return None
    pins = [p for p in slot.get("pins") or []
            if isinstance(p, dict) and p.get("path") == path]
    return pins[0] if len(pins) == 1 else None


def _load_manifest(repo, manifest_commit):
    raw = _git_blob(repo, manifest_commit, EXEC_MANIFEST_PATH)
    return json.loads(raw.decode("utf-8"))


def _load_v4_authority(repo, manifest):
    """Reopen the v4 authority from its MANIFEST PIN and verify it
    whole through the production authority verifier."""
    import w2_accrual_instrument_cayley as AI
    pin = _pin_for(manifest, "docs/f2g_window2_execution/"
                             + AI.EXPECTED_KEYS_BASENAME)
    if not pin:
        _refuse("the v4 expected-contracts authority is not a BOUND "
                "pin of the execution manifest -- an unpinned "
                "authority cannot ground an admission")
    raw = _git_blob(repo, pin["commit"], pin["path"])
    if hashlib.sha256(raw).hexdigest() != pin.get("blob_sha256"):
        _refuse("the reopened v4 authority diverges from its pin")
    auth = json.loads(raw.decode("utf-8"))
    try:
        AI._validate_expected_keys_authority(repo, auth)
    except Exception as e:                                # noqa: BLE001
        _refuse("the v4 authority failed the PRODUCTION authority "
                f"verifier ({type(e).__name__}: {str(e)[:110]})")
    return auth


def _load_probe_authority(repo, v4_authority):
    """Reopen the probe authority from the pin the V4 AUTHORITY
    carries -- registered lineage, not caller assertion. Blob-SHA
    equality authenticates the whole file."""
    pin = v4_authority.get("registered_probe_authority")
    if not isinstance(pin, dict) or not all(
            pin.get(k) for k in ("path", "commit", "blob_sha256")):
        _refuse("the v4 authority registers no probe-authority pin -- "
                "the predecessor lineage must be registered, never "
                "asserted by a caller")
    raw = _git_blob(repo, pin["commit"], pin["path"])
    got = hashlib.sha256(raw).hexdigest()
    if got != pin["blob_sha256"]:
        _refuse(f"the reopened probe authority ({got[:12]}) diverges "
                f"from the registered pin ({pin['blob_sha256'][:12]})")
    pa = json.loads(raw.decode("utf-8"))
    if pa.get("schema") != PROBE_AUTHORITY_SCHEMA or \
            set(pa) != PROBE_AUTHORITY_SECTIONS:
        _refuse("the probe authority is not the closed registered "
                f"shape {sorted(PROBE_AUTHORITY_SECTIONS)}")
    if _digest(pa["probe"]) != pa.get("probe_sha256"):
        _refuse("the probe block diverges from its own digest")
    az = pa.get("authorization") or {}
    if az.get("requests_authorized") != 1:
        _refuse(f"the probe authority authorizes "
                f"{az.get('requests_authorized')!r} requests, not 1")
    if (pa.get("claim_ceiling") or {}).get(
            "scientific_admission") != "NONE -- grammar evidence only":
        _refuse("the probe authority's claim ceiling is not the "
                "registered grammar-only ceiling")
    return pa


def _registered_dispatcher(repo, manifest):
    """Locate the accrual-operation PIN and invoke THAT.

    Fail closed when the transform is absent or pinned under the wrong
    authority (codex 1304Z finding 1; codex 0413Z circular-pin repair).
    """
    pin = _pin_for_slot(manifest, "accrual_impl", DISPATCHER_PATH)
    if not pin:
        _refuse("TRANSFORM_NOT_PINNED: the registered lane-transform "
                f"dispatcher ({DISPATCHER_PATH}) is not a BOUND pin "
                "of the accrual_impl operation authority. Scientific "
                "admission through an unpinned or misplaced transform "
                "is refused.")
    raw = _git_blob(repo, pin["commit"], pin["path"])
    if hashlib.sha256(raw).hexdigest() != pin.get("blob_sha256"):
        _refuse("the on-pin dispatcher bytes diverge from the pin")
    import w2_acquisition_capture_grassmann as CAP
    with open(os.path.join(repo, DISPATCHER_PATH.replace("/", os.sep)),
              "rb") as f:
        on_disk = f.read()
    if hashlib.sha256(on_disk.replace(b"\r\n", b"\n")).hexdigest() \
            != hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest():
        _refuse("the imported dispatcher on disk diverges from its "
                "pinned bytes")
    return CAP.admission_transform


def _bridge_core(repo, *, manifest_commit, transcript_path,
                 body_path, dispatcher, dispatcher_origin):
    import w2_accrual_instrument_cayley as AI
    import w2_producer_grassmann as PROD

    manifest = _load_manifest(repo, manifest_commit)
    v4 = _load_v4_authority(repo, manifest)
    pa = _load_probe_authority(repo, v4)
    probe = pa["probe"]
    lane, carrier, day = (probe["lane"], probe["carrier"],
                          probe["utc_day"])

    days = ((v4["prestart_expected_keys"].get(lane) or {})
            .get(carrier) or [])
    if day not in days:
        _refuse(f"{lane}/{carrier}/{day} is not an expected key of "
                "the v4 authority")

    # ---- the create-once evidence, reopened -----------------------
    for p, what in ((transcript_path, "transcript"),
                    (body_path, "body")):
        if not os.path.exists(p):
            _refuse(f"the create-once {what} is absent at {p} -- the "
                    "bridge reopens evidence, it never reconstructs "
                    "it")
    with open(body_path, "rb") as f:
        raw_body = f.read()
    with open(transcript_path, encoding="utf-8") as f:
        transcript = json.load(f)

    contract = AI.authoritative_static_contract(v4, lane, carrier, day)
    v4_url = PROD.requested_url_of(contract["endpoint"],
                                   contract["request_params"])
    probe_url = PROD.requested_url_of(probe["endpoint"],
                                      probe["request_params"])
    if v4_url != probe_url:
        _refuse("the v4 contract does not reproduce the probe request "
                f"byte-for-byte:\n  v4    {v4_url}\n  probe {probe_url}")

    # ---- T verified through the PRODUCTION verifier ---------------
    try:
        PROD.verify_transcript(transcript, contract, raw_body)
    except Exception as e:                                # noqa: BLE001
        _refuse("the create-once transcript failed the PRODUCTION "
                f"transcript verifier ({type(e).__name__}: "
                f"{str(e)[:110]})")
    if transcript.get("utc_day") != day:
        _refuse(f"transcript day {transcript.get('utc_day')!r} "
                f"diverges from the probe/contract day {day!r}")
    if transcript.get("requested_url") != probe_url:
        _refuse("the transcript's requested URL diverges from the "
                "registered probe request")
    if transcript.get("http_status") != 200:
        _refuse(f"transcript status {transcript.get('http_status')} "
                "!= 200; a non-200 probe is a typed refusal, never a "
                "bridge input")

    # ---- the REGISTERED transform, invoked internally -------------
    try:
        artifact = dispatcher(lane, raw_body, contract)
    except Exception as e:                                # noqa: BLE001
        _refuse("the registered v4 transform refused the reopened "
                f"probe body ({type(e).__name__}: {str(e)[:110]})")

    record = {
        "schema": BRIDGE_SCHEMA,
        "lane": lane, "carrier": carrier, "utc_day": day,
        "admitted_through": "predecessor-evidence bridge",
        "never": "relabelling of the probe record",
        "dispatcher_origin": dispatcher_origin,
        "request_identity": {"url": v4_url,
                             "reproduced_byte_for_byte": True},
        "evidence": {
            "raw_body_sha256": hashlib.sha256(raw_body).hexdigest(),
            "raw_body_bytes": len(raw_body),
            "transcript_sha256": _digest(transcript)},
        "lineages": {
            "manifest_commit": str(manifest_commit),
            "probe_authority_pin":
                dict(v4["registered_probe_authority"]),
            "probe_block_sha256": pa["probe_sha256"],
            "v4_authority_keys_sha256":
                v4.get("prestart_expected_keys_sha256"),
            "v4_contract_sha256": _digest(contract)},
        "artifact_sha256": _digest(artifact),
        "artifact": artifact,
        "claim_ceiling": {"counts_against_owner_ceiling_once": True,
                          "remaining_omni_requests": 211,
                          "lambda_geo": "INCONCLUSIVE"}}
    record["bridge_sha256"] = _digest(
        {k: v for k, v in record.items() if k != "bridge_sha256"})
    return record


def verify_predecessor_bridge(repo, *, manifest_commit,
                              transcript_path, body_path):
    """PRODUCTION entry. Takes NO dispatcher: the registered one is
    located from the manifest pin and invoked internally."""
    manifest = _load_manifest(repo, manifest_commit)
    dispatcher = _registered_dispatcher(repo, manifest)
    return _bridge_core(repo, manifest_commit=manifest_commit,
                        transcript_path=transcript_path,
                        body_path=body_path, dispatcher=dispatcher,
                        dispatcher_origin="registered manifest pin")


def fixture_only_bridge_with_dispatcher(repo, *, manifest_commit,
                                        transcript_path, body_path,
                                        transform_dispatcher):
    """FIXTURE-ONLY. Never reachable from the production entry: it is
    a separate function, and `verify_predecessor_bridge` resolves its
    dispatcher from the manifest pin without consulting any caller
    value. Present so KATs can exercise the core before the
    dispatcher is pinned; a caller reaching for injection must name
    this function explicitly, which is auditable."""
    return _bridge_core(repo, manifest_commit=manifest_commit,
                        transcript_path=transcript_path,
                        body_path=body_path,
                        dispatcher=transform_dispatcher,
                        dispatcher_origin="FIXTURE INJECTION -- not "
                                          "a production admission")
