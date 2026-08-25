#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OMNI PREDECESSOR-EVIDENCE BRIDGE (cayley) -- codex 0544Z step 3.

THE PROBLEM IT SOLVES
---------------------
The corrected-OMNI probe (vars 17/18/21) fires under a narrow
grammar-only probe authority that admits NOTHING scientifically. But
its day, 2026-01-01, is also a real expected key of the v4 authority.
So one of the 212 OMNI days already has its bytes on disk before the
production run begins.

Refetching it would waste a request against asylum's 636 ceiling and
discard the exact bytes that anchored the grammar. Relabelling the
probe record as a production record would be provenance fraud: the
probe was authorized under a different authority, for a different
purpose, with an explicit no-admission ceiling.

THE BRIDGE
----------
`verify_predecessor_bridge` is the ONLY path by which probe bytes
become scientifically admissible. It requires, all of them, closed:

  1. **byte-for-byte request identity** -- the v4 contract for that
     exact day, rendered through the PRODUCTION canonical builder,
     must equal the probe authority's registered request AND the
     envelope's actually-requested URL. One changed variable refuses.
  2. **day identity** -- probe day == contract day == envelope day.
     Copying genuine bytes under another day refuses.
  3. **evidence reopening** -- the raw body is reopened and its digest
     recomputed against the envelope; a substituted body refuses.
  4. **v4 transform rerun** -- the artifact is recomputed from the
     reopened bytes through the REGISTERED v4 dispatcher. A
     caller-supplied artifact is never accepted; bypassing the
     transform refuses.
  5. **dual lineage** -- BOTH authority lineages are bound: the probe
     authority (under which the bytes were lawfully obtained) and the
     v4 authority (under which they become admissible). Omitting
     either refuses.

Only when all five hold does the bridge emit an admission record --
and that record says the body was admitted THROUGH the bridge, never
that the probe admitted it.

Opens no window-2 value; makes no network call.
"""
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

BRIDGE_SCHEMA = "f2g-w2-predecessor-bridge-v4"
PROBE_AUTHORITY_SCHEMA = "f2g-w2-omni-probe-authority-v4"


class BridgeRefusal(ValueError):
    """Typed refusal; the code leads the message."""


def _refuse(detail):
    raise BridgeRefusal("PREDECESSOR_BRIDGE_REFUSED: " + str(detail))


def _digest(obj):
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def verify_predecessor_bridge(*, probe_authority, v4_authority,
                              envelope, raw_body,
                              transform_dispatcher=None):
    """THE bridge. Returns a closed admission record, or refuses typed.

    probe_authority -- the committed omni_probe_authority_v4 object
    v4_authority    -- the committed v4 expected-contracts authority
    envelope        -- the probe's create-once envelope
    raw_body        -- the probe's create-once raw bytes
    """
    import w2_accrual_instrument_cayley as AI
    import w2_producer_grassmann as PROD
    if transform_dispatcher is None:
        import w2_acquisition_capture_grassmann as CAP
        transform_dispatcher = CAP.admission_transform

    # ---- lineage 1: the probe authority ---------------------------
    if not isinstance(probe_authority, dict) or \
            probe_authority.get("schema") != PROBE_AUTHORITY_SCHEMA:
        _refuse("probe authority is not the registered probe-authority "
                "artifact")
    probe = probe_authority.get("probe")
    if not isinstance(probe, dict):
        _refuse("probe authority carries no closed probe block")
    if _digest(probe) != probe_authority.get("probe_sha256"):
        _refuse("probe authority block diverges from its own digest")
    lane = probe["lane"]
    carrier = probe["carrier"]
    day = probe["utc_day"]

    # ---- lineage 2: the v4 authority ------------------------------
    if not isinstance(v4_authority, dict) or \
            not v4_authority.get("prestart_expected_keys"):
        _refuse("v4 authority is not the registered expected-contracts "
                "authority")
    keys = v4_authority["prestart_expected_keys"]
    days = ((keys.get(lane) or {}).get(carrier) or [])
    if day not in days:
        _refuse(f"{lane}/{carrier}/{day} is not an expected key of "
                "the v4 authority -- the bridge cannot admit a day "
                "the authority does not expect")

    # ---- (1) byte-for-byte request identity -----------------------
    contract = AI.authoritative_static_contract(v4_authority, lane,
                                                carrier, day)
    v4_url = PROD.requested_url_of(contract["endpoint"],
                                   contract["request_params"])
    probe_url = PROD.requested_url_of(probe["endpoint"],
                                      probe["request_params"])
    if v4_url != probe_url:
        _refuse("the v4 contract does not reproduce the probe request "
                f"byte-for-byte:\n  v4    {v4_url}\n  probe {probe_url}")
    requested = envelope.get("requested_url")
    if requested != probe_url:
        _refuse("the envelope's requested URL diverges from the "
                f"registered probe request:\n  env   {requested}\n"
                f"  probe {probe_url}")

    # ---- (2) day identity ----------------------------------------
    env_day = envelope.get("probe_day_utc") or envelope.get("utc_day")
    if env_day != day:
        _refuse(f"envelope day {env_day!r} diverges from the probe/"
                f"contract day {day!r} -- genuine bytes carried under "
                "another day are refused")
    if contract.get("utc_day") != day:
        _refuse("the derived contract day diverges from the probe day")

    # ---- (3) evidence reopening ----------------------------------
    if envelope.get("http_status") != 200:
        _refuse(f"probe envelope status {envelope.get('http_status')} "
                "!= 200; a non-200 probe is a typed refusal, never a "
                "bridge input")
    env_sha = (envelope.get("raw_body_sha256")
               or envelope.get("body_sha256"))
    got = hashlib.sha256(raw_body).hexdigest()
    if not env_sha or got != env_sha:
        _refuse("reopened body digest diverges from the envelope "
                f"({got[:12]} != {str(env_sha)[:12]})")
    if envelope.get("raw_body_bytes") is not None and \
            len(raw_body) != envelope["raw_body_bytes"]:
        _refuse("reopened body size diverges from the envelope")

    # ---- (4) v4 transform rerun (never a supplied artifact) -------
    try:
        artifact = transform_dispatcher(lane, raw_body, contract)
    except Exception as e:                                # noqa: BLE001
        _refuse("the registered v4 transform refused the reopened "
                f"probe body ({type(e).__name__}: {str(e)[:100]}) -- "
                "the bytes cannot be admitted")

    # ---- (5) the closed admission record --------------------------
    record = {
        "schema": BRIDGE_SCHEMA,
        "lane": lane, "carrier": carrier, "utc_day": day,
        "admitted_through": "predecessor-evidence bridge",
        "never": "relabelling of the probe record",
        "request_identity": {"url": v4_url,
                             "reproduced_byte_for_byte": True},
        "evidence": {"raw_body_sha256": got,
                     "raw_body_bytes": len(raw_body),
                     "envelope_sha256": _digest(envelope)},
        "lineages": {
            "probe_authority_sha256": _digest(probe_authority),
            "probe_block_sha256": probe_authority["probe_sha256"],
            "v4_authority_keys_sha256":
                v4_authority.get("prestart_expected_keys_sha256"),
            "v4_contract_sha256": _digest(contract)},
        "artifact_sha256": _digest(artifact),
        "artifact": artifact,
        "claim_ceiling": {
            "counts_against_owner_ceiling_once": True,
            "remaining_omni_requests": 211,
            "lambda_geo": "INCONCLUSIVE"}}
    record["bridge_sha256"] = _digest(
        {k: v for k, v in record.items() if k != "bridge_sha256"})
    return record


def _selftest():
    """codex 0544Z abuse-class doctors. The probe has NOT fired, so the
    fixture stands in structurally: the REAL committed OMNIWeb body
    supplies authentic bytes/shape, and the envelope is synthesized to
    carry the corrected URL. It exercises the MECHANISM only, lives
    in memory, and is never written to the evidence tree."""
    import copy
    import subprocess
    import w2_expected_contracts_gen_cayley as GEN

    repo = os.path.abspath(os.path.join(_HERE, "..", ".."))
    probe_auth = json.load(open(os.path.join(
        repo, "docs", "f2g_window2_execution",
        "omni_probe_authority_v4.json"), encoding="utf-8"))
    probe = probe_auth["probe"]
    day = probe["utc_day"]

    # the v4 authority AS IT WILL BE once the probe closes the OMNI
    # lock: same build, OMNI template filled with the registered
    # corrected params
    auth = GEN.build(repo)
    omni = auth["static_layer"]["MAG_WEATHER_FEED"]["carriers"]["omni"]
    omni["endpoint"] = probe["endpoint"]
    omni["request_params"] = dict(probe["request_params"])
    omni["operation_params"] = {"carrier": "omni", "day": "{day}"}
    omni["static_contract_template"] = {
        "source": {"kind": "omniweb-highres-cgi",
                   "ref": probe["endpoint"]},
        "endpoint": probe["endpoint"],
        "request_params": dict(probe["request_params"]),
        "operation_params": {"carrier": "omni", "day": "{day}"}}
    omni.pop("fill_status", None)
    # the day token must survive: re-template the compact dates
    for k in ("start_date", "end_date"):
        omni["static_contract_template"]["request_params"][k] = \
            "{day_compact}"

    raw = subprocess.run(
        ["git", "-C", repo, "cat-file", "blob",
         "HEAD:docs/f2g_window2_execution/probe_evidence/"
         "mf4_feed_omni.body"], capture_output=True).stdout
    assert raw, "fixture body unreadable"
    # The real body is 2025 DOY 319; the probe day is 2026-01-01
    # (DOY 001). Re-date ONLY the year/DOY columns, leaving the row
    # format, token count, value columns and fill sentinels exactly as
    # the provider wrote them -- the fixture stays derived from the
    # authoritative body rather than hand-built.
    _txt = raw.decode("utf-8", "replace")
    _out = []
    for _l in _txt.splitlines():
        if _l[:4].isdigit() and _l[:8] == "2025 319":
            _out.append("2026   1" + _l[8:])
        else:
            _out.append(_l)
    _nl = chr(10)
    raw = (_nl.join(_out) + _nl).encode()
    import w2_producer_grassmann as PROD
    url = PROD.requested_url_of(probe["endpoint"],
                                probe["request_params"])
    env = {"schema": "f2g-w2-probe-envelope-v1",
           "requested_url": url, "effective_url": url,
           "http_status": 200, "probe_day_utc": day,
           "raw_body_sha256": hashlib.sha256(raw).hexdigest(),
           "raw_body_bytes": len(raw)}

    # SEAM (raised to grassmann): the registered dispatcher still
    # knows only the v3 lane name MF4_FEED; the successor rename to
    # MAG_WEATHER_FEED / MF4_MONITOR_FEED has not reached it. This
    # clearly-labelled shim lets the BRIDGE mechanism be exercised
    # now; it is NOT a production path and must disappear when the
    # dispatcher registers the v4 lane names.
    import w2_acquisition_capture_grassmann as CAP

    def lane_shim(lane, body, contract):
        return CAP.admission_transform(
            "MF4_FEED", body, dict(contract, lane="MF4_FEED"))

    def run(**over):
        kw = {"probe_authority": probe_auth, "v4_authority": auth,
              "envelope": env, "raw_body": raw,
              "transform_dispatcher": lane_shim}
        kw.update(over)
        return verify_predecessor_bridge(**kw)

    def must_refuse(label, needle, **over):
        try:
            run(**over)
        except BridgeRefusal as e:
            assert needle in str(e), (label, str(e)[:140])
            return
        raise AssertionError(f"doctor must refuse: {label}")

    rec = run()                                    # POSITIVE
    assert rec["schema"] == BRIDGE_SCHEMA
    assert rec["lineages"]["probe_authority_sha256"]
    assert rec["lineages"]["v4_authority_keys_sha256"]
    assert rec["utc_day"] == day
    print(f"  POSITIVE  bridge admits {rec['lane']}/"
          f"{rec['carrier']}/{rec['utc_day']} through BOTH lineages")

    # (2) genuine bytes carried under ANOTHER day
    must_refuse("day swap", "diverges from the probe/contract day",
                envelope=dict(env, probe_day_utc="2026-01-02"))
    # (1) ONE changed query variable
    bad_auth = copy.deepcopy(auth)
    bad_auth["static_layer"]["MAG_WEATHER_FEED"]["carriers"]["omni"][
        "static_contract_template"]["request_params"]["vars"] = \
        ["17", "18", "25"]
    must_refuse("one variable changed",
                "does not reproduce the probe request",
                v4_authority=bad_auth)
    # (3) substituted body
    must_refuse("body substitution", "body digest diverges",
                raw_body=raw + b"x")
    # (3) non-200 probe
    must_refuse("non-200 probe", "!= 200",
                envelope=dict(env, http_status=503))
    # (5) lineage omitted: probe side
    must_refuse("probe lineage omitted",
                "not the registered probe-authority",
                probe_authority={"schema": "something-else"})
    # (5) lineage omitted: v4 side (key absent)
    thin = copy.deepcopy(auth)
    thin["prestart_expected_keys"]["MAG_WEATHER_FEED"]["omni"] = [
        d for d in thin["prestart_expected_keys"][
            "MAG_WEATHER_FEED"]["omni"] if d != day]
    must_refuse("v4 lineage omitted", "is not an expected key",
                v4_authority=thin)
    # (5) probe block tampered under its own digest
    tampered = copy.deepcopy(probe_auth)
    tampered["probe"]["request_params"]["vars"] = ["17", "18", "25"]
    must_refuse("probe block tampered", "diverges from its own digest",
                probe_authority=tampered)
    # (4) the transform is genuinely consulted -- a refusing
    # dispatcher must sink the bridge (no supplied artifact exists)
    def refusing(lane, body, contract):
        raise ValueError("transform says no")
    must_refuse("transform consulted", "registered v4 transform "
                "refused", transform_dispatcher=refusing)
    # (4) structural: the bridge accepts NO caller artifact
    import inspect
    params = set(inspect.signature(
        verify_predecessor_bridge).parameters)
    assert "artifact" not in params, \
        "the bridge must never accept a caller-supplied artifact"
    print("  DOCTORS   day-swap, one-variable, body-substitution, "
          "non-200, both-lineage-omissions, probe-tamper, "
          "transform-consulted, no-artifact-parameter: ALL REFUSE")
    print("w2_predecessor_bridge selftest: ALL PASS (mechanism only; "
          "the real probe has not fired; nothing admitted)")


if __name__ == "__main__":
    _selftest()
