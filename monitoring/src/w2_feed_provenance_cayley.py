#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FEED PROVENANCE + RECOMPUTE-AT-BIND (cayley) -- codex 0530Z ruling.

THE GAP THIS CLOSES
-------------------
I raised that nothing bound a calibration envelope's series to the
boundary-admitted daily inputs, and proposed a canonical
(lane, carrier, day -> raw_body_sha256) input-map digest. codex
confirmed the gap and corrected the fix: an input map is **necessary
but not sufficient**, because an envelope can name exactly the right
inputs while carrying arbitrary, self-consistent series. Naming your
sources is not deriving from them.

So the rule here is stronger: at bind time the verifier **reopens the
admitted inputs, reruns the pinned join, and compares the ENTIRE
recomputed payload** to the staged envelope. A matching
`source.sha256` without that rebuild refuses.

THE THREE THINGS A FEED MUST BIND
---------------------------------
1. **A typed input set** -- every admitted input the payload consumes,
   keyed lane/carrier/day, each binding raw-body digest, admitted
   artifact digest, and transcript/static-contract linkage. For MAG
   that is observatory minutes PLUS every weather carrier; the
   multi-carrier join is never collapsed to one per-observatory map.
2. **A pinned operation** -- the join/resample identity and its
   parameters (time grid, fill/mask rules, units, the corrected Newell
   variables and formula, cutoff). Changing the operation changes the
   provenance.
3. **A recomputed output** -- the payload, rebuilt here, byte-equal to
   what was staged.

`source.sha256` may carry the provenance-manifest digest for schema
compatibility and `source.ref` may name the admitted boundary, but the
verifier RESOLVES and RECOMPUTES rather than trusting either field.

The join operation is injected: grassmann pins the production one.
This module owns the provenance contract and the bind-time gate.

Opens no window-2 value; makes no network call.
"""
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

PROVENANCE_SCHEMA = "f2g-w2-feed-provenance-v4"
INPUT_FIELDS = {"lane", "carrier", "utc_day", "raw_body_sha256",
                "artifact_sha256", "transcript_sha256",
                "static_contract_sha256"}
OPERATION_FIELDS = {"identity", "params"}


class ProvenanceRefusal(ValueError):
    """Typed refusal; the code leads the message."""


def _refuse(detail):
    raise ProvenanceRefusal("FEED_PROVENANCE_REFUSED: " + str(detail))


def _digest(obj):
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def _key(e):
    return f"{e['lane']}/{e['carrier']}/{e['utc_day']}"


def provenance_manifest_digest(manifest):
    """The canonical digest `source.sha256` may carry. Computing it is
    NOT admission -- the verifier still recomputes the payload."""
    if not isinstance(manifest, dict):
        _refuse("provenance manifest is not an object")
    return _digest({k: v for k, v in manifest.items()
                    if k != "manifest_sha256"})


def validate_provenance_manifest(manifest):
    """Closure + typing only. Deliberately does NOT admit anything."""
    if not isinstance(manifest, dict) or \
            manifest.get("schema") != PROVENANCE_SCHEMA:
        _refuse("provenance manifest is not the registered schema")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        _refuse("provenance manifest carries no typed input set")
    seen = set()
    for e in inputs:
        if not isinstance(e, dict) or set(e) != INPUT_FIELDS:
            _refuse("input entry is not the closed typed shape "
                    f"{sorted(INPUT_FIELDS)}: got "
                    f"{sorted(e) if isinstance(e, dict) else type(e)}")
        k = _key(e)
        if k in seen:
            _refuse(f"duplicate input entry {k}")
        seen.add(k)
    op = manifest.get("operation")
    if not isinstance(op, dict) or set(op) != OPERATION_FIELDS:
        _refuse("operation block is not the closed shape "
                f"{sorted(OPERATION_FIELDS)}")
    if not op.get("identity"):
        _refuse("operation carries no pinned identity")
    if manifest.get("manifest_sha256") != \
            provenance_manifest_digest(manifest):
        _refuse("provenance manifest diverges from its own digest")
    return sorted(seen)


def verify_feed_provenance(*, envelope, manifest, admitted_reader,
                           join_operation, expected_operation_identity
                           =None):
    """THE bind-time gate.

    envelope        -- the staged feed envelope (payload + source)
    manifest        -- the provenance manifest (typed inputs + op)
    admitted_reader -- (lane, carrier, day) -> {"raw_body": bytes,
                       "artifact": obj, "transcript_sha256": str,
                       "static_contract_sha256": str}; it must serve
                       ONLY boundary-ADMITTED inputs
    join_operation  -- the PINNED deterministic join:
                       (inputs, params) -> payload
    """
    keys = validate_provenance_manifest(manifest)
    op = manifest["operation"]
    if expected_operation_identity is not None and \
            op["identity"] != expected_operation_identity:
        _refuse(f"operation identity {op['identity']!r} is not the "
                f"registered {expected_operation_identity!r}")

    # ---- reopen EVERY declared input from the admitted set --------
    resolved = []
    for e in manifest["inputs"]:
        try:
            got = admitted_reader(e["lane"], e["carrier"],
                                  e["utc_day"])
        except Exception:                                 # noqa: BLE001
            _refuse(f"{_key(e)} is not resolvable in the ADMITTED "
                    "boundary set -- a feed may not consume an input "
                    "the boundary did not admit")
        raw = got["raw_body"]
        if hashlib.sha256(raw).hexdigest() != e["raw_body_sha256"]:
            _refuse(f"{_key(e)} raw body diverges from the declared "
                    "digest")
        if _digest(got["artifact"]) != e["artifact_sha256"]:
            _refuse(f"{_key(e)} admitted artifact diverges from the "
                    "declared digest")
        for f in ("transcript_sha256", "static_contract_sha256"):
            if got.get(f) != e[f]:
                _refuse(f"{_key(e)} {f} linkage diverges")
        resolved.append({"lane": e["lane"], "carrier": e["carrier"],
                         "utc_day": e["utc_day"],
                         "raw_body": raw, "artifact": got["artifact"]})

    # ---- rerun the PINNED join and compare the ENTIRE payload ------
    try:
        recomputed = join_operation(resolved, op["params"])
    except Exception as e:                                # noqa: BLE001
        _refuse("the pinned join operation refused the admitted "
                f"inputs ({type(e).__name__}: {str(e)[:100]})")
    staged = {k: v for k, v in envelope.items() if k != "source"}
    if _digest(recomputed) != _digest(staged):
        _refuse("the RECOMPUTED payload diverges from the staged "
                "envelope -- naming the right inputs is not deriving "
                "from them (codex 0530Z). A matching source.sha256 "
                "does not admit.")

    # ---- source fields are RESOLVED, never trusted -----------------
    src = envelope.get("source") or {}
    declared = src.get("sha256")
    actual = provenance_manifest_digest(manifest)
    if declared != actual:
        _refuse(f"envelope source.sha256 {str(declared)[:12]} is not "
                f"the provenance-manifest digest {actual[:12]}")
    return {"schema": "f2g-w2-feed-provenance-admission-v4",
            "keys_admitted": keys,
            "input_count": len(keys),
            "operation_identity": op["identity"],
            "provenance_manifest_sha256": actual,
            "payload_sha256": _digest(staged),
            "admitted_through": "recompute-at-bind",
            "never": "content-authentication alone",
            "lambda_geo": "INCONCLUSIVE"}


def _selftest():
    """codex 0530Z lock doctors. The reference join below is a
    deterministic stand-in so the MECHANISM is exercised now;
    grassmann pins the production MAG weather/Newell join, and this
    gate then runs against that instead -- nothing here changes."""
    import copy

    # ---- a multi-carrier admitted set: observatory + weather ------
    ADMITTED = {}
    for day in ("2026-01-01", "2026-01-02"):
        for lane, carrier in (("MAG_FEED", "izn"),
                              ("MAG_WEATHER_FEED", "sym_h")):
            raw = f"body-{lane}-{carrier}-{day}".encode()
            ADMITTED[(lane, carrier, day)] = {
                "raw_body": raw,
                "artifact": {"lane": lane, "carrier": carrier,
                             "utc_day": day, "n": len(raw)},
                "transcript_sha256": _digest(["T", lane, carrier,
                                              day]),
                "static_contract_sha256": _digest(["S", lane,
                                                   carrier, day])}

    def reader(lane, carrier, day):
        return ADMITTED[(lane, carrier, day)]

    OP_ID = "w2-mag-weather-join-reference-v0"

    def join(inputs, params):
        """Deterministic reference join: an ordered time grid from the
        observatory inputs plus an aligned weather series -- multi-
        carrier, so a swapped weather body changes the payload."""
        obs = sorted((i for i in inputs if i["lane"] == "MAG_FEED"),
                     key=lambda i: i["utc_day"])
        wx = sorted((i for i in inputs
                     if i["lane"] == "MAG_WEATHER_FEED"),
                    key=lambda i: i["utc_day"])
        if not obs or not wx:
            raise ValueError("join needs observatory AND weather")
        return {"times": [i["utc_day"] for i in obs],
                "components": {"X": [i["artifact"]["n"] * params[
                    "scale"] for i in obs]},
                "weather": {"sym_h": [len(i["raw_body"]) for i in wx]},
                "units": params["units"]}

    params = {"scale": 2, "units": "nT", "grid": "minute",
              "cutoff": "2026-07-31"}
    inputs = []
    for (lane, carrier, day), v in sorted(ADMITTED.items()):
        inputs.append({
            "lane": lane, "carrier": carrier, "utc_day": day,
            "raw_body_sha256": hashlib.sha256(
                v["raw_body"]).hexdigest(),
            "artifact_sha256": _digest(v["artifact"]),
            "transcript_sha256": v["transcript_sha256"],
            "static_contract_sha256": v["static_contract_sha256"]})
    man = {"schema": PROVENANCE_SCHEMA, "inputs": inputs,
           "operation": {"identity": OP_ID, "params": params}}
    man["manifest_sha256"] = provenance_manifest_digest(man)

    resolved = [dict(lane=i["lane"], carrier=i["carrier"],
                     utc_day=i["utc_day"],
                     raw_body=ADMITTED[(i["lane"], i["carrier"],
                                        i["utc_day"])]["raw_body"],
                     artifact=ADMITTED[(i["lane"], i["carrier"],
                                        i["utc_day"])]["artifact"])
                for i in inputs]
    payload = join(resolved, params)
    env = dict(payload, source={"kind": "staged-envelope",
                                "ref": "admitted-boundary",
                                "sha256": man["manifest_sha256"]})

    def run(**over):
        kw = {"envelope": env, "manifest": man,
              "admitted_reader": reader, "join_operation": join,
              "expected_operation_identity": OP_ID}
        kw.update(over)
        return verify_feed_provenance(**kw)

    def must_refuse(label, needle, **over):
        try:
            run(**over)
        except ProvenanceRefusal as e:
            assert needle in str(e), (label, str(e)[:150])
            return
        raise AssertionError(f"doctor must refuse: {label}")

    ok = run()                                          # POSITIVE
    assert ok["input_count"] == 4
    assert ok["admitted_through"] == "recompute-at-bind"
    print(f"  POSITIVE  {ok['input_count']} admitted inputs "
          "recomputed and byte-equal")

    # (1) correct map + ALTERED SERIES -- the headline case
    bad = copy.deepcopy(env)
    bad["components"]["X"][0] += 1
    must_refuse("correct map + altered series",
                "RECOMPUTED payload diverges", envelope=bad)

    # (2) correct observatory bodies + SWAPPED WEATHER BODY
    swapped = {k: dict(v) for k, v in ADMITTED.items()}
    swapped[("MAG_WEATHER_FEED", "sym_h", "2026-01-01")][
        "raw_body"] = b"a-different-weather-body"
    must_refuse("swapped weather body", "raw body diverges",
                admitted_reader=lambda l, c, d: swapped[(l, c, d)])

    # (3) OMITTED day/carrier
    thin = copy.deepcopy(man)
    thin["inputs"] = [i for i in thin["inputs"]
                      if not (i["carrier"] == "sym_h"
                              and i["utc_day"] == "2026-01-02")]
    thin["manifest_sha256"] = provenance_manifest_digest(thin)
    must_refuse("omitted day/carrier", "RECOMPUTED payload diverges",
                manifest=thin, envelope=dict(
                    env, source=dict(env["source"],
                                     sha256=thin["manifest_sha256"])))

    # (4) REORDERED time grid
    reord = copy.deepcopy(env)
    reord["times"] = list(reversed(reord["times"]))
    must_refuse("reordered time grid", "RECOMPUTED payload diverges",
                envelope=reord)

    # (5) ALTERED OPERATION PIN -- params, and identity
    alt = copy.deepcopy(man)
    alt["operation"]["params"]["scale"] = 3
    alt["manifest_sha256"] = provenance_manifest_digest(alt)
    must_refuse("altered operation params",
                "RECOMPUTED payload diverges", manifest=alt,
                envelope=dict(env, source=dict(
                    env["source"], sha256=alt["manifest_sha256"])))
    alt2 = copy.deepcopy(man)
    alt2["operation"]["identity"] = "some-other-join"
    alt2["manifest_sha256"] = provenance_manifest_digest(alt2)
    must_refuse("altered operation identity",
                "is not the registered", manifest=alt2)

    # (6) STAGED SOURCE DIGEST COPIED from a genuine-but-other
    # envelope -- content-auth alone must not admit
    must_refuse("copied source digest",
                "is not the provenance-manifest digest",
                envelope=dict(env, source=dict(
                    env["source"], sha256=alt["manifest_sha256"])))

    # structural: an input outside the ADMITTED set is unresolvable
    must_refuse("input outside the admitted set",
                "not resolvable in the ADMITTED boundary set",
                admitted_reader=lambda l, c, d: (_ for _ in ()).throw(
                    KeyError((l, c, d))))
    # structural: a non-closed input entry refuses on typing
    loose = copy.deepcopy(man)
    loose["inputs"][0]["extra"] = 1
    loose["manifest_sha256"] = provenance_manifest_digest(loose)
    must_refuse("non-closed input entry", "closed typed shape",
                manifest=loose)
    print("  DOCTORS   altered-series, swapped-weather, omitted-key, "
          "reordered-grid, altered-op-params, altered-op-identity, "
          "copied-source-digest, unadmitted-input, non-closed-entry: "
          "ALL REFUSE")
    print("w2_feed_provenance selftest: ALL PASS (reference join; "
          "grassmann pins the production Newell join; nothing "
          "admitted)")


if __name__ == "__main__":
    _selftest()
