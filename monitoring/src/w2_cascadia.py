#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 CASCADIA registry engine (cayley) -- per-day epoch/location
resolution over the PINNED station receipt, per the frozen capsule
docs/f2g_window2_freeze/cascadia_carrier_capsule.md (design freeze
CLOSED @ 12161f6/5fba544) and grassmann's bar seam pin
(test_f2g_window2_redkats_grassmann.py @ 8a78d5f). Seam name FIXED as
`w2_cascadia`.

Frozen rule (capsule, codex freeze-review fix 3): identity = NET.STA;
channel epochs are HALF-OPEN [start, end); for each required time FIRST
restrict to epochs ACTIVE at that time, THEN apply location precedence
blank -> 00 -> lowest lexicographic. A literally-blank-first rule would
select dead epochs (UW.TOUT..HHZ ends 2026-07-16T00:00 exactly as
UW.TOUT.00.HHZ opens).

Interpretation pin (disclosed): the required time for a UTC day is the
DAY-START instant 00:00:00Z. With half-open epochs this resolves the
TOUT transition exactly (07-15 -> blank epoch, 07-16 -> 00 epoch) and
makes mid-day transitions resolve to the day-start state; an epoch
opening mid-day first appears the following day. Deterministic; no
evaluation-window value is consulted.

The receipt body is read FROM GIT OBJECTS at the frozen design-manifest
commit and sha-verified BEFORE use (content-auth before use); the
working tree is never consulted. This module opens no window-2 value.
"""
import hashlib
import json
import os
import subprocess
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))

MANIFEST_COMMIT = "5fba5446cd96722fb86021df5065a46b8a1a78f5"
RECEIPT_PATH = ("docs/f2g_window2_freeze/receipts/"
                "cascadia_UW_CC_CN_HHZ.txt")
ENVELOPE_PATH = ("docs/f2g_window2_freeze/receipts/"
                 "cascadia_UW_CC_CN_HHZ.envelope.json")
RECEIPT_SHA = ("d4256792bf85edf855a4dbaf7841982824a020cd5e075c103d8322"
               "48c513a847")
CHANNEL = "HHZ"


class ReceiptIntegrityError(ValueError):
    """Typed RECEIPT_SHA_MISMATCH / RECEIPT_UNREADABLE."""


class RegistryInputInvalid(ValueError):
    """Typed BAD_DAY_FORMAT."""


class EpochOverlapError(ValueError):
    """Typed EPOCH_OVERLAP_SAME_LOCATION: two epochs of one NET.STA
    with the SAME location code active at the required instant --
    provider-metadata defect; refusing beats a silent choice."""


def _git_blob(repo, ref):
    p = subprocess.run(["git", "-C", repo, "cat-file", "blob", ref],
                       capture_output=True)
    if p.returncode != 0:
        raise ReceiptIntegrityError(f"RECEIPT_UNREADABLE: {ref}")
    return p.stdout


def _parse_time(s):
    s = s.strip()
    if not s:
        return None
    if "." in s:
        s = s.split(".")[0]
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")


def load_receipt(repo=None):
    """Pinned receipt body -> list of epoch dicts. Bytes verified
    against the capsule sha BEFORE parsing."""
    repo = repo or _DEFAULT_REPO
    raw = _git_blob(repo, f"{MANIFEST_COMMIT}:{RECEIPT_PATH}")
    got = hashlib.sha256(raw).hexdigest()
    if got != RECEIPT_SHA:
        raise ReceiptIntegrityError(
            f"RECEIPT_SHA_MISMATCH: got={got[:12]} "
            f"pinned={RECEIPT_SHA[:12]}")
    epochs = []
    for line in raw.decode("utf-8", "replace").splitlines():
        if not line or line.startswith("#"):
            continue
        f = [x.strip() for x in line.split("|")]
        if len(f) < 16 or f[3] != CHANNEL:
            continue
        epochs.append({
            "network": f[0], "station": f[1], "location": f[2],
            "channel": f[3], "latitude": float(f[4]),
            "longitude": float(f[5]),
            "epoch_start": _parse_time(f[15]),
            "epoch_end": _parse_time(f[16]) if len(f) > 16 else None})
    return epochs


def _active(e, t):
    # half-open [start, end)
    if e["epoch_start"] is not None and t < e["epoch_start"]:
        return False
    return e["epoch_end"] is None or t < e["epoch_end"]


def _loc_key(loc):
    if loc == "":
        return (0, "")
    if loc == "00":
        return (1, "00")
    return (2, loc)


def registry_for_day(utc_day, repo=None, epochs=None):
    """Per-day epoch/location-resolved NET.STA rows (the bar seam).
    utc_day = 'YYYY-MM-DD'; required instant = day start 00:00:00Z.
    `epochs` may inject fixture epochs (bar KATs); default = the pinned
    receipt."""
    try:
        t = datetime.strptime(utc_day, "%Y-%m-%d")
    except (TypeError, ValueError):
        raise RegistryInputInvalid(f"BAD_DAY_FORMAT: {utc_day!r}")
    if epochs is None:
        epochs = load_receipt(repo)

    by_id = {}
    for e in epochs:
        by_id.setdefault(f"{e['network']}.{e['station']}",
                         []).append(e)
    rows = []
    for ident in sorted(by_id):
        active = [e for e in by_id[ident] if _active(e, t)]
        if not active:
            continue
        active.sort(key=lambda e: _loc_key(e["location"]))
        chosen = active[0]
        same_loc = [e for e in active
                    if e["location"] == chosen["location"]]
        if len(same_loc) > 1:
            raise EpochOverlapError(
                f"EPOCH_OVERLAP_SAME_LOCATION: {ident} "
                f"loc={chosen['location']!r} at {utc_day}")
        rows.append({"id": ident, "network": chosen["network"],
                     "station": chosen["station"],
                     "location": chosen["location"],
                     "channel": chosen["channel"],
                     "latitude": chosen["latitude"],
                     "longitude": chosen["longitude"],
                     "epoch_start": chosen["epoch_start"],
                     "epoch_end": chosen["epoch_end"]})
    return rows


def receipt_summary(repo=None):
    """Recomputed body facts for the envelope-vs-body equality check
    (bar W-CAS-b): non-comment row count, unique NET.STA, per-network
    counts, body sha."""
    repo = repo or _DEFAULT_REPO
    raw = _git_blob(repo, f"{MANIFEST_COMMIT}:{RECEIPT_PATH}")
    rows = [l for l in raw.decode("utf-8", "replace").splitlines()
            if l and not l.startswith("#")]
    idents = set()
    by_net = {}
    for l in rows:
        f = [x.strip() for x in l.split("|")]
        idents.add(f"{f[0]}.{f[1]}")
    for ident in idents:
        by_net[ident.split(".")[0]] = \
            by_net.get(ident.split(".")[0], 0) + 1
    return {"rows": len(rows), "unique_net_sta": len(idents),
            "by_network": by_net,
            "body_sha256": hashlib.sha256(raw).hexdigest()}


def _selftest():
    # envelope-vs-body: recomputed facts equal the pinned envelope's
    s = receipt_summary()
    assert s["body_sha256"] == RECEIPT_SHA
    assert (s["rows"], s["unique_net_sta"]) == (203, 198)
    assert s["by_network"] == {"UW": 118, "CC": 43, "CN": 37}
    env = json.loads(_git_blob(_DEFAULT_REPO,
                               f"{MANIFEST_COMMIT}:{ENVELOPE_PATH}")
                     .decode("utf-8"))
    env_txt = json.dumps(env)
    assert RECEIPT_SHA in env_txt and "203" in env_txt

    # TOUT transition: 07-15 -> blank epoch; 07-16 -> 00 epoch (the
    # dead blank must NOT be selected)
    r15 = {r["id"]: r for r in registry_for_day("2026-07-15")}
    r16 = {r["id"]: r for r in registry_for_day("2026-07-16")}
    assert r15["UW.TOUT"]["location"] == ""
    assert r15["UW.TOUT"]["epoch_end"] == datetime(2026, 7, 16)
    assert r16["UW.TOUT"]["location"] == "00"
    assert r16["UW.TOUT"]["epoch_end"] is None

    # RER three adjacent epochs (same blank location, mid-day
    # transitions resolve at day start)
    for day, end in (("2026-07-30", datetime(2026, 7, 30, 21, 0)),
                     ("2026-07-31", datetime(2026, 8, 12, 18, 27)),
                     ("2026-08-13", None)):
        row = {r["id"]: r for r in registry_for_day(day)}["UW.RER"]
        assert row["location"] == "" and row["epoch_end"] == end, \
            (day, row)

    # synthetic: simultaneously ACTIVE blank + 00 -> blank wins
    fx = [{"network": "XX", "station": "AAA", "location": "00",
           "channel": "HHZ", "latitude": 46.0, "longitude": -122.0,
           "epoch_start": datetime(2026, 1, 1), "epoch_end": None},
          {"network": "XX", "station": "AAA", "location": "",
           "channel": "HHZ", "latitude": 46.0, "longitude": -122.0,
           "epoch_start": datetime(2026, 1, 1), "epoch_end": None}]
    row = registry_for_day("2026-06-01", epochs=fx)[0]
    assert row["location"] == ""

    # synthetic: same-location overlap -> typed refusal
    fx2 = [dict(fx[1]), dict(fx[1])]
    try:
        registry_for_day("2026-06-01", epochs=fx2)
        raise AssertionError("same-loc overlap must refuse")
    except EpochOverlapError as exc:
        assert "EPOCH_OVERLAP_SAME_LOCATION" in str(exc)

    # synthetic: epoch opening mid-day appears the FOLLOWING day
    fx3 = [{"network": "XX", "station": "BBB", "location": "",
            "channel": "HHZ", "latitude": 46.0, "longitude": -122.0,
            "epoch_start": datetime(2026, 6, 1, 12, 0),
            "epoch_end": None}]
    assert registry_for_day("2026-06-01", epochs=fx3) == []
    assert len(registry_for_day("2026-06-02", epochs=fx3)) == 1

    # typed input refusal
    try:
        registry_for_day("06/01/2026", epochs=fx3)
        raise AssertionError("bad day format must refuse")
    except RegistryInputInvalid as exc:
        assert "BAD_DAY_FORMAT" in str(exc)

    print("w2_cascadia selftest: ALL PASS")


if __name__ == "__main__":
    _selftest()
