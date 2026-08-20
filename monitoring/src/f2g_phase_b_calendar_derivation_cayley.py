#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Derive the canonical shared calendar + per-carrier registered-day masks
FROM THE PHASE-A ANCHOR (codex c33dc41f repair 3) -- metadata only, no r
values read.

Authority chain: the anchor phase_a_result.json (known SHA-256 0850cf3d...)
binds every output file digest; its output_digests KEYS name the snapshot
paths snapshots/<carrier>/<date>.json, so the registered-day masks are
INDEPENDENTLY RE-DERIVABLE by any party holding the anchor bytes -- no share
access or directory listing trusted. This script additionally verifies the
local artifact copy's snapshot files hash-match the anchor (proof the local
copy IS the sealed artifact), derives the 132-day civil calendar
2026-03-01..2026-07-10, and emits the mask authority JSON.
"""
import datetime
import hashlib
import json
import re
import sys

ANCHOR_SHA = "0850cf3d24602ab0ba420412f5b292c9d33464852fbb9de8d7363a019c7886ad"


def main(artifact_dir, out_json, anchor_copy_out):
    anchor_path = f"{artifact_dir}/phase_a_result.json"
    raw = open(anchor_path, "rb").read()
    got = hashlib.sha256(raw).hexdigest()
    if got != ANCHOR_SHA:
        print(f"ANCHOR MISMATCH: {got}")
        sys.exit(2)
    anchor = json.loads(raw.decode("utf-8"))
    pat = re.compile(r"^snapshots/([a-z_]+)/(\d{4}-\d{2}-\d{2})\.json$")
    masks = {}
    verified = 0
    for path, digest in sorted(anchor["output_digests"].items()):
        m = pat.match(path)
        if not m:
            continue
        carrier, day = m.group(1), m.group(2)
        masks.setdefault(carrier, []).append(day)
        local = open(f"{artifact_dir}/{path}", "rb").read()
        if hashlib.sha256(local).hexdigest() != digest:
            print(f"LOCAL SNAPSHOT MISMATCH: {path}")
            sys.exit(2)
        verified += 1
    for c in masks:
        masks[c] = sorted(masks[c])
    all_days = sorted({d for ds in masks.values() for d in ds})
    d0 = datetime.date.fromisoformat(all_days[0])
    d1 = datetime.date.fromisoformat(all_days[-1])
    calendar = [(d0 + datetime.timedelta(days=i)).isoformat()
                for i in range((d1 - d0).days + 1)]
    out = {
        "schema": "f2g-phase-b-shared-calendar-v1",
        "source_anchor_sha256": ANCHOR_SHA,
        "derivation": ("registered-day masks = the date components of the "
                       "anchor's output_digests keys matching "
                       "snapshots/<carrier>/<date>.json; independently "
                       "re-derivable from the anchor bytes alone"),
        "local_copy_verification": {
            "snapshot_files_hash_matched": verified,
            "note": "every mask snapshot's local bytes hash-match the "
                    "anchor digest (the local copy IS the sealed artifact)"},
        "shared_calendar_days": calendar,
        "calendar_positions": len(calendar),
        "carrier_masks": {
            c: {"registered_days": ds, "count": len(ds),
                "absent_days": [d for d in calendar if d not in set(ds)],
                "absent_count": sum(1 for d in calendar
                                    if d not in set(ds))}
            for c, ds in sorted(masks.items())},
    }
    with open(out_json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    with open(anchor_copy_out, "wb") as f:
        f.write(raw)
    print(json.dumps({
        "calendar_positions": len(calendar),
        "span": [calendar[0], calendar[-1]],
        "counts": {c: len(ds) for c, ds in sorted(masks.items())},
        "absent": {c: out["carrier_masks"][c]["absent_count"]
                   for c in sorted(masks)},
        "verified_files": verified}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
