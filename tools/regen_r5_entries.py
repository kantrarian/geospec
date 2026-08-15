#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-off R5 derived-record correction (codex 1246 contract, step 2).

Regenerates ONLY docs/r5_daily.json days entries for 2026-08-12 and 2026-08-13
using the repaired store-free historical path (r5_transform(..., historical=True)
at geospec ee77902). Input ratio per region = the entry's own recorded raw_ratio
(the R3 ratio actually scored that day). Entries without a computed record
({"r5_active": false}) are left as-is. No other surface is touched; the FC
artifacts are NOT rerun. A dated correction record is appended under
"corrections" (additive key) for provenance.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(r"C:\GeoSpec\geospec_sprint")
sys.path.insert(0, str(REPO / "monitoring" / "src"))

from precip_residual import r5_transform  # noqa: E402
from validate_predictions import REGION_DEFINITIONS  # noqa: E402

R5_DAILY = REPO / "docs" / "r5_daily.json"
DAYS = ["2026-08-12", "2026-08-13"]

data = json.loads(R5_DAILY.read_text())
summary = {}
for day in DAYS:
    entries = data["days"].get(day, {})
    day_sum = {}
    for region, rec in entries.items():
        if not (isinstance(rec, dict) and rec.get("r5_computed")):
            day_sum[region] = "kept (no computed record)"
            continue
        center = REGION_DEFINITIONS.get(region, {}).get("center")
        if not center:
            day_sum[region] = "kept (no center)"
            continue
        new = r5_transform(region, center[0], center[1], rec["raw_ratio"], day,
                           historical=True)
        if new is None:
            entries[region] = {"r5_active": False,
                               "note": "historical as-of fit ineligible (codex-1246 correction)"}
            day_sum[region] = "regenerated -> ineligible (honest None)"
        else:
            entries[region] = new
            day_sum[region] = (f"regenerated fitted_date={new['fitted_date']} "
                               f"n_fit={new['n_fit']} sha={new['model_sha256'][:12]}")
    summary[day] = day_sum

corr = data.setdefault("corrections", [])
corr.append({
    "applied_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "days": DAYS,
    "reason": "codex-1246 R5 replay-order finding: prior entries for these days were "
              "written through the shared live model store during explicit-date replays "
              "(order-dependent). Regenerated via the repaired store-free deterministic "
              "as-of-date path (geospec ee77902); FC artifacts untouched.",
    "by": "grassmann",
})
R5_DAILY.write_text(json.dumps(data, indent=1))
print(json.dumps(summary, indent=1))
