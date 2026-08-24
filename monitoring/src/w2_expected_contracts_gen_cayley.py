#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 INDEPENDENTLY-EXPECTED RECORD CONTRACTS v1 generator
(cayley) -- the static layer of the producer_boundary BIND condition
(codex 1544Z: "Cayley's prestart path supplies and pins the
independently expected record contracts").

INDEPENDENCE RULE (content-auth != derivation provenance): every value
here derives from REGISTERED artifacts -- the calendar authority
(day-set arithmetic), the pinned MAG probe envelopes (endpoints), the
schedule/renewal artifacts (cutoff) -- NEVER from the envelope records
the contracts will verify, and never from the acquisition code.

TWO-LAYER DESIGN (the receipt/capture seam, routed for ruling):
`verify_staged_day_set` compares ALL of (source, endpoint,
request_params, receipt, capture_time_utc, cutoff, operation_params,
expected_keys) to the independent contract. Receipt and capture
instant cannot exist before capture; a contract that copies them from
the records is vacuous for those fields. This artifact therefore
registers the STATIC layer and declares the DYNAMIC layer's carrier:
receipt + capture_time_utc enter the per-day contract FROM THE CAPTURE
TRANSCRIPT (the acquisition harness's recorded spec/transcript tree),
a separate carrier from the envelope records, before the day-set gate
runs. codex ruling requested; grassmann wires the transcript carrier.

Deterministic: same bytes every run. Opens no window-2 value.
"""
import datetime
import hashlib
import json
import os

OUT_REL = os.path.join("docs", "f2g_window2_execution",
                       "staged_expected_contracts_v2.json")

CUTOFF = "2026-08-25"
SELECTION_LOOKBACK_START = "2026-05-28"     # cutoff - 89 (90 days)
CALIBRATION_START = "2026-01-01"
CARRIERS = ("istanbul_marmara", "socal_coachella",
            "turkey_kahramanmaras", "cascadia")
MAG_OBSERVATORIES = ("izn", "frn", "tuc")
# MF4 driver-series carrier TOKENS (registered here; the key set is
# CLOSED even while endpoints stay OPEN for the specs round):
# sym_h = WDC-Kyoto SYM-H, kp = GFZ Kp, omni = NASA OMNI
MF4_DRIVERS = ("sym_h", "kp", "omni")

# endpoints: REGISTERED evidence = the pinned probe envelopes'
# requested_url hosts (mag_<obs>_probe.envelope.json); the per-day
# query params are request_params (OPEN below). Every value is
# verified against the pinned envelope bytes at generation.
MAG_ENDPOINTS = {
    "izn": "https://imag-data.bgs.ac.uk/GIN_V1/GINServices",
    "frn": "https://geomag.usgs.gov/ws/data/",
    "tuc": "https://geomag.usgs.gov/ws/data/",
}


def _span(a, b):
    d0 = datetime.date.fromisoformat(a)
    d1 = datetime.date.fromisoformat(b)
    assert d0 <= d1
    return [(d0 + datetime.timedelta(days=i)).isoformat()
            for i in range((d1 - d0).days + 1)]


def _digest(obj):
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def build(repo):
    sel_days = _span(SELECTION_LOOKBACK_START, CUTOFF)
    cal_days = _span(CALIBRATION_START, CUTOFF)
    assert len(sel_days) == 90
    assert sel_days[-1] == cal_days[-1] == CUTOFF
    # MAG endpoints: verify the izn value against the PINNED probe
    # envelope bytes (independent registered evidence, not the records)
    for obs in MAG_OBSERVATORIES:
        env = json.load(open(os.path.join(
            repo, "docs", "f2g_window2_execution", "mag_capsules",
            "receipts", f"mag_{obs}_probe.envelope.json"),
            encoding="utf-8"))
        assert env["requested_url"].startswith(MAG_ENDPOINTS[obs]), \
            f"{obs} endpoint diverges from the pinned probe evidence"

    def tmpl(source_kind, source_ref, endpoint):
        return {"source": {"kind": source_kind, "ref": source_ref},
                "endpoint": endpoint,
                "request_params": "OPEN_REVIEW_ROUND",
                "operation_params": "OPEN_REVIEW_ROUND"}

    lanes = {}
    lanes["SELECTION_RECORDS"] = {
        "carriers": {ck: {
            "expected_days": sel_days,
            "cutoff": CUTOFF,
            "source_class": "FDSN dataselect/station (registered "
                            "carrier networks)",
            "endpoint": "OPEN_REVIEW_ROUND",
            "request_params": "OPEN_REVIEW_ROUND",
            "operation_params": "OPEN_REVIEW_ROUND",
            "expected_keys": "OPEN_REVIEW_ROUND",
            "static_contract_template": tmpl(
                "fdsn", "OPEN_REVIEW_ROUND", "OPEN_REVIEW_ROUND")}
            for ck in CARRIERS},
        "day_set_rule": f"[cutoff-89, cutoff] = "
                        f"[{SELECTION_LOOKBACK_START}, {CUTOFF}], "
                        "90 days exact (selection frame)"}
    lanes["MAG_FEED"] = {
        "carriers": {obs: {
            "expected_days": cal_days,
            "cutoff": CUTOFF,
            "source_class": ("INTERMAGNET GIN" if obs == "izn"
                             else "USGS geomagnetism"),
            "endpoint": MAG_ENDPOINTS[obs],
            "request_params": "OPEN_REVIEW_ROUND",
            "operation_params": "OPEN_REVIEW_ROUND",
            "expected_keys": "OPEN_REVIEW_ROUND",
            "static_contract_template": tmpl(
                "gin-minute" if obs == "izn" else "usgs-minute",
                MAG_ENDPOINTS[obs], MAG_ENDPOINTS[obs])}
            for obs in MAG_OBSERVATORIES},
        "day_set_rule": f"calibration span [{CALIBRATION_START}, "
                        f"{CUTOFF}] (mag1 instantiation)"}
    lanes["MF4_FEED"] = {
        "carriers": {drv: {
            "expected_days": cal_days,
            "cutoff": CUTOFF,
            "source_class": {"sym_h": "WDC-Kyoto SYM-H",
                             "kp": "GFZ Kp",
                             "omni": "NASA OMNI"}[drv],
            "endpoint": "OPEN_REVIEW_ROUND",
            "request_params": "OPEN_REVIEW_ROUND",
            "operation_params": "OPEN_REVIEW_ROUND",
            "expected_keys": "OPEN_REVIEW_ROUND",
            "static_contract_template": tmpl(
                "driver-series", "OPEN_REVIEW_ROUND",
                "OPEN_REVIEW_ROUND")}
            for drv in MF4_DRIVERS},
        "day_set_rule": f"calibration span [{CALIBRATION_START}, "
                        f"{CUTOFF}]"}
    lanes["DAY_CAPSULE"] = {
        "carriers": "EXCLUDED_FROM_PRESTART (accrual-time lane per "
                    "codex 1843Z item 5 + 0238Z item 1: separate "
                    "per-day admission rule; a DAY_CAPSULE pin in "
                    "the PRESTART staged tree REFUSES)",
        "day_set_rule": "evaluation days at accrual time"}

    # codex 0238Z item 1: THE sole exact authority for the PRESTART
    # (lane, carrier, day) key set -- derived ONLY from the calendar/
    # probe/schedule registrations above, never from submitted pins
    prestart_keys = {}
    for lane in ("SELECTION_RECORDS", "MAG_FEED", "MF4_FEED"):
        prestart_keys[lane] = {
            ck: list(v["expected_days"])
            for ck, v in lanes[lane]["carriers"].items()}

    return {
        "schema": "f2g-w2-expected-contracts-v2",
        "prestart_expected_keys": prestart_keys,
        "prestart_expected_keys_sha256": _digest(prestart_keys),
        "static_layer": lanes,
        "dynamic_layer": {
            "fields": ["receipt", "capture_time_utc"],
            "carrier": "capture transcript tree (acquisition "
                       "harness recorded specs), NEVER the envelope "
                       "records under verification",
            "status": "DESIGN_QUESTION_ROUTED (codex ruling; "
                      "grassmann wires)"},
        "digests": {
            "selection_days_sha256": _digest(sel_days),
            "calibration_days_sha256": _digest(cal_days)},
        "provenance": {
            "producer": "monitoring/src/"
                        "w2_expected_contracts_gen_cayley.py",
            "independence": "derived from calendar authority + "
                            "pinned probe envelopes + schedule "
                            "artifacts only; no record or "
                            "acquisition-code derivation",
            "bind_note": "producer_boundary BINDS only when the OPEN "
                         "values are settled in review and this "
                         "artifact (or its successor) is "
                         "manifest-pinned alongside staged bytes + "
                         "records",
            "claim_ceiling": "registration only; no staging, no "
                             "power value; Lambda_geo INCONCLUSIVE"}}


def main():
    repo = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    body = json.dumps(build(repo), indent=1, sort_keys=True) + "\n"
    out = os.path.join(repo, OUT_REL)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
    print(f"wrote {OUT_REL}")
    print("artifact sha256:",
          hashlib.sha256(body.encode()).hexdigest())


if __name__ == "__main__":
    main()
