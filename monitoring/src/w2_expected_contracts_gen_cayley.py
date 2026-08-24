#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 INDEPENDENTLY-EXPECTED RECORD CONTRACTS v3 generator
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
                       "staged_expected_contracts_v3.json")

CUTOFF = "2026-08-27"
SELECTION_LOOKBACK_START = "2026-05-30"     # cutoff - 89 (90 days)
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

    # registered template token vocabulary (consumed by
    # authoritative_static_contract): {day} = the capture UTC day;
    # {day_next} = the UTC day after it (half-open [day, day_next)
    # request windows -- USGS/FDSN day forms). Any other brace token
    # survives substitution and fails downstream comparison closed.
    template_tokens = ["{day}", "{day_next}", "{day_compact}"]

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

    # ---------------- PHASE-A FILL (codex 1434Z limit 5) ----------
    # Fill the fillable (lane, carrier) templates from grassmann's
    # capture specs v1 + the six pinned probe envelopes. Every filled
    # template is VERIFIED at generation: substituting the probe day
    # into the template must reproduce the envelope's requested query
    # exactly (the verbatim-derivation lock). socal_coachella and kp
    # stay OPEN_REVIEW_ROUND: their probes are PROBE_REFUSED pinned
    # and their templates are BLOCKED pending a new codex ruling.
    import urllib.parse

    specs = json.load(open(os.path.join(
        repo, "docs", "f2g_window2_execution",
        "capture_specs_v1_grassmann.json"), encoding="utf-8"))
    probe_rec = json.load(open(os.path.join(
        repo, "docs", "f2g_window2_execution",
        "probe_input_record_v1_grassmann.json"), encoding="utf-8"))
    PROBE_DAY = "2025-11-15"
    PROBE_DAY_NEXT = "2025-11-16"
    PROBE_DAY_COMPACT = "20251115"

    def sub_probe(v):
        if isinstance(v, str):
            return (v.replace("{day_next}", PROBE_DAY_NEXT)
                     .replace("{day_compact}", PROBE_DAY_COMPACT)
                     .replace("{day}", PROBE_DAY))
        if isinstance(v, dict):
            return {k: sub_probe(x) for k, x in v.items()}
        if isinstance(v, list):
            return [sub_probe(x) for x in v]
        return v

    def envelope_query(env_rel, endpoint):
        env = json.load(open(os.path.join(
            repo, "docs", "f2g_window2_execution", "probe_evidence",
            env_rel), encoding="utf-8"))
        assert env["http_status"] == 200, env_rel
        req = env["requested_url"]
        assert req.startswith(endpoint), (env_rel, endpoint)
        q = urllib.parse.parse_qs(
            urllib.parse.urlsplit(req).query,
            keep_blank_values=True)
        return q, env

    def verify_probe_fill(env_rel, endpoint, tmpl_params):
        """The template with the probe day substituted must equal the
        envelope's actual requested query."""
        got, env = envelope_query(env_rel, endpoint)
        want = {}
        for k, v in sub_probe(tmpl_params).items():
            want[k] = v if isinstance(v, list) else [v]
        assert got == want, (env_rel, got, want)
        return {"probe_envelope": "docs/f2g_window2_execution/"
                                  "probe_evidence/" + env_rel,
                "probe_body_sha256": env["raw_body_sha256"],
                "probe_day_utc": PROBE_DAY}

    def fill(lane, ck, *, kind, endpoint, request_params,
             evidence, source_class=None):
        e = lanes[lane]["carriers"][ck]
        t = {"source": {"kind": kind, "ref": endpoint},
             "endpoint": endpoint,
             "request_params": request_params,
             "operation_params": {"carrier": ck, "day": "{day}"}}
        e["static_contract_template"] = t
        e["endpoint"] = endpoint
        e["request_params"] = dict(request_params)
        e["operation_params"] = dict(t["operation_params"])
        e["expected_keys"] = ("one key per expected day "
                              "(derived contract binds the day)")
        e["fill_evidence"] = evidence
        e["fill_status"] = "FILLED"
        if source_class:
            e["source_class"] = source_class

    # MAG izn/frn/tuc + SELECTION cascadia: EVIDENCE_PINNED verbatim
    # from capture specs v1 (MAG endpoints already asserted against
    # the pinned mag probe envelopes above)
    for lane, ck in (("MAG_FEED", "izn"), ("MAG_FEED", "frn"),
                     ("MAG_FEED", "tuc"),
                     ("SELECTION_RECORDS", "cascadia")):
        sp = specs["lanes"][lane][ck]
        assert sp["status"] == "EVIDENCE_PINNED", (lane, ck)
        fill(lane, ck, kind=sp["source"]["kind"],
             endpoint=sp["endpoint"],
             request_params=dict(sp["request_params"]),
             evidence=dict(sp["evidence"],
                           spec_status="EVIDENCE_PINNED"))

    # istanbul/turkey: probe-record params day-templated; the probe
    # envelope proves the exact grammar (TEMPLATE_GRAMMAR_CONFIRMED)
    for ck, env_rel in (
            ("istanbul_marmara",
             "selection_records_istanbul_marmara.envelope.json"),
            ("turkey_kahramanmaras",
             "selection_records_turkey_kahramanmaras.envelope.json")):
        pk = probe_rec["keys"][f"SELECTION_RECORDS/{ck}"]
        rp = dict(pk["request_params"])
        assert rp.pop("starttime") == PROBE_DAY
        assert rp.pop("endtime") == PROBE_DAY_NEXT
        rp["starttime"] = "{day}"
        rp["endtime"] = "{day_next}"
        ev = verify_probe_fill(env_rel, pk["endpoint"], rp)
        fill("SELECTION_RECORDS", ck, kind="fdsn-station-channel",
             endpoint=pk["endpoint"], request_params=rp,
             evidence=dict(ev,
                           verdict="TEMPLATE_GRAMMAR_CONFIRMED"))

    # sym_h/omni: OMNIWeb high-res CGI; compact-date {day_compact}
    for ck, env_rel, sclass in (
            ("sym_h", "mf4_feed_sym_h.envelope.json",
             "NASA OMNIWeb high-res SYM/H (var 41)"),
            ("omni", "mf4_feed_omni.envelope.json",
             "NASA OMNIWeb high-res BZ-GSM/flow/density "
             "(vars 17/21/25)")):
        pk = probe_rec["keys"][f"MF4_FEED/{ck}"]
        rp = dict(pk["request_params"])
        assert rp.pop("start_date") == PROBE_DAY_COMPACT
        assert rp.pop("end_date") == PROBE_DAY_COMPACT
        rp["start_date"] = "{day_compact}"
        rp["end_date"] = "{day_compact}"
        ev = verify_probe_fill(env_rel, pk["endpoint"], rp)
        fill("MF4_FEED", ck, kind="omniweb-highres-cgi",
             endpoint=pk["endpoint"], request_params=rp,
             evidence=dict(ev,
                           verdict="TEMPLATE_GRAMMAR_CONFIRMED"),
             source_class=sclass)

    # kp: CONFIRMED at attempt-2 (codex 1623Z two-retry ruling;
    # identical URL bytes, delta = verified TLS w/ a real CA bundle
    # -- attempt-1 was purely the local trust store)
    pk = probe_rec["keys"]["MF4_FEED/kp"]
    rp = dict(pk["request_params"])
    assert rp.pop("start") == PROBE_DAY + "T00:00:00Z"
    assert rp.pop("end") == PROBE_DAY + "T23:59:59Z"
    rp["start"] = "{day}T00:00:00Z"
    rp["end"] = "{day}T23:59:59Z"
    ev = verify_probe_fill("kp_attempt2.envelope.json",
                           pk["endpoint"], rp)
    fill("MF4_FEED", "kp", kind="gfz-kp-json",
         endpoint=pk["endpoint"], request_params=rp,
         evidence=dict(ev, verdict="TEMPLATE_GRAMMAR_CONFIRMED",
                       attempt="2 (verified TLS; attempt-1 refusal "
                               "was the local trust store)"),
         source_class="GFZ Kp JSON (three-hourly definitive)")

    # socal_coachella: REFUSED at BOTH ruled attempts (HTTP 400 on
    # abbreviated AND long-form FDSN grammars) -- template BLOCKED;
    # exclusion or another request = a separate codex decision. The
    # OPEN tokens keep the freeze gate refusing (structurally honest)
    lanes["SELECTION_RECORDS"]["carriers"]["socal_coachella"][
        "fill_status"] = ("BLOCKED_PROBE_REFUSED -- two ruled "
                          "attempts (abbrev + long-form) both HTTP "
                          "400; pending a separate codex decision")
    lanes["SELECTION_RECORDS"]["carriers"]["socal_coachella"][
        "fill_evidence"] = {
        "refusal": "SCEDC HTTP 400 on both authorized requests",
        "probe_envelopes": [
            "docs/f2g_window2_execution/probe_evidence/"
            "selection_records_socal_coachella.envelope.json",
            "docs/f2g_window2_execution/probe_evidence/"
            "socal_coachella_attempt2.envelope.json"]}

    # codex 0238Z item 1: THE sole exact authority for the PRESTART
    # (lane, carrier, day) key set -- derived ONLY from the calendar/
    # probe/schedule registrations above, never from submitted pins
    prestart_keys = {}
    for lane in ("SELECTION_RECORDS", "MAG_FEED", "MF4_FEED"):
        prestart_keys[lane] = {
            ck: list(v["expected_days"])
            for ck, v in lanes[lane]["carriers"].items()}

    return {
        "schema": "f2g-w2-expected-contracts-v3",
        "template_token_vocabulary": template_tokens,
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
