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
import urllib.parse
import subprocess

OUT_REL = os.path.join("docs", "f2g_window2_execution",
                       "staged_expected_contracts_v3.json")

# --- SUCCESSOR v4 (codex 0527Z postflight findings 1-3) ------------
# PER-LANE cutoffs: selection keeps 08-27; the MAG lanes cap at 07-31
# because NASA publishes high-resolution SYM/H only through 07-31, and
# the MAG fit needs observatory minutes AND weather regressors on ONE
# grid -- so observatory minutes past the weather cutoff carry no
# regressors. (The observatory cap is cayley's inference, flagged for
# codex in the 0538Z design spec; it is one constant, not a design.)
SELECTION_CUTOFF = "2026-08-27"
SELECTION_LOOKBACK_START = "2026-05-30"     # cutoff - 89 (90 days)
MAG_CUTOFF = "2026-07-31"
CALIBRATION_START = "2026-01-01"
CUTOFF = SELECTION_CUTOFF                   # back-compat alias
LANE_CUTOFF = {"SELECTION_RECORDS": SELECTION_CUTOFF,
               "MAG_FEED": MAG_CUTOFF,
               "MAG_WEATHER_FEED": MAG_CUTOFF}
CARRIERS = ("istanbul_marmara", "socal_coachella",
            "turkey_kahramanmaras", "cascadia")
# codex finding 3: one lane name was masking two carrier spaces.
# MAG_WEATHER_FEED = the MAG-1 regressors (sym_h/kp/corrected omni).
# MF4_MONITOR_FEED = the true M-F4 daily-risk monitor carrier, which
# is an ARCHIVE (risk_by_region/catalog_snapshot/...), NOT a per-day
# HTTP key set -- registered OPEN here pending its producer.
MAG_WEATHER_DRIVERS = ("sym_h", "kp", "omni")
MF4_DRIVERS = MAG_WEATHER_DRIVERS           # back-compat alias
DESIGN_MANIFEST_REL = "docs/f2g_window2_freeze/byte_pin_manifest.json"
EXEC_MANIFEST_REL = ("docs/f2g_window2_execution/"
                     "execution_manifest.json")
import re as _re                            # noqa: E402
_CAPSULE_RE = _re.compile(r"mag_capsule_[a-z0-9_]+\.json$")


def admitted_mag_observatories(repo):
    """codex 0527Z finding 1: THE admitted MAG observatory set, DERIVED
    from the typed capsules pinned by the two REGISTERED manifests --
    never a typed constant. A typed tuple is exactly what omitted VIC
    and NEW: it agreed with itself while diverging from the freeze.

    design/byte-pin manifest -> the 2026-08-22 cascadia amendment
    capsules (VIC, NEW); execution manifest -> IZN, FRN, TUC.
    Returns {iaga_lower: {"iaga", "capsule", "probe_envelope"}}.
    """
    def _load(rel):
        with open(os.path.join(repo, rel.replace("/", os.sep)),
                  encoding="utf-8") as f:
            return json.load(f)
    paths = set()
    dm = _load(DESIGN_MANIFEST_REL)
    pins = dm.get("pins")
    for e in (pins.values() if isinstance(pins, dict) else pins or []):
        p = e.get("path") if isinstance(e, dict) else None
        if p and _CAPSULE_RE.search(str(p)):
            paths.add(str(p))
    for slot in (_load(EXEC_MANIFEST_REL).get("slots") or {}).values():
        if not isinstance(slot, dict):
            continue
        for pin in slot.get("pins") or []:
            p = pin.get("path") if isinstance(pin, dict) else pin
            if p and _CAPSULE_RE.search(str(p)):
                paths.add(str(p))
    if not paths:
        raise AssertionError(
            "CARRIER_SET_UNDERIVABLE: no MAG capsule is pinned by "
            "either registered manifest")
    out = {}
    for p in sorted(paths):
        iaga = _load(p).get("iaga_code")
        if not isinstance(iaga, str) or not iaga:
            raise AssertionError(
                f"CARRIER_CAPSULE_UNTYPED: {p} carries no iaga_code")
        key = iaga.lower()
        if key in out:
            # codex 0532Z registered VIC at the execution tree as an
            # EXACT BYTE COPY of the preserved freeze capsule, so one
            # observatory can now be pinned at two paths (design
            # manifest -> freeze; execution manifest -> execution).
            # Byte-identical copies are ONE unambiguous authority --
            # dedupe, preferring the EXECUTION path the transform
            # reads. DIVERGENT bytes for one iaga remain the defect
            # this check exists to refuse.
            prev = json.dumps(_load(out[key]["capsule"]),
                              sort_keys=True, separators=(",", ":"))
            cur = json.dumps(_load(p), sort_keys=True,
                             separators=(",", ":"))
            if prev != cur:
                raise AssertionError(
                    f"CARRIER_CAPSULE_DUPLICATE: {iaga} pinned twice "
                    "with DIVERGENT bytes")
            # identical bytes: keep whichever entry can resolve its
            # probe envelope -- VIC's receipts deliberately stay at
            # the freeze path (codex 0532Z: the execution slot must
            # not imply receipts it does not have), so the freeze
            # entry is the one that carries the envelope.
            def _env_exists(capsule_path):
                dd = os.path.dirname(capsule_path)
                st = f"mag_{key}_probe.envelope.json"
                return (os.path.exists(os.path.join(
                            repo, dd.replace("/", os.sep),
                            "receipts", st))
                        or os.path.exists(os.path.join(
                            repo, dd.replace("/", os.sep), st)))
            if _env_exists(out[key]["capsule"]):
                continue      # existing entry resolves its envelope
            if not _env_exists(p):
                raise AssertionError(
                    f"CARRIER_ENVELOPE_UNDERIVABLE: {iaga} pinned "
                    "twice and neither path resolves a probe "
                    "envelope")
            # fall through: replace with the envelope-bearing path
        # the probe envelope sits beside the capsule, named for the obs
        d = os.path.dirname(p)
        stem = f"mag_{key}_probe.envelope.json"
        env = (f"{d}/receipts/{stem}" if os.path.exists(
            os.path.join(repo, d.replace("/", os.sep), "receipts",
                         stem)) else f"{d}/{stem}")
        out[key] = {"iaga": iaga, "capsule": p,
                    "probe_envelope": env}
    return out


# per-observatory request FORM, keyed by the provider the capsule
# names; the concrete endpoint + query are verified at generation
# against that observatory's PINNED probe envelope.
GIN_ENDPOINT = "https://imag-data.bgs.ac.uk/GIN_V1/GINServices"
USGS_ENDPOINT = "https://geomag.usgs.gov/ws/data/"


def mag_request(iaga):
    """The registered per-day request for one observatory, in the
    provider form its pinned probe envelope demonstrates."""
    up = iaga.upper()
    if up in ("IZN", "VIC"):            # INTERMAGNET GIN
        return GIN_ENDPOINT, "intermagnet-gin-minute", {
            "Request": "GetData", "format": "json", "testObsys": "0",
            "observatoryIagaCode": up, "samplesPerDay": "minute",
            "dataStartDate": "{day}", "dataDuration": "1",
            "publicationState": "adj-or-rep"}
    return USGS_ENDPOINT, "usgs-geomag-ws-minute", {  # USGS ws
        "id": up, "format": "json", "sampling_period": "60",
        "starttime": "{day}T00:00:00Z",
        "endtime": "{day_next}T00:00:00Z"}


def _span(a, b):
    d0 = datetime.date.fromisoformat(a)
    d1 = datetime.date.fromisoformat(b)
    assert d0 <= d1
    return [(d0 + datetime.timedelta(days=i)).isoformat()
            for i in range((d1 - d0).days + 1)]


def _blob_at_head(repo, rel):
    """Committed bytes only -- working copies EOL-convert."""
    r = subprocess.run(["git", "-C", repo, "cat-file", "blob",
                        f"HEAD:{rel}"], capture_output=True)
    if r.returncode != 0:
        raise AssertionError(f"EVIDENCE_UNREADABLE: {rel}")
    return r.stdout


def _digest(obj):
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


MAG_PROBE_DAY = "2026-08-19"   # the pinned MAG probe envelopes' day


def _sub_day(v, day):
    """Registered-token substitution at a concrete UTC day."""
    dn = (datetime.date.fromisoformat(day)
          + datetime.timedelta(days=1)).isoformat()
    dc = day.replace("-", "")

    def sub(x):
        if isinstance(x, str):
            return (x.replace("{day_next}", dn)
                     .replace("{day_compact}", dc)
                     .replace("{day}", day))
        if isinstance(x, dict):
            return {k: sub(y) for k, y in x.items()}
        if isinstance(x, list):
            return [sub(y) for y in x]
        return x
    return sub(v)


def validate_evidence_obj(env, body, *, endpoint, tmpl_params,
                          probe_day, time_fields=(),
                          evidence_time_values=None, label=""):
    """codex 2205Z finding 4: THE one executable evidence lock, used
    for ALL TEN templates by build() and exercised directly by the
    mutation KATs. HTTP 200; exact body sha (and size where
    recorded); requested origin+path == the registered endpoint
    EXACTLY; the registered query -- INCLUDING repeated parameters --
    compared through the PRODUCTION canonical builder
    (w2_producer_grassmann.requested_url_of): one builder, both
    sides. time_fields = the cascadia broad-window rule: those
    template values must be exactly the registered day tokens while
    the envelope keeps its own receipt window; every other field
    compares exactly."""
    import w2_producer_grassmann as PROD
    if env.get("http_status") != 200:
        raise AssertionError(f"{label}: evidence status "
                             f"{env.get('http_status')} != 200")
    sha = env.get("raw_body_sha256") or env.get("body_sha256")
    if not sha or hashlib.sha256(body).hexdigest() != sha:
        raise AssertionError(f"{label}: body digest diverges from "
                             "the pinned envelope")
    if env.get("raw_body_bytes") is not None and \
            len(body) != env["raw_body_bytes"]:
        raise AssertionError(f"{label}: body size diverges")
    r = urllib.parse.urlsplit(env["requested_url"])
    e = urllib.parse.urlsplit(endpoint)
    if (r.scheme, r.netloc, r.path) != (e.scheme, e.netloc, e.path):
        raise AssertionError(f"{label}: requested origin/path "
                             "diverges from the registered endpoint")
    actual = {}
    for k, v in urllib.parse.parse_qsl(r.query,
                                       keep_blank_values=True):
        actual.setdefault(k, []).append(v)
    actual = {k: (v[0] if len(v) == 1 else v)
              for k, v in actual.items()}
    tp = dict(tmpl_params)
    if time_fields:
        # codex 2329Z finding 4: the evidence-side broad window is a
        # CLOSED registered binding -- never copied from the envelope.
        # Values must be canonical ISO dates in strict order and the
        # envelope must carry them EXACTLY.
        if not isinstance(evidence_time_values, dict) or \
                set(evidence_time_values) != set(time_fields):
            raise AssertionError(
                f"{label}: date-transform -- evidence_time_values "
                "must be a closed mapping over exactly the time "
                "fields")
        parsed = {}
        for f in time_fields:
            v = evidence_time_values[f]
            try:
                parsed[f] = datetime.date.fromisoformat(str(v))
            except ValueError:
                raise AssertionError(
                    f"{label}: date-transform -- registered evidence "
                    f"time value {v!r} for {f} is not a canonical "
                    "ISO date")
        if "starttime" in parsed and "endtime" in parsed and \
                not parsed["starttime"] < parsed["endtime"]:
            raise AssertionError(
                f"{label}: date-transform -- registered evidence "
                "window is not strictly ordered")
        for f in time_fields:
            if tp.get(f) not in ("{day}", "{day_next}"):
                raise AssertionError(
                    f"{label}: date-transform -- {f} is not a "
                    "registered day token (the broad-window "
                    "transform substitutes time fields ONLY)")
            if f not in actual:
                raise AssertionError(f"{label}: date-transform -- "
                                     f"{f} absent in the evidence "
                                     "query")
            if actual[f] != str(evidence_time_values[f]):
                raise AssertionError(
                    f"{label}: date-transform -- evidence {f} "
                    f"{actual[f]!r} diverges from the registered "
                    f"broad-window value "
                    f"{evidence_time_values[f]!r}")
            tp[f] = actual[f]
    expected = _sub_day(tp, probe_day)
    if PROD.requested_url_of(endpoint, expected) != \
            PROD.requested_url_of(endpoint, actual):
        raise AssertionError(f"{label}: registered query diverges "
                             "from the pinned evidence "
                             "(canonical-builder comparison)")
    return env


def build(repo):
    sel_days = _span(SELECTION_LOOKBACK_START, SELECTION_CUTOFF)
    # successor v4: the MAG lanes end at their OWN cutoff (07-31)
    cal_days = _span(CALIBRATION_START, MAG_CUTOFF)
    mag_obs = admitted_mag_observatories(repo)
    assert len(sel_days) == 90
    # successor v4: cutoffs are PER LANE -- each day set must end at
    # its own registered cutoff, and the MAG cutoff must not exceed
    # the selection cutoff (the weather publishes with more lag)
    assert sel_days[-1] == SELECTION_CUTOFF
    assert cal_days[-1] == MAG_CUTOFF
    assert MAG_CUTOFF <= SELECTION_CUTOFF
    assert set(LANE_CUTOFF) == {"SELECTION_RECORDS", "MAG_FEED",
                                "MAG_WEATHER_FEED"}

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
    def _mag_carrier(obs):
        ep, kind, _rp = mag_request(mag_obs[obs]["iaga"])
        return {"expected_days": cal_days,
                "cutoff": MAG_CUTOFF,
                "source_class": ("INTERMAGNET GIN" if kind ==
                                 "intermagnet-gin-minute"
                                 else "USGS geomagnetism"),
                "capsule": mag_obs[obs]["capsule"],
                "endpoint": ep,
                "request_params": "OPEN_REVIEW_ROUND",
                "operation_params": "OPEN_REVIEW_ROUND",
                "expected_keys": "OPEN_REVIEW_ROUND",
                "static_contract_template": tmpl(kind, ep, ep)}
    lanes["MAG_FEED"] = {
        "carriers": {obs: _mag_carrier(obs)
                     for obs in sorted(mag_obs)},
        "day_set_rule": f"calibration span [{CALIBRATION_START}, "
                        f"{MAG_CUTOFF}] (mag1 instantiation); the "
                        "carrier set is DERIVED from the capsules "
                        "pinned by the design + execution manifests "
                        "(codex 0527Z finding 1), never typed"}
    # codex 0527Z finding 3: the captured sym_h/kp/omni objects are
    # MAG-1 WEATHER REGRESSORS, not the M-F4 monitor feed. The lane
    # name split follows the carrier spaces.
    lanes["MAG_WEATHER_FEED"] = {
        "carriers": {drv: {
            "expected_days": cal_days,
            "cutoff": MAG_CUTOFF,
            "source_class": {
                "sym_h": "NASA OMNIWeb high-res SYM/H",
                "kp": "GFZ Kp (def/pre)",
                "omni": ("NASA OMNIWeb high-res By_GSM/Bz_GSM/flow "
                         "speed (vars 17/18/21)")}[drv],
            "endpoint": "OPEN_REVIEW_ROUND",
            "request_params": "OPEN_REVIEW_ROUND",
            "operation_params": "OPEN_REVIEW_ROUND",
            "expected_keys": "OPEN_REVIEW_ROUND",
            "static_contract_template": tmpl(
                "driver-series", "OPEN_REVIEW_ROUND",
                "OPEN_REVIEW_ROUND")}
            for drv in MAG_WEATHER_DRIVERS},
        "day_set_rule": f"MAG regressor span [{CALIBRATION_START}, "
                        f"{MAG_CUTOFF}] (NASA publishes high-res "
                        "SYM/H only through 07-31)"}
    # the TRUE M-F4 carrier: a daily-risk ARCHIVE + pinned catalog
    # snapshot, NOT a per-day HTTP key set -- registered here so the
    # lane exists and is visibly unfilled; its producer and key rule
    # land with the M-F4 continuity carrier (codex finding 3).
    lanes["MF4_MONITOR_FEED"] = {
        "carriers": "OPEN_REVIEW_ROUND (daily-monitor risk archive + "
                    "pinned catalog snapshot; {risk_by_region, "
                    "catalog_snapshot, snapshot_end, freeze_day, "
                    "bboxes, regions} accepted by the REAL "
                    "run_mf4_calibration -- not a per-day HTTP lane)",
        "day_set_rule": "OPEN_REVIEW_ROUND (archive carrier)"}
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

    def _validate_evidence(env_rel, *, endpoint, tmpl_params,
                           probe_day, time_fields=(),
                           evidence_time_values=None):
        """File wrapper over validate_evidence_obj: reopens the named
        envelope AND its body from the repo (new schema: sibling .body
        + raw_body_sha256; old schema: body_path + body_sha256)."""
        envp = os.path.join(repo, env_rel.replace("/", os.sep))
        with open(envp, encoding="utf-8") as f:
            env = json.load(f)
        if env.get("raw_body_sha256"):
            body_rel = env_rel[:-len(".envelope.json")] + ".body"
        else:
            body_rel = env["body_path"]
        # the evidence bytes are the COMMITTED git object (working
        # copies suffer EOL conversion; the recorded sha binds the
        # blob, exactly how the reviewer recomputes it)
        r = subprocess.run(["git", "-C", repo, "cat-file", "blob",
                            f"HEAD:{body_rel}"], capture_output=True)
        if r.returncode != 0:
            raise AssertionError(f"{env_rel}: evidence body "
                                 f"{body_rel} unreadable at HEAD")
        body = r.stdout
        validate_evidence_obj(env, body, endpoint=endpoint,
                              tmpl_params=tmpl_params,
                              probe_day=probe_day,
                              time_fields=time_fields,
                              evidence_time_values=(
                                  evidence_time_values),
                              label=env_rel)
        return {"probe_envelope": env_rel,
                "probe_body_sha256":
                    env.get("raw_body_sha256") or env["body_sha256"],
                "probe_day_utc": probe_day}

    def verify_probe_fill(env_rel, endpoint, tmpl_params):
        """The six probe-evidence templates (probe day 2025-11-15)."""
        return _validate_evidence(
            "docs/f2g_window2_execution/probe_evidence/" + env_rel,
            endpoint=endpoint, tmpl_params=tmpl_params,
            probe_day=PROBE_DAY)

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
    # successor v4: the MAG fill iterates the DERIVED observatory set
    # (izn/frn/tuc + the amendment's VIC/NEW), building each request
    # from the ONE registered form in mag_request() and locking it
    # against that observatory's own pinned probe envelope -- so
    # adding a frozen carrier can never again mean an unfilled lane.
    for obs in sorted(mag_obs):
        ep, kind, rp = mag_request(mag_obs[obs]["iaga"])
        ev = _validate_evidence(mag_obs[obs]["probe_envelope"],
                                endpoint=ep, tmpl_params=rp,
                                probe_day=MAG_PROBE_DAY)
        fill("MAG_FEED", obs, kind=kind, endpoint=ep,
             request_params=rp,
             evidence=dict(ev, verdict="TEMPLATE_GRAMMAR_CONFIRMED",
                           capsule=mag_obs[obs]["capsule"],
                           spec_status="EVIDENCE_PINNED"))
    for lane, ck in (("SELECTION_RECORDS", "cascadia"),):
        sp = specs["lanes"][lane][ck]
        assert sp["status"] == "EVIDENCE_PINNED", (lane, ck)
        # finding 4: the SAME executable evidence lock as the probe
        # templates -- envelope+body reopened, digests recomputed,
        # origin/path bound, query compared through the production
        # canonical builder
        if ck == "cascadia":
            # broad-window receipt -> per-day transform: time fields
            # must be exactly the registered tokens; all other fields
            # compare exactly against the receipt query (probe_day is
            # inert here -- no token survives outside time_fields)
            _validate_evidence(
                sp["evidence"]["pinned_receipt_envelope"],
                endpoint=sp["endpoint"],
                tmpl_params=dict(sp["request_params"]),
                probe_day="2026-07-11",
                time_fields=("starttime", "endtime"),
                evidence_time_values={"starttime": "2026-07-11",
                                      "endtime": "2026-11-30"})
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
    # codex 0527Z finding 2 + 1746Z gate-1 finding 1: vars 17/21/25
    # were By_GSM / flow speed / proton density -- Bz_GSM is var 18,
    # so the frozen Newell coupling was UNCOMPUTABLE. The corrected
    # probe (request 1 of 636) FIRED and created closed evidence, so
    # OMNI is filled from THOSE COMMITTED BYTES -- reopened and
    # verified here, never transcribed.
    pe = "docs/f2g_window2_execution/probe_evidence/"
    stem = pe + "omni_corrected_probe_20260101"
    p_contract = json.loads(_blob_at_head(
        repo, stem + ".contract.json").decode("utf-8"))
    p_transcript = json.loads(_blob_at_head(
        repo, stem + ".transcript.json").decode("utf-8"))
    p_body = _blob_at_head(repo, stem + ".body")
    p_auth = json.loads(_blob_at_head(
        repo, "docs/f2g_window2_execution/"
              "omni_probe_authority_v4.json").decode("utf-8"))
    # the probe evidence must agree with the reviewed probe AUTHORITY
    pa_probe = p_auth["probe"]
    if p_contract.get("endpoint") != pa_probe["endpoint"]:
        raise AssertionError(
            "OMNI_PROBE_EVIDENCE_DIVERGENT: contract endpoint != the "
            "reviewed probe authority")
    if p_contract["request_params"].get("vars") != \
            pa_probe["request_params"]["vars"]:
        raise AssertionError(
            "OMNI_PROBE_EVIDENCE_DIVERGENT: contract vars != the "
            "reviewed corrected vars")
    # T must bind THIS body and THIS contract
    if hashlib.sha256(p_body).hexdigest() != \
            p_transcript.get("raw_body_sha256"):
        raise AssertionError(
            "OMNI_PROBE_EVIDENCE_DIVERGENT: transcript does not bind "
            "the committed body")
    if p_transcript.get("http_status") != 200:
        raise AssertionError(
            "OMNI_PROBE_EVIDENCE_DIVERGENT: non-200 probe transcript")
    # day-template the concrete probe day back out of the evidence
    o_rp = dict(p_contract["request_params"])
    pday_compact = pa_probe["utc_day"].replace("-", "")
    for k in ("start_date", "end_date"):
        if o_rp.get(k) != pday_compact:
            raise AssertionError(
                f"OMNI_PROBE_EVIDENCE_DIVERGENT: {k} is not the "
                "probe day")
        o_rp[k] = "{day_compact}"
    fill("MAG_WEATHER_FEED", "omni",
         kind=p_contract["source"]["kind"],
         endpoint=p_contract["endpoint"], request_params=o_rp,
         evidence={"probe_contract": stem + ".contract.json",
                   "probe_transcript": stem + ".transcript.json",
                   "probe_body_sha256":
                       hashlib.sha256(p_body).hexdigest(),
                   "probe_day_utc": pa_probe["utc_day"],
                   "verdict": "TEMPLATE_GRAMMAR_CONFIRMED",
                   "source": "corrected-OMNI probe, request 1 of 636 "
                             "(codex 1304Z step-2 clearance)"},
         source_class="NASA OMNIWeb high-res By_GSM/Bz_GSM/flow "
                      "speed (vars 17/18/21)")
    lanes["MAG_WEATHER_FEED"]["carriers"]["omni"][
        "superseded_evidence"] = {
        "envelope": pe + "mf4_feed_omni.envelope.json",
        "reason": "vars 17/21/25 mis-identified as Bz; retained as "
                  "evidence, never as an admitted grammar"}
    for ck, env_rel, sclass in (
            ("sym_h", "mf4_feed_sym_h.envelope.json",
             "NASA OMNIWeb high-res SYM/H (var 41)"),):
        pk = probe_rec["keys"][f"MF4_FEED/{ck}"]
        rp = dict(pk["request_params"])
        assert rp.pop("start_date") == PROBE_DAY_COMPACT
        assert rp.pop("end_date") == PROBE_DAY_COMPACT
        rp["start_date"] = "{day_compact}"
        rp["end_date"] = "{day_compact}"
        ev = verify_probe_fill(env_rel, pk["endpoint"], rp)
        fill("MAG_WEATHER_FEED", ck, kind="omniweb-highres-cgi",
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
    fill("MAG_WEATHER_FEED", "kp", kind="gfz-kp-json",
         endpoint=pk["endpoint"], request_params=rp,
         evidence=dict(ev, verdict="TEMPLATE_GRAMMAR_CONFIRMED",
                       attempt="2 (verified TLS; attempt-1 refusal "
                               "was the local trust store)"),
         source_class="GFZ Kp JSON (three-hourly definitive)")

    # socal_coachella: GRAMMAR CONFIRMED at attempt 4 (codex 1711Z
    # one-delta ruling; SCEDC requires FULL-DATETIME spellings -- its
    # own 400 body named the defect; attempts 1-3 pinned refusals).
    # Codex 1647Z parser contract: the registered bbox query + the
    # exact 12-station filter/active-epoch rule BOTH bind here.
    rp = {"net": "CI", "cha": "HHZ", "level": "channel",
          "format": "text",
          "minlatitude": "32.6500", "maxlatitude": "34.1500",
          "minlongitude": "-116.9500", "maxlongitude": "-115.0500",
          "starttime": "{day}T00:00:00.000000",
          "endtime": "{day_next}T00:00:00.000000",
          "nodata": "404"}
    ev = verify_probe_fill(
        "socal_coachella_attempt4.envelope.json",
        "https://service.scedc.caltech.edu/fdsnws/station/1/query",
        rp)
    e = lanes["SELECTION_RECORDS"]["carriers"]["socal_coachella"]
    fill("SELECTION_RECORDS", "socal_coachella",
         kind="fdsn-station-channel",
         endpoint="https://service.scedc.caltech.edu/fdsnws/"
                  "station/1/query",
         request_params=rp,
         evidence=dict(
             ev, verdict="TEMPLATE_GRAMMAR_CONFIRMED",
             attempt="4 (one-delta full-datetime spelling per "
                     "SCEDC's own 400 diagnosis; attempts 1-3 "
                     "pinned refusals, byte-untouched)"))
    # the 1647Z registered filter + epoch rule ride operation_params
    # (bbox responses may contain out-of-set stations -- retained in
    # raw evidence, EXCLUDED by this registered filter)
    e["operation_params"]["registered_station_filter"] = (
        "ACP,ANG,BAR,BC3,BEL,BLA2,BOM,BOR,COA,CRR,CSH,CTC")
    e["operation_params"]["presence_rule"] = (
        "CI stations from the registered filter with an HHZ channel "
        "epoch active in [{day}, {day_next}); outside-station rows "
        "retained in raw evidence, excluded by the filter; a proper "
        "subset of the 12 is an honest per-day presence result")
    e["static_contract_template"]["operation_params"] = dict(
        e["operation_params"])

    # codex 0238Z item 1: THE sole exact authority for the PRESTART
    # (lane, carrier, day) key set -- derived ONLY from the calendar/
    # probe/schedule registrations above, never from submitted pins
    prestart_keys = {}
    for lane in ("SELECTION_RECORDS", "MAG_FEED",
                 "MAG_WEATHER_FEED"):
        prestart_keys[lane] = {
            ck: list(v["expected_days"])
            for ck, v in lanes[lane]["carriers"].items()}

    # codex 1304Z bridge finding 3: the v4 authority PINS the probe
    # authority (commit/path/LF-blob sha) so the predecessor lineage
    # is REGISTERED rather than asserted by a caller.
    probe_rel = ("docs/f2g_window2_execution/"
                 "omni_probe_authority_v4.json")
    probe_blob = subprocess.run(
        ["git", "-C", repo, "cat-file", "blob", f"HEAD:{probe_rel}"],
        capture_output=True)
    probe_commit = subprocess.run(
        ["git", "-C", repo, "log", "-1", "--format=%H", "HEAD", "--",
         probe_rel], capture_output=True)
    if probe_blob.returncode != 0 or not probe_commit.stdout.strip():
        raise AssertionError(
            "PROBE_AUTHORITY_UNPINNABLE: the corrected-OMNI probe "
            "authority must be committed before the v4 authority can "
            "register its lineage")
    registered_probe_authority = {
        "path": probe_rel,
        "commit": probe_commit.stdout.decode().strip(),
        "blob_sha256": hashlib.sha256(probe_blob.stdout).hexdigest(),
        "role": "predecessor-evidence lineage for "
                "MAG_WEATHER_FEED/omni/2026-01-01; the bridge reopens "
                "THIS blob, never a caller-supplied object"}

    return {
        "schema": "f2g-w2-expected-contracts-v3",
        "registered_probe_authority": registered_probe_authority,
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


def _selftest():
    """codex 2205Z finding 4 mutation KATs against the REAL
    validate_evidence_obj (the same function build() routes every
    template through): changed query, status, body digest,
    origin/path, source host, date-transform violation,
    repeated-parameter positive + stringified negative. build() runs
    FIRST so all ten committed evidence locks execute."""
    import copy
    repo = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    art = build(repo)          # ALL TEN evidence locks execute here
    assert art["schema"] == "f2g-w2-expected-contracts-v3"

    body = b"17-col-rows"
    sha = hashlib.sha256(body).hexdigest()
    base_env = {"http_status": 200, "raw_body_sha256": sha,
                "raw_body_bytes": len(body),
                "requested_url": "https://x.example/q?a=1&"
                                 "starttime=2025-11-15&"
                                 "endtime=2025-11-16"}
    tmpl = {"a": "1", "starttime": "{day}", "endtime": "{day_next}"}

    def check(env, t, **kw):
        validate_evidence_obj(
            env, kw.pop("body_bytes", body),
            endpoint=kw.pop("endpoint", "https://x.example/q"),
            tmpl_params=t, probe_day=kw.pop("probe_day",
                                            "2025-11-15"),
            time_fields=kw.pop("time_fields", ()),
            evidence_time_values=kw.pop("evidence_time_values",
                                        None), label="kat")
    check(base_env, tmpl)                       # positive
    for doctor, want in (
            (lambda e: e.update(requested_url=e["requested_url"]
                                .replace("a=1", "a=2")), "query"),
            (lambda e: e.update(http_status=500), "status"),
            (lambda e: e.update(raw_body_sha256="0" * 64),
             "body digest"),
            (lambda e: e.update(requested_url=e["requested_url"]
                                .replace("/q?", "/qq?")),
             "origin/path"),
            (lambda e: e.update(requested_url=e["requested_url"]
                                .replace("x.example", "y.example")),
             "origin/path"),
    ):
        env2 = copy.deepcopy(base_env)
        doctor(env2)
        try:
            check(env2, dict(tmpl))
            raise SystemExit(f"doctor must refuse: {want}")
        except AssertionError as ex:
            assert want in str(ex), (want, str(ex))
    # date-transform violation: a literal where the token must be
    try:
        check(dict(base_env), dict(tmpl, starttime="2025-11-15"),
              time_fields=("starttime",),
              evidence_time_values={"starttime": "2025-11-15"})
        raise SystemExit("literal time field must refuse")
    except AssertionError as ex:
        assert "date-transform" in str(ex)
    # broad-window transform positive: the CLOSED registered window
    env_bw = dict(base_env,
                  requested_url="https://x.example/q?a=1&"
                                "starttime=2026-07-11&"
                                "endtime=2026-11-30")
    BW = {"starttime": "2026-07-11", "endtime": "2026-11-30"}
    check(env_bw, dict(tmpl),
          time_fields=("starttime", "endtime"),
          evidence_time_values=dict(BW))
    # codex 2329Z finding 4 doctors: envelope time mutations refuse
    for evil_url, why in (
            ("https://x.example/q?a=1&starttime=EVIL&"
             "endtime=ALSO_EVIL", "diverges"),
            ("https://x.example/q?a=1&starttime=2026-07-12&"
             "endtime=2026-11-30", "diverges"),
            ("https://x.example/q?a=1&starttime=2026-07-11&"
             "endtime=2026-11-29", "diverges"),
            ("https://x.example/q?a=1&starttime=2026-07-11",
             "absent"),
    ):
        try:
            check(dict(env_bw, requested_url=evil_url), dict(tmpl),
                  time_fields=("starttime", "endtime"),
                  evidence_time_values=dict(BW))
            raise SystemExit("envelope-time doctor must refuse: "
                             + why)
        except AssertionError as ex:
            assert why in str(ex), (why, str(ex))
    # registered-side doctors: malformed value, reversed window,
    # non-closed mapping, omitted binding
    for etv, why in (
            ({"starttime": "EVIL", "endtime": "2026-11-30"},
             "canonical ISO"),
            ({"starttime": "2026-11-30", "endtime": "2026-07-11"},
             "strictly ordered"),
            ({"starttime": "2026-07-11"}, "closed mapping"),
            (None, "closed mapping"),
    ):
        try:
            check(dict(env_bw), dict(tmpl),
                  time_fields=("starttime", "endtime"),
                  evidence_time_values=etv)
            raise SystemExit("registered-side doctor must refuse: "
                             + why)
        except AssertionError as ex:
            assert why in str(ex), (why, str(ex))
    # repeated parameters: registered form passes, stringified refuses
    envr = {"http_status": 200, "raw_body_sha256": sha,
            "raw_body_bytes": len(body),
            "requested_url": "https://x.example/q?vars=17&vars=21"}
    check(envr, {"vars": ["17", "21"]})
    envb = dict(envr, requested_url="https://x.example/q?"
                                    "vars=%5B%2717%27%2C+%2721%27%5D")
    try:
        check(envb, {"vars": ["17", "21"]})
        raise SystemExit("stringified list must refuse")
    except AssertionError as ex:
        assert "query" in str(ex)
    print("w2_expected_contracts_gen selftest: ALL PASS (all-ten "
          "committed evidence locks executed via build(); mutation "
          "doctors refuse through the REAL validator; "
          "canonical-builder comparison)")


if __name__ == "__main__":
    import sys as _sys
    if "--selftest" in _sys.argv:
        _selftest()
    else:
        main()
