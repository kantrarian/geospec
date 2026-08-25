#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OMNI CORRECTED-GRAMMAR PROBE AUTHORITY generator (cayley) --
codex 0544Z sequencing ruling step 1.

WHY THIS EXISTS
---------------
codex 0527Z finding 2: the v3 OMNI template requested vars 17/21/25,
which NASA's live form maps to By_GSM / flow speed / proton density.
The frozen Newell coupling needs **Bz_GSM = var 18**, so the captured
OMNI cannot compute the registered regressor. The corrected template
(17/18/21) has no pinned envelope, and a template with no evidence
cannot carry a derivation lock -- while production capture must not
fire under an unfrozen authority. codex 0544Z broke that cycle:
precommit THIS narrow one-request probe authority, fire exactly it,
then bind its evidence into the closed v4 authority through an
explicit predecessor-evidence bridge.

WHAT THIS AUTHORIZES
--------------------
Exactly ONE HTTP request: corrected-vars OMNI for ONE deterministically
named day. The day is `2026-01-01` -- the FIRST day of the registered
MAG interval, chosen structurally so no value-dependent selection is
possible. It counts as request 1 of asylum's authorized 636
(quote sha f2411fa7...), leaving 211 OMNI + 424 VIC/NEW after the
freeze closes; the probe day is never refetched.

CLAIM CEILING
-------------
Grammar evidence ONLY. This artifact admits nothing scientifically:
no value, no calibration input, no expected-key satisfaction. The
probe body becomes scientifically admissible only later, through the
v4 predecessor-evidence bridge that reruns the pinned v4 transform and
binds both authority lineages -- never by relabelling this record.
"""
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

OUT_REL = os.path.join("docs", "f2g_window2_execution",
                       "omni_probe_authority_v4.json")
SCHEMA = "f2g-w2-omni-probe-authority-v4"
LANE = "MAG_WEATHER_FEED"
CARRIER = "omni"
# DETERMINISTIC: the first day of the registered MAG interval. No
# value-dependent choice is possible -- the rule fixes the day before
# any byte is seen.
PROBE_DAY = "2026-01-01"
ENDPOINT = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"
CORRECTED_VARS = ["17", "18", "21"]         # By_GSM, Bz_GSM, flow
VAR_LABELS = {"17": "By_GSM (nT, GSM)",
              "18": "Bz_GSM (nT, GSM)",
              "21": "flow speed (km/s)"}


def _digest(obj):
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def build():
    import w2_expected_contracts_gen_cayley as GEN
    import w2_producer_grassmann as PROD

    # the day must lie inside the registered MAG interval, and the
    # rule that picks it must be structural, not chosen
    assert GEN.CALIBRATION_START <= PROBE_DAY <= GEN.MAG_CUTOFF
    assert PROBE_DAY == GEN.CALIBRATION_START, (
        "the probe day must be the FIRST registered MAG day -- any "
        "other choice admits value-dependent selection")
    compact = PROBE_DAY.replace("-", "")
    request_params = {"activity": "retrieve", "res": "min",
                      "spacecraft": "omni_min",
                      "start_date": compact, "end_date": compact,
                      "vars": list(CORRECTED_VARS)}
    # the exact bytes grassmann must request, through the PRODUCTION
    # canonical builder (repeated vars, one canonical spelling)
    requested_url = PROD.requested_url_of(ENDPOINT, request_params)
    body = {
        "schema": SCHEMA,
        "authorization": {
            "owner_quote": "go ahead with the 636 corrective capture",
            "owner_quote_sha256":
                "f2411fa7a4b828f0780bf643e452281a4748d5fc5d0bae690"
                "f8d86629586f4f1",
            "codex_ruling": "2026-08-25T05:44:19Z OMNI sequencing "
                            "(636 total, probe first, retroactive "
                            "lock REJECTED)",
            "requests_authorized": 1,
            "counts_against_owner_ceiling": 636,
            "remaining_after_freeze": {"omni": 211,
                                       "vic_new": 424}},
        "probe": {
            "lane": LANE, "carrier": CARRIER,
            "utc_day": PROBE_DAY,
            "day_selection_rule": "the FIRST day of the registered "
                                  "MAG interval (CALIBRATION_START) "
                                  "-- structural, fixed before any "
                                  "byte is seen",
            "endpoint": ENDPOINT,
            "request_params": request_params,
            "requested_url": requested_url,
            "vars_corrected": list(CORRECTED_VARS),
            "var_labels": {v: VAR_LABELS[v] for v in CORRECTED_VARS},
            "supersedes_vars": ["17", "21", "25"],
            "supersedes_reason": "vars 17/21/25 = By_GSM / flow "
                                 "speed / proton density; the frozen "
                                 "Newell coupling needs Bz_GSM = var "
                                 "18 (codex 0527Z finding 2)"},
        "discipline": {
            "requests": "EXACTLY ONE; no retry, fallback host, "
                        "alternate dataset, expanded range, or "
                        "second exploratory request",
            "on_failure": "pin a typed probe refusal; the corrected "
                          "OMNI template stays BLOCKED and the "
                          "matter returns to codex",
            "evidence": "create-once raw body + envelope binding "
                        "requested/effective URL, day, request start "
                        "and completion UTC, status, headers, byte "
                        "count, body sha256, parser note, and "
                        "independently recomputed minute coverage"},
        "claim_ceiling": {
            "scientific_admission": "NONE -- grammar evidence only",
            "expected_keys_satisfied": "NONE; this probe does not "
                                       "satisfy MAG_WEATHER_FEED/"
                                       "omni/2026-01-01 as a "
                                       "scientific key",
            "later_admission": "only through the v4 "
                               "predecessor-evidence bridge, which "
                               "must reproduce this request "
                               "byte-for-byte, reopen this "
                               "transcript/body, rerun the pinned v4 "
                               "transform, and bind BOTH authority "
                               "lineages; relabelling this record is "
                               "refused",
            "lambda_geo": "INCONCLUSIVE"},
        "producer": "monitoring/src/w2_omni_probe_authority_gen_"
                    "cayley.py"}
    body["probe_sha256"] = _digest(body["probe"])
    return body


def main():
    repo = os.path.abspath(os.path.join(_HERE, "..", ".."))
    out = json.dumps(build(), indent=1, sort_keys=True) + "\n"
    with open(os.path.join(repo, OUT_REL), "w", encoding="utf-8",
              newline="\n") as f:
        f.write(out)
    print(f"wrote {OUT_REL.replace(os.sep, '/')}")
    print("artifact sha256:",
          hashlib.sha256(out.encode()).hexdigest())


def _selftest():
    a, b = build(), build()
    assert a == b, "probe authority must be deterministic"
    assert a["authorization"]["requests_authorized"] == 1
    assert a["probe"]["request_params"]["vars"] == ["17", "18", "21"]
    assert "vars=17&vars=18&vars=21" in a["probe"]["requested_url"], \
        a["probe"]["requested_url"]
    assert "%5B" not in a["probe"]["requested_url"], \
        "stringified list spelling must be impossible"
    assert a["probe"]["utc_day"] == "2026-01-01"
    n = (a["authorization"]["remaining_after_freeze"]["omni"]
         + a["authorization"]["remaining_after_freeze"]["vic_new"]
         + a["authorization"]["requests_authorized"])
    assert n == 636, f"request ceiling arithmetic {n} != 636"
    print("w2_omni_probe_authority selftest: ALL PASS "
          "(1 request, corrected vars, 1+211+424=636, grammar only)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
