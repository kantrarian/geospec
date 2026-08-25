#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""THE cleared corrected-OMNI grammar probe -- request 1 of 636.

AUTHORIZATION (both gates closed, first time in this arc):
- owner: asylum spoke "fire the probe" IN THIS SESSION on devildog;
  codex's clearance records it: "The in-session owner consent is
  already recorded; no further owner ask is required for this one
  request."
- reviewer: codex 2026-08-25T13:04:35Z -- "PASS ... step 2 is
  CLEARED. Fire that exact request once, with no retry/fallback/
  alternate request; create-once the S/T/body evidence and route
  either the evidence or a typed probe refusal."

SCOPE CEILING: exactly ONE HTTP request. Grammar evidence only. This
probe satisfies NO scientific key; the day becomes admissible only
later through cayley's v4 predecessor-evidence bridge. Nothing here
authorizes relabelling, admission, the remaining 635, bind,
calibration, Tier-S/C, PRESTART, or a seal.

EVIDENCE (codex "S/T/body" + cayley's 1310Z pre-fire catch, which
codex's own wording independently confirms): FOUR create-once
artifacts about the SAME single exchange --
  1. the raw body,
  2. a real `f2g-w2-capture-transcript-v1` T that the PRODUCTION
     verifier PROD.verify_transcript(T, S, body) accepts, so the
     bridge has a genuine T to reopen and no caller-synthesized
     envelope ever stands in for it,
  3. the closed static contract S,
  4. the probe envelope the authority's discipline specifies.
Emitting T is additive evidence about the same request -- it changes
no URL, day, or count. Without it, 2026-01-01 would be permanently
unadmittable (bridge refuses, refetch forbidden, a replacement
request would breach the 636 ceiling).

S PROVENANCE (disclosed for cayley's bridge): lane, carrier, utc_day,
endpoint and request_params come VERBATIM from the reviewed probe
authority's `probe` block. The closed static-contract schema needs
four more fields, each taken from REGISTERED vocabulary rather than
invented: source.kind = the registered `omniweb-highres-cgi`;
source.ref = the endpoint; cutoff = the codex-confirmed weather
cutoff 2026-07-31; operation_params = the registered {carrier, day}
shape; expected_keys = [utc_day]. If v4 derives any of these
differently for this day, the bridge must reconcile against THESE
bytes -- they are stated here before the request is fired.
"""
import hashlib
import json
import os
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import w2_acquisition_capture_grassmann as CAP
import w2_producer_grassmann as PROD

REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
AUTH_COMMIT = "a81b878"
AUTH_PATH = ("docs/f2g_window2_execution/"
             "omni_probe_authority_v4.json")
AUTH_BLOB_SHA = ("efbda04e6fb26fedd7a8d9a6e25200ed"
                 "ebd840809e21dd0a4b09c388e9e9d562")
CLEARED_URL = (
    "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi?activity=retrieve&"
    "end_date=20260101&res=min&spacecraft=omni_min&"
    "start_date=20260101&vars=17&vars=18&vars=21")
OUT = os.path.join(REPO, "docs", "f2g_window2_execution",
                   "probe_evidence")
STEM = "omni_corrected_probe_20260101"
REGISTERED_SOURCE_KIND = "omniweb-highres-cgi"
WEATHER_CUTOFF = "2026-07-31"
TIMEOUT_S = 120
UA = "geospec-w2-probe/1.0 (kantrarian/geospec window-2)"


def _blob(commitish, path):
    p = subprocess.run(["git", "-C", REPO, "cat-file", "blob",
                        f"{commitish}:{path}"], capture_output=True)
    if p.returncode != 0:
        raise SystemExit(f"REFUSING: {path} unreadable at {commitish}")
    return p.stdout


def _resolve(commitish):
    p = subprocess.run(["git", "-C", REPO, "rev-parse",
                        f"{commitish}^{{commit}}"],
                       capture_output=True)
    full = p.stdout.decode().strip()
    if p.returncode != 0 or len(full) != 40:
        raise SystemExit(f"REFUSING: {commitish} does not resolve")
    return full


def _now_z():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main():
    raw_auth = _blob(AUTH_COMMIT, AUTH_PATH)
    got = hashlib.sha256(raw_auth).hexdigest()
    if got != AUTH_BLOB_SHA:
        raise SystemExit(
            f"REFUSING: probe authority blob {got[:12]} != the "
            f"codex-reviewed {AUTH_BLOB_SHA[:12]}")
    auth = json.loads(raw_auth.decode("utf-8"))
    probe = auth["probe"]
    if hashlib.sha256(json.dumps(
            probe, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest() != \
            auth["probe_sha256"]:
        raise SystemExit("REFUSING: probe self-digest does not "
                         "recompute")
    spec = {"lane": probe["lane"], "carrier": probe["carrier"],
            "utc_day": probe["utc_day"],
            "endpoint": probe["endpoint"],
            "request_params": probe["request_params"],
            "source": {"kind": REGISTERED_SOURCE_KIND,
                       "ref": probe["endpoint"]},
            "cutoff": WEATHER_CUTOFF,
            "operation_params": {"carrier": probe["carrier"],
                                 "day": probe["utc_day"]},
            "expected_keys": [probe["utc_day"]]}
    s = CAP.static_contract_of(spec)
    url = PROD.requested_url_of(s["endpoint"], s["request_params"])
    # the URL is DERIVED, then required to equal the cleared string
    # AND the authority's own recorded spelling -- three-way, before
    # any socket exists
    if url != CLEARED_URL or url != probe["requested_url"]:
        raise SystemExit(
            "REFUSING: derived URL diverges from the cleared "
            f"request\n derived: {url}\n cleared: {CLEARED_URL}")
    auth_id = {"commit": _resolve(AUTH_COMMIT), "path": AUTH_PATH,
               "blob_sha256": AUTH_BLOB_SHA,
               "keys_sha256": auth["probe_sha256"]}
    PROD._validate_authority_id(auth_id, "probe authority id")
    os.makedirs(OUT, exist_ok=True)
    for suf in (".body", ".transcript.json", ".contract.json",
                ".envelope.json"):
        if os.path.exists(os.path.join(OUT, STEM + suf)):
            raise SystemExit(f"REFUSING: {STEM + suf} exists -- "
                             "create-once, never a second request")
    if len(sys.argv) > 1 and sys.argv[1] == "preflight":
        print("PREFLIGHT OK -- no request fired")
        print("  derived == cleared URL:", url == CLEARED_URL)
        print("  S digest:", PROD._canon_digest(s))
        return
    # ---- the ONE request ----
    ctx = ssl.create_default_context(cafile=__import__(
        "certifi").where())
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    env = {"schema": "f2g-w2-probe-envelope-v1",
           "operation_id": STEM,
           "authorization": ("codex 2026-08-25T13:04:35Z step-2 "
                             "clearance; owner go in-session"),
           "authority_ref": dict(auth_id),
           "claim_ceiling": ("grammar evidence only; satisfies NO "
                             "scientific key; admission only via the "
                             "v4 predecessor-evidence bridge"),
           "requested_url": url, "request_start_utc": _now_z()}
    body = b""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=TIMEOUT_S,
                                    context=ctx) as r:
            body = r.read()
            env["http_status"] = getattr(r, "status", r.getcode())
            env["effective_url"] = r.geturl()
            env["response_headers"] = {
                k.lower(): v for k, v in r.headers.items()
                if k.lower() in ("content-type", "content-length",
                                 "server", "date")}
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read()
        except Exception:
            body = b""
        env["http_status"] = exc.code
        env["effective_url"] = getattr(exc, "url", url)
        env["response_headers"] = {}
        env["refusal"] = f"PROBE_REFUSED: HTTP {exc.code}"
    except Exception as exc:
        env["http_status"] = 0
        env["effective_url"] = url
        env["response_headers"] = {}
        env["refusal"] = (f"PROBE_REFUSED: {type(exc).__name__}: "
                          f"{exc}")
    env["response_complete_utc"] = _now_z()
    env["raw_body_bytes"] = len(body)
    env["raw_body_sha256"] = hashlib.sha256(body).hexdigest()
    if "refusal" not in env and (env["http_status"] != 200
                                 or not body):
        env["refusal"] = (f"PROBE_REFUSED: status "
                          f"{env['http_status']} bytes {len(body)}")
    # every artifact is written create-once, refusal or not
    with open(os.path.join(OUT, STEM + ".body"), "wb") as f:
        f.write(body)
    CAP._write_once_json(os.path.join(OUT, STEM + ".contract.json"),
                         s, "PROBE_ARTIFACT_DIVERGENT")
    transcript = None
    if "refusal" not in env:
        transcript = {
            "schema": PROD.TRANSCRIPT_SCHEMA, "lane": s["lane"],
            "carrier": s["carrier"], "utc_day": s["utc_day"],
            "static_contract_sha256": PROD._canon_digest(s),
            "requested_url": env["requested_url"],
            "effective_url": env["effective_url"],
            "request_start_utc": env["request_start_utc"],
            "response_complete_utc": env["response_complete_utc"],
            "http_status": env["http_status"],
            "headers": dict(env["response_headers"]),
            "raw_body_sha256": env["raw_body_sha256"],
            "raw_body_bytes": env["raw_body_bytes"],
            "authority": dict(auth_id)}
        # the PRODUCTION verifier must accept it, or the probe is a
        # typed refusal rather than a silently weak anchor
        PROD.verify_transcript(transcript, s, raw_body=body)
        env["transcript_sha256"] = PROD._canon_digest(transcript)
        CAP._write_once_json(
            os.path.join(OUT, STEM + ".transcript.json"), transcript,
            "PROBE_ARTIFACT_DIVERGENT")
    CAP._write_once_json(os.path.join(OUT, STEM + ".envelope.json"),
                         env, "PROBE_ARTIFACT_DIVERGENT")
    print("PROBE", env["http_status"], len(body),
          env.get("refusal", "OK"))
    if transcript is not None:
        print("transcript VERIFIED by PROD.verify_transcript;",
              "sha", env["transcript_sha256"][:16])


if __name__ == "__main__":
    main()
