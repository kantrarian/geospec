#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D2 STEP-4B PROVIDER/DRIVER red-KATs (cayley, 2026-08-09) — freezes codex `0347`
prelaunch findings P1/P2/P3 as executable gates. Contract `codex-d2-step4b-2026-08-09-v1`;
companion to `test_d2_step4b_redkats_cayley.py` (REV 2, closed) and codex's `0123` batch bar.
HERMETIC: no network — all HTTP is intercepted at `d2_step4b_providers._http_get` and
`urllib.request.urlopen`; obspy is stubbed in-process. Nothing here fetches, lifts, tunes,
or claims. Per codex 0347 the existing owner receipt stays valid: the first large waveform
request is held ONLY until these gates are red-green (a code gate, not an authorization gate).

CONTRACT SEAMS (grassmann implements UNEDITED; additive to the closed SB-0..8 seams)
====================================================================================
* d2_step4b_producer._acquire(plan: dict, ledger: dict, root: str) -> dict
    - the ONLY acquisition path. Called EXACTLY ONCE by run_campaign, strictly AFTER
      verify_launch_authorization succeeds; it (not run_campaign's prologue) lazily imports
      d2_step4b_providers. No other public/importable path issues provider I/O.
* d2_step4b_producer.run_campaign(plan, launch_authorization, root, ...) -> dict
    - executable order: verify_launch_authorization FIRST (invalid/missing receipt ->
      SystemExit/ValueError with ZERO provider-module calls and ZERO urllib traffic) ->
      bind/hash plan -> _acquire(plan, ledger, root) -> batch assembly.
      NotImplementedError is RETIRED from this path (P1).
    - importing d2_step4b_producer must NOT import d2_step4b_providers (module isolation);
      the provider import happens lazily below the gate.
* d2_step4b_providers.koeri_fetch(nslc, start, end, *, stage_dir, base=..., timeout=...)
    -> {"stream": <trimmed stream>, "raw_objects": [obj]}
    where obj = {"source": <exact request URL>, "staged_path": <file under stage_dir>,
                 "size_bytes": int, "sha256": <64-hex of the exact served bytes>}
    - raw provider bytes are PERSISTED to staged_path BEFORE parsing (P2); a parse failure
      after staging raises but must NOT delete the staged object; sha256/size describe the
      exact served bytes, byte-for-byte.
* d2_step4b_providers.scedc_fetch(nslc, start, end, *, stage_dir, base=..., timeout=...)
    -> same shape with ONE raw_object PER TOUCHED DAY-VOLUME, ordered by day; each object
      hashed SEPARATELY (the concatenated-bytes digest appears NOWHERE in the result) (P2);
      a missing day-volume is tolerated (partial coverage) but every existing touched object
      is retained.
* d2_step4b_providers.verify_staged_object(obj: dict) -> bool
    - reopens staged_path; True iff both size_bytes and sha256 match the file exactly;
      False on any byte mutation or missing file (P2).
* d2_step4b_providers._utc_days_spanned(start, end) -> list[date]
    - HALF-OPEN [start, end): ValueError if end <= start; the last enumerated day is
      (end - 1 microsecond).date() — an exact-midnight 24 h window touches exactly ONE day (P3).
* d2_step4b_providers.scedc_available(net, stations, channels, start, end, ...)
    - probes EVERY touched UTC day for every frozen candidate (urllib HEAD); the NSLC is
      present iff ANY touched day-volume object exists — data only in the second touched
      day must register as available (P3). The HTTP layer stays urllib (this bar intercepts
      urllib.request.urlopen; a transport swap breaks the interception contract).

RED AS AUTHORED vs GeoSpec `8864b3e`: exactly
['PV-1-GATE (_acquire driver seam)', 'PV-2-GATE (staged fetch signature)',
 'PV-3a half-open day enumeration', 'PV-3b every-touched-day availability'].
PV-0 and PV-1a are green context locks (module isolation already holds and stays held).
"""
import hashlib
import inspect
import io
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

FAILS = []


def check(desc, ok, detail=""):
    print(f"    [{'PASS' if ok else 'FAIL'}] {desc}" + (f" - {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(desc)


def raises(fn, exc=Exception):
    try:
        fn()
        return False
    except exc:
        return True


# ---- hermetic obspy stub (parse layer only; no scientific behavior) ----------
class _StubStream(list):
    def trim(self, *a, **k):
        return self

    def split(self):
        return self

    def merge(self, *a, **k):
        return self

    def __iadd__(self, other):
        self.extend(other)
        return self


def _stub_read(buf):
    body = buf.read() if hasattr(buf, "read") else bytes(buf)
    if b"CORRUPT" in body:
        raise ValueError("stub parse failure (corrupt fixture)")
    return _StubStream([object()])


def _install_obspy_stub():
    import types
    m = types.ModuleType("obspy")
    m.read = _stub_read
    m.Stream = _StubStream
    m.UTCDateTime = lambda *a, **k: None
    sys.modules["obspy"] = m


def main():
    _install_obspy_stub()
    try:
        import d2_step4b_producer as P
        import d2_step4b_providers as PR
    except ImportError as exc:
        check("PV-0 producer + providers modules import", False, str(exc))
        return

    U = timezone.utc
    d0700 = datetime(2026, 3, 31, 7, 0, 0, tzinfo=U)
    d0700e = datetime(2026, 4, 1, 7, 0, 0, tzinfo=U)
    mid0 = datetime(2026, 3, 31, 0, 0, 0, tzinfo=U)
    mid1 = datetime(2026, 4, 1, 0, 0, 0, tzinfo=U)

    # ---- PV-1a (green lock): producer import does not load the network stack -
    code = ("import sys; sys.path.insert(0, %r); import d2_step4b_producer as P; "
            "assert 'd2_step4b_providers' not in sys.modules, 'providers imported at load'; "
            "\ntry:\n    P.run_campaign(plan={}, launch_authorization=None, root='.')\n"
            "except TypeError:\n"
            "    try:\n        P.run_campaign(plan={}, launch_authorization=None)\n"
            "    except BaseException:\n        pass\n"
            "except BaseException:\n    pass\n"
            "assert 'd2_step4b_providers' not in sys.modules, 'providers imported below refusal'; "
            "print('ISOLATED')" % HERE)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       timeout=120)
    check("PV-1a module isolation: importing the producer (and a REFUSED run_campaign) never "
          "imports d2_step4b_providers",
          r.returncode == 0 and "ISOLATED" in r.stdout, r.stderr.strip()[-300:])

    # ---- PV-1: run_campaign is the single gated driver ----------------------
    if not hasattr(P, "_acquire"):
        check("PV-1-GATE run_campaign drives acquisition through _acquire strictly after the "
              "SB-8 receipt check (NotImplementedError retired)", False,
              "AWAITING _acquire driver seam -- red-first as authored (codex 0347 P1)")
    else:
        events = []
        real_verify = P.verify_launch_authorization
        real_acquire = P._acquire
        real_http = PR._http_get
        real_urlopen = urllib.request.urlopen

        def spy_verify(receipt):
            events.append("verify")
            return real_verify(receipt)

        def spy_acquire(plan, ledger, root):
            events.append("acquire")
            return {}

        def spy_http(url, timeout):
            events.append(f"http:{url}")
            raise PR.ProviderUnavailable("spy")

        def spy_urlopen(*a, **k):
            events.append("urlopen")
            raise AssertionError("network touched")

        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "published_phase_ledger.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"rows": []}, f)
            plan = P.build_campaign_plan(
                {"socal_coachella": {
                    "coachella_south": [["CI.BOR..BHZ"], ["CI.CTW..BHZ"]],
                    "brawley_seismic_zone": [["CI.BC3..BHZ"], ["CI.RXH..BHZ"]]}},
                "2026-08-09")
            # REV 2 (codex 0447 H2 compat): the fixture root stages the SAME plan the
            # caller supplies, so an H2-compliant run_campaign (which reopens and binds
            # root/campaign_plan.json) proceeds to _acquire on the matching path.
            with open(os.path.join(td, "campaign_plan.json"), "wb") as f:
                f.write((json.dumps(plan, sort_keys=True, separators=(",", ":"))
                         + "\n").encode("utf-8"))
            try:
                P.verify_launch_authorization = spy_verify
                P._acquire = spy_acquire
                PR._http_get = spy_http
                urllib.request.urlopen = spy_urlopen

                events.clear()
                bad_exc = None
                try:
                    P.run_campaign(plan=plan, launch_authorization=None, root=td)
                except NotImplementedError as exc:
                    bad_exc = exc
                except BaseException:
                    pass
                check("PV-1b INVALID RECEIPT refuses with ZERO provider/network calls and "
                      "without reaching _acquire",
                      bad_exc is None and "acquire" not in events
                      and not any(e.startswith("http") or e == "urlopen" for e in events),
                      f"events={events}")

                events.clear()
                receipt = {"status": "VERIFIED_DIRECT",
                           "in_session_timestamp_utc": "2026-08-09T02:04:49Z",
                           "owner_quote_sha256": "0658bdf0b498b551c433bb3f932a87a9"
                                                 "c06e28929703c22d9468507b1fc7d3f8"}
                bad_exc = None
                try:
                    P.run_campaign(plan=plan, launch_authorization=receipt, root=td)
                except NotImplementedError as exc:
                    bad_exc = exc
                except BaseException:
                    pass
                v_idx = events.index("verify") if "verify" in events else -1
                a_idx = events.index("acquire") if "acquire" in events else -1
                check("PV-1c VALID RECEIPT: verify precedes _acquire; _acquire called exactly "
                      "once; NotImplementedError retired from the real path",
                      bad_exc is None and v_idx >= 0 and a_idx >= 0 and v_idx < a_idx
                      and events.count("acquire") == 1,
                      f"events={events} bad_exc={bad_exc!r}")
            finally:
                P.verify_launch_authorization = real_verify
                P._acquire = real_acquire
                PR._http_get = real_http
                urllib.request.urlopen = real_urlopen

    # ---- PV-2: staged, separately-hashed raw objects -------------------------
    ksig = inspect.signature(PR.koeri_fetch).parameters
    ssig = inspect.signature(PR.scedc_fetch).parameters
    if "stage_dir" not in ksig or "stage_dir" not in ssig:
        check("PV-2-GATE fetches persist raw objects (stage_dir seam on koeri_fetch + "
              "scedc_fetch; typed raw_objects result)", False,
              "AWAITING staged-fetch seams -- red-first as authored (codex 0347 P2)")
    else:
        real_http = PR._http_get
        served = {}

        def fake_http(url, timeout):
            for frag, body in list(served.items()):
                if frag in url:
                    if body is None:
                        raise PR.ProviderUnavailable(f"404: {url}")
                    return body
            raise PR.ProviderUnavailable(f"404 (unserved): {url}")

        try:
            PR._http_get = fake_http
            with tempfile.TemporaryDirectory() as td:
                body = b"MSEED-KOERI-" + b"x" * 64
                served.clear()
                served["dataselect"] = body
                res = PR.koeri_fetch("KO.GAZK..HHZ", d0700, d0700e, stage_dir=td)
                objs = res.get("raw_objects") if isinstance(res, dict) else None
                ok_shape = (isinstance(res, dict) and "stream" in res
                            and isinstance(objs, list) and len(objs) == 1)
                o = objs[0] if ok_shape else {}
                staged_ok = (ok_shape and os.path.isfile(o.get("staged_path", ""))
                             and o.get("size_bytes") == len(body)
                             and o.get("sha256") == hashlib.sha256(body).hexdigest()
                             and open(o["staged_path"], "rb").read() == body
                             and "dataselect" in o.get("source", ""))
                check("PV-2a KOERI fetch returns typed raw_objects; exact served bytes staged "
                      "to disk with matching size/sha256 and the exact request URL bound",
                      staged_ok, f"res_type={type(res).__name__} obj={o}")
                if staged_ok:
                    check("PV-2b verify_staged_object: True on the intact object; False after "
                          "a one-byte mutation; False on a missing file",
                          hasattr(PR, "verify_staged_object")
                          and PR.verify_staged_object(o) is True
                          and (lambda: (open(o["staged_path"], "r+b").write(b"Y"),
                                        PR.verify_staged_object(o))[1])() is False
                          and (lambda: (os.remove(o["staged_path"]),
                                        PR.verify_staged_object(o))[1])() is False)

                b090 = b"MSEED-DAY090-" + b"a" * 64
                b091 = b"MSEED-DAY091-" + b"b" * 64
                served.clear()
                served["2026_090"] = b090
                served["2026_091"] = b091
                res2 = PR.scedc_fetch("CI.BOR..BHZ", d0700, d0700e, stage_dir=td)
                objs2 = res2.get("raw_objects") if isinstance(res2, dict) else []
                concat_sha = hashlib.sha256(b090 + b091).hexdigest()
                shas = [x.get("sha256") for x in objs2] if objs2 else []
                ok2 = (isinstance(res2, dict) and len(objs2) == 2
                       and shas == [hashlib.sha256(b090).hexdigest(),
                                    hashlib.sha256(b091).hexdigest()]
                       and concat_sha not in json.dumps(objs2)
                       and all(os.path.isfile(x["staged_path"])
                               and open(x["staged_path"], "rb").read() == body_
                               for x, body_ in zip(objs2, (b090, b091)))
                       and "2026_090" in objs2[0]["source"]
                       and "2026_091" in objs2[1]["source"])
                check("PV-2c SCEDC two touched day-volumes -> TWO separately-hashed staged "
                      "raw_objects in day order; the concatenated digest appears nowhere",
                      ok2, f"objs={objs2}")

                served.clear()
                served["dataselect"] = b"CORRUPT-MSEED"
                staged_before = set(os.listdir(td))
                parse_failed = raises(lambda: PR.koeri_fetch("KO.GAZK..HHZ", d0700, d0700e,
                                                             stage_dir=td))
                staged_after = set(os.listdir(td))
                new_files = staged_after - staged_before
                kept = any(open(os.path.join(td, n), "rb").read() == b"CORRUPT-MSEED"
                           for n in new_files if os.path.isfile(os.path.join(td, n)))
                check("PV-2d PERSIST-BEFORE-PARSE: a parse failure after staging raises but "
                      "the staged raw object survives byte-for-byte",
                      parse_failed and kept, f"new_files={sorted(new_files)}")
        finally:
            PR._http_get = real_http

    # ---- PV-3a: half-open day enumeration -----------------------------------
    two = None
    one = None
    try:
        two = PR._utc_days_spanned(d0700, d0700e)
        one = PR._utc_days_spanned(mid0, mid1)
    except Exception as exc:
        check("PV-3a day enumeration callable", False, repr(exc))
    check("PV-3a HALF-OPEN day enumeration: 07Z->07Z touches two days; an exact-midnight 24 h "
          "window touches exactly ONE day; end<=start refuses",
          two == [date(2026, 3, 31), date(2026, 4, 1)]
          and one == [date(2026, 3, 31)]
          and raises(lambda: PR._utc_days_spanned(mid0, mid0))
          and raises(lambda: PR._utc_days_spanned(mid1, mid0)),
          f"two={two} one={one}")

    # ---- PV-3b: availability probes EVERY touched day ------------------------
    real_urlopen = urllib.request.urlopen
    probed = []

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        probed.append(url)
        if "2026_090" in url:
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
        if "2026_091" in url:
            return _Resp()
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

    try:
        urllib.request.urlopen = fake_urlopen
        probed.clear()
        got = PR.scedc_available("CI", ["BOR"], ["BHZ"], d0700, d0700e)
        second_day_probed = any("2026_091" in u for u in probed)
        check("PV-3b SECOND-DAY-ONLY data registers as available (every touched day probed; "
              "first-day 404 does not mask the candidate)",
              got == {"CI.BOR..BHZ"} and second_day_probed,
              f"got={got} probed={probed}")
        probed.clear()
        got_mid = PR.scedc_available("CI", ["BOR"], ["BHZ"], mid0, mid1)
        check("PV-3c exact-midnight window probes ONLY its single touched day and honestly "
              "reports absence when that object is missing",
              got_mid == set() and all("2026_091" not in u for u in probed)
              and any("2026_090" in u for u in probed),
              f"got={got_mid} probed={probed}")
    finally:
        urllib.request.urlopen = real_urlopen

    # ---- PV-3d (REV 3): KOERI availability-text parsing regression lock ------
    # Fixture lines 2-4 are VERBATIM live service output captured 2026-08-09T05:56Z from
    # eida.koeri.boun.edu.tr/fdsnws/availability/1 — the fixed-width EMPTY Location column
    # collapses under split(), the defect that falsely marked every KOERI station
    # unavailable and aborted the 05:35Z fetch (fixed at 5bc7d2d). Green regression lock,
    # not red-first: it pins the landed fix against the AUTHORITATIVE text shape.
    koeri_text = (
        "#Network Station Location Channel Quality SampleRate Earliest                    "
        "Latest                     \n"
        "KO       BOTS             HHZ     D       100.0      2026-05-01T07:00:00.000000Z "
        "2026-05-01T07:24:42.000000Z\n"
        "KO       SAUV             HHZ     D       100.0      2026-05-01T07:00:00.000000Z "
        "2026-05-01T09:00:00.000000Z\n"
        "KO       GAZK             HHZ     D       100.0      2026-05-01T07:00:00.000000Z "
        "2026-05-01T09:00:00.000000Z\n"
        "KO       TEST     00      HHZ     D       100.0      2026-05-01T07:00:00.000000Z "
        "2026-05-01T09:00:00.000000Z\n"
        "KO       DASH     --      HHZ     D       100.0      2026-05-01T07:00:00.000000Z "
        "2026-05-01T09:00:00.000000Z\n")
    real_http2 = PR._http_get
    try:
        PR._http_get = lambda url, timeout: koeri_text.encode("utf-8")
        got_k = PR.koeri_available("KO", ["BOTS", "SAUV", "GAZK", "TEST", "DASH"],
                                   ["HHZ"], d0700, d0700e)
        check("PV-3d koeri_available parses the REAL availability text: EMPTY location "
              "column -> NET.STA..CHA (never NET.STA.CHA.QUALITY); '--' -> empty; a present "
              "location token is preserved",
              got_k == {"KO.BOTS..HHZ", "KO.SAUV..HHZ", "KO.GAZK..HHZ",
                        "KO.TEST.00.HHZ", "KO.DASH..HHZ"},
              f"got={got_k}")
    finally:
        PR._http_get = real_http2

    # ---- PV-3e (REV 4): _http_get transient-retry semantics (ea4e3e0 lock) ----
    # A ~2,400-request multi-hour campaign guarantees transient drops. Locks: transient
    # network fault -> capped-backoff retry then success; 204/404 no-data -> ProviderUnavailable
    # with NO retry (attempt-row semantics depend on this); persistent fault -> fail-closed
    # ProviderUnavailable after exactly `retries` attempts with the 1,2,4s backoff schedule.
    # Green regression lock (fix landed at ea4e3e0 after the real run died on
    # RemoteDisconnected at ~378 scorings). time.sleep is patched -- the schedule is asserted,
    # not waited out.
    import http.client
    import time as _time
    real_urlopen3 = urllib.request.urlopen
    real_sleep = _time.sleep
    slept, ncalls = [], []

    class _OKResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"BODY"

    try:
        _time.sleep = lambda s: slept.append(s)
        seq = [http.client.RemoteDisconnected("dropped"), _OKResp()]

        def urlopen_seq(req, timeout=None):
            ncalls.append(1)
            item = seq.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        urllib.request.urlopen = urlopen_seq
        body = PR._http_get("http://x/transient", 10)
        ok_a = body == b"BODY" and len(ncalls) == 2 and slept == [1]

        ncalls.clear()
        slept.clear()

        def urlopen_404(req, timeout=None):
            ncalls.append(1)
            raise urllib.error.HTTPError("http://x", 404, "Not Found", None, None)

        urllib.request.urlopen = urlopen_404
        ok_b = (raises(lambda: PR._http_get("http://x/nodata", 10), PR.ProviderUnavailable)
                and len(ncalls) == 1 and slept == [])

        ncalls.clear()
        slept.clear()

        def urlopen_drop(req, timeout=None):
            ncalls.append(1)
            raise ConnectionResetError("reset by peer")

        urllib.request.urlopen = urlopen_drop
        ok_c = (raises(lambda: PR._http_get("http://x/dead", 10), PR.ProviderUnavailable)
                and len(ncalls) == 4 and slept == [1, 2, 4])
        check("PV-3e _http_get retry semantics: transient drop -> retry then body (2 "
              "attempts, backoff [1]); 404 -> ProviderUnavailable with NO retry; persistent "
              "fault -> fail-closed after exactly 4 attempts, backoff [1,2,4]",
              ok_a and ok_b and ok_c,
              f"a={ok_a} b={ok_b} c={ok_c} calls={len(ncalls)} slept={slept}")

        # ---- PV-3f (REV 5, codex 112111Z F2 -- POST-RUN red-first): HTTP status
        # classifier. Retry ONLY transient statuses (408/425/429 + 500-599); a permanent
        # client failure (403 and other 4xx except the no-data pair) is an IMMEDIATE typed
        # ProviderUnavailable -- one call, no sleep, no 4x request multiplication.
        perm_ok = {}
        for code, reason in ((401, "Unauthorized"), (403, "Forbidden")):
            ncalls.clear()
            slept.clear()

            def urlopen_perm(req, timeout=None, _c=code, _r=reason):
                ncalls.append(1)
                raise urllib.error.HTTPError("http://x", _c, _r, None, None)

            urllib.request.urlopen = urlopen_perm
            perm_ok[code] = (raises(lambda: PR._http_get("http://x/denied", 10),
                                    PR.ProviderUnavailable)
                             and len(ncalls) == 1 and slept == [])

        ncalls.clear()
        slept.clear()

        def urlopen_503(req, timeout=None):
            ncalls.append(1)
            raise urllib.error.HTTPError("http://x", 503, "Service Unavailable", None, None)

        urllib.request.urlopen = urlopen_503
        ok_e = (raises(lambda: PR._http_get("http://x/busy", 10), PR.ProviderUnavailable)
                and len(ncalls) == 4 and slept == [1, 2, 4])

        trans_ok = {}
        for code, reason in ((408, "Request Timeout"), (425, "Too Early"),
                             (429, "Too Many Requests")):
            ncalls.clear()
            slept.clear()
            seq_t = [urllib.error.HTTPError("http://x", code, reason, None, None), _OKResp()]

            def urlopen_trans(req, timeout=None, _s=seq_t):
                ncalls.append(1)
                item = _s.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item

            urllib.request.urlopen = urlopen_trans
            trans_ok[code] = (PR._http_get("http://x/paced", 10) == b"BODY"
                              and len(ncalls) == 2 and slept == [1])
        check("PV-3f status classifier (codex 113942Z exact set): 401/403 -> IMMEDIATE "
              "ProviderUnavailable (1 call, no sleep); persistent 503 -> retried fail-closed "
              "(4 calls, [1,2,4]); 408/425/429 -> retried then body (the complete "
              "idempotent-GET transient set)",
              all(perm_ok.values()) and ok_e and all(trans_ok.values()),
              f"perm={perm_ok} e={ok_e} trans={trans_ok}")
    finally:
        urllib.request.urlopen = real_urlopen3
        _time.sleep = real_sleep


main()
print()
if FAILS:
    print(f"D2 STEP-4B PROVIDER RED-KAT FAILURES: {FAILS}")
    sys.exit(1)
print("ALL D2 STEP-4B PROVIDER RED-KATs PASS (gated single driver + staged separately-hashed "
      "raw objects + half-open every-day availability)")
