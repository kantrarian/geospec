#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hermetic KAT for the persistent-scoring-OOM typed-fatal fix (codex 0824). A TRANSIENT
MemoryError retries once after gc and returns the envelope (exactly two scorer calls); a
PERSISTENT MemoryError (second failure) raises ScoringResourceUnavailable and mints NO batch --
host resource exhaustion is never silently relabelled as data/support unavailability. No network,
no real DSP: providers/scoring/clock injected."""
import os
import sys
import tempfile
import types
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import d2_step4b_campaign_run as R  # noqa: E402

FAILS = []


def check(desc, ok, detail=""):
    print(f"    [{'PASS' if ok else 'FAIL'}] {desc}" + (f" - {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(desc)


class _ES:
    def __init__(self):
        self.coverage = 0.9
        self.valid_mask = np.ones(86400, dtype=bool)


class _SD:
    """Injected scorer: raises MemoryError the first `oom` calls, then returns an envelope."""
    class DataUnavailable(Exception):
        pass

    def __init__(self, oom):
        self.calls = 0
        self.oom = oom

    def compute_band_envelope_supported(self, frags, *, session_start_utc, session_seconds,
                                        source_id):
        self.calls += 1
        if self.calls <= self.oom:
            raise MemoryError("injected commit-limit OOM")
        return _ES()


# _fragments_from_stream content is irrelevant to the injected scorer
R._fragments_from_stream = lambda stream: [
    (np.zeros(10), 40.0, datetime(2026, 1, 1, tzinfo=timezone.utc))]

# ---- branch 1: transient OOM -> retry-after-gc returns the envelope (2 calls) ----
sd1 = _SD(oom=1)
es = R._station_series(sd1, object(), "KO.SAUV..HHZ", None)
check("TRANSIENT OOM: one failure then success returns the envelope after EXACTLY two scorer "
      "calls (retry-after-gc)", es is not None and sd1.calls == 2, f"calls={sd1.calls}")

# ---- branch 2: persistent OOM -> ScoringResourceUnavailable (never None) ----------
sd2 = _SD(oom=2)
raised = False
try:
    R._station_series(sd2, object(), "KO.SAUV..HHZ", None)
except R.ScoringResourceUnavailable:
    raised = True
except Exception:  # noqa: BLE001
    pass
check("PERSISTENT OOM: two failures raise ScoringResourceUnavailable (NOT None -> not relabelled "
      "as data unavailability)", raised and sd2.calls == 2, f"raised={raised} calls={sd2.calls}")

# ---- branch 2b: DataUnavailable still maps to None (unchanged) --------------------
sd3 = _SD(oom=0)

def _du(*a, **k):
    raise _SD.DataUnavailable()
sd3.compute_band_envelope_supported = _du
sd3.DataUnavailable = _SD.DataUnavailable
check("DATA unavailable still maps to None (resource vs data cleanly separated)",
      R._station_series(sd3, object(), "KO.SAUV..HHZ", None) is None)

# ---- branch 3: full acquire with a persistently-OOMing station mints NO batch -----
_orig_ss = R._station_series


def _always_oom(SD, stream, source_id, session_start):
    raise R.ScoringResourceUnavailable(f"persistent OOM {source_id}")


class _FakeProv:
    class ProviderUnavailable(Exception):
        pass

    def koeri_available(self, net, stas, chas, s, e):
        return {f"{net}.{st}..{c}" for st in stas for c in chas}

    def scedc_available(self, net, stas, chas, s, e):
        return {f"{net}.{st}..{c}" for st in stas for c in chas}

    def parse_staged(self, p):
        return [p]

    def fetch(self, provider, nslc, s, e, *, stage_dir, **kw):
        import hashlib
        raw = (nslc + s.isoformat()).encode()
        h = hashlib.sha256(raw).hexdigest()
        path = os.path.join(stage_dir, h + ".ms")
        open(path, "wb").write(raw)
        return {"stream": [nslc], "raw_objects": [{"source": "x", "staged_path": path,
                                                   "size_bytes": len(raw), "sha256": h}]}


try:
    R._station_series = _always_oom
    day = R.datetime.strptime("2026-08-09", "%Y-%m-%d")
    d = "2026-04-15"
    end = datetime.strptime(d + "T07:00:13.094647Z", "%Y-%m-%dT%H:%M:%S.%fZ").replace(
        tzinfo=timezone.utc)
    plan = {"activation_reference_day": "2026-08-09", "incident_reference_day": "2026-07-29",
            "scheduled_days": [d], "incident_days": [d], "activation_days": [d],
            "carriers": list(R.ELIGIBLE), "providers": R.PROVIDERS,
            "station_registry": {c: [{"segment_name": "s0", "station_id": "KO.A",
                                      "ordered_nslc_candidates": ["KO.A..HHZ"]},
                                     {"segment_name": "s0", "station_id": "KO.B",
                                      "ordered_nslc_candidates": ["KO.B..HHZ"]}]
                                 for c in R.ELIGIBLE}}
    ledger = {"rows": [{"carrier_key": "istanbul_marmara", "scored_day": d, "status": "REGISTERED",
                        "record_sha256": "c" * 64,
                        "request_start_utc": (end - timedelta(seconds=86400)).strftime(
                            "%Y-%m-%dT%H:%M:%S.%fZ"),
                        "request_end_utc": end.strftime("%Y-%m-%dT%H:%M:%S.%fZ")}]}
    root = tempfile.mkdtemp(prefix="d2s4b_oom_")
    raised3 = False
    try:
        R.acquire(plan, ledger, root, providers=_FakeProv(), receipt={
            "status": "VERIFIED_DIRECT", "in_session_timestamp_utc": "2026-08-09T02:00:00Z",
            "owner_quote_sha256": "a" * 64})
    except R.ScoringResourceUnavailable:
        raised3 = True
    except Exception:  # noqa: BLE001
        pass
    no_batch = not os.path.isfile(os.path.join(root, "batch_manifest.json"))
    check("FULL PATH: a persistent scoring OOM aborts acquire (ScoringResourceUnavailable) and "
          "leaves NO batch_manifest.json", raised3 and no_batch,
          f"raised={raised3} no_batch={no_batch}")
finally:
    R._station_series = _orig_ss

print()
if FAILS:
    print(f"D2 STEP-4B OOM-RESOURCE KAT FAILURES: {FAILS}")
    sys.exit(1)
print("ALL D2 STEP-4B OOM-RESOURCE KATs PASS (transient retry -> envelope; persistent -> "
      "ScoringResourceUnavailable + no batch; data-unavailable still None)")
