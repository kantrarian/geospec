#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HERMETIC pre-fire validation of d2_step4b_campaign_run.acquire against codex's 0123 bar.
No network, no real waveforms: providers + segmented scoring + git are mocked so acquire runs
fast, producing a full 540-row batch with a candidate. Then codex's REAL 0123 acceptance bar is
run against the produced root. Goal: EVIDENCE_BATCH_PASS -> acquire's 11-artifact assembly is
0123-conformant BEFORE any outward fetch. (Scoring itself is the accepted diagnostic-replay
pipeline; here it is stubbed to isolate the assembly.)"""
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import types
from datetime import datetime, timedelta, timezone

SRC = "E:/GeoSpec/geospec_sprint/monitoring/src"
sys.path.insert(0, SRC)
import numpy as np  # noqa: E402
import d2_step4b_producer as P  # noqa: E402
import d2_step4b_campaign_run as R  # noqa: E402

U = timezone.utc
DIAG = json.load(open(R.DIAGNOSTIC_FIXTURE, encoding="utf-8"))


# ---- synthetic plan + ledger (3 carriers x 2 seg x 2 sta; 90+90 days) -------
def schedule(ref):
    r = datetime.strptime(ref, "%Y-%m-%d").date()
    return [(r - timedelta(days=120) + timedelta(days=i)).isoformat() for i in range(90)]


incident_days = schedule("2026-07-29")
activation_days = schedule("2026-08-09")
scheduled = sorted(set(incident_days) | set(activation_days))
STA = {
    "istanbul_marmara": {"izmit": [("KO.SAUV..HHZ",), ("KO.GAZK..HHZ",)],
                         "marmara_west": [("KO.NMR6..HHZ",), ("KO.BOTS..HHZ",)]},
    "socal_coachella": {"coachella_south": [("CI.BOR..BHZ",), ("CI.CTW..BHZ",)],
                        "brawley_seismic_zone": [("CI.BC3..BHZ",), ("CI.RXH..BHZ",)]},
    "turkey_kahramanmaras": {"kahramanmaras": [("KO.KHMN..HHZ",), ("KO.NURH..HHZ",)],
                             "malatya": [("KO.MLTY..HHZ",), ("KO.SVRC..HHZ",)]},
}
station_registry = {}
for c, segs in STA.items():
    rows = []
    for seg, stations in segs.items():
        for cand in stations:
            rows.append({"segment_name": seg, "station_id": ".".join(cand[0].split(".")[:2]),
                         "ordered_nslc_candidates": list(cand)})
    station_registry[c] = rows
plan = {"schema": "geospec-d2-step4b-campaign-plan-v1", "contract_id": R.CONTRACT_ID,
        "registered_utc": "2026-08-09T03:07:19.713711Z",
        "activation_reference_day": "2026-08-09", "incident_reference_day": "2026-07-29",
        "carriers": list(R.ELIGIBLE), "providers": R.PROVIDERS, "station_registry": station_registry,
        "incident_days": incident_days, "activation_days": activation_days,
        "scheduled_days": scheduled, "acquisition_order": "KOERI_FIRST_THEN_SCEDC",
        "free_sources_only": True, "outcomes_inspected_before_schedule": False}

def _git_blob(raw):
    return hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()


# one self-consistent publication record per day (shared across carriers)
day_record = {}
for d in scheduled:
    raw = json.dumps({"date": d}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    day_record[d] = {"raw": raw, "sha256": hashlib.sha256(raw).hexdigest(),
                     "git_blob": _git_blob(raw)}
rows = []
for c in R.ELIGIBLE:
    for d in scheduled:
        end = datetime.strptime(d, "%Y-%m-%d").replace(hour=7, tzinfo=U)
        start = end - timedelta(seconds=86400)
        rec = day_record[d]
        rows.append({"carrier_key": c, "scored_day": d, "status": "REGISTERED",
                     "publication_commit": "0" * 40, "publication_repo_path": "docs/ensemble_latest.json",
                     "publication_record_artifact": f"publication_records/{d}.json",
                     "record_git_blob": rec["git_blob"], "record_sha256": rec["sha256"],
                     "request_start_utc": start.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                     "request_end_utc": end.strftime("%Y-%m-%dT%H:%M:%S.%fZ"), "reason_codes": []})
ledger = {"schema": "geospec-d2-published-phase-ledger-v1", "rows": rows}


# ---- stub providers + scoring + git ----------------------------------------
class _Stream(list):
    pass


def _mk_providers():
    m = types.ModuleType("providers_stub")
    m.ProviderUnavailable = P and Exception
    m.ProviderUnavailable = type("ProviderUnavailable", (Exception,), {})

    def koeri_available(net, stas, chas, s, e):
        return {f"{net}.{st}..{ch}" for st in stas for ch in chas}

    def scedc_available(net, stas, chas, s, e):
        return {f"{net}.{st}..{ch}" for st in stas for ch in chas}

    def fetch(provider, nslc, s, e, *, stage_dir, **kw):
        raw = ("MSEED-" + nslc + "-" + s.isoformat()).encode()
        digest = hashlib.sha256(raw).hexdigest()
        path = os.path.join(stage_dir, digest + ".ms")
        with open(path, "wb") as fh:
            fh.write(raw)
        return {"stream": _Stream([nslc]),
                "raw_objects": [{"source": f"http://x/{nslc}", "staged_path": path,
                                 "size_bytes": len(raw), "sha256": digest}]}

    def parse_staged(staged_path):     # H3 seam (mocked _fragments_from_stream ignores content)
        return [staged_path]
    m.koeri_available = koeri_available
    m.scedc_available = scedc_available
    m.fetch = fetch
    m.parse_staged = parse_staged
    return m


class _ES:
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.coverage = 0.95
        self.valid_mask = np.ones(86400, dtype=bool)
        self._v = rng.standard_normal(86400)


def _install_scoring():
    import seismic_data as SD
    import fault_correlation as FC

    R._fragments_from_stream = lambda stream: [
        (np.zeros(1000), 40.0, datetime(2026, 1, 1, tzinfo=U))]
    SD._orig_cbes = SD.compute_band_envelope_supported

    def cbes(frags, *, session_start_utc, session_seconds, source_id):
        return _ES(hash(source_id) % 10000)
    SD.compute_band_envelope_supported = cbes
    if not hasattr(SD, "DataUnavailable"):
        SD.DataUnavailable = type("DataUnavailable", (Exception,), {})

    def agg(st_series):
        if not st_series:
            return None
        a = _ES(0)
        a.valid_mask = np.ones(86400, dtype=bool)
        a._v = np.mean([s._v for s in st_series], axis=0)
        return a
    FC.aggregate_segment_supported = agg

    def ccm(seg_series, seg_names):
        if len(seg_series) < 2:
            return None, list(seg_names), ["INSUFFICIENT_SEGMENTS"]
        V = np.vstack([s._v for s in seg_series])
        C = np.corrcoef(V)
        return C, list(seg_names), []
    FC.compute_correlation_matrix_supported = ccm

    class _Mon:
        def __init__(self, **k):
            pass

        def analyze_eigenvalue_spectrum(self, C):
            w = np.linalg.eigvalsh(np.asarray(C))[::-1]
            pr = (w.sum() ** 2) / (np.square(w).sum())
            return w, None, pr, None
    FC.FaultCorrelationMonitor = _Mon


def main():
    providers = _mk_providers()
    _install_scoring()
    R._git_head = lambda repo=R.REPO: "a" * 40
    R._git_clean = lambda repo=R.REPO: True
    receipt = {"status": "VERIFIED_DIRECT", "in_session_timestamp_utc": "2026-08-09T02:04:49Z",
               "owner_quote_sha256": "0658bdf0b498b551c433bb3f932a87a9c06e28929703c22d9468507b1fc7d3f8"}

    root = tempfile.mkdtemp(prefix="d2s4b_hermetic_")
    # stage campaign_plan.json + published_phase_ledger.json + publication_records
    with open(os.path.join(root, "campaign_plan.json"), "wb") as fh:
        fh.write(R._canon(plan))
    with open(os.path.join(root, "published_phase_ledger.json"), "wb") as fh:
        fh.write(R._canon(ledger))
    os.makedirs(os.path.join(root, "publication_records"), exist_ok=True)
    for d in scheduled:
        with open(os.path.join(root, "publication_records", f"{d}.json"), "wb") as fh:
            fh.write(day_record[d]["raw"])

    # Pin the acquire clock to the activation reference day so this hermetic self-test is
    # date-independent (the default live clock made campaign_started_utc drift off the fixed
    # activation_reference_day after a date rollover -> a spurious "activation day != campaign
    # start UTC day" 0123 failure unrelated to acquisition).
    _t = {"v": datetime(2026, 8, 9, 12, 0, 0, tzinfo=U)}

    def _pinned_clock():
        _t["v"] = _t["v"] + timedelta(microseconds=1)
        return _t["v"]

    summary = R.acquire(plan, ledger, root, providers=providers, receipt=receipt,
                        clock=_pinned_clock)
    print("acquire ->", {k: summary[k] for k in ("status", "candidates", "attempts", "daily_rows",
                                                  "clean_tree")})

    # run codex's REAL 0123 bar against the produced root
    barpath = os.environ.get(
        "D2_0123_BAR", "C:/agent-framework/inbox/geospec_japan/"
        "2026-08-09-0123-codex-grassmann-cayley-d2-step4b-acceptance.py")
    spec = importlib.util.spec_from_file_location("bar0123", barpath)
    bar = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bar)
    print("\n===== 0123 acceptance bar vs produced root =====")
    rc = bar.main(root)
    print("0123 RC =", rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
