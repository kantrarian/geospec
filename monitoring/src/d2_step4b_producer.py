#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""d2_step4b_producer.py — D2 step-4b archive-calibration campaign PRODUCER.

Implements the seam contract of `monitoring/src/test_d2_step4b_redkats_cayley.py` (cayley
`65755de`, SB-0..8) under contract `codex-d2-step4b-2026-08-09-v1` (codex 0129); the produced
BATCH is separately verified by codex's 0123 acceptance entry point. Pinned segmented scoring
implementation: GeoSpec `3950a2c`.

Nothing here lifts a freeze, admits calibration values, tunes a rule, promotes a registry,
deploys, publishes, or makes a claim. The campaign is outcome-blind and may honestly mint 0..3
candidates. The FIRST archive request additionally requires a DIRECT verifiable owner launch go
bound in `run_campaign` (SB-8): `run_campaign` refuses — before ANY provider I/O — without a
`VERIFIED_DIRECT` receipt. This module cannot and does not mint that receipt.
"""
import hashlib
import json
import math
import os
import re
from datetime import datetime, timedelta, timezone

# ---- SB-1: frozen campaign constants ---------------------------------------
CAMPAIGN = {
    "contract_id": "codex-d2-step4b-2026-08-09-v1",
    "incident_reference": "2026-07-29",
    "min_admitted_days": 60,
    "window_days": 90,
    "lag_days": 30,
    "providers": {
        "istanbul_marmara": "eida.koeri.boun.edu.tr",
        "turkey_kahramanmaras": "eida.koeri.boun.edu.tr",
        "socal_coachella": "s3://scedc-pds",
    },
}

_HEX64 = re.compile(r"^[0-9a-f]{64}$")

# ---- SB-4 v2 / SB-7 v2 frozen constants (cayley REV 2 d1a58b3) --------------
# Exact carrier -> published-region mapping (codex phasebar 0313).
REGION_KEY = {
    "istanbul_marmara": "istanbul_marmara",
    "socal_coachella": "socal_saf_coachella",
    "turkey_kahramanmaras": "turkey_kahramanmaras",
}
# The accepted sealed diagnostic-result artifact (codex retention PASS 0043); pinned DIRECTLY,
# with no caller-supplied override / expected-sha path (SB-7b).
DIAGNOSTIC_RESULT_SHA256 = "ee75e449aa0b1003a3cf047432a91a9adc1db4c7497b1e1d9d47f01d552a4b35"
# A naive-or-Z serialized instant with up to 6 fractional digits; any explicit offset refuses.
_NAIVE_OR_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z?$")


def _canonical(obj) -> bytes:
    """Canonical JSON bytes: sorted keys, compact separators, UTF-8 (one logical value)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _parse_day(s: str):
    return datetime.strptime(s, "%Y-%m-%d").date()


def _parse_iso_utc(s):
    """Parse an ISO-8601 instant that MUST be aware UTC (trailing 'Z' tolerated)."""
    txt = str(s)
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    dt = datetime.fromisoformat(txt)
    if dt.tzinfo is None or dt.utcoffset() != timedelta(0):
        raise ValueError(f"timestamp {s!r} is not aware UTC")
    return dt


# ---- SB-2: exact arm schedules ---------------------------------------------
def schedule_days(reference_day: str):
    """The exact half-open [ref-120d, ref-30d) = 90 ascending ISO calendar days.
    (ref - (lag_days + window_days)) .. (ref - lag_days), left-closed / right-open."""
    ref = _parse_day(reference_day)
    lag = int(CAMPAIGN["lag_days"])          # 30
    win = int(CAMPAIGN["window_days"])       # 90
    start = ref - timedelta(days=lag + win)  # ref - 120d
    return [(start + timedelta(days=i)).isoformat() for i in range(win)]


# ---- SB-3: outcome-blind deterministic plan --------------------------------
def build_campaign_plan(carriers: dict, activation_reference: str) -> dict:
    """Freeze the outcome-blind campaign plan. Fail-closed: >= 2 stations/segment, >= 2
    segments/carrier, provider only from CAMPAIGN['providers']. NO outcome-bearing field is
    accepted or emitted. Deterministic: identical inputs -> byte-identical canonical JSON."""
    providers = CAMPAIGN["providers"]
    if not isinstance(carriers, dict) or not carriers:
        raise ValueError("carriers must be a non-empty mapping")
    plan_carriers = {}
    for carrier, segments in carriers.items():
        if carrier not in providers:
            raise ValueError(f"carrier {carrier!r} is outside the provider cap "
                             f"{sorted(providers)} (no expansion)")
        if not isinstance(segments, dict) or len(segments) < 2:
            raise ValueError(f"carrier {carrier!r} needs >= 2 segments")
        seg_out = {}
        for seg, stations in segments.items():
            if not isinstance(stations, list) or len(stations) < 2:
                raise ValueError(f"segment {carrier}/{seg} needs >= 2 stations")
            frozen = []
            for cand_list in stations:
                if not isinstance(cand_list, list) or not cand_list:
                    raise ValueError(f"station in {carrier}/{seg} needs an ordered NSLC "
                                     f"candidate list")
                frozen.append([str(c) for c in cand_list])   # ORDERED NSLC candidates, verbatim
            seg_out[seg] = frozen
        plan_carriers[carrier] = {"provider": providers[carrier], "segments": seg_out}
    return {
        "contract_id": CAMPAIGN["contract_id"],
        "incident_reference": CAMPAIGN["incident_reference"],
        "activation_reference": str(activation_reference),
        "arms": {
            "incident": {"reference": CAMPAIGN["incident_reference"],
                         "days": schedule_days(CAMPAIGN["incident_reference"])},
            "activation": {"reference": str(activation_reference),
                           "days": schedule_days(str(activation_reference))},
        },
        "carriers": plan_carriers,
    }


def plan_digest(plan) -> str:
    """64-hex SHA-256 over the canonical plan bytes."""
    return hashlib.sha256(_canonical(plan)).hexdigest()


# ---- SB-4 v2: published-end-anchored session binding (codex phasebar 0313) --
def _parse_published_end(value):
    """A naive-or-Z serialized instant interpreted as UTC (microseconds preserved). Any explicit
    non-Z offset / epoch / prose refuses. Returns aware-UTC."""
    if not isinstance(value, str) or not _NAIVE_OR_Z.fullmatch(value):
        raise ValueError(f"published region.date has unsupported form: {value!r}")
    text = value[:-1] if value.endswith("Z") else value
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


def session_from_record(record_bytes: bytes, *, carrier: str, scored_day: str):
    """published-end-anchored-segmented-v2 (codex phasebar 0313). The bound (start, end) for a
    (carrier, scored_day) comes ONLY from the portable public daily-monitoring record's
    region.date: require top-level record["date"] == scored_day; the carrier's REGION_KEY region
    present; its components.fault_correlation.available EXACTLY True; region["date"] (naive-or-Z,
    interpreted UTC, microseconds preserved) as the request END, which must lie on scored_day;
    request START = END - exactly 86,400 s. A legacy/decoy request_interval in the record is
    IGNORED; no midnight/nearest/cache/inferred fallback exists. Returns aware-UTC (start, end)."""
    if carrier not in REGION_KEY:
        raise ValueError(f"unknown carrier {carrier!r}")
    try:
        rec = json.loads(record_bytes.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"published record is not valid JSON: {exc}")
    if not isinstance(rec, dict):
        raise ValueError("published record is not a JSON object")
    if rec.get("date") != scored_day:
        raise ValueError(f"top-level record.date {rec.get('date')!r} != scored_day {scored_day!r}")
    regions = rec.get("regions")
    region = regions.get(REGION_KEY[carrier]) if isinstance(regions, dict) else None
    if not isinstance(region, dict):
        raise ValueError(f"carrier {carrier!r} mapped region {REGION_KEY[carrier]!r} absent")
    components = region.get("components")
    fc = components.get("fault_correlation") if isinstance(components, dict) else None
    if not (isinstance(fc, dict) and fc.get("available") is True):
        raise ValueError("region fault_correlation.available is not exactly True")
    end = _parse_published_end(region.get("date"))
    if end.date().isoformat() != scored_day:
        raise ValueError(f"published end {end} is not on the scored day {scored_day}")
    start = end - timedelta(seconds=86400)
    return start, end


# ---- SB-5: frozen NSLC selection -------------------------------------------
def select_channel(candidates, available):
    """The FIRST candidate present in `available`, in the frozen order; None if none. No
    re-sort, no post-QC swap (QC failure of fetched data never re-enters selection)."""
    for cand in candidates:
        if cand in available:
            return cand
    return None


# ---- SB-6: nearest-rank lower-5% quantile ----------------------------------
def threshold_from_admitted(ratios):
    """None below the MIN_ADMITTED_DAYS floor; else sort ascending and return the nearest-rank
    lower 5% quantile ratios_sorted[ceil(0.05*n)-1] (zero-based). Input-order-invariant."""
    n = len(ratios)
    if n < int(CAMPAIGN["min_admitted_days"]):
        return None
    ordered = sorted(float(x) for x in ratios)
    idx = math.ceil(0.05 * n) - 1
    return ordered[idx]


# ---- SB-7 v2: replay derived ONLY from the accepted sealed artifact bytes --
def _matrix_digest(matrix):
    """sha256 hex of the canonical JSON (sorted keys, ',:' separators) of a correlation matrix."""
    return hashlib.sha256(json.dumps(matrix, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def parse_diagnostic_results(doc: dict) -> dict:
    """Derive per-carrier replay values FROM the accepted diagnostic-result doc's bytes-parsed
    content (codex 0300 F2). Require results for EXACTLY the three campaign carriers; both phases
    status == "OK"; the ratio RECOMPUTED as ordered_eigenvalues[1]/[0] (descending) matching the
    stored lambda2_lambda1 within rel/abs 1e-6 (mismatch = tamper -> ValueError). Returns per
    carrier {incident_ratio, control_ratio, incident_common_support, control_common_support,
    incident_matrix_digest, control_matrix_digest}."""
    if not isinstance(doc, dict):
        raise ValueError("diagnostic result is not a JSON object")
    results = doc.get("results")
    if not isinstance(results, dict):
        raise ValueError("diagnostic result has no 'results' mapping")
    if set(results) != set(REGION_KEY):
        raise ValueError(f"diagnostic must cover exactly {sorted(REGION_KEY)}; "
                         f"got {sorted(results)}")
    out = {}
    for carrier in REGION_KEY:
        carrier_res = results.get(carrier)
        if not isinstance(carrier_res, dict):
            raise ValueError(f"{carrier} results missing")
        entry = {}
        for phase in ("incident", "control"):
            rec = carrier_res.get(phase)
            if not isinstance(rec, dict):
                raise ValueError(f"{carrier}/{phase} record missing")
            if rec.get("status") != "OK":
                raise ValueError(f"{carrier}/{phase} status {rec.get('status')!r} != OK")
            eig = rec.get("ordered_eigenvalues")
            if not (isinstance(eig, list) and len(eig) >= 2 and float(eig[0]) != 0.0):
                raise ValueError(f"{carrier}/{phase} ordered_eigenvalues unusable")
            recomputed = float(eig[1]) / float(eig[0])
            stored = rec.get("lambda2_lambda1")
            if not (isinstance(stored, (int, float))
                    and math.isclose(recomputed, float(stored), rel_tol=1e-6, abs_tol=1e-6)):
                raise ValueError(f"{carrier}/{phase} ratio tamper: recomputed {recomputed} "
                                 f"!= stored {stored}")
            entry[f"{phase}_ratio"] = float(stored)
            entry[f"{phase}_common_support"] = rec.get("common_support_count")
            entry[f"{phase}_matrix_digest"] = _matrix_digest(rec.get("correlation_matrix"))
        out[carrier] = entry
    return out


def derive_replay_ratios(diagnostic_result_bytes: bytes) -> dict:
    """Verify the diagnostic-result bytes against the DIRECT pin, then derive the sealed replay
    values from the bytes (codex 0300 F2). There is NO override / expected-sha parameter, and
    producer-entered values have no path in; `prior_evidence.json` stays a receipt wrapper only."""
    actual = hashlib.sha256(diagnostic_result_bytes).hexdigest()
    if actual != DIAGNOSTIC_RESULT_SHA256:
        raise ValueError(f"diagnostic-result sha256 {actual} != pin {DIAGNOSTIC_RESULT_SHA256}")
    doc = json.loads(diagnostic_result_bytes.decode("utf-8"))
    return parse_diagnostic_results(doc)


# ---- SB-7c: deterministic candidate rule -----------------------------------
def admit_candidate(carrier, incident_summary, activation_summary, replay):
    """(status, info). ADMITTED_CANDIDATE iff BOTH arms have >= 60 admitted days AND reproducible
    thresholds AND replay carries both ratios AND incident_ratio >= incident_threshold AND
    control_ratio >= incident_threshold. Otherwise one deterministic BLOCKED_* status."""
    min_days = int(CAMPAIGN["min_admitted_days"])
    inc_days = int(incident_summary.get("admitted_days", 0))
    act_days = int(activation_summary.get("admitted_days", 0))
    inc_thr = incident_summary.get("threshold")
    act_thr = activation_summary.get("threshold")
    if inc_days < min_days or act_days < min_days or inc_thr is None or act_thr is None:
        return "BLOCKED_INSUFFICIENT_CALIBRATION", {
            "carrier": carrier, "incident_admitted_days": inc_days,
            "activation_admitted_days": act_days, "min_admitted_days": min_days,
            "incident_threshold": inc_thr, "activation_threshold": act_thr}
    if not (isinstance(replay, dict) and "incident_ratio" in replay
            and "control_ratio" in replay):
        return "BLOCKED_REPLAY_UNAVAILABLE", {"carrier": carrier, "replay": replay}
    inc_ratio = float(replay["incident_ratio"])
    ctrl_ratio = float(replay["control_ratio"])
    inc_threshold = float(inc_thr)
    if inc_ratio < inc_threshold:
        return "BLOCKED_ARTIFACT_PERSISTS", {
            "carrier": carrier, "incident_ratio": inc_ratio, "incident_threshold": inc_threshold}
    if ctrl_ratio < inc_threshold:
        return "BLOCKED_NEGATIVE_CONTROL", {
            "carrier": carrier, "control_ratio": ctrl_ratio, "incident_threshold": inc_threshold}
    return "ADMITTED_CANDIDATE", {
        "carrier": carrier, "incident_ratio": inc_ratio, "control_ratio": ctrl_ratio,
        "incident_threshold": inc_threshold, "activation_threshold": float(act_thr),
        "artifact_removed": True, "control_clear": True}


# ---- Ridgecrest t2 extension seams (cayley t2 supplement; codex 0300 F3) ---
# Registers the outcome-blind little_lake redraw (shared CI.LRL/CI.WBS -> unshared CI.WBM/CI.DTP)
# per docs/AMENDMENT_2026-08-09_ridgecrest_t2_topology.md. This mints a redraw ONLY: the base
# three-carrier policy is unchanged (build_campaign_plan still refuses 'ridgecrest') and
# admit_candidate('ridgecrest', ...) with no replay evidence stays BLOCKED_REPLAY_UNAVAILABLE.
RIDGECREST_T2_REGISTRY_SHA256 = \
    "449273b866f682d1363806daef5509cac40f1480d003a7fd0731b71a365f2657"


def load_ridgecrest_t2(registry_bytes: bytes) -> dict:
    """Verify the redraw-registry bytes against the amendment pin, then parse. There is NO
    unpinned load path: a non-bytes input OR any byte difference refuses (ValueError)."""
    if not isinstance(registry_bytes, (bytes, bytearray)):
        raise ValueError("ridgecrest-t2 registry must be pinned bytes, not a caller object")
    actual = hashlib.sha256(bytes(registry_bytes)).hexdigest()
    if actual != RIDGECREST_T2_REGISTRY_SHA256:
        raise ValueError(f"ridgecrest-t2 registry sha256 {actual} != pin "
                         f"{RIDGECREST_T2_REGISTRY_SHA256}")
    return json.loads(bytes(registry_bytes).decode("utf-8"))


def build_ridgecrest_extension_plan(activation_reference: str, registry_bytes: bytes) -> dict:
    """PUBLIC builder (codex 0520 F1): consumes the PINNED registry BYTES only -- it verifies them
    itself via load_ridgecrest_t2 (a plain dict, however valid-looking, or any re-encoded
    candidate mutation refuses) and then builds the plan. There is NO public raw-dict builder."""
    registry = load_ridgecrest_t2(registry_bytes)
    return _build_ridgecrest_extension_plan(activation_reference, registry)


def _build_ridgecrest_extension_plan(activation_reference: str, registry: dict) -> dict:
    """Private structural builder (fail-closed checks on the ALREADY-attested registry). Not a
    public path: only build_ridgecrest_extension_plan (bytes-only) reaches it. Fail-closed on:
    wrong region/topology_version/provider; fewer than 2 segments; fewer than 2 stations in any
    segment; any NET.STA shared across segments. No outcome-bearing field is emitted; byte-
    identical canonical JSON on identical inputs (plan_digest-able). Registers a redraw only."""
    if not isinstance(registry, dict):
        raise ValueError("registry must be a mapping")
    if registry.get("region") != "ridgecrest":
        raise ValueError(f"registry region {registry.get('region')!r} != 'ridgecrest'")
    if registry.get("topology_version") != "t2":
        raise ValueError(f"registry topology_version {registry.get('topology_version')!r} != 't2'")
    if registry.get("provider") != "s3://scedc-pds":
        raise ValueError(f"registry provider {registry.get('provider')!r} != 's3://scedc-pds'")
    segments = registry.get("segments")
    if not isinstance(segments, dict) or len(segments) < 2:
        raise ValueError("ridgecrest-t2 registry needs >= 2 segments")
    seen_netsta, seg_out = {}, {}
    for seg, body in segments.items():
        stations = body.get("stations") if isinstance(body, dict) else None
        if not isinstance(stations, dict) or len(stations) < 2:
            raise ValueError(f"segment {seg!r} needs >= 2 stations")
        st_out = {}
        for st, sbody in stations.items():
            netsta = ".".join(str(st).split(".")[:2])
            if netsta in seen_netsta and seen_netsta[netsta] != seg:
                raise ValueError(f"NET.STA {netsta} shared across segments "
                                 f"{seen_netsta[netsta]!r} and {seg!r}")
            seen_netsta[netsta] = seg
            candidates = sbody.get("nslc_candidates") if isinstance(sbody, dict) else None
            if not isinstance(candidates, list) or not candidates:
                raise ValueError(f"station {st!r} in {seg!r} needs an nslc_candidate list")
            st_out[st] = {"nslc_candidates": [str(c) for c in candidates]}  # frozen order verbatim
        seg_out[seg] = {"stations": st_out}
    return {
        "carrier": "ridgecrest", "topology_version": "t2", "provider": "s3://scedc-pds",
        "incident_arm": schedule_days(CAMPAIGN["incident_reference"]),
        "activation_arm": schedule_days(str(activation_reference)),
        "segments": seg_out,
    }


# ---- SB-8: the direct-owner launch gate ------------------------------------
def verify_launch_authorization(receipt) -> bool:
    """True iff receipt == {status: 'VERIFIED_DIRECT', in_session_timestamp_utc: <aware-UTC>,
    owner_quote_sha256: <64-hex>}. RELAYED / missing / malformed -> False. This formalizes
    grassmann's session-level consent classifier; it cannot substitute for the owner's direct
    in-session word (which mints the receipt)."""
    if not isinstance(receipt, dict):
        return False
    if set(receipt.keys()) != {"status", "in_session_timestamp_utc", "owner_quote_sha256"}:
        return False
    if receipt.get("status") != "VERIFIED_DIRECT":
        return False
    try:
        _parse_iso_utc(receipt.get("in_session_timestamp_utc"))
    except Exception:
        return False
    quote = receipt.get("owner_quote_sha256")
    return isinstance(quote, str) and bool(_HEX64.match(quote))


# ---- P1 (codex 0347): run_campaign is the single receipt-gated real driver -
# run_campaign stashes the VERIFIED receipt here for _acquire's batch_manifest: the PV-spied
# _acquire(plan, ledger, root) signature carries no receipt slot, so this side channel (set only
# after the gate passes) is how the verified owner-launch receipt reaches the batch assembler.
_RECEIPT_HOLDER = {}


def _load_ledger(root):
    """Load the staged published_phase_ledger.json from the campaign root. A local JSON read only
    (no provider I/O); reached only after the launch gate succeeds."""
    with open(os.path.join(root, "published_phase_ledger.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_plan(root):
    """Reopen the staged campaign_plan.json from the campaign root (H2: the acquired plan IS the
    staged plan). Local JSON read only; reached only after the launch gate succeeds."""
    with open(os.path.join(root, "campaign_plan.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _acquire(plan, ledger, root):
    """The ONLY acquisition path (codex 0347 P1). Lazily imports the provider stack — keeping it
    out of the producer's import graph until strictly below the launch gate — and runs the full
    published-phase-bound campaign: per REGISTERED (carrier, day) fetch (KOERI first, then SCEDC)
    -> segmented score (pinned GeoSpec 3950a2c) -> the 0123 batch artifacts staged under `root`.
    Reached only after verify_launch_authorization succeeds and the plan is bound; called exactly
    once by run_campaign. The verified receipt + injectable clock ride the side-channel holder."""
    import d2_step4b_providers as providers        # lazy: below the gate, off the import graph
    import d2_step4b_campaign_run as campaign_run
    return campaign_run.acquire(plan, ledger, root, providers=providers,
                                receipt=_RECEIPT_HOLDER.get("receipt"),
                                clock=_RECEIPT_HOLDER.get("clock"))


def _writer_lock_path(root):
    """The single-writer lock lives at the SIBLING path `<canonical root> + '.writer.lock'` — never
    inside the root, so it can never be mistaken for a campaign artifact (H6b). H6d (codex 1339):
    the lock IDENTITY is derived from the CANONICAL physical root — normcase(realpath(abspath(...)))
    — not the caller's raw path spelling, so a trailing-separator / symlink / junction alias of the
    same root resolves to the SAME lock file. Raw string concatenation would let `root + os.sep`
    mint a second lock (and drop it INSIDE the root), defeating the same-root single-writer
    invariant; canonicalization collapses every alias to one lock identity."""
    canonical_root = os.path.normcase(
        os.path.realpath(os.path.abspath(os.fspath(root)))
    )
    return canonical_root + ".writer.lock"


def _acquire_writer_lock(root):
    """H6 (codex 112111Z F1): take an OS-level, non-blocking, EXCLUSIVE byte-range lock on the
    sibling lock file, returning the open fd to hold for the campaign's duration. The lock is a
    KERNEL lock bound to the file handle (msvcrt.locking here / fcntl.flock elsewhere), so:
      * a second same-root live invocation cannot take it -> SystemExit (single-writer), and
      * a RESIDUAL lock file left by a dead run has no live holder, so the kernel grants the lock
        and the fresh run proceeds — never an existence check, never an age break (H6c).
    The file is opened O_CREAT without O_TRUNC: its bytes are irrelevant; the invariant is the
    kernel lock on the handle, and truncating could clobber a concurrent holder's file."""
    fd = os.open(_writer_lock_path(root), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if os.name == "nt":
            import msvcrt
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)          # non-blocking exclusive; OSError if held
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        raise SystemExit("run_campaign REFUSED: another live writer holds the same-root "
                         "single-writer lock (%s) — no _acquire call, no provider I/O."
                         % _writer_lock_path(root))
    return fd


def _release_writer_lock(fd):
    """Release the kernel lock and close the handle. Closing the handle alone releases the kernel
    lock, so an explicit-unlock failure is non-fatal (the `finally` close is the real guarantee)."""
    try:
        if os.name == "nt":
            import msvcrt
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    except OSError:
        pass
    finally:
        os.close(fd)


def run_campaign(plan, launch_authorization, root=None, *, dry_run=False, clock=None,
                 providers=None, **kwargs):
    """The single real driver for the step-4b campaign (codex 0347 P1 + 0447 H2). Executable
    order: verify the VERIFIED_DIRECT owner receipt FIRST — invalid/missing raises SystemExit
    before ANY provider import/I/O and never reaches `_acquire` — then REOPEN
    `root/campaign_plan.json` and, if a caller-supplied plan differs from it, refuse (SystemExit,
    zero `_acquire`); the REOPENED staged plan is the acquisition authority (`plan=None` is
    allowed). Then load the ledger and drive acquisition through `_acquire` exactly once. The
    injectable `clock` (default: live now()) reaches the assembler via the side channel.

    RESUME SEAM (cayley resume bar `3a829cd`): `providers=None` is the production default and this
    function is byte-for-byte unchanged (the real provider stack is lazily imported below the gate
    by `_acquire`; the existing SB/PV/executor bars keep passing). When a `providers` object is
    INJECTED, the same receipt gate runs FIRST, then control passes to the resume-aware executor
    (`d2_step4b_resume.run_resume_campaign`) which is automatic-from-root-state: a COMPLETED root
    is verified and returned immutably; plan/ledger drift on a resumable root refuses before any
    provider I/O; PROCESS_STARTED is committed under the writer lock before the first probe."""
    if not verify_launch_authorization(launch_authorization):
        raise SystemExit("run_campaign REFUSED: no VERIFIED_DIRECT owner launch authorization "
                         "— no archive request issued, no provider I/O performed.")
    if providers is not None:
        # Injected-providers resume seam: the receipt gate above has already passed; the resume
        # executor owns the completed-root/drift checks + writer lock + PROCESS_STARTED ordering.
        import d2_step4b_resume as _resume
        staged_plan = _load_plan(root)                      # H2: reopen the staged plan
        if plan is not None and plan_digest(plan) != plan_digest(staged_plan):
            raise SystemExit("run_campaign REFUSED: caller plan differs from the staged "
                             "campaign_plan.json — no provider I/O (H2).")
        return _resume.run_resume_campaign(staged_plan, root, providers=providers,
                                           receipt=launch_authorization, clock=clock)
    staged_plan = _load_plan(root)                          # H2: reopen the staged plan
    if plan is not None and plan_digest(plan) != plan_digest(staged_plan):
        raise SystemExit("run_campaign REFUSED: caller plan differs from the staged "
                         "campaign_plan.json — no _acquire call, no provider I/O (H2).")
    plan = staged_plan                                      # reopened authority (H2)
    plan_sha = plan_digest(plan)
    if dry_run:
        return {"status": "LAUNCH_AUTHORIZED", "dry_run": True, "plan_digest": plan_sha}
    # H6 (codex 112111Z F1): same-root single-writer. Take the sibling kernel lock AFTER the
    # receipt + staged-plan gates and hold it through final manifest publication (the whole
    # _acquire), releasing in `finally`. A concurrent same-root live invocation refuses here
    # (SystemExit) before reaching _acquire; a residual lock with no live holder never blocks.
    _writer_fd = _acquire_writer_lock(root)
    try:
        _RECEIPT_HOLDER["receipt"] = launch_authorization   # side channel to the batch assembler
        _RECEIPT_HOLDER["clock"] = clock
        ledger = _load_ledger(root)
        return _acquire(plan, ledger, root)
    finally:
        _release_writer_lock(_writer_fd)
