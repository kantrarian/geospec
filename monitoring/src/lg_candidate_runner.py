"""LG candidate runner wiring: the integration boundary for the private LG candidate (cayley 2026-11-09; codex d7af675d
integration acceptance). OFF: nothing in run_ensemble_daily calls it, no schedule runs it, and it fetches only through
the session a caller passes (tests pass a recording fake; an activation would pass make_verified_session()).

What it pins down at the runner boundary:
- ONE RefreshPolicyNGL per run (open_run). Every region's refresh draws from that one object, so the per-run retry
  budget (RF.RETRY_BUDGET extra attempts) is shared across all regions and acquisitions; a fresh helper per region
  would silently reset it.
- calibration_run: refresh the catalog (calibration runs only; the declared carriers of the current baselines must
  survive it), RE-LOAD it, refresh every candidate station file the calibration windows will read, calibrate each
  region, and write ONE dated baseline file. A baseline whose generated day differs from calibrated_on (retrospective)
  is refused here; the planner would refuse it anyway.
- catalogs (codex d38ea435): every catalog a baseline uses is kept as immutable, hash-addressed metadata in the
  run's catalog store (<baselines>/catalogs/<sha256>.json). Calibration preserves the catalog in force BEFORE it
  refreshes, stores the refreshed one, and calibrates from the STORED bytes it binds.
- scoring_run: per region, the as-of baseline is selected first and ITS catalog is resolved from the store (verified
  by sha256; missing or altered refuses BOUND_CATALOG_UNAVAILABLE / _ALTERED by name, never falling back to the
  mutable cache copy). That catalog drives both the declared carrier's refresh and the producer. A later calibration
  that changes the catalog therefore cannot strand earlier scored days; the mutable cache catalog is irrelevant to
  scoring.
- the cache is CANDIDATE-OWNED: open_run refuses a cache directory listed in `forbidden_dirs` (an activation passes
  the legacy runner cache and the approved evidence snapshot).
- NGL only. A GeoNet (NZ) region is refused REFRESH_POLICY_NGL_ONLY: the refresh policy covers NGL files only.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List, Optional

import live_data as LD
import lg_producer_candidate as P
import lg_refresh_candidate as RF

VERSION = 'lg-candidate-runner-v1'


@dataclass
class RunContext:
    run_day: date
    run_id: str
    cache_dir: Path
    baselines_dir: Path
    acquisition: RF.RefreshPolicyNGL            # the ONE acquisition of this run (shared retry budget)
    log: List[dict] = field(default_factory=list)


def catalog_store(baselines_dir) -> Path:
    return Path(baselines_dir) / 'catalogs'


def store_catalog(store, raw: bytes) -> Path:
    """Keep catalog bytes as immutable, hash-addressed metadata; an existing entry must hold the same bytes."""
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(raw).hexdigest()
    path = store / f'{sha}.json'
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError(f'catalog store entry {sha} is altered; refusing to bind it')
        return path
    with open(path, 'xb') as fh:
        fh.write(raw)
    return path


def open_run(run_day: date, run_id: str, cache_dir, baselines_dir, ledger_path, *, session,
             clock: Callable[[], str], sleep: Callable[[float], None] = time.sleep, forbidden_dirs=()) -> RunContext:
    resolved = os.path.normcase(os.path.realpath(cache_dir))
    if any(resolved == os.path.normcase(os.path.realpath(d)) for d in forbidden_dirs):
        raise ValueError('CANDIDATE_CACHE_MUST_BE_CANDIDATE_OWNED: %s is a forbidden directory' % cache_dir)
    acq = RF.RefreshPolicyNGL(cache_dir, run_day=run_day, run_id=run_id, ledger_path=ledger_path, session=session,
                              clock=clock, sleep=sleep)
    acq.load_station_catalog()                                  # a run starts from the cached catalog (never fetched)
    return RunContext(run_day, run_id, Path(cache_dir), Path(baselines_dir), acq)


def _ngl_only(region: str) -> Optional[str]:
    if region not in LD.FAULT_POLYGONS:
        return 'REGION_HAS_NO_POLYGON'
    if region in LD.GeoNetLiveAcquisition.NZ_STATIONS:          # the class table is_nz_region reads
        return 'REFRESH_POLICY_NGL_ONLY'
    return None


def refresh_region_inputs(ctx: RunContext, region: str, target: date, days_back: int, declared=None) -> List[dict]:
    """Refresh the NGL files one window of `region` will read, through the run's one acquisition: the declared carrier
    on scoring runs; production's candidate rule (catalog stations in the polygon's bounding box, first 50) on
    calibration runs. Returns this call's refresh records."""
    acq = ctx.acquisition                                       # the caller loaded the catalog this run binds
    start = len(acq.records)
    codes = list(declared) if declared is not None else acq.find_stations_in_polygon(LD.FAULT_POLYGONS[region]['polygon'])[:50]
    for code in codes:
        acq.download_station(code, days_back, P.as_of_instant(target))
    return acq.records[start:]


def _declared_union(baselines_dir: Path, run_day: date, regions) -> set:
    selected, _ = P.select_as_of(baselines_dir, run_day)
    out = set()
    if selected:
        for region in regions:
            entry = (selected['doc'].get('regions') or {}).get(region) or {}
            if entry.get('available'):
                out.update(entry.get('station_codes') or [])
    return out


def calibration_run(ctx: RunContext, regions, *, generated_utc: str, code_id: Optional[str] = None) -> dict:
    """Catalog refresh -> catalog re-load -> station refresh -> calibrate -> one dated baseline file (calibrated_on =
    the run day). Prospective only: `generated_utc` must fall on the run day."""
    if str(generated_utc)[:10] != ctx.run_day.isoformat():
        return dict(status='RETROSPECTIVE_CALIBRATION_REFUSED', generated_utc=generated_utc,
                    calibrated_on=ctx.run_day.isoformat())
    declared = _declared_union(ctx.baselines_dir, ctx.run_day, regions)
    store = catalog_store(ctx.baselines_dir)
    current = ctx.cache_dir / 'ngl_catalog.json'
    if current.exists():                                        # preserve the catalog in force before replacing it
        store_catalog(store, current.read_bytes())
    catalog = ctx.acquisition.refresh_catalog(calibration=True, declared=declared or None)
    ctx.acquisition.load_station_catalog()                      # RE-LOAD the catalog this calibration will bind
    bound = store_catalog(store, current.read_bytes())          # the stored bytes this calibration binds
    code_id = code_id or P.code_identity()
    entries, refresh = {}, {}
    window_end = P.calibration_window_end(ctx.run_day)
    for region in regions:
        why = _ngl_only(region)
        if why:
            entries[region] = dict(available=False, reason=why)
            continue
        refresh[region] = refresh_region_inputs(ctx, region, window_end, P.BASELINE_DAYS)
        entries[region] = P.calibrate_region(region, ctx.run_day, ctx.cache_dir, code_id=code_id, catalog_path=bound)
    failed = {r: e.get('reason') or 'BASELINE_UNAVAILABLE'
              for r, e in entries.items() if not e.get('available')}
    if failed:
        out = dict(status='CALIBRATION_FAILED', version=VERSION, run_id=ctx.run_id,
                   baseline_file=None, failed_regions=failed, catalog=catalog,
                   bound_catalog=bound.name, retries_left=ctx.acquisition.retries_left)
        ctx.log.append(out)
        return out
    path = P.write_baselines(ctx.baselines_dir, ctx.run_day, entries, code_id, generated_utc)
    out = dict(status='CALIBRATED', version=VERSION, run_id=ctx.run_id, baseline_file=path.name, catalog=catalog,
               bound_catalog=bound.name,
               regions={r: dict(available=bool(e.get('available')), reason=e.get('reason'),
                                n_stations=e.get('n_stations'), last_epoch=e.get('last_epoch'),
                                catalog_sha256=(e.get('selection') or {}).get('catalog_sha256'),
                                not_admitted=(e.get('admission') or {}).get('not_admitted'))
                        for r, e in entries.items()},
               refresh={r: [x['outcome'] for x in recs] for r, recs in refresh.items()},
               retries_left=ctx.acquisition.retries_left)
    ctx.log.append(out)
    return out


def scoring_run(ctx: RunContext, regions, scored_day: date) -> Dict[str, dict]:
    """Per region: the catalog-binding GATE, then (only when bound) a refresh of the declared carrier's files and the
    producer. Each result carries what run_region_assessment takes: ratio, provenance, refusal (status, detail)."""
    results = {}
    for region in regions:
        why = _ngl_only(region)
        if why:
            results[region] = dict(status=why, ratio=None, provenance=None, refusal=(why, 'not refreshed or scored'))
            continue
        selected, why = P.select_as_of(ctx.baselines_dir, scored_day)
        entry = ((selected['doc'].get('regions') or {}).get(region) or {}) if selected else {}
        if not selected or not entry.get('available'):
            status = why or 'BASELINE_UNAVAILABLE'
            results[region] = dict(status=status, gate=None, ratio=None, provenance=None,
                                   refusal=(status, entry.get('reason') or 'no usable as-of baseline'), refresh=[])
            continue
        path, cstatus, cdetail = P.resolve_bound_catalog(catalog_store(ctx.baselines_dir), entry)
        gate = dict(status=cstatus, detail=cdetail, baseline=selected['path'].name)
        if path is None:                                        # never the mutable cache copy
            results[region] = dict(status=cstatus, gate=gate, ratio=None, provenance=None, refusal=(cstatus, cdetail),
                                   refresh=[])
            continue
        ctx.acquisition.station_catalog = json.loads(path.read_text(encoding='utf-8'))   # the baseline's catalog
        refresh = refresh_region_inputs(ctx, region, scored_day, P.DAYS_BACK, declared=entry['station_codes'])
        ratio, prov, trace = P.candidate_lambda_geo(region, scored_day, ctx.cache_dir, ctx.baselines_dir,
                                                    catalog_store=catalog_store(ctx.baselines_dir))
        results[region] = dict(status=trace['status'], gate=gate, ratio=ratio, provenance=prov, trace=trace,
                               refusal=None if ratio is not None else (trace['status'], trace.get('detail', '')),
                               refresh=refresh)
    ctx.log.append(dict(scored_day=scored_day.isoformat(), run_id=ctx.run_id, retries_left=ctx.acquisition.retries_left,
                        regions={r: v['status'] for r, v in results.items()}))
    return results
