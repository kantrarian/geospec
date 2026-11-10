"""LG candidate operation (cayley 2026-11-09; owner authorization relayed in codex 0c71239f1): the flag-gated,
default-OFF connection of the LG candidate producer to the daily runner, for an explicit two-region allowlist.

OFF is the default. With monitoring/config/lg_candidate_operation.json absent or `enabled: false`, or for any scored
day before the registered effective scored day E, the daily runner calls nothing here: every region takes the legacy
LG path unchanged (lg_candidate=None into run_all_regions). ON, for a scored day >= E and an allowlisted region only:

- one candidate run per daily run (lg_candidate_runner.open_run: one acquisition, one shared retry budget, verified
  TLS) on the candidate-owned state root (cache/, baselines/, ledger/, runs/). The legacy runner cache and every
  host-listed evidence copy are forbidden cache directories;
- a weekly PROSPECTIVE re-calibration (calibrated_on = the run day) when the as-of baseline is >= 7 days old or the
  daily actual-epoch freshness plan asks for it. It refreshes ONLY the declared carriers of the previous baseline and
  fetches NO catalog (0c71239f1: subsequent refreshes are only the declared target carriers). The FIRST baseline
  (catalog plus at most 50 candidate stations per region) is the separate operator command `seed`;
- scoring of the scored day (catalog-bound, declared-carrier refresh only), handed to run_region_assessment under the
  registered CandidatePolicy. Every failure is a TYPED exclusion of that region's LG; the daily service goes on.

Host-local controls in the state root, never in the repo:
- OFF: present -> every allowlisted region's LG is LG_CANDIDATE_OFF and no request is made;
- host_forbidden_dirs.json: REQUIRED, a JSON list of the host's evidence copies, bound as forbidden cache directories.

Operator commands (no daily-run side effects):
  python lg_candidate_operation.py plan                      read-only: config, targets, state, freshness plan
  python lg_candidate_operation.py seed --expect-run-day D   the first prospective calibration (an acquisition)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import calibration_eligibility as CE
import lg_candidate_runner as LCR
import lg_integration_candidate as LGI
import lg_producer_candidate as P
import lg_refresh_candidate as RF

VERSION = 'lg-candidate-operation-v1'
CONFIG_REL = 'monitoring/config/lg_candidate_operation.json'
ALLOWED_REGIONS = ('tokyo_kanto', 'kumamoto')          # the owner-authorized scope (0c71239f1); nothing else is valid
REGISTERED_POLICY = dict(baseline_max_age_days=45, calibration_min_lag_days=30, observation_max_age_days=45,
                         min_baseline_samples=30)
RECALIBRATION_DAYS = 7
CONFIG_KEYS = frozenset({'schema', 'enabled', 'regions', 'effective_scored_day', 'policy', 'recalibration_days',
                         'state_root', 'amendment'})
OFF_SWITCH = 'OFF'
HOST_FORBIDDEN = 'host_forbidden_dirs.json'
LEGACY_CACHE_REL = 'monitoring/data/gps_cache'
LEDGER_NAME = 'ngl_refresh_ledger.json'
ERROR_CHARS = 200


class OperationConfigInvalid(Exception):
    """The committed operation config is malformed. The production preflight refuses (exit 12) before any fetch."""


@dataclass(frozen=True)
class Operation:
    regions: Tuple[str, ...]
    effective_scored_day: date
    policy: LGI.CandidatePolicy
    state_root: Path
    config_sha256: str
    amendment: str

    @property
    def cache(self) -> Path:
        return self.state_root / 'cache'

    @property
    def baselines(self) -> Path:
        return self.state_root / 'baselines'

    @property
    def ledger(self) -> Path:
        return self.state_root / 'ledger' / LEDGER_NAME

    @property
    def runs(self) -> Path:
        return self.state_root / 'runs'


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _bounded(exc: BaseException) -> str:
    text = f'{type(exc).__name__}: {exc}'
    return text if len(text) <= ERROR_CHARS else text[:ERROR_CHARS - 1] + '…'


def load_operation(repo_root) -> Optional[Operation]:
    """None when OFF (config absent or `enabled: false`); the validated Operation when ON. A malformed config raises
    OperationConfigInvalid by name: the registered values are carried, never defaulted."""
    repo_root = Path(repo_root)
    path = repo_root / CONFIG_REL
    if not path.exists():
        return None
    raw = path.read_bytes()

    def bad(detail):
        raise OperationConfigInvalid(f'LG_OPERATION_CONFIG_INVALID: {CONFIG_REL}: {detail}')

    try:
        doc = json.loads(raw.decode('utf-8'))
    except (ValueError, UnicodeDecodeError) as exc:
        bad(f'not JSON ({type(exc).__name__})')
    if not isinstance(doc, dict) or doc.get('schema') != VERSION:
        bad(f'schema must be {VERSION}')
    if doc.get('enabled') is False:
        return None
    if doc.get('enabled') is not True:
        bad('enabled must be true or false')
    if set(doc) != CONFIG_KEYS:
        bad(f'keys must be exactly {sorted(CONFIG_KEYS)}')
    regions = doc['regions']
    if (not isinstance(regions, list) or not regions or len(set(regions)) != len(regions)
            or any(r not in ALLOWED_REGIONS for r in regions)):
        bad(f'regions must be a non-empty subset of {list(ALLOWED_REGIONS)} (the owner-authorized scope)')
    effective = LGI._day(doc['effective_scored_day'])
    if effective is None:
        bad('effective_scored_day must be an ISO day')
    if not CE.rule_active_for_scored_day(effective.isoformat()):
        bad('effective_scored_day precedes the active eligibility rule, under which the candidate is classified')
    if doc['policy'] != REGISTERED_POLICY:
        bad(f'policy must equal the registered values {REGISTERED_POLICY}; a change needs a new amendment')
    if doc['recalibration_days'] != RECALIBRATION_DAYS:
        bad(f'recalibration_days must be {RECALIBRATION_DAYS}')
    state = doc['state_root']
    if not isinstance(state, str) or not state or os.path.isabs(state):
        bad('state_root must be a relative path')
    state_root = (repo_root / state).resolve()
    checkout = repo_root.resolve()
    if state_root == checkout or checkout in state_root.parents:
        bad('state_root must lie outside the runner checkout (the committed-program preflight never sees it)')
    amendment = doc['amendment']
    if not isinstance(amendment, str) or os.path.isabs(amendment) or not (repo_root / amendment).is_file():
        bad('amendment must name the committed amendment document')
    text = (repo_root / amendment).read_text(encoding='utf-8')
    missing = [s for s in (VERSION, effective.isoformat(), *regions) if s not in text]
    if missing:
        bad(f'the amendment does not register {missing}')
    return Operation(tuple(regions), effective, LGI.CandidatePolicy(effective.isoformat(), **REGISTERED_POLICY),
                     state_root, hashlib.sha256(raw).hexdigest(), amendment)


def targets(op: Optional[Operation], regions, scored_day: date) -> List[str]:
    """The allowlisted regions of this run that the candidate scores: none while OFF or before E."""
    if op is None or scored_day < op.effective_scored_day:
        return []
    return [r for r in regions if r in op.regions]


def forbidden_dirs(op: Operation, repo_root) -> Tuple[Optional[List[str]], Optional[str]]:
    """(forbidden cache directories, None) or (None, typed reason). The legacy runner cache always; the host's evidence
    copies from the REQUIRED host_forbidden_dirs.json in the state root."""
    path = op.state_root / HOST_FORBIDDEN
    try:
        listed = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        return None, f'{HOST_FORBIDDEN} unreadable in the state root ({type(exc).__name__})'
    if not isinstance(listed, list) or not listed or not all(isinstance(d, str) and os.path.isabs(d) for d in listed):
        return None, f'{HOST_FORBIDDEN} must be a non-empty list of absolute directories'
    return [str(Path(repo_root) / LEGACY_CACHE_REL)] + listed, None


def recalibration_due(plan: dict, run_day: date) -> Tuple[bool, str]:
    """Weekly, or sooner when the daily actual-epoch freshness plan says a region is close to or past stale. Never
    without a previous prospective baseline: the first baseline is the operator's `seed`."""
    selected = plan.get('file')
    if selected is None:
        return False, 'NO_BASELINE: the first baseline is the operator seed'
    if selected.get('retrospective'):
        return False, 'RETROSPECTIVE_BASELINE_FILE: not a prospective baseline'
    calibrated_on = date.fromisoformat(selected['calibrated_on'])
    if calibrated_on >= run_day:
        return False, 'CALIBRATED_ON_THIS_RUN_DAY'
    if (run_day - calibrated_on).days >= RECALIBRATION_DAYS:
        return True, f'WEEKLY: previous calibrated_on {calibrated_on.isoformat()}'
    near = sorted(f"{r} {v['action']}" for r, v in plan['regions'].items()
                  if str(v.get('action', '')).startswith(('RECALIBRATE_BY', 'STALE')))
    if near:
        return True, 'FRESHNESS: ' + ', '.join(near)
    return False, 'NOT_DUE'


def recalibrate(ctx, regions, *, generated_utc: str, code_id: Optional[str] = None) -> dict:
    """The weekly prospective re-calibration: per region, the PREVIOUS baseline's bound catalog (no catalog fetch),
    a refresh of that baseline's declared carrier only, then calibrate_region; one dated file for the run day."""
    if str(generated_utc)[:10] != ctx.run_day.isoformat():
        return dict(status='RETROSPECTIVE_CALIBRATION_REFUSED', generated_utc=generated_utc,
                    calibrated_on=ctx.run_day.isoformat())
    selected, why = P.select_as_of(ctx.baselines_dir, ctx.run_day)
    if selected is None:
        return dict(status='NO_BASELINE', detail=why)
    store = LCR.catalog_store(ctx.baselines_dir)
    code_id = code_id or P.code_identity()
    window_end = P.calibration_window_end(ctx.run_day)
    entries, refresh, bound = {}, {}, {}
    for region in regions:
        why = LCR._ngl_only(region)
        previous = (selected['doc'].get('regions') or {}).get(region) or {}
        if why or not previous.get('available'):
            entries[region] = dict(available=False, reason=why or 'PREVIOUS_BASELINE_UNAVAILABLE: re-seeding is an '
                                                                    'operator step')
            continue
        path, cstatus, cdetail = P.resolve_bound_catalog(store, previous)
        if path is None:
            entries[region] = dict(available=False, reason=f'{cstatus}: {cdetail}')
            continue
        ctx.acquisition.station_catalog = json.loads(path.read_text(encoding='utf-8'))
        refresh[region] = LCR.refresh_region_inputs(ctx, region, window_end, P.BASELINE_DAYS,
                                                    declared=previous['station_codes'])
        entries[region] = P.calibrate_region(region, ctx.run_day, ctx.cache_dir, code_id=code_id, catalog_path=path)
        bound[region] = path.name
    # An unavailable candidate must not become the newest selectable baseline.
    # Keep the prior immutable baseline and retry the scheduled calibration on
    # the next run, subject to its normal freshness and acquisition limits.
    failed = {r: e.get('reason') or 'BASELINE_UNAVAILABLE'
              for r, e in entries.items() if not e.get('available')}
    if failed:
        return dict(status='RECALIBRATION_FAILED', failed_regions=failed,
                    previous=selected['path'].name, baseline_file=None,
                    scope='declared carriers only; no catalog fetch',
                    retries_left=ctx.acquisition.retries_left)
    written = P.write_baselines(ctx.baselines_dir, ctx.run_day, entries, code_id, generated_utc)
    return dict(status='RECALIBRATED', scope='declared carriers only; no catalog fetch', previous=selected['path'].name,
                baseline_file=written.name, bound_catalogs=bound,
                regions={r: dict(available=bool(e.get('available')), reason=e.get('reason'),
                                 n_stations=e.get('n_stations'), last_epoch=e.get('last_epoch'))
                         for r, e in entries.items()},
                refresh={r: [x['outcome'] for x in recs] for r, recs in refresh.items()},
                retries_left=ctx.acquisition.retries_left)


def _write_summary(op: Operation, name: str, summary: dict, log=None) -> Optional[str]:
    """Create-once run evidence in <state_root>/runs; a failure to write it is reported, never fatal."""
    try:
        op.runs.mkdir(parents=True, exist_ok=True)
        with open(op.runs / f'{name}.json', 'x', encoding='utf-8') as fh:
            json.dump(summary, fh, indent=1, sort_keys=True, default=str)
        return None
    except OSError as exc:
        if log is not None:
            log.warning(f'LG candidate run summary not written: {_bounded(exc)}')
        return _bounded(exc)


def daily_candidate(op: Optional[Operation], regions, scored_day: date, run_day: date, repo_root, *,
                    session_factory: Callable = RF.make_verified_session, clock: Callable[[], str] = _utc_now,
                    sleep: Callable[[float], None] = time.sleep, log=None) -> Dict[str, dict]:
    """region -> dict(ratio, provenance, refusal, policy, status) for each allowlisted region of a scored day >= E;
    {} otherwise (nothing is opened, read or requested). Never raises: a failure is that region's typed exclusion."""
    chosen = targets(op, regions, scored_day)
    if not chosen:
        return {}
    run_id = 'daily-' + clock().replace('-', '').replace(':', '')
    summary = dict(version=VERSION, run_id=run_id, run_day=run_day.isoformat(), scored_day=scored_day.isoformat(),
                   regions=chosen, effective_scored_day=op.effective_scored_day.isoformat(),
                   config_sha256=op.config_sha256, policy=str(op.policy))

    def every(status, detail):
        return {r: dict(ratio=None, provenance=None, refusal=(status, detail), policy=op.policy, status=status)
                for r in chosen}

    try:
        summary['code_identity'] = P.code_identity()
        if (op.state_root / OFF_SWITCH).exists():
            out = every('LG_CANDIDATE_OFF', 'operator off switch present in the candidate state root; no request made')
        else:
            forbidden, why = forbidden_dirs(op, repo_root)
            if why:
                out = every('LG_CANDIDATE_STATE_INVALID', why)
            else:
                ctx = LCR.open_run(run_day, run_id, op.cache, op.baselines, op.ledger, session=session_factory(),
                                   clock=clock, sleep=sleep, forbidden_dirs=forbidden)
                plan = RF.calibration_plan(op.baselines, run_day, chosen,
                                           baseline_max_age_days=op.policy.baseline_max_age_days)
                due, reason = recalibration_due(plan, run_day)
                summary['calibration_plan'] = plan
                summary['recalibration'] = dict(due=due, reason=reason)
                if due:
                    try:                                    # a failed re-calibration keeps the previous baseline
                        summary['recalibration']['result'] = recalibrate(ctx, chosen, generated_utc=clock())
                    except Exception as exc:
                        summary['recalibration']['result'] = dict(status='RECALIBRATION_FAILED', error=_bounded(exc))
                scored = LCR.scoring_run(ctx, chosen, scored_day)
                out = {r: dict(ratio=v['ratio'], provenance=v['provenance'], refusal=v['refusal'], policy=op.policy,
                               status=v['status']) for r, v in scored.items()}
                summary['scoring'] = {r: dict(status=v['status'], ratio=v['ratio'], refusal=v['refusal'],
                                              gate=v.get('gate'),
                                              refresh=[x.get('outcome') for x in (v.get('refresh') or [])])
                                      for r, v in scored.items()}
                summary['retries_left'] = ctx.acquisition.retries_left
    except Exception as exc:                                # the daily service goes on; this region's LG is excluded
        summary['error'] = _bounded(exc)
        out = every('LG_CANDIDATE_RUN_FAILED', _bounded(exc))
    summary['outcome'] = {r: v['status'] for r, v in out.items()}
    _write_summary(op, run_id, summary, log)
    if log is not None:
        log.info(f'LG candidate {run_id}: ' + ', '.join(f'{r} {v["status"]}' for r, v in out.items()))
    return out


def seed(op: Operation, run_day: date, repo_root, *, session_factory: Callable = RF.make_verified_session,
         clock: Callable[[], str] = _utc_now, sleep: Callable[[float], None] = time.sleep) -> dict:
    """The FIRST prospective calibration, run once by the operator: the NGL catalog plus at most 50 candidate
    stations per allowlisted region (lg_candidate_runner.calibration_run). Refuses by name, before any request, when
    the day is not today (UTC), the off switch is present, the forbidden list is invalid or a baseline already
    exists."""
    now = clock()
    if now[:10] != run_day.isoformat():
        return dict(status='SEED_NOT_PROSPECTIVE', detail=f'run day {run_day.isoformat()} is not today ({now[:10]} UTC)')
    if (op.state_root / OFF_SWITCH).exists():
        return dict(status='LG_CANDIDATE_OFF', detail='operator off switch present; nothing requested')
    forbidden, why = forbidden_dirs(op, repo_root)
    if why:
        return dict(status='LG_CANDIDATE_STATE_INVALID', detail=why)
    existing = sorted(p.name for p in op.baselines.glob('lambda_geo_baselines_rule_*.json')) if op.baselines.exists() else []
    if existing:
        return dict(status='SEED_ALREADY_DONE', detail=f'baselines exist: {existing}')
    for d in (op.cache, op.baselines, op.ledger.parent):
        if not d.is_dir():
            return dict(status='LG_CANDIDATE_STATE_INVALID', detail=f'missing state directory {d.name}')
    run_id = 'seed-' + now.replace('-', '').replace(':', '')
    ctx = LCR.open_run(run_day, run_id, op.cache, op.baselines, op.ledger, session=session_factory(), clock=clock,
                       sleep=sleep, forbidden_dirs=forbidden)
    out = LCR.calibration_run(ctx, list(op.regions), generated_utc=clock())
    stations = sorted({r['file'].split('.')[0] for r in ctx.acquisition.records if r.get('file') != 'ngl_catalog.json'})
    out = dict(out, operation=VERSION, run_day=run_day.isoformat(), station_products=len(stations),
               station_bound=50 * len(op.regions), config_sha256=op.config_sha256)
    _write_summary(op, run_id, out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='LG candidate operation: read-only plan, or the operator seed')
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('plan')
    sp = sub.add_parser('seed')
    sp.add_argument('--expect-run-day', required=True, help='today in UTC (YYYY-MM-DD); the seed refuses otherwise')
    args = ap.parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent.parent
    try:
        op = load_operation(repo_root)
    except OperationConfigInvalid as exc:
        print(str(exc))
        return 12
    if op is None:
        print(json.dumps(dict(status='OFF', config=CONFIG_REL)))
        return 0
    if args.command == 'plan':
        today = date.fromisoformat(_utc_now()[:10])
        forbidden, why = forbidden_dirs(op, repo_root)
        plan = RF.calibration_plan(op.baselines, today, list(op.regions),
                                   baseline_max_age_days=op.policy.baseline_max_age_days)
        print(json.dumps(dict(status='ON', version=VERSION, regions=list(op.regions),
                              effective_scored_day=op.effective_scored_day.isoformat(), policy=str(op.policy),
                              state_root=str(op.state_root), config_sha256=op.config_sha256,
                              off_switch=(op.state_root / OFF_SWITCH).exists(), forbidden=forbidden,
                              forbidden_problem=why, calibration_plan=plan,
                              recalibration=recalibration_due(plan, today)), indent=1, default=str))
        return 0
    out = seed(op, date.fromisoformat(args.expect_run_day), repo_root)
    print(json.dumps(out, indent=1, default=str))
    return 0 if out.get('status') == 'CALIBRATED' else 12


if __name__ == '__main__':
    sys.exit(main())
