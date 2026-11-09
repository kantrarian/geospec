"""LG candidate trace (cayley 2026-11-09): cache input -> selected observation -> baseline -> component -> combined score.

OFFLINE and read-only toward every issued record. For each region and scored day it
1. selects (or, when absent, writes once) the dated rule baseline as of the scored day (lg_producer_candidate),
2. measures the retained inputs with the unchanged production operator and the support rule,
3. replays the ISSUED record's non-LG components through the REAL GeoSpecEnsemble.compute_risk combiner, twice:
   - control: with the issued LG component as recorded and no candidate policy. It must reproduce the issued combined
     risk, tier, methods_available and effective weights exactly, or the region refuses REPLAY_CONTROL_MISMATCH;
   - candidate: with the candidate LG ratio (or its typed refusal) under an explicit CandidatePolicy.
The issued components are replayed as recorded values: THD / FC are not recomputed, and nothing is written outside
--baselines-dir and --out. A replay over a cache fetched after the scored day uses inputs AS RETAINED (the manifest
hashes them), not necessarily as available on the scored day; the trace says which.

  py -3.11 -B lg_candidate_trace.py --cache-dir DIR --issued RECORD.json --scored-day 2026-11-04 \\
      --calibrated-on 2026-10-30 --effective-scored-day 2026-11-04 --baselines-dir DIR --generated-utc <clock read> \\
      --region tokyo_kanto --region kumamoto --out trace.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import ensemble as E
import lg_integration_candidate as LGI
import lg_producer_candidate as P

VERSION = 'lg-candidate-trace-v1'


def method_result(component: dict, support=None) -> E.MethodResult:
    """A MethodResult carrying a recorded component's values exactly (no recomputation)."""
    r = E.MethodResult(name=component['name'], available=bool(component['available']),
                       raw_value=component.get('raw_value'), raw_secondary=component.get('raw_secondary'),
                       risk_score=component.get('risk_score'), is_elevated=component.get('is_elevated'),
                       is_critical=component.get('is_critical'), notes=component.get('notes', ''),
                       frozen=bool(component.get('frozen', False)))
    if 'score_basis' in component:
        r.score_basis = component['score_basis']
    cal = component.get('calibration')
    if isinstance(cal, dict):
        r.calibration_status = cal.get('status')
        r.eligible_for_tiering = cal.get('eligible_for_tiering')
        r.eligibility_rule_version = cal.get('rule_version')
        r.eligibility_reason = cal.get('reason')
    r.support = support
    return r


class ReplayEnsemble(E.GeoSpecEnsemble):
    """The production ensemble with the non-LG components (and, for the control, the LG component) supplied from an
    issued region record. compute_risk, the tier predicate, confidence and method_set are the real code."""

    def __init__(self, region, record, replay_lg, **kwargs):
        super().__init__(region, eligibility_rule_active=True, **kwargs)
        self._record = record
        self._replay_lg = replay_lg
        self._support = ((record.get('method_set') or {}).get('support') or {})

    def _recorded(self, name):
        return method_result(self._record['components'][name], self._support.get(name))

    def compute_fault_correlation_risk(self, date):
        cov = self._record.get('coverage') or {}
        return (self._recorded('fault_correlation'), int(cov.get('segments_defined', 0)),
                int(cov.get('segments_working', 0)), list(cov.get('segment_names', [])))

    def compute_thd_risk(self, date, station_network=None, station_code=None, **kwargs):
        return self._recorded('seismic_thd')

    def compute_lambda_geo_risk(self, date):
        if self._replay_lg:
            return self._recorded('lambda_geo')
        return super().compute_lambda_geo_risk(date)


def _combined(res: E.EnsembleResult) -> dict:
    return dict(combined_risk=res.combined_risk, tier=res.tier, tier_name=res.tier_name,
                methods_available=res.methods_available, effective_weights=dict(res.effective_weights),
                included=(res.method_set or {}).get('included'), notes=res.notes)


def _issued(record: dict) -> dict:
    return dict(combined_risk=record['combined_risk'], tier=record['tier'], tier_name=record['tier_name'],
                methods_available=record['methods_available'], effective_weights=dict(record['effective_weights']),
                included=(record.get('method_set') or {}).get('included'), notes=record.get('notes', ''))


def replay_region(region: str, scored_day: date, record: dict, policy: LGI.CandidatePolicy, ratio, provenance, status,
                  detail) -> dict:
    when = datetime(scored_day.year, scored_day.month, scored_day.day)
    methods = tuple(n for n in E.MC.METHOD_ORDER if n in record['components'])
    control = ReplayEnsemble(region, record, replay_lg=True).compute_risk(when, methods=methods)
    issued = _issued(record)
    got = _combined(control)
    same = (got['combined_risk'] == issued['combined_risk'] and got['tier'] == issued['tier']
            and got['methods_available'] == issued['methods_available']
            and got['effective_weights'] == issued['effective_weights'] and got['included'] == issued['included'])
    out = dict(issued=issued, control=dict(got, reproduces_issued=same))
    if not same:
        return dict(out, status='REPLAY_CONTROL_MISMATCH')
    cand = ReplayEnsemble(region, record, replay_lg=False, lg_candidate_policy=policy)
    if ratio is not None:
        cand.set_lambda_geo(when, ratio, provenance=provenance)
    else:
        cand.set_lambda_geo_unavailable(when, status, detail, provenance=provenance)
    res = cand.compute_risk(when, methods=methods)
    lg = res.components['lambda_geo']
    return dict(out, status='REPLAYED', candidate=dict(_combined(res), lambda_geo=lg.to_dict()),
                issued_lambda_geo=record['components'].get('lambda_geo'))


LEGACY_DAYS_BACK = 120                 # fetch_ngl_lambda_geo's default, used by the daily run


def legacy_reproduction(region: str, scored_day: date, cache_dir: Path, legacy_file: Path, issued_lg: dict) -> dict:
    """The PRODUCTION R3 ratio recomputed from the same retained cache: days_back 120, the naive-midnight target the
    daily run uses, the last grid row on or before it (the unsupported edge), over the legacy baseline file. Equality
    with the issued raw_value shows the retained bytes are the ones production read that day; inequality means the
    cache moved after the scored day (the R3 value is then not reproducible from it)."""
    import run_ensemble_daily as RD
    ngl, geonet = P.RetainedNGL(cache_dir), P.RetainedGeoNet(cache_dir)
    try:
        ngl.load_station_catalog()
    except FileNotFoundError:
        return dict(status='RETAINED_CATALOG_ABSENT')
    target = datetime(scored_day.year, scored_day.month, scored_day.day)
    res = P.LD.acquire_region_data(region, ngl, LEGACY_DAYS_BACK, target, geonet)
    cal = RD.read_lambda_geo_calibration(legacy_file)
    entry = (cal.get('regions') or {}).get(region) or {}
    issued = (issued_lg or {}).get('raw_value')
    if res is None or res.n_stations < 3 or not res.lambda_geo_max > 0:
        return dict(status='LEGACY_MEASUREMENT_UNAVAILABLE', issued_raw_value=issued, legacy_file_sha256=cal.get('sha256'))
    if not entry.get('available'):
        return dict(status='LEGACY_BASELINE_UNAVAILABLE', lambda_max=res.lambda_geo_max, issued_raw_value=issued,
                    legacy_file_sha256=cal.get('sha256'))
    ratio = max(0.1, min(50.0, res.lambda_geo_max / entry['mean_lambda_geo']))
    return dict(status='REPRODUCES_ISSUED' if issued is not None and ratio == issued else 'DIFFERS_FROM_ISSUED',
                lambda_max=res.lambda_geo_max, station_codes=list(res.station_codes),
                baseline_mean=entry['mean_lambda_geo'], ratio=ratio, issued_raw_value=issued,
                legacy_file_sha256=cal.get('sha256'))


def ensure_baseline(baselines_dir: Path, calibrated_on: date, regions, cache_dir, code_id, generated_utc):
    path = Path(baselines_dir) / f'lambda_geo_baselines_rule_{calibrated_on:%Y%m%d}.json'
    if path.exists():
        return path, False
    entries = {r: P.calibrate_region(r, calibrated_on, cache_dir, code_id=code_id) for r in regions}
    return P.write_baselines(baselines_dir, calibrated_on, entries, code_id, generated_utc), True


class TraceRefused(Exception):
    """The trace refused before writing anything: the input snapshot is not the one it must be."""
    def __init__(self, code, detail):
        super().__init__(code)
        self.code, self.detail = code, detail


def _snapshot(regions, windows, cache_dir):
    """Each window's selection: the calibration window under the declared-carrier admission (as calibrate_region makes
    it), the scored window with every candidate attempted (a superset of any declared carrier)."""
    return {r: {w: P.select_inputs(r, day, cache_dir, days, admit_until=P.admission_day(day) if w == 'calibration' else None)
                for w, (day, days) in windows.items()} for r in regions}


def _diffs(reference, candidate):
    out = {}
    for r in reference:
        for w in reference[r]:
            d = P.compare_selections(reference[r][w], candidate[r][w])
            if d:
                out[f'{r}/{w}'] = d
    return out


def _summary(selection):
    outcomes = {}
    for group in ('ngl_attempts', 'geonet_attempts'):
        for row in (selection.get(group) or {}).values():
            outcomes[row['outcome']] = outcomes.get(row['outcome'], 0) + 1
    return dict(sha256=P._sha(P.canonical(selection)), outcome=selection.get('outcome'), provider=selection.get('provider'),
                carrier=selection.get('carrier'), attempts_by_outcome=outcomes, as_of=selection.get('as_of'))


def run(args) -> dict:
    scored = date.fromisoformat(args.scored_day)
    calibrated_on = date.fromisoformat(args.calibrated_on)
    cache = Path(args.cache_dir)
    windows = dict(calibration=(P.calibration_window_end(calibrated_on), P.BASELINE_DAYS), scored=(scored, P.DAYS_BACK))
    # The selection each window makes from this cache (every attempted station and file, present or absent), taken
    # BEFORE anything else; a reference cache (the original, when tracing a copy) must make exactly the same selection.
    before = _snapshot(args.region, windows, cache)
    if args.selection_reference:
        diffs = _diffs(_snapshot(args.region, windows, Path(args.selection_reference)), before)
        if diffs:
            raise TraceRefused('SELECTION_DIFFERS_FROM_REFERENCE', diffs)
    raw = Path(args.issued).read_bytes()
    issued = json.loads(raw.decode('utf-8'))
    if issued.get('date') != args.scored_day:
        raise SystemExit(f"issued record is for {issued.get('date')!r}, not {args.scored_day}")
    policy = LGI.CandidatePolicy(args.effective_scored_day)
    code_id = P.code_identity()
    bpath, wrote = ensure_baseline(Path(args.baselines_dir), calibrated_on, args.region, Path(args.cache_dir), code_id,
                                   args.generated_utc)
    cache_files = sorted(p for p in Path(args.cache_dir).iterdir() if p.is_file())
    newest = max((datetime.fromtimestamp(p.stat().st_mtime, timezone.utc) for p in cache_files), default=None)
    trace = dict(version=VERSION, producer=P.VERSION, code_identity=code_id, scored_day=args.scored_day,
                 policy=dict(vars(policy)), issued_record=dict(path=str(args.issued), sha256=hashlib.sha256(raw).hexdigest()),
                 baseline_file=dict(name=bpath.name, written_by_this_run=wrote,
                                    sha256=hashlib.sha256(bpath.read_bytes()).hexdigest()),
                 cache=dict(dir=str(args.cache_dir), n_files=len(cache_files),
                            newest_mtime_utc=newest.strftime('%Y-%m-%dT%H:%M:%SZ') if newest else None,
                            retained_after_scored_day=bool(newest and newest.date() > scored)),
                 regions={})
    generated = str(json.loads(bpath.read_bytes().decode('utf-8')).get('generated_utc', ''))
    reasons = ['CANDIDATE_POLICY_NOT_REGISTERED']
    if generated[:10] > args.scored_day:
        reasons.append('BASELINE_GENERATED_AFTER_SCORED_DAY')
    if trace['cache']['retained_after_scored_day']:
        reasons.append('INPUTS_RETAINED_AFTER_SCORED_DAY')
    trace['counterfactual'] = dict(reasons=reasons, baseline_generated_utc=generated,
                                   boundary='offline counterfactual replay: not an issued, qualifying or warning result; '
                                            'the physical origin of any LG change is not attributed here')
    for region in args.region:
        ratio, prov, t = P.candidate_lambda_geo(region, scored, Path(args.cache_dir), Path(args.baselines_dir), code_id)
        record = (issued.get('regions') or {}).get(region)
        if record is None:
            trace['regions'][region] = dict(producer=t, replay=dict(status='REGION_NOT_IN_ISSUED_RECORD'))
            continue
        row = dict(producer=t, ratio=ratio, provenance=prov,
                   replay=replay_region(region, scored, record, policy, ratio, prov, t['status'], t.get('detail', '')))
        if args.legacy_baselines:
            row['legacy'] = legacy_reproduction(region, scored, Path(args.cache_dir), Path(args.legacy_baselines),
                                                record['components'].get('lambda_geo'))
        entry = (json.loads(bpath.read_bytes().decode('utf-8')).get('regions') or {}).get(region) or {}
        row['baseline_selection_check'] = (
            'SAME_INPUTS_AS_THIS_CACHE' if entry.get('selection_sha256') == P._sha(P.canonical(before[region]['calibration']))
            else 'BASELINE_FROM_OTHER_INPUTS')
        trace['regions'][region] = row
    after = _snapshot(args.region, windows, cache)
    if after != before:
        raise TraceRefused('INPUTS_CHANGED_DURING_TRACE', _diffs(before, after))
    trace['input_selection'] = {r: {w: _summary(before[r][w]) for w in before[r]} for r in before}
    trace['input_selection_reference'] = str(args.selection_reference) if args.selection_reference else None
    return trace


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--cache-dir', required=True)
    ap.add_argument('--issued', required=True, help='the issued ensemble record of the scored day (docs/ensemble/...)')
    ap.add_argument('--scored-day', required=True)
    ap.add_argument('--calibrated-on', required=True)
    ap.add_argument('--effective-scored-day', required=True)
    ap.add_argument('--baselines-dir', required=True)
    ap.add_argument('--generated-utc', required=True, help='a clock read, recorded in a newly written baseline file')
    ap.add_argument('--region', action='append', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--legacy-baselines', help='the host lambda_geo_baselines.json, to reproduce the issued R3 ratio')
    ap.add_argument('--selection-reference', help='the ORIGINAL cache when --cache-dir is a copy: both windows must '
                                                  'select identically from both, or the trace refuses')
    args = ap.parse_args(argv)
    try:
        trace = run(args)
    except TraceRefused as refusal:
        print('REFUSED', refusal.code, json.dumps(refusal.detail, sort_keys=True, default=str)[:4000], file=sys.stderr)
        return 3 if refusal.code == 'INPUTS_CHANGED_DURING_TRACE' else 4
    out = Path(args.out)
    with open(out, 'xb') as fh:
        fh.write(json.dumps(trace, indent=1, sort_keys=True, default=str).encode('utf-8') + b'\n')
    for region, row in trace['regions'].items():
        rep = row['replay']
        cand = rep.get('candidate') or {}
        print(region, row['producer']['status'], 'ratio', row.get('ratio'), '| replay', rep['status'],
              '| issued', (rep.get('issued') or {}).get('combined_risk'), '-> candidate', cand.get('combined_risk'),
              cand.get('tier_name'), '| legacy', (row.get('legacy') or {}).get('status'), '|',
              row.get('baseline_selection_check'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
