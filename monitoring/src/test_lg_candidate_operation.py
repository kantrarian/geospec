"""lg-candidate-operation-v1 tests (cayley 2026-11-09; owner authorization relayed in codex 0c71239f1). Every file lives
in a temp dir laid out like the host: <root>/repo (the committed config and amendment) and <root>/state (the
candidate-owned cache, baselines, ledger and runs). A recording fake session serves the producer tests' NGL-layout
files; clocks are fixtures. Checks: the config is carried, never defaulted; OFF and pre-E days touch nothing; the
operator seed is the only full acquisition; a daily run re-calibrates weekly on the declared carriers with no catalog
fetch; every failure is a typed exclusion; and the runner passes the candidate only for allowlisted regions."""
import io
import json
import logging
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import calibration_eligibility_fixtures as FX
FX.install_stubs()
import lg_candidate_operation as LCO           # noqa: E402
import lg_integration_candidate as LGI          # noqa: E402
import lg_producer_candidate as P               # noqa: E402
import run_ensemble_daily as RD                 # noqa: E402
import test_lg_candidate_runner as RT           # noqa: E402
import test_lg_producer_candidate as TT         # noqa: E402

E = RT.SCORED                                    # 2026-11-04
REGIONS = ['tokyo_kanto', 'kumamoto']


def config(**over):
    doc = dict(schema=LCO.VERSION, enabled=True, regions=list(REGIONS), effective_scored_day=E.isoformat(),
               policy=dict(LCO.REGISTERED_POLICY), recalibration_days=LCO.RECALIBRATION_DAYS, state_root='../state',
               amendment='docs/AMENDMENT_TEST_lg.md')
    doc.update(over)
    return doc


class Layout(RT.Base):
    """RT.Base (fake NGL, synthetic files) plus a host-like repo and candidate state root."""

    def setUp(self):
        super().setUp()
        self.repo = self.root / 'repo'
        (self.repo / 'monitoring' / 'config').mkdir(parents=True)
        (self.repo / 'docs').mkdir()
        (self.repo / 'docs' / 'AMENDMENT_TEST_lg.md').write_text(
            f'{LCO.VERSION}: effective scored day {E.isoformat()}; regions tokyo_kanto, kumamoto\n', encoding='utf-8')
        self.state = self.root / 'state'
        for d in ('cache', 'baselines', 'ledger'):
            (self.state / d).mkdir(parents=True)
        self.evidence = self.root / 'evidence_copy'
        self.evidence.mkdir()
        self.write_forbidden([str(self.evidence)])
        self.write_config(config())
        self.sessions = []

    def write_config(self, doc):
        (self.repo / LCO.CONFIG_REL).write_text(json.dumps(doc) if isinstance(doc, dict) else doc, encoding='utf-8')

    def write_forbidden(self, listed):
        (self.state / LCO.HOST_FORBIDDEN).write_text(json.dumps(listed), encoding='utf-8')

    def op(self):
        return LCO.load_operation(self.repo)

    def factory(self, failing=()):
        def make():
            s = self.serve(failing)
            self.sessions.append(s)
            return s
        return make

    def requests(self):
        return [u for s in self.sessions for u in s.calls]

    def seed(self, day=RT.CAL_DAY):
        return LCO.seed(self.op(), day, self.repo, session_factory=self.factory(),
                        clock=lambda: day.isoformat() + 'T12:05:00Z', sleep=self.sleeps.append)

    def daily(self, scored, run_day, **kw):
        kw.setdefault('session_factory', self.factory())
        return LCO.daily_candidate(self.op(), list(RD.REGIONS.keys()), scored, run_day, self.repo,
                                   clock=lambda: run_day.isoformat() + 'T12:00:30Z', sleep=self.sleeps.append, **kw)


class Config(Layout):
    def test_absent_or_disabled_is_off(self):
        (self.repo / LCO.CONFIG_REL).unlink()
        self.assertIsNone(self.op())
        self.write_config(dict(schema=LCO.VERSION, enabled=False))
        self.assertIsNone(self.op())

    def test_a_valid_config_carries_the_registered_values(self):
        op = self.op()
        self.assertEqual(op.regions, tuple(REGIONS))
        self.assertEqual(op.effective_scored_day, E)
        self.assertEqual(op.policy, LGI.CandidatePolicy(E.isoformat(), **LCO.REGISTERED_POLICY))
        self.assertEqual(op.state_root, self.state.resolve())
        self.assertEqual(op.ledger, self.state.resolve() / 'ledger' / LCO.LEDGER_NAME)

    def test_every_malformed_config_refuses_by_name(self):
        (self.repo / 'docs' / 'AMENDMENT_TEST_lg.md').write_text(            # registers every day the cases use
            f'{LCO.VERSION}: effective scored day {E.isoformat()} (also 2026-10-01); regions tokyo_kanto, kumamoto\n',
            encoding='utf-8')
        policy = dict(LCO.REGISTERED_POLICY, baseline_max_age_days=60)
        cases = {
            'outside the scope': (config(regions=['tokyo_kanto', 'cascadia']), 'owner-authorized scope'),
            'duplicate': (config(regions=['kumamoto', 'kumamoto']), 'owner-authorized scope'),
            'empty': (config(regions=[]), 'owner-authorized scope'),
            'malformed E': (config(effective_scored_day='2026-11-4'), 'must be an ISO day'),
            'E before the eligibility rule': (config(effective_scored_day='2026-10-01'), 'active eligibility rule'),
            'changed policy': (config(policy=policy), 'registered values'),
            'cadence': (config(recalibration_days=6), 'recalibration_days'),
            'absolute state': (config(state_root=str(self.state)), 'relative path'),
            'state inside the checkout': (config(state_root='lgstate'), 'outside the runner checkout'),
            'missing amendment': (config(amendment='docs/nothing.md'), 'committed amendment'),
            'amendment without E': (config(effective_scored_day='2026-11-05'), "does not register ['2026-11-05']"),
            'extra key': (dict(config(), extra=1), 'keys must be exactly'),
            'enabled not boolean': (config(enabled='yes'), 'enabled must be'),
            'schema': (config(schema='lg-candidate-operation-v0'), 'schema must be'),
            'not JSON': ('{"schema": ', 'not JSON'),
        }
        for name, (doc, fragment) in cases.items():
            with self.subTest(name):
                self.write_config(doc)
                with self.assertRaises(LCO.OperationConfigInvalid) as cm:
                    self.op()
                self.assertTrue(str(cm.exception).startswith('LG_OPERATION_CONFIG_INVALID: '), str(cm.exception))
                self.assertIn(fragment, str(cm.exception))

    def test_the_legacy_cache_is_always_forbidden_and_the_host_list_is_required(self):
        forbidden, why = LCO.forbidden_dirs(self.op(), self.repo)
        self.assertIsNone(why)
        self.assertEqual(forbidden, [str(self.repo / 'monitoring' / 'data' / 'gps_cache'), str(self.evidence)])
        for listed in ([], ['relative/dir'], 'C:/x'):
            self.write_forbidden(listed)
            self.assertEqual(LCO.forbidden_dirs(self.op(), self.repo)[0], None, listed)

    def test_recalibration_is_weekly_or_on_freshness_never_without_a_prospective_baseline(self):
        day = date(2026, 11, 10)

        def plan(calibrated_on, action='OK', retrospective=False, present=True):
            return dict(file=dict(calibrated_on=calibrated_on, retrospective=retrospective) if present else None,
                        regions={'tokyo_kanto': dict(action=action), 'kumamoto': dict(action='OK')})
        self.assertEqual(LCO.recalibration_due(plan('2026-11-03'), day), (True, 'WEEKLY: previous calibrated_on 2026-11-03'))
        self.assertEqual(LCO.recalibration_due(plan('2026-11-04'), day), (False, 'NOT_DUE'))
        self.assertEqual(LCO.recalibration_due(plan('2026-11-04', 'RECALIBRATE_BY 2026-11-11'), day),
                         (True, 'FRESHNESS: tokyo_kanto RECALIBRATE_BY 2026-11-11'))
        self.assertEqual(LCO.recalibration_due(plan('2026-11-04', 'STALE'), day), (True, 'FRESHNESS: tokyo_kanto STALE'))
        self.assertEqual(LCO.recalibration_due(plan('2026-11-04', 'UNAVAILABLE'), day), (False, 'NOT_DUE'))
        self.assertEqual(LCO.recalibration_due(plan('2026-11-10'), day), (False, 'CALIBRATED_ON_THIS_RUN_DAY'))
        self.assertFalse(LCO.recalibration_due(plan('2026-11-01', retrospective=True), day)[0])
        self.assertFalse(LCO.recalibration_due(plan(None, present=False), day)[0])

    def test_targets_are_the_allowlist_on_or_after_E_only(self):
        op = self.op()
        everything = list(RD.REGIONS.keys())
        self.assertEqual(LCO.targets(op, everything, E - timedelta(days=1)), [])
        self.assertEqual(LCO.targets(op, everything, E), [r for r in everything if r in REGIONS])
        self.assertEqual(LCO.targets(op, ['cascadia'], E), [])
        self.assertEqual(LCO.targets(None, everything, E), [])


class Seed(Layout):
    def test_the_seed_is_the_bounded_first_acquisition(self):
        out = self.seed()
        self.assertEqual(out['status'], 'CALIBRATED', out)
        reqs = self.requests()
        self.assertEqual(reqs[0], RT.CAT)                                       # the catalog first
        self.assertLessEqual(out['station_products'], out['station_bound'])     # at most 50 per region
        self.assertEqual(out['station_bound'], 100)
        doc = json.loads((self.state / 'baselines' / 'lambda_geo_baselines_rule_20261030.json').read_text(encoding='utf-8'))
        self.assertEqual((doc['calibrated_on'], doc['generated_utc']), ('2026-10-30', '2026-10-30T12:05:00Z'))
        for region in REGIONS:
            self.assertTrue(doc['regions'][region]['available'], region)
            self.assertEqual(doc['regions'][region]['origin_rule'], P.ORIGIN_RULE)
        self.assertEqual(len(list((self.state / 'runs').glob('seed-*.json'))), 1)

    def test_the_seed_refuses_by_name_before_any_request(self):
        self.assertEqual(self.seed(day=RT.CAL_DAY + timedelta(days=1))['status'], 'CALIBRATED')
        self.sessions.clear()
        self.assertEqual(self.seed()['status'], 'SEED_ALREADY_DONE')
        other = Layout('setUp')
        other.setUp()
        try:
            out = LCO.seed(other.op(), RT.CAL_DAY, other.repo, session_factory=other.factory(),
                           clock=lambda: '2026-10-31T00:00:01Z')
            self.assertEqual(out['status'], 'SEED_NOT_PROSPECTIVE')
            (other.state / LCO.OFF_SWITCH).write_text('', encoding='utf-8')
            self.assertEqual(other.seed()['status'], 'LG_CANDIDATE_OFF')
            (other.state / LCO.OFF_SWITCH).unlink()
            (other.state / LCO.HOST_FORBIDDEN).unlink()
            self.assertEqual(other.seed()['status'], 'LG_CANDIDATE_STATE_INVALID')
            self.assertEqual(other.requests(), [])
        finally:
            other.tearDown()
        self.assertEqual(self.requests(), [])


class Daily(Layout):
    def test_pre_E_and_off_touch_nothing(self):
        self.seed()
        self.sessions.clear()
        self.assertEqual(self.daily(E - timedelta(days=1), E + timedelta(days=1)), {})
        self.write_config(dict(schema=LCO.VERSION, enabled=False))
        self.assertEqual(LCO.daily_candidate(self.op(), list(RD.REGIONS.keys()), E, E + timedelta(days=2), self.repo,
                                             session_factory=self.factory()), {})
        self.assertEqual(self.sessions, [])                                     # no session was even opened
        self.assertFalse((self.state / 'runs').exists() and any((self.state / 'runs').glob('daily-*.json')))

    def test_a_daily_run_scores_the_allowlist_from_the_declared_carriers(self):
        self.seed()
        self.sessions.clear()
        out = self.daily(E, E + timedelta(days=1))                              # run day 11-05: 6 days, not due
        self.assertEqual(sorted(out), sorted(REGIONS))
        for region in REGIONS:
            self.assertEqual(out[region]['status'], 'OK', out[region]['refusal'])
            self.assertIsNotNone(out[region]['ratio'])
            self.assertIsNone(out[region]['refusal'])
            self.assertEqual(out[region]['policy'], self.op().policy)
            self.assertEqual(out[region]['provenance']['source'], f'lambda_geo_baselines_rule_20261030.json:{region}')
        self.assertNotIn(RT.CAT, self.requests())
        fetched = sorted({u.rsplit('/', 1)[1].split('.')[0] for u in self.requests()})
        self.assertEqual(fetched, sorted([c for c, _, _ in TT.STATIONS] + [c for c, _, _ in RT.KUMAMOTO]))
        summary = json.loads(next((self.state / 'runs').glob('daily-*.json')).read_text(encoding='utf-8'))
        self.assertEqual(summary['recalibration'], dict(due=False, reason='NOT_DUE'))
        self.assertEqual(summary['outcome'], {r: 'OK' for r in REGIONS})

    def test_the_weekly_recalibration_refreshes_declared_carriers_only_and_fetches_no_catalog(self):
        self.seed()
        self.sessions.clear()
        out = self.daily(E, RT.CAL_DAY + timedelta(days=7))                     # run day 11-06: 7 days, due
        summary = json.loads(next((self.state / 'runs').glob('daily-*.json')).read_text(encoding='utf-8'))
        result = summary['recalibration']['result']
        self.assertEqual(result['status'], 'RECALIBRATED', result)
        self.assertEqual(result['baseline_file'], 'lambda_geo_baselines_rule_20261106.json')
        self.assertNotIn(RT.CAT, self.requests())                               # no catalog fetch
        declared = {c for c, _, _ in TT.STATIONS if c != RT.DEAD[0]} | {c for c, _, _ in RT.KUMAMOTO}
        fetched = {u.rsplit('/', 1)[1].split('.')[0] for u in self.requests()}
        self.assertLessEqual(fetched, declared)                                 # no candidate expansion
        new = json.loads((self.state / 'baselines' / 'lambda_geo_baselines_rule_20261106.json').read_text(encoding='utf-8'))
        self.assertEqual(new['generated_utc'][:10], '2026-11-06')               # prospective
        for region in REGIONS:
            self.assertEqual(out[region]['status'], 'OK')                       # 11-04 still scores on its 10-30 baseline
            self.assertEqual(out[region]['provenance']['calibrated_on'], '2026-10-30')

    def test_a_recalibration_whose_clock_left_the_run_day_is_refused_before_any_request(self):
        self.seed()
        self.sessions.clear()
        run_day = RT.CAL_DAY + timedelta(days=7)
        ctx = RT.RUN.open_run(run_day, 'late', self.state / 'cache', self.state / 'baselines',
                              self.state / 'ledger' / LCO.LEDGER_NAME, session=self.factory()(),
                              clock=lambda: '2026-11-07T00:00:01Z', sleep=self.sleeps.append)
        out = LCO.recalibrate(ctx, REGIONS, generated_utc='2026-11-07T00:00:01Z')   # crossed UTC midnight
        self.assertEqual(out['status'], 'RETROSPECTIVE_CALIBRATION_REFUSED')
        self.assertEqual(self.requests(), [])
        self.assertFalse((self.state / 'baselines' / 'lambda_geo_baselines_rule_20261106.json').exists())

    def test_a_failed_recalibration_keeps_the_previous_baseline(self):
        self.seed()
        with mock.patch.object(P, 'write_baselines', side_effect=OSError('disk full')):
            out = self.daily(E, RT.CAL_DAY + timedelta(days=7))
        summary = json.loads(next((self.state / 'runs').glob('daily-*.json')).read_text(encoding='utf-8'))
        self.assertEqual(summary['recalibration']['result']['status'], 'RECALIBRATION_FAILED')
        self.assertEqual({r: v['status'] for r, v in out.items()}, {r: 'OK' for r in REGIONS})

    def test_every_failure_is_a_typed_exclusion_and_never_raises(self):
        self.seed()
        self.sessions.clear()
        (self.state / LCO.OFF_SWITCH).write_text('', encoding='utf-8')
        out = self.daily(E, E + timedelta(days=1))
        self.assertEqual({r: v['refusal'][0] for r, v in out.items()}, {r: 'LG_CANDIDATE_OFF' for r in REGIONS})
        self.assertEqual(self.sessions, [])
        (self.state / LCO.OFF_SWITCH).unlink()
        (self.state / LCO.HOST_FORBIDDEN).unlink()
        out = self.daily(E, E + timedelta(days=1))
        self.assertEqual({r: v['status'] for r, v in out.items()}, {r: 'LG_CANDIDATE_STATE_INVALID' for r in REGIONS})
        self.assertEqual(self.sessions, [])
        self.write_forbidden([str(self.state / 'cache')])                      # the candidate cache itself
        out = self.daily(E, E + timedelta(days=1))
        self.assertEqual({v['status'] for v in out.values()}, {'LG_CANDIDATE_RUN_FAILED'})
        self.assertIn('CANDIDATE_CACHE_MUST_BE_CANDIDATE_OWNED', out['tokyo_kanto']['refusal'][1])
        self.write_forbidden([str(self.evidence)])

        def boom():
            raise RuntimeError('no route to host')
        out = self.daily(E, E + timedelta(days=1), session_factory=boom)
        self.assertEqual(out['kumamoto']['refusal'], ('LG_CANDIDATE_RUN_FAILED', 'RuntimeError: no route to host'))
        for region in REGIONS:                                                  # each reaches the ensemble by name
            res = RD.run_region_assessment(region, datetime(2026, 11, 4), use_seismic=False,
                                           lambda_geo_provenance=out[region]['provenance'],
                                           lg_candidate_policy=out[region]['policy'],
                                           lg_candidate_refusal=out[region]['refusal'])
            lg = res.components['lambda_geo']
            self.assertFalse(lg.available)
            self.assertTrue(lg.eligibility_reason.startswith('LG_CANDIDATE_RUN_FAILED: RuntimeError'), lg.eligibility_reason)

    def test_no_baseline_refuses_without_a_request(self):
        out = self.daily(E, E + timedelta(days=1))
        self.assertEqual({v['ratio'] for v in out.values()}, {None})
        self.assertNotIn('LG_CANDIDATE_RUN_FAILED', {v['status'] for v in out.values()})
        self.assertEqual(self.requests(), [])
        summary = json.loads(next((self.state / 'runs').glob('daily-*.json')).read_text(encoding='utf-8'))
        self.assertEqual(summary['recalibration']['due'], False)
        self.assertTrue(summary['recalibration']['reason'].startswith('NO_BASELINE'))


class Runner(unittest.TestCase):
    """run_all_regions and main(): OFF / pre-E is the legacy call for every region; ON changes allowlisted regions only."""

    def calls(self, lg_candidate=None):
        recorded = []

        def assess(**kw):
            recorded.append(kw)
            return None
        legacy = {r: 1.0 + k for k, r in enumerate(RD.REGIONS)}
        with mock.patch.object(RD, 'run_region_assessment', side_effect=assess), \
                mock.patch.object(RD, 'fetch_ngl_lambda_geo', return_value=dict(legacy)), \
                mock.patch.object(RD, 'LAMBDA_GEO_PILOT_AVAILABLE', False), \
                mock.patch.object(RD, 'NGL_LAMBDA_GEO_AVAILABLE', True):
            RD.run_all_regions(datetime(2026, 11, 4), fetch_events=False, lg_candidate=lg_candidate)
        return {kw['region']: kw for kw in recorded}

    def test_off_is_the_legacy_call_for_every_region(self):
        off = self.calls()
        self.assertEqual(sorted(off), sorted(RD.REGIONS))
        for region, kw in off.items():
            self.assertEqual(set(kw), {'region', 'target_date', 'lambda_geo_ratio', 'use_seismic',
                                       'lambda_geo_provenance', 'station_regions'}, region)
        self.assertEqual(self.calls({}), off)

    def test_on_changes_allowlisted_regions_only(self):
        policy = LGI.CandidatePolicy(E.isoformat())
        cand = {'tokyo_kanto': dict(ratio=11.5, provenance={'source': 'x'}, refusal=None, policy=policy, status='OK'),
                'kumamoto': dict(ratio=None, provenance=None, refusal=('LG_CANDIDATE_OFF', 'off'), policy=policy,
                                 status='LG_CANDIDATE_OFF')}
        off, on = self.calls(), self.calls(cand)
        for region in RD.REGIONS:
            if region in cand:
                self.assertEqual(on[region]['lg_candidate_policy'], policy)
                self.assertEqual(on[region]['lambda_geo_ratio'], cand[region]['ratio'])
                self.assertEqual(on[region]['lg_candidate_refusal'], cand[region]['refusal'])
                self.assertEqual(on[region]['lambda_geo_provenance'], cand[region]['provenance'])
            else:
                self.assertEqual(on[region], off[region], region)              # non-target: the identical legacy call


class Main(unittest.TestCase):
    """main() on the production path with the revision preflight stubbed and every scoring seam recorded."""

    NOW = datetime(2026, 11, 12, 12, 5, tzinfo=timezone.utc)               # scores 2026-11-10 (latency 2)

    def run_main(self, argv, operation, daily=None, configured=True, block_import=False):
        seen = dict(run_all=[], daily=[], load=[])

        def run_all(**kw):
            seen['run_all'].append(kw)
            return {}, {}                                                      # main() then stops: "No results"

        def fake_daily(*a, **k):
            seen['daily'].append((a, k))
            return daily if daily is not None else {}

        def fake_load(repo_root):
            seen['load'].append(repo_root)
            if isinstance(operation, Exception):
                raise operation
            return operation
        fixed = self.NOW

        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed if tz is not None else fixed.replace(tzinfo=None)
        tmp = tempfile.TemporaryDirectory(prefix='lgmain-')
        self.addCleanup(tmp.cleanup)
        repo = Path(tmp.name)
        if configured:                                                          # its content is load_operation's job
            (repo / RD.LG_OPERATION_CONFIG_REL).parent.mkdir(parents=True)
            (repo / RD.LG_OPERATION_CONFIG_REL).write_text('{}', encoding='utf-8')
        blocked = mock.patch.dict(sys.modules, {'lg_candidate_operation': None} if block_import else {})
        view = {'commit': '1' * 40, 'code': {}, 'calibration': {}}
        with blocked, mock.patch.object(LCO, 'load_operation', side_effect=fake_load), \
                mock.patch.object(LCO, 'daily_candidate', side_effect=fake_daily), \
                mock.patch.object(RD, 'REPO_ROOT', repo), \
                mock.patch.object(RD, 'run_all_regions', side_effect=run_all), \
                mock.patch.object(RD, 'datetime', Clock), \
                mock.patch.object(RD, '_HISTORICAL_REPLAY', False), \
                mock.patch.object(RD.REV, 'committed_inputs_view', return_value=view), \
                mock.patch.object(RD.REV, 'checkout_matches_committed', return_value=None), \
                mock.patch.object(RD.REV, 'check_store_clean', return_value=None), \
                mock.patch.object(RD.REV, 'load_legacy_baseline', return_value={'records': []}), \
                mock.patch.object(RD.REV, 'journal_bytes', return_value=b''), \
                mock.patch.object(RD.REV, 'prior_days_view', return_value=[]), \
                mock.patch.object(sys, 'argv', ['run_ensemble_daily.py'] + list(argv)):
            rc = RD.main()
        return rc, seen

    def op(self, effective):
        return LCO.Operation(tuple(REGIONS), effective, LGI.CandidatePolicy(effective.isoformat()), Path('/state'),
                             '0' * 64, 'docs/AMENDMENT_TEST_lg.md')

    def test_the_runner_and_the_operation_name_the_same_config(self):
        self.assertEqual(RD.LG_OPERATION_CONFIG_REL, LCO.CONFIG_REL)

    def test_without_the_committed_config_nothing_is_imported_or_read(self):
        rc, seen = self.run_main([], self.op(date(2026, 11, 10)), configured=False, block_import=True)
        self.assertEqual((seen['load'], seen['daily']), ([], []))
        self.assertIsNone(seen['run_all'][0]['lg_candidate'])

    def test_a_committed_config_that_cannot_be_imported_refuses_before_any_scoring(self):
        rc, seen = self.run_main([], self.op(date(2026, 11, 10)), block_import=True)
        self.assertEqual(rc, 12)
        self.assertEqual((seen['run_all'], seen['daily'], seen['load']), ([], [], []))

    def test_a_malformed_config_refuses_before_any_scoring(self):
        rc, seen = self.run_main([], LCO.OperationConfigInvalid('LG_OPERATION_CONFIG_INVALID: x'))
        self.assertEqual(rc, 12)
        self.assertEqual((seen['run_all'], seen['daily']), ([], []))
        self.assertEqual(len(seen['load']), 1)

    def test_off_and_pre_E_pass_no_candidate(self):
        for operation in (None, self.op(date(2026, 11, 11))):
            rc, seen = self.run_main([], operation)
            self.assertEqual(seen['daily'], [])
            self.assertIsNone(seen['run_all'][0]['lg_candidate'])

    def test_on_and_due_passes_the_daily_candidate_through(self):
        op = self.op(date(2026, 11, 10))
        marker = {'tokyo_kanto': dict(ratio=None, provenance=None, refusal=('X', 'y'), policy=op.policy, status='X')}
        rc, seen = self.run_main([], op, daily=marker)
        (args, kwargs), = seen['daily']
        self.assertEqual(args[0], op)
        self.assertEqual(sorted(args[1]), sorted(RD.REGIONS))
        self.assertEqual((args[2], args[3]), (date(2026, 11, 10), date(2026, 11, 12)))   # scored day, run day (UTC)
        self.assertIs(seen['run_all'][0]['lg_candidate'], marker)

    def test_an_explicit_date_replay_makes_no_acquisition(self):
        rc, seen = self.run_main(['--date', '2026-11-10'], self.op(date(2026, 11, 10)))
        self.assertEqual(seen['daily'], [])
        passed = seen['run_all'][0]['lg_candidate']
        self.assertEqual(sorted(passed), sorted(REGIONS))
        self.assertEqual({v['refusal'][0] for v in passed.values()}, {'LG_CANDIDATE_NOT_ON_REPLAY'})


if __name__ == '__main__':
    unittest.main()
