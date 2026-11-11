"""lg-candidate-runner-v1 tests (cayley 2026-11-09; codex d7af675d integration checks). A recording fake session serves
NGL-layout files written by the producer tests' writer (Tokyo stations plus a Kumamoto set); every file lives in a
temp dir; clocks are fixtures. Checks: one shared retry budget per run, the catalog-binding GATE, catalog re-load
before calibration, declared-carrier-only refresh, and the path into run_region_assessment."""
import json
import os
import tempfile
import time
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import calibration_eligibility_fixtures as FX
FX.install_stubs()
import live_data as LD                          # noqa: E402
import lg_candidate_runner as RUN               # noqa: E402
import lg_integration_candidate as LGI          # noqa: E402
import lg_producer_candidate as P               # noqa: E402
import lg_refresh_candidate as RF               # noqa: E402
import run_ensemble_daily as RD                 # noqa: E402
import test_lg_producer_candidate as TT         # noqa: E402

KUMAMOTO = (('KMT1', 32.0, 130.5), ('KMT2', 32.4, 131.1), ('KMT3', 32.9, 130.7), ('KMT4', 32.2, 131.6),
            ('KMT5', 33.1, 131.4))
DEAD = ('QTK9', 35.7, 140.4)                     # parses at calibration, last sample 08-03: NOT_ADMITTED
CAL_DAY = date(2026, 10, 30)
SCORED = date(2026, 11, 4)
POLICY = LGI.CandidatePolicy(SCORED.isoformat())
IGS20 = LD.NGLLiveAcquisition.BASE_URL_IGS20
IGS14 = LD.NGLLiveAcquisition.BASE_URL_IGS14
CAT = LD.NGLLiveAcquisition.CATALOG_URL


class FakeSession:
    verify = True

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, timeout=None):
        self.calls.append(url)
        status, text = self.routes(url)
        return SimpleNamespace(status_code=status, text=text, raise_for_status=lambda: None)


def llh(stations):
    return ''.join(f'{c} {lat:.4f} {lon - 360.0:.4f} 50.0\n' for c, lat, lon in stations)


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix='lgrun-')
        self.root = Path(self._tmp.name)
        self.src = self.root / 'source'
        self.stations = list(TT.STATIONS) + list(KUMAMOTO) + [DEAD]
        TT.write_cache(self.src, extra=tuple(KUMAMOTO) + (DEAD,), last={'QTK9': date(2026, 8, 3)})
        self.cache = self.root / 'cache'
        self.cache.mkdir()
        (self.cache / 'ngl_catalog.json').write_text(json.dumps({c: [lat, lon - 360.0] for c, lat, lon in TT.STATIONS}),
                                                     encoding='utf-8')
        self.base = self.root / 'baselines'
        self.sleeps = []

    def tearDown(self):
        self._tmp.cleanup()

    def serve(self, failing=()):
        def routes(url):
            if url == CAT:
                return 200, llh(self.stations)
            name = url.rsplit('/', 1)[1]
            code = name.split('.')[0]
            if url.startswith(IGS20) and code in failing:
                return 503, ''
            if url.startswith(IGS20) and (self.src / f'{code}.igs20.tenv').exists():
                return 200, (self.src / f'{code}.igs20.tenv').read_text(encoding='utf-8')
            return 404, ''
        return FakeSession(routes)

    def run_ctx(self, session, run_day, run_id):
        return RUN.open_run(run_day, run_id, self.cache, self.base, self.root / 'ledger.json', session=session,
                            clock=lambda: run_day.isoformat() + 'T12:00:30Z', sleep=self.sleeps.append)

    def calibrate(self):
        s = self.serve()
        ctx = self.run_ctx(s, CAL_DAY, 'cal-1030')
        return ctx, s, RUN.calibration_run(ctx, ['tokyo_kanto', 'kumamoto'], generated_utc='2026-10-30T12:05:00Z')


class Calibration(Base):
    def test_calibration_refreshes_and_reloads_the_catalog_it_binds(self):
        ctx, s, out = self.calibrate()
        self.assertEqual(out['status'], 'CALIBRATED')
        self.assertEqual(out['catalog']['outcome'], 'REFRESHED_CATALOG')
        self.assertEqual(s.calls[0], CAT)
        current = RF._sha((self.cache / 'ngl_catalog.json').read_bytes())
        for region in ('tokyo_kanto', 'kumamoto'):
            row = out['regions'][region]
            self.assertTrue(row['available'], (region, row))
            self.assertEqual(row['catalog_sha256'], current, region)             # the refreshed catalog was bound
        self.assertEqual(out['regions']['kumamoto']['n_stations'], 5)            # only in the refreshed catalog
        self.assertEqual(out['regions']['tokyo_kanto']['not_admitted'], {'QTK9': '2026-08-03'})
        doc = json.loads((self.base / 'lambda_geo_baselines_rule_20261030.json').read_text(encoding='utf-8'))
        self.assertEqual(doc['generated_utc'], '2026-10-30T12:05:00Z')

    def test_a_catalog_that_would_drop_a_declared_station_is_refused_at_the_next_calibration(self):
        self.calibrate()
        before = (self.cache / 'ngl_catalog.json').read_bytes()
        self.stations = [s for s in self.stations if s[0] != 'QTK4'] + [(f'QX{k}', 35.0 + k / 50, 139.5) for k in range(3)]
        s = self.serve()
        ctx = self.run_ctx(s, date(2026, 11, 6), 'cal-1106')
        out = RUN.calibration_run(ctx, ['tokyo_kanto', 'kumamoto'], generated_utc='2026-11-06T12:05:00Z')
        cat = out['catalog']
        self.assertEqual((cat['outcome'], cat['cause']), ('FAILED_KEPT_PREVIOUS', 'DECLARED_STATION_MISSING_FROM_CATALOG'))
        self.assertEqual(cat['detail'], 'QTK4')
        self.assertEqual((self.cache / 'ngl_catalog.json').read_bytes(), before)
        self.assertEqual(out['regions']['tokyo_kanto']['catalog_sha256'], RF._sha(before))   # bound to the kept catalog

    def test_a_retrospective_calibration_is_refused_before_any_request(self):
        s = self.serve()
        ctx = self.run_ctx(s, CAL_DAY, 'late')
        out = RUN.calibration_run(ctx, ['tokyo_kanto'], generated_utc='2026-11-09T02:00:00Z')
        self.assertEqual(out['status'], 'RETROSPECTIVE_CALIBRATION_REFUSED')
        self.assertEqual(s.calls, [])
        self.assertFalse(self.base.exists() and any(self.base.iterdir()))


class CatalogRollover(Base):
    """codex d38ea435 P1, locked as a POSITIVE regression: a calibration that refreshes the catalog must not strand
    earlier scored days. Each baseline's catalog is immutable, hash-addressed; scoring loads the as-of baseline's."""

    def test_lagged_days_keep_their_baseline_and_catalog_after_a_catalog_changing_calibration(self):
        regions = ['tokyo_kanto', 'kumamoto']
        initial_sha = RF._sha((self.cache / 'ngl_catalog.json').read_bytes())   # the catalog before any calibration
        self.calibrate()
        old_file = self.base / 'lambda_geo_baselines_rule_20261030.json'
        old_bytes = old_file.read_bytes()
        old_sha = json.loads(old_bytes)['regions']['tokyo_kanto']['selection']['catalog_sha256']
        before = RUN.scoring_run(self.run_ctx(self.serve(), date(2026, 11, 6), 'pre-calibration'), regions, SCORED)
        self.assertEqual({r: v['status'] for r, v in before.items()}, {'tokyo_kanto': 'OK', 'kumamoto': 'OK'})
        self.stations.append(('QNEW', 0.1, 0.1))                                # outside both regions (codex repro)
        s = self.serve()
        ctx = self.run_ctx(s, date(2026, 11, 6), 'new-calibration')
        cal = RUN.calibration_run(ctx, regions, generated_utc='2026-11-06T12:05:00Z')
        self.assertEqual(cal['catalog']['outcome'], 'REFRESHED_CATALOG')
        new_sha = json.loads((self.base / 'lambda_geo_baselines_rule_20261106.json').read_text(encoding='utf-8'))[
            'regions']['tokyo_kanto']['selection']['catalog_sha256']
        self.assertNotEqual(new_sha, old_sha)
        self.assertEqual(old_file.read_bytes(), old_bytes)                       # old baseline untouched
        store = RUN.catalog_store(self.base)
        self.assertEqual(sorted(p.name for p in store.iterdir()),
                         sorted([initial_sha + '.json', old_sha + '.json', new_sha + '.json']))   # each preserved
        for day, file, sha in ((date(2026, 11, 4), '20261030', old_sha), (date(2026, 11, 5), '20261030', old_sha),
                               (date(2026, 11, 6), '20261106', new_sha)):
            res = RUN.scoring_run(ctx, regions, day)
            for region in regions:
                r = res[region]
                self.assertEqual(r['status'], 'OK', (day, region, r.get('refusal')))
                self.assertEqual(r['gate'], dict(status='BOUND_CATALOG_VERIFIED', detail=sha,
                                                 baseline=f'lambda_geo_baselines_rule_{file}.json'), (day, region))
                self.assertEqual(r['trace']['input_selection']['catalog_sha256'], sha, (day, region))
                self.assertEqual(r['provenance']['calibrated_on'], f'{file[:4]}-{file[4:6]}-{file[6:]}')

    def test_a_missing_or_altered_bound_catalog_refuses_by_name_without_falling_back(self):
        self.calibrate()
        entry = json.loads((self.base / 'lambda_geo_baselines_rule_20261030.json').read_text(encoding='utf-8'))['regions']['tokyo_kanto']
        stored = RUN.catalog_store(self.base) / (entry['selection']['catalog_sha256'] + '.json')
        good = stored.read_bytes()
        stored.write_bytes(good.replace(b'}', b', "QALT": [35.0, -220.0]}'))     # altered in place
        s = self.serve()
        r = RUN.scoring_run(self.run_ctx(s, SCORED + timedelta(days=2), 't-alt'), ['tokyo_kanto'], SCORED)['tokyo_kanto']
        self.assertEqual(r['refusal'][0], 'BOUND_CATALOG_ALTERED')
        self.assertEqual(s.calls, [])
        os.remove(stored)                                                        # gone; the cache copy is still there
        self.assertTrue((self.cache / 'ngl_catalog.json').exists())
        r = RUN.scoring_run(self.run_ctx(s, SCORED + timedelta(days=2), 't-gone'), ['tokyo_kanto'], SCORED)['tokyo_kanto']
        self.assertEqual(r['refusal'][0], 'BOUND_CATALOG_UNAVAILABLE')
        out = RD.run_region_assessment('tokyo_kanto', datetime(2026, 11, 4), use_seismic=False,
                                       lg_candidate_policy=POLICY, lg_candidate_refusal=r['refusal'])
        self.assertTrue(out.components['lambda_geo'].eligibility_reason.startswith('BOUND_CATALOG_UNAVAILABLE'))

    def test_calibration_binds_the_stored_bytes_and_preserves_the_previous_catalog(self):
        old = (self.cache / 'ngl_catalog.json').read_bytes()
        ctx, s, out = self.calibrate()
        store = RUN.catalog_store(self.base)
        self.assertTrue((store / (RF._sha(old) + '.json')).exists())               # preserved before replacement
        bound = store / out['bound_catalog']
        self.assertEqual(RF._sha(bound.read_bytes()), out['regions']['tokyo_kanto']['catalog_sha256'])
        self.assertEqual(bound.read_bytes(), (self.cache / 'ngl_catalog.json').read_bytes())
        with self.assertRaises(ValueError):                                      # an altered store entry is never re-bound
            bound.write_bytes(bound.read_bytes() + b' ')
            RUN.store_catalog(store, (self.cache / 'ngl_catalog.json').read_bytes())

    def test_the_cache_must_be_candidate_owned(self):
        with self.assertRaises(ValueError):
            self.run_ctx_forbidden = RUN.open_run(SCORED, 't', self.cache, self.base, self.root / 'l.json',
                                                  session=self.serve(), clock=lambda: 'x', forbidden_dirs=[self.cache])
        RUN.open_run(SCORED, 't', self.cache, self.base, self.root / 'l.json', session=self.serve(), clock=lambda: 'x',
                     forbidden_dirs=[self.root / 'legacy-runner-cache', self.root / 'evidence-snapshot'])


class Scoring(Base):
    def test_scoring_refreshes_only_the_declared_carrier_and_reaches_the_ensemble(self):
        self.calibrate()
        s = self.serve()
        ctx = self.run_ctx(s, SCORED + timedelta(days=2), 'tick-45')
        res = RUN.scoring_run(ctx, ['tokyo_kanto', 'kumamoto'], SCORED)
        for region in ('tokyo_kanto', 'kumamoto'):
            self.assertEqual(res[region]['status'], 'OK', (region, res[region].get('refusal')))
            self.assertEqual(res[region]['gate']['status'], 'BOUND_CATALOG_VERIFIED')
        fetched = sorted({u.rsplit('/', 1)[1].split('.')[0] for u in s.calls})
        self.assertEqual(fetched, sorted([c for c, _, _ in TT.STATIONS] + [c for c, _, _ in KUMAMOTO]))
        self.assertNotIn(CAT, s.calls)                                           # no catalog fetch on a scoring run
        for region in ('tokyo_kanto', 'kumamoto'):
            r = res[region]
            out = RD.run_region_assessment(region, datetime(2026, 11, 4), lambda_geo_ratio=r['ratio'],
                                           use_seismic=False, lambda_geo_provenance=r['provenance'],
                                           lg_candidate_policy=POLICY, lg_candidate_refusal=r['refusal'])
            lg = out.components['lambda_geo']
            self.assertTrue(lg.eligible_for_tiering, (region, lg.eligibility_reason))
            self.assertEqual(out.method_set['included'], ['lambda_geo'])

    def test_station_file_freshness_names_the_earliest_file_without_moving_the_operator(self):
        # cayley 2026-11-11 (GNSS freshness): QTK3's provider file ends 8 days before the scored day.
        TT.write_cache(self.src, extra=tuple(KUMAMOTO) + (DEAD,), last={'QTK9': date(2026, 8, 3), 'QTK3': SCORED - timedelta(days=8)})
        self.calibrate()
        identity_before = P.code_identity()
        ctx = self.run_ctx(self.serve(), SCORED + timedelta(days=2), 'tick-fresh')
        r = RUN.scoring_run(ctx, ['tokyo_kanto'], SCORED)['tokyo_kanto']
        self.assertEqual(r['status'], 'OK', r.get('refusal'))
        m = r['provenance']['measurement']
        f = m['station_freshness']
        self.assertEqual(f['basis'], 'SELECTED_FILE_REFRESH_RECORD_END')
        self.assertEqual(f['earliest_file_end_stations'], ['QTK3'])
        self.assertTrue(f['all_carrier_file_ends_known'])
        self.assertEqual(f['support_limiter_status'], 'UNKNOWN_FILE_ENDS_DO_NOT_RESOLVE_INTERIOR_GAPS')
        self.assertNotIn('limiting_stations', f, 'a file-end bound is not a claim about actual support limiters')
        self.assertEqual(f['earliest_file_end_day'], (SCORED - timedelta(days=8)).isoformat())
        self.assertEqual(sorted(f['station_file_end_day']), sorted(r['provenance']['measurement'].get('declared', []) or
                                                                  r['trace']['declared_carrier']))
        self.assertEqual(m['data_end_day'], SCORED.isoformat(), 'the union end hides the lagging station')
        self.assertEqual(m['observation_day'], (SCORED - timedelta(days=8 + P.SUPPORT_MARGIN)).isoformat(),
                         'the rule recorded beside it explains the observation day')
        self.assertEqual(P.code_identity(), identity_before, 'recording freshness does not move the operator identity')
        out = RD.run_region_assessment('tokyo_kanto', datetime(2026, 11, 4), lambda_geo_ratio=r['ratio'], use_seismic=False,
                                       lambda_geo_provenance=r['provenance'], lg_candidate_policy=POLICY,
                                       lg_candidate_refusal=r['refusal'])
        lg = out.components['lambda_geo']
        self.assertTrue(lg.eligible_for_tiering, lg.eligibility_reason)
        self.assertEqual(lg.baseline_provenance['measurement']['station_freshness']['earliest_file_end_stations'], ['QTK3'],
                         'the record carries it to the report')

    def test_station_file_freshness_reads_only_refresh_records(self):
        recs = [dict(file='AAA1.igs20.tenv', data_end_after='2026-10-24'), dict(file='AAA1.tenv3', data_end_after='2026-12-01'),
                dict(file='BBB2.igs20.tenv', outcome='NOT_DUE', data_end_day='2026-10-20'),
                dict(file='CCC3.igs20.tenv', data_end_after='2026-11-30'), dict(file='ZZZ9.igs20.tenv', data_end_after='2026-01-01'),
                dict(file='ngl_catalog.json')]
        for rec in recs:
            rec['sha256'] = 'a' * 64
        selected = {c: dict(file=f'{c}.igs20.tenv', frame='IGS20', sha256='a' * 64) for c in ['AAA1', 'BBB2', 'CCC3']}
        f = RUN.station_file_freshness(recs, ['AAA1', 'BBB2', 'CCC3', 'DDD4'], date(2026, 11, 12), selected_files=selected)
        self.assertEqual(f['station_file_end_day'], {'AAA1': '2026-10-24', 'BBB2': '2026-10-20', 'CCC3': '2026-11-30', 'DDD4': None})
        self.assertEqual(f['station_file_kind']['AAA1'], 'IGS20', 'the selected manifest, not extension preference, binds the file')
        self.assertEqual((f['earliest_file_end_stations'], f['earliest_file_end_day'], f['latest_file_end_day']), (['BBB2'], '2026-10-20', '2026-11-30'))
        self.assertEqual((f['stations_ending_after_scored_day'], f['stations_without_file_end']), (['CCC3'], ['DDD4']))
        self.assertIsNone(f['station_asof_end_day']['CCC3'], 'a future file end does not prove a scored-day sample')
        self.assertEqual(f['station_asof_end_upper_bound_day']['CCC3'], '2026-11-12')
        self.assertFalse(f['all_carrier_file_ends_known'])
        self.assertNotIn('ZZZ9', f['station_file_end_day'], 'an undeclared station is not counted')

    def test_station_file_freshness_uses_the_actual_fallback_and_binds_its_bytes(self):
        recs = [dict(file='AAA1.igs20.tenv', data_end_after='2026-11-10', sha256_after='a' * 64),
                dict(file='AAA1.tenv3', data_end_after='2026-10-20', sha256_after='b' * 64)]
        selected = {'AAA1': dict(file='AAA1.tenv3', frame='IGS14', sha256='b' * 64)}
        f = RUN.station_file_freshness(recs, ['AAA1'], SCORED, selected_files=selected)
        self.assertEqual((f['station_file_end_day']['AAA1'], f['station_file_kind']['AAA1']), ('2026-10-20', 'IGS14'))
        self.assertTrue(f['all_carrier_file_ends_known'])
        selected['AAA1']['sha256'] = 'c' * 64
        f = RUN.station_file_freshness(recs, ['AAA1'], SCORED, selected_files=selected)
        self.assertIsNone(f['station_file_end_day']['AAA1'])
        self.assertEqual(f['unknown_reasons']['AAA1'], 'SELECTED_FILE_REFRESH_RECORD_UNBOUND')

    def test_station_file_freshness_missing_or_conflicting_metadata_does_not_break_scoring(self):
        selected = {'AAA1': dict(file='AAA1.igs20.tenv', frame='IGS20', sha256='a' * 64)}
        rec = dict(file='AAA1.igs20.tenv', data_end_after='2026-10-20', sha256_after='a' * 64)
        for recs, chosen, reason in [([rec], {}, 'SELECTED_FILE_IDENTITY_UNKNOWN'),
                                     ([dict(rec, data_end_after='not-a-date')], selected, 'FILE_END_UNKNOWN'),
                                     ([rec, dict(rec, data_end_after='2026-10-21')], selected, 'FILE_END_RECORD_CONFLICT')]:
            with self.subTest(reason=reason):
                f = RUN.station_file_freshness(recs, ['AAA1'], SCORED, selected_files=chosen)
                self.assertIsNone(f['station_file_end_day']['AAA1'])
                self.assertEqual(f['unknown_reasons']['AAA1'], reason)

    def test_a_legacy_rewrite_of_the_mutable_catalog_does_not_change_candidate_scoring(self):
        """The legacy NGLLiveAcquisition refetches the cache catalog on its 7-day wall clock. Scoring reads the as-of
        baseline's STORED catalog, so the rewrite neither strands the day nor leaks into the measurement."""
        self.calibrate()
        now = time.time()
        cat = self.cache / 'ngl_catalog.json'
        os.utime(cat, (now, now - 8 * 86400))
        legacy = LD.NGLLiveAcquisition(self.cache)
        legacy._session = FakeSession(lambda url: (200, llh([x for x in self.stations if x[0] != 'QTK4'] + [('QNEW', 35.3, 139.3)])))
        legacy.load_station_catalog()                                            # production's 7-day refetch
        s = self.serve()
        ctx = self.run_ctx(s, SCORED + timedelta(days=2), 'tick-45')
        res = RUN.scoring_run(ctx, ['tokyo_kanto'], SCORED)
        r = res['tokyo_kanto']
        self.assertEqual(r['status'], 'OK', r.get('refusal'))
        self.assertEqual(RF.catalog_bound(self.base, SCORED, self.cache, ['tokyo_kanto'])['regions']['tokyo_kanto']['status'],
                         'CATALOG_CHANGED_SINCE_CALIBRATION')                   # the mutable copy did move
        self.assertEqual(r['trace']['input_selection']['catalog_sha256'], r['gate']['detail'])   # the bound one was read
        qtk4 = [x for x in r['refresh'] if x['file'] == 'QTK4.igs20.tenv']          # dropped from the mutable catalog only
        self.assertTrue(qtk4 and qtk4[0]['outcome'].startswith('REFRESHED'), qtk4)    # refreshed via the bound catalog
        self.assertNotIn('window_usable false', qtk4[0]['validation_detail'])     # validated WITH its usable window

    def test_a_producer_refusal_reaches_the_ensemble_by_name(self):
        self.calibrate()
        os.remove(self.cache / 'QTK4.igs20.tenv')
        os.remove(self.src / 'QTK4.igs20.tenv')                                  # NGL now 404s it too
        s = self.serve()
        ctx = self.run_ctx(s, SCORED + timedelta(days=2), 'tick-45')
        r = RUN.scoring_run(ctx, ['tokyo_kanto'], SCORED)['tokyo_kanto']
        self.assertIsNone(r['ratio'])
        self.assertEqual(r['refusal'], ('DECLARED_CARRIER_STATION_UNAVAILABLE', 'QTK4'))
        out = RD.run_region_assessment('tokyo_kanto', datetime(2026, 11, 4), use_seismic=False,
                                       lambda_geo_provenance=r['provenance'], lg_candidate_policy=POLICY,
                                       lg_candidate_refusal=r['refusal'])
        lg = out.components['lambda_geo']
        self.assertFalse(lg.available)
        self.assertTrue(lg.eligibility_reason.startswith('DECLARED_CARRIER_STATION_UNAVAILABLE: QTK4'))

    def test_one_retry_budget_is_shared_across_regions_in_a_run(self):
        self.calibrate()
        failing = {c for c, _, _ in TT.STATIONS} | {c for c, _, _ in KUMAMOTO}
        s = self.serve(failing=failing)
        ctx = self.run_ctx(s, SCORED + timedelta(days=2), 'tick-45-outage')
        self.sleeps.clear()
        res = RUN.scoring_run(ctx, ['tokyo_kanto', 'kumamoto'], SCORED)
        n_files = len(failing)
        self.assertEqual(len(s.calls), n_files + RF.RETRY_BUDGET)                # shared: 10 + 10 = 20, not 30
        self.assertEqual(ctx.acquisition.retries_left, 0)
        self.assertEqual(len(self.sleeps), RF.RETRY_BUDGET)
        outcomes = [x['outcome'] for r in res.values() for x in r['refresh']]
        self.assertEqual(outcomes, ['FAILED_KEPT_PREVIOUS'] * n_files)
        self.assertEqual({r['status'] for r in res.values()}, {'OK'})            # kept bytes, with their true age

    def test_a_geonet_region_is_refused_by_name(self):
        s = self.serve()
        ctx = self.run_ctx(s, SCORED, 'tick')
        res = RUN.scoring_run(ctx, ['kaikoura'], SCORED)
        self.assertEqual(res['kaikoura']['refusal'][0], 'REFRESH_POLICY_NGL_ONLY')
        self.assertEqual(s.calls, [])


if __name__ == '__main__':
    unittest.main()
