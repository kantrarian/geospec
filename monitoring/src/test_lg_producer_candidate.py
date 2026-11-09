"""lg-producer-candidate-v1 tests (cayley 2026-11-09). Synthetic retained GNSS cache in NGL's file layouts (IGS20 tenv:
decimal year in column 2, E/N/U metres in columns 7-9; IGS14 tenv3: E/N/U fractional metres in columns 8/10/12;
catalog longitudes in NGL's -360..0 convention) parsed by the UNCHANGED production parsers and operator, then the
real GeoSpecEnsemble combiner and the issued 2026-11-04 record. No network; every file lives in a temp dir.
The fixture layout follows NGL's documented columns and the production parsers; no retained real NGL file exists on
this host, so the Tokyo/Kumamoto trace over the host cache is the real-bytes check."""
import copy
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

import calibration_eligibility_fixtures as FX
FX.install_stubs()
import ensemble as E                       # noqa: E402
import lg_integration_candidate as LGI     # noqa: E402
import lg_numerator_support_rule as R      # noqa: E402
import lg_producer_candidate as P          # noqa: E402
import lg_candidate_trace as T             # noqa: E402
import live_data as LD                     # noqa: E402
import run_ensemble_daily as RD            # noqa: E402

HERE = Path(__file__).resolve().parent
ISSUED_1104 = HERE.parent.parent / 'docs' / 'ensemble' / '2026-11-04' / '20261106T120920984447Z-67cafafa.json'
SCORED = date(2026, 11, 4)
CALIBRATED_ON = date(2026, 10, 30)
FIRST, LAST = date(2026, 5, 1), date(2026, 11, 12)
POLICY = LGI.CandidatePolicy(SCORED.isoformat())
STATIONS = [('QTK1', 35.0, 139.0), ('QTK2', 35.4, 139.6), ('QTK3', 35.9, 139.2), ('QTK4', 35.2, 140.1),
            ('QTK5', 36.1, 140.0)]
CLOCK = '2026-11-09T00:00:00Z'             # fixture clock: recorded in the baseline file, never compared to now


def decyr(d: date) -> str:
    return '%.4f' % (d.year + (d.timetuple().tm_yday - 0.5) / 365.25)


def series(seed, days, still_until=None):
    rng = np.random.default_rng(seed)
    e = np.cumsum(rng.normal(0.0, 0.6, len(days))) + 0.05 * np.arange(len(days))
    n = np.cumsum(rng.normal(0.0, 0.6, len(days))) - 0.03 * np.arange(len(days))
    if still_until is not None:                        # a motionless stretch: lambda is exactly zero there
        k = sum(1 for d in days if d <= still_until)
        e[:k], n[:k] = 0.0, 0.0
        e[k:] -= e[k - 1] if k else 0.0
    return e, n


def write_cache(root: Path, *, last=None, tenv3_only=(), drop=(), still_until=None, first=FIRST, extra=()):
    """NGL-layout cache: ngl_catalog.json + one file per station (tenv for IGS20, tenv3 for IGS14). `extra` adds
    (code, lat, lon) stations after the standard five."""
    root.mkdir(parents=True, exist_ok=True)
    last = dict(last or {})
    stations = list(STATIONS) + list(extra)
    catalog = {code: [lat, lon - 360.0] for code, lat, lon in stations if code not in drop}
    (root / 'ngl_catalog.json').write_text(json.dumps(catalog), encoding='utf-8')
    for k, (code, lat, lon) in enumerate(stations):
        if code in drop:
            continue
        end = last.get(code, LAST)
        days = [first + timedelta(days=i) for i in range((end - first).days + 1)]
        e, n = series(100 + k, days, still_until)
        rows = []
        for d, ee, nn in zip(days, e, n):
            mjd = (d - date(1858, 11, 17)).days
            stamp = d.strftime('%y%b%d').upper()
            if code in tenv3_only:
                rows.append(f'{code} {stamp} {decyr(d)} {mjd} 2400 0 {lon - 360.0:.1f} 0 {ee / 1000:.6f} 0 {nn / 1000:.6f} '
                            f'0 0.000000 0.0000 0.00100 0.00100 0.00400 0.000 0.000 0.000')
            else:
                rows.append(f'{code} {stamp} {decyr(d)} {mjd} 2400 0 {lon - 360.0:.1f} {ee / 1000:.6f} {nn / 1000:.6f} '
                            f'0.000000 0.0000 0.00100 0.00100 0.00400 0.000 0.000 0.000')
        if code in tenv3_only:
            head = 'site YYMMMDD yyyy.yyyy __MJD week d reflon _e0(m) __east(m) ____n0(m) _north(m) u0(m) ____up(m)'
            (root / f'{code}.tenv3').write_text(head + '\n' + '\n'.join(rows) + '\n', encoding='utf-8')
        else:                                  # retained IGS20 bytes have NO header line (grassmann inventory 29b18201)
            (root / f'{code}.igs20.tenv').write_text('\n'.join(rows) + '\n', encoding='utf-8')
    return root


class _NoNet:
    def get(self, *a, **k):
        raise AssertionError('network access attempted')


def _snapshot(root: Path):
    return {p.name: p.read_bytes() for p in sorted(root.iterdir())}


class Fixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix='lgpc-')
        self.root = Path(self._tmp.name)
        self.cache = self.root / 'cache'
        self.base = self.root / 'baselines'

    def tearDown(self):
        self._tmp.cleanup()

    def calibrate(self, cache=None, on=CALIBRATED_ON, regions=('tokyo_kanto',)):
        cid = P.code_identity()
        entries = {r: P.calibrate_region(r, on, cache or self.cache, code_id=cid) for r in regions}
        return P.write_baselines(self.base, on, entries, cid, CLOCK)


class FixtureLayoutContract(unittest.TestCase):
    def test_the_fixture_writer_emits_the_byte_verified_igs20_layout(self):
        """Fixtures must carry the WRITER's layout, settled from retained bytes (grassmann inventory 29b18201: no header,
        17 fields per row, legacy-reflon-17), not the reader's comment."""
        import lg_refresh_candidate as RFC
        with tempfile.TemporaryDirectory(prefix='lgfx-') as tmp:
            write_cache(Path(tmp))
            for f in Path(tmp).glob('*.igs20.tenv'):
                lines = f.read_text(encoding='utf-8').splitlines()
                self.assertFalse(lines[0].startswith('site'), f.name)
                self.assertEqual({len(l.split()) for l in lines if l.strip()}, {RFC.IGS20_TENV_FIELDS}, f.name)
        self.assertEqual((RFC.IGS20_TENV_LAYOUT, RFC.IGS20_TENV_FIELDS), ('legacy-reflon-17', 17))


class OperatorReach(unittest.TestCase):
    def test_support_margin_covers_the_measured_reach_of_the_production_operator(self):
        """An epoch m rows before one station's last real sample still changes when that station's tail is cut, for
        every m < reach. SUPPORT_MARGIN and the rule's default EDGE_MARGIN must both be >= reach (5 since grassmann 0008)."""
        rng = np.random.default_rng(20261109)
        n = 120
        times = [datetime(2026, 7, 1, 12) + timedelta(days=i) for i in range(n)]
        coords = [(35.0, 139.0), (35.4, 139.6), (35.9, 139.2), (35.2, 140.1), (36.1, 140.0), (35.6, 139.9)]
        walk = [(np.cumsum(rng.normal(0, 1, n)), np.cumsum(rng.normal(0, 1, n))) for _ in coords]

        def grid(cut=None):
            st = []
            for k, ((lat, lon), (e, nn)) in enumerate(zip(coords, walk)):
                end = cut + 1 if (cut is not None and k == 2) else n
                st.append(LD.StationData(f'R{k}', lat, lon, times[:end], e[:end].copy(), nn[:end].copy(), np.zeros(end)))
            sc = LD.StrainComputer(smoothing_window=7)
            v, p, _ = sc.compute_velocities(st)
            strain, _ = sc.compute_strain_field(v, p)
            return LD.LambdaGeoComputer().compute(strain)
        last = n - 20
        full, cut = grid(), grid(cut=last)
        changed = [m for m in range(10) if np.nanmax(np.abs(full[last - m] - cut[last - m])) > 1e-12]
        self.assertEqual(changed, list(range(max(changed) + 1)))
        reach = max(changed) + 1
        self.assertEqual(reach, 5)
        self.assertGreaterEqual(P.SUPPORT_MARGIN, reach)
        self.assertGreaterEqual(R.EDGE_MARGIN, reach)  # grassmann 0008: the rule default now covers the measured reach


class PositiveEndToEnd(Fixture):
    def test_lg_included_through_the_real_ensemble_and_runner(self):
        write_cache(self.cache)
        before = _snapshot(self.cache)
        path = self.calibrate()
        ratio, prov, trace = P.candidate_lambda_geo('tokyo_kanto', SCORED, self.cache, self.base)
        self.assertEqual(_snapshot(self.cache), before)                     # retained inputs never written
        self.assertEqual(trace['status'], 'OK', trace)
        m = prov['measurement']
        # clocks: the observation is SUPPORT_MARGIN days before the last sample; support runs through the scored day
        self.assertEqual(m['observation_day'], (SCORED - timedelta(days=P.SUPPORT_MARGIN)).isoformat())
        self.assertEqual(m['support_through_day'], SCORED.isoformat())
        self.assertEqual(trace['grid_last'], SCORED.isoformat())
        # the baseline: the bytes selected, the window from accepted epochs, the same carrier and operator
        self.assertEqual(prov['calibration_sha256'], P._sha(path.read_bytes()))
        self.assertEqual(prov['calibrated_on'], CALIBRATED_ON.isoformat())
        self.assertEqual(prov['window_end'], (CALIBRATED_ON - timedelta(days=30 + P.SUPPORT_MARGIN)).isoformat())
        self.assertEqual(m['operator_identity'], prov['operator_identity'])
        self.assertTrue(trace['carrier_equal'] and trace['identity_equal'])
        self.assertEqual(trace['station_codes'], [s[0] for s in STATIONS])
        self.assertEqual(trace['frame'], 'IGS20')
        # the numerator reproduces from the PRODUCTION acquisition class over the same files (network blocked)
        ngl = LD.NGLLiveAcquisition(self.cache)
        ngl._session = _NoNet()
        ngl.load_station_catalog()
        prod = LD.acquire_region_data('tokyo_kanto', ngl, P.DAYS_BACK, P.as_of_instant(SCORED), None)
        i = trace['selection']['index']
        self.assertEqual(m['lambda_max'], float(np.max(prod.lambda_geo_grid[i][np.isfinite(prod.lambda_geo_grid[i])])))
        self.assertEqual(m['unclamped_ratio'], m['lambda_max'] / prov['baseline_value'])
        self.assertEqual(ratio, max(0.1, min(50.0, m['unclamped_ratio'])))
        # component and combined score through the real combiner
        ens = E.GeoSpecEnsemble('tokyo_kanto', eligibility_rule_active=True, lg_candidate_policy=POLICY)
        day = datetime(SCORED.year, SCORED.month, SCORED.day)
        ens.set_lambda_geo(day, ratio, provenance=prov)
        res = ens.compute_risk(day, methods=('lambda_geo',))
        lg = res.components['lambda_geo']
        self.assertTrue(lg.eligible_for_tiering, lg.eligibility_reason)
        self.assertEqual(lg.calibration_status, 'calibrated')
        self.assertEqual(res.method_set['included'], ['lambda_geo'])
        self.assertEqual(res.combined_risk, E.lambda_geo_to_risk(ratio))
        self.assertEqual(lg.to_dict()['baseline_provenance'], prov)
        # the runner path, LG only, with the explicit policy
        out = RD.run_region_assessment('tokyo_kanto', day, lambda_geo_ratio=ratio, use_seismic=False,
                                       lambda_geo_provenance=prov, lg_candidate_policy=POLICY)
        self.assertEqual(out.combined_risk, res.combined_risk)
        self.assertTrue(out.components['lambda_geo'].eligible_for_tiering)

    def test_the_same_input_without_the_candidate_policy_stays_excluded(self):
        write_cache(self.cache)
        self.calibrate()
        ratio, prov, _ = P.candidate_lambda_geo('tokyo_kanto', SCORED, self.cache, self.base)
        day = datetime(SCORED.year, SCORED.month, SCORED.day)
        ens = E.GeoSpecEnsemble('tokyo_kanto', eligibility_rule_active=True)
        ens.set_lambda_geo(day, ratio, provenance=prov)
        lg = ens.compute_lambda_geo_risk(day)
        self.assertFalse(lg.eligible_for_tiering)
        self.assertIn('NO_REGISTERED_FRESHNESS_POLICY', lg.eligibility_reason)
        self.assertIsNone(E.LAMBDA_GEO_BASELINE_MAX_AGE_DAYS)
        self.assertIsNone(E.LAMBDA_GEO_BASELINE_MIN_LAG_DAYS)

    def test_issued_record_replay_control_then_lg_included_combined_score(self):
        write_cache(self.cache)
        self.calibrate()
        ratio, prov, t = P.candidate_lambda_geo('tokyo_kanto', SCORED, self.cache, self.base)
        record = json.loads(ISSUED_1104.read_text(encoding='utf-8'))['regions']['tokyo_kanto']
        rep = T.replay_region('tokyo_kanto', SCORED, record, POLICY, ratio, prov, t['status'], t['detail'])
        self.assertEqual(rep['status'], 'REPLAYED')
        self.assertTrue(rep['control']['reproduces_issued'])
        thd = record['components']['seismic_thd']['risk_score']
        lg_risk = E.lambda_geo_to_risk(ratio)
        w = E.WEIGHTS
        expected = (w['seismic_thd'] * thd + w['lambda_geo'] * lg_risk) / (w['seismic_thd'] + w['lambda_geo'])
        self.assertAlmostEqual(rep['candidate']['combined_risk'], expected, places=15)
        self.assertEqual(rep['candidate']['included'], ['lambda_geo', 'seismic_thd'])
        self.assertEqual(rep['issued']['included'], ['seismic_thd'])

    def test_legacy_reproduction_uses_the_production_edge_row_and_compares_exactly(self):
        write_cache(self.cache)
        legacy = self.root / 'lambda_geo_baselines.json'
        legacy.write_text(json.dumps({'regions': {'tokyo_kanto': {'available': True, 'mean_lambda_geo': 0.05}}}),
                          encoding='utf-8')
        ngl = LD.NGLLiveAcquisition(self.cache)
        ngl._session = _NoNet()
        ngl.load_station_catalog()
        prod = LD.acquire_region_data('tokyo_kanto', ngl, 120, datetime(2026, 11, 4), None)
        expected = max(0.1, min(50.0, prod.lambda_geo_max / 0.05))
        self.assertEqual(prod.lambda_geo_max, float(np.nanmax(prod.lambda_geo_grid[-1])))   # the last (edge) row
        got = T.legacy_reproduction('tokyo_kanto', SCORED, self.cache, legacy, {'raw_value': expected})
        self.assertEqual((got['status'], got['ratio']), ('REPRODUCES_ISSUED', expected))
        got = T.legacy_reproduction('tokyo_kanto', SCORED, self.cache, legacy, {'raw_value': expected * (1 + 1e-15)})
        self.assertEqual(got['status'], 'DIFFERS_FROM_ISSUED')

    def test_trace_cli_names_why_a_replay_is_counterfactual(self):
        write_cache(self.cache)
        args = ['--cache-dir', str(self.cache), '--issued', str(ISSUED_1104), '--scored-day', '2026-11-04',
                '--calibrated-on', '2026-10-30', '--effective-scored-day', '2026-11-04',
                '--baselines-dir', str(self.base), '--region', 'tokyo_kanto']
        T.main(args + ['--generated-utc', '2026-11-03T23:00:00Z', '--out', str(self.root / 'a.json')])
        a = json.loads((self.root / 'a.json').read_text(encoding='utf-8'))
        self.assertTrue(a['baseline_file']['written_by_this_run'])
        self.assertNotIn('BASELINE_GENERATED_AFTER_SCORED_DAY', a['counterfactual']['reasons'])
        self.assertIn('CANDIDATE_POLICY_NOT_REGISTERED', a['counterfactual']['reasons'])
        self.assertEqual(a['regions']['tokyo_kanto']['replay']['status'], 'REPLAYED')
        os.remove(self.base / 'lambda_geo_baselines_rule_20261030.json')
        T.main(args + ['--generated-utc', '2026-11-09T00:00:00Z', '--out', str(self.root / 'b.json')])
        b = json.loads((self.root / 'b.json').read_text(encoding='utf-8'))
        self.assertIn('BASELINE_GENERATED_AFTER_SCORED_DAY', b['counterfactual']['reasons'])
        with self.assertRaises(FileExistsError):                                  # outputs are never overwritten
            T.main(args + ['--generated-utc', CLOCK, '--out', str(self.root / 'b.json')])

    def test_replay_control_refuses_a_record_it_does_not_reproduce(self):
        record = json.loads(ISSUED_1104.read_text(encoding='utf-8'))['regions']['tokyo_kanto']
        bent = copy.deepcopy(record)
        bent['combined_risk'] += 1e-9
        rep = T.replay_region('tokyo_kanto', SCORED, bent, POLICY, None, None, 'X', 'y')
        self.assertEqual(rep['status'], 'REPLAY_CONTROL_MISMATCH')
        self.assertNotIn('candidate', rep)


class Refusals(Fixture):
    def _component(self, ratio, prov, trace, policy=POLICY):
        day = datetime(SCORED.year, SCORED.month, SCORED.day)
        ens = E.GeoSpecEnsemble('tokyo_kanto', eligibility_rule_active=True, lg_candidate_policy=policy)
        if ratio is None:
            ens.set_lambda_geo_unavailable(day, trace['status'], trace['detail'], provenance=prov)
        else:
            ens.set_lambda_geo(day, ratio, provenance=prov)
        return ens.compute_risk(day, methods=('lambda_geo',))

    def test_a_declared_station_missing_at_scoring_refuses_by_name_and_stays_out_of_the_tier(self):
        write_cache(self.cache)
        self.calibrate()
        alt = self.root / 'cache-after'
        write_cache(alt, drop=('QTK4',))
        ratio, prov, trace = P.candidate_lambda_geo('tokyo_kanto', SCORED, alt, self.base)
        self.assertIsNone(ratio)
        self.assertEqual((trace['status'], trace['detail']), ('DECLARED_CARRIER_STATION_UNAVAILABLE', 'QTK4'))
        self.assertEqual(trace['declared_carrier'], [s[0] for s in STATIONS])
        res = self._component(ratio, prov, trace)
        lg = res.components['lambda_geo']
        self.assertFalse(lg.available)
        self.assertEqual(lg.to_dict()['calibration']['status'], 'missing')
        self.assertTrue(lg.eligibility_reason.startswith('DECLARED_CARRIER_STATION_UNAVAILABLE: QTK4'))
        self.assertIn('DECLARED_CARRIER_STATION_UNAVAILABLE', lg.notes)
        self.assertEqual(res.tier_name, 'DEGRADED')
        out = RD.run_region_assessment('tokyo_kanto', datetime(2026, 11, 4), use_seismic=False,
                                       lambda_geo_provenance=prov, lg_candidate_policy=POLICY,
                                       lg_candidate_refusal=(trace['status'], trace['detail']))
        self.assertIn('DECLARED_CARRIER_STATION_UNAVAILABLE', out.components['lambda_geo'].notes)

    def test_a_station_outside_the_declared_carrier_is_not_used(self):
        write_cache(self.cache)
        self.calibrate()
        alt = self.root / 'cache-new-station'
        write_cache(alt, extra=(('QTK7', 35.3, 139.3),))
        ratio, prov, trace = P.candidate_lambda_geo('tokyo_kanto', SCORED, alt, self.base)
        self.assertEqual(trace['status'], 'OK')
        self.assertEqual(trace['station_codes'], [s[0] for s in STATIONS])
        self.assertEqual(trace['input_selection']['ngl_attempts']['QTK7'],
                         dict(outcome='NOT_IN_DECLARED_CARRIER', files=[]))

    def test_mixed_frames_are_a_different_operator(self):
        write_cache(self.cache)
        self.calibrate()
        alt = self.root / 'cache-mixed'
        write_cache(alt, tenv3_only=('QTK2',))
        ratio, prov, trace = P.candidate_lambda_geo('tokyo_kanto', SCORED, alt, self.base)
        self.assertEqual(trace['frame'], 'MIXED:IGS14+IGS20')
        self.assertEqual(trace['status'], 'OPERATOR_MISMATCH')
        self.assertTrue(trace['carrier_equal'])

    def test_lagging_station_moves_the_observation_back_without_pruning(self):
        write_cache(self.cache, last={'QTK3': SCORED - timedelta(days=8)})
        self.calibrate()
        ratio, prov, trace = P.candidate_lambda_geo('tokyo_kanto', SCORED, self.cache, self.base)
        self.assertEqual(trace['status'], 'OK')
        self.assertEqual(prov['measurement']['observation_day'],
                         (SCORED - timedelta(days=8 + P.SUPPORT_MARGIN)).isoformat())
        self.assertEqual(prov['measurement']['support_through_day'], (SCORED - timedelta(days=8)).isoformat())
        self.assertEqual(prov['measurement']['data_end_day'], SCORED.isoformat())
        self.assertEqual(len(trace['station_codes']), 5)

    def test_lagging_station_beyond_the_age_bound_is_typed(self):
        write_cache(self.cache, last={'QTK3': SCORED - timedelta(days=41)})
        self.calibrate()
        ratio, prov, trace = P.candidate_lambda_geo('tokyo_kanto', SCORED, self.cache, self.base)
        self.assertIsNone(ratio)
        self.assertEqual(trace['status'], 'FULL_SUPPORT_EPOCH_TOO_OLD')
        lg = self._component(ratio, prov, trace).components['lambda_geo']
        self.assertTrue(lg.eligibility_reason.startswith('FULL_SUPPORT_EPOCH_TOO_OLD'))

    def test_stale_baseline_is_excluded_by_the_candidate_policy(self):
        write_cache(self.cache)
        self.calibrate(on=date(2026, 9, 30))
        ratio, prov, trace = P.candidate_lambda_geo('tokyo_kanto', SCORED, self.cache, self.base)
        self.assertEqual(trace['status'], 'OK')
        lg = self._component(ratio, prov, trace).components['lambda_geo']
        self.assertTrue(lg.available)
        self.assertFalse(lg.eligible_for_tiering)
        self.assertEqual(lg.calibration_status, 'stale')

    def test_valid_zero_epochs_are_kept_in_the_denominator(self):
        write_cache(self.cache, still_until=date(2026, 8, 31))
        path = self.calibrate()
        entry = json.loads(path.read_text(encoding='utf-8'))['regions']['tokyo_kanto']
        self.assertTrue(entry['available'], entry)
        self.assertGreater(entry['n_zero_days'], 0)
        m, _ = P.measure('tokyo_kanto', CALIBRATED_ON - timedelta(days=30), self.cache, P.BASELINE_DAYS)
        idx = R.valid_denominator_epochs(m.support, P.SUPPORT_MARGIN)
        values = [float(np.max(m.grid[i][np.isfinite(m.grid[i])])) for i in idx]
        self.assertEqual(entry['n_samples'], len(values))
        self.assertEqual(entry['mean_lambda_geo'], float(np.mean(values)))

    def test_all_zero_baseline_is_unavailable(self):
        write_cache(self.cache, still_until=LAST)
        path = self.calibrate()
        entry = json.loads(path.read_text(encoding='utf-8'))['regions']['tokyo_kanto']
        self.assertFalse(entry['available'])
        self.assertEqual(entry['reason'], 'DENOMINATOR_UNAVAILABLE')
        ratio, prov, trace = P.candidate_lambda_geo('tokyo_kanto', SCORED, self.cache, self.base)
        self.assertEqual(trace['status'], 'BASELINE_UNAVAILABLE')
        self.assertIn('DENOMINATOR_UNAVAILABLE', trace['detail'])


class EpochsAndRevisions(Fixture):
    def test_appending_later_samples_revises_only_provisional_epochs(self):
        """Measure the same carrier as of D and as of D+7 (seven more daily rows exist). Every epoch FINAL as of D is
        bit-identical as of D+7 and still final; at least one PROVISIONAL epoch of D revises."""
        write_cache(self.cache)
        early, _ = P.measure('tokyo_kanto', SCORED, self.cache)
        late, _ = P.measure('tokyo_kanto', SCORED + timedelta(days=7), self.cache)
        self.assertEqual(early.station_codes, late.station_codes)
        a = {d: (v, f) for d, v, f in P.epoch_series(early)}
        b = {d: (v, f) for d, v, f in P.epoch_series(late)}
        common = sorted(set(a) & set(b))
        final_a = [d for d in common if a[d][1]]
        start_b = min(b)
        comparable = [d for d in final_a if date.fromisoformat(d) >= date.fromisoformat(start_b) + timedelta(days=P.SUPPORT_MARGIN)]
        self.assertGreater(len(comparable), 50)
        for d in comparable:
            self.assertEqual(a[d][0], b[d][0], d)
            self.assertTrue(b[d][1], d)
        provisional = [d for d in common if not a[d][1] and date.fromisoformat(d) > SCORED - timedelta(days=P.SUPPORT_MARGIN)]
        self.assertEqual(len(provisional), P.SUPPORT_MARGIN)
        self.assertTrue(any(a[d][0] != b[d][0] for d in provisional))
        self.assertEqual(max(d for d in a if a[d][1]), (SCORED - timedelta(days=P.SUPPORT_MARGIN)).isoformat())

    def test_selected_observation_is_the_newest_final_epoch_and_clocks_stay_separate(self):
        write_cache(self.cache, last={c: SCORED - timedelta(days=11) for c, _, _ in STATIONS})
        self.calibrate()
        ratio, prov, trace = P.candidate_lambda_geo('tokyo_kanto', SCORED, self.cache, self.base)
        m = prov['measurement']
        self.assertEqual(m['data_end_day'], (SCORED - timedelta(days=11)).isoformat())
        self.assertEqual(m['observation_day'], (SCORED - timedelta(days=11 + P.SUPPORT_MARGIN)).isoformat())
        self.assertEqual(m['support_through_day'], m['data_end_day'])
        self.assertEqual(m['observation_age_days'], 11 + P.SUPPORT_MARGIN)
        self.assertEqual(m['as_of'], '2026-11-04T23:59:59')
        finals = [d for d, v, f in m['series'] if f]
        self.assertEqual(finals[-1], m['observation_day'])
        self.assertEqual([v for d, v, f in m['series'] if d == m['observation_day']], [m['lambda_max']])
        self.assertEqual(sorted(m['retrieval']['files_mtime_utc']), sorted(trace['station_codes']))

    def test_retrieval_clock_is_recorded_beside_not_inside_the_content_identity(self):
        write_cache(self.cache)
        self.calibrate()
        _, p1, t1 = P.candidate_lambda_geo('tokyo_kanto', SCORED, self.cache, self.base)
        f = self.cache / 'QTK2.igs20.tenv'
        st = f.stat()
        os.utime(f, (st.st_atime, st.st_mtime - 86400 * 3))
        _, p2, t2 = P.candidate_lambda_geo('tokyo_kanto', SCORED, self.cache, self.base)
        self.assertEqual(t1['input_sha256'], t2['input_sha256'])
        self.assertNotEqual(p1['measurement']['retrieval']['files_mtime_utc']['QTK2'],
                            p2['measurement']['retrieval']['files_mtime_utc']['QTK2'])
        self.assertEqual(p2['measurement']['retrieval']['oldest_mtime_utc'],
                         p2['measurement']['retrieval']['files_mtime_utc']['QTK2'])

    def test_an_included_ratio_in_the_score_dead_band_still_moves_the_combined_score(self):
        """value -> method score -> effective weight -> combined: an eligible ratio <= 1.5 maps to score 0 under the
        unchanged mapping, yet joins the included set with its weight, so the combined score falls below THD alone."""
        write_cache(self.cache)
        path = self.calibrate()
        _, prov, _ = P.candidate_lambda_geo('tokyo_kanto', SCORED, self.cache, self.base)
        doc = json.loads(path.read_text(encoding='utf-8'))
        doc['regions']['tokyo_kanto']['mean_lambda_geo'] = prov['measurement']['lambda_max'] / 1.2
        other = self.root / 'baselines-deadband'
        other.mkdir()
        (other / path.name).write_text(json.dumps(doc), encoding='utf-8')
        ratio, prov, t = P.candidate_lambda_geo('tokyo_kanto', SCORED, self.cache, other)
        self.assertAlmostEqual(ratio, 1.2, places=12)
        self.assertEqual(E.lambda_geo_to_risk(ratio), 0.0)
        record = json.loads(ISSUED_1104.read_text(encoding='utf-8'))['regions']['tokyo_kanto']
        rep = T.replay_region('tokyo_kanto', SCORED, record, POLICY, ratio, prov, t['status'], t['detail'])
        cand = rep['candidate']
        self.assertEqual(cand['included'], ['lambda_geo', 'seismic_thd'])
        self.assertEqual(cand['effective_weights'], {'lambda_geo': 0.4 / 0.7, 'seismic_thd': 0.3 / 0.7})
        thd = record['components']['seismic_thd']['risk_score']
        self.assertAlmostEqual(cand['combined_risk'], 0.3 * thd / 0.7, places=15)
        self.assertLess(cand['combined_risk'], rep['issued']['combined_risk'])


class InputSelection(Fixture):
    """codex bc6c1542: a station that qualifies only in the CALIBRATION window. On the complete cache the carriers
    differ and the producer refuses by name; a partial copy that loses that file would calibrate without it and match
    falsely, so the two-window selection record must expose the copy before any trace uses it."""
    EARLY = ('QTK6', 35.8, 140.6)
    EARLY_LAST = SCORED - timedelta(days=42)            # admitted at calibration (ends 7 d before 09-30); lags at scoring
    DEAD_LAST = SCORED - timedelta(days=93)             # 2026-08-03 = G107: 33 samples parse; rule alone keeps 23 < 30 epochs

    def complete_and_partial(self, last=None):
        write_cache(self.cache, extra=(self.EARLY,), last={'QTK6': last or self.EARLY_LAST})
        copy = self.root / 'partial-copy'
        shutil.copytree(self.cache, copy)
        os.remove(copy / 'QTK6.igs20.tenv')
        return copy

    def test_an_admitted_station_stays_in_the_declared_carrier_and_its_lag_is_reported_by_name(self):
        self.complete_and_partial()
        end = P.calibration_window_end(CALIBRATED_ON)
        cal = P.select_inputs('tokyo_kanto', end, self.cache, P.BASELINE_DAYS, admit_until=P.admission_day(end))
        self.assertEqual(cal['ngl_attempts']['QTK6']['outcome'], 'PARSED')
        self.assertIn('QTK6', cal['carrier'])
        self.calibrate()
        ratio, prov, trace = P.candidate_lambda_geo('tokyo_kanto', SCORED, self.cache, self.base)
        self.assertIsNone(ratio)
        self.assertEqual(trace['status'], 'FULL_SUPPORT_EPOCH_TOO_OLD')
        self.assertIn('QTK6', trace['declared_carrier'])
        self.assertEqual(prov['measurement']['observation_day'], (self.EARLY_LAST - timedelta(days=P.SUPPORT_MARGIN)).isoformat())

    def test_a_station_ending_long_before_the_window_end_is_not_admitted_by_name(self):
        """The Tokyo/G107 shape: under support rule v3 alone it would cut the baseline to ~20 epochs; the declared
        carrier names it NOT_ADMITTED with its last sample day and calibrates without it."""
        self.complete_and_partial(last=self.DEAD_LAST)
        path = self.calibrate()
        entry = json.loads(path.read_text(encoding='utf-8'))['regions']['tokyo_kanto']
        self.assertTrue(entry['available'], entry.get('reason'))
        self.assertEqual(entry['admission']['not_admitted'], {'QTK6': self.DEAD_LAST.isoformat()})
        self.assertEqual(entry['admission']['admit_until'], '2026-09-23')
        self.assertNotIn('QTK6', entry['station_codes'])
        self.assertEqual(entry['selection']['ngl_attempts']['QTK6']['outcome'], 'NOT_ADMITTED')
        now = P.select_inputs('tokyo_kanto', SCORED, self.cache)                  # present, no sample in this window
        self.assertEqual(now['ngl_attempts']['QTK6']['outcome'], 'UNPARSED')
        ratio, prov, trace = P.candidate_lambda_geo('tokyo_kanto', SCORED, self.cache, self.base)
        self.assertEqual(trace['status'], 'OK')
        self.assertEqual(trace['input_selection']['ngl_attempts']['QTK6']['outcome'], 'NOT_IN_DECLARED_CARRIER')
        legacy = R.denominator_from_grid(*(lambda m: (m.times, m.support, m.grid))(
            P.measure('tokyo_kanto', P.calibration_window_end(CALIBRATED_ON), self.cache, P.BASELINE_DAYS)[0]),
            edge_margin=P.SUPPORT_MARGIN)
        self.assertEqual(legacy['reason'], 'INSUFFICIENT_VALID_DAYS')       # what the rule alone does with that station

    def test_a_partial_copy_matches_falsely_and_the_selection_check_names_the_missing_file(self):
        copy = self.complete_and_partial()
        self.calibrate(cache=copy)
        ratio, prov, trace = P.candidate_lambda_geo('tokyo_kanto', SCORED, copy, self.base)
        self.assertEqual(trace['status'], 'OK')    # the hazard: the complete cache says FULL_SUPPORT_EPOCH_TOO_OLD
        end = P.calibration_window_end(CALIBRATED_ON)
        adm = P.admission_day(end)
        diffs = P.compare_selections(P.select_inputs('tokyo_kanto', end, self.cache, P.BASELINE_DAYS, admit_until=adm),
                                     P.select_inputs('tokyo_kanto', end, copy, P.BASELINE_DAYS, admit_until=adm))
        rows = [d for d in diffs if d.get('station') == 'QTK6']
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]['reference']['outcome'], rows[0]['candidate']['outcome']), ('PARSED', 'ABSENT'))
        self.assertEqual(rows[0]['candidate']['files'], [dict(file='QTK6.igs20.tenv', present=False),
                                                         dict(file='QTK6.tenv3', present=False)])
        self.assertIn('carrier', [d['field'] for d in diffs])
        self.assertEqual(P.compare_selections(P.select_inputs('tokyo_kanto', end, self.cache, P.BASELINE_DAYS),
                                              P.select_inputs('tokyo_kanto', end, self.cache, P.BASELINE_DAYS)), [])

    def _args(self, cache, out, extra=()):
        return ['--cache-dir', str(cache), '--issued', str(ISSUED_1104), '--scored-day', '2026-11-04',
                '--calibrated-on', '2026-10-30', '--effective-scored-day', '2026-11-04', '--baselines-dir', str(self.base),
                '--region', 'tokyo_kanto', '--generated-utc', CLOCK, '--out', str(out)] + list(extra)

    def test_trace_refuses_a_copy_whose_selection_differs_and_writes_nothing(self):
        copy = self.complete_and_partial()
        out = self.root / 'refused.json'
        rc = T.main(self._args(copy, out, ['--selection-reference', str(self.cache)]))
        self.assertEqual(rc, 4)
        self.assertFalse(out.exists())
        self.assertFalse(self.base.exists() and any(self.base.iterdir()))
        rc = T.main(self._args(self.cache, out, ['--selection-reference', str(self.cache)]))
        self.assertEqual(rc, 0)
        doc = json.loads(out.read_text(encoding='utf-8'))
        self.assertEqual(doc['input_selection_reference'], str(self.cache))
        self.assertEqual(doc['regions']['tokyo_kanto']['producer']['status'], 'FULL_SUPPORT_EPOCH_TOO_OLD')
        self.assertIn('QTK6', doc['input_selection']['tokyo_kanto']['calibration']['carrier'])

    def test_trace_refuses_inputs_that_change_during_the_run(self):
        write_cache(self.cache)
        real = P.candidate_lambda_geo

        def touching(*a, **k):
            f = self.cache / 'QTK1.igs20.tenv'
            f.write_bytes(b'# appended while tracing\n' + f.read_bytes())
            return real(*a, **k)
        out = self.root / 'changed.json'
        with mock.patch.object(P, 'candidate_lambda_geo', side_effect=touching):
            rc = T.main(self._args(self.cache, out))
        self.assertEqual(rc, 3)
        self.assertFalse(out.exists())

    def test_a_reused_baseline_says_whether_it_came_from_these_inputs(self):
        write_cache(self.cache)
        T.main(self._args(self.cache, self.root / 'one.json'))
        one = json.loads((self.root / 'one.json').read_text(encoding='utf-8'))
        self.assertEqual(one['regions']['tokyo_kanto']['baseline_selection_check'], 'SAME_INPUTS_AS_THIS_CACHE')
        f = self.cache / 'QTK1.igs20.tenv'
        f.write_bytes(b'# a later fetch\n' + f.read_bytes())
        T.main(self._args(self.cache, self.root / 'two.json'))
        two = json.loads((self.root / 'two.json').read_text(encoding='utf-8'))
        self.assertFalse(two['baseline_file']['written_by_this_run'])
        self.assertEqual(two['regions']['tokyo_kanto']['baseline_selection_check'], 'BASELINE_FROM_OTHER_INPUTS')


class BaselineFiles(Fixture):
    def test_dated_files_are_immutable_and_selected_as_of(self):
        write_cache(self.cache)
        early = self.calibrate(on=date(2026, 10, 1))
        late = self.calibrate(on=date(2026, 11, 3))
        raw = late.read_bytes()
        with self.assertRaises(FileExistsError):
            self.calibrate(on=date(2026, 11, 3))
        self.assertEqual(late.read_bytes(), raw)
        self.assertEqual(P.select_as_of(self.base, date(2026, 11, 2))[0]['path'], early)
        self.assertEqual(P.select_as_of(self.base, date(2026, 11, 3))[0]['path'], late)
        self.assertEqual(P.select_as_of(self.base, date(2026, 9, 30)), (None, 'NO_BASELINE_AS_OF_SCORED_DAY'))

    def test_a_file_whose_name_and_content_disagree_is_refused_not_skipped(self):
        write_cache(self.cache)
        self.calibrate(on=date(2026, 10, 1))
        late = self.calibrate(on=date(2026, 10, 30))
        doc = json.loads(late.read_text(encoding='utf-8'))
        doc['calibrated_on'] = '2026-10-29'
        late.write_text(json.dumps(doc), encoding='utf-8')
        self.assertEqual(P.select_as_of(self.base, SCORED), (None, 'BASELINE_FILE_INCONSISTENT'))

    def test_a_file_from_another_support_policy_is_refused(self):
        write_cache(self.cache)
        path = self.calibrate()
        for key, value in (('support_margin', 4), ('days_back', 120)):
            doc = json.loads(path.read_text(encoding='utf-8'))
            doc[key] = value
            path.write_text(json.dumps(doc), encoding='utf-8')
            self.assertEqual(P.select_as_of(self.base, SCORED), (None, 'BASELINE_FILE_INCONSISTENT'), key)

    def test_input_bytes_are_bound_even_when_the_value_is_not_changed(self):
        write_cache(self.cache)
        self.calibrate()
        _, p1, t1 = P.candidate_lambda_geo('tokyo_kanto', SCORED, self.cache, self.base)
        for code, row in t1['input_manifest']['stations'].items():
            raw = (self.cache / row['file']).read_bytes()
            self.assertEqual((row['sha256'], row['bytes']), (P._sha(raw), len(raw)), code)
        self.assertEqual(t1['input_manifest']['catalog_sha256'], P._sha((self.cache / 'ngl_catalog.json').read_bytes()))
        self.assertEqual(t1['input_sha256'], P._sha(P.canonical(t1['input_manifest'])))
        f = self.cache / 'QTK1.igs20.tenv'
        f.write_bytes(b'# retained comment line\n' + f.read_bytes())
        _, p2, t2 = P.candidate_lambda_geo('tokyo_kanto', SCORED, self.cache, self.base)
        self.assertEqual(p1['measurement']['lambda_max'], p2['measurement']['lambda_max'])
        self.assertNotEqual(t1['input_sha256'], t2['input_sha256'])
        self.assertEqual(t1['status'], t2['status'])

    def test_no_baseline_and_no_catalog_are_typed(self):
        write_cache(self.cache)
        r = P.candidate_lambda_geo('tokyo_kanto', SCORED, self.cache, self.base)
        self.assertEqual((r[0], r[1], r[2]['status']), (None, None, 'NO_BASELINE_AS_OF_SCORED_DAY'))
        self.calibrate()
        os.remove(self.cache / 'ngl_catalog.json')
        r = P.candidate_lambda_geo('tokyo_kanto', SCORED, self.cache, self.base)
        self.assertEqual(r[2]['status'], 'RETAINED_CATALOG_ABSENT')

    def test_retained_acquisition_refuses_network(self):
        with self.assertRaises(RuntimeError):
            P.RetainedNGL(self.cache).session
        with self.assertRaises(RuntimeError):
            P.RetainedGeoNet(self.cache).session

    def test_refusal_hook_requires_the_explicit_candidate_policy(self):
        ens = E.GeoSpecEnsemble('tokyo_kanto', eligibility_rule_active=True)
        with self.assertRaises(ValueError):
            ens.set_lambda_geo_unavailable(datetime(2026, 11, 4), 'X', 'y')
        with self.assertRaises(ValueError):
            RD.run_region_assessment('tokyo_kanto', datetime(2026, 11, 4), use_seismic=False,
                                     lg_candidate_refusal=('X', 'y'))
        with self.assertRaises(ValueError):
            RD.run_region_assessment('tokyo_kanto', datetime(2026, 10, 1), use_seismic=False,
                                     lg_candidate_policy=POLICY)


if __name__ == '__main__':
    unittest.main()
