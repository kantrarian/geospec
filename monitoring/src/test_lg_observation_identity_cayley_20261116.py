"""lg-observation-content-v1 tests (cayley 2026-11-16; codex 5851b0649 / 8c8d84db9).

WRITER-BUILT. NGL-layout caches are written by the producer tests' writer (test_lg_producer_candidate.write_cache) and
served by the runner tests' recording fake session. They are refreshed, calibrated and scored by the REAL
lg_candidate_runner, lg_producer_candidate and live_data operator, and the identities are read back by the REAL
method_comparability reader. The GNSS values are synthetic. No retained real NGL bytes exist on this host, so
grassmann's retained-input reproduction (tick 51/52 Tokyo and Kumamoto) is the real-bytes check.

OPERATOR-LEVEL. A nondegenerate synthetic carrier, with an interior gap that crosses the window edge, goes through the
production operator functions. Perturbations CORROBORATE the closure that the module derives from the source
composition: a change outside the closure leaves EVERY triangle value at the epoch bit-identical (not merely the
maximum, which can hide a change). They do not replace that argument, and the per-run recomputation check enforces it.
"""
import json
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

import test_lg_candidate_runner as RT          # installs the stubs; the runner fixture and its writer
import live_data as LD                         # noqa: E402
import lg_numerator_support_rule as R          # noqa: E402
import lg_observation_identity as OI           # noqa: E402
import method_comparability as MC              # noqa: E402

TT, RUN, P, RD = RT.TT, RT.RUN, RT.P, RT.RD
END = RT.SCORED - timedelta(days=8)            # every carrier file ends here, so the selected epoch stops advancing
EPOCH = END - timedelta(days=P.SUPPORT_MARGIN)
REGION = 'tokyo_kanto'


def row_of(prov):
    """The issued-row shape the reader reads: components.lambda_geo.baseline_provenance (through JSON, as issued)."""
    return {'components': {'lambda_geo': {'baseline_provenance': json.loads(json.dumps(prov))}}}


def step(newer_prov, older_prov):
    return MC.observation_step(MC.observation_identity('lambda_geo', row_of(newer_prov)),
                               MC.observation_identity('lambda_geo', row_of(older_prov)))


def edit(path: Path, day: date, *, de_mm=0.0, dn_mm=0.0, drop=False):
    """Change ONE retained IGS20 row (found by its MJD column) in place: shift E/N by mm, or drop the row."""
    mjd = str((day - date(1858, 11, 17)).days)
    out, hit = [], 0
    for line in path.read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if len(parts) > 8 and parts[3] == mjd:
            hit += 1
            if drop:
                continue
            parts[7] = '%.6f' % (float(parts[7]) + de_mm / 1000.0)
            parts[8] = '%.6f' % (float(parts[8]) + dn_mm / 1000.0)
            line = ' '.join(parts)
        out.append(line)
    assert hit == 1, (path.name, day, hit)
    path.write_text('\n'.join(out) + '\n', encoding='utf-8')


class WriterBuilt(RT.Base):
    def setUp(self):
        super().setUp()
        last = {c: END for c, _, _ in list(TT.STATIONS) + list(RT.KUMAMOTO)}
        last['QTK9'] = date(2026, 8, 3)
        TT.write_cache(self.src, extra=tuple(RT.KUMAMOTO) + (RT.DEAD,), last=last)
        self.calibrate()

    def scored(self, day, run_id):
        ctx = self.run_ctx(self.serve(), day + timedelta(days=2), run_id)
        return RUN.scoring_run(ctx, [REGION], day)[REGION]

    def bound(self, day):
        store = RUN.catalog_store(self.base)
        selected, _ = P.select_as_of(self.base, day)
        entry = selected['doc']['regions'][REGION]
        return store, entry, P.resolve_bound_catalog(store, entry)[0]

    def direct(self, day, margin=OI.MARGIN):
        """The producer and the identity over the CURRENT cache bytes, no refresh (so in-place edits stay as made)."""
        store, entry, path = self.bound(day)
        ratio, prov, trace = P.candidate_lambda_geo(REGION, day, self.cache, self.base, catalog_store=store)
        self.assertEqual(trace['status'], 'OK', trace.get('detail'))
        oc, rec = OI.observation_content(REGION, day, self.cache, path, trace, prov['measurement'], entry, margin=margin)
        prov['measurement']['observation_content'] = oc
        return ratio, prov, trace, rec

    # ---- the defect: unchanged files on two report days ------------------------------------------------------------
    def test_unchanged_files_on_two_report_days_are_one_observation(self):
        a = self.scored(RT.SCORED, 'tick-a')
        b = self.scored(RT.SCORED + timedelta(days=1), 'tick-b')
        ma, mb = a['provenance']['measurement'], b['provenance']['measurement']
        self.assertEqual((ma['observation_day'], mb['observation_day']), (EPOCH.isoformat(),) * 2)
        self.assertEqual(b['trace']['selection']['index'], a['trace']['selection']['index'] - 1,
                         'the rolling window start moved by one row')
        self.assertNotEqual(ma['input_sha256'], mb['input_sha256'], 'the whole-product digest embeds the as-of')
        self.assertNotEqual(ma['input_selection_sha256'], mb['input_selection_sha256'])
        oa, ob = ma['observation_content'], mb['observation_content']
        self.assertEqual((oa['status'], ob['status']), ('KNOWN', 'KNOWN'), (oa, ob))
        self.assertEqual(oa['sha256'], ob['sha256'])
        self.assertEqual(oa['window'], [(EPOCH - timedelta(days=5)).isoformat(), END.isoformat()])
        self.assertEqual(a['observation_record'], b['observation_record'])
        self.assertEqual(OI._sha(OI.canonical(a['observation_record'])), oa['sha256'], 'the digest is of the record')
        self.assertEqual(oa['closure_recompute'], {'equal': True, 'lambda_max': ma['lambda_max']})
        self.assertEqual(step(b['provenance'], a['provenance']), MC.STEP_REPEATED)
        # through the real assessment and the issued-row form, the reader gives the same answer
        outs = [RD.run_region_assessment(REGION, datetime(2026, 11, 4), lambda_geo_ratio=r['ratio'], use_seismic=False,
                                         lambda_geo_provenance=r['provenance'], lg_candidate_policy=RT.POLICY,
                                         lg_candidate_refusal=r['refusal']) for r in (a, b)]
        prior = json.loads(json.dumps(outs[0].to_dict()))
        self.assertEqual(MC.observation_step(MC.observation_identity('lambda_geo', outs[1]),
                                             MC.observation_identity('lambda_geo', prior)), MC.STEP_REPEATED)

    def test_a_later_supported_epoch_is_an_advance(self):
        _, p0, _, _ = self.direct(RT.SCORED)
        last = {c: END + timedelta(days=1) for c, _, _ in list(TT.STATIONS) + list(RT.KUMAMOTO)}
        last['QTK9'] = date(2026, 8, 3)
        TT.write_cache(self.cache, extra=tuple(RT.KUMAMOTO) + (RT.DEAD,), last=last)   # every file grows by one day
        _, p1, _, _ = self.direct(RT.SCORED + timedelta(days=1))
        self.assertEqual(p1['measurement']['observation_day'], (EPOCH + timedelta(days=1)).isoformat())
        self.assertEqual(step(p1, p0), MC.STEP_ADVANCED)
        self.assertEqual(step(p0, p1), MC.STEP_REGRESSED)

    # ---- outside / inside the closure -----------------------------------------------------------------------------------
    def test_a_change_outside_the_closure_keeps_the_observation(self):
        f = self.cache / 'QTK2.igs20.tenv'
        _, p0, _, _ = self.direct(RT.SCORED)
        original = f.read_bytes()
        for back in (6, 20, 80):
            with self.subTest(days_before_epoch=back):
                f.write_bytes(original)
                edit(f, EPOCH - timedelta(days=back), de_mm=2.0, dn_mm=-3.0)
                _, p1, _, _ = self.direct(RT.SCORED)
                m0, m1 = p0['measurement'], p1['measurement']
                self.assertNotEqual(m0['input_sha256'], m1['input_sha256'], 'the whole-file provenance moved')
                self.assertEqual(m0['lambda_max'], m1['lambda_max'])
                self.assertEqual(m0['observation_content']['sha256'], m1['observation_content']['sha256'])
                self.assertEqual(step(p1, p0), MC.STEP_REPEATED)

    def test_a_change_inside_the_closure_is_a_revision(self):
        f = self.cache / 'QTK2.igs20.tenv'
        _, p0, _, _ = self.direct(RT.SCORED)
        original = f.read_bytes()
        for offset, de, dn in ((-5, 2.0, 0.0), (-5, -2.0, 0.0), (0, 0.0, 0.001), (3, -50.0, 50.0), (5, 0.001, 0.0)):
            with self.subTest(offset=offset, de=de, dn=dn):
                f.write_bytes(original)
                edit(f, EPOCH + timedelta(days=offset), de_mm=de, dn_mm=dn)
                _, p1, _, _ = self.direct(RT.SCORED)
                self.assertEqual(p1['measurement']['observation_day'], EPOCH.isoformat())
                self.assertEqual(p1['measurement']['observation_content']['status'], 'KNOWN')
                self.assertNotEqual(p1['measurement']['observation_content']['sha256'],
                                    p0['measurement']['observation_content']['sha256'])
                self.assertEqual(step(p1, p0), MC.STEP_REVISED)

    def test_a_gap_bracket_outside_the_window_is_part_of_the_closure(self):
        f = self.cache / 'QTK3.igs20.tenv'
        for back in (6, 5, 4):
            edit(f, EPOCH - timedelta(days=back), drop=True)
        _, g0, _, rec = self.direct(RT.SCORED)
        oc = g0['measurement']['observation_content']
        self.assertEqual((oc['status'], g0['measurement']['observation_day']), ('KNOWN', EPOCH.isoformat()), oc)
        self.assertEqual((oc['n_interpolated_rows'], oc['n_brackets_outside_window']), (2, 2))
        qtk3 = next(s for s in rec['carrier'] if s['code'] == 'QTK3')
        self.assertEqual([r[1:4] for r in qtk3['rows'] if r[0] == 'I'], [[-5, -7, -3], [-4, -7, -3]])
        saved = f.read_bytes()
        edit(f, EPOCH - timedelta(days=7), de_mm=2.0)               # the left bracket, outside the window
        _, g1, _, rec1 = self.direct(RT.SCORED)
        # The maximum over triangles need not move (here it does not: QTK3's triangles are not the maximum), which is
        # why the identity hashes the consumed inputs, never the reduced value. The bracket's numerical effect on the
        # triangle values is shown at triangle level in OperatorClosurePerturbation.
        self.assertEqual(g1['measurement']['observation_content']['status'], 'KNOWN')
        self.assertNotEqual([s for s in rec1['carrier'] if s['code'] == 'QTK3'], [qtk3])
        self.assertEqual(step(g1, g0), MC.STEP_REVISED)
        f.write_bytes(saved)
        edit(f, EPOCH - timedelta(days=8), de_mm=2.0)               # one row further out: not a dependency
        _, g2, _, _ = self.direct(RT.SCORED)
        self.assertEqual(g2['measurement']['lambda_max'], g0['measurement']['lambda_max'])
        self.assertEqual(step(g2, g0), MC.STEP_REPEATED)

    # ---- unverifiable input: a typed UNKNOWN identity, never a refusal of the value ---------------------------------------
    def test_bytes_changed_after_the_producer_read_are_unknown_and_keep_the_value(self):
        ratio, prov, trace, _ = self.direct(RT.SCORED)
        store, entry, path = self.bound(RT.SCORED)
        m = prov['measurement']
        f = self.cache / 'QTK1.igs20.tenv'
        f.write_bytes(f.read_bytes() + b'\n')
        oc, rec = OI.observation_content(REGION, RT.SCORED, self.cache, path, trace, m, entry)
        self.assertEqual((oc['status'], oc['reason'], rec), ('UNKNOWN', 'INPUT_DIGEST_MISMATCH', None))
        f.unlink()
        self.assertEqual(OI.observation_content(REGION, RT.SCORED, self.cache, path, trace, m, entry)[0]['reason'],
                         'INPUT_UNREADABLE')
        altered = self.root / 'altered_catalog.json'
        altered.write_bytes(path.read_bytes() + b' ')
        got = OI.observation_content(REGION, RT.SCORED, self.cache, altered, trace, m, entry)[0]
        self.assertEqual((got['reason'], got['detail'].split(':')[0]), ('INPUT_DIGEST_MISMATCH', 'catalog'))
        m['observation_content'] = oc
        out = RD.run_region_assessment(REGION, datetime(2026, 11, 4), lambda_geo_ratio=ratio, use_seismic=False,
                                       lambda_geo_provenance=prov, lg_candidate_policy=RT.POLICY,
                                       lg_candidate_refusal=None)
        self.assertTrue(out.components['lambda_geo'].eligible_for_tiering, out.components['lambda_geo'].eligibility_reason)
        self.assertEqual(out.method_set['included'], ['lambda_geo'])
        ident = MC.observation_identity('lambda_geo', out)
        self.assertEqual(ident['status'], 'UNKNOWN')
        self.assertIn('INPUT_DIGEST_MISMATCH', ident['reason'])

    def test_unbindable_input_is_a_typed_unknown_never_an_exception(self):
        ratio, prov, trace, _ = self.direct(RT.SCORED)
        store, entry, path = self.bound(RT.SCORED)
        m = prov['measurement']

        def got(trace_=trace, m_=m, entry_=entry, path_=path):
            oc, rec = OI.observation_content(REGION, RT.SCORED, self.cache, path_, trace_, m_, entry_)
            self.assertIsNone(rec)
            self.assertEqual(oc['status'], 'UNKNOWN')
            return oc['reason']

        def but(obj, **changes):
            out = json.loads(json.dumps(obj))
            out.update(changes)
            return out
        self.assertEqual(got(trace_=None, m_=None, entry_=None), 'NO_SELECTED_OBSERVATION')
        self.assertEqual(got(trace_=but(trace, status='NO_FULL_SUPPORT_EPOCH')), 'NO_SELECTED_OBSERVATION')
        self.assertEqual(got(m_=but(m, observation_day='2026-10-01')), 'SELECTION_UNBOUND')
        self.assertEqual(got(trace_=but(trace, input_manifest=but(trace['input_manifest'], provider='GeoNet'))),
                         'MANIFEST_UNBOUND')
        self.assertEqual(got(trace_=but(trace, input_manifest=but(trace['input_manifest'], as_of='2026-11-05T23:59:59'))),
                         'MANIFEST_UNBOUND')
        self.assertEqual(got(entry_=but(entry, origins=None)), 'ORIGIN_UNBOUND')
        self.assertEqual(got(path_=None), 'CATALOG_UNBOUND')
        sel = but(trace['selection'], index=trace['selection']['index'] - 1)
        self.assertEqual(got(trace_=but(trace, selection=sel)), 'GRID_RECONSTRUCTION_MISMATCH')
        self.assertEqual(got(m_=but(m, lambda_max=m['lambda_max'] * (1 + 1e-15) + 1e-300)), 'CLOSURE_RECOMPUTE_MISMATCH')
        self.assertEqual(got(m_=but(m, operator_identity='x')), 'OPERATOR_IDENTITY_UNBOUND')

    # ---- nothing numerical moves ------------------------------------------------------------------------------------------
    def test_the_value_the_operator_and_the_baseline_binding_do_not_move(self):
        before = P.code_identity()
        res = self.scored(RT.SCORED, 'tick-v')
        ratio, prov, trace = P.candidate_lambda_geo(REGION, RT.SCORED, self.cache, self.base,
                                                    catalog_store=RUN.catalog_store(self.base))
        self.assertEqual(res['ratio'], ratio)
        added = {'station_freshness', 'observation_content'}
        self.assertEqual({k: v for k, v in res['provenance']['measurement'].items() if k not in added}, prov['measurement'])
        self.assertEqual({k: v for k, v in res['provenance'].items() if k != 'measurement'},
                         {k: v for k, v in prov.items() if k != 'measurement'})
        self.assertEqual(P.code_identity(), before)
        self.assertNotIn('lg_observation_identity.py', P.OPERATOR_SOURCES)
        self.assertNotIn('lg_candidate_runner.py', P.OPERATOR_SOURCES)
        self.assertEqual(res['trace']['identity_equal'], True)

    def test_a_closure_narrower_than_the_composition_is_never_known(self):
        ratio, prov, trace, _ = self.direct(RT.SCORED)
        _, p4, _, rec4 = self.direct(RT.SCORED, margin=4)
        self.assertEqual((p4['measurement']['observation_content']['reason'], rec4), ('CLOSURE_MARGIN_INVALID', None))
        # and the per-run recomputation is what would catch a too-narrow closure: four rows do not reproduce the value
        store, entry, path = self.bound(RT.SCORED)
        stations, _ = OI._verified_stations(self.cache, path, trace['input_manifest'], trace['station_codes'],
                                            trace['input_manifest']['days_back'], P.as_of_instant(RT.SCORED))
        times, support = OI.union_grid(stations)
        cl5 = OI.closure(stations, entry['origins'], times, support, trace['selection']['index'], 5)
        self.assertEqual(OI.recompute(cl5, 5)[0], prov['measurement']['lambda_max'])
        cl4 = [dict(st, rows=st['rows'][1:-1]) for st in cl5]
        self.assertNotEqual(OI.recompute(cl4, 4)[0], prov['measurement']['lambda_max'])


class OperatorClosurePerturbation(unittest.TestCase):
    """The production operator functions on a nondegenerate synthetic carrier (six stations, random walks with distinct
    trends, one interior gap of three rows crossing the window edge). Perturbation sizes 1e-3 mm and 50 mm, both signs,
    on E and N."""

    N = 100

    def stations(self):
        rng = np.random.default_rng(20261116)
        times = [datetime(2026, 7, 1, 12) + timedelta(days=d) for d in range(self.N)]
        coords = [(35.0, 139.0), (35.4, 139.6), (35.9, 139.2), (35.2, 140.1), (36.1, 140.0), (35.6, 139.9)]
        out = []
        for k, (lat, lon) in enumerate(coords):
            e = np.cumsum(rng.normal(0, 1, self.N)) + (0.3 + 0.1 * k) * np.arange(self.N)
            n = np.cumsum(rng.normal(0, 1, self.N)) - (0.2 - 0.07 * k) * np.arange(self.N)
            keep = [d for d in range(self.N) if not (k == 2 and d in (88, 89, 90))]
            out.append(LD.StationData(f'S{k}', lat, lon, [times[d] for d in keep], e[keep].copy(), n[keep].copy(),
                                      np.zeros(len(keep))))
        return out

    @staticmethod
    def operator(stations):
        sc = LD.StrainComputer(smoothing_window=7)
        vel, pos, _ = sc.compute_velocities(stations)
        strain, _ = sc.compute_strain_field(vel, pos)
        return LD.LambdaGeoComputer().compute(strain)

    def test_outside_the_closure_every_triangle_value_at_the_epoch_is_unchanged(self):
        st = self.stations()
        times, support = OI.union_grid(st)
        i = int(R.valid_denominator_epochs(support, P.SUPPORT_MARGIN)[-1])
        self.assertEqual(i, self.N - 1 - P.SUPPORT_MARGIN)
        base = self.operator(st)[i].copy()
        zero = {s.code: [0.0, 0.0] for s in st}
        cl = OI.closure(st, zero, times, support, i)
        self.assertEqual(OI.recompute(cl)[0], R._epoch_max(base)[0], 'the closure alone reproduces the value')
        inside = {(k, r) for k, c in enumerate(cl) for row in c['rows']
                  for r in ((row['row'],) if row['kind'] == 'R' else (row['left'], row['right']))}
        self.assertIn((2, 87), inside, 'the left bracket of the gap is outside the window but inside the closure')
        index = {t: r for r, t in enumerate(times)}
        changed_inside = checked_outside = 0
        moved = set()
        for k, s in enumerate(st):
            for j, t in enumerate(s.times):
                r = index[t]
                if r < i - 14 and r % 7:
                    continue                                  # every row near the closure; a sample of the rest
                for attr, deltas in (('east_mm', (1e-3, -1e-3, 50.0, -50.0)), ('north_mm', (1e-3, -50.0))):
                    for delta in deltas:
                        pert = [LD.StationData(x.code, x.lat, x.lon, list(x.times), x.east_mm.copy(), x.north_mm.copy(),
                                               x.up_mm.copy()) for x in st]
                        getattr(pert[k], attr)[j] += delta
                        same = np.array_equal(self.operator(pert)[i], base)
                        if (k, r) in inside:
                            changed_inside += not same
                            if not same:
                                moved.add((k, r))
                        else:
                            checked_outside += 1
                            self.assertTrue(same, f'{s.code} row {r} ({attr} {delta:+g}) is outside the closure but '
                                                  f'moved a triangle at the epoch')
        self.assertGreater(checked_outside, 100)
        self.assertGreater(changed_inside, 100, 'the closure is not vacuous: its rows do move the epoch')
        self.assertIn((2, 87), moved, 'the out-of-window bracket moves a triangle value at the epoch')
        self.assertEqual({(k, r) for k, r in inside if r in (i - 5, i + 5)} - moved, set(),
                         'both window edges are real dependencies of every station')


if __name__ == '__main__':
    unittest.main(verbosity=2)
