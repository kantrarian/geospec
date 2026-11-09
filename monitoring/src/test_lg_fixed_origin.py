"""fixed-origin-v1 tests (cayley 2026-11-09; codex 397b7e96 numerical closure). NGL positions carry absolute
northings near 3.87e9 mm; the candidate subtracts ONE fixed E/N origin per station (first calibration-window row,
metre-rounded, recorded in the baseline, reused at scoring) before interpolation and smoothing. Production
live_data is untouched. Synthetic data only; offsets are exact integers and walks sit on a 2**-10 mm grid so that
offset + walk is exactly representable (the operator, not the input quantisation, is under test)."""
import json
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest import mock

import numpy as np

import calibration_eligibility_fixtures as FX
FX.install_stubs()
import live_data as LD                          # noqa: E402
import lg_numerator_support_rule as R           # noqa: E402
import lg_producer_candidate as P               # noqa: E402
import test_lg_producer_candidate as TT         # noqa: E402

T0 = datetime(2026, 7, 5, 12)
N = 80
COORDS = [(35.0, 139.0), (35.4, 139.6), (35.9, 139.2), (35.2, 140.1), (36.1, 140.0)]
OFFSETS = [(66196 + 1000 * k, 3870980688 + 7500 * k) for k in range(len(COORDS))]     # mm, retained-file magnitude


def walk_stations(offset, seed=20261109):
    rng = np.random.default_rng(seed)
    out = []
    for k, (lat, lon) in enumerate(COORDS):
        e = np.round(np.cumsum(rng.normal(0, 0.8, N)) * 1024) / 1024
        n = np.round(np.cumsum(rng.normal(0, 0.8, N)) * 1024) / 1024
        if offset:
            e, n = e + OFFSETS[k][0], n + OFFSETS[k][1]
        out.append(LD.StationData(f'S{k}', lat, lon, [T0 + timedelta(days=i) for i in range(N)], e, n, np.zeros(N)))
    return out


def lam(stations):
    sc = LD.StrainComputer(smoothing_window=7)
    v, p, _ = sc.compute_velocities(stations)
    E, _ = sc.compute_strain_field(v, p)
    return v, LD.LambdaGeoComputer().compute(E)


class Operator(unittest.TestCase):
    def test_a_quadratic_at_a_northing_offset_gives_its_exact_derivative_only_with_the_fixed_origin(self):
        b, c = 0.25, 0.0625
        t = np.arange(N, dtype=float)
        raw = [LD.StationData(f'Q{k}', lat, lon, [T0 + timedelta(days=i) for i in range(N)],
                              OFFSETS[k][0] + b * t + c * t * t, OFFSETS[k][1] - b * t + c * t * t, np.zeros(N))
               for k, (lat, lon) in enumerate(COORDS)]
        conditioned = [P.condition_origin(s, P.derive_origin(s)) for s in raw]
        expect_e, expect_n = b + 2 * c * t[1:-1], -b + 2 * c * t[1:-1]
        v, _, _ = LD.StrainComputer(smoothing_window=7).compute_velocities(conditioned)
        np.testing.assert_allclose(v[1:-1, :, 0], np.repeat(expect_e[:, None], len(COORDS), 1), rtol=0, atol=1e-10)
        np.testing.assert_allclose(v[1:-1, :, 1], np.repeat(expect_n[:, None], len(COORDS), 1), rtol=0, atol=1e-10)
        v_raw, _, _ = LD.StrainComputer(smoothing_window=7).compute_velocities(raw)       # the unconditioned operator
        self.assertGreater(float(np.max(np.abs(v_raw[1:-1, :, 1] - expect_n[:, None]))), 1e-8)

    def test_origin_shift_invariance_within_the_frozen_tolerance(self):
        small = [P.condition_origin(s, P.derive_origin(s)) for s in walk_stations(offset=False)]
        large = [P.condition_origin(s, P.derive_origin(s)) for s in walk_stations(offset=True)]
        v0, L0 = lam(small)
        v1, L1 = lam(large)
        np.testing.assert_allclose(v1, v0, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(L1, L0, rtol=1e-9, atol=1e-10)
        _, L_raw = lam(walk_stations(offset=True))                                    # unconditioned: outside
        self.assertFalse(np.allclose(L_raw, L0, rtol=1e-9, atol=1e-10))

    def test_the_origin_is_metre_rounded_exact_and_leaves_the_source_untouched(self):
        s = walk_stations(offset=True)[0]
        before = s.north_mm.copy()
        o = P.derive_origin(s)
        self.assertEqual([x % 1000.0 for x in o], [0.0, 0.0])
        c = P.condition_origin(s, o)
        np.testing.assert_array_equal(s.north_mm, before)
        np.testing.assert_array_equal(c.east_mm + o[0], s.east_mm)                    # subtraction was exact, E and N
        np.testing.assert_array_equal(c.north_mm + o[1], s.north_mm)
        np.testing.assert_array_equal(c.up_mm, s.up_mm)
        self.assertLess(float(np.max(np.abs(c.east_mm))), 2000.0)
        self.assertLess(float(np.max(np.abs(c.north_mm))), 2000.0)
        near = LD.StationData('Z', 35.0, 139.0, [T0], np.array([-3.0]), np.array([-499.0]), np.zeros(1))
        self.assertEqual(json.dumps(P.derive_origin(near)), '[0.0, 0.0]')            # no negative zero recorded


class Candidate(TT.Fixture):
    def offset_cache(self):
        TT.write_cache(self.cache)
        for k, f in enumerate(sorted(self.cache.glob('*.igs20.tenv'))):
            rows = []
            for line in f.read_text(encoding='utf-8').splitlines():
                p = line.split()
                p[7] = '%.6f' % (float(p[7]) + OFFSETS[k][0] / 1000.0)
                p[8] = '%.6f' % (float(p[8]) + OFFSETS[k][1] / 1000.0)
                rows.append(' '.join(p))
            f.write_text('\n'.join(rows) + '\n', encoding='utf-8')

    def test_calibration_records_one_metre_rounded_origin_per_carrier_station_and_scoring_reuses_them(self):
        self.offset_cache()
        path = self.calibrate()
        entry = json.loads(path.read_text(encoding='utf-8'))['regions']['tokyo_kanto']
        self.assertEqual(entry['origin_rule'], P.ORIGIN_RULE)
        self.assertEqual(sorted(entry['origins']), sorted(entry['station_codes']))
        self.assertTrue(all(v % 1000.0 == 0.0 for o in entry['origins'].values() for v in o))
        m, _ = P.measure('tokyo_kanto', TT.SCORED, self.cache, declared=entry['station_codes'], origins=entry['origins'])
        self.assertEqual(m.origins, entry['origins'])                                   # the SAME origins, both windows
        ratio, prov, trace = P.candidate_lambda_geo('tokyo_kanto', TT.SCORED, self.cache, self.base)
        self.assertEqual((trace['status'], trace['origin_rule']), ('OK', P.ORIGIN_RULE))
        i = trace['selection']['index']
        conditioned = float(np.max(m.grid[i][np.isfinite(m.grid[i])]))
        raw, _ = P.measure('tokyo_kanto', TT.SCORED, self.cache, declared=entry['station_codes'])
        unconditioned = float(np.max(raw.grid[i][np.isfinite(raw.grid[i])]))
        self.assertEqual(prov['measurement']['lambda_max'], conditioned)              # the candidate used the origins
        self.assertNotEqual(prov['measurement']['lambda_max'], unconditioned)          # and the origins matter here

    def test_the_retained_reader_conditions_and_refuses_an_unbound_station(self):
        self.offset_cache()
        as_of = P.as_of_instant(TT.SCORED)
        acq = P.RetainedNGL(self.cache, origins='derive')
        acq.load_station_catalog()
        st = acq.download_station('QTK1', 90, as_of)
        self.assertLess(float(np.max(np.abs(st.east_mm))), 2000.0)
        self.assertLess(float(np.max(np.abs(st.north_mm))), 2000.0)
        self.assertEqual(sorted(acq.origins), ['QTK1'])
        bound = P.RetainedNGL(self.cache, origins={'QTK2': [0.0, 0.0]})
        bound.load_station_catalog()
        self.assertIsNone(bound.download_station('QTK1', 90, as_of))
        self.assertEqual(bound.attempts['QTK1']['outcome'], 'ORIGIN_UNBOUND')
        plain = P.RetainedNGL(self.cache)
        plain.load_station_catalog()
        self.assertGreater(float(np.max(np.abs(plain.download_station('QTK1', 90, as_of).north_mm))), 3.8e9)

    def test_support_and_selection_are_unchanged_by_the_origin(self):
        self.offset_cache()
        end = P.calibration_window_end(TT.CALIBRATED_ON)
        a, _ = P.measure('tokyo_kanto', end, self.cache, P.BASELINE_DAYS)                      # unconditioned
        b, _ = P.measure('tokyo_kanto', end, self.cache, P.BASELINE_DAYS, origins='derive')    # fixed origin
        np.testing.assert_array_equal(a.support, b.support)
        self.assertEqual(a.times, b.times)
        self.assertEqual(list(R.valid_denominator_epochs(a.support, P.SUPPORT_MARGIN)),
                         list(R.valid_denominator_epochs(b.support, P.SUPPORT_MARGIN)))
        self.assertEqual(a.station_codes, b.station_codes)
        self.assertEqual(a.input_sha256, b.input_sha256)                                       # same source bytes

    def test_the_matched_baseline_guard(self):
        self.offset_cache()
        path = self.calibrate()
        doc = json.loads(path.read_text(encoding='utf-8'))
        e = doc['regions']['tokyo_kanto']
        # (1) a baseline with no recorded origins is refused by name
        without = json.loads(json.dumps(doc))
        del without['regions']['tokyo_kanto']['origins']
        other = self.root / 'b-no-origins'
        other.mkdir()
        (other / path.name).write_text(json.dumps(without), encoding='utf-8')
        t = P.candidate_lambda_geo('tokyo_kanto', TT.SCORED, self.cache, other)[2]
        self.assertEqual((t['status'], t['detail']),
                         ('BASELINE_ORIGIN_UNBOUND', f'baseline records no {P.ORIGIN_RULE} origins'))
        # (2) a baseline missing one declared station's origin names it
        partial = json.loads(json.dumps(doc))
        partial['regions']['tokyo_kanto']['origins'].pop('QTK3')
        other2 = self.root / 'b-partial'
        other2.mkdir()
        (other2 / path.name).write_text(json.dumps(partial), encoding='utf-8')
        t = P.candidate_lambda_geo('tokyo_kanto', TT.SCORED, self.cache, other2)[2]
        self.assertEqual((t['status'], t['detail']), ('BASELINE_ORIGIN_UNBOUND', 'QTK3'))
        # (2b) origins present but recorded under another (or no) origin rule are not reused
        for k, rule in enumerate(('pre-fixed-origin', None)):
            foreign = json.loads(json.dumps(doc))
            if rule is None:
                del foreign['regions']['tokyo_kanto']['origin_rule']
            else:
                foreign['regions']['tokyo_kanto']['origin_rule'] = rule
            d = self.root / f'b-foreign-rule-{k}'
            d.mkdir()
            (d / path.name).write_text(json.dumps(foreign), encoding='utf-8')
            t = P.candidate_lambda_geo('tokyo_kanto', TT.SCORED, self.cache, d)[2]
            self.assertEqual((t['status'], t['detail']),
                             ('BASELINE_ORIGIN_UNBOUND', f'baseline records no {P.ORIGIN_RULE} origins'))
        # (3) a baseline from an operator version without this origin rule never pairs with a centred numerator
        with mock.patch.object(P, 'ORIGIN_RULE', 'pre-fixed-origin'):
            old = P.calibrate_region('tokyo_kanto', TT.CALIBRATED_ON, self.cache, code_id=P.code_identity())
        old['origin_rule'] = P.ORIGIN_RULE                                                   # even if relabelled
        other3 = self.root / 'b-old-operator'
        P.write_baselines(other3, TT.CALIBRATED_ON, {'tokyo_kanto': old}, P.code_identity(), TT.CLOCK)
        self.assertNotEqual(old['operator_identity'], e['operator_identity'])
        self.assertEqual(P.candidate_lambda_geo('tokyo_kanto', TT.SCORED, self.cache, other3)[2]['status'],
                         'OPERATOR_MISMATCH')


if __name__ == '__main__':
    unittest.main()
