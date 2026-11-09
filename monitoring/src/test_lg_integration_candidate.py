"""Offline integration tests; synthetic GNSS provenance, real ensemble arithmetic."""
import copy
from datetime import datetime
import json
from pathlib import Path
import unittest

import calibration_eligibility_fixtures as FX
FX.install_stubs()
import calibration_eligibility as CE
import ensemble
import lg_integration_candidate as LGI
import raw_history_cayley as RH

DAY = datetime(2026, 11, 10)
POLICY = LGI.CandidatePolicy('2026-11-10')


def provenance(ratio=4.0):
    return dict(source='fixture-only-regional-baseline', n_days=70,
                window_start='2026-07-05', window_end='2026-10-03', calibrated_on='2026-11-02',
                calibration_sha256='b' * 64, operator_identity='a' * 64,
                statistic=LGI.STATISTIC, baseline_value=2.0,
                measurement=dict(operator_identity='a' * 64, input_sha256='c' * 64,
                                 observation_day='2026-11-01', support_through_day='2026-11-05',
                                 selection_status='OK', lambda_max=ratio * 2,
                                 unclamped_ratio=ratio))


class QualificationTests(unittest.TestCase):
    def test_positive_input_is_admitted(self):
        self.assertTrue(LGI.classify(provenance(), DAY, 4, POLICY).eligible_for_tiering)

    def test_legacy_production_policy_is_not_changed(self):
        self.assertIsNone(ensemble.LAMBDA_GEO_BASELINE_MAX_AGE_DAYS)
        self.assertIsNone(ensemble.LAMBDA_GEO_BASELINE_MIN_LAG_DAYS)
        obj = ensemble.GeoSpecEnsemble('tokyo_kanto', eligibility_rule_active=True)
        obj.set_lambda_geo(DAY, 4, provenance())
        result = obj.compute_lambda_geo_risk(DAY)
        self.assertFalse(result.eligible_for_tiering)
        self.assertIn('NO_REGISTERED_FRESHNESS_POLICY', result.eligibility_reason)

    def test_provenance_survives_exclusion_and_history_export(self):
        obj = ensemble.GeoSpecEnsemble('tokyo_kanto', eligibility_rule_active=True)
        prov = provenance()
        obj.set_lambda_geo(DAY, 4, prov)
        component = obj.compute_lambda_geo_risk(DAY).to_dict()
        self.assertEqual(component['baseline_provenance'], prov)
        self.assertIn('support', component)
        entry = RH.method_entry('lambda_geo', component, {})
        self.assertEqual(entry['calibration']['baseline_provenance'], prov)
        self.assertEqual(entry['support'], component['support'])

    def test_refusals(self):
        cases = [
            ('n_days', 29, 'LG_INSUFFICIENT_BASELINE_SAMPLES'),
            ('calibration_sha256', None, 'LG_BASELINE_BYTES_UNBOUND'),
            ('statistic', 'legacy-zeros-dropped', 'LG_BASELINE_STATISTIC_UNKNOWN'),
            ('baseline_value', 0, 'LG_DENOMINATOR_UNAVAILABLE'),
            ('baseline_value', float('nan'), 'LG_DENOMINATOR_UNAVAILABLE'),
            ('operator_identity', 'd' * 64, 'LG_OPERATOR_MISMATCH'),
            ('operator_identity', None, 'LG_OPERATOR_IDENTITY_UNBOUND'),
            ('measurement', None, 'LG_MEASUREMENT_PROVENANCE_UNKNOWN'),
            ('calibrated_on', '2026-11-11', 'CALIBRATED_AFTER_SCORED_DAY'),
            ('calibrated_on', '2026-10-31', 'LAG_NOT_HONORED'),
            ('window_end', '2026-08-01', 'WINDOW_STALE'),
        ]
        for key, value, code in cases:
            with self.subTest(key=key, value=value):
                p = provenance(); p[key] = value
                self.assertEqual(LGI.classify(p, DAY, 4, POLICY).code, code)

    def test_measurement_refusals(self):
        cases = [('input_sha256', None, 'LG_INPUT_BYTES_UNBOUND'),
                 ('observation_day', None, 'LG_OBSERVATION_TIME_UNKNOWN'),
                 ('support_through_day', '2026-11-11', 'LG_FUTURE_SUPPORT'),
                 ('observation_day', '2026-11-06', 'LG_FUTURE_SUPPORT'),
                 ('observation_day', '2026-09-01', 'LG_OBSERVATION_STALE'),
                 ('selection_status', 'NO_FULL_SUPPORT_EPOCH', 'LG_SUPPORT_SELECTION_UNAVAILABLE'),
                 ('lambda_max', -1, 'LG_NUMERATOR_UNAVAILABLE'),
                 ('lambda_max', float('inf'), 'LG_NUMERATOR_UNAVAILABLE'),
                 ('unclamped_ratio', 8, 'LG_RATIO_LINEAGE_MISMATCH')]
        for key, value, code in cases:
            with self.subTest(key=key):
                p = provenance(); p['measurement'][key] = value
                self.assertEqual(LGI.classify(p, DAY, 4, POLICY).code, code)

    def test_no_retroactive_candidate(self):
        self.assertEqual(LGI.classify(provenance(), datetime(2026, 11, 9), 4, POLICY).code,
                         'LG_BEFORE_CANDIDATE_BOUNDARY')

    def test_zero_is_valid_but_clamping_is_reproducible(self):
        for q in (0, 0.05, 4, 100):
            self.assertTrue(LGI.classify(provenance(q), DAY, max(0.1, min(50, q)), POLICY).eligible_for_tiering)
        self.assertEqual(LGI.classify(provenance(0), DAY, 0, POLICY).code, 'LG_RATIO_LINEAGE_MISMATCH')

    def test_input_not_mutated(self):
        p = provenance(); before = copy.deepcopy(p)
        LGI.classify(p, DAY, 4, POLICY)
        self.assertEqual(p, before)

    def test_candidate_cannot_disable_checks(self):
        with self.assertRaises(ValueError):
            ensemble.GeoSpecEnsemble('tokyo_kanto', eligibility_rule_active=False, lg_candidate_policy=POLICY)
        for fields in ({'effective_scored_day': ''}, {'baseline_max_age_days': 20}, {'min_baseline_samples': True}):
            with self.assertRaises(ValueError):
                LGI.CandidatePolicy(**dict({'effective_scored_day': '2026-11-10'}, **fields))


class EnsembleIntegrationTests(unittest.TestCase):
    def compute(self, ratio, p=None):
        obj = ensemble.GeoSpecEnsemble('tokyo_kanto', eligibility_rule_active=True, lg_candidate_policy=POLICY)
        obj.set_lambda_geo(DAY, ratio, p or provenance(ratio))
        thd = ensemble.MethodResult(name='seismic_thd', available=True, raw_value=0.5, risk_score=0.2)
        CE.attach(thd, CE.Eligibility('calibrated', True, 'CALIBRATED: fixture', 'CALIBRATED'))
        thd.support = dict(identity='fixture-station', estimator='fixture-estimator',
                           calibration_class='fixture-baseline', shared_with=['tokyo_kanto'])
        obj.compute_thd_risk = lambda *a, **k: thd
        return obj.compute_risk(DAY, methods=('lambda_geo', 'seismic_thd'))

    def test_lg_reaches_real_combiner_with_nonzero_weight_and_response(self):
        low, high = self.compute(1), self.compute(4)
        for result in (low, high):
            self.assertEqual(result.methods_available, 2)
            self.assertAlmostEqual(result.effective_weights['lambda_geo'], 4 / 7)
            self.assertEqual(result.method_set['included'], ['lambda_geo', 'seismic_thd'])
        self.assertGreater(high.combined_risk, low.combined_risk)
        self.assertAlmostEqual(high.combined_risk,
                               (0.4 * ensemble.lambda_geo_to_risk(4) + 0.3 * 0.2) / 0.7)

    def test_excluded_lg_does_not_silently_enter_weight_denominator(self):
        p = provenance(); p['measurement']['operator_identity'] = 'd' * 64
        result = self.compute(4, p)
        self.assertEqual(result.effective_weights, {'seismic_thd': 1})
        self.assertAlmostEqual(result.combined_risk, 0.2)

    def test_current_japan_steps_are_lost_by_level_mapping_not_by_combiner(self):
        for value in (0.1, 0.9321286354018504, 0.10299787949606348, 0.18376583173976863):
            result = self.compute(value)
            self.assertEqual(result.components['lambda_geo'].risk_score, 0)
            self.assertEqual(result.method_set['states']['lambda_geo'], 'VALID_ZERO')
            self.assertAlmostEqual(result.combined_risk, 0.2 * 3 / 7)


if __name__ == '__main__':
    unittest.main()
