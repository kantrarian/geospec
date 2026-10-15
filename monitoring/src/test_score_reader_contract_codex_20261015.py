"""Real reader calls, not a copied loop/source-text check; isolated synthetic reports.

Run from the candidate monitoring/src with PYTHONPATH containing that directory.
No network or live reports. Marked missing z must not be published as 0.0 or a
tier-derived delta-z. Unmarked legacy fallback and independent numeric controls
are deliberately retained; no scientific interpretation is asserted.
"""
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import stress_release_detector as subject

MARKER = "NOT_COMPUTED_NO_EMPIRICAL_BASELINE"


class ReaderContract(unittest.TestCase):
    def run_case(self, marked_days=(), independent=False, no_numeric_legacy=False):
        first = datetime(2026, 10, 10)
        with tempfile.TemporaryDirectory(prefix="score-reader-") as temp:
            for offset in range(4):
                day = (first + timedelta(days=offset)).strftime("%Y-%m-%d")
                z = 3.0 if offset < 3 else 0.0
                component = dict(available=True, raw_value=0.3 + 0.07 * z,
                                 z_score=z, baseline=dict(mean=0.3, std=0.07))
                if offset in marked_days:
                    component.update(z_score=None, risk_score=None, is_elevated=None,
                                     is_critical=None, score_basis=MARKER)
                if no_numeric_legacy:
                    component.update(z_score=None, baseline=None)
                components = dict(seismic_thd=component)
                if independent:
                    components["lambda_geo"] = dict(available=True, z_score=z)
                record = dict(date=day, regions={"fixture_region": dict(
                    tier=2 if offset < 3 else 0, combined_risk=0.6 if offset < 3 else 0.0,
                    components=components)})
                (Path(temp) / ("ensemble_" + day + ".json")).write_text(json.dumps(record), encoding="utf-8")
            return subject.detect_stress_release_drops(
                ensemble_dir=Path(temp), target_date=first + timedelta(days=3),
                lookback_days=4, require_multi_region=False)

    def test_numeric_positive_control_fires(self):
        drops = self.run_case()
        self.assertEqual(len(drops), 1)
        self.assertEqual((drops[0].prior_z_score, drops[0].drop_z_score, drops[0].delta_z), (3.0, 0.0, 3.0))

    def test_explicitly_withheld_prior_z_cannot_become_zero(self):
        self.assertEqual(self.run_case(marked_days=(0, 1, 2)), [])

    def test_explicitly_withheld_current_z_cannot_become_zero(self):
        self.assertEqual(self.run_case(marked_days=(3,)), [])

    def test_other_numeric_component_remains_usable(self):
        drops = self.run_case(marked_days=(0, 1, 2, 3), independent=True)
        self.assertEqual(len(drops), 1)
        self.assertEqual(drops[0].delta_z, 3.0)

    def test_unmarked_legacy_tier_fallback_is_unchanged(self):
        drops = self.run_case(no_numeric_legacy=True)
        self.assertEqual(len(drops), 1)
        self.assertEqual(drops[0].delta_z, 2.0)


if __name__ == "__main__":
    unittest.main()
