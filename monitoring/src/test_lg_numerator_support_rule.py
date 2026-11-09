"""Boundary tests for lg_numerator_support_rule v3 (grassmann 2026-10-04): the v2 locks (mine + codex's 13 LGGuards),
codex's 7 CalibrationControls from review_round2.py (eb83ad47) reproduced in substance, metadata-validation and
days_back-equality locks, and the two real-input istanbul locks (skipped off devildog)."""
import json
import os
import sys
import unittest
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lg_numerator_support_rule as R  # noqa: E402

ID = R.production_operator_identity(provider="NGL", station_codes=["P1", "P2", "P3", "P4"], code_version="44fa5abf", days_back=90)


def synthetic(n_t=60, n_s=4, truncate=None, seed=0):
    rng = np.random.default_rng(seed)
    t0 = datetime(2026, 7, 1)
    times = [t0 + timedelta(days=i) for i in range(n_t)]
    support = np.ones((n_t, n_s), dtype=bool)
    for s, miss in (truncate or {}).items():
        support[n_t - miss:, s] = False
    full = support.sum(axis=1) == n_s
    grid = np.full((n_t, 3), 0.05)
    for i in range(n_t):
        lo, hi = max(0, i - 4), min(n_t, i + 5)
        if full[lo:hi].all() and 4 <= i < n_t - 4:
            grid[i] = 1.0 + 0.01 * rng.standard_normal(3)
    return times, support, grid


def sel(times, support, grid, target, **kw):
    args = dict(denominator=1.0, live_identity=ID, denominator_identity=ID)
    args.update(kw)
    return R.select_numerator(times, support, grid, target, **args)


class Synthetic(unittest.TestCase):
    def test_full_support_series_selects_an_interior_epoch_at_unit_ratio(self):
        times, support, grid = synthetic()
        s = sel(times, support, grid, times[-1].date())
        self.assertEqual(s.status, "OK"); self.assertEqual(s.index, len(times) - 1 - R.EDGE_MARGIN)
        self.assertAlmostEqual(s.unclamped_ratio, 1.0, places=1); self.assertFalse(s.clamped)

    def test_one_station_truncated_10_epochs_production_style_drops_rule_holds(self):
        times, support, grid = synthetic(truncate={1: 10})
        self.assertLess(float(np.nanmax(grid[-1])), 0.1)
        s = sel(times, support, grid, times[-1].date())
        self.assertEqual(s.status, "OK"); self.assertEqual(s.stations_with_real_samples, s.of_stations)
        self.assertLessEqual(s.index, len(times) - 10 - 1 - R.EDGE_MARGIN); self.assertAlmostEqual(s.unclamped_ratio, 1.0, places=1)

    def test_no_full_support_epoch_refuses_by_name(self):
        times, support, grid = synthetic(n_t=20); support[:, 2] = False
        self.assertEqual(sel(times, support, grid, times[-1].date()).status, "NO_FULL_SUPPORT_EPOCH")

    def test_full_support_epoch_too_old_refuses_instead_of_using_the_sparse_tail(self):
        times, support, grid = synthetic(n_t=70, truncate={0: 50})  # full-support prefix 20 epochs: exists under the margin, 55 d old
        s = sel(times, support, grid, times[-1].date())
        self.assertEqual(s.status, "FULL_SUPPORT_EPOCH_TOO_OLD"); self.assertGreater(s.days_before_target, R.MAX_LAG_DAYS)

    def test_denominator_epochs_exclude_edges_and_tails(self):
        times, support, grid = synthetic(n_t=30, truncate={3: 5})
        idx = R.valid_denominator_epochs(support)
        self.assertEqual(int(idx[0]), R.EDGE_MARGIN); self.assertLessEqual(int(idx[-1]), 30 - 5 - 1 - R.EDGE_MARGIN)

    def test_clamp_flag_is_reported_not_hidden(self):
        times, support, grid = synthetic()
        s = sel(times, support, grid, times[-1].date(), denominator=100.0)
        self.assertEqual(s.status, "OK"); self.assertTrue(s.clamped); self.assertEqual(s.clamped_ratio, 0.1); self.assertLess(s.unclamped_ratio, 0.1)


class CodexNumeratorGuards(unittest.TestCase):
    def fixture(self):
        ts = [datetime(2026, 7, 1) + timedelta(days=i) for i in range(20)]
        return ts, np.ones((20, 4), dtype=bool), np.ones((20, 2))

    def call(self, identities=True, **overrides):
        ts, support, grid = self.fixture()
        args = dict(times=ts, support=support, grid=grid, target=ts[-1].date(), denominator=1.0)
        if identities:
            args.update(live_identity="a" * 64, denominator_identity="a" * 64)
        args.update(overrides)
        return R.select_numerator(**args)

    def test_nominal_bound_unit_ratio(self):
        got = self.call(); self.assertEqual(got.status, "OK"); self.assertEqual(got.unclamped_ratio, 1.0); self.assertEqual(got.index, 19 - R.EDGE_MARGIN)

    def test_both_identities_missing_refuse(self):
        self.assertEqual(self.call(False).status, "OPERATOR_IDENTITY_UNBOUND")

    def test_one_identity_missing_refuses(self):
        self.assertEqual(self.call(denominator_identity=None).status, "OPERATOR_IDENTITY_UNBOUND")

    def test_empty_identity_refuses(self):
        self.assertEqual(self.call(live_identity="", denominator_identity="").status, "OPERATOR_IDENTITY_UNBOUND")

    def test_different_bound_identities_refuse(self):
        self.assertEqual(self.call(denominator_identity="b" * 64).status, "OPERATOR_MISMATCH")

    def test_nan_numerator_never_becomes_upper_clamp(self):
        got = self.call(grid=np.full((20, 2), np.nan)); self.assertEqual(got.status, "NUMERATOR_UNAVAILABLE"); self.assertIsNone(got.clamped_ratio)

    def test_infinite_numerator_refuses(self):
        self.assertEqual(self.call(grid=np.full((20, 2), np.inf)).status, "NUMERATOR_UNAVAILABLE")

    def test_negative_numerator_refuses(self):
        self.assertEqual(self.call(grid=np.full((20, 2), -1.0)).status, "NUMERATOR_UNAVAILABLE")

    def test_finite_zero_is_not_missing(self):
        got = self.call(grid=np.zeros((20, 2))); self.assertEqual(got.status, "OK"); self.assertEqual(got.unclamped_ratio, 0.0); self.assertTrue(got.clamped)

    def test_partial_finite_triangles_remain_usable(self):
        grid = np.ones((20, 2)); grid[:, 0] = np.nan
        self.assertEqual(self.call(grid=grid).status, "OK")

    def test_future_input_refuses(self):
        self.assertEqual(self.call(target=date(2026, 7, 19)).status, "FUTURE_EPOCH")

    def test_misaligned_arrays_refuse_typed(self):
        self.assertEqual(self.call(grid=np.ones((19, 2))).status, "INPUT_SHAPE_INVALID")

    def test_zero_margin_cannot_disable_edge_guard(self):
        self.assertEqual(self.call(edge_margin=0).status, "SUPPORT_POLICY_INVALID")

    def test_non_daily_grid_refuses(self):
        ts, support, grid = self.fixture(); ts[10] = ts[10] + timedelta(days=1)
        self.assertEqual(self.call(times=ts).status, "INPUT_TIME_INVALID")


class CodexCalibrationControls(unittest.TestCase):
    """review_round2.CalibrationControls (codex eb83ad47) against the integrated module."""
    def fixture(self):
        return ([datetime(2026, 1, 1) + timedelta(days=i) for i in range(68)], np.ones((68, 3), dtype=bool), np.ones((68, 2)))

    N_VALID = 68 - 2 * R.EDGE_MARGIN  # the interior epochs the rule admits on a fully supported 68-day grid

    def test_constant_positive_control(self):
        t, s, g = self.fixture(); out = R.denominator_from_grid(t, s, g)
        self.assertTrue(out["available"]); self.assertEqual((out["mean_lambda_geo"], out["n_samples"]), (1.0, self.N_VALID))

    def test_zero_days_keep_their_weight(self):
        t, s, g = self.fixture(); nz = self.N_VALID // 2; m = R.EDGE_MARGIN; g[m:m + nz] = 0.0; out = R.denominator_from_grid(t, s, g)
        self.assertTrue(out["available"]); self.assertEqual((out["mean_lambda_geo"], out["n_samples"]), ((self.N_VALID - nz) / self.N_VALID, self.N_VALID)); self.assertEqual(out["n_zero_days"], nz)

    def test_all_zero_denominator_is_unavailable(self):
        t, s, g = self.fixture(); g[:] = 0.0; out = R.denominator_from_grid(t, s, g)
        self.assertFalse(out["available"]); self.assertEqual(out["reason"], "DENOMINATOR_UNAVAILABLE")

    def test_invalid_triangle_epoch_not_laundered_by_positive_max(self):
        t, s, g = self.fixture(); m = R.EDGE_MARGIN; g[m, 0] = -1.0; out = R.denominator_from_grid(t, s, g)  # the first ADMITTED epoch carries the invalid triangle
        self.assertEqual(out["n_samples"], self.N_VALID - 1); self.assertEqual(out["rejected_epochs"][0]["epoch"], t[m].date().isoformat())

    def test_finite_support_dates_exclude_invalid_edge_epoch(self):
        t, s, g = self.fixture(); m = R.EDGE_MARGIN; g[m] = np.nan; out = R.denominator_from_grid(t, s, g)  # an invalid FIRST admitted epoch must not become first_epoch
        self.assertEqual(out["first_epoch"], t[m + 1].date().isoformat())

    def test_non_daily_calibration_grid_refuses(self):
        t, s, g = self.fixture(); t[12] += timedelta(days=1); out = R.denominator_from_grid(t, s, g)
        self.assertFalse(out["available"]); self.assertEqual(out["reason"], "INPUT_TIME_INVALID")

    def test_null_metadata_not_stringified_to_an_identity(self):
        kwargs = dict(provider=None, frame="ENU", units="u", station_codes=["A", "B", "C"], triangulation="Delaunay", smoothing={"window": 7},
                      derivative="d", edge_policy="interior", gap_policy="linear", estimator="e", reduction="max", baseline_statistic="mean",
                      processing_version="v2", code_version="code", days_back=90)
        with self.assertRaises(ValueError):
            R.operator_identity(**kwargs)


class Identity(unittest.TestCase):
    def test_identity_from_metadata_is_order_independent_and_field_sensitive(self):
        a = R.production_operator_identity(provider="NGL", station_codes=["B", "A", "C"], code_version="x", days_back=90)
        b = R.production_operator_identity(provider="NGL", station_codes=["A", "B", "C"], code_version="x", days_back=90)
        self.assertEqual(a, b)
        self.assertNotEqual(a, R.production_operator_identity(provider="GeoNet", station_codes=["A", "B", "C"], code_version="x", days_back=90))
        self.assertNotEqual(a, R.production_operator_identity(provider="NGL", station_codes=["A", "B", "D"], code_version="x", days_back=90))
        self.assertNotEqual(a, R.production_operator_identity(provider="NGL", station_codes=["A", "B", "C"], code_version="y", days_back=90))

    def test_days_back_is_an_identity_field_and_a_mismatch_refuses(self):
        a = R.production_operator_identity(provider="NGL", station_codes=["A", "B", "C"], code_version="x", days_back=90)
        b = R.production_operator_identity(provider="NGL", station_codes=["A", "B", "C"], code_version="x", days_back=120)
        self.assertNotEqual(a, b)
        times, support, grid = synthetic(n_s=3)
        self.assertEqual(sel(times, support, grid, times[-1].date(), live_identity=b, denominator_identity=a).status, "OPERATOR_MISMATCH")

    def test_invalid_metadata_refuses_before_hashing(self):
        base = dict(provider="NGL", station_codes=["A", "B", "C"], code_version="x", days_back=90)
        for bad in (dict(provider=""), dict(station_codes=["A", "B"]), dict(station_codes=["A", "A", "B"]), dict(days_back=0), dict(code_version=None)):
            with self.assertRaises(ValueError):
                R.production_operator_identity(**{**base, **bad})
        with self.assertRaises(ValueError):
            R.operator_identity(provider="NGL", frame="f", units="u", station_codes=["A", "B", "C"], triangulation="t", smoothing={"window": None},
                                derivative="d", edge_policy="e", gap_policy="g", estimator="e", reduction="r", baseline_statistic="b",
                                processing_version="p", code_version="c", days_back=90)

    def test_denominator_requires_three_stations(self):
        times, support, grid = synthetic(n_s=2)
        self.assertEqual(R.denominator_from_grid(times, support, grid)["reason"], "INSUFFICIENT_STATIONS")


import lg_real_istanbul_fixture as FX

_CACHE = FX.cache_dir()


@unittest.skipUnless(FX.available(_CACHE), "retained NGL cache absent (set LG_TEST_CACHE to a preserved copy)")
class RealIstanbul(unittest.TestCase):
    """Real retained inputs, deterministic expectation (codex 7ebb9c65): the exact epoch/station pins are asserted only when
    the inputs' identity equals the pinned sidecar; structural properties of the rule are asserted on any bytes."""

    @classmethod
    def setUpClass(cls):
        cls.fx = FX.load(_CACHE)
        cls.den, cls.ident_d, cls.ident_l, cls.sel = FX.selection(cls.fx)
        cls.pinned = json.loads(FX.SIDE_CAR.read_text(encoding="utf-8")) if FX.SIDE_CAR.exists() else None

    def test_structural_full_support_selection_on_real_inputs(self):
        fx, sel = self.fx, self.sel
        self.assertTrue(self.den["available"]); self.assertEqual(self.ident_d, self.ident_l)
        self.assertEqual(sel.status, "OK")
        self.assertEqual(sel.stations_with_real_samples, len(fx["cap"]["codes"]))
        sup = fx["cap"]["support"]; times = fx["cap"]["times"]; i = sel.index
        # the selected epoch is the NEWEST admissible epoch under the rule's own definition, recomputed here independently
        self.assertEqual(i, int(R.valid_denominator_epochs(sup, R.EDGE_MARGIN).max()))  # returns indices, not a mask
        self.assertEqual(sel.epoch, times[i].date().isoformat())
        # the rule's own definition, recomputed here without the module: every station has a real sample AT i, no station's
        # series starts after i-m or ends before i+m (interior gaps stay interpolated), and i is >= m rows from either grid end
        m = R.EDGE_MARGIN
        self.assertTrue(sup[i].all())
        for k in range(sup.shape[1]):
            real = np.flatnonzero(sup[:, k]); self.assertLessEqual(int(real[0]), i - m); self.assertGreaterEqual(int(real[-1]), i + m)
        self.assertTrue(m <= i <= len(times) - 1 - m)
        # and it is the NEWEST such epoch: every later epoch violates at least one of those conditions
        for j in range(i + 1, len(times) - m):
            ok = sup[j].all() and all(np.flatnonzero(sup[:, k])[-1] >= j + m for k in range(sup.shape[1]))
            self.assertFalse(ok)
        self.assertLessEqual((FX.TARGET.date() - times[i].date()).days, R.MAX_LAG_DAYS)

    def test_pinned_expectation_binds_to_input_identity(self):
        if self.pinned is None:
            self.skipTest("no sidecar: run `python lg_real_istanbul_fixture.py <cache_dir>` against immutable retained bytes")
        if self.pinned["input_manifest_sha256"] != self.fx["input_manifest_sha256"] or self.pinned.get("edge_margin") != R.EDGE_MARGIN:
            self.skipTest(f"retained inputs/rule differ from the pinned sidecar: pinned {self.pinned['input_manifest_sha256'][:16]} margin {self.pinned.get('edge_margin')}"
                          f" vs current {self.fx['input_manifest_sha256'][:16]} margin {R.EDGE_MARGIN}; current selection {self.sel.status} {self.sel.epoch}"
                          f" ({self.sel.stations_with_real_samples} stations) -- regenerate the sidecar deliberately")
        e = self.pinned["expected"]
        self.assertEqual(self.sel.status, e["status"]); self.assertEqual(self.sel.epoch, e["epoch"])
        self.assertEqual(self.sel.stations_with_real_samples, e["stations_with_real_samples"])

    def test_unbound_identity_refuses_on_real_inputs(self):
        s = R.select_numerator(self.fx["cap"]["times"], self.fx["cap"]["support"], self.fx["res"].lambda_geo_grid, FX.TARGET.date(), denominator=0.0185)
        self.assertEqual(s.status, "OPERATOR_IDENTITY_UNBOUND")

if __name__ == "__main__":
    unittest.main()
