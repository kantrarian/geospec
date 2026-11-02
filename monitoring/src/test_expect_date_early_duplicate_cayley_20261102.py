#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""--expect-date ASSERTION and the EARLY duplicate refusal (cayley 2026-11-02; codex afb97398 findings 1 and 2).

Attempt B (2026-11-02 04:36Z) showed two producer defects: a manual run collected the wrong day (D3's New York rule vs
the runner's UTC B6 rule), and a run of an already-published day re-scored and OVERWROTE the canonical local record
before publish_revision refused REVISION_EXISTS. Both refusals now happen in main() BEFORE the preflight-dependent
scoring, the provider fetch and save_results. Hermetic: the REAL main() over this checkout's REAL public journal, with
every fetch seam tripwired and the store + default output directory snapshotted; nothing is fetched or written.
  python -I -B monitoring/src/test_expect_date_early_duplicate_cayley_20261102.py"""
import ast
import io
import logging
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_daily_revision_model_redkats_cayley as T  # noqa: E402  (the existing bar's announced obspy shim)

T._shim_obspy_if_absent()
import run_ensemble_daily as RED  # noqa: E402

SEAMS = ["run_all_regions", "fetch_earthquake_events", "fetch_ngl_lambda_geo", "fetch_region_events", "save_results"]


def _snapshot(p):
    if not os.path.isdir(p):
        return None
    return sorted(os.path.relpath(os.path.join(r, f), p) for r, _d, fs in os.walk(p) for f in fs)


def _published_days():
    raw = RED.REV.journal_bytes(RED.REPO_ROOT)
    return set(RED.REV.current_map(RED.REV.parse_journal(raw)))


class _Run:
    """main() with argv, tripwires on every fetch/write seam, optional REV stubs; returns (rc, calls, logs, unchanged)."""

    def __init__(self, argv, *, stub_view=False, refuse_checkout=None, now=None):
        self.argv, self.stub_view, self.refuse_checkout, self.now = argv, stub_view, refuse_checkout, now
        self.replay_mode = None

    def __call__(self):
        calls, saved, saved_rev = [], {}, {}
        store = os.path.join(str(RED.REPO_ROOT), "docs", "ensemble")
        out_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(RED.__file__)), "..", "data", "ensemble_results"))
        before = (_snapshot(store), _snapshot(out_dir))
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        RED.logger.addHandler(handler)

        def trip(name):
            def _t(*a, **k):
                calls.append(name)
                raise AssertionError("TRIPWIRE " + name)
            return _t
        saved_argv = sys.argv
        saved_dt, saved_mode = RED.datetime, RED._HISTORICAL_REPLAY
        RED._HISTORICAL_REPLAY = False                      # the module default; --date sets it inside main()
        if self.now is not None:
            fixed = self.now

            class _FixedClock(saved_dt):
                @classmethod
                def now(cls, tz=None):
                    return fixed if tz is not None else fixed.replace(tzinfo=None)
            RED.datetime = _FixedClock
        try:
            for name in SEAMS:
                if hasattr(RED, name):
                    saved[name] = getattr(RED, name)
                    setattr(RED, name, trip(name))
            if self.stub_view:
                saved_rev["committed_inputs_view"] = RED.REV.committed_inputs_view
                RED.REV.committed_inputs_view = lambda _repo, _commit="HEAD", **_k: {"commit": "1" * 40, "code": {}, "calibration": {}}
                saved_rev["checkout_matches_committed"] = RED.REV.checkout_matches_committed

                def _check(_repo, _view):
                    if self.refuse_checkout:
                        raise RED.REV.RevisionRefusal(self.refuse_checkout)
                RED.REV.checkout_matches_committed = _check
            sys.argv = ["run_ensemble_daily.py"] + list(self.argv)
            try:
                rc = RED.main()
            except AssertionError as e:
                rc = "TRIPWIRE:%s" % e
        finally:
            self.replay_mode = RED._HISTORICAL_REPLAY
            RED.datetime, RED._HISTORICAL_REPLAY = saved_dt, saved_mode
            sys.argv = saved_argv
            for n, f in saved.items():
                setattr(RED, n, f)
            for n, f in saved_rev.items():
                setattr(RED.REV, n, f)
            RED.logger.removeHandler(handler)
        after = (_snapshot(store), _snapshot(out_dir))
        return rc, calls, buf.getvalue(), before == after


class ExpectDateAndEarlyDuplicate(unittest.TestCase):
    def test_ast_both_refusals_precede_scoring_fetch_and_local_write(self):
        src = io.open(os.path.join(HERE, "run_ensemble_daily.py"), encoding="utf-8").read()
        main = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
        lines = src.splitlines()

        def first(text):
            return min(i + 1 for i in range(main.lineno - 1, main.end_lineno) if text in lines[i])

        def first_call(name):
            return min(n.lineno for n in ast.walk(main) if isinstance(n, ast.Call) and
                       ((isinstance(n.func, ast.Name) and n.func.id == name) or (isinstance(n.func, ast.Attribute) and n.func.attr == name)))
        l_expect, l_dup = first("EXPECTED_DAY_MISMATCH"), first("REVISION_EXISTS_BEFORE_FETCH")
        for later in ("run_all_regions", "save_results", "publish_revision"):
            self.assertLess(l_expect, first_call(later), later)
            self.assertLess(l_dup, first_call(later), later)

    def test_expect_date_mismatch_refuses_before_anything(self):
        rc, calls, logs, unchanged = _Run(["--date", "2026-10-31", "--expect-date", "2026-10-30"])()
        self.assertEqual(rc, 12)
        self.assertEqual(calls, [])
        self.assertTrue(unchanged)
        self.assertIn("EXPECTED_DAY_MISMATCH", logs)

    def test_malformed_expect_date_refuses(self):
        rc, calls, logs, unchanged = _Run(["--date", "2026-10-31", "--expect-date", "2026-10-31T00"])()
        self.assertEqual((rc, calls, unchanged), (12, [], True))
        self.assertIn("EXPECTED_DAY_MISMATCH", logs)

    def test_matching_expect_date_passes_through_to_the_next_check(self):
        rc, calls, logs, unchanged = _Run(["--date", "2026-10-31", "--expect-date", "2026-10-31"], stub_view=True,
                                          refuse_checkout="INPUTS_CODE_DIRTY: planted after the assertion")()
        self.assertEqual((rc, calls, unchanged), (12, [], True))
        self.assertNotIn("EXPECTED_DAY_MISMATCH", logs)
        self.assertIn("planted after the assertion", logs)

    def test_published_day_refuses_before_fetch_and_before_overwriting_the_local_record(self):
        days = _published_days()
        day = max(days)
        rc, calls, logs, unchanged = _Run(["--date", day], stub_view=True)()
        self.assertEqual(rc, 12)
        self.assertEqual(calls, [], "no scoring, fetch or save_results before the duplicate refusal")
        self.assertTrue(unchanged)
        self.assertIn("REVISION_EXISTS_BEFORE_FETCH", logs)

    def test_unpublished_day_is_not_refused_by_the_duplicate_check(self):
        import datetime as d
        day = (d.date.fromisoformat(max(_published_days())) + d.timedelta(days=1)).isoformat()
        rc, calls, logs, unchanged = _Run(["--date", day], stub_view=True)()
        self.assertTrue(str(rc).startswith("TRIPWIRE"), rc)       # reached scoring: the check let it through
        self.assertEqual(calls[:1], ["run_all_regions"])
        self.assertNotIn("REVISION_EXISTS_BEFORE_FETCH", logs)

    def test_auto_and_auto_with_matching_expectation_behave_identically(self):
        """codex 05f6019e: the assertion never changes the ordinary mode. A fixed clock puts the auto day on the first
        unpublished day; both runs reach scoring the same way and neither selects the --date historical replay."""
        import datetime as d
        day = (d.date.fromisoformat(max(_published_days())) + d.timedelta(days=1))
        now = d.datetime(day.year, day.month, day.day, 12, 0, tzinfo=d.timezone.utc) + d.timedelta(days=2)
        plain = _Run([], stub_view=True, now=now)
        r1 = plain()
        expect = _Run(["--expect-date", day.isoformat()], stub_view=True, now=now)
        r2 = expect()
        self.assertTrue(str(r1[0]).startswith("TRIPWIRE") and str(r2[0]).startswith("TRIPWIRE"), (r1[0], r2[0]))
        self.assertEqual(r1[1], r2[1])
        self.assertIs(plain.replay_mode, False)
        self.assertIs(expect.replay_mode, False)
        self.assertIn("Starting ensemble assessment for %s" % day.isoformat(), r2[2])

    def test_utc_midnight_crossed_between_claim_and_scoring_refuses_without_substitution(self):
        """D3 claimed day X before UTC midnight; recalibration (or a late launch) carried the runner past midnight, so
        its auto day is X+1. The bound expectation X refuses -- the day is never moved to X+1 silently."""
        import datetime as d
        claimed = d.date.fromisoformat(max(_published_days())) + d.timedelta(days=1)
        after_midnight = d.datetime(claimed.year, claimed.month, claimed.day, 0, 5, tzinfo=d.timezone.utc) + d.timedelta(days=3)
        rc, calls, logs, unchanged = _Run(["--expect-date", claimed.isoformat()], stub_view=True, now=after_midnight)()
        self.assertEqual((rc, calls, unchanged), (12, [], True))
        self.assertIn("EXPECTED_DAY_MISMATCH: this run scores %s" % (claimed + d.timedelta(days=1)).isoformat(), logs)

    def test_explicit_replay_remains_store_free_with_or_without_expectation(self):
        import datetime as d
        day = (d.date.fromisoformat(max(_published_days())) + d.timedelta(days=1)).isoformat()
        for argv in (["--date", day], ["--date", day, "--expect-date", day]):
            run = _Run(argv, stub_view=True)
            rc = run()[0]
            self.assertTrue(str(rc).startswith("TRIPWIRE"), (argv, rc))
            self.assertIs(run.replay_mode, True, argv)

    def test_rescore_bypasses_only_the_early_check(self):
        day = max(_published_days())
        rc, calls, logs, _u = _Run(["--date", day, "--rescore", "owner-authorized test"], stub_view=True)()
        self.assertNotIn("REVISION_EXISTS_BEFORE_FETCH", logs)
        self.assertTrue(str(rc).startswith("TRIPWIRE"), rc)


if __name__ == "__main__":
    unittest.main()
