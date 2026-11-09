"""ngl-refresh-candidate-v1 / lg-calibration-plan-v1 tests (cayley 2026-11-09). A recording fake session stands in for
NGL; every file lives in a temp dir; the run day and attempt instant are fixture clocks. The two production
characterisation tests read production's own wall-clock rule with mtimes set relative to now (+/- 60 s margins)."""
import json
import os
import tempfile
import time
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import calibration_eligibility_fixtures as FX
FX.install_stubs()
import live_data as LD                          # noqa: E402
import lg_producer_candidate as P               # noqa: E402
import lg_refresh_candidate as RF               # noqa: E402
import test_lg_producer_candidate as TT         # noqa: E402

RUN_DAY = date(2026, 11, 9)
URL20 = LD.NGLLiveAcquisition.BASE_URL_IGS20 + '/QTK1.tenv'
URL14 = LD.NGLLiveAcquisition.BASE_URL_IGS14 + '/QTK1.tenv3'


class FakeSession:
    verify = True  # recording fake explicitly models a verified session

    def __init__(self, responses):
        self.responses = {k: list(v) for k, v in responses.items()}
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        item = self.responses[url].pop(0)
        if isinstance(item, Exception):
            raise item
        def raise_for_status(status=item[0]):
            if status >= 400:
                raise RuntimeError('HTTP %d' % status)
        return SimpleNamespace(status_code=item[0], text=item[1], raise_for_status=raise_for_status)


def clock():
    return '2026-11-09T12:00:30Z'


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix='lgrf-')
        self.root = Path(self._tmp.name)
        self.cache = self.root / 'cache'
        self.src = self.root / 'source'

    def tearDown(self):
        self._tmp.cleanup()

    def body(self, name='QTK1.igs20.tenv', **kw):
        """NGL-layout text from the producer tests' writer (one file of a fresh source cache)."""
        src = self.root / ('src-%d' % len(list(self.root.glob('src-*'))))
        TT.write_cache(src, **kw)
        return (src / name).read_text(encoding='utf-8')

    def acq(self, session, run_day=RUN_DAY, run_id='tick-45'):
        self.sleeps = []
        a = RF.RefreshPolicyNGL(self.cache, run_day=run_day, run_id=run_id, ledger_path=self.root / 'ledger.json',
                                session=session, clock=clock, sleep=self.sleeps.append)
        TT.write_cache(self.src)
        a.station_catalog = json.loads((self.src / 'ngl_catalog.json').read_text(encoding='utf-8'))
        return a


class ProductionCharacterisation(Base):
    def test_production_refresh_is_a_wall_clock_age_test_that_races_the_cadence(self):
        TT.write_cache(self.cache)
        f = self.cache / 'QTK1.igs20.tenv'
        prod = LD.NGLLiveAcquisition(self.cache)
        prod.station_catalog = json.loads((self.cache / 'ngl_catalog.json').read_text(encoding='utf-8'))
        prod._session = FakeSession({URL20: [(200, f.read_text(encoding='utf-8'))]})
        now = time.time()
        os.utime(f, (now, now - 86400 + 60))                  # 23 h 59 m old: reused, no request
        prod.download_station('QTK1', 90, datetime(2026, 11, 4, 23, 59, 59))
        self.assertEqual(prod._session.calls, [])
        os.utime(f, (now, now - 86400 - 60))                  # 24 h 1 m old: fetched
        prod.download_station('QTK1', 90, datetime(2026, 11, 4, 23, 59, 59))
        self.assertEqual(prod._session.calls, [URL20])

    def test_production_failed_igs20_refresh_switches_the_station_to_igs14(self):
        TT.write_cache(self.cache)                                              # IGS20 file, data to LAST
        tenv3 = self.body('QTK1.tenv3', tenv3_only=('QTK1',), last={'QTK1': TT.SCORED - timedelta(days=20)})
        (self.cache / 'QTK1.tenv3').write_text(tenv3, encoding='utf-8')
        now = time.time()
        os.utime(self.cache / 'QTK1.igs20.tenv', (now, now - 2 * 86400))       # stale: production tries a refresh
        prod = LD.NGLLiveAcquisition(self.cache)
        prod.station_catalog = json.loads((self.cache / 'ngl_catalog.json').read_text(encoding='utf-8'))
        prod._session = FakeSession({URL20: [(503, '')]})
        st = prod.download_station('QTK1', 90, datetime(2026, 11, 4, 23, 59, 59))
        self.assertEqual(st.times[-1].date(), TT.SCORED - timedelta(days=20))  # the IGS14 series, not the IGS20 one
        cand = self.acq(FakeSession({URL20: [(503, '')] * RF.MAX_ATTEMPTS}))
        st = cand.download_station('QTK1', 90, datetime(2026, 11, 4, 23, 59, 59))
        self.assertEqual(st.times[-1].date(), TT.SCORED)                       # the kept IGS20 file
        self.assertEqual(cand.records[0]['outcome'], 'FAILED_KEPT_PREVIOUS')
        self.assertEqual(cand.session.calls, [URL20] * RF.MAX_ATTEMPTS)        # retried (transient); IGS14 never asked


class RefreshPolicy(Base):
    def test_one_attempt_per_run_day_and_none_on_a_repeat_call(self):
        text = self.body()
        s = FakeSession({URL20: [(200, text), (200, text)]})
        a = self.acq(s)
        a.download_station('QTK1', 90, datetime(2026, 11, 4, 23, 59, 59))
        a.download_station('QTK1', 90, datetime(2026, 11, 4, 23, 59, 59))
        self.assertEqual(s.calls, [URL20])
        self.assertEqual([r['outcome'] for r in a.records], ['REFRESHED_NEW_EPOCHS', 'NOT_DUE'])
        again = self.acq(s, run_id='rerun-same-day')                           # ledger persists across instances
        again.download_station('QTK1', 90, datetime(2026, 11, 4, 23, 59, 59))
        self.assertEqual(s.calls, [URL20])
        self.assertEqual(again.records[0]['prior_run_id'], 'tick-45')
        nxt = self.acq(s, run_day=RUN_DAY + timedelta(days=1), run_id='tick-46')
        nxt.download_station('QTK1', 90, datetime(2026, 11, 5, 23, 59, 59))
        self.assertEqual(s.calls, [URL20, URL20])
        self.assertEqual(nxt.records[0]['outcome'], 'REFRESHED_SAME_CONTENT')

    def test_success_with_old_observations_is_distinguished_from_new_data(self):
        old = self.body(last={'QTK1': TT.SCORED - timedelta(days=7)})
        new = self.body()
        reprocessed = old.replace('0.00100 0.00100', '0.00200 0.00100', 1)
        self.assertNotEqual(reprocessed, old)
        (self.cache).mkdir(parents=True, exist_ok=True)
        (self.cache / 'QTK1.igs20.tenv').write_text(old, encoding='utf-8')
        s = FakeSession({URL20: [(200, reprocessed), (200, new)]})
        a = self.acq(s)
        a.download_station('QTK1', 90, datetime(2026, 11, 4, 23, 59, 59))
        r = a.records[0]
        self.assertEqual(r['outcome'], 'REFRESHED_CHANGED_NO_NEW_EPOCHS')
        self.assertEqual(r['data_end_before'], r['data_end_after'])
        self.assertEqual(r['data_end_after'], (TT.SCORED - timedelta(days=7)).isoformat())
        self.assertEqual(r['data_age_days'], (RUN_DAY - (TT.SCORED - timedelta(days=7))).days)
        b = self.acq(s, run_day=RUN_DAY + timedelta(days=1), run_id='tick-46')
        b.download_station('QTK1', 90, datetime(2026, 11, 4, 23, 59, 59))
        self.assertEqual(b.records[0]['outcome'], 'REFRESHED_NEW_EPOCHS')
        self.assertEqual(b.records[0]['data_end_after'], TT.LAST.isoformat())

    def test_failure_keeps_the_previous_bytes_and_reports_their_true_age(self):
        text = self.body(last={c: TT.SCORED - timedelta(days=11) for c, _, _ in TT.STATIONS})
        self.cache.mkdir(parents=True, exist_ok=True)
        f = self.cache / 'QTK1.igs20.tenv'
        f.write_text(text, encoding='utf-8')
        before = f.read_bytes()
        a = self.acq(FakeSession({URL20: [ConnectionError('x' * 5000)] * RF.MAX_ATTEMPTS}))
        st = a.download_station('QTK1', 90, datetime(2026, 11, 4, 23, 59, 59))
        r = a.records[0]
        self.assertEqual(f.read_bytes(), before)
        self.assertEqual((r['outcome'], r['failure_class'], r['cause']), ('FAILED_KEPT_PREVIOUS', 'NETWORK', 'ConnectionError'))
        self.assertTrue(r['detail'].startswith('ConnectionError: x'))
        self.assertEqual(len(r['detail']), len('ConnectionError: ') + RF.ERROR_CHARS)
        self.assertEqual(r['sha256_before'], r['sha256_after'])
        self.assertEqual(r['data_age_days'], (RUN_DAY - (TT.SCORED - timedelta(days=11))).days)
        self.assertEqual(st.times[-1].date(), TT.SCORED - timedelta(days=11))

    def test_no_previous_file_and_a_failed_refresh_leaves_the_station_unavailable_by_record(self):
        a = self.acq(FakeSession({URL20: [(404, 'nf')], URL14: [(404, 'nf')]}))
        self.assertIsNone(a.download_station('QTK1', 90, datetime(2026, 11, 4, 23, 59, 59)))
        self.assertEqual([(r['file'], r['outcome'], r['cause']) for r in a.records],
                         [('QTK1.igs20.tenv', 'FAILED_NO_PREVIOUS', 'HTTP 404'),
                          ('QTK1.tenv3', 'FAILED_NO_PREVIOUS', 'HTTP 404')])
        self.assertFalse((self.cache / 'QTK1.igs20.tenv').exists())

    def test_refresh_is_atomic_and_leaves_no_partial_file(self):
        text = self.body()
        a = self.acq(FakeSession({URL20: [(200, text)]}))
        a.download_station('QTK1', 90, datetime(2026, 11, 4, 23, 59, 59))
        self.assertEqual((self.cache / 'QTK1.igs20.tenv').read_text(encoding='utf-8'), text)
        self.assertEqual(list(self.cache.glob('*.part')) + list(self.root.glob('*.part')), [])
        self.assertEqual(a.records[0]['sha256_after'], RF._sha(text.encode('utf-8')))

    def test_data_end_matches_the_production_parser(self):
        text = self.body(last={'QTK1': TT.SCORED - timedelta(days=3)})
        prod = LD.NGLLiveAcquisition(self.root / 'p')
        TT.write_cache(self.src)
        prod.station_catalog = json.loads((self.src / 'ngl_catalog.json').read_text(encoding='utf-8'))
        st = prod._parse_tenv('QTK1', text, 10 ** 5, datetime(2100, 1, 1))
        self.assertEqual(RF.data_end_day(text), st.times[-1].date().isoformat())

    def test_an_explicit_session_and_run_identity_are_required(self):
        with self.assertRaises(ValueError):
            RF.RefreshPolicyNGL(self.cache, run_day=RUN_DAY, run_id='t', ledger_path=self.root / 'l.json',
                                session=None, clock=clock)
        with self.assertRaises(ValueError):
            RF.RefreshPolicyNGL(self.cache, run_day=datetime(2026, 11, 9), run_id='t',
                                ledger_path=self.root / 'l.json', session=FakeSession({}), clock=clock)


class ValidationRepair(Base):
    """codex 7ebb9c65 P1: an HTTP 200 body is staged and validated before it may replace a usable file. Each refused
    body leaves the previous bytes, the previous IGS20 station, no IGS14 request and a typed failure record."""
    TARGET = datetime(2026, 11, 4, 23, 59, 59)

    def previous(self, **kw):
        self.cache.mkdir(parents=True, exist_ok=True)
        f = self.cache / 'QTK1.igs20.tenv'
        f.write_text(self.body(**kw), encoding='utf-8')
        return f, f.read_bytes()

    def refused(self, body, cause):
        f, before = self.previous()
        s = FakeSession({URL20: [(200, body)], URL14: [(404, '')]})
        a = self.acq(s)
        st = a.download_station('QTK1', 90, self.TARGET)                        # must not raise
        r = a.records[0]
        self.assertEqual(f.read_bytes(), before)
        self.assertIsNotNone(st)
        self.assertEqual(st.times[-1].date(), TT.SCORED)                         # the previous IGS20 station
        self.assertEqual(s.calls, [URL20])                                       # IGS14 never asked
        self.assertEqual((r['outcome'], r['failure_class'], r['cause']), ('FAILED_KEPT_PREVIOUS', 'VALIDATION', cause))
        self.assertEqual(r['sha256_after'], r['sha256_before'])
        self.assertEqual(r['data_end_after'], TT.LAST.isoformat())
        self.assertEqual(len(a.records), 1)
        self.assertEqual(a.ledger.get('QTK1.igs20.tenv')['cause'], cause)
        self.assertEqual(list(self.cache.glob('*.part')), [])
        return r

    def test_html_200_is_refused(self):                                          # codex reproduction 1
        self.refused('<html>Service temporarily unavailable</html>', 'FOREIGN_STATION_ROWS')

    def test_truncated_200_is_refused(self):                                     # codex reproduction 2
        text = (self.root / 'x').parent and self.body()
        self.refused('\n'.join(text.splitlines()[:10]), 'NO_USABLE_WINDOW')

    def test_nan_epoch_200_is_refused_without_an_exception(self):                # codex reproduction 3
        # codex's exact 10-field body now fails the byte-verified 17-field IGS20 layout first; a 17-field NaN epoch
        # still reaches the epoch check
        self.refused('QTK1 26NOV04 nan 0 0 0 0 0.1 0.2 0.3', 'LAYOUT_CONTRACT_MISMATCH')

    def test_a_17_field_nan_epoch_is_refused_as_nonfinite(self):
        self.refused('QTK1 26NOV04 nan 0 0 0 139.1 0.1 0.2 0.3 0 0.001 0.001 0.004 0 0 0', 'NONFINITE_EPOCH')

    def test_a_body_in_the_16_field_readme_layout_is_refused_not_misread(self):
        """NGL's README_tenv describes 16 fields; the retained bytes have 17 (grassmann inventory 29b18201). A body
        without the reference-longitude column would shift E/N/U by one under the production parser."""
        rows = [l.split() for l in self.body().splitlines() if l.strip()]
        self.assertEqual({len(r) for r in rows}, {RF.IGS20_TENV_FIELDS})
        readme16 = '\n'.join(' '.join(r[:6] + r[7:]) for r in rows) + '\n'
        r = self.refused(readme16, 'LAYOUT_CONTRACT_MISMATCH')
        self.assertIn('0 valid rows', r['detail'])

    def test_a_nonfinite_coordinate_in_the_window_is_refused(self):
        text = self.body()
        lines = text.splitlines()
        i = next(k for k, l in enumerate(lines) if l.startswith('QTK1 26OCT20'))
        parts = lines[i].split()
        parts[7] = 'nan'                                                         # east (m), inside the window
        lines[i] = ' '.join(parts)
        self.refused('\n'.join(lines) + '\n', 'NONFINITE_COORDINATES')

    def test_another_stations_rows_are_refused(self):
        self.refused(self.body('QTK2.igs20.tenv'), 'FOREIGN_STATION_ROWS')

    def test_a_body_that_loses_covered_window_days_is_refused(self):
        r = self.refused(self.body(first=date(2026, 9, 1)), 'COVERAGE_REDUCED')
        self.assertIn('first 2026-08-07', r['detail'])

    def test_a_body_whose_data_end_moves_backwards_is_refused(self):
        r = self.refused(self.body(last={'QTK1': TT.SCORED + timedelta(days=4)}), 'DATA_END_REGRESSED')
        self.assertIn('< previous ' + TT.LAST.isoformat(), r['detail'])

    def test_a_valid_changed_body_is_accepted(self):                            # success control
        f, before = self.previous(last={'QTK1': TT.SCORED - timedelta(days=7)})
        new = self.body()
        a = self.acq(FakeSession({URL20: [(200, new)]}))
        st = a.download_station('QTK1', 90, self.TARGET)
        r = a.records[0]
        self.assertEqual((r['outcome'], r['failure_class'], r['cause']), ('REFRESHED_NEW_EPOCHS', None, None))
        self.assertEqual(f.read_text(encoding='utf-8'), new)
        self.assertEqual(r['data_end_after'], TT.LAST.isoformat())
        self.assertEqual(st.times[-1].date(), TT.SCORED)

    def test_a_write_failure_keeps_the_previous_file(self):
        f, before = self.previous()
        new = self.body(extra=())
        new = new.replace('0.00100 0.00100', '0.00300 0.00100', 1)
        a = self.acq(FakeSession({URL20: [(200, new)]}))
        with mock.patch.object(RF.os, 'replace', side_effect=PermissionError('locked by another process')):
            st = a.download_station('QTK1', 90, self.TARGET)
        r = a.records[0]
        self.assertEqual(f.read_bytes(), before)
        self.assertEqual((r['outcome'], r['failure_class'], r['cause']), ('FAILED_KEPT_PREVIOUS', 'WRITE', 'PermissionError'))
        self.assertEqual(list(self.cache.glob('*.part')) + list(self.root.glob('*.part')), [])
        self.assertTrue(r['ledger_error'].startswith('PermissionError'))          # the ledger write failed too: recorded
        self.assertEqual(a.ledger.get('QTK1.igs20.tenv')['cause'], 'PermissionError')
        self.assertIsNotNone(st)

    def test_a_validator_defect_is_a_typed_failure(self):
        f, before = self.previous()
        a = self.acq(FakeSession({URL20: [(200, self.body())]}))
        with mock.patch.object(RF, '_station_ok', side_effect=RuntimeError('boom')):
            st = a.download_station('QTK1', 90, self.TARGET)
        r = a.records[0]
        self.assertEqual((r['outcome'], r['failure_class'], r['cause']), ('FAILED_KEPT_PREVIOUS', 'VALIDATION', 'VALIDATOR_ERROR'))
        self.assertEqual(r['detail'], 'RuntimeError: boom')
        self.assertEqual(f.read_bytes(), before)
        self.assertIsNotNone(st)

    def test_an_invalid_200_with_no_previous_file_writes_nothing(self):
        s = FakeSession({URL20: [(200, '<html>x</html>')], URL14: [(200, '<html>x</html>')]})
        a = self.acq(s)
        self.assertIsNone(a.download_station('QTK1', 90, self.TARGET))
        self.assertEqual([(r['file'], r['outcome'], r['cause']) for r in a.records],
                         [('QTK1.igs20.tenv', 'FAILED_NO_PREVIOUS', 'FOREIGN_STATION_ROWS'),
                          ('QTK1.tenv3', 'FAILED_NO_PREVIOUS', 'FOREIGN_STATION_ROWS')])
        self.assertEqual(sorted(p.name for p in self.cache.glob('QTK1*')), [])

    def test_a_recovering_station_without_a_usable_window_is_not_blocked(self):
        """The previous bytes end long before the window (no usable window); a body with new but still too few window
        rows is structurally valid and accepted, recorded as an unusable window; a truncation of it is still refused."""
        f, _ = self.previous(last={'QTK1': date(2026, 7, 20)})
        revived = self.body(last={'QTK1': TT.SCORED}, first=date(2026, 10, 20))   # 16 rows in the window (< 30)
        a = self.acq(FakeSession({URL20: [(200, revived)], URL14: [(404, '')]}))
        st = a.download_station('QTK1', 90, self.TARGET)
        r = a.records[0]
        self.assertIsNone(st)                                                    # not usable yet, by the parser's rule
        self.assertEqual((r['outcome'], r['failure_class']), ('REFRESHED_NEW_EPOCHS', None))
        self.assertIn('window_usable false', r['validation_detail'])
        self.assertEqual(f.read_text(encoding='utf-8'), revived)
        self.assertEqual(r['data_end_after'], TT.SCORED.isoformat())
        back = self.acq(FakeSession({URL20: [(200, self.body(last={'QTK1': date(2026, 7, 10)}))]}),
                        run_day=RUN_DAY + timedelta(days=1), run_id='tick-46')
        back.download_station('QTK1', 90, self.TARGET)
        self.assertEqual((back.records[0]['outcome'], back.records[0]['cause']), ('FAILED_KEPT_PREVIOUS', 'DATA_END_REGRESSED'))

    def test_epoch_scan_never_raises(self):
        for body, cause in (('', 'NO_EPOCH_ROWS'), ('site x y\n# c\n', 'NO_EPOCH_ROWS'), ('QTK1 a inf', 'NONFINITE_EPOCH'),
                            ('QTK1 a 1e308', 'NONFINITE_EPOCH'), ('QTK1 a b', 'UNPARSEABLE_EPOCH'), ('QTK1 a', 'UNPARSEABLE_EPOCH')):
            self.assertEqual(RF.epoch_scan(body, 'QTK1')[1], cause, body)


class PreActivation(Base):
    """Bounded retry and same-day recovery, TLS-verified sessions, catalog refresh only on calibration runs, and the
    catalog binding check (codex 7ebb9c65 integrated-packet items owned by cayley)."""
    TARGET = datetime(2026, 11, 4, 23, 59, 59)
    CAT = LD.NGLLiveAcquisition.CATALOG_URL

    def previous(self):
        self.cache.mkdir(parents=True, exist_ok=True)
        f = self.cache / 'QTK1.igs20.tenv'
        f.write_text(self.body(last={'QTK1': TT.SCORED - timedelta(days=7)}), encoding='utf-8')
        return f

    def llh(self, stations=None):
        return '\n'.join(f'{c} {lat:.4f} {lon - 360.0:.4f} 50.0' for c, lat, lon in (stations or TT.STATIONS)) + '\n'

    def test_transient_failures_are_retried_with_backoff_and_bounded(self):
        self.previous()
        new = self.body()
        a = self.acq(FakeSession({URL20: [(503, ''), ConnectionError('reset'), (200, new)]}))
        a.download_station('QTK1', 90, self.TARGET)
        r = a.records[0]
        self.assertEqual(r['outcome'], 'REFRESHED_NEW_EPOCHS')
        self.assertEqual([(x['http_status'], x['failure_class']) for x in r['attempts']],
                         [(503, 'HTTP'), (None, 'NETWORK'), (200, None)])
        self.assertEqual(self.sleeps, [5, 30])
        b = self.acq(FakeSession({URL20: [(503, '')] * 5}), run_day=RUN_DAY + timedelta(days=1), run_id='tick-46')
        b.download_station('QTK1', 90, self.TARGET)
        self.assertEqual((len(b.records[0]['attempts']), b.records[0]['outcome']), (RF.MAX_ATTEMPTS, 'FAILED_KEPT_PREVIOUS'))
        self.assertEqual(len(b.session.calls), RF.MAX_ATTEMPTS)

    def test_a_per_run_retry_budget_bounds_an_outage(self):
        """A provider outage costs at most RETRY_BUDGET extra attempts (and their backoff) per run, not 2 per file."""
        urls = [f'{LD.NGLLiveAcquisition.BASE_URL_IGS20}/QTK{k}.tenv' for k in (1, 2, 3)]
        tenv3 = [f'{LD.NGLLiveAcquisition.BASE_URL_IGS14}/QTK{k}.tenv3' for k in (1, 2, 3)]
        s = FakeSession({u: [(503, '')] * 5 for u in urls + tenv3})
        self.sleeps = []
        a = RF.RefreshPolicyNGL(self.cache, run_day=RUN_DAY, run_id='outage', ledger_path=self.root / 'ledger.json',
                                session=s, clock=clock, sleep=self.sleeps.append, retry_budget=3)
        a.station_catalog = {}
        for k in (1, 2, 3):
            a.download_station(f'QTK{k}', 90, self.TARGET)
        per_file = [len(r['attempts']) for r in a.records]
        self.assertEqual(per_file, [3, 2, 1, 1, 1, 1])                           # budget 3 = 2 + 1, then single attempts
        self.assertEqual(len(s.calls), sum(per_file))
        self.assertEqual(self.sleeps, [5, 30, 5])
        self.assertTrue(a.records[1]['attempts'][-1]['retry_budget_exhausted'])
        self.assertTrue(all(r['attempts'][-1].get('retry_budget_exhausted') for r in a.records[2:]))
        self.assertEqual(a.retries_left, 0)

    def test_permanent_and_validation_failures_are_not_retried(self):
        self.previous()
        for body in ((404, 'nf'), (200, '<html>x</html>')):
            a = self.acq(FakeSession({URL20: [body] * 3, URL14: [(404, '')] * 3}),
                         run_day=RUN_DAY + timedelta(days=10 + body[0] % 7), run_id='t-%d' % body[0])
            a.download_station('QTK1', 90, self.TARGET)
            self.assertEqual(len(a.records[0]['attempts']), 1, body)
            self.assertEqual(self.sleeps, [], body)

    def test_a_failed_file_may_be_retried_by_a_later_authorized_run_the_same_day(self):
        f = self.previous()
        s = FakeSession({URL20: [(503, '')] * RF.MAX_ATTEMPTS + [(200, self.body())] + [(200, self.body())]})
        a = self.acq(s, run_id='tick-45')
        a.download_station('QTK1', 90, self.TARGET)
        self.assertEqual(a.records[0]['outcome'], 'FAILED_KEPT_PREVIOUS')
        again = self.acq(s, run_id='tick-45')                                    # the same run: exhausted
        again.download_station('QTK1', 90, self.TARGET)
        self.assertEqual((again.records[0]['outcome'], again.records[0]['reason']), ('NOT_DUE', 'ATTEMPTS_EXHAUSTED_THIS_RUN'))
        recovery = self.acq(s, run_id='tick-45-recovery')                        # a later authorized run, same day
        recovery.download_station('QTK1', 90, self.TARGET)
        self.assertEqual(recovery.records[0]['outcome'], 'REFRESHED_NEW_EPOCHS')
        later = self.acq(s, run_id='tick-45-check')
        later.download_station('QTK1', 90, self.TARGET)
        self.assertEqual((later.records[0]['outcome'], later.records[0]['reason']), ('NOT_DUE', 'REFRESHED_THIS_RUN_DAY'))
        self.assertEqual(len(s.calls), RF.MAX_ATTEMPTS + 1)

    def test_an_unverified_tls_session_is_refused_and_the_wiring_session_verifies(self):
        bad = FakeSession({})
        bad.verify = False
        with self.assertRaises(ValueError):
            self.acq(bad)
        good = RF.make_verified_session()
        self.assertIs(good.verify, True)
        self.assertEqual(good.headers['User-Agent'], RF.USER_AGENT)
        self.acq(good)                                                           # constructing makes no request

    def test_tls_guard_rejects_unverified_and_untyped_settings(self):
        for value in (None, 0, '', '   ', False, 1, object()):
            with self.subTest(verify=repr(value)):
                bad = FakeSession({})
                bad.verify = value
                with self.assertRaises(ValueError):
                    self.acq(bad)
                self.assertEqual(bad.calls, [])
        with self.assertRaises(ValueError):
            self.acq(SimpleNamespace())  # no explicit verification setting

    def test_tls_guard_accepts_an_explicit_ca_bundle_path(self):
        import requests
        good = FakeSession({})
        good.verify = requests.certs.where()
        self.assertTrue(Path(good.verify).is_file())
        self.acq(good)
        self.assertEqual(good.calls, [])

    def test_the_catalog_is_refreshed_only_on_calibration_runs(self):
        TT.write_cache(self.cache)
        cat = self.cache / 'ngl_catalog.json'
        before = cat.read_bytes()
        now = time.time()
        os.utime(cat, (now, now - 10 * 86400))                                  # production would refetch at 7 days
        s = FakeSession({self.CAT: [(200, self.llh())]})
        a = self.acq(s)
        self.assertEqual(a.load_station_catalog(), json.loads(before))
        r = a.refresh_catalog(calibration=False)
        self.assertEqual((r['outcome'], r['reason']), ('NOT_DUE', 'CATALOG_REFRESH_ONLY_ON_CALIBRATION_RUNS'))
        self.assertEqual(s.calls, [])
        r = a.refresh_catalog(calibration=True, declared=[c for c, _, _ in TT.STATIONS])
        self.assertEqual(s.calls, [self.CAT])
        self.assertIn(r['outcome'], ('REFRESHED_CATALOG', 'REFRESHED_SAME_CONTENT'))
        prod = LD.NGLLiveAcquisition(self.root / 'prod')
        prod._session = FakeSession({self.CAT: [(200, self.llh())]})
        prod.load_station_catalog()                                              # production's parse + dump of the same text
        self.assertEqual(cat.read_bytes(), (self.root / 'prod' / 'ngl_catalog.json').read_bytes())

    def test_production_refreshes_the_catalog_on_a_seven_day_wall_clock(self):
        TT.write_cache(self.cache)
        now = time.time()
        os.utime(self.cache / 'ngl_catalog.json', (now, now - 7 * 86400 - 60))
        prod = LD.NGLLiveAcquisition(self.cache)
        prod._session = FakeSession({self.CAT: [(200, self.llh())]})
        prod.load_station_catalog()
        self.assertEqual(prod._session.calls, [self.CAT])

    def test_a_catalog_that_shrinks_loses_a_declared_station_or_is_invalid_is_refused(self):
        TT.write_cache(self.cache, extra=tuple((f'QX{k:02d}', 35.0 + k / 100, 139.0 + k / 100) for k in range(20)))
        cat = self.cache / 'ngl_catalog.json'
        before = cat.read_bytes()
        declared = [c for c, _, _ in TT.STATIONS]
        cases = [(self.llh(), 'CATALOG_SHRANK'),                                 # 5 of 25 stations
                 (self.llh([s for s in TT.STATIONS if s[0] != 'QTK4'] + [(f'QX{k:02d}', 35.0, 139.0 + k / 100) for k in range(20)]),
                  'DECLARED_STATION_MISSING_FROM_CATALOG'),
                 ('QTK1 abc 139.0 1\n', 'CATALOG_UNPARSEABLE'),
                 ('QTK1 95.0 139.0 1\n', 'CATALOG_INVALID_COORDINATES'),
                 ('# nothing\n', 'CATALOG_EMPTY')]
        for k, (body, cause) in enumerate(cases):
            a = self.acq(FakeSession({self.CAT: [(200, body)]}), run_day=RUN_DAY + timedelta(days=k), run_id=f'cal-{k}')
            r = a.refresh_catalog(calibration=True, declared=declared)
            self.assertEqual((r['outcome'], r['failure_class'], r['cause']), ('FAILED_KEPT_PREVIOUS', 'VALIDATION', cause), cause)
            self.assertEqual(cat.read_bytes(), before, cause)

    def test_scoring_checks_that_the_catalog_is_the_one_the_baseline_bound(self):
        TT.write_cache(self.cache)
        cid = P.code_identity()
        entry = P.calibrate_region('tokyo_kanto', date(2026, 10, 30), self.cache, code_id=cid)
        P.write_baselines(self.root / 'b', date(2026, 10, 30), {'tokyo_kanto': entry}, cid, '2026-10-30T12:00:00Z')
        ok = RF.catalog_bound(self.root / 'b', date(2026, 11, 4), self.cache, ['tokyo_kanto'])
        self.assertEqual(ok['regions']['tokyo_kanto']['status'], 'CATALOG_BOUND')
        cat = self.cache / 'ngl_catalog.json'
        cat.write_text(cat.read_text(encoding='utf-8').replace('}', ', "QZZZ": [35.5, -220.5]}'), encoding='utf-8')
        moved = RF.catalog_bound(self.root / 'b', date(2026, 11, 4), self.cache, ['tokyo_kanto'])
        self.assertEqual(moved['regions']['tokyo_kanto']['status'], 'CATALOG_CHANGED_SINCE_CALIBRATION')
        none = RF.catalog_bound(self.root / 'b', date(2026, 10, 1), self.cache, ['tokyo_kanto'])
        self.assertEqual(none['regions']['tokyo_kanto'], dict(status='UNAVAILABLE', reason='NO_BASELINE_AS_OF_SCORED_DAY'))


class CalibrationPlan(Base):
    def baseline(self, on, generated=None, **cache_kw):
        cache = self.root / ('c-%s' % on.isoformat())
        TT.write_cache(cache, **cache_kw)
        cid = P.code_identity()
        entry = P.calibrate_region('tokyo_kanto', on, cache, code_id=cid)
        return P.write_baselines(self.root / 'b', on, {'tokyo_kanto': entry}, cid,
                                 generated or on.isoformat() + 'T12:00:00Z'), entry

    def test_freshness_comes_from_the_actual_last_accepted_epoch(self):
        _, entry = self.baseline(date(2026, 10, 30))
        last = date.fromisoformat(entry['last_epoch'])
        self.assertEqual(last, date(2026, 9, 25))                               # window end 09-30 minus margin 5
        for day, action in ((date(2026, 11, 3), 'OK'),
                            (date(2026, 11, 7), 'OK'),                      # 3 days left > warn_days 2
                            (date(2026, 11, 8), 'RECALIBRATE_BY 2026-11-09'),
                            (date(2026, 11, 9), 'RECALIBRATE_BY 2026-11-09'),
                            (date(2026, 11, 10), 'STALE')):
            plan = RF.calibration_plan(self.root / 'b', day, ['tokyo_kanto'])
            row = plan['regions']['tokyo_kanto']
            self.assertEqual(row['action'], action, day)
            self.assertEqual(row['age_days'], (day - last).days)
            self.assertEqual(row['stale_on'], '2026-11-10')

    def test_a_lagging_station_brings_the_stale_day_forward(self):
        _, entry = self.baseline(date(2026, 10, 30), last={'QTK3': date(2026, 9, 26)})
        self.assertEqual(entry['last_epoch'], '2026-09-21')                     # QTK3's end minus margin 5
        row = RF.calibration_plan(self.root / 'b', date(2026, 11, 3), ['tokyo_kanto'])['regions']['tokyo_kanto']
        self.assertEqual(row['stale_on'], '2026-11-06')                          # a nominal 09-25 would say 11-10
        self.assertEqual(row['action'], 'OK')
        self.assertEqual(row['days_until_stale'], 3)

    def test_retrospective_and_missing_files_are_unavailable_by_name(self):
        self.assertEqual(RF.calibration_plan(self.root / 'b', date(2026, 11, 3), ['tokyo_kanto'])['regions'],
                         {'tokyo_kanto': dict(action='UNAVAILABLE', reason='NO_BASELINE_AS_OF_SCORED_DAY')})
        self.baseline(date(2026, 10, 30), generated='2026-11-09T02:00:00Z')
        plan = RF.calibration_plan(self.root / 'b', date(2026, 11, 3), ['tokyo_kanto'])
        self.assertTrue(plan['file']['retrospective'])
        self.assertEqual(plan['regions']['tokyo_kanto'], dict(action='UNAVAILABLE', reason='RETROSPECTIVE_BASELINE_FILE'))


if __name__ == '__main__':
    unittest.main()
