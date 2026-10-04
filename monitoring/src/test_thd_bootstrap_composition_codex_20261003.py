"""Regression locks for the two remaining bootstrap composition boundaries. Offline only."""
import json
import pickle
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from obspy import Stream, Trace, UTCDateTime
import thd_bootstrap as tb
import test_thd_bootstrap_grassmann_20261003 as fx
import station_baselines as sb


class Support(unittest.TestCase):
    def q(self, observation):
        return tb.qualify(observation, fx.WIN, today=fx.TODAY, station=fx.STA, expected_location=fx.LOC)

    def test_complete_control(self):
        self.assertEqual(self.q(fx.obs('2026-07-01')), [])

    def test_short_support_cannot_hide_behind_empty_missing_list(self):
        self.assertIn('SUPPORT_INCOMPLETE', self.q(fx.obs('2026-07-01', hours=12)))

    def test_operator_bound(self):
        self.assertIn('OPERATOR_MISMATCH', self.q(replace(fx.obs('2026-07-01'), operator='daily_ensemble')))

    def test_support_hash_required(self):
        self.assertIn('SUPPORT_HASH_MISSING_OR_INVALID', self.q(replace(fx.obs('2026-07-01'), support_sha256=None)))

    def test_last_day_ineligible_even_when_no_data_and_not_requested(self):
        obs = replace(fx.obs('2026-09-03'), npts=0, start_utc='', end_utc='', support_sha256=None)
        reasons = self.q(obs)
        self.assertIn('DAY_TOO_RECENT', reasons)
        spec = tb.acquisition_spec([obs], {obs.day: reasons}, 40)
        self.assertEqual(spec['strict_full_windows']['n_intervals'], 0)
        self.assertEqual(spec['excluded_ineligible_days'], [{'day': obs.day, 'reasons': ['DAY_TOO_RECENT']}])

    def test_complete_window_must_fit_epoch(self):
        reasons = tb.qualify(fx.obs('2026-07-01'), fx.WIN, today=fx.TODAY, station=fx.STA,
                             expected_location=fx.LOC, epoch=(None, __import__('datetime').date(2026, 7, 1)))
        self.assertIn('EPOCH_MISMATCH', reasons)

    def stitch(self, mutation=None):
        with tempfile.TemporaryDirectory(prefix='codex-stitch-test-') as tmp:
            for i, day in enumerate(('2026-06-30', '2026-07-01')):
                tr = Trace(np.arange(86400, dtype=np.int32))
                tr.stats.starttime, tr.stats.sampling_rate = UTCDateTime(day)+7*3600, 1.0
                tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel = 'IU', 'SNZO', '00', 'BHZ'
                if i == 1 and mutation:
                    setattr(tr.stats, mutation[0], mutation[1])
                folder = Path(tmp)/day.replace('-', ''); folder.mkdir()
                with (folder/'x_waveforms.pkl').open('wb') as handle:
                    pickle.dump({'IU.SNZO': Stream([tr])}, handle)
            return tb.stitch_cached_window(tmp, '2026-07-01', fx.STA, fx.LOC, analyzer=fx._FakeAnalyzer())

    def test_real_trace_nominal(self):
        self.assertEqual(self.q(self.stitch()), [])

    def test_actual_trace_station_not_cache_key(self):
        with self.assertRaisesRegex(tb.BootstrapRefused, 'NSLC_MISMATCH'):
            self.stitch(('station', 'TUC'))

    def test_later_piece_channel_bound(self):
        with self.assertRaisesRegex(tb.BootstrapRefused, 'CHANNEL_MISMATCH'):
            self.stitch(('channel', 'HHZ'))

    def test_later_piece_rate_bound(self):
        with self.assertRaisesRegex(tb.BootstrapRefused, 'RATE_MISMATCH'):
            self.stitch(('sampling_rate', 2.0))


class BaseSelection(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='codex-base-test-')
        self.root = Path(self.temp.name); self.bdir = self.root/'base'; self.bdir.mkdir()
        self.base = {'IU.TUC': dict(station='IU.TUC', mean_thd=.37, std_thd=.11, n_samples=91)}
        self.write('20261001', self.base)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, day, data):
        (self.bdir/f'thd_baselines_{day}.json').write_text(json.dumps(data), encoding='utf8')

    def test_malformed_newest_same_selection_as_loader(self):
        self.write('20261002', {'IU.TUC': dict(self.base['IU.TUC'], mean_thd='invalid')})
        snapshot = tb.snapshot_effective_base(self.bdir)
        saved = dict(sb.STATION_BASELINES)
        try:
            selected = sb._load_newest_baseline_file(self.bdir)
        finally:
            sb.STATION_BASELINES.clear(); sb.STATION_BASELINES.update(saved)
        self.assertEqual(snapshot['name'], selected)
        self.assertEqual(selected, 'thd_baselines_20261001.json')

    def test_newer_effective_file_refuses_stale_candidate(self):
        snapshot = tb.snapshot_effective_base(self.bdir)
        self.write('20261002', {'IU.TUC': dict(self.base['IU.TUC'], mean_thd=.42)})
        result = fx.run([fx.obs(d, thd=.30+.01*(i%7)) for i,d in enumerate(fx.days(80))])
        with self.assertRaisesRegex(tb.BootstrapRefused, 'BASE_SNAPSHOT_STALE'):
            tb.compose_candidate_file(result, snapshot, str(self.root/'CANDIDATE.json'))


if __name__ == '__main__':
    unittest.main()
