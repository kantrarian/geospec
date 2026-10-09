"""thd-baseline-as-of-v1 (cayley 2026-10-09; grassmann 8af63873, run da191bd4). Needs obspy (1.5.1 pinned env).

On 2026-10-09 the weekly recal wrote thd_baselines_20261009 before the ensemble scored 2026-10-07; the import-time
loader selected that newest file, and calibration-eligibility-v3 correctly refused it (CALIBRATED_AFTER_SCORED_DAY) in
every THD region. The baseline in force for a scored day is now the newest dated file ON OR BEFORE that day.

The baseline files are written by the RECAL'S OWN WRITER (run_thd_recal.run_recal with calibrate_station replaced by the
moments grassmann read on the host for 20261001 / 20261009); the 20261001 file then has `calibration_date` removed, as
the host file has (written before the 2026-10-04 writer added it). Nothing touches a network or the runner tree.
"""
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import calibrate_thd_baselines as C  # noqa: E402
import calibration_eligibility as CE  # noqa: E402
import ensemble as E  # noqa: E402
import run_thd_recal as R  # noqa: E402
import station_baselines as SB  # noqa: E402

# grassmann b9c6ca5a / 8af63873, as read on the host (BK.BKS and IU.TUC; IU.COLA only in 20261001 for the gap case)
MOMENTS = {
    "20261001": {"BK.BKS": (0.372086, 0.136103, 91), "IU.TUC": (0.372277, 0.11271, 91), "IU.COLA": (0.284404, 0.179171, 91)},
    "20261009": {"BK.BKS": (0.364152, 0.135428, 91), "IU.TUC": (0.351704, 0.104826, 91)},
}
PERIOD = {"20261001": "2026-06-03 to 2026-09-01", "20261009": "2026-06-11 to 2026-09-09"}


def write_recal(bdir, stamp, legacy=False):
    """One dated file from run_recal itself; `legacy` drops calibration_date like the pre-2026-10-04 writer."""
    def calibrate(network, station, **kw):
        mean, std, n = MOMENTS[stamp]["%s.%s" % (network, station)]
        return {"mean_thd": mean, "std_thd": std, "n_samples": n, "calibration_period": PERIOD[stamp]}
    with mock.patch.object(R, "BASELINE_DIR", bdir), mock.patch.object(C, "calibrate_station", side_effect=calibrate), \
            mock.patch.object(R, "_prior_effective_entries", return_value={}):
        path = Path(R.run_recal(sorted(MOMENTS[stamp]), end_date=datetime.strptime(stamp, "%Y%m%d")))
    if legacy:
        data = json.loads(path.read_text())
        for entry in data.values():
            if isinstance(entry, dict):
                entry.pop("calibration_date", None)
        path.write_text(json.dumps(data, indent=2))
    return path


class AsOfSelection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="thd-as-of-")
        self.addCleanup(self.tmp.cleanup)
        self.bdir = Path(self.tmp.name)
        self.old = write_recal(self.bdir, "20261001", legacy=True)
        self.new = write_recal(self.bdir, "20261009")

    def at(self, station, day):
        net, sta = station.split(".")
        return SB.baseline_as_of(sta, net, day, bdir=self.bdir)

    def test_the_writer_files_are_what_the_host_holds(self):
        self.assertEqual([p.name for p in sorted(self.bdir.glob("thd_baselines_*.json"))],
                         ["thd_baselines_20261001.json", "thd_baselines_20261009.json"])
        old = json.loads(self.old.read_text())
        self.assertNotIn("calibration_date", old["BK.BKS"])
        self.assertEqual(json.loads(self.new.read_text())["BK.BKS"]["calibration_date"], "2026-10-09")

    def test_each_scored_day_gets_the_newest_file_on_or_before_it(self):
        expect = {"2026-09-30": None, "2026-10-01": "20261001", "2026-10-07": "20261001", "2026-10-08": "20261001",
                  "2026-10-09": "20261009", "2026-10-20": "20261009"}
        for day, stamp in expect.items():
            with self.subTest(day=day):
                got = self.at("BK.BKS", day)
                if stamp is None:
                    self.assertIs(got, SB._BUILTIN_BASELINES["BK.BKS"])
                else:
                    self.assertEqual((got.mean_thd, got.calibration_period), (MOMENTS[stamp]["BK.BKS"][0], PERIOD[stamp]))

    def test_the_as_of_baseline_qualifies_where_the_newest_one_is_refused(self):
        lag, age = R.EXCLUDE_RECENT_DAYS, E.MAX_BASELINE_AGE_DAYS
        for day in ("2026-10-07", "2026-10-08"):
            with self.subTest(day=day):
                newest = SB._file_baselines(self.new)["BK.BKS"]
                refused = CE.classify_thd_baseline(newest, day, max_age_days=age, min_lag_days=lag)
                self.assertEqual((refused.eligible_for_tiering, refused.code), (False, CE.CALIBRATED_AFTER_SCORED_DAY))
                in_force = CE.classify_thd_baseline(self.at("BK.BKS", day), day, max_age_days=age, min_lag_days=lag)
                self.assertTrue(in_force.eligible_for_tiering, in_force.reason)
        on_recal = CE.classify_thd_baseline(self.at("BK.BKS", "2026-10-09"), "2026-10-09", max_age_days=age,
                                            min_lag_days=lag)
        self.assertTrue(on_recal.eligible_for_tiering, on_recal.reason)

    def test_a_station_the_selected_file_lacks_keeps_its_builtin_default(self):
        self.assertIs(self.at("IU.COLA", "2026-10-09"), SB._BUILTIN_BASELINES["IU.COLA"])
        self.assertEqual(self.at("IU.COLA", "2026-10-07").mean_thd, MOMENTS["20261001"]["IU.COLA"][0])

    def test_unreadable_or_undated_files_are_skipped_never_guessed(self):
        (self.bdir / "thd_baselines_20261005.json").write_text("{not json")
        (self.bdir / "thd_baselines_latest.json").write_text(self.new.read_text())
        self.assertEqual(self.at("BK.BKS", "2026-10-07").mean_thd, MOMENTS["20261001"]["BK.BKS"][0])
        for bad in ("2026-13-01", "10/07/2026", None, 20261007):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.at("BK.BKS", bad)

    def test_the_import_time_selection_is_replaced_only_when_dated_after_the_scored_day(self):
        with mock.patch.dict(SB.STATION_BASELINES), mock.patch.object(SB, "BASELINE_DIR", self.bdir):
            self.assertEqual(SB._load_newest_baseline_file(), "thd_baselines_20261009.json")   # the loader's own pick
            selected = SB.STATION_BASELINES["BK.BKS"]
            self.assertIs(SB.get_baseline("BKS", "BK"), selected)
            self.assertIs(SB.get_baseline("BKS", "BK", as_of="2026-10-09"), selected)
            self.assertIs(SB.get_baseline("BKS", "BK", as_of="2026-10-20"), selected)
            for day in ("2026-10-07", "2026-10-08"):
                with self.subTest(day=day):
                    self.assertEqual(SB.get_baseline("BKS", "BK", as_of=day).mean_thd, MOMENTS["20261001"]["BK.BKS"][0])
            injected = SB.StationBaseline(station="BK.BKS", mean_thd=0.5, std_thd=0.1, n_samples=60,
                                          calibration_period="2026-06-01 to 2026-08-30", calibration_date="2026-09-30")
            SB.STATION_BASELINES["BK.BKS"] = injected
            self.assertIs(SB.get_baseline("BKS", "BK", as_of="2026-10-07"), injected, "not later than the day: kept")
            default = SB._BUILTIN_BASELINES["BK.BKS"]
            SB.STATION_BASELINES["BK.BKS"] = default
            self.assertIs(SB.get_baseline("BKS", "BK", as_of="2026-10-07"), default, "undated default: kept")


class TheDailyPathAsksForTheScoredDay(unittest.TestCase):
    def test_compute_thd_risk_scores_against_the_baseline_in_force_for_its_day(self):
        from obspy import Stream
        from test_thd_daily_measurement_cayley_20261008 import DAY, TrimmingClient, waveform
        with tempfile.TemporaryDirectory(prefix="thd-as-of-daily-") as tmp:
            bdir = Path(tmp)
            for stamp, period, mean in (("20260726", "2026-04-27 to 2026-06-26", 0.31),
                                        ("20260803", "2026-05-05 to 2026-07-04", 0.42)):
                (bdir / ("thd_baselines_%s.json" % stamp)).write_text(json.dumps({"IU.TUC": {
                    "station": "IU.TUC", "mean_thd": mean, "std_thd": 0.1, "n_samples": 60, "calibration_period": period,
                    "calibration_date": "%s-%s-%s" % (stamp[:4], stamp[4:6], stamp[6:])}}))
            TrimmingClient.STREAM, TrimmingClient.asked = Stream([waveform("IU", "TUC", "00")]), []
            with mock.patch("obspy.clients.fdsn.Client", TrimmingClient), mock.patch.object(SB, "BASELINE_DIR", bdir), \
                    mock.patch.dict(SB.STATION_BASELINES):
                self.assertEqual(SB._load_newest_baseline_file(), "thd_baselines_20260803.json")   # newest-first pick
                result = E.GeoSpecEnsemble(region="ridgecrest", eligibility_rule_active=False).compute_thd_risk(
                    DAY, station_network="IU", station_code="TUC")
        self.assertEqual(DAY.strftime("%Y-%m-%d"), "2026-08-01")
        self.assertEqual((result.baseline_window, result.baseline_mean), ("2026-04-27 to 2026-06-26", 0.31),
                         "the 2026-08-03 recal is after the scored day and is never used for it")


if __name__ == "__main__":
    unittest.main()
