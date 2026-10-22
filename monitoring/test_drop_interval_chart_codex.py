"""Offline chart acceptance: real Turkey clocks, reconstruction links, and false-match controls."""
import csv
import io
import json
from pathlib import Path
import re
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
PAGE = (ROOT / "docs/index.html").read_text(encoding="utf-8-sig")
NAMES = ("historyValue", "retrospectiveTrace", "retrospectiveConnectors", "detectStressReleaseDrops", "getDropOverlays")
CODE = "\n".join(re.search(r"^        function " + name + r"\(.*?^        \}", PAGE, re.M | re.S).group() for name in NAMES)


def js(expression):
    proc = subprocess.run(["node", "-"], input=CODE + "\nconsole.log(JSON.stringify(" + expression + "));",
                          text=True, encoding="utf-8", capture_output=True, check=True)
    return json.loads(proc.stdout)


def rows(values, region="test", start=1):
    return [dict(region=region, date=f"2026-10-{i:02}", risk=str(v), methods="1", tier="1" if v >= .25 else "0")
            for i, v in enumerate(values, start)]


class DeclineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = subprocess.check_output(["git", "-C", str(ROOT), "show", "0eca6fab:docs/data.csv"], text=True)
        cls.actual = [r for r in csv.DictReader(io.StringIO(data)) if r["region"] == "turkey_kahramanmaras"
                      and "2026-09-28" <= r["date"] <= "2026-10-09"]

    def actual_drop(self):
        return js("detectStressReleaseDrops(" + json.dumps(self.actual) + ")")[0]

    def test_real_turkey_four_clocks(self):
        d = self.actual_drop()
        self.assertEqual((d["peakDate"], d["firstDeclineDate"], d["dropDate"], d["lowDate"], d["lowConfirmedOn"]),
                         ("2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05", "2026-10-06"))
        self.assertEqual([d["peakRisk"], d["currRisk"], d["lowRisk"]], [.8115, .011, .0082])
        self.assertEqual([p["date"] for p in d["decline"]], ["2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05"])

    def test_peak_is_not_backdated_detection_and_future_low_is_separate(self):
        before = [r for r in self.actual if r["date"] <= "2026-10-03"]
        self.assertEqual(js("detectStressReleaseDrops(" + json.dumps(before) + ")"), [])
        at = [r for r in self.actual if r["date"] <= "2026-10-04"]
        d = js("detectStressReleaseDrops(" + json.dumps(at) + ")")[0]
        self.assertEqual(d["dropDate"], "2026-10-04")
        self.assertEqual(d["lowDate"], "2026-10-04")
        self.assertIsNone(d["lowConfirmedOn"])

    def test_render_has_peak_to_low_line_same_day_event_and_unmoved_window(self):
        d = self.actual_drop()
        events = [dict(region=d["region"], time="2026-10-04T11:43:23Z", mag=5)]
        result = js("getDropOverlays(" + json.dumps([d]) + ", {}, " + json.dumps(events) + ")")
        self.assertEqual((result["shapes"][0]["x0"], result["shapes"][0]["x1"]), ("2026-10-02", "2026-10-05"))
        self.assertEqual((result["shapes"][1]["x0"], result["shapes"][1]["x1"]), ("2026-10-04", "2026-10-07"))
        text = result["traces"][0]["text"][0]
        self.assertIn("within peak-to-low scored-day interval", text)
        self.assertIn("intraday order unknown", text)
        self.assertIn("NOT a backdated detection", text)
        self.assertEqual(result["traces"][0]["y"], [.8115, .612, .011, .0082])

    def test_unrelated_region_and_event_before_peak_do_not_match(self):
        d = self.actual_drop()
        events = [dict(region="other", time="2026-10-04", mag=9),
                  dict(region=d["region"], time="2026-10-01", mag=8)]
        result = js("getDropOverlays(" + json.dumps([d]) + ", {}, " + json.dumps(events) + ")")
        self.assertIn("No retained same-region", result["traces"][0]["text"][0])
        self.assertFalse(any(a["text"] == "\u25cf" for a in result["annotations"]))

    def test_same_date_regions_are_not_collapsed(self):
        a = self.actual_drop()
        b = dict(a, region="other")
        result = js("getDropOverlays(" + json.dumps([a, b]) + ", {}, [])")
        self.assertEqual(len(result["traces"]), 2)

    def test_missing_or_unavailable_day_breaks_consecutive_run(self):
        base = rows([.4, .5, .6, .7, .6, .05])
        gap = [r for r in base if r["date"] != "2026-10-04"]
        self.assertEqual(js("detectStressReleaseDrops(" + json.dumps(gap) + ")"), [])
        base[3]["methods"] = "0"
        self.assertEqual(js("detectStressReleaseDrops(" + json.dumps(base) + ")"), [])

    def test_plateau_uses_last_peak_then_first_decrease(self):
        data = rows([.4, .6, .8, .8, .7, .1, .08, .09])
        d = js("detectStressReleaseDrops(" + json.dumps(data) + ")")[0]
        self.assertEqual((d["peakDate"], d["firstDeclineDate"], d["lowDate"]), ("2026-10-04", "2026-10-05", "2026-10-07"))

    def test_future_gap_does_not_create_confirmed_low(self):
        data = rows([.4, .6, .8, .7, .6, .1, .05, .3])
        data[6]["methods"] = "0"
        d = js("detectStressReleaseDrops(" + json.dumps(data) + ")")[0]
        self.assertEqual(d["lowDate"], "2026-10-06")
        self.assertIsNone(d["lowConfirmedOn"])


class GreyLinksTests(unittest.TestCase):
    def connect(self, data, retro):
        return js("retrospectiveConnectors(" + json.dumps(data) + "," + json.dumps(retro) + ")")

    def test_adjacent_gap_has_two_grey_dashed_segments(self):
        data = rows([.4, 0, .2], start=6)
        data[1]["methods"] = "0"
        retro = dict(x=["2026-10-07"], y=[.07], customdata=["test"])
        result = self.connect(data, retro)
        self.assertEqual(result["x"], ["2026-10-06", "2026-10-07", None, "2026-10-07", "2026-10-08", None])
        self.assertEqual(result["y"], [.4, .07, None, .07, .2, None])
        self.assertEqual(result["line"], dict(color="#9ca3af", width=2, dash="dash"))
        self.assertIn("NOT an issued continuous series", result["text"][0])

    def test_never_bridge_date_gap_missing_value_or_different_region(self):
        retro = dict(x=["2026-10-07"], y=[0], customdata=["test"])
        data = rows([.4, 0, .2], start=6)
        data[0]["date"] = "2026-10-05"
        data[2]["methods"] = "0"
        self.assertEqual(self.connect(data, retro)["x"], [])
        data[0]["date"] = "2026-10-06"
        data[0]["region"] = "other"
        self.assertEqual(self.connect(data, retro)["x"], [])

    def test_no_reconstruction_is_no_connector(self):
        self.assertEqual(self.connect(rows([.4, .5]), dict(x=[], y=[], customdata=[]))["x"], [])

    def test_input_rows_are_not_mutated(self):
        data = rows([.4, .6, .8, .7, .6, .1])
        expr = "(() => {const rows=" + json.dumps(data) + "; const before=JSON.stringify(rows); detectStressReleaseDrops(rows); return before===JSON.stringify(rows);})()"
        self.assertTrue(js(expr))


if __name__ == "__main__":
    unittest.main()
