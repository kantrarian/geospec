"""Grey reconstruction overlay: real Oct 7 records plus declared negative controls; offline only."""
import copy
import json
from pathlib import Path
import re
import subprocess
import unittest

from monitoring.src import raw_history_cayley as raw

ROOT = Path(__file__).resolve().parents[1]
BASE = "290d9163"
REPORT = "docs/ensemble/2026-10-07/20261009T114617132021Z-7adb6101.json"


def git_bytes(path):
    return subprocess.check_output(["git", "-C", str(ROOT), "show", BASE + ":" + path])


def real_day(region="hualien"):
    body = json.loads(git_bytes("docs/raw_history/" + region + ".json"))
    return next(d for d in body["days"] if d["date"] == "2026-10-07")


class ReconstructionTests(unittest.TestCase):
    def test_oct7_retained_values_not_placeholder_zero_and_no_mutation(self):
        for region, value in (("hualien", 0.04718178923375134), ("turkey_kahramanmaras", 0.07060029198077417)):
            day = real_day(region)
            before = copy.deepcopy(day)
            point = raw.retrospective_combined(day)
            self.assertAlmostEqual(point["value"], value)
            self.assertEqual(day, before)
            self.assertEqual(day["combined"]["recorded"]["combined_risk"], 0)
            self.assertEqual(day["combined"]["recorded"]["tier_name"], "DEGRADED")
            self.assertEqual(point["components"]["seismic_thd"]["admission"]["code"], "CALIBRATED_AFTER_SCORED_DAY")

    def test_weighted_mean_is_named_display_rule_not_a_calibration(self):
        day = real_day()
        thd = day["methods"]["seismic_thd"]
        lg = copy.deepcopy(thd)
        lg["score"]["value"] = 0.2
        lg["raw"]["value"] = 1.2
        day["methods"]["lambda_geo"] = lg
        point = raw.retrospective_combined(day)
        self.assertAlmostEqual(point["value"], (0.2 * 0.4 + thd["score"]["value"] * 0.3) / 0.7)
        self.assertAlmostEqual(sum(c["effective_weight"] for c in point["components"].values()), 1)
        self.assertEqual(point["components"]["seismic_thd"]["calibration"], thd["calibration"])

    def test_unavailable_unconfirmed_frozen_withheld_and_invalid_are_not_fabricated(self):
        for field, key, value in (("raw", "state", "UNAVAILABLE_ZERO_PLACEHOLDER"),
                                  ("raw", "state", "UNCONFIRMED_NONZERO"),
                                  ("raw", "value", None), ("raw", "value", True),
                                  ("score", "state", "WITHHELD"), ("score", "value", None),
                                  ("score", "value", float("nan")), ("score", "value", True),
                                  ("score", "value", 1.1), ("score", "value", -0.1),
                                  ("recorded", "frozen", True)):
            with self.subTest(field=field, key=key, value=value):
                day = real_day()
                day["methods"]["seismic_thd"][field][key] = value
                self.assertIsNone(raw.retrospective_combined(day))

    def test_genuine_measured_zero_is_preserved(self):
        day = real_day()
        day["methods"]["seismic_thd"]["raw"]["value"] = 0
        day["methods"]["seismic_thd"]["score"]["value"] = 0
        self.assertEqual(raw.retrospective_combined(day)["value"], 0)

    def test_original_scored_and_csv_only_days_not_overwritten(self):
        for state in ("SCORED", "CSV_ONLY_SCORED", "CSV_ONLY_NO_METHOD", "REGION_NOT_IN_REPORT"):
            day = real_day()
            day["combined"]["state"] = state
            self.assertIsNone(raw.retrospective_combined(day))


class PageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page = (ROOT / "docs/index.html").read_text(encoding="utf-8-sig")
        cls.code = "\n".join(re.search(r"^        function " + name + r"\(.*?^        \}",
                                       cls.page, re.M | re.S).group() for name in ("historyValue", "retrospectiveTrace"))

    def run_trace(self, change=None):
        point = raw.retrospective_combined(real_day())
        row = dict(point["csv"], date=point["date"], region="hualien")
        block = {"rule": raw.RETRO_RULE, "regions": {"hualien": [point]}}
        if change:
            change(row, point, block)
        script = self.code + "\nconsole.log(JSON.stringify(retrospectiveTrace(" + json.dumps([row]) + "," + json.dumps(block) + ")));"
        result = subprocess.run(["node", "-"], input=script, text=True, encoding="utf-8", capture_output=True, check=True)
        return json.loads(result.stdout)

    def test_grey_disconnected_marker_with_formula_inputs_reason_and_source(self):
        trace = self.run_trace()
        self.assertEqual(trace["x"], ["2026-10-07"])
        self.assertAlmostEqual(trace["y"][0], 0.04718178923375134)
        self.assertEqual(trace["marker"]["color"], "#9ca3af")
        self.assertEqual(trace["marker"]["symbol"], "diamond")
        self.assertEqual(trace["mode"], "markers")
        self.assertFalse(trace["connectgaps"])
        for text in ("CALIBRATED_AFTER_SCORED_DAY", "RETROSPECTIVE", "weight 1.0000", "raw 0.423267", REPORT):
            self.assertIn(text, trace["text"][0])

    def test_mismatched_rows_new_qualified_scores_and_unknown_rule_do_not_overlay(self):
        for change in (lambda r, p, b: r.update(risk="0.1234"),
                       lambda r, p, b: r.update(methods="1"),
                       lambda r, p, b: r.update(date="2026-10-08"),
                       lambda r, p, b: b.update(rule="other"),
                       lambda r, p, b: p.update(value=None)):
            self.assertEqual(self.run_trace(change)["x"], [])

    def test_tooltip_escapes_retained_text(self):
        trace = self.run_trace(lambda r, p, b: p["components"]["seismic_thd"]["admission"].update(reason="<img src=x>"))
        self.assertIn("&lt;img src=x&gt;", trace["text"][0])
        self.assertNotIn("<img", trace["text"][0])

    def test_toggle_and_failure_message_and_drop_inputs_stay_separate(self):
        self.assertIn('id="show-retrospective" type="checkbox" checked', self.page)
        self.assertIn("Plotly.restyle('history-chart', { visible: toggle.checked }", self.page)
        self.assertIn("Retrospective overlay unavailable:", self.page)
        self.assertIn("const drops = detectStressReleaseDrops(csvData)", self.page)
        script = re.search(r"<script>(.*?)</script>", self.page, re.S).group(1)
        result = subprocess.run(["node", "--check"], input=script, text=True, encoding="utf-8", capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_actual_chart_adds_toggleable_grey_trace_without_replacing_original(self):
        result = self.render_chart(False)
        self.assertEqual(result["traces"][0]["y"], [None])
        self.assertAlmostEqual(result["traces"][1]["y"][0], 0.04718178923375134)
        self.assertEqual(result["traces"][1]["marker"]["color"], "#9ca3af")
        self.assertEqual(result["restyle"], ["history-chart", {"visible": False}, [1]])
        self.assertIn("1 grey reconstructed region-days", result["status"])

    def test_unavailable_reconstruction_artifact_does_not_break_original_chart(self):
        result = self.render_chart(True)
        self.assertEqual(len(result["traces"]), 3)  # original region and two threshold lines
        self.assertEqual(result["traces"][0]["y"], [None])
        self.assertIn("Original history is unchanged", result["status"])

    def render_chart(self, failure):
        point = raw.retrospective_combined(real_day())
        row = dict(point["csv"], date=point["date"], region="hualien")
        block = {"rule": raw.RETRO_RULE, "regions": {"hualien": [point]}}
        render = re.search(r"^        async function renderChart\(.*?^        \}", self.page, re.M | re.S).group()
        harness = """
const status = {textContent: ''}, toggle = {checked: true};
const document = {getElementById: id => id === 'retrospective-status' ? status : toggle};
const result = {};
const Plotly = {newPlot: (id, traces) => {result.traces = traces;}, restyle: (...args) => {result.restyle = args;}};
const getDropOverlays = () => ({shapes: [], annotations: []});
const fetch = async () => { if (FAILURE) throw new Error('offline');
  return {ok:true, json: async () => ({retrospective_combined: BLOCK})}; };
(async () => {await renderChart(ROWS); toggle.checked = false;
  if (toggle.onchange) toggle.onchange(); result.status = status.textContent;
  console.log(JSON.stringify(result));})();
""".replace("FAILURE", json.dumps(failure)).replace("BLOCK", json.dumps(block)).replace("ROWS", json.dumps([row]))
        result = subprocess.run(["node", "-"], input=self.code + render + harness, text=True,
                                encoding="utf-8", capture_output=True, check=True)
        return json.loads(result.stdout)


if __name__ == "__main__":
    unittest.main()
