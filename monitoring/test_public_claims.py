"""Offline claim-surface regression tests; never run the daily publisher.

Run: python -m unittest discover -s monitoring -p test_public_claims.py -v
PowerShell exercises the extracted real README template, Node exercises the
actual UI functions. No monitoring data, network service or Git ref is written.
These are presentation/implementation tests, not scientific validation.
"""

import ast
import base64
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
PAGES = (ROOT / "docs/index.html", ROOT / "monitoring/dashboard/index.html")
INCIDENT = "INCIDENT_2026-09-17_shared_support_claims.md"
STATUS = "Research status: INCONCLUSIVE. Not an earthquake warning or forecasting service."


def read(path):
    return path.read_text(encoding="utf-8-sig")


def js_function(page, name):
    match = re.search(r"^        function " + name + r"\(.*?^        \}",
                      page, flags=re.MULTILINE | re.DOTALL)
    if not match:
        raise AssertionError(f"Missing function: {name}")
    return match.group(0)


def node_eval(script):
    node = shutil.which("node")
    if not node:
        raise AssertionError("Node is required for UI regression tests")
    result = subprocess.run([node, "-"], input=script, check=True, capture_output=True,
                            text=True, encoding="utf-8", timeout=30)
    return json.loads(result.stdout)


class PublicClaimsTests(unittest.TestCase):
    def test_status_precedes_metrics_on_all_entry_points(self):
        readme = read(ROOT / "README.md")
        self.assertLess(readme.index(STATUS), readme.index("## Current Status"))
        for path in PAGES:
            with self.subTest(path=path):
                page = read(path)
                self.assertLess(page.index(STATUS), page.index('id="status-grid"'))
                self.assertIn("not calibrated earthquake probabilities", page)
                self.assertIn("not independent confirmations", page)

    def test_correction_links_resolve_from_each_entry_point(self):
        for path in (ROOT / "README.md", *PAGES):
            with self.subTest(path=path):
                links = re.findall(r'["(]([^"()\s]*' + re.escape(INCIDENT) + r')[")]', read(path))
                self.assertTrue(links)
                for link in links:
                    self.assertEqual((path.parent / link).resolve(),
                                     ROOT / "docs" / INCIDENT)
                    self.assertTrue((path.parent / link).is_file())

    def test_actual_powershell_readme_template_retains_status_on_regeneration(self):
        source = read(ROOT / "run_and_publish.ps1")
        match = re.search(r'^\$ReadmeContent = @"\n.*?^"@', source,
                          flags=re.MULTILINE | re.DOTALL)
        self.assertIsNotNone(match)
        # Execute only the exact template assignment, never the publisher.
        command = """
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$AssessmentDate = '2099-01-02'
$MaxRegion = 'TEST_REGION'
$MaxRisk = 0.433
$TierCounts = @{'0'=12; '1'=2; '2'=0; '3'=0}
$EnsembleData = @{summary=@{total_regions=14}}
""" + match.group(0) + "\n[Console]::Write($ReadmeContent)"
        shell = shutil.which("pwsh") or shutil.which("powershell")
        self.assertIsNotNone(shell, "PowerShell required to test the real README generator")
        encoded = base64.b64encode(command.encode("utf-16le")).decode("ascii")
        result = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                                check=True, capture_output=True, text=True,
                                encoding="utf-8", timeout=30)
        generated = result.stdout
        self.assertLess(generated.index(STATUS), generated.index("## Current Status"))
        self.assertIn(f"docs/{INCIDENT}", generated)
        self.assertIn("**Last Update**: 2099-01-02", generated)
        self.assertIn("| Highest Risk Region | TEST_REGION |", generated)
        self.assertIn("| Risk Score | 0.433 |", generated)
        self.assertIn("| WATCH (1) | 2 |", generated)
        banner = lambda value: "\n".join(line for line in value.splitlines() if line.startswith(">"))
        self.assertEqual(banner(generated), banner(read(ROOT / "README.md")))

    def test_calibration_is_not_labelled_out_of_sample(self):
        report = read(ROOT / "docs/CALIBRATION_BACKTEST_REPORT.md")
        self.assertIn("#### 4.2.4 Training Set Performance", report)
        self.assertIn("**Detection Rate (training):** 5/5", report)
        self.assertIn("**Validation Hit Rate:** 2/5", report)
        for path in PAGES:
            with self.subTest(path=path):
                page = read(path)
                self.assertIn("Calibration / Training Backtest (Not Out-of-Sample)", page)
                self.assertIn("5/5 calibration events", page)
                self.assertIn("separately lists 2/5", page)
                self.assertNotIn("recalibrated May 2026", page)
                self.assertNotIn("5/5 events detected", page)

    def test_strong_interpretations_are_withdrawn_from_active_pages(self):
        retired = ("move as one tectonic unit", "190 km/hr", "by 76%",
                   "all validated precursor signals", "Tidal aliasing rejected",
                   "Strain eigenframe rotation completes", "often preceding rupture")
        for path in PAGES:
            with self.subTest(path=path):
                for claim in retired:
                    self.assertNotIn(claim, read(path))

    def test_incident_states_unchanged_operations_and_unknown_independent_count(self):
        note = read(ROOT / "docs" / INCIDENT)
        self.assertIn("Source-aware detector recomputation has not been performed", note)
        self.assertIn("number of independently supported detections is unknown", note)
        self.assertIn("not a claim that every component", note)
        self.assertIn("thresholds, calibration,\ncollected data and monitoring runs are unchanged", note)
        self.assertIn("append-only incident record", note)
        for claim in ("190 km/hr", "76%", "not supported as stated"):
            self.assertIn(claim, note)

    def test_documented_shared_support_matches_public_station_map(self):
        tree = ast.parse(read(ROOT / "monitoring/src/run_ensemble_daily.py"))
        config = next(ast.literal_eval(item.value) for item in tree.body
                      if isinstance(item, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "REGIONS"
                              for target in item.targets))
        for region in ("ridgecrest", "socal_saf_mojave", "socal_saf_coachella"):
            self.assertEqual((config[region]["thd_network"], config[region]["thd_station"]), ("IU", "TUC"))
        for region in ("istanbul_marmara", "turkey_kahramanmaras"):
            self.assertEqual((config[region]["thd_network"], config[region]["thd_station"]), ("IU", "ANTO"))
        self.assertEqual((config["kumamoto"]["thd_network"], config["kumamoto"]["thd_station"]), ("IU", "MAJO"))
        self.assertEqual(config["tokyo_kanto"]["thd_network"], "HINET")
        self.assertEqual((config["tokyo_kanto"]["fallback_network"], config["tokyo_kanto"]["fallback_station"]), ("IU", "MAJO"))

    def test_drop_algorithm_unchanged_by_claim_correction(self):
        # Normalized-newline function hash at public base e207b477. Updating the
        # detector requires a separate reviewed change and explicit new test.
        function = js_function(read(PAGES[0]), "detectStressReleaseDrops")
        self.assertEqual(hashlib.sha256(function.encode("utf-8")).hexdigest(),
                         "6b795770c84b5fb664a6c1d36d4a9d3c3c31c9315ef5cc3599681108d8c60c0e")

    def test_existing_region_count_and_solo_exceptions_match_corrected_text(self):
        function = js_function(read(PAGES[0]), "detectStressReleaseDrops")
        # Synthetic implementation probes only, not a detector recomputation.
        result = node_eval(function + """
function rows(region, n, tier=1) {
    const data = Array.from({length:n}, (_, i) => ({region,
        date:`2026-01-${String(i+1).padStart(2,'0')}`, tier:String(tier), risk:'0.4'}));
    data.push({region, date:`2026-01-${String(n+1).padStart(2,'0')}`, tier:'0', risk:'0.1'});
    return data;
}
console.log(JSON.stringify([
    detectStressReleaseDrops(rows('solo-short',3)).length,
    detectStressReleaseDrops(rows('label-a',3).concat(rows('label-b',3))).length,
    detectStressReleaseDrops(rows('solo-long',5)).length,
    detectStressReleaseDrops(rows('solo-tier2',3,2)).length
]));
""")
        self.assertEqual(result, [0, 2, 1, 1])

    def test_old_json_cannot_reintroduce_causal_method_claims(self):
        methods = json.loads(read(ROOT / "docs/backtest_timeseries.json"))["method_descriptions"]
        for path in PAGES:
            with self.subTest(path=path):
                function = js_function(read(path), "renderMethodsLegend")
                script = "const container={innerHTML:''}; const document={getElementById:()=>container};\n"
                script += "console.log=()=>{};\n" + function
                script += "\nrenderMethodsLegend(" + json.dumps(methods) + ");"
                script += "\nprocess.stdout.write(JSON.stringify(container.innerHTML));"
                rendered = node_eval(script)
                self.assertNotIn("often preceding rupture", rendered)
                self.assertNotIn("indicate decorrelation suggesting stress transfer", rendered)
                self.assertIn("do not establish crustal stress accumulation", rendered)
                self.assertIn("do not by themselves establish tectonic stress transfer", rendered)
                self.assertIn(methods["LG"]["derivation"], rendered)
                self.assertIn(methods["THD"]["threshold_elevated"], rendered)

    def test_all_inline_javascript_still_parses(self):
        for path in PAGES:
            with self.subTest(path=path):
                scripts = re.findall(r"<script(?:\s[^>]*)?>(.*?)</script>", read(path), re.DOTALL)
                result = node_eval("const scripts=" + json.dumps(scripts) +
                                   "; scripts.forEach(s => new Function(s)); console.log(JSON.stringify(scripts.length));")
                self.assertGreater(result, 0)


if __name__ == "__main__":
    unittest.main()
