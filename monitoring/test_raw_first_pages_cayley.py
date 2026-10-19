"""Raw-first Pages increment (cayley 2026-10-19; codex bbe31a10, owner "agreed - proceed"): offline tests.

Run: python -m unittest discover -s monitoring -p test_raw_first_pages_cayley.py -v
Every record case starts from a REAL retained component, read by git identity (commit and path or blob); the one
case history does not contain (a genuine raw zero) is a declared single-field edit of a real record. The page tests
run the actual functions extracted from docs/index.html under Node. Nothing is fetched, published or written in the
repository; build tests skip (with the reason) when the pinned base commit is absent from this clone.
"""
import csv
import importlib.util
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("raw_history_cayley", ROOT / "monitoring/src/raw_history_cayley.py")
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)
BASE = "08d72b537af0aa6c4b88bd3c9eb041f278e14ba4"  # the commit docs/raw_history/ was written from
SEP20 = "docs/ensemble/2026-09-20/20260922T053344926589Z-63b9971e.json"
OCT07 = "docs/ensemble/2026-10-07/20261009T114617132021Z-7adb6101.json"
OCT16 = "docs/ensemble/2026-10-16/20261018T110515494888Z-4b8f597c.json"
DECLARED_PATCH_PATHS = ("docs/index.html", "docs/raw_history/", "monitoring/src/raw_history_cayley.py",
                        "monitoring/test_raw_first_pages_cayley.py")


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT)] + list(args), capture_output=True, check=True).stdout


def have(rev):
    return subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", rev + "^{commit}"], capture_output=True).returncode == 0


def report(rev, path):
    return json.loads(git("show", rev + ":" + path))


def region(rev, path, name):
    return report(rev, path)["regions"][name]


def capsule_report(date):
    capsule = report(BASE, R.CAPSULE_REL)
    record = next(r for r in capsule["records"] if r["date"] == date)
    return json.loads(git("cat-file", "blob", record["git_blob"])), record


def node_eval(script):
    node = shutil.which("node")
    if not node:
        raise AssertionError("Node is required for the page tests")
    result = subprocess.run([node, "-"], input=script, check=True, capture_output=True, text=True, encoding="utf-8", timeout=60)
    return json.loads(result.stdout)


def page_functions():
    page = (ROOT / "docs/index.html").read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    block = re.search(r"^        const RAW_METHOD_LABELS.*?(?=^        async function rawLoadRegion)", page, re.MULTILINE | re.DOTALL)
    history = re.search(r"^        function historyValue\(.*?^        \}", page, re.MULTILINE | re.DOTALL)
    if not block or not history:
        raise AssertionError("raw-first functions missing from docs/index.html")
    return block.group(0) + "\n" + history.group(0) + "\n"


@unittest.skipUnless(have(BASE), "pinned base commit absent from this clone")
class RecordTests(unittest.TestCase):
    def test_hualien_pair_measured_values_kept_and_scores_as_recorded(self):
        july = region("0c99e0ec", "docs/ensemble_latest.json", "hualien")
        sept = region(BASE, SEP20, "hualien")
        a = R.method_entry("seismic_thd", july["components"]["seismic_thd"], july)
        b = R.method_entry("seismic_thd", sept["components"]["seismic_thd"], sept)
        self.assertEqual((a["raw"]["value"], b["raw"]["value"]), (0.5233468943622342, 0.5267691230866207))
        self.assertEqual((a["score"]["value"], b["score"]["value"]), (0.5560382408066546, 0.24788143925048328))
        self.assertEqual((a["station"]["value"], b["station"]["value"]), ("IU.TATO", "IU.TATO"))
        self.assertEqual((a["admission"]["weight"], b["admission"]["weight"]), (1.0, 1.0))
        windows = (a["calibration"]["baseline"]["window"], b["calibration"]["baseline"]["window"])
        self.assertEqual(windows, ("2025-12-09 to 2026-01-09", "2026-05-18 to 2026-08-16"), "the calibration change")
        self.assertEqual(a["recorded"]["raw_value"], july["components"]["seismic_thd"]["raw_value"])

    def test_legacy_zero_placeholder_is_kept_verbatim_but_never_a_measurement(self):
        july = region("0c99e0ec", "docs/ensemble_latest.json", "hualien")
        comp = july["components"]["lambda_geo"]
        self.assertEqual((comp["available"], comp["raw_value"]), (False, 0.0))
        e = R.method_entry("lambda_geo", comp, july)
        self.assertEqual((e["raw"]["state"], e["raw"]["value"], e["score"]["state"], e["score"]["value"]),
                         ("UNAVAILABLE_ZERO_PLACEHOLDER", None, "METHOD_UNAVAILABLE", None))
        self.assertEqual((e["recorded"]["raw_value"], e["recorded"]["risk_score"]), (0.0, 0.0), "original preserved")

    def test_genuine_raw_zero_is_a_measurement(self):
        # DECLARED SYNTHETIC: history holds no available zero; one field of a real record is edited.
        july = region("0c99e0ec", "docs/ensemble_latest.json", "hualien")
        comp = dict(july["components"]["seismic_thd"], raw_value=0.0)
        e = R.method_entry("seismic_thd", comp, july)
        self.assertEqual((e["raw"]["state"], e["raw"]["value"]), ("MEASURED", 0.0))

    def test_unconfirmed_nonzero_is_kept_flagged_never_a_measurement(self):
        # DECLARED SYNTHETIC: history holds no unavailable nonzero raw field; the real record's available flag is edited.
        july = region("0c99e0ec", "docs/ensemble_latest.json", "hualien")
        for available in (False, None):
            comp = dict(july["components"]["seismic_thd"], available=available)
            e = R.method_entry("seismic_thd", comp, july)
            self.assertEqual((e["raw"]["state"], e["raw"]["value"], e["raw"]["unconfirmed_value"]),
                             ("UNCONFIRMED_NONZERO", None, 0.5233468943622342))
            self.assertEqual((e["score"]["state"], e["score"]["value"]), ("METHOD_UNAVAILABLE", None))
            self.assertIn("UNCONFIRMED_MEASUREMENT_STATUS", e["warnings"])

    def test_raw_above_one_is_not_clamped(self):
        rep, _ = capsule_report("2026-01-12")
        r = rep["regions"]["ridgecrest"]
        comp = r["components"]["lambda_geo"]
        self.assertGreater(comp["raw_value"], 1)
        e = R.method_entry("lambda_geo", comp, r)
        self.assertEqual((e["raw"]["state"], e["raw"]["value"]), ("MEASURED", comp["raw_value"]))

    def test_withheld_score_keeps_the_measured_value_and_the_recorded_basis(self):
        r = region(BASE, OCT16, "kaikoura")
        e = R.method_entry("seismic_thd", r["components"]["seismic_thd"], r)
        self.assertEqual((e["raw"]["state"], e["raw"]["value"]), ("MEASURED", 0.5262871366899935))
        self.assertEqual((e["score"]["state"], e["score"]["value"]), ("WITHHELD", None))
        self.assertEqual(e["calibration"]["score_basis"], "NOT_COMPUTED_NO_EMPIRICAL_BASELINE")
        for warning in ("CALIBRATION_N0_DEFAULT", "BASELINE_QUALITY_POOR", "ZERO_SAMPLE_BASELINE"):
            self.assertIn(warning, e["warnings"])

    def test_qc_flagged_value_carries_a_warning(self):
        rep, _ = capsule_report("2026-07-30")
        r = rep["regions"]["anchorage"]
        comp = r["components"]["seismic_thd"]
        self.assertIs(comp["frozen"], True)
        e = R.method_entry("seismic_thd", comp, r)
        self.assertIn("FROZEN", e["warnings"])
        self.assertEqual(e["recorded"]["frozen"], True)
        self.assertEqual(e["raw"]["value"], comp["raw_value"] if comp["available"] is True else None)

    def test_october_7_measured_value_survives_while_no_method_carried_the_day(self):
        r = region(BASE, OCT07, "hualien")
        e = R.method_entry("seismic_thd", r["components"]["seismic_thd"], r)
        self.assertEqual((e["raw"]["state"], e["raw"]["value"]), ("MEASURED", 0.42326726420803307))
        self.assertEqual((e["admission"]["state"], e["admission"]["included"], e["admission"]["code"]),
                         ("NOT_QUALIFIED", False, "CALIBRATED_AFTER_SCORED_DAY"))
        self.assertIn("NOT_ADMITTED_NOT_QUALIFIED", e["warnings"])
        c = R.combined_entry(r)
        self.assertEqual((c["state"], c["value"], c["recorded"]["combined_risk"]), ("NO_QUALIFIED_METHOD", None, 0.0))

    def test_missing_component_is_not_in_report(self):
        e = R.method_entry("lambda_geo", None, {})
        self.assertEqual((e["raw"]["state"], e["raw"]["value"], e["score"]["state"]), ("NOT_IN_REPORT", None, "NOT_IN_REPORT"))


@unittest.skipUnless(have(BASE), "pinned base commit absent from this clone")
class BuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = R.build(str(ROOT), BASE)
        cls.index = json.loads(cls.files["index.json"])
        cls.regions = {name[:-5]: json.loads(data) for name, data in cls.files.items() if name != "index.json"}

    def day(self, region_name, date):
        return next(d for d in self.regions[region_name]["days"] if d["date"] == date)

    def test_build_is_deterministic(self):
        self.assertEqual(R.build(str(ROOT), BASE), self.files)

    def test_committed_artifact_is_this_writer_output_at_its_source_commit(self):
        try:
            committed = json.loads(git("show", "HEAD:docs/raw_history/index.json"))
        except subprocess.CalledProcessError:
            self.skipTest("docs/raw_history/ is not committed in this checkout")
        self.assertEqual(committed["source_commit"], BASE)
        for name, data in self.files.items():
            self.assertEqual(git("show", "HEAD:docs/raw_history/" + name), data, name)

    def test_patch_touches_only_declared_paths(self):
        try:
            patch = git("log", "-1", "--format=%H", "--", "docs/raw_history/index.json").decode().strip()
        except subprocess.CalledProcessError:
            patch = ""
        if not patch:
            self.skipTest("docs/raw_history/ is not committed in this checkout")
        changed = git("diff", "--name-only", BASE, patch).decode().split()
        self.assertTrue(changed)
        self.assertEqual([p for p in changed if not p.startswith(DECLARED_PATCH_PATHS)], [], "existing history untouched")

    def test_every_retained_value_is_preserved_exactly(self):
        reports, _ = R.load_reports(str(ROOT), BASE)
        checked = 0
        for name, body in self.regions.items():
            for d in body["days"]:
                entry = reports.get(d["date"])
                if not entry or not entry.get("current"):
                    self.assertIn(d["source"]["kind"], ("NO_REPORT", "LEGACY_UNPARSEABLE"))
                    continue
                rep, source = entry["current"]
                self.assertEqual(d["source"], source)
                comps = ((rep.get("regions") or {}).get(name) or {}).get("components") or {}
                for method in R.METHODS:
                    m, comp = d["methods"][method], comps.get(method)
                    if comp is None:
                        self.assertEqual(m["raw"]["state"], "NOT_IN_REPORT")
                        continue
                    self.assertEqual(m["recorded"], {k: comp[k] for k in R.RECORDED_KEYS if k in comp})
                    if m["raw"]["state"] == "MEASURED":
                        self.assertEqual(m["raw"]["value"], comp["raw_value"])
                    else:
                        self.assertIsNone(m["raw"]["value"])
                    if m["score"]["state"] == "RECORDED":
                        self.assertEqual(m["score"]["value"], comp["risk_score"])
                    checked += 1
        self.assertGreater(checked, 9000)

    def test_multiple_report_revisions_are_separate(self):
        d = self.day("hualien", "2026-01-09")
        self.assertEqual(d["source"]["kind"], "LEGACY_LAST_COMMITTED_BLOB")
        self.assertEqual(len(d["superseded"]), 3)
        self.assertTrue(all(s["source"]["kind"] == "LEGACY_EARLIER_COMMITTED_BLOB" for s in d["superseded"]))
        self.assertNotIn(d["source"]["git_blob"], [s["source"]["git_blob"] for s in d["superseded"]])

    def test_repeated_identical_value_is_flagged_never_changed(self):
        # Ridgecrest LG 2026-10-05 differs from 10-04; 10-06 repeats 10-05 exactly (the report says frozen false).
        a, b = self.day("ridgecrest", "2026-10-05")["methods"]["lambda_geo"], self.day("ridgecrest", "2026-10-06")["methods"]["lambda_geo"]
        self.assertEqual(a["raw"]["value"], b["raw"]["value"])
        self.assertNotIn("REPEATS_PREVIOUS_MEASURED_VALUE", a["warnings"])
        self.assertIn("REPEATS_PREVIOUS_MEASURED_VALUE", b["warnings"])
        self.assertIs(b["recorded"].get("frozen"), False)
        self.assertEqual(b["raw"]["value"], b["recorded"]["raw_value"])
        for body in self.regions.values():
            for method in R.METHODS:
                previous = None
                for d in body["days"]:
                    m = d["methods"][method]
                    if m["raw"]["state"] != "MEASURED":
                        self.assertNotIn("REPEATS_PREVIOUS_MEASURED_VALUE", m.get("warnings", []))
                        continue
                    self.assertEqual("REPEATS_PREVIOUS_MEASURED_VALUE" in m["warnings"], previous == m["raw"]["value"])
                    previous = m["raw"]["value"]

    def test_missing_raw_dates_are_gaps_with_a_reason(self):
        d = self.day("hualien", "2025-12-01")
        self.assertEqual((d["source"]["kind"], d["methods"]["seismic_thd"]["raw"]), ("NO_REPORT", {"value": None, "state": "NO_REPORT"}))
        self.assertEqual(d["combined"]["state"], "CSV_ONLY_SCORED")

    def test_existing_history_preservation_comparison(self):
        p = self.index["preservation"]
        self.assertEqual((p["compared"], p["equal"], len(p["mismatches"])), (3682, 3370, 312))
        self.assertEqual(p["mismatch_reasons"], {"CSV_EQUALS_A_SUPERSEDED_REVISION": 45, "CSV_EQUALS_NO_RETAINED_REVISION": 36,
                                                 "NO_OTHER_RETAINED_REVISION": 185, "REGION_NOT_IN_REPORT": 46})
        rows = {(r["date"], r["region"]): r for r in csv.DictReader(io.StringIO(git("show", BASE + ":docs/data.csv").decode(), newline=""))}
        for m in p["mismatches"]:
            d = self.day(m["region"], m["date"])
            row = rows[(m["date"], m["region"])]
            self.assertEqual(d["csv"], {k: row[k] for k in ("risk", "tier", "methods", "agreement")}, "kept as published")
            self.assertEqual(d["csv_disagrees"]["reason"], m["reason"])
            for rev in m["matching_revisions"]:
                s = next(s for s in d["superseded"] if rev in (s["source"].get("commit"), s["source"].get("run_id")))
                self.assertEqual(list(R._csv_shape(s["combined"]["recorded"])), m["csv"])

    def test_a_retained_report_that_does_not_match_its_recorded_identity_refuses(self):
        real_show = R._show
        def tampered(repo, rev, path):
            data = real_show(repo, rev, path)
            if path == R.CAPSULE_REL:
                capsule = json.loads(data)
                capsule["records"][-1]["sha256"] = "0" * 64
                return json.dumps(capsule).encode("utf-8")
            if path == R.JOURNAL_REL:
                return data
            return data
        R._show = tampered
        try:
            with self.assertRaises(R.Refused) as caught:
                R.load_reports(str(ROOT), BASE)
            self.assertTrue(str(caught.exception).startswith("LEGACY_BLOB_MISMATCH"))
        finally:
            R._show = real_show
        def tampered_journal(repo, rev, path):
            data = real_show(repo, rev, path)
            if path == R.JOURNAL_REL:
                lines = data.decode("utf-8").splitlines()
                entry = json.loads(lines[-1]); entry["sha256"] = "0" * 64; lines[-1] = json.dumps(entry)
                return ("\n".join(lines) + "\n").encode("utf-8")
            return data
        R._show = tampered_journal
        try:
            with self.assertRaises(R.Refused) as caught:
                R.load_reports(str(ROOT), BASE)
            self.assertTrue(str(caught.exception).startswith("REVISION_SHA256_MISMATCH"))
        finally:
            R._show = real_show

    def test_writer_refuses_a_directory_it_did_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "raw"
            out.mkdir()
            (out / "other.txt").write_text("x", encoding="utf-8")
            with self.assertRaises(R.Refused):
                R.write(self.files, str(out))
            with self.assertRaises(R.Refused):
                R.write(self.files, str(out), replace_own=True)
            fresh = Path(tmp) / "fresh"
            R.write(self.files, str(fresh))
            self.assertEqual(sorted(p.name for p in fresh.iterdir()), sorted(self.files))
            R.write(self.files, str(fresh), replace_own=True)
            self.assertEqual({p.name: p.read_bytes() for p in fresh.iterdir()}, self.files)


@unittest.skipUnless(have(BASE), "pinned base commit absent from this clone")
class PageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        files = R.build(str(ROOT), BASE)
        cls.hualien = json.loads(files["hualien.json"])["days"]
        cls.ridgecrest = json.loads(files["ridgecrest.json"])["days"]
        cls.anchorage = json.loads(files["anchorage.json"])["days"]
        cls.functions = page_functions()

    def run_js(self, body):
        return node_eval(self.functions + body)

    def test_gaps_are_null_never_zero(self):
        out = self.run_js("const s=rawSeries(" + json.dumps(self.hualien) + ",'seismic_thd');" +
                          "process.stdout.write(JSON.stringify(s));")
        by = dict(zip(out["dates"], range(len(out["dates"]))))
        days = {d["date"]: d for d in self.hualien}
        for date, i in by.items():
            thd = days[date]["methods"]["seismic_thd"]
            measured = thd["raw"]["value"] if thd["raw"]["state"] == "MEASURED" else None
            frozen = "FROZEN" in thd.get("warnings", [])
            self.assertEqual((out["raw"][i], out["frozen"][i]), (None if frozen else measured, measured if frozen else None), date)
        i = by["2026-10-07"]
        self.assertEqual((out["raw"][i], out["score"][i], out["scoreNotAdmitted"][i], out["combined"][i], out["csvCombined"][i]),
                         (0.42326726420803307, None, 0.04718178923375134, None, None))
        self.assertIsNone(out["raw"][by["2025-12-01"]])
        # A CSV-only day has no report: the published CSV value is its own trace, never labelled as the report's.
        self.assertEqual((out["combined"][by["2025-12-01"]], out["csvCombined"][by["2025-12-01"]]), (None, 0.0791))
        self.assertEqual(len(out["gaps"]), sum(1 for d in self.hualien if d["methods"]["seismic_thd"]["raw"]["state"]
                                               not in ("MEASURED", "UNCONFIRMED_NONZERO")))
        self.assertNotIn(0, [out["raw"][by[g["date"]]] for g in out["gaps"]])

    def test_frozen_and_unconfirmed_values_are_drawn_apart_and_whiskers_follow_the_record(self):
        out = self.run_js("process.stdout.write(JSON.stringify(rawSeries(" + json.dumps(self.anchorage) + ",'seismic_thd')));")
        i = out["dates"].index("2026-07-30")
        day = next(d for d in self.anchorage if d["date"] == "2026-07-30")["methods"]["seismic_thd"]
        self.assertIn("FROZEN", day["warnings"])
        self.assertEqual((out["raw"][i], out["frozen"][i]), (None, day["raw"]["value"]))
        self.assertNotIn("2026-07-30", [g["date"] for g in out["gaps"]], "a frozen value is not a gap")
        for d, mean, std in zip(self.anchorage, out["baselineMean"], out["baselineStd"]):
            b = (d["methods"]["seismic_thd"].get("calibration") or {}).get("baseline") or {}
            ok = isinstance(b.get("mean"), (int, float)) and isinstance(b.get("std"), (int, float))
            self.assertEqual((mean, std), (b["mean"], b["std"]) if ok else (None, None), d["date"])
        lg = self.run_js("process.stdout.write(JSON.stringify(rawSeries(" + json.dumps(self.anchorage) + ",'lambda_geo')));")
        self.assertEqual(set(lg["baselineMean"]), {None}, "whiskers are a THD reference only")
        # DECLARED SYNTHETIC: one real day's THD entry set to an unconfirmed nonzero value (none in history).
        edited = json.loads(json.dumps(self.anchorage))
        target = next(d for d in edited if d["date"] == "2026-07-29")
        target["methods"]["seismic_thd"]["raw"] = {"state": "UNCONFIRMED_NONZERO", "value": None, "unconfirmed_value": 0.4}
        u = self.run_js("process.stdout.write(JSON.stringify(rawSeries(" + json.dumps(edited) + ",'seismic_thd')));")
        j = u["dates"].index("2026-07-29")
        self.assertEqual((u["raw"][j], u["unconfirmed"][j]), (None, 0.4))
        self.assertNotIn("2026-07-29", [g["date"] for g in u["gaps"]], "an unconfirmed value is shown, not a gap")

    def test_baseline_changes_follow_the_recorded_windows(self):
        out = self.run_js("process.stdout.write(JSON.stringify(rawSeries(" + json.dumps(self.hualien) + ",'seismic_thd').baselineChanges));")
        expected, last = [], None
        for d in self.hualien:
            w = ((d["methods"]["seismic_thd"].get("calibration") or {}).get("baseline") or {}).get("window")
            if isinstance(w, str):
                if last is not None and w != last:
                    expected.append({"date": d["date"], "from": last, "to": w})
                last = w
        self.assertEqual(out, expected)
        self.assertTrue(any(c["to"] == "2026-05-18 to 2026-08-16" for c in out))

    def test_table_and_csv_expose_the_same_values_unclamped(self):
        out = self.run_js("const rows=rawTableRows(" + json.dumps(self.ridgecrest) + ",'lambda_geo');" +
                          "process.stdout.write(JSON.stringify({rows, csv: rawCsv(rows)}));")
        parsed = list(csv.DictReader(io.StringIO(out["csv"], newline="")))
        self.assertEqual(len(parsed), len(self.ridgecrest))
        for table, row, day in zip(out["rows"], parsed, self.ridgecrest):
            self.assertEqual(table, row, day["date"])
            m = day["methods"]["lambda_geo"]
            raw = m["raw"]["value"] if m["raw"]["state"] == "MEASURED" else None
            self.assertEqual(float(row["raw_value"]) if row["raw_value"] else None, raw, day["date"])
            score = m["score"]["value"] if m["score"]["state"] == "RECORDED" else None
            self.assertEqual(float(row["score"]) if row["score"] else None, score, day["date"])
        self.assertTrue(any(float(r["raw_value"]) > 1 for r in parsed if r["raw_value"]), "raw > 1 kept")
        for row, day in zip(parsed, self.ridgecrest):
            recorded = day["methods"]["lambda_geo"].get("recorded") or {}
            self.assertEqual(row["recorded_raw_value"], "" if "raw_value" not in recorded else
                             ("0" if recorded["raw_value"] == 0 else row["recorded_raw_value"]), day["date"])
            if "raw_value" in recorded:
                self.assertEqual(float(row["recorded_raw_value"]), recorded["raw_value"], day["date"])
        placeholders = [r for r, d in zip(parsed, self.ridgecrest) if d["methods"]["lambda_geo"]["raw"]["state"] == "UNAVAILABLE_ZERO_PLACEHOLDER"]
        self.assertTrue(placeholders)
        self.assertTrue(all((r["raw_value"], r["recorded_raw_value"]) == ("", "0") for r in placeholders), "placeholder inspectable, not measured")

    def test_independent_axes_without_clamping(self):
        out = self.run_js("const s=rawSeries(" + json.dumps(self.hualien) + ",'seismic_thd');" +
                          "process.stdout.write(JSON.stringify(rawPlotSpec(s,'seismic_thd',{raw:'THD value (dimensionless)'})));")
        layout = out["layout"]
        self.assertNotIn("overlaying", layout["yaxis2"], "stacked panels, each on its own axis")
        self.assertLessEqual(layout["yaxis2"]["domain"][1], layout["yaxis"]["domain"][0])
        self.assertNotIn("range", layout["yaxis"])
        self.assertNotIn("range", layout["yaxis2"])
        self.assertTrue(all(t["connectgaps"] is False for t in out["traces"] if "lines" in t["mode"]))
        self.assertIn("unit not recorded", layout["yaxis"]["title"])

    def test_combined_history_plots_no_method_days_as_gaps(self):
        out = self.run_js("process.stdout.write(JSON.stringify([historyValue({methods:'0',risk:'0.0000'})," +
                          "historyValue({methods:'1',risk:'0.5560'}),historyValue({methods:'2',risk:''}),historyValue({methods:'',risk:'0.3'})]));")
        self.assertEqual(out, [None, 0.556, None, None])


if __name__ == "__main__":
    unittest.main()
