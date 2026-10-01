"""
calibration_eligibility_report.py -- the before/after report for the calibration-eligibility rule (generated, never
hand-written): for every fixture, the status, eligibility, the CURRENT score and the tier with the rule OFF vs ON,
plus (optionally) a read-only projection over a served display database's retained THD notes.

  python calibration_eligibility_report.py --base-ensemble <path to base-commit ensemble.py copy>
                                           [--served-db <display.sqlite>] [--out-dir ../data/reports]

Timestamp = live clock read at generation. The rule itself stays OFF; this script activates it only inside the
harness for the ON column. No collection, no recalibration, no writes to any store.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import calibration_eligibility_fixtures as FX  # noqa: E402

FX.install_stubs()
import calibration_eligibility as CE  # noqa: E402
import ensemble  # noqa: E402


def fixture_rows(base_module):
    rows = []
    for name, fx in FX.fixture_baselines().items():
        args = dict(region=fx["regions"][0], network=fx["network"], station=fx["station"],
                    thd_baseline=fx["baseline"], thd_value=fx["thd"], shared=fx["regions"])
        off = FX.build(ensemble, active=False, **args)
        on = FX.build(ensemble, active=True, **args)
        base = FX.build(base_module, **args) if base_module is not None else None
        thd_on = on.components["seismic_thd"]
        rows.append(dict(
            fixture=name, family="seismic_thd", station=f'{fx["network"]}.{fx["station"]}', regions=fx["regions"],
            thd_value=fx["thd"], expected_status=fx["expected"],
            status=thd_on.calibration_status, eligible=thd_on.eligible_for_tiering, reason=thd_on.eligibility_reason,
            current_score=off.components["seismic_thd"].risk_score, z_score=off.components["seismic_thd"].z_score,
            baseline_n=off.components["seismic_thd"].baseline_n, baseline_quality=off.components["seismic_thd"].baseline_quality,
            tier_rule_off=dict(tier=off.tier, name=off.tier_name, methods_available=off.methods_available),
            tier_rule_on=dict(tier=on.tier, name=on.tier_name, methods_available=on.methods_available, notes=on.notes),
            tier_changes=(off.tier != on.tier),
            flag_off_equals_base=(None if base is None else
                                  json.dumps(FX.result_dict(base), sort_keys=True) == json.dumps(FX.result_dict(off), sort_keys=True)),
        ))
    for name, fx in FX.fc_fixtures().items():
        e = CE.classify_fc_calibration(fx["state"], fx["reasons"], capsule=fx.get("capsule"),
                                       scored_day=FX.SCORED_DAY, embargo_days=fx.get("embargo"))
        rows.append(dict(fixture=name, family="fault_correlation", status=e.status, eligible=e.eligible_for_tiering,
                         code=e.code, reason=e.reason, expected_status=fx["expected"], expected_code=fx["code"],
                         note="unavailable capsules are already available=False in the runner; the rule records why; "
                              "admitted capsules are re-checked against valid_through and the registered embargo"))
    for name, fx in FX.LG_FIXTURES.items():
        e = CE.classify_lambda_geo(fx["provenance"], FX.SCORED_DAY, max_age_days=fx["max_age"])
        rows.append(dict(fixture=name, family="lambda_geo", status=e.status, eligible=e.eligible_for_tiering,
                         code=e.code, reason=e.reason, expected_status=fx["expected"], expected_code=fx["code"],
                         provenance=fx["provenance"], max_age_days_policy=fx["max_age"],
                         policy_basis=("EXPLICIT_TEST_POLICY_NOT_REGISTERED" if fx["max_age"] is not None
                                       else "RUNNER_REGISTERS_NONE")))
    for name, fx in FX.input_validation_fixtures().items():
        b = fx["baseline"]
        e = CE.classify_thd_baseline(b, FX.SCORED_DAY, max_age_days=FX.MAX_AGE)
        rows.append(dict(fixture=name, family="seismic_thd_input_validation", status=e.status,
                         eligible=e.eligible_for_tiering, code=e.code, reason=e.reason, expected_status=fx["expected"],
                         expected_code=fx["code"], inputs=dict(mean_thd=repr(b.mean_thd), std_thd=repr(b.std_thd),
                                                               n_samples=repr(b.n_samples),
                                                               calibration_period=b.calibration_period)))
    return rows


NOTE_RE = re.compile(r"sta=([A-Z0-9.]+),.*?baseline_mean=([0-9.]+), baseline_std=([0-9.]+), n=(\d+)")


def served_projection(db_path):
    """Read-only, notes-derived projection over a served display database: which retained THD observations the
    active rule would classify n0_default (n=0 in the retained notes) versus a positive n. The window end is not
    in the notes, so 'n>0' rows are labelled `calibrated_by_n_only_window_not_in_notes` -- a projection under the
    prospective rule from the retained record, NOT a re-score of history and NOT a claim about staleness."""
    conn = sqlite3.connect("file:" + db_path.replace("\\", "/") + "?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT scored_day, region, method, available, raw_value, risk_score, notes "
                            "FROM d3_daily_method_observations ORDER BY scored_day, region, method").fetchall()
        results = {(d, r): (t, n) for d, r, t, n in conn.execute(
            "SELECT scored_day, region, tier, tier_name FROM d3_daily_region_results")}
        span = conn.execute("SELECT min(scored_day), max(scored_day), count(*) FROM d3_daily_method_observations").fetchone()
    finally:
        conn.close()
    counts, would_change = {}, []
    for day, region, method, available, raw, score, notes in rows:
        if method != "seismic_thd":
            key = (method, "available" if available else "unavailable")
            counts[key] = counts.get(key, 0) + 1
            continue
        if not available:
            klass = "unavailable"
        else:
            m = NOTE_RE.search(notes or "")
            if m is None:
                klass = "no_baseline_fields_in_notes"
            elif m.group(4) == "0":
                klass = "n0_default"
            else:
                klass = "calibrated_by_n_only_window_not_in_notes"
        counts[("seismic_thd", klass)] = counts.get(("seismic_thd", klass), 0) + 1
        if klass == "n0_default":
            tier = results.get((day, region), (None, None))
            would_change.append(dict(scored_day=day, region=region, station=(NOTE_RE.search(notes or "") or [None, None])[1],
                                     raw_value=raw, saved_score=score, saved_tier=tier[0], saved_tier_name=tier[1],
                                     projection="THD would be n0_default -> not counted toward the tier if the rule were active"))
    return dict(database=db_path, scored_day_span=[span[0], span[1]], observations=span[2],
                counts={"%s:%s" % k: v for k, v in sorted(counts.items())}, n0_default_region_days=would_change,
                basis="NOTES_DERIVED_PROJECTION_NOT_A_RESCORE")


def markdown(report):
    lines = ["# %s -- before/after report (NOT ACTIVATED)" % report["rule_version"], "",
             "Generated %s from fixtures; rule flag OFF in the runner. `tier_rule_on` is the harness activating the rule; "
             "saved historical scores are untouched." % report["generated_utc"], "",
             "| fixture | family | status | eligible | current score | z | n | tier OFF | tier ON | tier changes | flag-off == base |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in report["fixtures"]:
        if r["family"] != "seismic_thd":
            continue
        lines.append("| %s | %s | %s | %s | %.7f | %.2f | %s | %s (%s) | %s (%s) | %s | %s |" % (
            r["fixture"], r["family"], r["status"], r["eligible"], r["current_score"], r["z_score"], r["baseline_n"],
            r["tier_rule_off"]["tier"], r["tier_rule_off"]["name"], r["tier_rule_on"]["tier"], r["tier_rule_on"]["name"],
            r["tier_changes"], r["flag_off_equals_base"]))
    others = [r for r in report["fixtures"] if r["family"] != "seismic_thd"]
    if others:
        lines += ["", "## Classifier fixtures (typed codes; v2 input validation and registered policies)", "",
                  "| fixture | family | status | eligible | code | expected | reason |", "|---|---|---|---|---|---|---|"]
        lines += ["| %s | %s | %s | %s | %s | %s/%s | %s |" % (
            r["fixture"], r["family"], r["status"], r["eligible"], r.get("code"), r["expected_status"],
            r.get("expected_code"), str(r.get("reason")).replace("|", "/")) for r in others]
    proj = report.get("served_projection")
    if proj:
        lines += ["", "## Served-record projection (read-only, notes-derived; basis %s)" % proj["basis"], "",
                  "Database `%s`, scored days %s..%s, %d observations." % (proj["database"], proj["scored_day_span"][0],
                                                                            proj["scored_day_span"][1], proj["observations"]),
                  "", "| class | rows |", "|---|---|"]
        lines += ["| %s | %d |" % (k, v) for k, v in proj["counts"].items()]
        lines += ["", "Region-days whose THD is n0_default in the retained notes (would not count toward the tier under the rule):", ""]
        lines += ["- %s %s (%s): raw %.4f, saved score %.4f, saved tier %s" % (
            d["scored_day"], d["region"], d["station"], d["raw_value"], d["saved_score"], d["saved_tier_name"]) for d in proj["n0_default_region_days"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--base-ensemble", default=None, help="base-commit ensemble.py copy (flag-off equality column)")
    ap.add_argument("--served-db", default=None, help="a served display.sqlite copy (read-only projection)")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "data", "reports"))
    ap.add_argument("--base-sha", default=None)
    args = ap.parse_args(argv)
    base = FX.load_module_from_file(args.base_ensemble, "ensemble_base") if args.base_ensemble else None
    now = datetime.now(timezone.utc)
    report = dict(schema="geospec-calibration-eligibility-before-after-v1", rule_version=CE.ELIGIBILITY_RULE_VERSION,
                  rule_active_in_runner=CE.ELIGIBILITY_RULE_ACTIVE, generated_utc=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  base_commit=args.base_sha, scored_day_fixture=FX.SCORED_DAY.strftime("%Y-%m-%d"),
                  max_baseline_age_days=ensemble.MAX_BASELINE_AGE_DAYS,
                  lambda_geo_baseline_max_age_days=ensemble.LAMBDA_GEO_BASELINE_MAX_AGE_DAYS,
                  fault_correlation_registered_embargo_days=FX.fc_registered_embargo_days(),
                  stubbed_modules=list(FX.STUBBED),
                  fixtures=fixture_rows(base))
    if args.served_db:
        report["served_projection"] = served_projection(args.served_db)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, "calibration_eligibility_before_after_%s" % now.strftime("%Y%m%d"))
    with open(stem + ".json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
        fh.write("\n")
    with open(stem + ".md", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(markdown(report))
    print(json.dumps(dict(json=stem + ".json", md=stem + ".md", fixtures=len(report["fixtures"]),
                          tier_changes=[r["fixture"] for r in report["fixtures"] if r.get("tier_changes")],
                          flag_off_equals_base=[r.get("flag_off_equals_base") for r in report["fixtures"] if r["family"] == "seismic_thd"],
                          expectation_mismatches=[r["fixture"] for r in report["fixtures"]
                                                  if r.get("expected_status") != r.get("status")
                                                  or (r.get("expected_code") and r.get("expected_code") != r.get("code"))]),
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
