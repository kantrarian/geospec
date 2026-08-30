#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 CALIBRATION-LEDGER production runner (cayley) -- the
orchestration that, AT THE AVAILABILITY CUTOFF, fits the frozen
apply-never-refit ledgers and emits them with input-bound receipts.
Fills the `calibration_ledgers` execution-manifest slot when run for
real; until then this module is fixture-verified machinery.

FEED CONTRACT (the producer targets these shapes -- the panel-shape
precedent):
- MF4 feed: {"risk_by_region": {region: {iso_day: float}},
   "catalog_snapshot": [{"day","lat","lon","mag"}...],
   "snapshot_end": iso_day, "freeze_day": iso_day,
   "bboxes": {region: bbox}, "regions": [...]}
- MAG feed (one per observatory): {"observatory": iaga,
   "lon_east": deg, "times": [iso minute stamps],
   "components": {"X": [...], "Y": [...]},
   "weather": {name: [...aligned...]},
   "m3_reference": iaga-or-None}
  Calibration interval: 2026-01-01 -> the cutoff (the caller slices;
  the engines refuse unsupported/rank-deficient designs typed).

PROVENANCE RULE (the content-auth != derivation-provenance lesson):
every receipt binds (a) the INPUT CARRIER digests -- each feed's
canonical-JSON sha256 computed BEFORE fitting, (b) the PRODUCER
identity handed in by the producer (their code blob sha -- recorded,
not attested here), (c) THIS runner's own executed-source sha, and
(d) the output ledger digests. verify_receipt() recomputes the output
digests from the written artifacts and refuses on any divergence
(typed CALIBRATION_RECEIPT_MISMATCH). Cutoff ordering vs
evaluation_start is validated by the BARRIER at PRESTART assembly (the
cutoff exists before evaluation_start does); this runner records the
cutoff verbatim.

This module opens no window-2 value: calibration uses strictly
pre-evaluation bytes by construction of the interval.
"""
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import w2_mf4 as MF4
import w2_mag1 as MAG

OUT_DIR = "docs/f2g_window2_execution/calibration"


class CalibrationRunnerError(ValueError):
    """Typed refusal; the code leads the message."""


CAL_EPOCH_DAY = "2026-01-01"
RECEIPT_FIELDS_MF4 = {"schema", "lane", "cutoff", "input_feed_sha256",
                      "input_feed_path", "producer_identity",
                      "runner_source_sha256_normalized", "ledger_path",
                      "ledger_sha256", "training_digest", "n_rows"}
RECEIPT_FIELDS_MAG = {"schema", "lane", "cutoff", "input_feed_sha256",
                      "input_feed_path", "producer_identity",
                      "runner_source_sha256_normalized", "results"}
RECEIPT_FIELDS_MF4_AMENDED = {
    "schema", "lane", "cutoff", "input_feed_sha256",
    "input_feed_path", "producer_identity",
    "runner_source_sha256_normalized", "ledger_path",
    "ledger_sha256", "training_digest", "amended_training_digest",
    "catalog_binding", "n_rows"}


def _validate_mag_times(obs, times, cutoff):
    """codex 1358Z item 3 + the 1721Z canonical-frame grammar (their
    producer item 4 applies to this runner identically): one timestamp
    frame -- canonical UTC. Timezone-AWARE parsing; naive stamps are
    UTC by declaration and 'Z' is UTC; any NON-UTC offset refuses (a
    +14:00 stamp under a naive strip would pass the wrong UTC day).
    Interval/order/uniqueness checks run on the NORMALIZED UTC
    instants."""
    from datetime import datetime, timezone
    prev = None
    for t in times:
        try:
            dt = datetime.fromisoformat(
                str(t).replace("Z", "+00:00"))
        except ValueError:
            raise CalibrationRunnerError(
                f"CALIBRATION_TIME_INDEX_INVALID: {obs} unparseable "
                f"{t!r}")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        elif dt.utcoffset() != timezone.utc.utcoffset(None):
            raise CalibrationRunnerError(
                f"CALIBRATION_TIME_INDEX_INVALID: {obs} non-UTC "
                f"offset in {t!r} (canonical frame is UTC)")
        day = dt.astimezone(timezone.utc).date().isoformat()
        if day < CAL_EPOCH_DAY or day > str(cutoff):
            raise CalibrationRunnerError(
                f"CALIBRATION_AFTER_CUTOFF: {obs} {t} outside "
                f"[{CAL_EPOCH_DAY}, {cutoff}]")
        if prev is not None and dt <= prev:
            raise CalibrationRunnerError(
                f"CALIBRATION_TIME_INDEX_INVALID: {obs} not strictly "
                f"increasing at {t}")
        prev = dt


def _validate_mf4_temporal(feed, cutoff):
    """codex 1358Z item 3 (MF4 side): the registered cutoff/maturity
    relations bind the feed -- risk rows <= cutoff, catalog events <=
    snapshot_end, snapshot_end <= freeze_day."""
    for region, series in feed["risk_by_region"].items():
        for d in series:
            if str(d) > str(cutoff):
                raise CalibrationRunnerError(
                    f"CALIBRATION_AFTER_CUTOFF: {region} risk row {d} "
                    f"> cutoff {cutoff}")
    for ev in feed["catalog_snapshot"]:
        if str(ev["day"]) > str(feed["snapshot_end"]):
            raise CalibrationRunnerError(
                f"CALIBRATION_TIME_INDEX_INVALID: catalog event "
                f"{ev['day']} beyond snapshot_end "
                f"{feed['snapshot_end']}")
    if str(feed["snapshot_end"]) > str(feed["freeze_day"]):
        raise CalibrationRunnerError(
            "CALIBRATION_TIME_INDEX_INVALID: snapshot_end after "
            "freeze_day")


def _canon_digest(obj):
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def _self_sha():
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(
            f.read().replace(b"\r\n", b"\n")).hexdigest()


def _write(repo, rel, obj):
    p = os.path.join(repo, rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")
    return rel


def run_mf4_calibration(repo, feed, cutoff, producer_identity):
    """MF4 fit-once ledger + receipt. Input digest FIRST, then fit."""
    for k in ("risk_by_region", "catalog_snapshot", "snapshot_end",
              "freeze_day", "bboxes", "regions"):
        if k not in feed:
            raise CalibrationRunnerError(f"MF4_FEED_INCOMPLETE: {k}")
    _validate_mf4_temporal(feed, cutoff)
    feed_digest = _canon_digest(feed)
    # persist the canonical input carrier (codex item 4: receipts must
    # be independently reopenable -- the feed bytes ARE the carrier)
    feed_path = _write(repo, f"{OUT_DIR}/mf4_input_feed.json", feed)
    ledger = MF4.calibrate(feed["risk_by_region"],
                           feed["catalog_snapshot"], feed["bboxes"],
                           feed["regions"], feed["freeze_day"],
                           feed["snapshot_end"])
    led_path = _write(repo, f"{OUT_DIR}/mf4_ledger.json", ledger)
    receipt = {"schema": "f2g-w2-calibration-receipt-v1",
               "lane": "MF4", "cutoff": str(cutoff),
               "input_feed_sha256": feed_digest,
               "input_feed_path": feed_path,
               "producer_identity": dict(producer_identity),
               "runner_source_sha256_normalized": _self_sha(),
               "ledger_path": led_path,
               "ledger_sha256": _canon_digest(ledger),
               "training_digest": ledger["training_digest"],
               "n_rows": ledger["n_rows"]}
    rec_path = _write(repo, f"{OUT_DIR}/mf4_ledger.receipt.json",
                      receipt)
    return {"ledger": led_path, "receipt": rec_path,
            "ledger_sha256": receipt["ledger_sha256"]}


AMENDED_LEDGER_FIELDS = {
    "calibration_start", "calibration_issue_end", "training_digest",
    "n_rows", "regions", "scaler_mean", "scaler_std", "coef",
    "intercept", "baseline_coef", "baseline_intercept",
    "amended_training_digest", "amended_training_binding"}
AMENDED_PROVENANCE_FIELDS = {
    "capsule_sha256", "rows_sha256", "snapshot_sha256",
    "acquisition_receipt_sha256", "result_commit",
    "supported_cells", "producer_source_sha256_normalized"}
ADAPTER_MODULE_PATH = ("monitoring/src/"
                       "w2_mf4_catalog_adapter_grassmann.py")
PRODUCER_MODULE_PATH = ("monitoring/src/"
                        "w2_calibration_feed_producer_cayley.py")


def _manifest_pin_norm_sha(repo, rel):
    """The module's normalized-source pin from the committed
    execution manifest at `repo` -- the same blob a clean checkout
    materializes; absence refuses (a claim-bearing lane never runs
    against unpinned machinery)."""
    mp = os.path.join(repo, "docs", "f2g_window2_execution",
                      "execution_manifest.json")
    if not os.path.isfile(mp):
        raise CalibrationRunnerError(
            "MF4_AMENDED_MANIFEST_ABSENT: the amended lane runs "
            "only inside a checkout carrying the execution manifest")
    man = json.load(open(mp, encoding="utf-8"))
    for slot in (man.get("slots") or {}).values():
        if not isinstance(slot, dict) or                 slot.get("status") != "BOUND":
            continue
        for pin in slot.get("pins") or ():
            if pin.get("path") == rel:
                return pin.get("blob_sha256")
    raise CalibrationRunnerError(
        f"MF4_AMENDED_MODULE_UNPINNED: {rel} is not a BOUND pin of "
        "the execution manifest")


def _norm_file_sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(
            f.read().replace(b"\r\n", b"\n")).hexdigest()


def run_mf4_calibration_amended(repo):
    """THE registered amended-lane entry (codex 0439Z E1): repo
    only. The manifest-pinned producer EXECUTES here -- inputs are
    derived, never accepted -- so caller-constructed data can never
    wear the producer's identity. Fit + receipts continue in the
    private checked seam below (unregistered; fixtures only)."""
    import w2_calibration_feed_producer_cayley as FPM
    fp_sha = _norm_file_sha(os.path.abspath(FPM.__file__))
    if fp_sha != _manifest_pin_norm_sha(repo, PRODUCER_MODULE_PATH):
        raise CalibrationRunnerError(
            "MF4_AMENDED_PRODUCER_UNPINNED: the executing feed "
            "producer source diverges from its manifest pin")
    inputs = FPM.build_mf4_calibration_inputs(repo)
    ident = {"module": os.path.basename(PRODUCER_MODULE_PATH),
             "source_sha256_normalized": fp_sha}
    return _run_mf4_calibration_amended_with_inputs(repo, inputs,
                                                    ident)


def _amended_carrier(inputs, snap_sha, rcpt_sha):
    """THE one canonical carrier constructor (codex 0439Z F1) --
    the runner persists it and the verifier independently rebuilds
    it from committed bytes."""
    prov = inputs["provenance"]
    return {"risk_by_region": inputs["risk_by_region"],
            "regions": inputs["regions"],
            "bboxes": inputs["bboxes"],
            "freeze_day": inputs["freeze_day"],
            "snapshot_end": inputs["snapshot_end"],
            "requested_issue_end": inputs["requested_issue_end"],
            "catalog": {"snapshot_sha256": snap_sha,
                        "acquisition_receipt_sha256": rcpt_sha},
            "provenance": {k: prov[k] for k in sorted(prov)}}


def _run_mf4_calibration_amended_with_inputs(repo, inputs,
                                             producer_identity):
    """The AMENDED MF4 lane (codex 1758Z option 1; 0411Z P04-A): the
    fit runs ONLY through the registered authenticated adapter --
    resolved UNCONDITIONALLY inside this entry, never injectable (a
    caller-supplied fit path could mint a verifiable fake receipt).
    The executing adapter and the claimed feed producer must both
    equal their manifest pins; the returned ledger must satisfy the
    CLOSED amended schema and every cross-binding BEFORE anything is
    written. The runner never parses the snapshot and never calls
    the frozen engine directly for this lane."""
    import w2_mf4_catalog_adapter_grassmann as ADAPT
    adapter = ADAPT.calibrate_with_snapshot
    adapter_sha = _norm_file_sha(os.path.abspath(ADAPT.__file__))
    if adapter_sha != _manifest_pin_norm_sha(repo,
                                             ADAPTER_MODULE_PATH):
        raise CalibrationRunnerError(
            "MF4_AMENDED_ADAPTER_UNPINNED: the executing adapter "
            "source diverges from its manifest pin")
    for k in ("risk_by_region", "snapshot_bytes", "receipt_bytes",
              "bboxes", "regions", "freeze_day", "snapshot_end",
              "requested_issue_end", "provenance"):
        if k not in inputs:
            raise CalibrationRunnerError(
                f"MF4_AMENDED_INPUTS_INCOMPLETE: {k}")
    if not isinstance(inputs["snapshot_bytes"],
                      (bytes, bytearray)) or             not isinstance(inputs["receipt_bytes"],
                           (bytes, bytearray)):
        raise CalibrationRunnerError(
            "MF4_AMENDED_INPUTS_INCOMPLETE: snapshot/receipt must "
            "be the ORIGINAL bytes, never parsed objects")
    prov = inputs["provenance"]
    if not isinstance(prov, dict) or \
            set(prov) != AMENDED_PROVENANCE_FIELDS:
        raise CalibrationRunnerError(
            "MF4_AMENDED_PROVENANCE_NOT_CLOSED: "
            f"{sorted(prov) if isinstance(prov, dict) else prov!r}")
    snap_sha = hashlib.sha256(inputs["snapshot_bytes"]).hexdigest()
    rcpt_sha = hashlib.sha256(inputs["receipt_bytes"]).hexdigest()
    if prov["snapshot_sha256"] != snap_sha or \
            prov["acquisition_receipt_sha256"] != rcpt_sha:
        raise CalibrationRunnerError(
            "MF4_AMENDED_PROVENANCE_DIVERGENT: producer provenance "
            "does not bind the supplied snapshot/receipt bytes")
    if prov["producer_source_sha256_normalized"] != \
            dict(producer_identity).get("source_sha256_normalized"):
        raise CalibrationRunnerError(
            "MF4_AMENDED_PROVENANCE_DIVERGENT: provenance producer "
            "identity != claimed producer identity")
    if prov["producer_source_sha256_normalized"] != \
            _manifest_pin_norm_sha(repo, PRODUCER_MODULE_PATH):
        raise CalibrationRunnerError(
            "MF4_AMENDED_PRODUCER_UNPINNED: the claimed feed "
            "producer identity diverges from its manifest pin")
    cutoff = str(inputs["requested_issue_end"])
    for region, series in inputs["risk_by_region"].items():
        for d in series:
            if str(d) > cutoff:
                raise CalibrationRunnerError(
                    f"CALIBRATION_AFTER_CUTOFF: {region} risk row "
                    f"{d} > issue end {cutoff}")
    carrier = _amended_carrier(inputs, snap_sha, rcpt_sha)
    feed_digest = _canon_digest(carrier)
    # digest FIRST; the carrier PERSISTS only after every pre-write
    # validation below -- a refused fit leaves ZERO artifacts
    ledger = adapter(inputs["risk_by_region"],
                     inputs["snapshot_bytes"],
                     inputs["receipt_bytes"], inputs["bboxes"],
                     inputs["regions"], inputs["freeze_day"],
                     inputs["snapshot_end"],
                     requested_issue_end=
                     inputs["requested_issue_end"])
    if not isinstance(ledger, dict) or \
            set(ledger) != AMENDED_LEDGER_FIELDS:
        raise CalibrationRunnerError(
            "MF4_AMENDED_LEDGER_SCHEMA: the returned ledger is not "
            "the CLOSED amended shape "
            f"({sorted(ledger) if isinstance(ledger, dict) else ledger!r})")
    if sorted(ledger["regions"]) != sorted(inputs["regions"]):
        raise CalibrationRunnerError(
            "MF4_AMENDED_LEDGER_UNBOUND: ledger regions diverge "
            "from the carrier regions")
    if str(ledger["calibration_issue_end"]) != cutoff:
        raise CalibrationRunnerError(
            "MF4_AMENDED_LEDGER_UNBOUND: ledger "
            "calibration_issue_end diverges from the accepted "
            "issue end")
    bind = ledger["amended_training_binding"]
    if not isinstance(bind, dict) or \
            bind.get("snapshot_sha256") != snap_sha or \
            bind.get("engine_training_digest") != \
            ledger["training_digest"]:
        raise CalibrationRunnerError(
            "MF4_AMENDED_LEDGER_UNBOUND: the amended binding does "
            "not bind the carrier snapshot/engine digest")
    ra = bind.get("result_authentication")
    if not isinstance(ra, dict) or \
            ra.get("catalog_commit") != prov["result_commit"]:
        raise CalibrationRunnerError(
            "MF4_AMENDED_LEDGER_UNBOUND: result authentication "
            "does not bind the provenance result commit")
    for k_ra, k_pv in (("snapshot_sha256", "snapshot_sha256"),
                       ("receipt_sha256",
                        "acquisition_receipt_sha256")):
        if k_ra in ra and ra[k_ra] != prov[k_pv]:
            raise CalibrationRunnerError(
                "MF4_AMENDED_LEDGER_UNBOUND: result "
                f"authentication {k_ra} diverges from provenance")
    feed_path = _write(repo,
                       f"{OUT_DIR}/mf4_input_feed_amended.json",
                       carrier)
    led_path = _write(repo, f"{OUT_DIR}/mf4_ledger_amended.json",
                      ledger)
    receipt = {"schema": "f2g-w2-calibration-receipt-v1",
               "lane": "MF4_AMENDED", "cutoff": cutoff,
               "input_feed_sha256": feed_digest,
               "input_feed_path": feed_path,
               "producer_identity": dict(producer_identity),
               "runner_source_sha256_normalized": _self_sha(),
               "ledger_path": led_path,
               "ledger_sha256": _canon_digest(ledger),
               "training_digest": ledger["training_digest"],
               "amended_training_digest":
                   ledger["amended_training_digest"],
               "catalog_binding": dict(carrier["catalog"]),
               "n_rows": ledger.get("n_rows")}
    rec_path = _write(repo,
                      f"{OUT_DIR}/mf4_ledger_amended.receipt.json",
                      receipt)
    return {"ledger": led_path, "receipt": rec_path,
            "ledger_sha256": receipt["ledger_sha256"],
            "amended_training_digest":
                receipt["amended_training_digest"]}


def run_mag_calibration(repo, feeds, cutoff, producer_identity):
    """MAG subtraction ledgers per observatory per horizontal
    component, plus M3 reference regressions per the frozen
    instantiation. Input digests FIRST, then fits."""
    out = {"observatories": {}, "m3": {}}
    residuals = {}
    for obs in sorted(feeds):
        feed = feeds[obs]
        for k in ("observatory", "lon_east", "times", "components",
                  "weather"):
            if k not in feed:
                raise CalibrationRunnerError(
                    f"MAG_FEED_INCOMPLETE: {obs}:{k}")
        _validate_mag_times(obs, feed["times"], cutoff)
        n_t = len(feed["times"])
        for cname, series in list(feed["components"].items()) + \
                list(feed["weather"].items()):
            if len(series) != n_t:
                raise CalibrationRunnerError(
                    f"CALIBRATION_TIME_INDEX_INVALID: {obs} series "
                    f"{cname!r} length {len(series)} != times {n_t}")
        feed_digest = _canon_digest(
            {k: v for k, v in feed.items() if k != "m3_reference"})
        obs_rec = {"input_feed_sha256": feed_digest, "components": {}}
        residuals[obs] = {}
        for comp in ("X", "Y"):
            if comp not in feed["components"]:
                raise CalibrationRunnerError(
                    f"MAG_FEED_INCOMPLETE: {obs}:components:{comp}")
            led = MAG.fit_subtraction(
                feed["times"], feed["components"][comp],
                feed["lon_east"], feed["weather"],
                meta={"observatory": obs, "component": comp,
                      "cutoff": str(cutoff)})
            rel = _write(repo,
                         f"{OUT_DIR}/mag_{obs.lower()}_{comp}"
                         f"_ledger.json", led)
            obs_rec["components"][comp] = {
                "ledger_path": rel, "ledger_sha256": _canon_digest(led),
                "ledger_digest_field": led["digest"]}
            residuals[obs][comp] = MAG.apply_subtraction(
                led, feed["times"], feed["components"][comp],
                feed["weather"])
        out["observatories"][obs] = obs_rec
    # M3 references: local residual ~ reference residual + weather.
    # codex item 3: BYTE-EQUAL time indices required between local and
    # reference -- positional pairing of shifted clocks refuses.
    for obs in sorted(feeds):
        ref = feeds[obs].get("m3_reference")
        if not ref:
            continue
        if ref not in residuals:
            raise CalibrationRunnerError(
                f"MAG_M3_REFERENCE_ABSENT: {obs} -> {ref}")
        if list(map(str, feeds[obs]["times"])) != \
                list(map(str, feeds[ref]["times"])):
            raise CalibrationRunnerError(
                f"M3_TIME_INDEX_MISMATCH: {obs} vs {ref} time indices "
                "are not byte-equal")
        for comp in ("X", "Y"):
            led = MAG.fit_m3_reference(
                residuals[obs][comp], residuals[ref][comp],
                {n: feeds[obs]["weather"][n]
                 for n in sorted(feeds[obs]["weather"])},
                meta={"local": obs, "reference": ref,
                      "component": comp, "cutoff": str(cutoff)})
            rel = _write(repo,
                         f"{OUT_DIR}/mag_m3_{obs.lower()}_on_"
                         f"{ref.lower()}_{comp}_ledger.json", led)
            out["m3"][f"{obs}:{ref}:{comp}"] = {
                "ledger_path": rel,
                "ledger_sha256": _canon_digest(led)}
    # persist the canonical input carrier (codex item 4)
    feed_carrier = {obs: {k: v for k, v in feeds[obs].items()}
                    for obs in sorted(feeds)}
    feed_path = _write(repo, f"{OUT_DIR}/mag_input_feeds.json",
                       feed_carrier)
    receipt = {"schema": "f2g-w2-calibration-receipt-v1",
               "lane": "MAG", "cutoff": str(cutoff),
               "input_feed_sha256": _canon_digest(feed_carrier),
               "input_feed_path": feed_path,
               "producer_identity": dict(producer_identity),
               "runner_source_sha256_normalized": _self_sha(),
               "results": out}
    rec_path = _write(repo, f"{OUT_DIR}/mag_ledgers.receipt.json",
                      receipt)
    return {"receipt": rec_path, "results": out}


def verify_receipt(repo, receipt_rel, *, expected_cutoff,
                   expected_producer, expected_runner_sha=None,
                   expected_input_sha256=None,
                   expected_ledger_sha256=None):
    """codex 1358Z item 4 + 1815Z item 3: NO claim-bearing defaults.
    Expected cutoff and pinned producer identity are REQUIRED on every
    call; the executing runner is ALWAYS compared (to
    expected_runner_sha when a manifest pin is supplied, else to the
    executing bytes -- a supplied pin also must match the executing
    bytes). The lane enum and every nested result schema are CLOSED.
    The exact required ledger keyset is DERIVED from the persisted
    input feed and must be EQUAL -- an empty or subset result refuses.
    Any divergence refuses CALIBRATION_RECEIPT_MISMATCH."""
    with open(os.path.join(repo, receipt_rel.replace("/", os.sep)),
              encoding="utf-8") as f:
        rec = json.load(f)

    def refuse(detail):
        raise CalibrationRunnerError(
            f"CALIBRATION_RECEIPT_MISMATCH: {detail}")

    if rec.get("lane") not in ("MF4", "MAG", "MF4_AMENDED"):
        refuse(f"lane not in the closed enum: {rec.get('lane')!r}")
    want_fields = (RECEIPT_FIELDS_MF4 if rec["lane"] == "MF4"
                   else RECEIPT_FIELDS_MF4_AMENDED
                   if rec["lane"] == "MF4_AMENDED"
                   else RECEIPT_FIELDS_MAG)
    if set(rec) != want_fields:
        refuse(f"receipt schema not closed: "
               f"{sorted(set(rec) ^ want_fields)}")
    if rec["schema"] != "f2g-w2-calibration-receipt-v1":
        refuse(f"schema id {rec['schema']!r}")
    if rec["cutoff"] != str(expected_cutoff):
        refuse(f"cutoff {rec['cutoff']} != expected {expected_cutoff}")
    if rec["producer_identity"] != dict(expected_producer):
        refuse("producer identity diverges from the pinned identity")
    if rec["runner_source_sha256_normalized"] != _self_sha():
        refuse("runner sha claim does not match the executing runner")
    if expected_runner_sha is not None and \
            rec["runner_source_sha256_normalized"] != \
            expected_runner_sha:
        refuse("runner sha diverges from the manifest pin")

    def check(path_rel, want, what):
        with open(os.path.join(repo, path_rel.replace("/", os.sep)),
                  encoding="utf-8") as f:
            got = _canon_digest(json.load(f))
        if got != want:
            refuse(f"{what} {path_rel} {got[:12]} != {want[:12]}")

    # input carrier recomputation (independently reopenable) + the
    # REQUIRED keyset derived from it
    check(rec["input_feed_path"], rec["input_feed_sha256"], "input")
    with open(os.path.join(repo, rec["input_feed_path"]
                           .replace("/", os.sep)),
              encoding="utf-8") as f:
        feed = json.load(f)
    n = 0
    if rec["lane"] == "MF4":
        check(rec["ledger_path"], rec["ledger_sha256"], "output")
        n = 1
    elif rec["lane"] == "MF4_AMENDED":
        check(rec["ledger_path"], rec["ledger_sha256"], "output")
        with open(os.path.join(repo, rec["ledger_path"]
                               .replace("/", os.sep)),
                  encoding="utf-8") as f:
            led = json.load(f)
        if led.get("amended_training_digest") != \
                rec["amended_training_digest"] or \
                led.get("training_digest") != \
                rec["training_digest"]:
            refuse("amended/engine training digests do not "
                   "recompute from the persisted ledger")
        if rec["n_rows"] != led.get("n_rows"):
            refuse("receipt n_rows does not equal the persisted "
                   "ledger's n_rows")
        if feed.get("catalog") != rec.get("catalog_binding"):
            refuse("receipt catalog_binding does not equal the "
                   "persisted carrier's catalog block")
        prov2 = feed.get("provenance")
        if not isinstance(prov2, dict) or \
                set(prov2) != AMENDED_PROVENANCE_FIELDS:
            refuse("carrier provenance absent or not the closed "
                   "shape")
        if prov2["snapshot_sha256"] != \
                feed["catalog"]["snapshot_sha256"] or \
                prov2["acquisition_receipt_sha256"] != \
                feed["catalog"]["acquisition_receipt_sha256"]:
            refuse("carrier provenance does not bind the carrier "
                   "catalog digests")
        if str(led.get("calibration_issue_end")) != rec["cutoff"] \
                or sorted(led.get("regions") or ()) != \
                sorted(feed.get("regions") or ()):
            refuse("ledger issue-end/regions diverge from the "
                   "receipt cutoff / persisted carrier")
        bind2 = led.get("amended_training_binding") or {}
        if bind2.get("snapshot_sha256") != \
                prov2["snapshot_sha256"]:
            refuse("ledger amended binding does not bind the "
                   "carrier provenance snapshot")
        ra2 = bind2.get("result_authentication") or {}
        if ra2.get("catalog_commit") != prov2["result_commit"]:
            refuse("ledger result authentication does not bind the "
                   "provenance result commit")
        n = 1
        # codex 0439Z F1: self-consistency is never provenance. The
        # carrier is REBUILT read-only through the manifest-pinned
        # producer and the ONE canonical constructor; exact
        # canonical equality or refusal. Where the raw store is
        # unreachable the result is TYPED consistency-only -- a
        # coordinated carrier+receipt mutation can therefore never
        # masquerade as provenance-checked on any host.
        amended_provenance = "INTERNAL_CONSISTENCY_ONLY"
        try:
            import w2_calibration_feed_producer_cayley as _FPMV
            _re_in = _FPMV.build_mf4_calibration_inputs(repo)
            _re_carrier = _amended_carrier(
                _re_in,
                hashlib.sha256(_re_in["snapshot_bytes"]).hexdigest(),
                hashlib.sha256(_re_in["receipt_bytes"]).hexdigest())
            if _canon_digest(_re_carrier) != _canon_digest(feed):
                refuse("persisted carrier does not equal the "
                       "independently rebuilt producer carrier")
            amended_provenance = "REBUILT_FROM_COMMITTED_BYTES"
        except CalibrationRunnerError:
            raise
        except BaseException as _exc:
            _passthru = ("MF4_ARCHIVE_OBJECT_MISSING",
                         "MF4_AMENDED_MANIFEST_ABSENT",
                         "FEED_STAGED_ABSENT",
                         "MF4_INPUTS_REPO_MISMATCH")
            if not any(t in str(_exc) for t in _passthru):
                refuse(f"carrier rebuild failed: "
                       f"{type(_exc).__name__}: {str(_exc)[:140]}")
        # codex 0439Z F2: the ledger's independent bind arrives only
        # with the separately committed post-run final-bind record;
        # until its hashes are supplied as MANDATORY expected values
        # the ledger claim stays typed consistency-only.
        if expected_input_sha256 is not None or \
                expected_ledger_sha256 is not None:
            if rec["input_feed_sha256"] != expected_input_sha256:
                refuse("receipt input digest does not equal the "
                       "final-bind record's expected value")
            if rec["ledger_sha256"] != expected_ledger_sha256:
                refuse("receipt ledger digest does not equal the "
                       "final-bind record's expected value")
            amended_ledger_binding = "FINAL_BIND_EXPECTED_VERIFIED"
        else:
            amended_ledger_binding = "INTERNAL_CONSISTENCY_ONLY"
    else:
        res = rec["results"]
        if set(res) != {"observatories", "m3"}:
            refuse(f"results schema not closed: {sorted(res)}")
        want_obs = set(feed)
        if set(res["observatories"]) != want_obs:
            refuse(f"observatory set {sorted(res['observatories'])} "
                   f"!= required {sorted(want_obs)} (derived from the "
                   "persisted feed)")
        want_m3 = {f"{o}:{feed[o]['m3_reference']}:{c}"
                   for o in feed if feed[o].get("m3_reference")
                   for c in ("X", "Y")}
        if set(res["m3"]) != want_m3:
            refuse(f"m3 set {sorted(res['m3'])} != required "
                   f"{sorted(want_m3)}")
        for oname, obs in res["observatories"].items():
            if set(obs) != {"input_feed_sha256", "components"}:
                refuse(f"observatory schema not closed: {oname}")
            if set(obs["components"]) != {"X", "Y"}:
                refuse(f"component set not closed: {oname}")
            for c in obs["components"].values():
                if set(c) != {"ledger_path", "ledger_sha256",
                              "ledger_digest_field"}:
                    refuse(f"component schema not closed: {oname}")
                check(c["ledger_path"], c["ledger_sha256"], "output")
                n += 1
        for key, m3 in res["m3"].items():
            if set(m3) != {"ledger_path", "ledger_sha256"}:
                refuse(f"m3 schema not closed: {key}")
            check(m3["ledger_path"], m3["ledger_sha256"], "output")
            n += 1
    if n == 0:
        refuse("zero-ledger receipt")
    if rec["lane"] == "MF4_AMENDED":
        return {"verified_ledgers": n, "lane": rec["lane"],
                "provenance_checked": amended_provenance,
                "ledger_binding": amended_ledger_binding}
    return {"verified_ledgers": n, "lane": rec["lane"],
            "provenance_checked": True}


# ---------------------------------------------------------------- selftest
def _selftest():
    import tempfile
    import numpy as np
    from datetime import date, datetime, timedelta
    repo = tempfile.mkdtemp(prefix="w2_cal_kat_")
    rng = np.random.Generator(np.random.PCG64(17))
    producer = {"name": "kat-producer", "code_blob_sha256": "ab" * 32}

    # MF4: synthetic feed -> ledger + receipt -> verify -> tamper
    days = [(date(2025, 10, 10) + timedelta(days=i)).isoformat()
            for i in range(120)]
    bbox = {"min_lat": 30, "max_lat": 40, "min_lon": -125,
            "max_lon": -115}
    feed = {"risk_by_region": {r: {d: float(rng.uniform(0, 1))
                                   for d in days} for r in ("ra", "rb")},
            "catalog_snapshot": [
                {"day": (date(2025, 11, 1) + timedelta(days=7 * i))
                 .isoformat(), "lat": 35.0, "lon": -120.0, "mag": 4.5}
                for i in range(8)],
            "snapshot_end": "2026-02-08", "freeze_day": "2026-02-10",
            "bboxes": {"ra": bbox, "rb": bbox},
            "regions": ["ra", "rb"]}
    res = run_mf4_calibration(repo, feed, "2026-02-09", producer)
    v = verify_receipt(repo, res["receipt"],
                       expected_cutoff="2026-02-09",
                       expected_producer=producer)
    assert v == {"verified_ledgers": 1, "lane": "MF4",
                 "provenance_checked": True}
    # codex item 3 (MF4): a post-cutoff risk row refuses typed
    bad_feed = json.loads(json.dumps(feed))
    bad_feed["risk_by_region"]["ra"]["2026-02-10"] = 0.5
    try:
        run_mf4_calibration(repo, bad_feed, "2026-02-09", producer)
        raise AssertionError("post-cutoff risk must refuse")
    except CalibrationRunnerError as e:
        assert "CALIBRATION_AFTER_CUTOFF" in str(e)
    # determinism: rerun -> identical ledger digest
    res2 = run_mf4_calibration(repo, feed, "2026-02-09", producer)
    assert res2["ledger_sha256"] == res["ledger_sha256"]
    # tamper a written ledger -> receipt verification refuses
    lp = os.path.join(repo, res["ledger"].replace("/", os.sep))
    led = json.load(open(lp))
    led["intercept"] = 99.9
    json.dump(led, open(lp, "w"))
    try:
        verify_receipt(repo, res["receipt"],
                       expected_cutoff="2026-02-09",
                       expected_producer=producer)
        raise AssertionError("tampered ledger must refuse")
    except CalibrationRunnerError as e:
        assert "CALIBRATION_RECEIPT_MISMATCH" in str(e)
    try:
        run_mf4_calibration(repo, {k: v_ for k, v_ in feed.items()
                                   if k != "bboxes"},
                            "2026-02-09", producer)
        raise AssertionError("incomplete feed must refuse")
    except CalibrationRunnerError as e:
        assert "MF4_FEED_INCOMPLETE" in str(e)

    # MAG: two observatories, one M3 pair, per-component ledgers
    n = 3000
    times = [(datetime(2026, 1, 1) + timedelta(minutes=i)).isoformat()
             for i in range(n)]
    weather = {"symh": rng.normal(size=n).tolist()}

    def obs_feed(name, ref):
        return {"observatory": name, "lon_east": -120.0,
                "times": times,
                "components": {
                    "X": rng.normal(20000, 5, size=n).tolist(),
                    "Y": rng.normal(4000, 5, size=n).tolist()},
                "weather": weather, "m3_reference": ref}
    feeds = {"FRN": obs_feed("FRN", "TUC"),
             "TUC": obs_feed("TUC", None)}
    res = run_mag_calibration(repo, feeds, "2026-08-24", producer)
    v = verify_receipt(repo, res["receipt"],
                       expected_cutoff="2026-08-24",
                       expected_producer=producer)
    assert v["verified_ledgers"] == 6      # 2 obs x2 comps + 2 M3
    assert "FRN:TUC:X" in res["results"]["m3"]
    try:
        run_mag_calibration(
            repo, {"FRN": obs_feed("FRN", "NEW")}, "2026-08-24",
            producer)
        raise AssertionError("absent M3 reference must refuse")
    except CalibrationRunnerError as e:
        assert "MAG_M3_REFERENCE_ABSENT" in str(e)

    # provenance + keyset doctor battery FIRST, while the disk state
    # (receipt + input carrier) is the coherent two-observatory run
    rec_path = os.path.join(repo, res["receipt"].replace("/", os.sep))
    orig = open(rec_path, encoding="utf-8").read()

    def doctor(mut, label, **expect):
        rec = json.loads(orig)
        mut(rec)
        json.dump(rec, open(rec_path, "w"))
        try:
            verify_receipt(repo, res["receipt"],
                           expected_cutoff="2026-08-24",
                           expected_producer=producer, **expect)
            raise AssertionError(f"{label} must refuse")
        except CalibrationRunnerError as e:
            assert "CALIBRATION_RECEIPT_MISMATCH" in str(e), \
                (label, str(e))
        finally:
            open(rec_path, "w").write(orig)
    doctor(lambda r: r.__setitem__("cutoff", "2027-01-01"),
           "doctored cutoff")
    doctor(lambda r: r["producer_identity"].__setitem__(
        "name", "evil"), "doctored producer")
    doctor(lambda r: r.__setitem__(
        "runner_source_sha256_normalized", "0" * 64),
        "doctored runner sha")
    doctor(lambda r: r.__setitem__("extra_field", 1),
           "receipt schema not closed")
    # codex 1815Z item-3 doctors: derived-keyset equality + closed
    # nested schemas + lane enum (the forged-empty-receipt class)
    doctor(lambda r: r["results"].__setitem__(
        "observatories", {}), "empty observatories")
    doctor(lambda r: r["results"]["m3"].pop("FRN:TUC:X"),
           "removed M3 entry")
    doctor(lambda r: r.__setitem__("lane", "EVIL"), "changed lane")
    doctor(lambda r: r["results"]["observatories"]["FRN"]
           .__setitem__("extra", 1), "extra nested field")
    doctor(lambda r: r.__setitem__(
        "results", {"observatories": {}, "m3": {}}),
        "zero-ledger receipt")
    # doctored INPUT CARRIER file -> input digest recomputation catches
    fp = os.path.join(repo, "docs/f2g_window2_execution/calibration/"
                      "mag_input_feeds.json".replace("/", os.sep))
    saved_feed = open(fp, encoding="utf-8").read()
    fdoc = json.loads(saved_feed)
    fdoc["TUC"]["lon_east"] = -119.0
    json.dump(fdoc, open(fp, "w"))
    try:
        verify_receipt(repo, res["receipt"],
                       expected_cutoff="2026-08-24",
                       expected_producer=producer)
        raise AssertionError("doctored input carrier must refuse")
    except CalibrationRunnerError as e:
        assert "CALIBRATION_RECEIPT_MISMATCH" in str(e)
    finally:
        open(fp, "w").write(saved_feed)

    # codex item 3 doctors (the exact KAT list)
    def expect_refuse(feeds_d, cutoff, code, label):
        try:
            run_mag_calibration(repo, feeds_d, cutoff, producer)
            raise AssertionError(f"{label} must refuse")
        except CalibrationRunnerError as e:
            assert code in str(e), (label, str(e))
    # one-minute-after-cutoff (times run into 01-03; cutoff 01-02)
    expect_refuse({"TUC": obs_feed("TUC", None)}, "2026-01-02",
                  "CALIBRATION_AFTER_CUTOFF", "after-cutoff")
    # duplicate timestamp
    f_dup = {"TUC": obs_feed("TUC", None)}
    f_dup["TUC"]["times"] = list(times)
    f_dup["TUC"]["times"][100] = f_dup["TUC"]["times"][99]
    expect_refuse(f_dup, "2026-08-24",
                  "CALIBRATION_TIME_INDEX_INVALID", "duplicate")
    # reordered timestamps
    f_re = {"TUC": obs_feed("TUC", None)}
    f_re["TUC"]["times"] = list(times)
    f_re["TUC"]["times"][10], f_re["TUC"]["times"][11] = \
        f_re["TUC"]["times"][11], f_re["TUC"]["times"][10]
    expect_refuse(f_re, "2026-08-24",
                  "CALIBRATION_TIME_INDEX_INVALID", "reordered")
    # shifted-equal-length M3 clocks (codex's exact repro class)
    f_sh = {"FRN": obs_feed("FRN", "TUC"),
            "TUC": obs_feed("TUC", None)}
    f_sh["FRN"]["times"] = [
        (datetime(2026, 1, 2) + timedelta(minutes=i)).isoformat()
        for i in range(n)]
    expect_refuse(f_sh, "2026-08-24", "M3_TIME_INDEX_MISMATCH",
                  "shifted-clocks")
    # missing-row (one dropped mid-index; series consistently trimmed
    # so the per-obs alignment guard passes and the M3 equality check
    # is what refuses)
    f_mr = {"FRN": obs_feed("FRN", "TUC"),
            "TUC": obs_feed("TUC", None)}
    f_mr["FRN"]["times"] = times[:1500] + times[1501:]
    f_mr["FRN"]["components"] = {
        "X": f_mr["FRN"]["components"]["X"][:2999],
        "Y": f_mr["FRN"]["components"]["Y"][:2999]}
    f_mr["FRN"]["weather"] = {"symh": weather["symh"][:2999]}
    expect_refuse(f_mr, "2026-08-24", "M3_TIME_INDEX_MISMATCH",
                  "missing-row")
    # codex 1721Z canonical-frame doctors: a non-UTC offset refuses
    # even when its LOCAL date sits inside the window (the +14:00
    # trap); 'Z' and naive-as-UTC both pass
    f_tz = {"TUC": obs_feed("TUC", None)}
    f_tz["TUC"]["times"] = list(times)
    f_tz["TUC"]["times"][0] = "2026-01-01T00:00:00+14:00"
    expect_refuse(f_tz, "2026-08-24",
                  "CALIBRATION_TIME_INDEX_INVALID", "non-utc-offset")
    f_z = {"TUC": obs_feed("TUC", None)}
    f_z["TUC"]["times"] = [t + "Z" for t in times]
    run_mag_calibration(repo, f_z, "2026-08-24", producer)  # Z ok

    # misaligned series/times (the new alignment guard)
    f_al = {"TUC": obs_feed("TUC", None)}
    f_al["TUC"]["components"] = dict(
        f_al["TUC"]["components"],
        X=f_al["TUC"]["components"]["X"][:2999])
    expect_refuse(f_al, "2026-08-24",
                  "CALIBRATION_TIME_INDEX_INVALID", "misaligned")

    print("w2_calibration_runner selftest: ALL PASS")


if __name__ == "__main__":
    _selftest()
