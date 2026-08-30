#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lock tests for the P0-4 calibration feed producer (codex 1733Z
D1/D2 required list). Fixtures mint their staged artifacts through
THE REAL CAP.admission_transform -- the anti-second-implementation
guarantee is itself under test: the producer's extraction must agree
with what the pinned transform admitted, and every doctored surface
must refuse typed. All writes confined to temp dirs."""
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import w2_accrual_instrument_cayley as ACC
import w2_acquisition_capture_grassmann as CAP
import w2_calibration_feed_producer_cayley as FP
import w2_calibration_runner_cayley as RUN
import w2_mag1 as MAG1

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  -- {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def refuses(fn, code):
    try:
        fn()
    except BaseException as exc:
        return code in str(exc)
    return False


DAYS = ["2026-01-01", "2026-06-30"]
CUTOFF = "2026-06-30"
DOY = {"2026-01-01": 1, "2026-06-30": 181}
MAG_CARRIERS = {"frn": "FRN", "tuc": "TUC"}
KP_VALS = {"2026-01-01": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
           "2026-06-30": [2.5, 3.5, 1.5, 0.5, 4.5, 5.5, 6.5, 7.5]}
# registered missingness planted in the fixtures:
SYMH_FILL_MIN = 5        # sym_h fill sentinel at minute 5, day 1
NEWELL_FILL_MIN = 7      # omni var-18 fill at minute 7, day 1
MAG_NULL_MIN = range(10, 20)   # FRN X nulls, day 1


def _tmpl(kind, **rp):
    return {"source": {"kind": kind, "ref": "https://kat.example"},
            "endpoint": "https://kat.example/q",
            "request_params": dict(rp, d="{day}"),
            "operation_params": {"day": "{day}"}}


def build_authority():
    keys = {"MAG_FEED": {c: list(DAYS) for c in MAG_CARRIERS},
            "MAG_WEATHER_FEED": {c: list(DAYS)
                                 for c in FP.WEATHER_CARRIERS}}
    static = {
        "MAG_FEED": {"carriers": {
            c: {"cutoff": CUTOFF,
                "static_contract_template":
                    _tmpl("usgs-geomag-ws-minute",
                          id=MAG_CARRIERS[c])}
            for c in MAG_CARRIERS}},
        "MAG_WEATHER_FEED": {"carriers": {
            "kp": {"cutoff": CUTOFF,
                   "static_contract_template":
                       _tmpl("gfz-kp-json")},
            "omni": {"cutoff": CUTOFF,
                     "static_contract_template":
                         _tmpl("omniweb-highres-cgi",
                               vars=["17", "18", "21"])},
            "sym_h": {"cutoff": CUTOFF,
                      "static_contract_template":
                          _tmpl("omniweb-highres-cgi", vars="41")},
        }}}
    return {"schema": ACC.AUTHORITY_SCHEMA,
            "registered_probe_authority": {
                "path": "docs/kat", "commit": "a" * 40,
                "blob_sha256": "b" * 64, "role": "kat"},
            "template_token_vocabulary":
                list(ACC.TEMPLATE_TOKEN_VOCABULARY),
            "prestart_expected_keys": keys,
            "prestart_expected_keys_sha256": hashlib.sha256(
                json.dumps(keys, sort_keys=True,
                           separators=(",", ":")).encode()
                ).hexdigest(),
            "static_layer": static, "dynamic_layer": {},
            "digests": {}, "provenance": {}}


def mag_body(obs, day):
    n = 1440
    x = [10.0 + math.sin(i / 97.0) for i in range(n)]
    y = [-3.0 + math.cos(i / 83.0) for i in range(n)]
    z = [40000.0] * n
    f = [40010.0] * n
    if obs == "FRN" and day == DAYS[0]:
        for i in MAG_NULL_MIN:
            x[i] = None
    doc = {"metadata": {"intermagnet": {
               "imo": {"iaga_code": obs},
               "reported_orientation": "XYZF"}},
           "times": FP.canonical_grid(day),
           "values": [{"id": "X", "values": x},
                      {"id": "Y", "values": y},
                      {"id": "Z", "values": z},
                      {"id": "F", "values": f}]}
    return json.dumps(doc).encode()


def kp_body(day):
    stamps = [f"{day}T{h:02d}:00:00Z" for h in range(0, 24, 3)]
    return json.dumps({"Kp": KP_VALS[day], "datetime": stamps,
                       "status": ["def"] * 8}).encode()


def omni_body(day, var_list):
    fills = [CAP._OMNIWEB_VAR_FILL[v] for v in var_list]
    lines = ["<pre>", "YYYY DOY HR MN " + " ".join(var_list)]
    for m in range(1440):
        vals = []
        for i, v in enumerate(var_list):
            if day == DAYS[0] and (
                    (var_list == ["41"] and m == SYMH_FILL_MIN)
                    or (v == "18" and m == NEWELL_FILL_MIN)):
                vals.append(fills[i])
            elif v == "41":
                vals.append(
                    f"{-20.0 + DOY[day] * 0.37 + m / 500.0:.1f}")
            elif v == "21":
                vals.append(
                    f"{420.0 + DOY[day] * 1.3 + m / 300.0:.1f}")
            else:
                vals.append(
                    f"{1.0 + DOY[day] / 90.0 + m / 700.0:.2f}")
        lines.append(f"2026 {DOY[day]} {m // 60} {m % 60} "
                     + " ".join(vals))
    lines.append("</pre>")
    return ("\n".join(lines) + "\n").encode()


def build_fixture(root):
    """A complete fixture repo: authority + staged classes minted by
    the REAL transform + inventory + descriptor + store + capsules."""
    staged = os.path.join(root, *ACC.STAGED_PREFIX.split("/"))
    store = os.path.join(root, "store")
    os.makedirs(staged)
    os.makedirs(store)
    auth = build_authority()

    def wjson(path, obj):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, sort_keys=True, separators=(",", ":"))
            f.write("\n")

    inv_entries = {}

    def stage(lane, carrier, day, body):
        s = ACC.authoritative_static_contract(auth, lane, carrier,
                                              day)
        art = CAP.admission_transform(lane, body, s)
        sha = hashlib.sha256(body).hexdigest()
        with open(os.path.join(store, sha + ".body"), "wb") as f:
            f.write(body)
        stem = f"{lane.lower()}_{carrier}_{day}"
        wjson(os.path.join(staged, stem + ".artifact.json"), art)
        wjson(os.path.join(staged, stem + ".transcript.json"),
              {"raw_body_sha256": sha, "lane": lane,
               "carrier": carrier, "utc_day": day})
        inv_entries[f"{lane}/{carrier}/{day}"] = {
            "sha256": sha, "bytes": len(body)}

    for day in DAYS:
        for c, obs in MAG_CARRIERS.items():
            stage("MAG_FEED", c, day, mag_body(obs, day))
        stage("MAG_WEATHER_FEED", "kp", day, kp_body(day))
        stage("MAG_WEATHER_FEED", "omni", day,
              omni_body(day, ["17", "18", "21"]))
        stage("MAG_WEATHER_FEED", "sym_h", day,
              omni_body(day, ["41"]))
    wjson(os.path.join(staged, ACC.EXPECTED_KEYS_BASENAME), auth)
    inv = CAP.build_staged_body_inventory("kat-store", "kat://x",
                                          inv_entries)
    wjson(os.path.join(staged, ACC.STAGED_INVENTORY_BASENAME), inv)
    wjson(os.path.join(staged, ACC.STORE_DESCRIPTOR_BASENAME),
          {"schema": "f2g-w2-store-descriptor-v1",
           "store_id": "kat-store", "store_root": "kat://x",
           "physical_root": store})
    capdir = os.path.join(root, "docs", "f2g_window2_execution",
                          "mag_capsules")
    os.makedirs(capdir)
    for c, obs in MAG_CARRIERS.items():
        wjson(os.path.join(capdir, f"mag_capsule_{c}.json"),
              {"iaga_code": obs, "coordinates_lat_lon":
               [36.0, -119.7 if obs == "FRN" else -110.9]})
    return root


def _redir(src_root, mutate):
    """Copy the fixture tree, retarget the store descriptor at the
    COPY's store (physical_root is absolute), apply `mutate(root)`,
    return the copy's root."""
    dst = tempfile.mkdtemp(prefix="fpkat_")
    shutil.rmtree(dst)
    shutil.copytree(src_root, dst)
    dp = os.path.join(dst, *ACC.STAGED_PREFIX.split("/"),
                      ACC.STORE_DESCRIPTOR_BASENAME)
    d = json.load(open(dp))
    d["physical_root"] = os.path.join(dst, "store")
    json.dump(d, open(dp, "w"))
    mutate(dst)
    return dst


def _staged_file(root, lane, carrier, day, cls):
    stem = f"{lane.lower()}_{carrier}_{day}"
    return os.path.join(root, *ACC.STAGED_PREFIX.split("/"),
                        stem + f".{cls}.json")


def main():
    root = build_fixture(tempfile.mkdtemp(prefix="fpkat_root_"))

    # ---- G1 golden build --------------------------------------
    feeds, prov = FP.build_mag_feeds(root)
    n_min = len(DAYS) * 1440
    check("G1a golden build yields both observatories with the "
          "full retained minute grid",
          sorted(feeds) == ["FRN", "TUC"]
          and all(len(feeds[o]["times"]) == n_min
                  and len(feeds[o]["components"]["X"]) == n_min
                  and len(feeds[o]["weather"]["sym_h"]) == n_min
                  for o in feeds))
    check("G1b cutoff and day census derive from the authority",
          prov["cutoff"] == CUTOFF and prov["days"] == len(DAYS))
    frn = feeds["FRN"]
    check("G1c unsupported MAG minutes carry the registered None "
          "state (never compacted)",
          all(frn["components"]["X"][i] is None
              and frn["components"]["Y"][i] is None
              for i in MAG_NULL_MIN)
          and frn["components"]["X"][25] is not None)
    check("G1d SYM-H fill minute is None; Newell fill minute is "
          "None; numeric minutes are floats",
          frn["weather"]["sym_h"][SYMH_FILL_MIN] is None
          and frn["weather"]["newell"][NEWELL_FILL_MIN] is None
          and isinstance(frn["weather"]["sym_h"][6], float)
          and isinstance(frn["weather"]["newell"][6], float))
    check("G1e byte-equal time grids across all feeds "
          "(the M3 join precondition)",
          feeds["FRN"]["times"] == feeds["TUC"]["times"])

    # ---- D2: the frozen map, arrow direction, refusals --------
    check("M1 the frozen map is exactly FRN->TUC, VIC->NEW, "
          "IZN/TUC/NEW null",
          FP.M3_REFERENCE_MAP == {"FRN": "TUC", "VIC": "NEW",
                                  "IZN": None, "TUC": None,
                                  "NEW": None})
    check("M2 golden feeds carry FRN->TUC and TUC->null",
          frn["m3_reference"] == "TUC"
          and feeds["TUC"]["m3_reference"] is None)

    def stub(**m3):
        return {o: {"m3_reference": r} for o, r in m3.items()}
    check("M3 the inverse arrow refuses (NEW->VIC inverts the "
          "runner semantics)",
          refuses(lambda: FP.validate_m3_map(
              stub(NEW="VIC", VIC=None)), "M3_PAIR_UNREGISTERED"))
    check("M4 self-reference refuses",
          refuses(lambda: FP.validate_m3_map(stub(FRN="FRN")),
                  "M3_PAIR_UNREGISTERED"))
    check("M5 an unknown station refuses",
          refuses(lambda: FP.validate_m3_map(stub(KAT="TUC")),
                  "M3_PAIR_UNREGISTERED"))
    check("M6 a registered target with a MISSING reference feed "
          "refuses",
          refuses(lambda: FP.validate_m3_map(stub(FRN="TUC")),
                  "M3_REFERENCE_ABSENT"))
    check("M7 IZN must carry null (a novel IZN pair refuses)",
          refuses(lambda: FP.validate_m3_map(
              stub(IZN="FRN", FRN="TUC", TUC=None)),
              "M3_PAIR_UNREGISTERED"))

    # ---- D1: Kp half-open interval semantics ------------------
    kp = frn["weather"]["kp"]
    k1 = KP_VALS[DAYS[0]]
    check("K1 the 02:59/03:00 boundary is half-open (minute 179 = "
          "interval 0, minute 180 = interval 1)",
          kp[179] == k1[0] and kp[180] == k1[1]
          and kp[359] == k1[1] and kp[360] == k1[2])
    check("K2 day 2's first minute takes day 2's own first "
          "interval (no carry across days)",
          kp[1440] == KP_VALS[DAYS[1]][0])
    r_unsup = _redir(root, lambda d: None)
    art_p = _staged_file(r_unsup, "MAG_WEATHER_FEED", "kp",
                         DAYS[0], "artifact")
    art = json.load(open(art_p))
    art["support_mask"][1] = False
    art["definitive_samples"] = 7
    json.dump(art, open(art_p, "w"), sort_keys=True,
              separators=(",", ":"))
    f2, _ = FP.build_mag_feeds(r_unsup)
    check("K3 an unsupported Kp interval yields None for exactly "
          "its own minutes",
          all(f2["FRN"]["weather"]["kp"][m] is None
              for m in range(180, 360))
          and f2["FRN"]["weather"]["kp"][179] == k1[0]
          and f2["FRN"]["weather"]["kp"][360] == k1[2])
    r_absent = _redir(root, lambda d: os.remove(
        _staged_file(d, "MAG_WEATHER_FEED", "kp", DAYS[1],
                     "artifact")))
    check("K4 an ABSENT staged Kp day refuses -- no carry, no fill",
          refuses(lambda: FP.build_mag_feeds(r_absent),
                  "FEED_STAGED_ABSENT"))

    # ---- D1: shifted / doctored surfaces ----------------------
    def shift_symh(d):
        key = f"MAG_WEATHER_FEED/sym_h/{DAYS[0]}"
        body = omni_body(DAYS[0], ["41"]).decode()
        body = body.replace(f"\n2026 1 0 30 ", f"\n2026 1 0 31 ", 1)
        raw = body.encode()
        sha = hashlib.sha256(raw).hexdigest()
        store = os.path.join(d, "store")
        with open(os.path.join(store, sha + ".body"), "wb") as f:
            f.write(raw)
        tr_p = _staged_file(d, "MAG_WEATHER_FEED", "sym_h", DAYS[0],
                            "transcript")
        tr = json.load(open(tr_p))
        tr["raw_body_sha256"] = sha
        json.dump(tr, open(tr_p, "w"))
        inv_p = os.path.join(d, *ACC.STAGED_PREFIX.split("/"),
                             ACC.STAGED_INVENTORY_BASENAME)
        inv = json.load(open(inv_p))
        inv["objects"][key]["sha256"] = sha
        inv["objects"][key]["path"] = sha + ".body"
        json.dump(inv, open(inv_p, "w"))
    check("S1 a shifted SYM-H minute violates the registered "
          "cadence and refuses (never realigned)",
          refuses(lambda: FP.build_mag_feeds(_redir(root,
                  shift_symh)), "FEED_OMNI_CADENCE_VIOLATION"))

    def doctor_mask(d):
        p = _staged_file(d, "MAG_FEED", "frn", DAYS[0], "artifact")
        a = json.load(open(p))
        a["support_mask"][0] = not a["support_mask"][0]
        json.dump(a, open(p, "w"), sort_keys=True,
                  separators=(",", ":"))
    check("A1 a support-mask mutation with unchanged numeric bytes "
          "refuses at adjudication",
          refuses(lambda: FP.build_mag_feeds(_redir(root,
                  doctor_mask)), "FEED_ADJUDICATION_DIVERGENT"))

    def doctor_body(d):
        key_sha = json.load(open(_staged_file(
            d, "MAG_FEED", "tuc", DAYS[0],
            "transcript")))["raw_body_sha256"]
        p = os.path.join(d, "store", key_sha + ".body")
        with open(p, "ab") as f:
            f.write(b" ")
    check("B1 doctored store bytes refuse at the digest join",
          refuses(lambda: FP.build_mag_feeds(_redir(root,
                  doctor_body)), "FEED_BODY_DIGEST_DIVERGENT"))

    def beyond_cutoff(d):
        p = os.path.join(d, *ACC.STAGED_PREFIX.split("/"),
                         ACC.EXPECTED_KEYS_BASENAME)
        a = json.load(open(p))
        a["prestart_expected_keys"]["MAG_FEED"]["frn"].append(
            "2026-07-01")
        json.dump(a, open(p, "w"))
    check("C1 an authority day beyond the registered cutoff "
          "refuses",
          refuses(lambda: FP.build_mag_feeds(_redir(root,
                  beyond_cutoff)), "FEED_AFTER_CUTOFF"))

    # ---- D1: exact production design-column names -------------
    _, names = MAG1.build_design_matrix(
        frn["times"][:64], frn["lon_east"],
        {"kp": [1.0] * 64, "newell": [1.0] * 64,
         "sym_h": [1.0] * 64})
    check("D1 the production design columns are exactly the frozen "
          "recipe over {kp, newell, sym_h}",
          names == ["intercept", "weather:kp", "weather:newell",
                    "weather:sym_h",
                    "lst_sin_24.0h", "lst_cos_24.0h",
                    "lst_sin_12.0h", "lst_cos_12.0h",
                    "lst_sin_8.0h", "lst_cos_8.0h",
                    "seasonal_sin_365.25d", "seasonal_cos_365.25d",
                    "seasonal_sin_182.63d",
                    "seasonal_cos_182.63d"], str(names))

    # ---- E2E: the produced feeds drive the REAL runner --------
    runroot = tempfile.mkdtemp(prefix="fpkat_run_")
    try:
        res = RUN.run_mag_calibration(
            runroot, feeds, CUTOFF,
            {"module": "w2_calibration_feed_producer_cayley.py",
             "source_sha256_normalized":
                 prov["producer_source_sha256_normalized"]})
        out = res["results"]
        m3_keys = sorted(out.get("m3", ()))
        e1_ok = (m3_keys == ["FRN:TUC:X", "FRN:TUC:Y"]
                 and set(out["observatories"]) == {"FRN", "TUC"})
        if e1_ok:
            RUN.verify_receipt(
                runroot, res["receipt"], expected_cutoff=CUTOFF,
                expected_producer={
                    "module":
                        "w2_calibration_feed_producer_cayley.py",
                    "source_sha256_normalized":
                        prov["producer_source_sha256_normalized"]})
        e1_why = str(m3_keys)
    except BaseException as exc:
        e1_ok, e1_why = False, f"{type(exc).__name__}: {exc}"
    check("E1 the real runner accepts the produced feeds and emits "
          "exactly the frozen M3 ledgers", e1_ok, e1_why)

    # ---- MF4 amended lane (codex 1758Z opt 1 + 0317Z + 0411Z) --
    import w2_calibration_feed_producer_cayley as FPM
    import w2_mf4_archive_capsule_gen_grassmann as ARCH
    import w2_mf4_catalog_adapter_grassmann as ADAPT
    repo_real = ARCH.REPO

    cap = json.load(open(os.path.join(
        repo_real, *ARCH.CAPSULE_REL.split("/"))))
    # the FULL binding verification reopens the raw S4T objects, so
    # it is HOST-BOUND. codex 0411Z item 4: catch ONLY the typed
    # raw-object-absent refusal -- every other exception is a real
    # defect and re-raises.
    inputs = None
    try:
        inputs = FPM.build_mf4_calibration_inputs(repo_real)
        store_here = True
    except BaseException as exc:
        if "MF4_ARCHIVE_OBJECT_MISSING" not in str(exc):
            raise
        store_here = False
    if store_here:
        ma1_ok = (sorted(inputs["regions"])
                  == sorted(cap["region_sets"]["admitted_regions"])
                  and len(inputs["regions"]) == 13
                  and inputs["requested_issue_end"]
                  == cap["maturity_bounds"]
                  ["calibration_interval"][1]
                  and isinstance(inputs["snapshot_bytes"], bytes)
                  and inputs["provenance"]["supported_cells"] > 0
                  and all(set(b) == {"min_lat", "max_lat",
                                     "min_lon", "max_lon"}
                          for b in inputs["bboxes"].values()))
        check("MA1 the committed capsule/rows/snapshot bindings "
              "verify read-only; engine-exact inner bboxes",
              ma1_ok)
    else:
        check("MA1t store-less host fails CLOSED with the typed "
              "raw-object refusal (full MA suite runs where the "
              "S4T store exists)", True)
        print("  [NOTE] HOST-TRUNCATED: MA1/MA2/MA7 binding checks "
              "require the S4T raw store (evidence/review hosts); "
              "the locks below run on CONSTRUCTED inputs")
        inputs = {
            "risk_by_region": {"anchorage": {"2026-01-05": 0.4,
                                             "2026-02-01": 0.6}},
            "snapshot_bytes": b'{"kat": "snapshot"}',
            "receipt_bytes": b'{"kat": "receipt"}',
            "bboxes": {"anchorage": {"min_lat": 60.0,
                                     "max_lat": 62.0,
                                     "min_lon": -152.0,
                                     "max_lon": -148.0}},
            "regions": ["anchorage"],
            "freeze_day": "2026-08-28",
            "snapshot_end": "2026-08-27",
            "requested_issue_end": "2026-08-20",
            "provenance": {
                "capsule_sha256": "1" * 64,
                "rows_sha256": "2" * 64,
                "snapshot_sha256": hashlib.sha256(
                    b'{"kat": "snapshot"}').hexdigest(),
                "acquisition_receipt_sha256": hashlib.sha256(
                    b'{"kat": "receipt"}').hexdigest(),
                "result_commit": "4" * 40,
                "supported_cells": 2,
                "producer_source_sha256_normalized":
                    FPM._self_norm_sha()}}

    if store_here:
        rbr = inputs["risk_by_region"]
        days_all = sorted({d for m in rbr.values() for d in m})
        check("MA2 risk rows stay inside the registered interval "
              "and excluded regions never enter",
              days_all[0] >= "2025-10-18"
              and days_all[-1] <= inputs["requested_issue_end"]
              and "tokyo_kanto" not in rbr)

    # ---- P04 locks (codex 0411Z), every host --------------------
    # a fixture CHECKOUT: temp root carrying the real committed
    # manifest, so pin checks run against the true pins while all
    # writes stay in the temp root
    def mk_ckpt():
        root = tempfile.mkdtemp(prefix="fpkat_amn_")
        d = os.path.join(root, "docs", "f2g_window2_execution")
        os.makedirs(d)
        shutil.copyfile(
            os.path.join(repo_real, "docs", "f2g_window2_execution",
                         "execution_manifest.json"),
            os.path.join(d, "execution_manifest.json"))
        return root

    ident = {"module": "w2_calibration_feed_producer_cayley.py",
             "source_sha256_normalized":
                 inputs["provenance"]
                 ["producer_source_sha256_normalized"]}
    import inspect
    check("P04-E1 the registered production entry accepts REPO "
          "ONLY -- no inputs, identity, or adapter parameter",
          list(inspect.signature(
              RUN.run_mf4_calibration_amended).parameters)
          == ["repo"])
    try:
        RUN.run_mf4_calibration_amended(mk_ckpt(), inputs, ident)
        a_ok = False
    except TypeError:
        a_ok = True
    except BaseException:
        a_ok = False
    check("P04-E1b caller-supplied inputs are impossible at the "
          "production entry (TypeError)", a_ok)
    # P04-E2: mutations can only enter through COMMITTED surfaces,
    # which the internally-executed producer refuses BEFORE the
    # adapter. Sentinel proves the adapter is never reached.
    class _Sentinel(Exception):
        pass

    def _sentinel(*a, **k):
        raise _Sentinel("REACHED_ADAPTER")
    real_cws0 = ADAPT.calibrate_with_snapshot
    try:
        ADAPT.calibrate_with_snapshot = _sentinel
        try:
            # Exercise the real registered repo-only path. A temp root
            # containing only a copied manifest must refuse at
            # MF4_INPUTS_REPO_MISMATCH, which would make the store-less
            # negative vacuous and the store-backed positive fail before
            # testing producer execution.
            RUN.run_mf4_calibration_amended(repo_real)
            e2_ok, e2_why = False, "no refusal"
        except _Sentinel:
            e2_ok, e2_why = store_here, "adapter reached"
        except BaseException as exc:
            e2_ok = "REACHED_ADAPTER" not in str(exc)
            e2_why = str(exc)[:90]
    finally:
        ADAPT.calibrate_with_snapshot = real_cws0
    if store_here:
        # with the real store, the sentinel firing means the whole
        # internally-derived path verified -- that IS the positive;
        # the mutations below must then refuse pre-sentinel
        check("P04-E2pos the internally-executed producer path "
              "reaches the adapter on genuine committed surfaces",
              e2_why == "adapter reached")
        e2_muts = [
            ("bbox coordinate", lambda c: c["bboxes"]["anchorage"]
             ["bbox"].__setitem__("min_lat", 0.0)),
            ("admitted regions", lambda c: c["region_sets"]
             ["admitted_regions"].append("kat_region")),
            ("freeze day", lambda c: c["maturity_bounds"]
             .__setitem__("freeze_day", "2026-08-29")),
            ("snapshot end", lambda c: c["maturity_bounds"]
             .__setitem__("snapshot_end", "2026-08-26")),
            ("issue end", lambda c: c["maturity_bounds"]
             ["calibration_interval"].__setitem__(1, "2026-08-21")),
            ("supported census", lambda c: c["support_census"]
             ["anchorage"].__setitem__("days_supported", 1)),
        ]
        for label, mut in e2_muts:
            fx2 = tempfile.mkdtemp(prefix="fpkat_e2_")
            for rel in (ARCH.CAPSULE_REL, ARCH.RECEIPT_REL,
                        ARCH.ROWS_REL,
                        cap["catalog_binding"]["snapshot_path"],
                        cap["catalog_binding"]["receipt_path"],
                        cap["maturity_bounds"]["source"],
                        "docs/f2g_window2_execution/"
                        "execution_manifest.json"):
                src = os.path.join(repo_real, *rel.split("/"))
                dst = os.path.join(fx2, *rel.split("/"))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(src, dst)
            cp = os.path.join(fx2, *ARCH.CAPSULE_REL.split("/"))
            cobj = json.load(open(cp, encoding="utf-8"))
            mut(cobj)
            json.dump(cobj, open(cp, "w", encoding="utf-8"),
                      sort_keys=True, separators=(",", ":"))
            _r0 = ARCH.REPO
            try:
                ARCH.REPO = fx2
                ADAPT.calibrate_with_snapshot = _sentinel
                try:
                    RUN.run_mf4_calibration_amended(fx2)
                    got = False
                except _Sentinel:
                    got = False
                except BaseException:
                    got = True
            finally:
                ARCH.REPO = _r0
                ADAPT.calibrate_with_snapshot = real_cws0
            check(f"P04-E2 a doctored {label} refuses before the "
                  "adapter sentinel", got)
    else:
        check("P04-E2t store-less host: the repo-only entry refuses "
              "typed BEFORE the adapter sentinel (full nine-surface "
              "battery runs where the store exists)", e2_ok, e2_why)

    # P04-pos: the REGISTERED adapter runs for real and refuses the
    # non-fitting fixture bytes TYPED, before any write -- the one
    # real fit stays owner-gated
    root_pos = mk_ckpt()
    # Keep this negative non-fitting on every host. When `store_here`
    # is true, `inputs` contains the genuine catalog bytes; passing it
    # here would execute the owner-gated real fit from a selftest.
    _bad_snap = b'{"kat":"snapshot"}'
    _bad_rcpt = b'{"kat":"receipt"}'
    fixture_inputs = dict(
        inputs,
        snapshot_bytes=_bad_snap,
        receipt_bytes=_bad_rcpt,
        provenance=dict(
            inputs["provenance"],
            snapshot_sha256=hashlib.sha256(_bad_snap).hexdigest(),
            acquisition_receipt_sha256=
                hashlib.sha256(_bad_rcpt).hexdigest()))
    try:
        RUN._run_mf4_calibration_amended_with_inputs(
            root_pos, fixture_inputs, ident)
        pos_ok, pos_why = False, "did not refuse"
    except BaseException as exc:
        pos_ok = "MF4_CATALOG" in str(exc)
        pos_why = str(exc)[:110]
    caldir = os.path.join(root_pos, "docs",
                          "f2g_window2_execution", "calibration")
    check("P04-pos the registered adapter is reached and refuses "
          "fixture bytes typed, with ZERO artifacts written",
          pos_ok and not os.path.isdir(caldir), pos_why)

    # P04-B: an invented two-digest ledger (hostile state
    # CONSTRUCTED by module-attr patch) cannot write or verify
    real_cws = ADAPT.calibrate_with_snapshot
    try:
        ADAPT.calibrate_with_snapshot = lambda *a, **k: {
            "training_digest": "e" * 64,
            "amended_training_digest": "a" * 64, "n_rows": 1}
        root_b = mk_ckpt()
        okb = refuses(
            lambda: RUN._run_mf4_calibration_amended_with_inputs(
                root_b, inputs, ident), "MF4_AMENDED_LEDGER_SCHEMA")
        okb = okb and not os.path.isdir(os.path.join(
            root_b, "docs", "f2g_window2_execution", "calibration"))
    finally:
        ADAPT.calibrate_with_snapshot = real_cws
    check("P04-B an invented two-digest ledger refuses on the "
          "CLOSED schema and writes nothing", okb)

    # P04-C: provenance omission / divergence refuse
    bad = json.loads(json.dumps(
        {k: v for k, v in inputs.items()
         if k not in ("snapshot_bytes", "receipt_bytes")}))
    bad["snapshot_bytes"] = inputs["snapshot_bytes"]
    bad["receipt_bytes"] = inputs["receipt_bytes"]
    del bad["provenance"]["result_commit"]
    check("P04-C1 a provenance field omission refuses (closed "
          "keyset)",
          refuses(lambda: RUN._run_mf4_calibration_amended_with_inputs(
              mk_ckpt(), bad, ident),
              "MF4_AMENDED_PROVENANCE_NOT_CLOSED"))
    bad2 = dict(inputs, provenance=dict(inputs["provenance"],
                                        snapshot_sha256="9" * 64))
    check("P04-C2 a provenance/bytes divergence refuses",
          refuses(lambda:
              RUN._run_mf4_calibration_amended_with_inputs(
                  mk_ckpt(), bad2, ident),
              "MF4_AMENDED_PROVENANCE_DIVERGENT"))
    check("P04-C3 a producer identity that is not the manifest pin "
          "refuses",
          refuses(lambda:
              RUN._run_mf4_calibration_amended_with_inputs(
                  mk_ckpt(),
                  dict(inputs, provenance=dict(
                  inputs["provenance"],
                  producer_source_sha256_normalized="8" * 64)),
              {"module": "kat",
               "source_sha256_normalized": "8" * 64}),
              "MF4_AMENDED_PRODUCER_UNPINNED"))

    # P04-D: a WELL-FORMED ledger (constructed, consistent with the
    # carrier) writes in the fixture root; the receipt verifies;
    # every mutation refuses independently
    snap_sha_fix = hashlib.sha256(inputs["snapshot_bytes"]).hexdigest()

    def full_ledger(*a, **k):
        return {"calibration_start": "2025-10-18",
                "calibration_issue_end":
                    inputs["requested_issue_end"],
                "training_digest": "e" * 64, "n_rows": 42,
                "regions": sorted(inputs["regions"]),
                "scaler_mean": [0.0], "scaler_std": [1.0],
                "coef": [0.0], "intercept": 0.0,
                "baseline_coef": [0.0], "baseline_intercept": 0.0,
                "amended_training_digest": "a" * 64,
                "amended_training_binding": {
                    "engine_training_digest": "e" * 64,
                    "temporal_role_policy_sha256": "b" * 64,
                    "canonical_event_table_sha256": "c" * 64,
                    "snapshot_sha256": snap_sha_fix,
                    "receipt_schema": "kat",
                    "authorization_sha256": "d" * 64,
                    "trust_anchor": {},
                    "result_authentication": {
                        "catalog_commit":
                            inputs["provenance"]["result_commit"]},
                    "policy_source": "kat"}}
    root_d = mk_ckpt()
    try:
        ADAPT.calibrate_with_snapshot = full_ledger
        r = RUN._run_mf4_calibration_amended_with_inputs(
            root_d, inputs, ident)
    finally:
        ADAPT.calibrate_with_snapshot = real_cws

    # Make the receipt fixture a genuine rebuild host when the S4T
    # store is present. A manifest-only temp root necessarily trips
    # MF4_INPUTS_REPO_MISMATCH and can prove only the typed fallback,
    # never F1's positive rebuild/refusal path.
    if store_here:
        for rel in (ARCH.CAPSULE_REL, ARCH.RECEIPT_REL,
                    ARCH.ROWS_REL,
                    cap["catalog_binding"]["snapshot_path"],
                    cap["catalog_binding"]["receipt_path"],
                    cap["maturity_bounds"]["source"]):
            src = os.path.join(repo_real, *rel.split("/"))
            dst = os.path.join(root_d, *rel.split("/"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)

    def verify_d(**kwargs):
        real_repo_d = ARCH.REPO
        try:
            if store_here:
                ARCH.REPO = root_d
            return RUN.verify_receipt(
                root_d, r["receipt"],
                expected_cutoff=inputs["requested_issue_end"],
                expected_producer=ident, **kwargs)
        finally:
            ARCH.REPO = real_repo_d

    try:
        verify_d()
        d_ok, d_why = True, ""
    except BaseException as exc:
        d_ok, d_why = False, f"{type(exc).__name__}: {exc}"
    check("P04-D1 a consistent constructed ledger writes and the "
          "receipt verifies through production verify_receipt "
          "(fixture root only)", d_ok, d_why)
    if d_ok:
        v0 = verify_d()
        check("P04-F0 the amended lane NEVER reports bare "
              "provenance_checked=True without an independent "
              "rebuild; ledger stays typed consistency-only",
              v0.get("provenance_checked") in (
                  "INTERNAL_CONSISTENCY_ONLY",
                  "REBUILT_FROM_COMMITTED_BYTES")
              and v0.get("provenance_checked") is not True
              and v0.get("ledger_binding")
              == "INTERNAL_CONSISTENCY_ONLY")

        def coordinated(mutfile, mutate, refresh_receipt_key):
            pth = os.path.join(root_d, mutfile.replace("/", os.sep))
            raw = json.load(open(pth, encoding="utf-8"))
            keep = json.dumps(raw)
            mutate(raw)
            json.dump(raw, open(pth, "w", encoding="utf-8"))
            rp2 = os.path.join(root_d,
                               r["receipt"].replace("/", os.sep))
            rec2 = json.load(open(rp2, encoding="utf-8"))
            keep_r = json.dumps(rec2)
            import w2_calibration_runner_cayley as _R
            rec2[refresh_receipt_key] = _R._canon_digest(raw)
            json.dump(rec2, open(rp2, "w", encoding="utf-8"))
            try:
                out2 = verify_d()
                masq = out2.get("provenance_checked") is True
                refused = False
            except BaseException:
                masq, refused = False, True
            finally:
                open(pth, "w", encoding="utf-8").write(keep)
                open(rp2, "w", encoding="utf-8").write(keep_r)
            return masq, refused
        masq1, ref1 = coordinated(
            "docs/f2g_window2_execution/calibration/"
            "mf4_input_feed_amended.json",
            lambda d: d["risk_by_region"]["anchorage"]
            .__setitem__("2026-01-05", 0.99),
            "input_feed_sha256")
        check("P04-F1 a coordinated carrier mutation + refreshed "
              "receipt hash can never masquerade as "
              "provenance-checked (refuses on rebuild hosts)",
              (ref1 if store_here else not masq1))
        masq2, ref2 = coordinated(
            r["ledger"],
            lambda d: d.__setitem__("coef", [999.0]),
            "ledger_sha256")
        check("P04-F2a a coordinated coefficient mutation stays "
              "typed consistency-only (never provenance)",
              not masq2)
        check("P04-F2b with the final-bind expected values "
              "supplied, the original digests verify and a "
              "mutation refuses",
              (lambda: (verify_d(
                  expected_input_sha256=json.load(open(os.path.join(
                      root_d, r["receipt"].replace("/", os.sep))))
                  ["input_feed_sha256"],
                  expected_ledger_sha256=r["ledger_sha256"])
                  .get("ledger_binding")
                  == "FINAL_BIND_EXPECTED_VERIFIED"
                  and refuses(lambda: verify_d(
                      expected_input_sha256="0" * 64,
                      expected_ledger_sha256=r["ledger_sha256"]),
                      "final-bind record")))())

    def mutate_verify(mutfile, mutate, needle):
        pth = os.path.join(root_d, mutfile.replace("/", os.sep))
        raw = json.load(open(pth, encoding="utf-8"))
        keep = json.dumps(raw)
        mutate(raw)
        json.dump(raw, open(pth, "w", encoding="utf-8"))
        try:
            got = refuses(lambda: verify_d(), needle)
        finally:
            open(pth, "w", encoding="utf-8").write(keep)
        return got
    check("P04-D2 a doctored receipt n_rows refuses",
          mutate_verify(r["receipt"],
                        lambda d: d.__setitem__("n_rows", 999999),
                        "CALIBRATION_RECEIPT_MISMATCH"))
    check("P04-D3 a doctored carrier catalog digest refuses",
          mutate_verify(
              "docs/f2g_window2_execution/calibration/"
              "mf4_input_feed_amended.json",
              lambda d: d["catalog"].__setitem__(
                  "snapshot_sha256", "7" * 64),
              "CALIBRATION_RECEIPT_MISMATCH"))
    check("P04-D4 a doctored carrier provenance result commit "
          "refuses",
          mutate_verify(
              "docs/f2g_window2_execution/calibration/"
              "mf4_input_feed_amended.json",
              lambda d: d["provenance"].__setitem__(
                  "result_commit", "5" * 40),
              "CALIBRATION_RECEIPT_MISMATCH"))
    bad3 = dict(inputs)
    bad3["snapshot_bytes"] = json.loads(
        inputs["snapshot_bytes"].decode("utf-8"))
    check("P04-D5 a parsed snapshot OBJECT refuses -- bytes-only "
          "end to end",
          refuses(lambda:
              RUN._run_mf4_calibration_amended_with_inputs(
                  mk_ckpt(), bad3, ident),
              "MF4_AMENDED_INPUTS_INCOMPLETE"))

    if store_here:
        # MA7 mutations against a fixture COPY of the committed
        # surfaces (ARCH.REPO retargeted, restored in finally)
        fx = tempfile.mkdtemp(prefix="fpkat_mf4fx_")
        for rel in (ARCH.CAPSULE_REL, ARCH.RECEIPT_REL,
                    ARCH.ROWS_REL,
                    cap["catalog_binding"]["snapshot_path"],
                    cap["catalog_binding"]["receipt_path"],
                    cap["maturity_bounds"]["source"]):
            src = os.path.join(repo_real, *rel.split("/"))
            dst = os.path.join(fx, *rel.split("/"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
        real_repo_const = ARCH.REPO
        try:
            ARCH.REPO = fx
            ok_base = True
            try:
                FPM.build_mf4_calibration_inputs(fx)
            except BaseException as exc:
                ok_base, why_b = False, str(exc)[:120]
            check("MA7a the untouched fixture copy still verifies",
                  ok_base, "" if ok_base else why_b)
            rp = os.path.join(fx, *ARCH.ROWS_REL.split("/"))
            raw = open(rp, "rb").read()
            open(rp, "wb").write(raw + b" ")
            check("MA7b a doctored rows file refuses at the digest",
                  refuses(lambda:
                          FPM.build_mf4_calibration_inputs(fx),
                          "MF4_ARCHIVE") or
                  refuses(lambda:
                          FPM.build_mf4_calibration_inputs(fx),
                          "MF4_INPUTS_ROWS_DIGEST_DIVERGENT"))
            open(rp, "wb").write(raw)
            sp = os.path.join(
                fx, *cap["catalog_binding"]["snapshot_path"]
                .split("/"))
            sraw = open(sp, "rb").read()
            open(sp, "wb").write(sraw + b" ")
            # codex 0411Z item 4: EITHER valid refusal layer -- the
            # full capsule reconstruction catches it first where it
            # runs; the producer-local digest is the later gate
            check("MA7c doctored snapshot bytes refuse at a valid "
                  "layer",
                  refuses(lambda:
                          FPM.build_mf4_calibration_inputs(fx),
                          "MF4_ARCHIVE_CAPSULE_RECONSTRUCTION") or
                  refuses(lambda:
                          FPM.build_mf4_calibration_inputs(fx),
                          "MF4_INPUTS_SNAPSHOT_DIGEST_DIVERGENT"))
        finally:
            ARCH.REPO = real_repo_const

    print()
    if FAILS:
        print(f"FEED PRODUCER LOCK-TEST FAILURES ({len(FAILS)}): "
              f"{FAILS}")
        return 1
    print("CALIBRATION FEED PRODUCER: ALL LOCK TESTS PASS "
          "(fixtures minted by the real transform; writes confined "
          "to temp dirs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
