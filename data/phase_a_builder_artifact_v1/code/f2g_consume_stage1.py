"""fault2graph Phase A -- builder-consumption STAGE 1 (cayley).

Runs under the MIRRORED numeric lock (py3.11.9/numpy2.3.5/scipy1.17.1/obspy1.4.2,
codex consumer-environment ruling 25a6af75). For each packet matrix: ingest via
d2_f2g_graph_builder.ingest_matrix -- the pinned B9 boundary (pre-use
recompute=True through the REAL producer verify seam, post-use recompute=False)
-- against a local consume root whose raw objects are lazily cached from the
staged s4t v2 root (byte-verified per object on copy). Every consumed packet
file is checked against the canonical packet summary before use. Receipts are
appended per day (resume-safe); verified matrices+manifests are copied to a
local verified store for stage 2 (graph construction).

Read-only on all shares. No claims. Usage:
  python f2g_consume_stage1.py smoke   (istanbul 03-01 + socal 03-04 + turkey 04-10)
  python f2g_consume_stage1.py full
"""
import hashlib
import json
import os
import shutil
import sys
import time

SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "C:/geospec/monitoring/src")
import d2_f2g_graph_builder as B                                   # noqa: E402

PKT = "//192.168.50.1/s4t/geospec_f2g_phase_a"
V2ROOT = "//192.168.50.1/s4t/geospec_d2_campaign_v2/d2_campaign_v2_20260811"
CONSUME_ROOT = os.path.join(SCRATCH, "consume_root")
VERIFIED = os.path.join(SCRATCH, "verified_matrices")
RECEIPTS = os.path.join(SCRATCH, "consume_receipts.jsonl")
PACKET_SUMMARY_SHA = ("df1e37ec6ca95dab1b4b24cfc5e7e3603f8fffe0"
                      "a8c4f76e4cfd20fa7cadc15c")  # supersedes e502ee14 (A4 members added, prior members byte-unchanged; grassmann b3129ba5)
INPUT_MANIFEST_SHA = ("e7ea157f7ca9011e1ec68a7de76446bb1f7c3e53"
                      "124b12a8bbfc98a81b8b43f9")
SMOKE = [("istanbul_marmara", "2026-03-01"), ("socal_coachella", "2026-03-04"),
         ("turkey_kahramanmaras", "2026-04-10")]
SMOKE_ISTANBUL_SHA = ("10fc8be595cb58d99951ab2c0f91fbf1c69c7e1d"
                      "16a5da0973d9f6215a79c5b6")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def rd(p):
    with open(p, "rb") as fh:
        return fh.read()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    os.makedirs(os.path.join(CONSUME_ROOT, "raw_objects"), exist_ok=True)
    os.makedirs(VERIFIED, exist_ok=True)

    summary_bytes = rd(f"{PKT}/f2g_phase_a_packet.json")
    assert sha(summary_bytes) == PACKET_SUMMARY_SHA, "packet summary sha drift"
    summary = json.loads(summary_bytes.decode("utf-8"))
    sfiles = summary.get("files") or summary.get("artifacts") or {}

    def summary_check(rel, body):
        ent = sfiles.get(rel)
        if ent is None:
            return f"NOT_IN_SUMMARY:{rel}"
        want = ent["sha256"] if isinstance(ent, dict) else ent
        return None if sha(body) == want else f"SUMMARY_MISMATCH:{rel}"

    im = rd(os.path.join(CONSUME_ROOT, "input_manifest.json"))
    assert sha(im) == INPUT_MANIFEST_SHA, "consume-root input_manifest drift"
    root_objects = json.loads(im.decode("utf-8"))["objects"]
    # P16: recompute rebuilds the candidate UNIVERSE from the ROOT manifest --
    # the cache must therefore cover EVERY root object for the (carrier, day),
    # including stations the production run refused (their objects exist and
    # the frozen gate must be re-run over them), never just the result
    # manifest's eligible-slim records.
    objects_by_day = {}
    for o in root_objects:
        objects_by_day.setdefault((o["carrier_key"], o["scored_day"]),
                                  []).append(o)

    done = set()
    if os.path.exists(RECEIPTS):
        for line in rd(RECEIPTS).decode("utf-8").splitlines():
            r = json.loads(line)
            if r.get("ok"):
                done.add((r["carrier"], r["day"]))

    days = []
    if mode == "smoke":
        days = SMOKE
    else:
        for carrier in sorted(os.listdir(f"{PKT}/matrices_v2")):
            cdir = f"{PKT}/matrices_v2/{carrier}"
            for f in sorted(os.listdir(cdir)):
                if f.endswith(".manifest.json"):
                    days.append((carrier, f[:-len(".manifest.json")]))
    todo = [(c, d) for c, d in days if (c, d) not in done]
    print(f"[stage1:{mode}] {len(todo)} to ingest ({len(done)} already done)",
          flush=True)

    n_ok = n_fail = 0
    for carrier, day in todo:
        t0 = time.time()
        rec = {"carrier": carrier, "day": day, "ok": False}
        try:
            fp = f"{PKT}/matrices_v2/{carrier}/{day}.manifest.json"
            mp = f"{PKT}/matrices_v2/{carrier}/{day}.matrix.npy"
            man_b, mat_b = rd(fp), rd(mp)
            for rel, body in ((f"matrices_v2/{carrier}/{day}.manifest.json",
                               man_b),
                              (f"matrices_v2/{carrier}/{day}.matrix.npy",
                               mat_b)):
                err = summary_check(rel, body)
                if err:
                    raise RuntimeError(err)
            man = json.loads(man_b.decode("utf-8"))
            # lazy raw-object cache over the ROOT-manifest universe (P16)
            for o in objects_by_day.get((carrier, day), []):
                osha = o["sha256"]
                dst = os.path.join(CONSUME_ROOT, "raw_objects", osha + ".ms")
                if not os.path.exists(dst):
                    shutil.copyfile(f"{V2ROOT}/{o['relative_path']}", dst)
                    if sha(rd(dst)) != osha:
                        os.remove(dst)
                        raise RuntimeError(f"RAW_OBJECT_COPY_DRIFT:{osha}")
            # local staging of the pair (paths handed to the seam)
            lfp = os.path.join(VERIFIED, f"{carrier}__{day}.manifest.json")
            lmp = os.path.join(VERIFIED, f"{carrier}__{day}.matrix.npy")
            with open(lfp, "wb") as fh:
                fh.write(man_b)
            with open(lmp, "wb") as fh:
                fh.write(mat_b)
            # cross_host_consumer_v1 (codex ruling 82c31cf3): exact mode
            # first; fixed 1e-9 comparator only on pure DERIVATION_MISMATCH
            r, man2, prof = B.ingest_matrix_cross_host(CONSUME_ROOT, lmp, lfp)
            rec.update(ok=True, matrix_sha256=sha(mat_b),
                       n_stations=len(man2["station_ids"]),
                       status=man2["status"],
                       reason_codes=man2.get("reason_codes", [])[:6],
                       profile=prof["profile"], mode=prof["mode"],
                       observed_max_abs_delta=prof["observed_max_abs_delta"],
                       producer_environment_lock_digest=prof[
                           "producer_environment_lock_digest"],
                       consumer_environment_lock=prof[
                           "consumer_environment_lock"])
            n_ok += 1
        except Exception as exc:                                   # noqa: BLE001
            rec.update(error=f"{type(exc).__name__}: {exc}"[:300])
            n_fail += 1
        rec["elapsed_s"] = round(time.time() - t0, 1)
        with open(RECEIPTS, "ab") as fh:
            fh.write((json.dumps(rec, sort_keys=True) + "\n").encode("utf-8"))
        print(f"  {carrier} {day}: {'OK' if rec['ok'] else 'FAIL'} "
              f"({rec['elapsed_s']}s)"
              + (f" {rec.get('error', '')}" if not rec["ok"] else ""),
              flush=True)

    print(f"[stage1:{mode}] DONE ok={n_ok} fail={n_fail}", flush=True)
    if mode == "smoke":
        ist = f"{PKT}/matrices_v2/istanbul_marmara/2026-03-01.matrix.npy"
        got = sha(rd(ist))
        print(f"smoke istanbul matrix sha {'MATCHES' if got == SMOKE_ISTANBUL_SHA else 'MISMATCH'}: {got}",
              flush=True)
    sys.exit(1 if n_fail else 0)


main()
