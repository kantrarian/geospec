#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""W2 POWER SEED-AUTHORITY record generator (cayley) -- codex w2r1
pre-fire ruling 2026-08-31T15:07Z item 3 (CRITICAL): the power seed
authority is NOT a new choice. It is the already-frozen Amendment-2
root, and the geometry lane must resolve it from a manifest-pinned
record instead of accepting any 64-hex caller value.

This generator emits ONE closed record binding:

1. the registered ROOT digest, and its source identity {commit, path,
   LF blob sha256} -- read from the FROZEN commit, never from a
   working tree, with the frozen commit required to be an ancestor of
   the carrier commit;
2. the registered draw GRAMMAR, bound by the identity of the code that
   implements it: the design-pinned engine module (whose digest must
   equal the byte-pin manifest's registered value at the execution
   manifest's declared design commit -- a lookalike at the same path
   refuses) and the registered harness entrypoint;
3. the registered FAMILY tokens, read from the harness's own GRAPH
   constant -- never a literal list that could drift from it;
4. the REPLICATE indexing rule, bound as executable evidence: the
   master substream seed and the first replicate seeds per family,
   computed through the REGISTERED function `rep_seed_registered`
   itself (never a re-implementation -- a convenient proxy that agreed
   with nothing would be a false pass);
5. the non-claim ceiling.

Rebuild-or-refuse: a verifier re-running build() at the same commit
must reproduce these bytes exactly. Any root, grammar-module, harness,
or family drift refuses typed. Opens no window-2 value; no network; no
fit; admits nothing; draws no replicate. Lambda_geo INCONCLUSIVE.
"""
import hashlib
import json
import os
import re
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import w2_power_harness_cayley as PH  # noqa: E402

OUT_REL = ("docs/f2g_window2_execution/"
           "power_seed_authority_w2_v1.json")
SCHEMA = "f2g-w2-power-seed-authority-v1"

# The registered Amendment-2 root (codex 1507Z item 3). Both the
# digest and the commit that froze it are registered constants: the
# record proves them, it never discovers them.
REGISTERED_ROOT = ("58b513b6c30b70c8014510788da9d7d819ce8971"
                   "ca59b7dfdc11c57a1664586f")
ROOT_COMMIT = "337571c81df8d2a8242867fda69406bd67de9446"
ROOT_REL = "docs/f2g_phase_b_prereg_amendment2_DRAFT.md"

GRAMMAR_MODULE_REL = "monitoring/src/d2_f2g_phase_b_stats.py"
GRAMMAR_DESIGN_PIN = "engine_b2a_b3a"
HARNESS_REL = "monitoring/src/w2_power_harness_cayley.py"
MANIFEST_REL = "docs/f2g_window2_execution/execution_manifest.json"
DESIGN_MANIFEST_REL = "docs/f2g_window2_freeze/byte_pin_manifest.json"

# how many replicate seeds the record binds as executable evidence
KAT_REPLICATES = 3


class SeedAuthorityRefusal(ValueError):
    pass


def _refuse(detail):
    raise SeedAuthorityRefusal(f"POWER_SEED_AUTHORITY_REFUSED: {detail}")


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _lf_sha(b):
    """LF-normalized digest: the registered identity of a text
    artifact, so a CRLF checkout can never steer it."""
    return _sha(b.replace(b"\r\n", b"\n"))


def _resolve_commit(repo, commit):
    r = subprocess.run(["git", "-C", repo, "rev-parse",
                        f"{commit}^{{commit}}"],
                       capture_output=True, text=True)
    c = r.stdout.strip()
    if r.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", c):
        _refuse(f"unresolvable carrier commit {commit!r}")
    return c


def _blob(repo, rel, commit):
    r = subprocess.run(["git", "-C", repo, "cat-file", "blob",
                        f"{commit}:{rel}"], capture_output=True)
    if r.returncode != 0 or not r.stdout:
        _refuse(f"blob unreadable at {commit[:12]}: {rel}")
    return r.stdout


def _is_ancestor(repo, a, b):
    return subprocess.run(
        ["git", "-C", repo, "merge-base", "--is-ancestor", a, b],
        capture_output=True).returncode == 0


def _design_pin_digest(repo, commit, pin_name):
    """The registered digest of a design pin, resolved through the
    EXECUTION manifest's declared design commit -- content identity
    alone is not derivation provenance, so the grammar module is
    checked against the registry that froze it."""
    man = json.loads(_blob(repo, MANIFEST_REL, commit).decode("utf-8"))
    dmc = man.get("design_manifest_commit")
    if not (isinstance(dmc, str)
            and re.fullmatch(r"[0-9a-f]{40}", dmc)):
        _refuse("execution manifest declares no design_manifest_commit")
    design = json.loads(
        _blob(repo, DESIGN_MANIFEST_REL, dmc).decode("utf-8"))
    entry = (design.get("pins") or {}).get(pin_name)
    if not isinstance(entry, dict):
        _refuse(f"design pin {pin_name!r} absent at {dmc[:12]}")
    return dmc, entry


def build(repo, *, commit="HEAD", loaders=None):
    """`commit` is resolved ONCE and carries every read. loaders is a
    KAT-only seam ({rel: bytes}) for the refusal doctors; production
    passes None and reads committed bytes only."""
    carrier = _resolve_commit(repo, commit)

    def raw(rel, at=None):
        if loaders is not None and rel in loaders:
            return loaders[rel]
        return _blob(repo, rel, at or carrier)

    # --- 1. the root, proved at the FROZEN commit -----------------
    if not _is_ancestor(repo, ROOT_COMMIT, carrier):
        _refuse(f"the registered root commit {ROOT_COMMIT[:12]} is "
                f"not an ancestor of the carrier {carrier[:12]} -- "
                "the frozen amendment must already be history")
    root_b = raw(ROOT_REL, at=ROOT_COMMIT)
    root_lf = _lf_sha(root_b)
    if root_lf != REGISTERED_ROOT:
        _refuse("the amendment bytes at the registered root commit do "
                f"not carry the registered digest ({root_lf[:12]} != "
                f"{REGISTERED_ROOT[:12]}) -- the seed authority is "
                "never taken from bytes that fail this check")
    # disclosure, not a gate: whether the live path still equals the
    # frozen root (a later amendment would NOT move this authority)
    live_b = raw(ROOT_REL)
    live_lf = _lf_sha(live_b)

    # --- 2. the grammar, bound by CODE identity -------------------
    gram_b = raw(GRAMMAR_MODULE_REL)
    gram_lf = _lf_sha(gram_b)
    dmc, dpin = _design_pin_digest(repo, carrier, GRAMMAR_DESIGN_PIN)
    if dpin.get("path") != GRAMMAR_MODULE_REL:
        _refuse(f"design pin {GRAMMAR_DESIGN_PIN!r} names "
                f"{dpin.get('path')!r}, not the grammar module")
    if dpin.get("blob_sha256") != gram_lf:
        _refuse("the grammar module at the carrier is not the "
                "design-pinned engine (a lookalike at the registered "
                f"path: {gram_lf[:12]} != "
                f"{str(dpin.get('blob_sha256'))[:12]})")
    harness_b = raw(HARNESS_REL)

    # --- 3. families, from the registered constant ----------------
    families = list(PH.GRAPH)
    if not families or sorted(set(families)) != sorted(families):
        _refuse(f"registered family tokens are malformed: {families}")

    # --- 4. replicate indexing, as EXECUTABLE evidence ------------
    # computed through the REGISTERED entrypoint, never a local
    # re-implementation of the grammar
    import d2_f2g_phase_b_stats as _pb
    kat = {}
    for fam in families:
        master = _pb.derive_substream_seed(
            REGISTERED_ROOT, fam, "full", "power")
        reps = [int(PH.rep_seed_registered(REGISTERED_ROOT, fam, r))
                for r in range(KAT_REPLICATES)]
        if len(set(reps)) != len(reps):
            _refuse(f"replicate seeds for {fam} are not distinct -- "
                    "the registered sequential draw is not behaving "
                    "as registered")
        kat[fam] = {"master_substream_seed": int(master),
                    "replicate_seeds": reps}
    if len({v["master_substream_seed"] for v in kat.values()}) != \
            len(families):
        _refuse("master substream seeds are not distinct per family "
                "-- the family token is not entering the derivation")

    return {
        "schema": SCHEMA,
        "state": "REGISTERED",
        "ruling_basis": "codex w2r1 pre-fire ruling 2026-08-31T15:07Z "
                        "item 3 (CRITICAL): the seed authority is the "
                        "already-frozen Amendment-2 root, not a new "
                        "choice; the geometry lane must resolve it "
                        "from this manifest-pinned record rather than "
                        "accept any 64-hex caller value",
        "seed_authority_sha256": REGISTERED_ROOT,
        "root_source": {
            "path": ROOT_REL,
            "commit": ROOT_COMMIT,
            "blob_sha256_lf": root_lf,
            "blob_sha256_raw": _sha(root_b),
            "bytes": len(root_b),
            "identity_rule": "LF-normalized sha256 of the committed "
                             "blob at the registered root commit; the "
                             "frozen commit is required to be an "
                             "ancestor of the carrier",
            "live_path_matches_frozen_root": live_lf == root_lf,
            "live_path_blob_sha256_lf": live_lf},
        "grammar": {
            "purpose_token": "power",
            "fold_token": "full",
            "substream_rule": "derive_substream_seed(root, family, "
                              "'full', 'power') = int.from_bytes("
                              "sha256(f'{root}||{family}||{fold}||"
                              "{purpose}').digest()[:8], 'big')",
            "master_rule": "one master numpy PCG64 Generator per "
                           "(authority, family), seeded with that "
                           "substream seed",
            "replicate_rule": "replicate r's seed = the r-th "
                              "sequential int64 draw of that master: "
                              "master.integers(0, 2**63, size=r+1, "
                              "dtype=int64)[r] -- sequential, so "
                              "replicate r is only defined relative "
                              "to the same master stream",
            "engine_module": {
                "path": GRAMMAR_MODULE_REL,
                "blob_sha256_lf": gram_lf,
                "design_pin": GRAMMAR_DESIGN_PIN,
                "design_manifest_commit": dmc,
                "function": "derive_substream_seed"},
            "registered_entrypoint": {
                "path": HARNESS_REL,
                "blob_sha256_lf": _lf_sha(harness_b),
                "function": "rep_seed_registered"}},
        "families": families,
        "grammar_evidence": {
            "method": "master + first "
                      f"{KAT_REPLICATES} replicate seeds per family, "
                      "computed through the REGISTERED entrypoint at "
                      "the carrier commit (never re-implemented "
                      "here); a verifier rebuilding this record "
                      "recomputes them, so any drift in the root, the "
                      "engine module, or the entrypoint refuses",
            "replicates_bound": KAT_REPLICATES,
            "by_family": kat},
        "prohibitions": [
            "never mint a random seed",
            "never hash the current clock, the target commit, or the "
            "capsule to obtain a seed",
            "never accept a caller-supplied 64-hex seed authority: the "
            "geometry validator requires this record's exact root, "
            "resolved from its manifest pin"],
        "claim_ceiling": "seed-authority registration only; binds the "
                         "draw grammar and its root identity. It "
                         "draws no replicate, certifies nothing, "
                         "admits nothing, opens no window-2 value, "
                         "and is not a power result; Lambda_geo "
                         "INCONCLUSIVE"}


def main():
    repo = os.path.abspath(os.path.join(_HERE, "..", ".."))
    commit = "HEAD"
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        commit = args[0]
    body = json.dumps(build(repo, commit=commit),
                      indent=1, sort_keys=True) + "\n"
    out = os.path.join(repo, OUT_REL.replace("/", os.sep))
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
    print(f"wrote {OUT_REL}")
    print("record sha256:", _sha(body.encode()))


def _selftest():
    repo = os.path.abspath(os.path.join(_HERE, "..", ".."))
    rec = build(repo)
    assert rec["state"] == "REGISTERED"
    assert rec["seed_authority_sha256"] == REGISTERED_ROOT
    assert rec["root_source"]["blob_sha256_lf"] == REGISTERED_ROOT
    assert rec["families"] == list(PH.GRAPH)
    print("  control: REGISTERED at the frozen root "
          f"{REGISTERED_ROOT[:12]}.. ({len(rec['families'])} families, "
          f"{KAT_REPLICATES} replicate seeds each)")

    # determinism: two builds agree byte-for-byte
    if json.dumps(build(repo), sort_keys=True) != \
            json.dumps(rec, sort_keys=True):
        raise SystemExit("seed-authority record is not deterministic")
    print("  determinism: two builds agree byte-for-byte")

    # anti-vacuity: the bound evidence must actually depend on the
    # root and the family, or it proves nothing
    import d2_f2g_phase_b_stats as _pb
    alt = _pb.derive_substream_seed(
        "0" * 64, rec["families"][0], "full", "power")
    if alt == rec["grammar_evidence"]["by_family"][
            rec["families"][0]]["master_substream_seed"]:
        raise SystemExit("ANTI_VACUITY: a different root produced the "
                         "same substream seed")
    masters = {f: v["master_substream_seed"]
               for f, v in rec["grammar_evidence"]["by_family"].items()}
    if len(set(masters.values())) != len(masters):
        raise SystemExit("ANTI_VACUITY: family token does not steer "
                         "the derivation")
    print("  anti-vacuity: the bound seeds move with BOTH the root "
          "and the family token")

    def doctored(rel, mutate, why, at_root=False):
        base = _blob(repo, rel, ROOT_COMMIT if at_root
                     else _resolve_commit(repo, "HEAD"))
        try:
            build(repo, loaders={rel: mutate(base)})
            raise SystemExit("seed-authority doctor must refuse: " + why)
        except SeedAuthorityRefusal as ex:
            assert why in str(ex), (why, str(ex))

    # a doctored amendment root refuses (the loader seam overrides the
    # frozen-commit read too, so this constructs the exact precondition)
    doctored(ROOT_REL, lambda b: b + b"\ntampered\n",
             "do not carry the registered digest", at_root=True)
    print("  doctor: doctored amendment bytes REFUSE (registered "
          "digest gate)")
    # a lookalike grammar module at the registered path refuses
    doctored(GRAMMAR_MODULE_REL,
             lambda b: b.replace(b"def derive_substream_seed",
                                 b"def derive_substream_seed_", 1),
             "not the design-pinned engine")
    print("  doctor: a lookalike engine module at the registered path "
          "REFUSES (design-pin digest gate)")
    # an unresolvable carrier refuses before any read
    try:
        build(repo, commit="not-a-commit")
        raise SystemExit("unresolvable carrier must refuse")
    except SeedAuthorityRefusal as ex:
        assert "unresolvable carrier commit" in str(ex), str(ex)
    print("  doctor: an unresolvable carrier commit REFUSES before "
          "any read")
    print("w2_power_seed_authority selftest: ALL PASS")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
