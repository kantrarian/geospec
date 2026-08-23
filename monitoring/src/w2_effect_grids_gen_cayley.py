#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-2 REGISTERED EFFECT GRIDS transcription (cayley) -- the
machine-readable form of the FROZEN annex grid definitions, for the
bound geometry capsule's `effect_grids` field and the Tier selector's
`registered_grid_index` semantics.

SOURCES (frozen texts, transcribed verbatim -- this artifact ADDS
nothing and each family's count is asserted against the annex):
- B2A  (annex_b2a "Registered grid (3 points)"):
        m in {1,2,3}, ties ascending.
- B2B  (annex_b2b "Effect grid (power)"): the three swap classes PLUS
        two churn-robustness classes (m=2 swap) x dropout {10%, 25%};
        swaps ascending then churn ascending.
- B1B  (annex_b1b -> B1A "Registered grid (48 points)"):
        delta_lat {0.3,0.6,1.2,2.4} x k {3,10,25,50} x n_e {3,10,33},
        tie-break (delta_lat, k, n_e) ascending; PLUS the two
        specificity gain points x{3,10} (annex_b1b) appended
        ascending at indices 48, 49.
- B3A  (annex_b3a "Registered grid (24 points)"):
        delta_lat {0.3,0.6,1.2,2.4} x n_cross {3,8} x k {10,25,50},
        tie-break (delta_lat, n_cross, k) ascending.

registered_grid_index = position in each family's list below (the
selector amendment's tie-break coordinate). Deterministic: same bytes
every run. Transcription verification (annex text vs this JSON) is a
codex/grassmann review item -- routed, not self-cleared. Opens no
window-2 value.
"""
import hashlib
import itertools
import json
import os

OUT_REL = os.path.join("docs", "f2g_window2_execution",
                       "effect_grids_w2_v1.json")


def build():
    b2a = [{"m": m} for m in (1, 2, 3)]
    b2b = [{"m": m} for m in (1, 2, 3)] + \
          [{"m": 2, "dropout": d} for d in (0.1, 0.25)]
    b1b = [{"delta_lat": dl, "k": k, "n_e": ne}
           for dl, k, ne in itertools.product(
               (0.3, 0.6, 1.2, 2.4), (3, 10, 25, 50), (3, 10, 33))]
    b1b += [{"gain": 3.0}, {"gain": 10.0}]
    b3a = [{"delta_lat": dl, "n_cross": nc, "k": k}
           for dl, nc, k in itertools.product(
               (0.3, 0.6, 1.2, 2.4), (3, 8), (10, 25, 50))]
    assert len(b2a) == 3 and len(b2b) == 5
    assert len(b1b) == 50 and len(b3a) == 24
    assert sum(1 for p in b1b if "gain" not in p) == 48
    grids = {"B2A": b2a, "B2B": b2b, "B1B": b1b, "B3A": b3a}
    assert sum(len(g) for g in grids.values()) == 82
    return {
        "schema": "f2g-w2-effect-grids-v1",
        "grids": grids,
        "counts": {"B2A": 3, "B2B": 5, "B1B_detection": 48,
                   "B1B_specificity": 2, "B3A": 24, "total": 82},
        "provenance": {
            "sources": ["docs/f2g_window2_freeze/annex_b2b.md",
                        "docs/f2g_phase_b_power_annex_b2a.md",
                        "docs/f2g_phase_b_power_annex_b1a.md",
                        "docs/f2g_window2_freeze/annex_b1b.md",
                        "docs/f2g_phase_b_power_annex_b3a.md"],
            "producer": "monitoring/src/w2_effect_grids_gen_cayley.py",
            "note": "verbatim transcription of the frozen grid "
                    "definitions; registered_grid_index = list "
                    "position; k=3 B1B cells are the registered "
                    "DILUTION probes (annex B1A disclosure); "
                    "transcription verification is a review item",
            "claim_ceiling": "registration only; no power value; "
                             "Lambda_geo INCONCLUSIVE"}}


def main():
    repo = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    body = json.dumps(build(), indent=1, sort_keys=True) + "\n"
    out = os.path.join(repo, OUT_REL)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
    print(f"wrote {OUT_REL}")
    print("artifact sha256:",
          hashlib.sha256(body.encode()).hexdigest())


if __name__ == "__main__":
    main()
