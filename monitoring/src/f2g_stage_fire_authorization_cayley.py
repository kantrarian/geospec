#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FIRE-AUTHORIZATION STAGING (cayley) -- builds the committed record the
v2.1 driver requires, binding the codex PASS note to the exact reviewed
source blobs. Run AFTER codex's v2.1 PASS note is copied into the repo
and committed. Emits docs/f2g_sealed_run_fire_authorization.json for
review + manual commit; every sha is machine-computed from HEAD blobs
(nothing hand-typed).
Usage: stage.py <repo> <note_repo_relpath> <codex_pass_ref>
Then: commit the record; run the live refusal KATs (wrong ref ->
AUTH_REF_MISMATCH; --resume -> RESERVATION_MISSING); then fire:
  python monitoring/src/f2g_sealed_run_driver_cayley.py . \
      --codex-pass <codex_pass_ref>
"""
import hashlib
import json
import os
import subprocess
import sys

DRIVER_REL = "monitoring/src/f2g_sealed_run_driver_cayley.py"
INSTRUMENT_REL = "monitoring/src/f2g_sealed_run_instrument_cayley.py"
VERIFIER_REL = "monitoring/src/f2g_sealed_run_result_verifier_cayley.py"
MANIFEST_REL = "docs/f2g_sealed_run_input_manifest.json"
AUTH_PATH = "docs/f2g_sealed_run_fire_authorization.json"
SEAL_QUOTE_SHA = ("bb94a28bec0060d7d45b799f17536c499539f324da0f58ac3a1edcf"
                  "df594a7e4")


def head_blob_sha(repo, rel):
    b = subprocess.check_output(["git", "cat-file", "blob", f"HEAD:{rel}"],
                                cwd=repo)
    return hashlib.sha256(b).hexdigest()


def main(repo, note_rel, ref):
    repo = os.path.abspath(repo)
    # the note must already be committed AND byte-equal on disk
    note_sha = head_blob_sha(repo, note_rel)
    disk = open(os.path.join(repo, note_rel), "rb").read() \
        .replace(b"\r\n", b"\n")
    assert hashlib.sha256(disk).hexdigest() == note_sha, \
        "note disk bytes differ from HEAD blob"
    auth = {"schema": "f2g-sealed-run-fire-authorization-v2",
            "codex_pass_note": {"path": note_rel,
                                "blob_sha256": note_sha,
                                "ref": ref},
            "driver_blob_sha256": head_blob_sha(repo, DRIVER_REL),
            "instrument_blob_sha256": head_blob_sha(repo, INSTRUMENT_REL),
            "verifier_blob_sha256": head_blob_sha(repo, VERIFIER_REL),
            "manifest_sha256": head_blob_sha(repo, MANIFEST_REL),
            "seal_quote_sha256": SEAL_QUOTE_SHA}
    out = os.path.join(repo, AUTH_PATH)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(auth, f, indent=1, sort_keys=True)
        f.write("\n")
    print(json.dumps(auth, indent=1, sort_keys=True))
    print(f"\nwrote {AUTH_PATH} -- review, commit, run refusal KATs, "
          "then fire")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
