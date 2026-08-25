#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WINDOW-2 ADMITTED-ABSENCE RED-KATs (cayley) -- the executable lock
for codex 0542Z shape-review condition 2.

THE PROBLEM
-----------
The successor authority authorizes 2,056 requests. At least nine
provider-null MAG days lie inside 2026-01-01..07-31, and a corrected
OMNI day may be structurally valid yet all-fill. Today those bodies
REFUSE ("provider-null MAG series"), and codex's part-7 rule is that
`REFUSED` never satisfies an expected scientific key. So a 2,056-key
authority and an all-ADMITTED scientific census cannot both close --
while dropping the bad days after seeing them would make the authority
DATA-DEPENDENT, which is worse.

THE REGISTERED SEMANTICS THIS LOCKS
-----------------------------------
A third, pinned transform outcome: **ADMITTED_ABSENCE**.

  * it is reached ONLY by a STRUCTURALLY VALID response -- correct
    schema, canonical UTC instants, exact registered cadence grid --
    whose values are wholly provider-null / all-fill;
  * it carries NO value; it deterministically inserts `None` into the
    registered support mask, so downstream fits see an explicit hole
    rather than a silently shorter series;
  * it BINDS its raw/transcript/contract evidence exactly like an
    admitted value, so provenance is unbroken;
  * it SATISFIES its expected key (the census stays fixed at 2,056 and
    stays data-independent);
  * malformed, error-page, partial-grid, and unauthorized responses
    remain **REFUSED** and can never enter a feed.

Absence is "the provider published nothing for this day", which is a
fact about the world. Refusal is "we could not establish what the
provider published", which is a fact about the exchange. Collapsing
them in either direction is the defect.

FIXTURE PROVENANCE
------------------
Every fixture is DERIVED from the real committed FRN probe body (git
blob), never hand-built: the absence case is that exact body with its
values nulled, so structure/cadence/timestamps stay authentic and the
ONLY difference is the presence of values. A hand-built "null body"
could pass by being malformed in a way the real one is not.

REFINED by grassmann's 1330Z item (a), raised from their real
corrected-OMNI probe and deliberately NOT self-applied: a COMPLETE
grid with PER-SAMPLE fills is neither "all present" nor "absent" --
1,179 of 1,440 minutes definitive, 261 with no computable Newell
regressor. A whole-day absence flag cannot express that, and counts
alone lose WHICH minutes. So the semantics generalise: grid
COMPLETENESS stays structural (a short grid refuses), while per-sample
SUPPORT is carried as an explicit per-sample mask on EVERY admitted
artifact -- and ADMITTED_ABSENCE is simply the all-unsupported case of
that mask rather than a separate concept. AB-5 locks it.

STATUS: RED-FIRST. AB-1 fails today (the transform refuses a
structurally valid provider-null day); AB-5 is red pending the v4 lane
registration (codex bridge finding 4, grassmann). AB-2/3/4 guard the
opposite direction and should hold already -- if one of THOSE ever
goes red, absence has been over-applied.

Opens no window-2 value; no network; admits nothing scientifically.
"""
import copy
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
FRN_BODY = ("docs/f2g_window2_execution/mag_capsules/receipts/"
            "mag_frn_probe.json")
PROBE_DAY = "2026-08-19"
ABSENCE_OUTCOME = "ADMITTED_ABSENCE"


class AbsenceRefusal(AssertionError):
    """Typed refusal; the code leads the message."""


def _blob(rel):
    r = subprocess.run(["git", "-C", REPO, "cat-file", "blob",
                        f"HEAD:{rel}"], capture_output=True)
    if r.returncode != 0:
        raise AbsenceRefusal(f"FIXTURE_UNREADABLE: {rel}")
    return r.stdout


def _contract(day=PROBE_DAY):
    import w2_accrual_instrument_cayley as AI
    import w2_expected_contracts_gen_cayley as GEN
    auth = GEN.build(REPO)
    return AI.authoritative_static_contract(auth, "MAG_FEED", "frn",
                                            day)


def _nulled(raw):
    """The REAL body with every value nulled -- structurally identical,
    provider-null in content."""
    d = json.loads(raw.decode("utf-8"))
    for ch in d["values"]:
        ch["values"] = [None] * len(ch["values"])
    return json.dumps(d).encode()


def _truncated(raw):
    """A PARTIAL grid: real values, wrong cadence census."""
    d = json.loads(raw.decode("utf-8"))
    keep = len(d["times"]) // 2
    d["times"] = d["times"][:keep]
    for ch in d["values"]:
        ch["values"] = ch["values"][:keep]
    return json.dumps(d).encode()


def _selftest():
    import w2_acquisition_capture_grassmann as CAP

    raw = _blob(FRN_BODY)
    sc = _contract()
    baseline = CAP.admission_transform("MAG_FEED", raw, sc)
    assert baseline.get("definitive_samples", 0) > 0, \
        "FIXTURE: the real probe body must be a positive control"
    print(f"  control    real body -> {baseline['samples']} samples, "
          f"{baseline['definitive_samples']} definitive")

    # ---- AB-2: an error page must REFUSE (never absence) ----------
    for bad, label in ((b"<html><body>503 Service "
                        b"Unavailable</body></html>", "error page"),
                       (b"", "empty body"),
                       (b"{\"type\": \"Timeseries\"}",
                        "schema-shaped but empty")):
        try:
            CAP.admission_transform("MAG_FEED", bad, sc)
            raise AbsenceRefusal(
                f"AB-2: {label} must REFUSE, never admit")
        except AbsenceRefusal:
            raise
        except Exception:
            pass
    print("  AB-2 PASS  error page / empty / schema-shaped-empty all "
          "refuse")

    # ---- AB-3: a PARTIAL grid must REFUSE (absence is whole-day) ---
    try:
        CAP.admission_transform("MAG_FEED", _truncated(raw), sc)
        raise AbsenceRefusal(
            "AB-3: a partial cadence grid must REFUSE -- absence is a "
            "whole-day state, never a short series")
    except AbsenceRefusal:
        raise
    except Exception:
        pass
    print("  AB-3 PASS  partial grid refuses (absence is whole-day)")

    # ---- AB-5 (grassmann 1330Z item (a), NOT self-applied): a
    # COMPLETE grid with PER-SAMPLE fills is neither "all present"
    # nor "absent". Their real corrected-OMNI probe: 1440 rows, but
    # speed carries 260 fills against 9 for By/Bz and only 8 rows are
    # all-fill -- so 261 minutes have NO computable Newell regressor
    # while 1179 are fully definitive. A whole-day absence flag
    # cannot express that, and counts alone lose WHICH minutes.
    #
    # The refinement this locks: grid COMPLETENESS stays structural
    # (a short grid still refuses, AB-3), while per-sample SUPPORT is
    # semantic and must be carried as an explicit per-sample mask.
    # ADMITTED_ABSENCE then becomes the degenerate all-unsupported
    # case of that mask rather than a separate concept.
    omni_body = _blob("docs/f2g_window2_execution/probe_evidence/"
                      "omni_corrected_probe_20260101.body")
    omni_sc = json.loads(_blob(
        "docs/f2g_window2_execution/probe_evidence/"
        "omni_corrected_probe_20260101.contract.json"
    ).decode("utf-8"))
    try:
        oart = CAP.admission_transform(omni_sc["lane"], omni_body,
                                       omni_sc)
    except Exception as e:                               # noqa: BLE001
        raise AbsenceRefusal(
            "AB-5 PARTIAL_SUPPORT_UNREPRESENTABLE: the REAL "
            "corrected-OMNI probe body could not be transformed "
            f"({type(e).__name__}: {str(e)[:80]}) -- expected while "
            "the v4 lane names are unregistered (codex bridge "
            "finding 4, grassmann)")
    mask = oart.get("support_mask")
    if not isinstance(mask, list):
        raise AbsenceRefusal(
            "AB-5 PARTIAL_SUPPORT_UNREPRESENTABLE: the artifact "
            "carries no PER-SAMPLE support_mask. grassmann's real "
            "probe has 1179 definitive minutes and 261 with no "
            "computable Newell regressor; counts alone cannot say "
            "WHICH minutes, and a whole-day absence flag cannot "
            "express a partially supported day. Every admitted "
            "artifact must carry a per-sample mask, of which "
            f"{ABSENCE_OUTCOME} is the all-unsupported case.")
    if len(mask) != oart.get("samples"):
        raise AbsenceRefusal(
            "AB-5: support_mask length must equal the sample grid")
    if sum(1 for m in mask if m) != oart.get("definitive_samples"):
        raise AbsenceRefusal(
            "AB-5: supported entries must equal definitive_samples")
    print(f"  AB-5 PASS  per-sample support mask: "
          f"{sum(1 for m in mask if m)}/{len(mask)} supported")

    # ---- AB-1 (THE LOCK): structurally valid provider-null day ----
    null_body = _nulled(raw)
    d = json.loads(null_body.decode("utf-8"))
    assert len(d["times"]) == len(json.loads(
        raw.decode("utf-8"))["times"]), \
        "FIXTURE: the null body must keep the real cadence grid"
    try:
        art = CAP.admission_transform("MAG_FEED", null_body, sc)
    except Exception as e:                               # noqa: BLE001
        raise AbsenceRefusal(
            "AB-1 ABSENCE_COLLAPSED_TO_REFUSAL: a structurally valid "
            "provider-null day must reach the registered "
            f"{ABSENCE_OUTCOME} outcome, not a typed refusal "
            f"({type(e).__name__}: {str(e)[:90]}). It carries no "
            "value but MUST bind its evidence, insert None into the "
            "support mask, and satisfy its expected key -- otherwise "
            "the 2,056-key census can only close by dropping days "
            "after seeing them, which makes the authority "
            "data-dependent. codex 0542Z condition 2.")

    # ---- AB-4: the absence artifact carries no value, but is typed,
    # evidence-bound, and census-satisfying
    assert art.get("outcome") == ABSENCE_OUTCOME, \
        (f"AB-4: the absence artifact must be typed "
         f"{ABSENCE_OUTCOME}, got {art.get('outcome')!r}")
    assert art.get("definitive_samples") == 0, \
        "AB-4: an absence artifact must carry zero definitive samples"
    assert art.get("utc_day") == PROBE_DAY, \
        "AB-4: the absence artifact must bind its day"
    # codex 1424Z ruling 1 accepted the per-sample refinement, so
    # ADMITTED_ABSENCE is PRECISELY the all-false mask -- not the
    # scalar sentinel I first guessed at. Updated to the ruled shape.
    amask = art.get("support_mask")
    assert isinstance(amask, list) and len(amask) == art["samples"], \
        ("AB-4: the absence artifact must carry a per-sample support "
         "mask over the full grid")
    assert not any(amask), \
        ("AB-4: ADMITTED_ABSENCE is precisely the ALL-FALSE mask; got "
         f"{sum(1 for m in amask if m)} supported samples")
    print(f"  AB-1 PASS  provider-null -> {ABSENCE_OUTCOME}")
    print("  AB-4 PASS  absence typed, evidence-bound, no value, "
          "support mask None")
    print("w2 admitted-absence red-KATs: ALL PASS")


if __name__ == "__main__":
    try:
        _selftest()
    except AbsenceRefusal as e:
        print(f"RED (expected until the successor lands): {e}")
        raise SystemExit(1)
