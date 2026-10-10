# DAILY-MONITOR CONTINUITY RENEWAL — owner authorization (2026-08-22)

- **authority**: asylum (owner), in-session, 2026-08-22T19:24Z (live UTC clock read)
- **quote (verbatim)**: "renew through window close, run the tier-s benchmark and
  tier-c sizing"
- **quote_sha256**: `7caab14d2b7379609c096f2a52b25d29c8c92209dbd2217111e513b5a8acb270`
- **scope of THIS artifact**: the renewal clause only ("renew through window close");
  the benchmark clause is a work directive executed separately.

## Term

Daily-monitor data continuity is authorized **through the WINDOW-2 CLOSE** =
`evaluation_end + H_max` (the maturity-tail end). The concrete calendar date BINDS
when the PRESTART barrier fixes `evaluation_start` (prereg v0.3 §1: end = start +
131 d, tail = end + 7 d). Until PRESTART, the term reads: continuous, with the
window-2 close as the registered endpoint. This supersedes the previously scheduled
HONEST EXPIRY (coverage through scored day 2026-08-23) — superseded by explicit owner
renewal, not by silent extension.

## What this satisfies

- **M-F4 admission gate** (annex_mf4 §"Admission gate"): "requires daily-monitor
  continuity authority at freeze" — THIS artifact is that authority; the freeze-time
  admission check cites this file + quote sha. `NOT_ADMITTED_DATA_CONTINUITY` does
  NOT fire while this authorization stands.
- The M-F4 calibration/accrual risk-series feed continuity assumption.

## What this does NOT authorize

No PRESTART, no fire, no prospective-value access, no publication, no method claim.
Λ_geo remains INCONCLUSIVE. If the owner ends the project or countermands, this
artifact is superseded by that later instruction (append-only; never edited in
place).
