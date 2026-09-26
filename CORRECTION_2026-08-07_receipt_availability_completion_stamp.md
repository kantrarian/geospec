# CORRECTION (dated, prospective): publication-receipt availability = the server COMPLETION stamp

- **date registered:** 2026-08-07 (cayley, under codex review `f5296dc`; prospective only, no retroactive edits)
- **applies to:** the R6 §1 publication-receipt correction (server-stamped hit-clock), schema v2, all days going forward. No existing scored outcome changes (the live record has 0 hits; kumamoto episode left-censored).

## The finding (codex, live reproduction)

GitHub Pages build records expose BOTH `created_at` (build start) and `updated_at` (completion): live daily build `1137391428` showed `created_at 11:08:05Z` vs `updated_at 11:08:33Z` (~29 s of build time). A build-START stamp is not proof the public artifacts were available.

## The corrected rule (schema `geospec-publication-receipt-v2`)

1. **Availability** (`availability_utc`) = the server record's **`updated_at`** for a **`built`, error-free** Pages build — the completion stamp — never `created_at`. (If a still-later independently observed server stamp is ever adopted, that requires a further dated correction.)
2. A receipt **binds its day** (`day` field inside the hashed payload) — a valid receipt transplanted to another day's slot is detectable and inadmissible.
3. `commit_sha` must be full 40-hex; the deployment must carry the **server API URL + id + status + both timestamps**; any producer fallback that cannot obtain the real server record **fails closed** (no receipt written; the day stays ceiling + hit-ineligible).
4. **Standing requires verification, not structure:** hit eligibility is conferred ONLY by a typed verified-receipt result minted after (a) re-hashing the recorded artifacts from independently loaded bytes (the receipt's own commit via git object storage) AND (b) reopening the named server record and matching id/status/commit/timestamp. An unverified receipt dict NEVER confers eligibility, whatever its fields claim.

Absence/failure at any step degrades to the R6 §1 conservative ceiling (23:59:59Z) + hit-ineligible — never earlier, never synthesized, never a crash of the live monitor.

## Prospective clarification (2026-08-07, rev-3 bars; codex `d76842a` finding #3)

Rule 2 above under-specified WHAT binds the day. Prospectively: **the day is bound by the
server-attested commit's canonical alarm artifact** — admission re-opens `docs/ensemble_latest.json`
at the receipt's commit (via the independent artifact loader) and requires its `date` field to equal
BOTH `receipt.day` AND the requested day. The receipt JSON's `day` field alone is mutable metadata and
never binds anything by itself: a receipt whose `day` is edited to match the requested day still fails
admission when the reopened carrier says otherwise. Receipts must attest the mandatory carriers
(`docs/ensemble_latest.json`, `docs/data.csv`) within an exact artifact allowlist; a receipt lacking a
mandatory carrier, or naming a path outside the allowlist (however loadable/hash-valid), is inadmissible.
This clarification is prospective; no existing scored outcome changes (the live record remains 0 hits).
