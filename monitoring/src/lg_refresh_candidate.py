"""NGL input refresh policy + LG calibration planner: offline CANDIDATE (cayley 2026-11-09; codex e61e7603 P1-4, bc6c1542,
repair per codex 7ebb9c65).

OFF unless explicitly constructed. Production live_data.py is unchanged, and so is every scheduled run. Nothing here
fetches unless a caller passes a session; the tests pass a recording fake. This module proposes behaviour; it does
not activate any.

What it replaces (production live_data.NGLLiveAcquisition.download_station, read from the code at 3ef935d2):
- the refresh decision is "cached file younger than 24 h", a wall-clock test that races a 12:00Z daily cadence by
  seconds, so files were fetched on alternate runs (grassmann eecd7e4cf);
- a successful refresh that returns OLD observations is indistinguishable from new data (no data-end record);
- a FAILED IGS20 refresh does not fall back to the cached IGS20 file: the code falls through to the IGS14 tenv3 path,
  so a transient failure silently switches that station's frame, or drops it. (Code path; characterised by a test
  here, not observed on a host.)

Refresh policy (ngl-refresh-candidate-v2):
- keyed by the AUTHORIZED RUN DAY, not file age. Within one run a file gets up to MAX_ATTEMPTS requests, retried only
  for TRANSIENT failures (network errors, HTTP 408/425/429/5xx) with an injectable backoff, within a per-run budget
  of RETRY_BUDGET extra attempts across all files (then one attempt each, flagged); a validation failure is
  never retried (the same bytes would fail again). After a success the file is NOT_DUE for the rest of the run day; a
  file whose attempts all failed is NOT_DUE for the rest of THAT run, and a later authorized run of the same day (a
  different run id, e.g. a recovery run) may try it again. The run day and run id are the caller's.
- TLS: a session must declare verification positively (`verify` True or a nonblank CA-bundle path; codex d7af675d);
  None, 0, empty, False, other values or a missing setting are refused. make_verified_session() is the proposed wiring.
- catalog: the cached ngl_catalog.json is never fetched outside calibration (production refreshes on a 7-day wall
  clock); refresh_catalog(calibration=True) refreshes it on calibration runs only, validated (parseable, finite,
  in-range coordinates; not shrunk below CATALOG_MIN_RETAINED of the previous; every declared carrier station still
  present) and installed atomically; catalog_bound() checks before scoring that the current catalog bytes are the
  ones bound into the selected baseline's calibration selection.
- an HTTP 200 body is STAGED, never installed unvalidated (codex 7ebb9c65 P1). It is accepted only if every data row
  names the requested station, every epoch is finite, the production parser for that format accepts the requested
  window with finite, equal-length coordinate arrays, no window day the previous file covered is lost, and the data
  end does not move backwards. Only then is it written atomically (temp file + replace). When the previous bytes had
  no usable window either (a station returning from an outage), a structurally valid body is accepted and its record
  says the window is unusable: the rule protects existing coverage, it does not block a recovering station.
- every attempt leaves a record: outcome, failure class and typed cause (HTTP / NETWORK / VALIDATION / WRITE), HTTP
  status, content sha256 before/after, bytes, the source DATA END before/after, and the data age on the run day.
  Outcomes: REFRESHED_NEW_EPOCHS, REFRESHED_SAME_CONTENT, REFRESHED_CHANGED_NO_NEW_EPOCHS (reprocessed, same end),
  FAILED_KEPT_PREVIOUS (previous bytes untouched and used, with their true age), FAILED_NO_PREVIOUS, NOT_DUE.
  No path raises: validator and write errors become typed failure records.
- the frame never switches on failure: IGS14 is used only when no IGS20 file exists or it does not parse, exactly
  the production preference minus the failure fall-through.

Calibration planner (lg-calibration-plan-v1): freshness is judged from the ACTUAL last accepted epoch in the selected
dated baseline, never from a nominal cadence; it says when each region's baseline goes stale and refuses to treat a
retrospective file (generated after its calibrated_on) as a prospective one.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import live_data as LD
import lg_producer_candidate as P

VERSION = 'ngl-refresh-candidate-v3'
PLAN_VERSION = 'lg-calibration-plan-v1'
OUTCOMES = ('REFRESHED_NEW_EPOCHS', 'REFRESHED_SAME_CONTENT', 'REFRESHED_CHANGED_NO_NEW_EPOCHS',
            'FAILED_KEPT_PREVIOUS', 'FAILED_NO_PREVIOUS', 'NOT_DUE')
VALIDATION_CAUSES = ('NO_EPOCH_ROWS', 'FOREIGN_STATION_ROWS', 'LAYOUT_CONTRACT_MISMATCH', 'UNPARSEABLE_EPOCH', 'NONFINITE_EPOCH',
                     'NO_USABLE_WINDOW', 'NONFINITE_COORDINATES', 'INCONSISTENT_ARRAYS', 'COVERAGE_REDUCED',
                     'DATA_END_REGRESSED', 'VALIDATOR_ERROR')
ERROR_CHARS = 200
TRANSIENT_HTTP = frozenset({408, 425, 429, 500, 502, 503, 504})
MAX_ATTEMPTS = 3                       # per file per run, transient failures only
BACKOFF_SECONDS = (0, 5, 30)           # before attempt 1, 2, 3
RETRY_BUDGET = 10                      # extra attempts per RUN across all files; then one attempt each (outage bound)
CATALOG_MIN_RETAINED = 0.9             # a refreshed catalog keeps at least this share of the previous station count
FAILED = ('FAILED_KEPT_PREVIOUS', 'FAILED_NO_PREVIOUS')
# IGS20 .tenv layout contract, settled from BYTES, not from a README: grassmann's inventory of the 80 retained
# Tokyo/Kumamoto files (lg_tenv_layout_inventory_grassmann_20261109T140642Z.json, sha256 29b18201f4ad49d1...) found no
# header line and exactly 17 fields in every one of 634,889 rows ("legacy-reflon-17": 0 site, 1 YYMMMDD, 2 decimal
# year, 3 MJD, 4 week, 5 day, 6 reference longitude, 7/8/9 east/north/up m, ...), which is what live_data._parse_tenv
# reads. NGL's README_tenv describes a 16-field layout; a body in THAT layout would be misread by the production
# parser (columns shift by one), so a refresh refuses any IGS20 body whose data rows are not exactly 17 fields.
IGS20_TENV_LAYOUT = 'legacy-reflon-17'
IGS20_TENV_FIELDS = 17
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) GeoSpec/1.0'     # production's header, unchanged


def make_verified_session():
    """The proposed runner wiring's session: production's headers, TLS verification ON (production sets it off)."""
    import requests
    session = requests.Session()
    session.verify = True
    session.headers.update({'User-Agent': USER_AGENT, 'Accept': 'text/plain,*/*'})
    return session


def _transient(failure: Optional[str], status: Optional[int]) -> bool:
    return failure == 'NETWORK' or (failure == 'HTTP' and status in TRANSIENT_HTTP)


def parse_catalog(text: str):
    """(catalog | None, cause | None, detail): NGL llh text parsed exactly as production does (code, lat, lon from the
    first three columns of every line with >= 3 fields), validated: finite, |lat| <= 90, |lon| <= 360, non-empty."""
    catalog = {}
    for k, line in enumerate((text or '').strip().split('\n'), 1):
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            lat, lon = float(parts[1]), float(parts[2])
        except ValueError:
            return None, 'CATALOG_UNPARSEABLE', f'line {k}'
        if not (math.isfinite(lat) and math.isfinite(lon)) or abs(lat) > 90 or abs(lon) > 360:
            return None, 'CATALOG_INVALID_COORDINATES', f'line {k} ({parts[0]})'
        catalog[parts[0]] = (lat, lon)
    if not catalog:
        return None, 'CATALOG_EMPTY', 'no station lines'
    return catalog, None, f'{len(catalog)} stations'


def _sha(raw: Optional[bytes]) -> Optional[str]:
    return hashlib.sha256(raw).hexdigest() if raw is not None else None


def _rows(text: str):
    for line in text.replace('\r\n', '\n').split('\n'):
        if not line.strip() or line.startswith('#') or line.startswith('site'):
            continue
        yield line.split()


def epoch_scan(text: Optional[str], code: Optional[str] = None, width: Optional[int] = None) -> Tuple[Optional[str], Optional[str], int]:
    """(data end day | None, typed cause | None, rows). Decimal year in column 2, converted exactly as production does.
    With `code`, every data row must name that station; with `width`, every data row must have exactly that many
    fields (LAYOUT_CONTRACT_MISMATCH otherwise). Never raises."""
    if not text:
        return None, 'NO_EPOCH_ROWS', 0
    last, n = None, 0
    for parts in _rows(text):
        if code is not None and parts[0].upper() != code.upper():
            return None, 'FOREIGN_STATION_ROWS', n
        if width is not None and len(parts) != width:
            return None, 'LAYOUT_CONTRACT_MISMATCH', n
        if len(parts) < 3:
            return None, 'UNPARSEABLE_EPOCH', n
        try:
            decimal_year = float(parts[2])
        except ValueError:
            return None, 'UNPARSEABLE_EPOCH', n
        if not math.isfinite(decimal_year) or not 1 <= decimal_year < 10000:
            return None, 'NONFINITE_EPOCH', n
        year = int(decimal_year)
        try:
            day = (datetime(year, 1, 1) + timedelta(days=(decimal_year - year) * 365.25)).date()
        except (OverflowError, ValueError):
            return None, 'NONFINITE_EPOCH', n
        n += 1
        last = day if last is None or day > last else last
    if last is None:
        return None, 'NO_EPOCH_ROWS', 0
    return last.isoformat(), None, n


def data_end_day(text: Optional[str]) -> Optional[str]:
    """The last valid epoch day of a body, or None when it has none or any epoch is invalid."""
    return epoch_scan(text)[0]


def _bounded(exc: BaseException) -> str:
    """A bounded, single-line error: the class and the first characters of the message, nothing else."""
    text = ' '.join(str(exc).split())
    return f'{type(exc).__name__}: {text[:ERROR_CHARS]}'


class RefreshLedger:
    """file name -> the LATEST attempt record; persisted atomically. The full per-run records are returned to the
    caller (RefreshPolicyNGL.records) for the run's evidence."""

    def __init__(self, path):
        self.path = Path(path)
        self.entries: Dict[str, dict] = {}
        if self.path.exists():
            self.entries = json.loads(self.path.read_text(encoding='utf-8'))

    def get(self, name: str) -> Optional[dict]:
        return self.entries.get(name)

    def put(self, name: str, record: dict) -> Optional[str]:
        """Persist; on a write failure keep the in-memory entry, remove the temp file and return a bounded error."""
        self.entries[name] = record
        tmp = self.path.with_name(self.path.name + '.part')
        try:
            tmp.write_text(json.dumps(self.entries, indent=1, sort_keys=True), encoding='utf-8')
            os.replace(tmp, self.path)
            return None
        except OSError as exc:
            try:
                tmp.unlink()
            except OSError:
                pass
            return _bounded(exc)


def _station_ok(station) -> Optional[str]:
    """The typed cause when a parsed station's arrays are unusable, else None."""
    n = len(station.times)
    if any(len(a) != n for a in (station.east_mm, station.north_mm, station.up_mm)):
        return 'INCONSISTENT_ARRAYS'
    if not all(np.isfinite(np.asarray(a, dtype=float)).all() for a in (station.east_mm, station.north_mm, station.up_mm)):
        return 'NONFINITE_COORDINATES'
    if not (math.isfinite(float(station.lat)) and math.isfinite(float(station.lon))):
        return 'NONFINITE_COORDINATES'
    return None


class RefreshPolicyNGL(LD.NGLLiveAcquisition):
    """NGL acquisition with the run-day refresh policy. `session` must be given explicitly (a TLS-verifying
    requests.Session in a future activation; a recording fake in tests); `clock` returns the attempt instant."""

    def __init__(self, cache_dir, *, run_day: date, run_id: str, ledger_path, session, clock: Callable[[], str],
                 sleep: Callable[[float], None] = time.sleep, max_attempts: int = MAX_ATTEMPTS,
                 retry_budget: int = RETRY_BUDGET):
        super().__init__(Path(cache_dir))
        if type(run_day) is not date or not run_id:
            raise ValueError('an explicit run day (date) and run id are required')
        if session is None:
            # production's lazy default builds a requests.Session with verify=False; never fall back to it silently
            raise ValueError('an explicit session is required (production default disables TLS verification)')
        verify = getattr(session, 'verify', None)
        if verify is not True and not (isinstance(verify, str) and verify.strip()):
            raise ValueError('TLS verification must be True or an explicit CA bundle path (use make_verified_session)')
        if type(max_attempts) is not int or max_attempts < 1 or type(retry_budget) is not int or retry_budget < 0:
            raise ValueError('max_attempts must be a positive int and retry_budget a non-negative int')
        self.retries_left = retry_budget
        self.run_day, self.run_id, self.clock = run_day, str(run_id), clock
        self.sleep, self.max_attempts = sleep, max_attempts
        self.ledger = RefreshLedger(ledger_path)
        self._session = session
        self.records: List[dict] = []

    def _parse(self, parser, code, text, days_back, target_date):
        try:
            return parser(code, text, days_back, target_date), None
        except Exception as exc:
            return None, _bounded(exc)

    def _validate(self, code, text, parser, days_back, target_date, previous, previous_end, width=None) -> Tuple[Optional[str], Optional[str], str]:
        """(data end | None, cause | None, detail) for a STAGED body; nothing is written here."""
        end, cause, rows = epoch_scan(text, code, width)
        if cause:
            return None, cause, f'{rows} valid rows before the first refused one'
        station, err = self._parse(parser, code, text, days_back, target_date)
        if station is None:
            if previous is not None:            # the previous bytes had a usable window: this body would destroy it
                return None, 'NO_USABLE_WINDOW', err or 'the production parser accepts no usable window for this request'
            if previous_end is not None and end < previous_end:
                return None, 'DATA_END_REGRESSED', f'data end {end} < previous {previous_end}'
            return end, None, f'{rows} rows; no usable window for this request, and none before (window_usable false)'
        bad = _station_ok(station)
        if bad:
            return None, bad, f'{len(station.times)} window rows'
        if previous is not None:
            lost = sorted({t.date() for t in previous.times} - {t.date() for t in station.times})
            if lost:
                return None, 'COVERAGE_REDUCED', f'{len(lost)} previously covered window days lost, first {lost[0]}'
        if previous_end is not None and end < previous_end:                 # the previous FILE's end, not the window's
            return None, 'DATA_END_REGRESSED', f'data end {end} < previous {previous_end}'
        return end, None, f'{rows} rows, {len(station.times)} in the window'

    def _refresh(self, url: str, path: Path, code: str, parser, days_back, target_date, width=None) -> dict:
        prior = self.ledger.get(path.name)
        base = dict(version=VERSION, file=path.name, url=url, run_day=self.run_day.isoformat(), run_id=self.run_id)
        before = path.read_bytes() if path.exists() else None
        before_text = before.decode('utf-8', 'replace') if before is not None else None
        before_end = data_end_day(before_text)
        not_due = self._not_due(prior)
        if not_due:
            record = dict(base, outcome='NOT_DUE', reason=not_due, attempted_utc=None,
                          prior_outcome=prior.get('outcome'), prior_run_id=prior.get('run_id'), sha256=_sha(before),
                          data_end_day=before_end, data_age_days=self._age(before_end))
            self.records.append(record)
            return record
        status, text, failure, cause, detail, attempts = self._fetch_with_retries(url)
        vdetail = None
        after, after_end = before, before_end
        if failure is None:
            previous = self._parse(parser, code, before_text, days_back, target_date)[0] if before_text else None
            try:
                end, vcause, vdetail = self._validate(code, text, parser, days_back, target_date, previous, before_end,
                                                      width)
            except Exception as exc:                                 # a validator defect is a typed failure too
                end, vcause, vdetail = None, 'VALIDATOR_ERROR', _bounded(exc)
            if vcause:
                failure, cause, detail = 'VALIDATION', vcause, vdetail
            else:
                staged = text.encode('utf-8')
                tmp = path.with_name(path.name + '.part')
                try:
                    tmp.write_bytes(staged)
                    os.replace(tmp, path)
                    after, after_end = staged, end
                except OSError as exc:
                    failure, cause, detail = 'WRITE', type(exc).__name__, _bounded(exc)
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
        if failure is None:
            if before is not None and _sha(after) == _sha(before):
                outcome = 'REFRESHED_SAME_CONTENT'
            elif before_end is not None and after_end <= before_end:
                outcome = 'REFRESHED_CHANGED_NO_NEW_EPOCHS'
            else:
                outcome = 'REFRESHED_NEW_EPOCHS'
        else:
            outcome = 'FAILED_KEPT_PREVIOUS' if before is not None else 'FAILED_NO_PREVIOUS'
        record = dict(base, outcome=outcome, attempted_utc=self.clock(), http_status=status, failure_class=failure,
                      cause=cause, detail=detail, validation_detail=vdetail, attempts=attempts,
                      sha256_before=_sha(before),
                      sha256_after=_sha(after),
                      bytes_after=len(after) if after else 0, data_end_before=before_end, data_end_after=after_end,
                      data_age_days=self._age(after_end))
        ledger_error = self.ledger.put(path.name, record)
        if ledger_error:
            record['ledger_error'] = ledger_error      # the run keeps the record; a later run day is unaffected
        self.records.append(record)
        return record

    def _not_due(self, prior: Optional[dict]) -> Optional[str]:
        """Why this file is not due in this run, or None when it is."""
        if prior is None or prior.get('run_day') != self.run_day.isoformat():
            return None
        if prior.get('outcome') not in FAILED:
            return 'REFRESHED_THIS_RUN_DAY'
        if prior.get('run_id') == self.run_id:
            return 'ATTEMPTS_EXHAUSTED_THIS_RUN'
        return None                                     # a failed file: a later authorized run of the day may retry

    def _fetch_once(self, url: str):
        try:
            response = self.session.get(url, timeout=30)
            status = int(response.status_code)
        except Exception as exc:                                     # network / TLS / timeout: recorded, bounded
            return None, None, 'NETWORK', type(exc).__name__, _bounded(exc)
        text = response.text if status == 200 else None
        if status == 200 and text:
            return status, text, None, None, None
        return status, None, 'HTTP', f'HTTP {status}' if status != 200 else 'EMPTY_RESPONSE', None

    def _fetch_with_retries(self, url: str):
        """Up to max_attempts requests, retrying only transient failures, with BACKOFF_SECONDS between them."""
        attempts = []
        for k in range(self.max_attempts):
            if k:
                if self.retries_left <= 0:                       # an outage must not stretch the run without bound
                    attempts[-1]['retry_budget_exhausted'] = True
                    break
                self.retries_left -= 1
                self.sleep(BACKOFF_SECONDS[min(k, len(BACKOFF_SECONDS) - 1)])
            status, text, failure, cause, detail = self._fetch_once(url)
            attempts.append(dict(http_status=status, failure_class=failure, cause=cause))
            if not _transient(failure, status):
                break
        return status, text, failure, cause, detail, attempts

    def load_station_catalog(self):
        """The cached catalog only; never fetched here (production refreshes on a 7-day wall clock). The catalog is
        refreshed only by refresh_catalog(calibration=True)."""
        try:
            self.station_catalog = json.loads((self.cache_dir / 'ngl_catalog.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            self.station_catalog = {}
        return self.station_catalog

    def refresh_catalog(self, *, calibration: bool, declared=None) -> dict:
        """Refresh ngl_catalog.json on a CALIBRATION run only, validated before an atomic replace; the result is the
        catalog the calibration then binds. Outside calibration: NOT_DUE, no request."""
        path = self.cache_dir / 'ngl_catalog.json'
        base = dict(version=VERSION, file=path.name, url=self.CATALOG_URL, run_day=self.run_day.isoformat(),
                    run_id=self.run_id)
        before = path.read_bytes() if path.exists() else None
        prior = self.ledger.get(path.name)
        not_due = 'CATALOG_REFRESH_ONLY_ON_CALIBRATION_RUNS' if not calibration else self._not_due(prior)
        if not_due:
            record = dict(base, outcome='NOT_DUE', reason=not_due, attempted_utc=None, sha256=_sha(before))
            self.records.append(record)
            return record
        status, text, failure, cause, detail, attempts = self._fetch_with_retries(self.CATALOG_URL)
        after, n_before, n_after = before, None, None
        try:
            previous = json.loads(before.decode('utf-8')) if before is not None else None
        except ValueError:
            previous = None
        n_before = len(previous) if isinstance(previous, dict) else None
        if failure is None:
            catalog, vcause, vdetail = parse_catalog(text)
            if catalog is not None and n_before and len(catalog) < CATALOG_MIN_RETAINED * n_before:
                vcause, vdetail = 'CATALOG_SHRANK', f'{len(catalog)} stations < {CATALOG_MIN_RETAINED} x {n_before}'
            if catalog is not None and not vcause and declared:
                missing = sorted(set(declared) - set(catalog))
                if missing:
                    vcause, vdetail = 'DECLARED_STATION_MISSING_FROM_CATALOG', ', '.join(missing)
            if vcause:
                failure, cause, detail = 'VALIDATION', vcause, vdetail
            else:
                staged = json.dumps(catalog).encode('utf-8')         # production's json.dump of the same dict
                tmp = path.with_name(path.name + '.part')
                try:
                    tmp.write_bytes(staged)
                    os.replace(tmp, path)
                    after, n_after = staged, len(catalog)
                except OSError as exc:
                    failure, cause, detail = 'WRITE', type(exc).__name__, _bounded(exc)
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
        if failure is None:
            outcome = 'REFRESHED_SAME_CONTENT' if before is not None and _sha(after) == _sha(before) else 'REFRESHED_CATALOG'
        else:
            outcome = 'FAILED_KEPT_PREVIOUS' if before is not None else 'FAILED_NO_PREVIOUS'
        record = dict(base, outcome=outcome, attempted_utc=self.clock(), http_status=status, failure_class=failure,
                      cause=cause, detail=detail, attempts=attempts, sha256_before=_sha(before),
                      sha256_after=_sha(after), n_stations_before=n_before,
                      n_stations_after=n_after if n_after is not None else n_before)
        ledger_error = self.ledger.put(path.name, record)
        if ledger_error:
            record['ledger_error'] = ledger_error
        self.records.append(record)
        return record

    def _age(self, end: Optional[str]) -> Optional[int]:
        return (self.run_day - date.fromisoformat(end)).days if end else None

    def download_station(self, code, days_back=120, target_date=None):
        """Production's preference (IGS20 while USE_IGS20, IGS14 tenv3 otherwise) without the failure fall-through:
        IGS14 is used only when no IGS20 file exists after the attempt or it does not parse."""
        options = [(f'{self.BASE_URL_IGS20}/{code}.tenv', self.cache_dir / f'{code}.igs20.tenv', self._parse_tenv,
                    IGS20_TENV_FIELDS)] if self.USE_IGS20 else []
        options.append((f'{self.BASE_URL_IGS14}/{code}.tenv3', self.cache_dir / f'{code}.tenv3', self._parse_tenv3,
                        None))                             # tenv3 width not yet byte-verified: no contract claimed
        for url, path, parser, width in options:
            self._refresh(url, path, code, parser, days_back, target_date, width)
            if not path.exists():
                continue
            station, _ = self._parse(parser, code, path.read_text(encoding='utf-8'), days_back, target_date)
            if station:
                return station
        return None


def catalog_bound(baselines_dir, run_day: date, cache_dir, regions) -> dict:
    """Per region, before scoring: are the current ngl_catalog.json bytes the ones bound into the calibration selection
    of the baseline selected as of `run_day`? CATALOG_BOUND / CATALOG_CHANGED_SINCE_CALIBRATION / CATALOG_UNBOUND (an
    entry without a selection record) / UNAVAILABLE (typed reason)."""
    path = Path(cache_dir) / 'ngl_catalog.json'
    current = _sha(path.read_bytes()) if path.exists() else None
    selected, reason = P.select_as_of(baselines_dir, run_day)
    out = dict(run_day=run_day.isoformat(), current_catalog_sha256=current, regions={})
    for region in regions:
        if selected is None:
            out['regions'][region] = dict(status='UNAVAILABLE', reason=reason)
            continue
        entry = (selected['doc'].get('regions') or {}).get(region) or {}
        bound = (entry.get('selection') or {}).get('catalog_sha256')
        status = ('CATALOG_UNBOUND' if bound is None else 'CATALOG_BOUND' if bound == current
                  else 'CATALOG_CHANGED_SINCE_CALIBRATION')
        out['regions'][region] = dict(status=status, bound_catalog_sha256=bound)
    return out


def calibration_plan(baselines_dir, run_day: date, regions, *, baseline_max_age_days: int = 45,
                     warn_days: int = 2) -> dict:
    """Per region: the dated baseline selected as of `run_day`, its ACTUAL last accepted epoch, today's age, the first
    day it is stale (age > baseline_max_age_days) and an action: OK, RECALIBRATE_BY <day> (within warn_days of stale),
    STALE, or UNAVAILABLE (typed reason). A file generated after its calibrated_on is RETROSPECTIVE and is never
    counted as a prospective baseline. Plans only: writing a baseline is a separate, authorized step."""
    selected, reason = P.select_as_of(baselines_dir, run_day)
    plan = dict(version=PLAN_VERSION, run_day=run_day.isoformat(), baseline_max_age_days=baseline_max_age_days,
                file=None, regions={})
    if selected is None:
        plan['regions'] = {r: dict(action='UNAVAILABLE', reason=reason) for r in regions}
        return plan
    doc = selected['doc']
    generated = str(doc.get('generated_utc', ''))[:10]
    retrospective = generated > doc['calibrated_on']
    plan['file'] = dict(name=selected['path'].name, sha256=selected['sha256'], calibrated_on=doc['calibrated_on'],
                        generated_utc=doc.get('generated_utc'), retrospective=retrospective)
    for region in regions:
        entry = (doc.get('regions') or {}).get(region)
        if retrospective:
            plan['regions'][region] = dict(action='UNAVAILABLE', reason='RETROSPECTIVE_BASELINE_FILE')
            continue
        if not isinstance(entry, dict) or not entry.get('available'):
            plan['regions'][region] = dict(action='UNAVAILABLE',
                                           reason=(entry or {}).get('reason') if isinstance(entry, dict) else 'REGION_ABSENT')
            continue
        last = date.fromisoformat(entry['last_epoch'])
        age = (run_day - last).days
        stale_on = last + timedelta(days=baseline_max_age_days + 1)
        if age > baseline_max_age_days:
            action = 'STALE'
        elif (stale_on - run_day).days <= warn_days:
            action = f'RECALIBRATE_BY {(stale_on - timedelta(days=1)).isoformat()}'
        else:
            action = 'OK'
        plan['regions'][region] = dict(action=action, last_accepted_epoch=entry['last_epoch'], age_days=age,
                                       stale_on=stale_on.isoformat(), days_until_stale=(stale_on - run_day).days)
    return plan
