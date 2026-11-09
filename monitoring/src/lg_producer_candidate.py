"""LG producer + matched dated baseline: offline CANDIDATE (cayley 2026-11-09; owner priority via codex a145358c / 76525f35).

OFF unless explicitly called. The production LG path (run_ensemble_daily.fetch_ngl_lambda_geo) and every issued record
are unchanged. This module produces the provenance that lg_integration_candidate.classify (codex) checks, from RETAINED
inputs only, so the numerator and the denominator of every ratio come from one carrier and one operator:

- inputs: the retained cache bytes (NGL IGS20 tenv / IGS14 tenv3, GeoNet JSON, ngl_catalog.json), read with the
  production parsers and file preference, decoded as production's text-mode read decodes them. Every file is hashed
  exactly as parsed. No network I/O and no cache writes.
- operator: the UNCHANGED production live_data.acquire_region_data (StrainComputer / LambdaGeoComputer).
- support rule: grassmann's lg_numerator_support_rule v3, UNMODIFIED, called with SUPPORT_MARGIN = 5. The v3 default
  EDGE_MARGIN = 4 counts the Savitzky-Golay half-window (3) and the velocity central difference (1) but not
  LambdaGeoComputer's second central difference (E_dot), so an epoch 4 rows before a station's last real sample still
  depends on the tail-held value. test_lg_producer_candidate MEASURES the reach on the production operator.
- identity: operator_identity over provider, frame (derived from the formats of the files actually parsed), the station
  carrier, triangulation, smoothing, both differences, edge and gap policy, estimator, statistic, the operator CODE
  identity (sha256 of the operator source bytes, LF-normalised) and days_back. Both sides use DAYS_BACK = BASELINE_DAYS.
- fixed origin (codex 397b7e96; LG source audit numerical closure): NGL positions carry absolute northings near
  3.9e9 mm, and smoothing/differencing them in float64 lost ~1e-6 relative precision (paired stage comparison
  lg_stage_compare_devildog_grassmann_20261109T150231Z.json). The candidate subtracts ONE fixed E/N origin per
  station before interpolation and smoothing: at calibration the origin is the station's first calibration-window
  row rounded to the metre (an exact integer in mm, so the subtraction is exact), recorded in the dated baseline and
  REUSED for every window scored against it. No rolling recentering, no detrending; source bytes untouched. The rule
  is part of the operator identity, so a baseline without it can never be paired with a centred numerator.
- clocks: a scored day D is measured as of the END of D (every sample dated D or earlier, none later). The selected
  observation day and the day its support runs through are kept separate from D.
- baselines: immutable dated files lambda_geo_baselines_rule_YYYYMMDD.json; the as-of selection is the newest file
  dated on or before the scored day. With a catalog store (codex d38ea435), each baseline's catalog is immutable,
  hash-addressed metadata (<store>/<sha256>.json): calibration and scoring read THAT catalog, never the mutable
  cache copy, so a later calibration that refreshes the catalog cannot strand earlier scored days. Valid zero epochs are kept by the rule; a zero denominator is refused by the rule.
- carrier (declared-carrier rule, fixed 2026-11-09 before any Tokyo result under it): at calibration a parsed station
  is ADMITTED only if its last real sample is no more than ADMISSION_LAG_DAYS before the window end; others are named
  NOT_ADMITTED with their last sample day. The admitted stations are the baseline's DECLARED carrier. The scored side
  uses exactly that carrier: undeclared stations are not used (NOT_IN_DECLARED_CARRIER) and a declared station that
  is missing or unparsed refuses DECLARED_CARRIER_STATION_UNAVAILABLE by name. The rule reads only series end dates,
  never a value or a score. It replaces neither the support rule nor the identity check.

Nothing here registers a policy. The 45-day baseline-age / 30-day lag / 45-day observation-age values are codex's
CandidatePolicy defaults, applied only when a caller passes that policy, with an explicit effective day, to the ensemble.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np

import live_data as LD
import lg_numerator_support_rule as R
import lg_integration_candidate as LGI

VERSION = 'lg-producer-candidate-v1'
SCHEMA = 'geospec.lambda-geo-rule-baselines.v1'
SUPPORT_MARGIN = 5                    # real-sample rows required after the selected epoch (measured; see module doc)
BASELINE_DAYS = 90                    # the calibration window, and the acquisition window on BOTH sides
DAYS_BACK = BASELINE_DAYS
CALIBRATION_LAG_DAYS = 30             # the window ends this many days before calibrated_on
ADMISSION_LAG_DAYS = 7                # declared-carrier rule: a station's last real sample within this of the window end
ORIGIN_RULE = 'fixed-origin-v1'       # per-station E/N origin: first calibration-window row, rounded to the metre
ORIGIN_QUANTUM_MM = 1000.0
MIN_VALID_DAYS = 30
OPERATOR_SOURCES = ('live_data.py', 'lg_numerator_support_rule.py', 'lg_producer_candidate.py')
_FILE = re.compile(r'lambda_geo_baselines_rule_(\d{8})\.json')


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def as_of_instant(day: date) -> datetime:
    """The acquisition instant for a scored day: its last second, so samples dated `day` count and later ones do not."""
    return datetime.combine(day, time(23, 59, 59))


def code_identity(src_dir: Optional[Path] = None) -> str:
    """sha256 over the operator source files' bytes, LF-normalised so a CRLF checkout of the same code agrees."""
    src = Path(src_dir) if src_dir else Path(__file__).resolve().parent
    return _sha(canonical({n: _sha((src / n).read_bytes().replace(b'\r\n', b'\n')) for n in OPERATOR_SOURCES}))


def _mtime_utc(path: Path) -> str:
    """The retrieval clock a cache file carries (its last write), recorded beside, never inside, the content identity."""
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _text(raw: bytes) -> str:
    """What production's open(path, 'r').read() returns for these ASCII files (universal newlines)."""
    return raw.decode('utf-8').replace('\r\n', '\n').replace('\r', '\n')


def derive_origin(station) -> list:
    """The fixed-origin-v1 origin of a station from its FIRST row in this (calibration) window: E/N rounded to the
    metre, exact integers in mm (+ 0.0: a recorded origin never carries a negative zero)."""
    return [float(np.round(station.east_mm[0] / ORIGIN_QUANTUM_MM) * ORIGIN_QUANTUM_MM) + 0.0,
            float(np.round(station.north_mm[0] / ORIGIN_QUANTUM_MM) * ORIGIN_QUANTUM_MM) + 0.0]


def condition_origin(station, origin):
    """A copy of `station` with the fixed E/N origin subtracted (exact: values and origin agree to well within a factor
    of two). Up and positions are unchanged; the parsed source values are not modified."""
    return LD.StationData(code=station.code, lat=station.lat, lon=station.lon, times=list(station.times),
                          east_mm=np.asarray(station.east_mm, dtype=float) - origin[0],
                          north_mm=np.asarray(station.north_mm, dtype=float) - origin[1],
                          up_mm=np.asarray(station.up_mm, dtype=float).copy())


def _apply_origin(acq, code, station):
    """None when origins are not in use; otherwise the conditioned station, deriving (calibration) or reusing
    (scoring) the origin. A station with no bound origin is refused ORIGIN_UNBOUND."""
    if acq.origins_mode is None:
        return station
    if acq.origins_mode == 'derive':
        acq.origins[code] = derive_origin(station)
    elif code not in acq.origins:
        return None
    return condition_origin(station, acq.origins[code])


def _admitted(station, admit_until: Optional[date]) -> bool:
    """The declared-carrier admission test: True unless an admission day is set and the station's last real sample is
    before it. Reads only the series end date."""
    return admit_until is None or station.times[-1].date() >= admit_until


class RetainedNGL(LD.NGLLiveAcquisition):
    """NGL acquisition over the RETAINED cache only: production's parsers and file preference (IGS20 tenv while
    USE_IGS20, IGS14 tenv3 otherwise or as fallback), no download and no write. Records every file it parsed."""

    def __init__(self, cache_dir, admit_until: Optional[date] = None, declared=None, catalog_path=None, origins=None):
        self.cache_dir = Path(cache_dir)
        self.admit_until = admit_until
        self.declared = None if declared is None else set(declared)
        self.catalog_path = Path(catalog_path) if catalog_path is not None else None
        self.origins_mode = None if origins is None else ('derive' if origins == 'derive' else 'bound')
        self.origins = {} if origins in (None, 'derive') else {c: list(v) for c, v in origins.items()}
        self.station_catalog = None
        self._session = None
        self.inputs: Dict[str, dict] = {}
        self.retrieved: Dict[str, str] = {}
        self.parsed: Dict[str, LD.StationData] = {}
        self.attempts: Dict[str, dict] = {}
        self.catalog_sha256 = None
        self.catalog_mtime_utc = None

    @property
    def session(self):
        raise RuntimeError('retained inputs only: network access refused')

    def load_station_catalog(self):
        """The BOUND catalog when one is given (never falling back to the cache copy), else the cache's catalog."""
        path = self.catalog_path if self.catalog_path is not None else self.cache_dir / 'ngl_catalog.json'
        raw = path.read_bytes()
        self.catalog_sha256 = _sha(raw)
        self.catalog_mtime_utc = _mtime_utc(path)
        self.station_catalog = json.loads(_text(raw))
        return self.station_catalog

    def download_station(self, code, days_back=120, target_date=None):
        if self.declared is not None and code not in self.declared:
            self.attempts[code] = dict(outcome='NOT_IN_DECLARED_CARRIER', files=[])
            return None
        options = [(f'{code}.igs20.tenv', 'IGS20', self._parse_tenv)] if self.USE_IGS20 else []
        options.append((f'{code}.tenv3', 'IGS14', self._parse_tenv3))
        tried = []
        for name, fmt, parser in options:
            path = self.cache_dir / name
            if not path.exists():
                tried.append(dict(file=name, present=False))
                continue
            raw = path.read_bytes()
            try:
                station = parser(code, _text(raw), days_back, target_date)
            except Exception:
                station = None
            tried.append(dict(file=name, present=True, sha256=_sha(raw), bytes=len(raw), parsed=bool(station)))
            if station and not _admitted(station, self.admit_until):
                self.attempts[code] = dict(outcome='NOT_ADMITTED', files=tried,
                                           last_real_day=station.times[-1].date().isoformat())
                return None
            if station:
                station = _apply_origin(self, code, station)
                if station is None:
                    self.attempts[code] = dict(outcome='ORIGIN_UNBOUND', files=tried)
                    return None
                self.inputs[code] = dict(file=name, frame=fmt, sha256=_sha(raw), bytes=len(raw))
                self.retrieved[code] = _mtime_utc(path)
                self.parsed[code] = station
                self.attempts[code] = dict(outcome='PARSED', files=tried)
                return station
        self.attempts[code] = dict(outcome='UNPARSED' if any(t['present'] for t in tried) else 'ABSENT', files=tried)
        return None


class RetainedGeoNet(LD.GeoNetLiveAcquisition):
    """GeoNet acquisition over the RETAINED cache JSON only (no Tilde API call, no cache-age shortcut)."""

    def __init__(self, cache_dir, admit_until: Optional[date] = None, declared=None, origins=None):
        self.cache_dir = Path(cache_dir)
        self.admit_until = admit_until
        self.declared = None if declared is None else set(declared)
        self.origins_mode = None if origins is None else ('derive' if origins == 'derive' else 'bound')
        self.origins = {} if origins in (None, 'derive') else {c: list(v) for c, v in origins.items()}
        self._session = None
        self.inputs: Dict[str, dict] = {}
        self.retrieved: Dict[str, str] = {}
        self.parsed: Dict[str, LD.StationData] = {}
        self.attempts: Dict[str, dict] = {}

    @property
    def session(self):
        raise RuntimeError('retained inputs only: network access refused')

    def download_station(self, code, days_back=120, target_date=None):
        if self.declared is not None and code not in self.declared:
            self.attempts[code] = dict(outcome='NOT_IN_DECLARED_CARRIER', files=[])
            return None
        path = self.cache_dir / f'{code}.geonet.json'
        if not path.exists():
            self.attempts[code] = dict(outcome='ABSENT', files=[dict(file=path.name, present=False)])
            return None
        raw = path.read_bytes()
        try:
            station = self._parse_cached_data(code, json.loads(_text(raw)),
                                              target_date - timedelta(days=days_back), target_date)
        except Exception:
            station = None
        if station and not _admitted(station, self.admit_until):
            self.attempts[code] = dict(outcome='NOT_ADMITTED', last_real_day=station.times[-1].date().isoformat(),
                                       files=[dict(file=path.name, present=True, sha256=_sha(raw), bytes=len(raw),
                                                   parsed=True)])
            return None
        if station:
            station = _apply_origin(self, code, station)
            if station is None:
                self.attempts[code] = dict(outcome='ORIGIN_UNBOUND', files=[dict(file=path.name, present=True,
                                           sha256=_sha(raw), bytes=len(raw), parsed=True)])
                return None
        if station:
            self.inputs[code] = dict(file=path.name, frame='GeoNet-Tilde', sha256=_sha(raw), bytes=len(raw))
            self.retrieved[code] = _mtime_utc(path)
            self.parsed[code] = station
        self.attempts[code] = dict(outcome='PARSED' if station else 'UNPARSED',
                                   files=[dict(file=path.name, present=True, sha256=_sha(raw), bytes=len(raw),
                                               parsed=bool(station))])
        return station


@dataclass
class Measurement:
    region: str
    target: date
    days_back: int
    provider: str
    frame: str
    station_codes: list
    times: list
    support: np.ndarray
    grid: np.ndarray
    input_manifest: dict
    input_sha256: str
    identity: str
    retrieval: dict
    selection: dict
    selection_sha256: str
    origins: dict


def epoch_series(m: Measurement) -> list:
    """The epoch-indexed LG series behind a selection: [day, daily max over finite triangles or None, final]. An epoch
    is FINAL when the support rule admits it (every station has real samples at it and SUPPORT_MARGIN rows on both
    sides); appending later samples cannot change a final value. Non-final epochs are PROVISIONAL and may revise."""
    final = set(int(i) for i in R.valid_denominator_epochs(m.support, SUPPORT_MARGIN))
    rows = []
    for i, t in enumerate(m.times):
        value, _ = R._epoch_max(m.grid[i])
        rows.append([t.date().isoformat(), value, i in final])
    return rows


def operator_identity(provider, frame, station_codes, days_back, code_id) -> str:
    return R.operator_identity(
        provider=provider, frame=frame, units='mm/day-strain-rate-commutator', station_codes=list(station_codes),
        triangulation='scipy-Delaunay-lat-lon-km-local', smoothing=dict(filter='savgol', window=7, polyorder=2),
        derivative='central-difference-1day-velocity-and-strain-rate', edge_policy=f'full-support-interior-margin-{SUPPORT_MARGIN}',
        gap_policy='interior-linear-interp-no-tail-hold', estimator='LambdaGeoComputer-commutator-norm',
        reduction='nanmax-over-triangles', baseline_statistic=LGI.STATISTIC,
        processing_version=f'lg-support-rule-v3+{VERSION}+declared-carrier-admission-{ADMISSION_LAG_DAYS}d+{ORIGIN_RULE}',
        code_version=code_id, days_back=int(days_back),
        edge_margin=SUPPORT_MARGIN)


def calibration_window_end(calibrated_on: date) -> date:
    return calibrated_on - timedelta(days=CALIBRATION_LAG_DAYS)


def _acquire(region: str, target: date, cache_dir, days_back: int, admit_until: Optional[date] = None, declared=None,
             catalog_path=None, origins=None):
    """(result | None, ngl, geonet, reason | None, selection): one production acquisition over the retained cache and
    the SELECTION it made: every candidate station attempted, each file looked for (present or ABSENT, its sha256 and
    size, parsed or not), which acquisition supplied the carrier, the carrier and any co-located drop. The selection
    is the input-completeness record: a partial copy of the cache that loses a file changes it."""
    ngl = RetainedNGL(cache_dir, admit_until=admit_until, declared=declared, catalog_path=catalog_path, origins=origins)
    geonet = RetainedGeoNet(cache_dir, admit_until=admit_until, declared=declared, origins=origins)
    selection = dict(region=region, as_of=as_of_instant(target).isoformat(), days_back=int(days_back),
                     admit_until=admit_until.isoformat() if admit_until else None,
                     declared_carrier=sorted(declared) if declared is not None else None)
    try:
        ngl.load_station_catalog()
    except FileNotFoundError:
        return None, ngl, geonet, 'RETAINED_CATALOG_ABSENT', dict(selection, outcome='RETAINED_CATALOG_ABSENT')
    result = LD.acquire_region_data(region, ngl, days_back, as_of_instant(target), geonet)
    # acquire_region_data uses GeoNet for an NZ region when >= 3 GeoNet stations parse, otherwise NGL: the same test
    # names which acquisition supplied the carrier (a station code can exist in both).
    geonet_used = geonet.is_nz_region(region) and len(geonet.parsed) >= 3
    source = geonet if geonet_used else ngl
    codes = list(result.station_codes) if result is not None else []
    reason = None
    if result is None:
        reason = 'REGION_HAS_NO_POLYGON'
    elif result.n_stations < 3 or getattr(result.lambda_geo_grid, 'size', 0) == 0:
        reason = 'INSUFFICIENT_STATIONS'
    selection.update(outcome=reason or 'MEASURED', provider='GeoNet' if geonet_used else 'NGL',
                     catalog_sha256=ngl.catalog_sha256, ngl_attempts=ngl.attempts, geonet_attempts=geonet.attempts,
                     carrier=codes, colocated_dropped=sorted(c for c in source.parsed if c not in codes))
    return result, ngl, geonet, reason, selection


def select_inputs(region: str, target: date, cache_dir, days_back: int = DAYS_BACK, admit_until: Optional[date] = None,
                  declared=None, catalog_path=None) -> dict:
    """The selection record of one window (see _acquire), whether or not a measurement results."""
    return _acquire(region, target, cache_dir, days_back, admit_until, declared, catalog_path)[4]


def admission_day(window_end: date) -> date:
    return window_end - timedelta(days=ADMISSION_LAG_DAYS)


def compare_selections(reference: dict, candidate: dict) -> list:
    """Every difference between two selection records of the same window, by station and file. Empty = the candidate
    cache makes exactly the reference cache's selection (same attempts, files, bytes, outcomes, carrier)."""
    diffs = []
    for key in ('region', 'as_of', 'days_back', 'admit_until', 'declared_carrier', 'outcome', 'provider', 'catalog_sha256',
                'carrier', 'colocated_dropped'):
        if reference.get(key) != candidate.get(key):
            diffs.append(dict(field=key, reference=reference.get(key), candidate=candidate.get(key)))
    for group in ('ngl_attempts', 'geonet_attempts'):
        ref, cand = reference.get(group) or {}, candidate.get(group) or {}
        for code in sorted(set(ref) | set(cand)):
            if ref.get(code) != cand.get(code):
                diffs.append(dict(field=group, station=code, reference=ref.get(code), candidate=cand.get(code)))
    return diffs


def measure(region: str, target: date, cache_dir, days_back: int = DAYS_BACK, code_id: Optional[str] = None,
            admit_until: Optional[date] = None, declared=None, catalog_path=None, origins=None):
    """Run the production operator over retained inputs as of the end of `target`. (Measurement | None, reason).
    `admit_until` applies the declared-carrier admission (calibration); `declared` restricts the carrier to a baseline's
    declared stations (scored side), refusing 'DECLARED_CARRIER_STATION_UNAVAILABLE: <codes>' when one is missing."""
    result, ngl, geonet, reason, selection = _acquire(region, target, cache_dir, days_back, admit_until, declared,
                                                      catalog_path, origins)
    if reason in ('RETAINED_CATALOG_ABSENT', 'REGION_HAS_NO_POLYGON'):
        return None, reason
    if declared is not None:
        missing = sorted(set(declared) - set(selection.get('carrier') or []))
        if missing:
            return None, 'DECLARED_CARRIER_STATION_UNAVAILABLE: ' + ', '.join(missing)
    if reason:
        return None, reason
    geonet_used = selection['provider'] == 'GeoNet'
    source = geonet if geonet_used else ngl
    codes = list(result.station_codes)
    if any(c not in source.parsed for c in codes):
        return None, 'INPUT_RECORD_INCOMPLETE'
    stations = [source.parsed[c] for c in codes]
    times = sorted({t for s in stations for t in s.times})            # compute_velocities' union grid
    index = {t: i for i, t in enumerate(times)}
    support = np.zeros((len(times), len(stations)), dtype=bool)
    for k, s in enumerate(stations):
        for t in s.times:
            support[index[t], k] = True
    grid = np.asarray(result.lambda_geo_grid, dtype=float)
    if grid.shape[0] != len(times):
        return None, 'GRID_TIME_MISMATCH'
    provider = 'GeoNet' if geonet_used else 'NGL'
    frames = sorted({source.inputs[c]['frame'] for c in codes})
    frame = frames[0] if len(frames) == 1 else 'MIXED:' + '+'.join(frames)
    manifest = dict(region=region, provider=provider, as_of=as_of_instant(target).isoformat(), days_back=int(days_back),
                    stations={c: source.inputs[c] for c in codes},
                    catalog_sha256=None if geonet_used else ngl.catalog_sha256)
    code_id = code_id or code_identity()
    stamps = sorted(source.retrieved[c] for c in codes)
    retrieval = dict(files_mtime_utc={c: source.retrieved[c] for c in codes}, oldest_mtime_utc=stamps[0],
                     newest_mtime_utc=stamps[-1], catalog_mtime_utc=None if geonet_used else ngl.catalog_mtime_utc)
    return Measurement(region=region, target=target, days_back=int(days_back), provider=provider, frame=frame,
                       station_codes=codes, times=times, support=support, grid=grid, input_manifest=manifest,
                       input_sha256=_sha(canonical(manifest)),
                       identity=operator_identity(provider, frame, codes, days_back, code_id), retrieval=retrieval,
                       selection=selection, selection_sha256=_sha(canonical(selection)),
                       origins={c: source.origins[c] for c in codes} if source.origins_mode else {}), None


def calibrate_region(region: str, calibrated_on: date, cache_dir, code_id: Optional[str] = None,
                     catalog_path=None) -> dict:
    """The rule denominator over the BASELINE_DAYS window ending CALIBRATION_LAG_DAYS before `calibrated_on`, read
    with the catalog at `catalog_path` when given (the stored catalog this baseline binds)."""
    end = calibration_window_end(calibrated_on)
    admit = admission_day(end)
    nominal = dict(nominal_window_end=end.isoformat(), nominal_days=BASELINE_DAYS)
    m, reason = measure(region, end, cache_dir, days_back=BASELINE_DAYS, code_id=code_id, admit_until=admit,
                        catalog_path=catalog_path, origins='derive')
    selection = m.selection if m is not None else select_inputs(region, end, cache_dir, BASELINE_DAYS, admit_until=admit,
                                                                catalog_path=catalog_path)
    excluded = {c: row.get('last_real_day') for group in ('ngl_attempts', 'geonet_attempts')
                for c, row in (selection.get(group) or {}).items() if row.get('outcome') == 'NOT_ADMITTED'}
    admission = dict(rule=f'declared-carrier-admission-{ADMISSION_LAG_DAYS}d', admit_until=admit.isoformat(),
                     not_admitted=dict(sorted(excluded.items())))
    if m is None:
        return dict(available=False, reason=reason, **nominal, admission=admission, selection=selection,
                    selection_sha256=_sha(canonical(selection)))
    d = R.denominator_from_grid(m.times, m.support, m.grid, edge_margin=SUPPORT_MARGIN, min_valid_days=MIN_VALID_DAYS)
    return dict(d, **nominal, provider=m.provider, frame=m.frame, station_codes=m.station_codes,
                n_stations=len(m.station_codes), operator_identity=m.identity, input_manifest=m.input_manifest,
                input_sha256=m.input_sha256, days_back=m.days_back, admission=admission, selection=m.selection,
                selection_sha256=m.selection_sha256, origin_rule=ORIGIN_RULE, origins=m.origins)


def write_baselines(out_dir, calibrated_on: date, regions: Dict[str, dict], code_id: str, generated_utc: str) -> Path:
    """Write ONE immutable dated baseline file; an existing file of that date is never overwritten (FileExistsError).
    `generated_utc` is the caller's clock read; it is recorded, not used for selection."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'lambda_geo_baselines_rule_{calibrated_on:%Y%m%d}.json'
    doc = dict(schema=SCHEMA, version=VERSION, calibrated_on=calibrated_on.isoformat(), generated_utc=generated_utc,
               rule='lg-support-rule-v3', support_margin=SUPPORT_MARGIN, baseline_days=BASELINE_DAYS,
               calibration_lag_days=CALIBRATION_LAG_DAYS, min_valid_days=MIN_VALID_DAYS, days_back=DAYS_BACK,
               code_identity=code_id, statistic=LGI.STATISTIC, regions=regions)
    with open(path, 'xb') as fh:
        fh.write(json.dumps(doc, indent=1, sort_keys=True, allow_nan=False).encode('utf-8') + b'\n')
    return path


def select_as_of(baselines_dir, scored_day: date):
    """(selected | None, reason): the newest dated file with calibrated_on <= scored_day, the sha256 of its bytes and
    its parsed document. A file whose name and content disagree is refused, never skipped for an older one."""
    best = None
    for path in Path(baselines_dir).glob('lambda_geo_baselines_rule_*.json'):
        match = _FILE.fullmatch(path.name)
        if not match:
            continue
        day = datetime.strptime(match.group(1), '%Y%m%d').date()
        if day <= scored_day and (best is None or day > best[0]):
            best = (day, path)
    if best is None:
        return None, 'NO_BASELINE_AS_OF_SCORED_DAY'
    raw = best[1].read_bytes()
    doc = json.loads(raw.decode('utf-8'))
    if (doc.get('schema') != SCHEMA or doc.get('calibrated_on') != best[0].isoformat()
            or doc.get('support_margin') != SUPPORT_MARGIN or doc.get('days_back') != DAYS_BACK):
        return None, 'BASELINE_FILE_INCONSISTENT'
    return dict(path=best[1], sha256=_sha(raw), doc=doc), None


def resolve_bound_catalog(catalog_store, entry: dict):
    """(path | None, status, detail): the stored catalog a baseline entry binds, verified by its sha256. Never the
    mutable cache copy: a missing or altered stored catalog refuses by name."""
    sha = (entry.get('selection') or {}).get('catalog_sha256')
    if not sha:
        return None, 'BOUND_CATALOG_UNRECORDED', 'the baseline entry records no catalog sha256'
    path = Path(catalog_store) / f'{sha}.json'
    if not path.exists():
        return None, 'BOUND_CATALOG_UNAVAILABLE', f'catalog {sha} is not in the store'
    if _sha(path.read_bytes()) != sha:
        return None, 'BOUND_CATALOG_ALTERED', f'stored catalog {sha} no longer hashes to its name'
    return path, 'BOUND_CATALOG_VERIFIED', sha


def candidate_lambda_geo(region: str, scored_day: date, cache_dir, baselines_dir, code_id: Optional[str] = None,
                         catalog_store=None):
    """(ratio | None, provenance | None, trace). `ratio` is the rule ratio, clamped exactly as production clamps, when
    the selection is OK; otherwise None, and trace['status'] / trace['detail'] give the typed reason. `provenance` is
    the record codex's classify checks; it is present whenever a baseline and a measurement exist (refusals included)."""
    code_id = code_id or code_identity()
    trace = dict(version=VERSION, region=region, scored_day=scored_day.isoformat(), code_identity=code_id,
                 support_margin=SUPPORT_MARGIN, days_back=DAYS_BACK)
    selected, reason = select_as_of(baselines_dir, scored_day)
    if selected is None:
        return None, None, dict(trace, status=reason, detail='no usable dated baseline file on or before the scored day')
    doc, entry = selected['doc'], (selected['doc'].get('regions') or {}).get(region)
    trace['baseline_file'] = dict(name=selected['path'].name, sha256=selected['sha256'], calibrated_on=doc['calibrated_on'])
    if not isinstance(entry, dict) or not entry.get('available'):
        why = (entry or {}).get('reason') if isinstance(entry, dict) else 'region absent from the file'
        return None, None, dict(trace, status='BASELINE_UNAVAILABLE', detail=f'selected baseline file: {why}')
    catalog_path = None
    if catalog_store is not None:                      # the baseline's own catalog; no fallback to the cache copy
        catalog_path, cstatus, cdetail = resolve_bound_catalog(catalog_store, entry)
        trace['bound_catalog'] = dict(status=cstatus, detail=cdetail)
        if catalog_path is None:
            return None, None, dict(trace, status=cstatus, detail=cdetail, declared_carrier=entry['station_codes'])
    origins = entry.get('origins') if entry.get('origin_rule') == ORIGIN_RULE else None
    unbound = sorted(set(entry['station_codes']) - set(origins or {}))
    if origins is None or unbound:
        return None, None, dict(trace, status='BASELINE_ORIGIN_UNBOUND',
                                detail=f'baseline records no {ORIGIN_RULE} origins' if origins is None
                                else ', '.join(unbound),
                                declared_carrier=entry['station_codes'])
    m, reason = measure(region, scored_day, cache_dir, days_back=DAYS_BACK, code_id=code_id,
                        declared=entry['station_codes'], catalog_path=catalog_path, origins=origins)
    if m is None:
        code, _, names = reason.partition(':')
        return None, None, dict(trace, status=code, detail=names.strip() or 'retained-input measurement not produced',
                                declared_carrier=entry['station_codes'])
    sel = R.select_numerator(m.times, m.support, m.grid, scored_day, denominator=entry['mean_lambda_geo'],
                             edge_margin=SUPPORT_MARGIN, live_identity=m.identity,
                             denominator_identity=entry['operator_identity'])
    through = None
    if sel.index is not None and sel.index + SUPPORT_MARGIN < len(m.times):
        through = m.times[sel.index + SUPPORT_MARGIN].date().isoformat()
    measurement = dict(operator_identity=m.identity, input_sha256=m.input_sha256, observation_day=sel.epoch,
                       support_through_day=through, data_end_day=m.times[-1].date().isoformat(),
                       as_of=as_of_instant(scored_day).isoformat(), observation_age_days=sel.days_before_target,
                       selection_status=sel.status, lambda_max=sel.lambda_max, unclamped_ratio=sel.unclamped_ratio,
                       selection_reason=sel.reason, retrieval=m.retrieval, series=epoch_series(m),
                       input_selection_sha256=m.selection_sha256)
    provenance = dict(source=f"{selected['path'].name}:{region}", n_days=entry['n_samples'],
                      window_start=entry['first_epoch'], window_end=entry['last_epoch'],
                      calibrated_on=doc['calibrated_on'], calibration_sha256=selected['sha256'],
                      baseline_value=entry['mean_lambda_geo'], statistic=entry['statistic'],
                      operator_identity=entry['operator_identity'], measurement=measurement)
    trace['origin_rule'] = ORIGIN_RULE
    trace.update(status=sel.status, detail=sel.reason, provider=m.provider, frame=m.frame,
                 station_codes=m.station_codes, input_sha256=m.input_sha256, input_manifest=m.input_manifest,
                 grid_first=m.times[0].date().isoformat(), grid_last=m.times[-1].date().isoformat(),
                 carrier_equal=sorted(m.station_codes) == sorted(entry['station_codes']),
                 identity_equal=m.identity == entry['operator_identity'], selection=R.as_dict(sel),
                 input_selection=m.selection, baseline_input_selection_sha256=entry.get('selection_sha256'),
                 declared_carrier=entry['station_codes'], admission=entry.get('admission'),
                 baseline=dict(mean=entry['mean_lambda_geo'], n_samples=entry['n_samples'],
                               n_zero_days=entry['n_zero_days'], first_epoch=entry['first_epoch'],
                               last_epoch=entry['last_epoch'], station_codes=entry['station_codes'],
                               input_sha256=entry['input_sha256']))
    ratio = sel.clamped_ratio if sel.status == 'OK' else None
    return ratio, provenance, trace
