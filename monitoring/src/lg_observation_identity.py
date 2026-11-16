"""lg_observation_identity.py -- the identity of the LG SELECTED OBSERVATION (lg-observation-content-v1; cayley 2026-11-16,
codex 5851b0649 / 8c8d84db9). Runner side, OUTSIDE lg_producer_candidate.OPERATOR_SOURCES: code_identity(), every
operator identity and every seeded baseline binding are unchanged, and no numerical value moves (raw LG, ratio,
eligibility, weight and combined value are the producer's, untouched).

Why. The v2 LG observation content (input_sha256, input_selection_sha256) digests the whole report input. Both embed the
scored day's as-of instant and every byte of every carrier file, so an unchanged observation reads REVISED_SAME_EPOCH on
every day its data do not advance (cayley 475ebc86a). This module digests what the SELECTED value actually consumed.

What the selected value depends on (derived from the operator SOURCE, live_data.py, read at 6f270885):
  Lambda(i) = max over triangles of || [E(i), E_dot(i)] ||_F          LambdaGeoComputer.compute + R._epoch_max
  E_dot(i)  = (E(i+1) - E(i-1)) / 2                                   central difference (i is interior)
  E(t)      = symmetric velocity gradient of each Delaunay triangle   lstsq on its 3 vertex velocities at the SAME row t
  v(t)      = (s(t+1) - s(t-1)) / 2 per station and component         central difference
  s(j)      = savgol(7, 2)(x)(j): a fixed 7-tap combination of x(j-3..j+3) when 3 <= j <= n-4 (interior)
  x(r)      = the parsed E/N value minus the station's fixed origin at union-grid row r or, where the station has no
              sample at r, np.interp over the ROW INDEX between its nearest real samples on each side.
So Lambda(i) is a function of x(i-5..i+5) of every carrier station, the positions (bound-catalog lat/lon) and their
order (Delaunay simplices and vertex order), and nothing else in the files. The support rule (R.valid_denominator_epochs:
every station real at i, i >= 5 rows from both grid ends, every station's first real sample <= i-5 and last >= i+5)
makes every s(j) used interior and brackets every gap row of the window on BOTH sides. A bracket can lie OUTSIDE the
window; its row offset and value are part of the closure. The window is five union-grid ROWS each side; the support rule
also requires a daily grid, so a row is a day.

The argument is the source composition above. Perturbation tests corroborate it (a change outside the closure leaves
every triangle value at the epoch bit-identical), and every run CHECKS it: before the identity is KNOWN, Lambda(i) is
recomputed FROM THE CLOSURE ALONE with the production functions and must equal the producer's selected value exactly.

The record (encoding lg-observation-content-c14n-v1: sorted-key compact ASCII JSON, every float as float.hex(),
non-finite refused, never rounded) holds:
- the schema, region, provider, operator identity and selected epoch;
- the margin and the dependency grid (the row datetimes of the window);
- the ordered carrier, with each station's frame, catalog lat/lon and fixed origin;
- each station's window rows: a parsed E/N pair, or, for a gap row, its interpolation (the row offsets and parsed values
  of both brackets);
- the triangle mapping, as vertex codes in operator order.

It excludes:
- the as-of instant, admit_until, days_back and the absolute row index, which moves with the rolling window start;
- file mtimes and whole-file digests;
- Up, uncertainties and headers;
- every row outside the closure.

Those stay in the existing whole-product provenance (input_sha256, input_selection_sha256), which is unchanged.

Every input is read ONCE. The bytes are hashed and compared with the producer's manifest, then parsed FROM THOSE SAME
BYTES by the production parser, with the baseline's bound catalog (itself read once and hashed). A mismatch, or a
missing, malformed or non-finite input, gives a typed UNKNOWN identity, and this module never raises. An UNKNOWN identity
blocks confirmation (method_comparability) but never removes a qualified LG value or changes its score.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import live_data as LD
import lg_numerator_support_rule as R
import lg_producer_candidate as P

SCHEMA = 'lg-observation-content-v1'
ENCODING = 'lg-observation-content-c14n-v1'
MARGIN = P.SUPPORT_MARGIN            # closure half-width in union-grid rows: savgol 3 + velocity 1 + E_dot 1
CLOSURE_MIN_MARGIN = 5               # the composition above needs at least this; a smaller margin cannot be KNOWN
SMOOTHING_WINDOW, SMOOTHING_ORDER = 7, 2   # acquire_region_data: StrainComputer(smoothing_window=7), savgol order 2
FRAMES = {'IGS20': ('.igs20.tenv', '_parse_tenv'), 'IGS14': ('.tenv3', '_parse_tenv3')}
_HEX = set('0123456789abcdef')


class IdentityUnknown(Exception):
    def __init__(self, reason: str, detail: str = ''):
        super().__init__(reason)
        self.reason, self.detail = reason, str(detail)[:300]


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _hex64(v) -> bool:
    return isinstance(v, str) and len(v) == 64 and set(v) <= _HEX


def _f(x) -> str:
    """One exact float: float.hex() (round-trips every bit, keeps -0.0); non-finite is refused, never rounded."""
    v = float(x)
    if not math.isfinite(v):
        raise IdentityUnknown('NON_FINITE_CONTENT', repr(x))
    return v.hex()


def canonical(record) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')


# ---- pure part: union grid, closure, recomputation (no I/O) ---------------------------------------------------------

def union_grid(stations: Sequence) -> Tuple[list, np.ndarray]:
    """compute_velocities' union grid and the producer's per-station real-sample mask."""
    times = sorted({t for s in stations for t in s.times})
    index = {t: i for i, t in enumerate(times)}
    support = np.zeros((len(times), len(stations)), dtype=bool)
    for k, s in enumerate(stations):
        for t in s.times:
            support[index[t], k] = True
    return times, support


def closure(stations: Sequence, origins: Dict[str, Sequence[float]], times: list, support: np.ndarray, i: int,
            margin: int = MARGIN) -> List[dict]:
    """Per station (carrier order): the consumed content of rows i-margin..i+margin. A real row carries the PARSED E/N
    (origin not yet subtracted); a gap row carries its interpolation: the nearest real rows on each side (either may be
    outside the window) with their parsed values. Raises IdentityUnknown when the support rule does not hold at i."""
    n, lo, hi = len(times), i - margin, i + margin
    if type(margin) is not int or margin < CLOSURE_MIN_MARGIN:
        raise IdentityUnknown('CLOSURE_MARGIN_INVALID', f'margin {margin!r} < {CLOSURE_MIN_MARGIN}')
    if lo < 0 or hi > n - 1:
        raise IdentityUnknown('CLOSURE_OUTSIDE_GRID', f'i={i} margin={margin} n={n}')
    out = []
    for k, s in enumerate(stations):
        real = np.where(support[:, k])[0]
        if not support[i, k] or real[0] > lo or real[-1] < hi:
            raise IdentityUnknown('SUPPORT_RULE_NOT_MET', s.code)
        at = {t: j for j, t in enumerate(s.times)}
        e = {int(r): float(s.east_mm[at[times[r]]]) for r in real}
        nn = {int(r): float(s.north_mm[at[times[r]]]) for r in real}
        rows = []
        for r in range(lo, hi + 1):
            if r in e:
                rows.append(dict(kind='R', row=r, e=e[r], n=nn[r]))
            else:
                left = int(real[real < r][-1])
                right = int(real[real > r][0])
                rows.append(dict(kind='I', row=r, left=left, right=right, e_left=e[left], n_left=nn[left],
                                 e_right=e[right], n_right=nn[right]))
        out.append(dict(code=s.code, lat=float(s.lat), lon=float(s.lon), origin=[float(origins[s.code][0]),
                        float(origins[s.code][1])], rows=rows))
    return out


def recompute(cl: List[dict], margin: int = MARGIN) -> Tuple[Optional[float], Optional[str], int]:
    """Lambda(i) FROM THE CLOSURE ALONE, with the production functions: conditioned values (parsed - origin, gap rows by
    np.interp over the row index between the two brackets), savgol(7, 2) over the window (rows i-2..i+2 are interior),
    central-difference velocities at i-1, i, i+1, StrainComputer.compute_strain_field, LambdaGeoComputer.compute,
    R._epoch_max of the centre row. (value | None, reason | None, n_triangles)."""
    width = 2 * margin + 1
    velocities = np.zeros((3, len(cl), 2))
    for k, st in enumerate(cl):
        oe, on = st['origin']
        xe, xn = np.empty(width), np.empty(width)
        for w, row in enumerate(st['rows']):
            if row['kind'] == 'R':
                xe[w], xn[w] = row['e'] - oe, row['n'] - on
            else:
                xp = [float(row['left']), float(row['right'])]
                xe[w] = np.interp(float(row['row']), xp, [row['e_left'] - oe, row['e_right'] - oe])
                xn[w] = np.interp(float(row['row']), xp, [row['n_left'] - on, row['n_right'] - on])
        se = LD.savgol_filter(xe, SMOOTHING_WINDOW, SMOOTHING_ORDER)
        sn = LD.savgol_filter(xn, SMOOTHING_WINDOW, SMOOTHING_ORDER)
        for t, j in enumerate((margin - 1, margin, margin + 1)):        # rows i-1, i, i+1 of the operator
            velocities[t, k, 0] = (se[j + 1] - se[j - 1]) / 2
            velocities[t, k, 1] = (sn[j + 1] - sn[j - 1]) / 2
    positions = np.array([[st['lat'], st['lon']] for st in cl])
    strain, _ = LD.StrainComputer(smoothing_window=SMOOTHING_WINDOW).compute_strain_field(velocities, positions)
    if strain is None:
        return None, 'TRIANGULATION_FAILED', 0
    lam = LD.LambdaGeoComputer().compute(strain)
    value, why = R._epoch_max(lam[1])
    return value, why, int(strain.shape[1])


def triangles(cl: List[dict]) -> List[List[str]]:
    """The operator's triangle mapping: Delaunay over the carrier positions in carrier order (live_data), as codes."""
    positions = np.array([[st['lat'], st['lon']] for st in cl])
    return [[cl[int(a)]['code'] for a in simplex] for simplex in LD.Delaunay(positions).simplices]


def record_of(region: str, provider: str, operator_identity: str, epoch: str, times: list, i: int, cl: List[dict],
              frames: Dict[str, str], margin: int = MARGIN) -> dict:
    """The canonical record. Rows are given as OFFSETS from the selected row, never as absolute indices."""
    def row(r):
        if r['kind'] == 'R':
            return ['R', r['row'] - i, _f(r['e']), _f(r['n'])]
        return ['I', r['row'] - i, r['left'] - i, r['right'] - i, _f(r['e_left']), _f(r['n_left']),
                _f(r['e_right']), _f(r['n_right']), times[r['left']].isoformat(), times[r['right']].isoformat()]
    return dict(schema=SCHEMA, encoding=ENCODING, region=region, provider=provider, operator_identity=operator_identity,
                epoch=epoch, margin_rows=margin, grid=[times[r].isoformat() for r in range(i - margin, i + margin + 1)],
                carrier=[dict(code=st['code'], frame=frames[st['code']], lat=_f(st['lat']), lon=_f(st['lon']),
                              origin=[_f(st['origin'][0]), _f(st['origin'][1])], rows=[row(r) for r in st['rows']])
                         for st in cl],
                triangles=triangles(cl))


# ---- I/O part: verified single reads of the producer's retained inputs ----------------------------------------------

def _read_verified(path: Path, sha256: str, length: Optional[int], what: str) -> bytes:
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise IdentityUnknown('INPUT_UNREADABLE', f'{what}: {type(exc).__name__}')
    if _sha(raw) != sha256 or (length is not None and len(raw) != length):
        raise IdentityUnknown('INPUT_DIGEST_MISMATCH', f'{what}: bytes differ from the producer manifest')
    return raw


def _verified_stations(cache_dir, catalog_path, manifest: dict, codes: Sequence[str], days_back: int, as_of):
    if not _hex64(manifest.get('catalog_sha256')) or catalog_path is None:
        raise IdentityUnknown('CATALOG_UNBOUND', 'no bound catalog digest or path')
    catalog = json.loads(P._text(_read_verified(catalog_path, manifest['catalog_sha256'], None, 'catalog')))
    acq = P.RetainedNGL(cache_dir)
    acq.station_catalog = catalog
    stations, frames = [], {}
    for code in codes:
        rec = (manifest.get('stations') or {}).get(code)
        if not isinstance(rec, dict) or rec.get('frame') not in FRAMES or not _hex64(rec.get('sha256')):
            raise IdentityUnknown('INPUT_RECORD_UNRECOGNISED', code)
        suffix, parser = FRAMES[rec['frame']]
        if rec.get('file') != code + suffix or type(rec.get('bytes')) is not int:
            raise IdentityUnknown('INPUT_RECORD_UNRECOGNISED', code)
        raw = _read_verified(Path(cache_dir) / rec['file'], rec['sha256'], rec['bytes'], code)
        try:
            station = getattr(acq, parser)(code, P._text(raw), days_back, as_of)
        except Exception as exc:
            raise IdentityUnknown('INPUT_UNPARSED', f'{code}: {type(exc).__name__}')
        if station is None:
            raise IdentityUnknown('INPUT_UNPARSED', code)
        stations.append(station)
        frames[code] = rec['frame']
    return stations, frames


def observation_content(region: str, scored_day: date, cache_dir, catalog_path, trace: dict, measurement: dict,
                        entry: dict, *, margin: int = MARGIN) -> Tuple[dict, Optional[dict]]:
    """(summary, record | None). The summary goes into provenance['measurement']['observation_content']; the record is
    run evidence. Never raises: anything unverifiable gives status UNKNOWN with a typed reason."""
    head = dict(schema=SCHEMA, encoding=ENCODING)
    try:
        if not isinstance(trace, dict) or trace.get('status') != 'OK' or not isinstance(measurement, dict):
            raise IdentityUnknown('NO_SELECTED_OBSERVATION', (trace or {}).get('status') if isinstance(trace, dict) else '')
        sel = trace.get('selection') or {}
        i, epoch = sel.get('index'), sel.get('epoch')
        if type(i) is not int or not isinstance(epoch, str) or measurement.get('observation_day') != epoch:
            raise IdentityUnknown('SELECTION_UNBOUND', 'selection index/epoch absent or not the recorded observation day')
        manifest = trace.get('input_manifest') or {}
        codes = list(trace.get('station_codes') or [])
        if (manifest.get('provider') != 'NGL' or len(codes) < 3 or len(set(codes)) != len(codes)
                or sorted(codes) != sorted((manifest.get('stations') or {}).keys())
                or manifest.get('as_of') != P.as_of_instant(scored_day).isoformat()
                or type(manifest.get('days_back')) is not int):
            raise IdentityUnknown('MANIFEST_UNBOUND', 'provider, carrier, as-of or window does not bind')
        origins = entry.get('origins') if isinstance(entry, dict) and entry.get('origin_rule') == P.ORIGIN_RULE else None
        if not isinstance(origins, dict) or any(c not in origins or len(origins[c]) != 2 for c in codes):
            raise IdentityUnknown('ORIGIN_UNBOUND', 'baseline origins do not cover the carrier')
        stations, frames = _verified_stations(cache_dir, catalog_path, manifest, codes, manifest['days_back'],
                                              P.as_of_instant(scored_day))
        times, support = union_grid(stations)
        if not (0 <= i < len(times)) or times[i].date().isoformat() != epoch:
            raise IdentityUnknown('GRID_RECONSTRUCTION_MISMATCH', f'row {i} is not {epoch}')
        valid = R.valid_denominator_epochs(support, P.SUPPORT_MARGIN)
        if not len(valid) or int(valid[-1]) != i:
            raise IdentityUnknown('SELECTION_RECONSTRUCTION_MISMATCH', 'the support rule selects another row')
        cl = closure(stations, origins, times, support, i, margin)
        value, why, n_tri = recompute(cl, margin)
        produced = measurement.get('lambda_max')
        if why or value is None or not isinstance(produced, float) or value != produced:
            raise IdentityUnknown('CLOSURE_RECOMPUTE_MISMATCH',
                                  f'closure {value!r} ({why}) vs producer {produced!r}')
        operator_identity = measurement.get('operator_identity')
        if not _hex64(operator_identity):
            raise IdentityUnknown('OPERATOR_IDENTITY_UNBOUND', '')
        record = record_of(region, 'NGL', operator_identity, epoch, times, i, cl, frames, margin)
        digest = _sha(canonical(record))
        rows = [r for st in cl for r in st['rows']]
        gaps = [r for r in rows if r['kind'] == 'I']
        outside = sum((r['left'] < i - margin) + (r['right'] > i + margin) for r in gaps)
        return dict(head, status='KNOWN', sha256=digest, epoch=epoch, margin_rows=margin,
                    window=[times[i - margin].date().isoformat(), times[i + margin].date().isoformat()],
                    n_stations=len(cl), n_rows=2 * margin + 1, n_interpolated_rows=len(gaps),
                    n_brackets_outside_window=int(outside), n_triangles=n_tri,
                    closure_recompute=dict(equal=True, lambda_max=produced)), record
    except IdentityUnknown as exc:
        return dict(head, status='UNKNOWN', reason=exc.reason, detail=exc.detail), None
    except Exception as exc:                       # the daily service goes on; the identity is merely unknown
        return dict(head, status='UNKNOWN', reason='IDENTITY_COMPUTATION_FAILED',
                    detail=f'{type(exc).__name__}: {str(exc)[:200]}'), None
