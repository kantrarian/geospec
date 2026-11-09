"""lg_real_istanbul_fixture.py -- the retained-input istanbul_marmara fixture used by test_lg_numerator_support_rule.RealIstanbul
and by its expectation generator (grassmann 2026-11-09, codex 7ebb9c65: 'do not repair the fixture by merely replacing one
hardcoded live-cache date with another').

The fixture is DETERMINISTIC in the following sense: the expected numerator epoch and station count are pinned in a sidecar
(test_fixtures/lg_real_istanbul_expectation.json) TOGETHER WITH the sha256 of the exact input bytes they were derived from
(the station files the production acquisition actually used for this region + the cached catalog). The test asserts the
pinned values only when the current inputs have that identity; otherwise it SKIPS with a typed message naming both digests,
and the sidecar is regenerated deliberately by `python lg_real_istanbul_fixture.py <cache_dir> <out.json>`. Structural
properties of the rule are asserted regardless of the bytes.

Read-only toward the runner: the catalog is read from the cache file, station files are read from disk, and the NGL session
is never created (download_station is overridden). The cache directory defaults to the runner's live cache and can be
pointed at a preserved copy with the LG_TEST_CACHE environment variable."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

RUNNER = Path("C:/GeoSpec/geospec_runner")
DEFAULT_CACHE = RUNNER / "monitoring/data/gps_cache"
REGION = "istanbul_marmara"
DAYS_BACK = 90
TARGET = datetime(2026, 10, 2)
TARGET_B = datetime(2026, 9, 1)
SIDE_CAR = Path(__file__).resolve().parent / "test_fixtures" / "lg_real_istanbul_expectation.json"


def cache_dir() -> Path:
    return Path(os.environ.get("LG_TEST_CACHE") or DEFAULT_CACHE)


def available(cache: Path) -> bool:
    return (cache / "ngl_catalog.json").exists() and (cache / "ISTA.igs20.tenv").exists()


def load(cache: Path) -> dict:
    """Acquire the region twice (target 2026-10-02 and 2026-09-01) through the production operator with the support
    mask captured, exactly as the round-3 test did; also compute the input identity over the files actually used."""
    sys.path.insert(0, str(RUNNER / "monitoring/src"))
    sys.path.insert(0, str(RUNNER))
    cwd = os.getcwd()
    os.chdir(str(RUNNER))
    try:
        import live_data as LD

        used: dict = {}

        class NGLCache(LD.NGLLiveAcquisition):
            def load_station_catalog(self):
                self.station_catalog = json.load(open(self.cache_dir / "ngl_catalog.json"))
                return self.station_catalog

            def download_station(self, code, days_back=120, target_date=None):
                f20 = self.cache_dir / f"{code}.igs20.tenv"
                if f20.exists():
                    r = self._parse_tenv(code, f20.read_text(), days_back, target_date)
                    if r:
                        used[code] = f20.name
                        return r
                f14 = self.cache_dir / f"{code}.tenv3"
                if f14.exists():
                    try:
                        r = self._parse_tenv3(code, f14.read_text(), days_back, target_date)
                        if r:
                            used[code] = f14.name
                        return r
                    except Exception:
                        return None
                return None

        cap: dict = {}
        orig = LD.StrainComputer.compute_velocities

        def cv(self, stations):
            allt = set()
            for s in stations:
                allt.update(s.times)
            common = sorted(allt)
            idx = {t: i for i, t in enumerate(common)}
            sup = np.zeros((len(common), len(stations)), dtype=bool)
            for si, st in enumerate(stations):
                for t in st.times:
                    sup[idx[t], si] = True
            cap.update(times=common, support=sup, codes=[s.code for s in stations])
            return orig(self, stations)

        LD.StrainComputer.compute_velocities = cv
        try:
            ngl = NGLCache(cache)
            ngl.load_station_catalog()
            res = LD.acquire_region_data(REGION, ngl, DAYS_BACK, TARGET, None)
            cap_a = dict(cap)
            res_b = LD.acquire_region_data(REGION, ngl, DAYS_BACK, TARGET_B, None)
            cap_b = dict(cap)
        finally:
            LD.StrainComputer.compute_velocities = orig
    finally:
        os.chdir(cwd)
    files = sorted(set(used.values()) | {"ngl_catalog.json"})
    h = hashlib.sha256()
    for name in files:
        h.update(name.encode("utf-8") + b"\0" + hashlib.sha256((cache / name).read_bytes()).digest())
    return dict(res=res, cap=cap_a, res_b=res_b, cap_b=cap_b, files=files, input_manifest_sha256=h.hexdigest(), cache=str(cache))


def selection(fx: dict):
    """The rule's selection on the loaded fixture (identities bound on both sides, equal configuration)."""
    import lg_numerator_support_rule as R

    den = R.denominator_from_grid(fx["cap_b"]["times"], fx["cap_b"]["support"], fx["res_b"].lambda_geo_grid)
    ident_d = R.production_operator_identity(provider="NGL", station_codes=fx["cap_b"]["codes"], code_version="44fa5abf", days_back=DAYS_BACK)
    ident_l = R.production_operator_identity(provider="NGL", station_codes=fx["cap"]["codes"], code_version="44fa5abf", days_back=DAYS_BACK)
    sel = R.select_numerator(fx["cap"]["times"], fx["cap"]["support"], fx["res"].lambda_geo_grid, TARGET.date(),
                             denominator=den["mean_lambda_geo"] if den["available"] else None, live_identity=ident_l, denominator_identity=ident_d)
    return den, ident_d, ident_l, sel


def main(argv):
    cache = Path(argv[1]) if len(argv) > 1 else cache_dir()
    out = Path(argv[2]) if len(argv) > 2 else SIDE_CAR
    fx = load(cache)
    den, ident_d, ident_l, sel = selection(fx)
    import lg_numerator_support_rule as R

    doc = dict(schema="grassmann.lg-real-istanbul-expectation.v1", generated_utc=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
               region=REGION, days_back=DAYS_BACK, target=TARGET.date().isoformat(), target_b=TARGET_B.date().isoformat(), edge_margin=R.EDGE_MARGIN,
               files=fx["files"], input_manifest_sha256=fx["input_manifest_sha256"], source_cache=fx["cache"],
               expected=dict(status=sel.status, epoch=sel.epoch, stations_with_real_samples=sel.stations_with_real_samples,
                             denominator_available=bool(den["available"]), identities_equal=ident_d == ident_l))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(doc["expected"]), "manifest", fx["input_manifest_sha256"][:16], "->", out)


if __name__ == "__main__":
    main(sys.argv)
