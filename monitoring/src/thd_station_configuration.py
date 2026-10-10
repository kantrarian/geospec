"""thd-station-configuration-v1 (cayley 2026-10-10; codex b7a29458 correction 3): the EFFECTIVE THD station
configuration of one run, retained in every region's record so a report can be read against the configuration that
produced it -- not against today's station map.

Per region, every CONFIGURED station in role order (primary, fallback, fallback2) with the exact selector the run would
request, the selector's basis kind, the provider routing (adapters in order), the declared TLS trust of each provider and
the measurement operator's version and identity. This is configuration, not outcome: which station actually supplied
the value stays in `thd_attempts`; a station that was never attempted is still listed here.

Identity: `region_configuration_sha256` is the sha256 of the canonical JSON of the region's versions and stations;
`configuration_sha256` is the sha256 of the run's versions and every configured region's region hash. Both cover
CONTENT only. Provenance -- the source commit and the sha256 of the configuration-bearing source files -- is carried
beside them, never hashed into them, so an unrelated code change does not change a configuration that did not change.
The source commit is what the checkout says: HEAD_CLEAN (configuration files equal HEAD), WORKTREE_DIFFERS (commit
withheld), NOT_A_GIT_CHECKOUT or GIT_UNAVAILABLE. Never guessed.
"""
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

SCHEMA = "thd-station-configuration-v1"
ROLES = ("CONFIGURED_PRIMARY", "CONFIGURED_FALLBACK", "CONFIGURED_FALLBACK2")
# The files whose bytes decide the configuration (station lists, selectors, routing, trust, operators).
SOURCE_FILES = ("monitoring/src/run_ensemble_daily.py", "monitoring/src/thd_provider_routing.py",
                "monitoring/src/thd_daily_measurement.py", "monitoring/src/thd_bound_station_operator.py",
                "monitoring/src/thd_station_configuration.py")
FILES_HASH_RULE = "SHA256_OF_BYTES_CRLF_TO_LF"
_REPO = Path(__file__).resolve().parents[2]


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def configured_stations(config: Dict) -> List[tuple]:
    """(role, network, station) in role order, with run_region_assessment's network defaults ('CI' primary, 'IU'
    fallbacks). A region without seismic THD has none."""
    if not config.get("seismic_available") or not config.get("thd_station"):
        return []
    slots = ((config.get("thd_station"), config.get("thd_network", "CI")),
             (config.get("fallback_station"), config.get("fallback_network", "IU")),
             (config.get("fallback2_station"), config.get("fallback2_network", "IU")))
    return [(role, network, station) for role, (station, network) in zip(ROLES, slots) if station]


def station_entry(role: str, network: str, station: str, analyzer) -> Dict:
    """One configured station: exact selector, basis kind, routing, trust and operator identity."""
    import thd_bound_station_operator as OP
    import thd_daily_measurement as TDM
    import thd_provider_routing as TPR
    key = "%s.%s" % (network, station)
    adapters = TPR.route(network)
    providers = None if adapters is None else ["%s:%s" % (kind, name) for kind, name in adapters]
    trust = {} if adapters is None else {name: (TPR.tls_trust_for(name) or {}).get("source") for _, name in adapters}
    if OP.is_bound(network, station):
        spec = OP.BOUND_STATIONS[key]
        location, channel, rate, kind = spec["location"], spec["channel"], spec["expected_rate_hz"], "BOUND_OPERATOR"
        operator = dict(kind="BOUND_STATION_OPERATOR", version=spec["operator_version"],
                        identity=OP.daily_operator_identity(key))
    else:
        declared = TPR.declared_channel(network, station)
        location, _basis = TPR.selector_for(network, station)
        if declared is not None:
            channel, rate, kind = declared["channel"], declared["native_rate_hz"], "DECLARED_CHANNEL"
        else:
            channel, rate = TPR.DEFAULT_THD_CHANNEL, None
            kind = "WILDCARD_LOCATION" if location == "*" else "PINNED_LOCATION"
        m = TDM.measurement_record(network, station, analyzer)
        operator = dict(kind="DAILY_MEASUREMENT", version=m["version"], identity=m["identity"],
                        operator_class=TDM.operator_class_part(network, station, analyzer) or "LEGACY_OPERATOR_CLASS")
    return dict(role=role, station=key, selector="%s.%s.%s.%s" % (network, station, location, channel),
                selector_kind=kind, native_rate_hz=rate, providers=providers,
                routing="NETWORK_NOT_ROUTED" if adapters is None else TPR.ROUTING_VERSION,
                tls_trust=trust, operator=operator)


def versions() -> Dict:
    import thd_bound_station_operator as OP
    import thd_daily_measurement as TDM
    import thd_provider_routing as TPR
    return dict(schema=SCHEMA, routing=TPR.ROUTING_VERSION, station_channels=TPR.STATION_CHANNELS_VERSION,
                tls_trust=TPR.TLS_TRUST_VERSION, coverage=TPR.COVERAGE_VERSION, measurement=TDM.MEASUREMENT_VERSION,
                operator_class=TDM.OPERATOR_CLASS_VERSION,
                bound_operators={k: v["operator_version"] for k, v in sorted(OP.BOUND_STATIONS.items())})


def source_provenance(repo: Optional[Path] = None, runner=subprocess.run) -> Dict:
    """sha256 of every configuration-bearing file (CRLF -> LF first, so a Windows checkout hashes like its blob), and
    the checkout's commit only when those files equal HEAD."""
    repo = Path(repo or _REPO)
    files = {}
    for rel in SOURCE_FILES:
        p = repo / rel
        files[rel] = hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() if p.is_file() else None
    out = dict(commit=None, files=files, files_hash_rule=FILES_HASH_RULE)
    try:
        head = runner(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20)
        if head.returncode != 0:
            return dict(out, commit_status="NOT_A_GIT_CHECKOUT")
        dirty = runner(["git", "-C", str(repo), "status", "--porcelain", "--"] + list(SOURCE_FILES),
                       capture_output=True, text=True, timeout=20)
        if dirty.returncode != 0:
            return dict(out, commit_status="GIT_UNAVAILABLE")
        if dirty.stdout.strip():
            return dict(out, commit_status="WORKTREE_DIFFERS")
        return dict(out, commit=head.stdout.strip(), commit_status="HEAD_CLEAN")
    except (OSError, subprocess.SubprocessError):
        return dict(out, commit_status="GIT_UNAVAILABLE")


def run_configuration(regions_map: Dict[str, Dict], analyzer, provenance: Optional[Dict] = None) -> Dict:
    """{region: block} for every configured region, each block self-contained (versions, stations, both hashes,
    provenance). `provenance` is injected by tests; otherwise read once from the checkout."""
    ver = versions()
    regions = {}
    for region, config in sorted(regions_map.items()):
        stations = [station_entry(r, n, s, analyzer) for r, n, s in configured_stations(config)]
        regions[region] = dict(stations=stations, region_configuration_sha256=_sha(dict(versions=ver, stations=stations)))
    run_sha = _sha(dict(versions=ver, regions={r: b["region_configuration_sha256"] for r, b in regions.items()}))
    source = provenance if provenance is not None else source_provenance()
    return {region: dict(schema=SCHEMA, configuration_sha256=run_sha, versions=ver, source=source, **block)
            for region, block in regions.items()}


_PROVENANCE: Dict[str, Dict] = {}


def region_block(region: str, regions_map: Dict[str, Dict], analyzer) -> Optional[Dict]:
    """The region's block from the run's configuration, computed from the tables as they stand (never cached, so a
    changed table is never served stale); the checkout provenance is read once per process."""
    if "source" not in _PROVENANCE:
        _PROVENANCE["source"] = source_provenance()
    return run_configuration(regions_map, analyzer, provenance=_PROVENANCE["source"]).get(region)
