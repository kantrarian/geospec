"""
config.py - Feature flags for trans-Pacific correlation research.

This module provides a clean isolation layer between experimental
code and production monitoring. All experimental features are
DISABLED by default.

Usage:
    from experimental.trans_pacific.config import get_config, is_trans_pacific_enabled

    if is_trans_pacific_enabled():
        cfg = get_config()
        if cfg.tidal_phase_correction:
            # Run tidal correction analysis
            pass
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import logging

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)

# Path resolution: experimental/trans_pacific/config.py -> monitoring/config/
MONITORING_ROOT = Path(__file__).parent.parent.parent.parent  # Up to monitoring/
CONFIG_PATH = MONITORING_ROOT / 'config' / 'experimental.yaml'
DATA_DIR = MONITORING_ROOT / 'data'
EXPERIMENTAL_DATA_DIR = DATA_DIR / 'experimental' / 'trans_pacific'
ENSEMBLE_RESULTS_DIR = DATA_DIR / 'ensemble_results'


@dataclass
class TransPacificConfig:
    """Configuration for trans-Pacific correlation research."""

    # Master toggle - DISABLED by default
    enabled: bool = False

    # Sub-feature toggles
    tidal_phase_correction: bool = True
    thd_thd_correlation: bool = True
    lambda_geo_correlation: bool = True
    vertical_transient_detection: bool = False
    moment_ratio_analysis: bool = False

    # Region pairs to analyze
    # Note: Hualien is THD-only (single station IU.TATO)
    region_pairs: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("norcal_hayward", "hualien"),      # THD-only
        ("norcal_hayward", "tokyo_kanto"),  # Full coverage
    ])

    # Safety settings
    write_to_production_csv: bool = False  # Never pollute production
    generate_paper_figures: bool = True
    backtest_only: bool = True  # Only run in backtest mode by default

    # Paths (computed)
    data_dir: Path = field(default_factory=lambda: EXPERIMENTAL_DATA_DIR)
    ensemble_results_dir: Path = field(default_factory=lambda: ENSEMBLE_RESULTS_DIR)

    @classmethod
    def load(cls) -> 'TransPacificConfig':
        """Load configuration from YAML file."""
        if not YAML_AVAILABLE:
            logger.warning("PyYAML not available, using defaults (disabled)")
            return cls()

        if not CONFIG_PATH.exists():
            logger.info(f"Experimental config not found: {CONFIG_PATH} - using defaults (disabled)")
            return cls()

        try:
            with open(CONFIG_PATH) as f:
                data = yaml.safe_load(f)

            if not data:
                logger.warning("Empty experimental config, using defaults")
                return cls()

            tpc = data.get('features', {}).get('trans_pacific_correlation', {})

            # Parse region pairs from list of lists to list of tuples
            raw_pairs = tpc.get('region_pairs', [])
            region_pairs = [tuple(p) for p in raw_pairs] if raw_pairs else cls().region_pairs

            return cls(
                enabled=tpc.get('enabled', False),
                tidal_phase_correction=tpc.get('tidal_phase_correction', True),
                thd_thd_correlation=tpc.get('thd_thd_correlation', True),
                lambda_geo_correlation=tpc.get('lambda_geo_correlation', True),
                vertical_transient_detection=tpc.get('vertical_transient_detection', False),
                moment_ratio_analysis=tpc.get('moment_ratio_analysis', False),
                region_pairs=region_pairs,
                write_to_production_csv=tpc.get('write_to_production_csv', False),
                generate_paper_figures=tpc.get('generate_paper_figures', True),
                backtest_only=tpc.get('backtest_only', True),
            )
        except Exception as e:
            logger.error(f"Failed to load experimental config: {e}")
            return cls()  # Return defaults (all disabled)

    def get_output_dir(self, subdir: str = '') -> Path:
        """Get output directory, creating if needed."""
        path = self.data_dir / subdir if subdir else self.data_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def is_thd_only_pair(self, region_a: str, region_b: str) -> bool:
        """Check if a region pair can only use THD (not Lambda_geo)."""
        # Single-station regions that don't support Lambda_geo
        thd_only_regions = {'hualien', 'anchorage'}
        return region_a in thd_only_regions or region_b in thd_only_regions


def is_trans_pacific_enabled() -> bool:
    """Quick check if trans-Pacific correlation is enabled."""
    cfg = TransPacificConfig.load()
    return cfg.enabled


# Singleton config instance (lazy loaded)
_config: Optional[TransPacificConfig] = None


def get_config(force_reload: bool = False) -> TransPacificConfig:
    """Get cached configuration."""
    global _config
    if _config is None or force_reload:
        _config = TransPacificConfig.load()
    return _config


def reset_config() -> None:
    """Reset cached configuration (for testing)."""
    global _config
    _config = None
