"""
Trans-Pacific Correlation Hypothesis Testing Module.

This module investigates potential stress coupling mechanisms between
tectonically active regions on opposing sides of the Pacific Plate.

Primary Hypotheses:
- H0 (Null): Observed correlations are tidal aliasing artifacts
- H1: Short-term lithospheric stress transfer (hours to days)
- H2: Mantle convection deflection (weeks to months) - deferred

Key Components:
- config.py: Feature flags and configuration
- tidal_correction.py: M2 phase correction for H0 testing
- thd_correlation.py: THD-THD cross-region correlation
- lambda_geo_correlation.py: For full-coverage region pairs only
- vertical_transient.py: Slab orphaning signature detection
- analyzer.py: Main orchestrator
- reporter.py: Paper figure generation

Reference: docs/TRANS_PACIFIC_CORRELATION_PAPER_SKELETON.md
"""

from .config import get_config, is_trans_pacific_enabled, TransPacificConfig

__all__ = [
    'get_config',
    'is_trans_pacific_enabled',
    'TransPacificConfig',
]
