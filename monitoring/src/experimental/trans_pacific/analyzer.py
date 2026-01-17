"""
analyzer.py - Main Orchestrator for Trans-Pacific Correlation Analysis

This module orchestrates the full analysis pipeline:
1. Load configuration and validate prerequisites
2. Run tidal phase correction analysis (H0 testing)
3. Compute THD-THD correlations for configured pairs
4. Generate summary report with hypothesis conclusions

Usage:
    from experimental.trans_pacific.analyzer import run_trans_pacific_analysis

    # Called from run_ensemble_daily.py when feature is enabled
    results = run_trans_pacific_analysis(ensemble_results, target_date, output_dir)

Reference: docs/TRANS_PACIFIC_CORRELATION_PAPER_SKELETON.md
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging

from .config import get_config, TransPacificConfig, EXPERIMENTAL_DATA_DIR
from .tidal_correction import TidalPhaseCorrector, TidalCorrectionResult
from .thd_correlation import THDCorrelationAnalyzer, CorrelationResult

logger = logging.getLogger(__name__)


@dataclass
class TransPacificAnalysisResult:
    """Complete results from trans-Pacific analysis run."""
    timestamp: datetime
    config: Dict[str, Any]

    # Tidal phase analysis for each pair
    tidal_results: Dict[str, TidalCorrectionResult]

    # THD correlation results for each pair
    correlation_results: Dict[str, CorrelationResult]

    # Summary statistics
    n_pairs_analyzed: int = 0
    n_significant_correlations: int = 0
    n_h0_supported: int = 0  # Pairs where tidal aliasing explains correlation
    n_h0_rejected: int = 0   # Pairs where correlation persists after correction

    # Interpretation
    summary: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'config': self.config,
            'tidal_results': {k: v.to_dict() for k, v in self.tidal_results.items()},
            'correlation_results': {k: v.to_dict() for k, v in self.correlation_results.items()},
            'n_pairs_analyzed': self.n_pairs_analyzed,
            'n_significant_correlations': self.n_significant_correlations,
            'n_h0_supported': self.n_h0_supported,
            'n_h0_rejected': self.n_h0_rejected,
            'summary': self.summary,
            'notes': self.notes,
        }

    def save(self, output_path: Path) -> None:
        """Save results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved trans-Pacific analysis to {output_path}")


class TransPacificAnalyzer:
    """
    Main orchestrator for trans-Pacific correlation hypothesis testing.

    This class:
    1. Validates that the feature is properly enabled
    2. Runs tidal phase analysis for each configured region pair
    3. Computes THD correlations with tidal correction
    4. Generates a comprehensive summary with hypothesis conclusions
    """

    def __init__(self, config: Optional[TransPacificConfig] = None):
        """
        Initialize the analyzer.

        Args:
            config: Optional config override (uses get_config() if not provided)
        """
        self.config = config or get_config()
        self.tidal_corrector = TidalPhaseCorrector()
        self.thd_analyzer = THDCorrelationAnalyzer(
            ensemble_dir=self.config.ensemble_results_dir,
            min_samples=self.config.region_pairs and 20 or 20,  # Could be configurable
        )

        logger.info("TransPacificAnalyzer initialized")

    def run_analysis(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> TransPacificAnalysisResult:
        """
        Run complete trans-Pacific correlation analysis.

        Args:
            start_date: Start of analysis period (default: 31 days ago)
            end_date: End of analysis period (default: today)

        Returns:
            TransPacificAnalysisResult with all analysis outputs
        """
        if not self.config.enabled:
            logger.warning("Trans-Pacific analysis requested but feature is disabled")
            return TransPacificAnalysisResult(
                timestamp=datetime.now(),
                config={'enabled': False},
                tidal_results={},
                correlation_results={},
                summary="Feature disabled in configuration",
            )

        # Default date range: last 31 days
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=31)

        logger.info(f"Running trans-Pacific analysis: {start_date.date()} to {end_date.date()}")
        logger.info(f"Region pairs: {self.config.region_pairs}")

        tidal_results = {}
        correlation_results = {}
        notes = []

        # Analyze each configured region pair
        for region_a, region_b in self.config.region_pairs:
            pair_key = f"{region_a}|{region_b}"
            logger.info(f"Analyzing pair: {region_a} ↔ {region_b}")

            # Step 1: Tidal phase analysis
            if self.config.tidal_phase_correction:
                tidal_result = self.tidal_corrector.compute_phase_difference(region_a, region_b)
                tidal_results[pair_key] = tidal_result

                if tidal_result.is_opposing_phase:
                    notes.append(f"{pair_key}: Opposing M2 phases ({tidal_result.phase_difference_degrees:.1f}°) - high tidal aliasing risk")
                else:
                    notes.append(f"{pair_key}: Non-opposing phases ({tidal_result.phase_difference_degrees:.1f}°) - low tidal aliasing risk")

            # Step 2: THD correlation analysis
            if self.config.thd_thd_correlation:
                correlation_result = self.thd_analyzer.analyze_region_pair(
                    region_a, region_b,
                    start_date, end_date,
                    apply_tidal_correction=self.config.tidal_phase_correction,
                    compute_lags=True,
                )
                correlation_results[pair_key] = correlation_result

                if correlation_result.is_significant:
                    notes.append(f"{pair_key}: Significant correlation r={correlation_result.pearson_r:.3f} (p={correlation_result.pearson_p:.4f})")

                if correlation_result.tidal_correction:
                    tc = correlation_result.tidal_correction
                    if tc.h0_supported:
                        notes.append(f"{pair_key}: H0 SUPPORTED - tidal aliasing explains correlation")
                    elif tc.is_opposing_phase and correlation_result.is_significant:
                        notes.append(f"{pair_key}: H0 REJECTED - correlation persists after tidal correction")

        # Compute summary statistics
        n_pairs = len(self.config.region_pairs)
        n_significant = sum(1 for r in correlation_results.values() if r.is_significant)
        n_h0_supported = sum(1 for r in correlation_results.values()
                            if r.tidal_correction and r.tidal_correction.h0_supported)
        n_h0_rejected = sum(1 for r in correlation_results.values()
                           if r.tidal_correction and r.tidal_correction.is_opposing_phase
                           and r.is_significant and not r.tidal_correction.h0_supported)

        # Generate summary
        summary_parts = [
            f"Analyzed {n_pairs} region pairs from {start_date.date()} to {end_date.date()}",
            f"Significant correlations: {n_significant}/{n_pairs}",
        ]

        if self.config.tidal_phase_correction:
            summary_parts.append(f"H0 (tidal aliasing) supported: {n_h0_supported}/{n_pairs}")
            summary_parts.append(f"H0 rejected (genuine correlation): {n_h0_rejected}/{n_pairs}")

            if n_h0_rejected > 0:
                summary_parts.append("⚠️ Evidence for genuine trans-Pacific coupling found")
            elif n_significant > 0 and n_h0_supported == n_significant:
                summary_parts.append("✓ All significant correlations explained by tidal aliasing")

        result = TransPacificAnalysisResult(
            timestamp=datetime.now(),
            config={
                'enabled': self.config.enabled,
                'tidal_phase_correction': self.config.tidal_phase_correction,
                'thd_thd_correlation': self.config.thd_thd_correlation,
                'region_pairs': [list(p) for p in self.config.region_pairs],
                'date_range': [str(start_date.date()), str(end_date.date())],
            },
            tidal_results=tidal_results,
            correlation_results=correlation_results,
            n_pairs_analyzed=n_pairs,
            n_significant_correlations=n_significant,
            n_h0_supported=n_h0_supported,
            n_h0_rejected=n_h0_rejected,
            summary=' | '.join(summary_parts),
            notes=notes,
        )

        return result


def run_trans_pacific_analysis(
    ensemble_results: Optional[Dict] = None,
    target_date: Optional[datetime] = None,
    output_dir: Optional[Path] = None,
) -> Optional[TransPacificAnalysisResult]:
    """
    Entry point for trans-Pacific analysis from run_ensemble_daily.py.

    This function is called when the feature is enabled and backtest_only is False.

    Args:
        ensemble_results: Current day's ensemble results (optional context)
        target_date: Target date for analysis
        output_dir: Directory for output files

    Returns:
        TransPacificAnalysisResult if successful, None if disabled or failed
    """
    config = get_config()

    if not config.enabled:
        logger.debug("Trans-Pacific analysis skipped: feature disabled")
        return None

    if config.backtest_only:
        logger.debug("Trans-Pacific analysis skipped: backtest_only mode")
        return None

    try:
        analyzer = TransPacificAnalyzer(config)

        # Default to last 31 days ending at target_date
        end_date = target_date or datetime.now()
        start_date = end_date - timedelta(days=31)

        result = analyzer.run_analysis(start_date, end_date)

        # Save results
        out_dir = output_dir or config.get_output_dir('correlation_results')
        date_str = end_date.strftime('%Y-%m-%d')
        result.save(out_dir / f"trans_pacific_{date_str}.json")

        logger.info(f"Trans-Pacific analysis complete: {result.summary}")
        return result

    except Exception as e:
        logger.error(f"Trans-Pacific analysis failed: {e}")
        return None


def run_backtest(
    start_date: datetime,
    end_date: datetime,
    output_dir: Optional[Path] = None,
) -> TransPacificAnalysisResult:
    """
    Run trans-Pacific analysis in backtest mode.

    This bypasses the backtest_only check for explicit backtest runs.

    Args:
        start_date: Start of backtest period
        end_date: End of backtest period
        output_dir: Directory for output files

    Returns:
        TransPacificAnalysisResult
    """
    config = get_config()

    # Force enable for backtest
    if not config.enabled:
        logger.warning("Force-enabling trans-Pacific for backtest run")
        config.enabled = True

    analyzer = TransPacificAnalyzer(config)
    result = analyzer.run_analysis(start_date, end_date)

    # Save results
    out_dir = output_dir or config.get_output_dir('backtest_results')
    date_str = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    result.save(out_dir / f"backtest_trans_pacific_{date_str}.json")

    return result


# Quick test when run directly
if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("=== Trans-Pacific Correlation Analysis ===\n")

    # Force enable for testing
    config = get_config(force_reload=True)
    config.enabled = True

    analyzer = TransPacificAnalyzer(config)

    # Default date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=31)

    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Region pairs: {config.region_pairs}\n")

    result = analyzer.run_analysis(start_date, end_date)

    print("\n=== Results ===")
    print(f"Summary: {result.summary}")
    print(f"\nNotes:")
    for note in result.notes:
        print(f"  - {note}")

    print(f"\n=== Correlation Details ===")
    for key, corr in result.correlation_results.items():
        print(f"\n{key}:")
        print(f"  Pearson r: {corr.pearson_r:.4f}")
        print(f"  p-value: {corr.pearson_p:.6f}")
        print(f"  Significant: {corr.is_significant}")
        print(f"  Valid pairs: {corr.n_valid_pairs}")
        if corr.tidal_correction:
            print(f"  H0 supported: {corr.tidal_correction.h0_supported}")

    # Save results
    out_dir = EXPERIMENTAL_DATA_DIR / 'test_results'
    result.save(out_dir / 'test_trans_pacific_analysis.json')
    print(f"\nResults saved to {out_dir}")
