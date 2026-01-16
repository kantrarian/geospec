#!/usr/bin/env python3
"""
run_backtest.py - CLI Wrapper for GeoSpec Backtest

Convenience wrapper for running backtests from the validation directory.

Usage:
    python run_backtest.py --start 2019-01-01 --end 2019-12-31
    python run_backtest.py --start 2019-01-01 --end 2019-12-31 --config ../monitoring/config/backtest_config.yaml

Author: R.J. Mathews / Claude
Date: January 2026
"""

import sys
from pathlib import Path

# Add monitoring/src to path
src_dir = Path(__file__).parent.parent / 'monitoring' / 'src'
sys.path.insert(0, str(src_dir))

# Import and run main
from backtest import main

if __name__ == '__main__':
    sys.exit(main())
