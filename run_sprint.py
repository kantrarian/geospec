#!/usr/bin/env python3
"""
run_sprint.py
Master execution script for the GeoSpec Lambda_geo Validation Sprint.

Execute the complete 5-day sprint pipeline:
1. Data Acquisition (synthetic strain data generation)
2. Core Lambda_geo Analysis
3. Validation against historical earthquakes
4. Spatial analysis and visualization
5. Patent evidence generation

Author: R.J. Mathews
Date: January 2026
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding for Unicode
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def print_banner(title: str, char: str = "="):
    """Print a formatted banner."""
    width = 70
    print("\n" + char * width)
    print(f" {title}")
    print(char * width)


def print_step(step_num: int, title: str):
    """Print a step header."""
    print(f"\n{'='*70}")
    print(f"STEP {step_num}: {title}")
    print(f"{'='*70}")


def run_data_acquisition():
    """Step 1: Generate synthetic strain data."""
    print_step(1, "DATA ACQUISITION")
    
    from data_acquisition import main as acquire_data
    acquire_data()
    
    print("\n[OK] Data acquisition complete")


def run_validation():
    """Step 2: Run Lambda_geo validation pipeline."""
    print_step(2, "Lambda_geo VALIDATION")
    
    from validate_lambda_geo import main as validate
    validate()
    
    print("\n[OK] Validation complete")


def run_spatial_analysis():
    """Step 3: Generate spatial analysis figures."""
    print_step(3, "SPATIAL ANALYSIS")
    
    from spatial_analysis import main as spatial
    spatial()
    
    print("\n[OK] Spatial analysis complete")


def run_patent_evidence():
    """Step 4: Generate patent evidence package."""
    print_step(4, "PATENT EVIDENCE GENERATION")
    
    from generate_patent_evidence import generate_evidence_package
    generate_evidence_package()
    
    print("\n[OK] Patent evidence package complete")


def print_summary():
    """Print final summary of outputs."""
    print_banner("SPRINT COMPLETE", "=")
    
    results_dir = PROJECT_ROOT / "results"
    figures_dir = PROJECT_ROOT / "figures"
    docs_dir = PROJECT_ROOT / "docs"
    
    print("\n[OUTPUT LOCATIONS]")
    print(f"   Results:  {results_dir}")
    print(f"   Figures:  {figures_dir}")
    print(f"   Docs:     {docs_dir}")
    
    print("\n[KEY OUTPUTS]")
    
    # List results
    if results_dir.exists():
        for f in results_dir.glob("*.json"):
            print(f"   - {f.name}")
        for f in results_dir.glob("*.png"):
            print(f"   - {f.name}")
    
    # List figures
    if figures_dir.exists():
        for f in figures_dir.glob("*.png"):
            print(f"   - {f.name}")
    
    # List docs
    if docs_dir.exists():
        for f in docs_dir.glob("*"):
            print(f"   - {f.name}")
    
    print("\n" + "="*70)
    print("SUCCESS CRITERIA CHECKLIST:")
    print("="*70)
    print("""
| Criterion               | Target        | Check validation_summary.json |
|-------------------------|---------------|-------------------------------|
| Detection lead time     | 24-72 hours   | first_detection_hours_before  |
| Statistical significance| Z > 2.0       | zscore_max                    |
| Spatial localization    | < 200 km      | epicenter_proximity_at_peak   |
| Amplification factor    | > 5x          | amplification_factor          |
| Success rate            | > 66%         | success_rate                  |
""")
    
    print("="*70)
    print(f"Sprint completed at: {datetime.now().isoformat()}")
    print("="*70)


def main():
    """Execute the complete GeoSpec Λ_geo Validation Sprint."""
    
    print_banner("GEOSPEC Lambda_geo VALIDATION SPRINT", "=")
    print(f"""
    Author: R.J. Mathews
    Date: {datetime.now().strftime('%Y-%m-%d')}
    
    Objective: Validate Lambda_geo = ||[E, E_dot]||_F as earthquake precursor diagnostic
    Success Criteria: Detect anomalous Lambda_geo 24-72 hours before M>6 earthquakes
    
    Target Earthquakes:
      • 2011 Tohoku M9.0
      • 2019 Ridgecrest M7.1
      • 2023 Turkey-Syria M7.8
    """)
    
    start_time = time.time()
    
    try:
        # Step 1: Data Acquisition
        run_data_acquisition()
        
        # Step 2: Validation
        run_validation()
        
        # Step 3: Spatial Analysis
        run_spatial_analysis()
        
        # Step 4: Patent Evidence
        run_patent_evidence()
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    elapsed = time.time() - start_time
    
    # Print summary
    print_summary()
    
    print(f"\n[TIME] Total execution time: {elapsed:.1f} seconds")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
