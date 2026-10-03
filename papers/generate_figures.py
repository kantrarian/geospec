#!/usr/bin/env python3
"""
Generate figures for the Lambda_geo technical paper.
All figures use white backgrounds for professional presentation.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon
from matplotlib.collections import PatchCollection
import numpy as np
from pathlib import Path

# Set style for all figures
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def create_system_architecture():
    """Create system architecture overview diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('GeoSpec Lambda_geo Monitoring System Architecture',
                 fontsize=16, fontweight='bold', pad=20)

    # Colors
    data_color = '#E3F2FD'
    process_color = '#E8F5E9'
    output_color = '#FFF3E0'
    alert_color = '#FFEBEE'

    # Data Sources (left column)
    ax.add_patch(FancyBboxPatch((0.5, 7.5), 3, 1.5, boxstyle="round,pad=0.1",
                                 facecolor=data_color, edgecolor='#1565C0', linewidth=2))
    ax.text(2, 8.25, 'Nevada Geodetic Lab\n(NGL)', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(2, 7.7, 'IGS20 / IGS14 Reference\n17,000+ GPS Stations', ha='center', va='center', fontsize=8)

    ax.add_patch(FancyBboxPatch((0.5, 5.5), 3, 1.5, boxstyle="round,pad=0.1",
                                 facecolor=data_color, edgecolor='#1565C0', linewidth=2))
    ax.text(2, 6.25, 'Station Catalog', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(2, 5.7, 'llh.out (lat/lon/height)', ha='center', va='center', fontsize=8)

    # Processing Pipeline (center)
    ax.add_patch(FancyBboxPatch((4.5, 7.5), 3, 1.5, boxstyle="round,pad=0.1",
                                 facecolor=process_color, edgecolor='#2E7D32', linewidth=2))
    ax.text(6, 8.25, 'GPS Data Acquisition', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(6, 7.7, 'Download & Parse\nTENV/TENV3 Files', ha='center', va='center', fontsize=8)

    ax.add_patch(FancyBboxPatch((4.5, 5.5), 3, 1.5, boxstyle="round,pad=0.1",
                                 facecolor=process_color, edgecolor='#2E7D32', linewidth=2))
    ax.text(6, 6.25, 'Velocity Computation', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(6, 5.7, 'Temporal Derivatives\nSavitzky-Golay Filter', ha='center', va='center', fontsize=8)

    ax.add_patch(FancyBboxPatch((4.5, 3.5), 3, 1.5, boxstyle="round,pad=0.1",
                                 facecolor=process_color, edgecolor='#2E7D32', linewidth=2))
    ax.text(6, 4.25, 'Strain Tensor Field', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(6, 3.7, 'Delaunay Triangulation\nGradient Computation', ha='center', va='center', fontsize=8)

    ax.add_patch(FancyBboxPatch((4.5, 1.5), 3, 1.5, boxstyle="round,pad=0.1",
                                 facecolor='#E1BEE7', edgecolor='#7B1FA2', linewidth=2))
    ax.text(6, 2.25, 'Lambda_geo\nComputation', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(6, 1.7, r'$\Lambda_{geo} = ||[E, \dot{E}]||_F$', ha='center', va='center', fontsize=10)

    # Analysis & Output (right)
    ax.add_patch(FancyBboxPatch((8.5, 6.5), 3, 1.5, boxstyle="round,pad=0.1",
                                 facecolor=output_color, edgecolor='#EF6C00', linewidth=2))
    ax.text(10, 7.25, 'Rolling Baseline', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(10, 6.7, '90-day Lookback\nSeasonal Detrending', ha='center', va='center', fontsize=8)

    ax.add_patch(FancyBboxPatch((8.5, 4.5), 3, 1.5, boxstyle="round,pad=0.1",
                                 facecolor=output_color, edgecolor='#EF6C00', linewidth=2))
    ax.text(10, 5.25, 'Spatial Coherence', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(10, 4.7, 'Cluster Detection\nFraction Elevated', ha='center', va='center', fontsize=8)

    ax.add_patch(FancyBboxPatch((8.5, 2.5), 3, 1.5, boxstyle="round,pad=0.1",
                                 facecolor=alert_color, edgecolor='#C62828', linewidth=2))
    ax.text(10, 3.25, 'Alert State Machine', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(10, 2.7, 'Tier 0-3 Classification\nHysteresis Logic', ha='center', va='center', fontsize=8)

    # Output storage
    ax.add_patch(FancyBboxPatch((8.5, 0.5), 3, 1.5, boxstyle="round,pad=0.1",
                                 facecolor='#F3E5F5', edgecolor='#6A1B9A', linewidth=2))
    ax.text(10, 1.25, 'Results Storage', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(10, 0.7, 'CSV, JSON, SQLite\nDaily Logs', ha='center', va='center', fontsize=8)

    # Arrows
    arrow_props = dict(arrowstyle='->', color='#424242', lw=2)

    # Data to processing
    ax.annotate('', xy=(4.5, 8.25), xytext=(3.5, 8.25), arrowprops=arrow_props)
    ax.annotate('', xy=(4.5, 6.25), xytext=(3.5, 6.25), arrowprops=arrow_props)

    # Processing chain
    ax.annotate('', xy=(6, 7.5), xytext=(6, 7.0), arrowprops=arrow_props)
    ax.annotate('', xy=(6, 5.5), xytext=(6, 5.0), arrowprops=arrow_props)
    ax.annotate('', xy=(6, 3.5), xytext=(6, 3.0), arrowprops=arrow_props)

    # Processing to analysis
    ax.annotate('', xy=(8.5, 7.25), xytext=(7.5, 4.25), arrowprops=arrow_props)
    ax.annotate('', xy=(8.5, 5.25), xytext=(7.5, 3.0), arrowprops=arrow_props)
    ax.annotate('', xy=(8.5, 3.25), xytext=(7.5, 2.25), arrowprops=arrow_props)

    # Analysis chain
    ax.annotate('', xy=(10, 6.5), xytext=(10, 6.0), arrowprops=arrow_props)
    ax.annotate('', xy=(10, 4.5), xytext=(10, 4.0), arrowprops=arrow_props)
    ax.annotate('', xy=(10, 2.5), xytext=(10, 2.0), arrowprops=arrow_props)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=data_color, edgecolor='#1565C0', label='Data Sources'),
        mpatches.Patch(facecolor=process_color, edgecolor='#2E7D32', label='Processing'),
        mpatches.Patch(facecolor='#E1BEE7', edgecolor='#7B1FA2', label='Core Algorithm'),
        mpatches.Patch(facecolor=output_color, edgecolor='#EF6C00', label='Analysis'),
        mpatches.Patch(facecolor=alert_color, edgecolor='#C62828', label='Alerting'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'system_architecture.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: system_architecture.png")


def create_lambda_geo_pipeline():
    """Create detailed Lambda_geo computation pipeline diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title(r'Lambda$_{geo}$ Computation Pipeline: GPS to Strain Tensor Commutator',
                 fontsize=14, fontweight='bold', pad=20)

    # Step boxes
    steps = [
        (1, 6, 'GPS Positions\n(E, N, U)', r'$\vec{x}_i(t) = [E_i, N_i, U_i]$', '#BBDEFB'),
        (4, 6, 'Velocities', r'$\vec{v}_i = \frac{d\vec{x}_i}{dt}$', '#C8E6C9'),
        (7, 6, 'Triangulation', 'Delaunay\nMesh', '#FFF9C4'),
        (10, 6, 'Strain Tensor', r'$E_{ij} = \frac{1}{2}(\nabla u + \nabla u^T)$', '#FFCCBC'),
        (4, 3, 'Strain Rate', r'$\dot{E}_{ij} = \frac{dE_{ij}}{dt}$', '#D1C4E9'),
        (7, 3, 'Commutator', r'$[E, \dot{E}] = E\dot{E} - \dot{E}E$', '#B2EBF2'),
        (10, 3, r'$\Lambda_{geo}$', r'$||[E, \dot{E}]||_F$', '#F8BBD9'),
    ]

    for x, y, title, formula, color in steps:
        ax.add_patch(FancyBboxPatch((x-1, y-0.8), 2.5, 1.6, boxstyle="round,pad=0.1",
                                     facecolor=color, edgecolor='#424242', linewidth=1.5))
        ax.text(x+0.25, y+0.3, title, ha='center', va='center', fontsize=9, fontweight='bold')
        ax.text(x+0.25, y-0.3, formula, ha='center', va='center', fontsize=9)

    # Arrows
    arrow_props = dict(arrowstyle='->', color='#424242', lw=2)
    ax.annotate('', xy=(3, 6), xytext=(2.5, 6), arrowprops=arrow_props)
    ax.annotate('', xy=(6, 6), xytext=(5.5, 6), arrowprops=arrow_props)
    ax.annotate('', xy=(9, 6), xytext=(8.5, 6), arrowprops=arrow_props)
    ax.annotate('', xy=(4.25, 3.8), xytext=(10.25, 5.2), arrowprops=arrow_props)
    ax.annotate('', xy=(6, 3), xytext=(5.5, 3), arrowprops=arrow_props)
    ax.annotate('', xy=(9, 3), xytext=(8.5, 3), arrowprops=arrow_props)

    # Math explanation box
    ax.add_patch(FancyBboxPatch((0.5, 0.3), 13, 1.8, boxstyle="round,pad=0.1",
                                 facecolor='#FAFAFA', edgecolor='#9E9E9E', linewidth=1))
    explanation = (
        r"Mathematical Foundation:  The strain tensor $E$ captures instantaneous deformation, while $\dot{E}$ captures deformation rate."
        + "\n" +
        r"The commutator $[E, \dot{E}]$ is non-zero when strain directions rotate over time — a signature of active fault mechanics."
        + "\n" +
        r"The Frobenius norm $||\cdot||_F$ provides a scalar measure of this non-commutativity: $\Lambda_{geo} = \sqrt{\sum_{ij}[E,\dot{E}]_{ij}^2}$"
    )
    ax.text(7, 1.2, explanation, ha='center', va='center', fontsize=9,
            family='serif', style='italic')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'lambda_geo_pipeline.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: lambda_geo_pipeline.png")


def create_triangulation_diagram():
    """Create Delaunay triangulation visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Generate sample station positions
    np.random.seed(42)
    n_stations = 15
    x = np.random.uniform(0, 10, n_stations)
    y = np.random.uniform(0, 8, n_stations)

    # Add some displacement vectors
    dx = np.random.uniform(-0.3, 0.3, n_stations)
    dy = np.random.uniform(-0.3, 0.3, n_stations)

    from scipy.spatial import Delaunay
    tri = Delaunay(np.column_stack([x, y]))

    # Panel 1: GPS Stations
    ax1 = axes[0]
    ax1.scatter(x, y, s=100, c='#1565C0', zorder=5, edgecolors='white', linewidths=2)
    for i in range(n_stations):
        ax1.annotate(f'S{i+1}', (x[i]+0.2, y[i]+0.2), fontsize=8)
    ax1.set_xlim(-0.5, 10.5)
    ax1.set_ylim(-0.5, 8.5)
    ax1.set_title('(a) GPS Station Network', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Easting (km)')
    ax1.set_ylabel('Northing (km)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Delaunay Triangulation
    ax2 = axes[1]
    ax2.triplot(x, y, tri.simplices, 'k-', lw=1, alpha=0.7)
    ax2.scatter(x, y, s=100, c='#1565C0', zorder=5, edgecolors='white', linewidths=2)

    # Highlight one triangle
    t_idx = 5
    triangle = tri.simplices[t_idx]
    tx = x[triangle]
    ty = y[triangle]
    ax2.fill(tx, ty, alpha=0.3, color='#4CAF50')
    ax2.plot(np.append(tx, tx[0]), np.append(ty, ty[0]), 'g-', lw=3)

    ax2.set_xlim(-0.5, 10.5)
    ax2.set_ylim(-0.5, 8.5)
    ax2.set_title('(b) Delaunay Triangulation', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Easting (km)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Strain computation on triangle
    ax3 = axes[2]
    # Show the highlighted triangle enlarged
    cx, cy = np.mean(tx), np.mean(ty)
    scale = 2.5
    tx_scaled = (tx - cx) * scale + 5
    ty_scaled = (ty - cy) * scale + 4

    ax3.fill(tx_scaled, ty_scaled, alpha=0.2, color='#4CAF50')
    ax3.plot(np.append(tx_scaled, tx_scaled[0]), np.append(ty_scaled, ty_scaled[0]), 'g-', lw=2)
    ax3.scatter(tx_scaled, ty_scaled, s=150, c='#1565C0', zorder=5, edgecolors='white', linewidths=2)

    # Add velocity vectors
    for i in range(3):
        orig_idx = triangle[i]
        ax3.quiver(tx_scaled[i], ty_scaled[i], dx[orig_idx]*3, dy[orig_idx]*3,
                  angles='xy', scale_units='xy', scale=1, color='#D32F2F', width=0.03, zorder=10)

    # Add strain tensor visualization
    ax3.text(5, 7.5, r'Strain: $E_{ij} = \frac{1}{2}\left(\frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i}\right)$',
            ha='center', fontsize=10)
    ax3.text(5, 0.8, 'Velocity vectors (red)\nStrain computed from\ngradients within triangle',
            ha='center', fontsize=9, style='italic')

    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 8)
    ax3.set_title('(c) Strain Tensor Computation', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Local coordinates')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'triangulation_strain.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: triangulation_strain.png")


def create_alert_tiers():
    """Create alert tier state diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Alert Tier State Machine with Hysteresis', fontsize=14, fontweight='bold', pad=20)

    # Tier definitions
    tiers = [
        (2, 6, 'TIER 0', 'NORMAL', '#C8E6C9', '#2E7D32', r'$\Lambda_{max} < 2\times$ baseline'),
        (6, 6, 'TIER 1', 'WATCH', '#FFF9C4', '#F57F17', r'$2\times \leq \Lambda_{max} < 5\times$'),
        (10, 6, 'TIER 2', 'ELEVATED', '#FFCCBC', '#E64A19', r'$5\times \leq \Lambda_{max} < 10\times$'),
        (6, 2, 'TIER 3', 'HIGH', '#FFCDD2', '#C62828', r'$\Lambda_{max} \geq 10\times$ baseline'),
    ]

    for x, y, tier, status, fill, edge, threshold in tiers:
        # Draw circle
        circle = plt.Circle((x, y), 1.2, facecolor=fill, edgecolor=edge, linewidth=3)
        ax.add_patch(circle)
        ax.text(x, y+0.4, tier, ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(x, y, status, ha='center', va='center', fontsize=10)
        ax.text(x, y-0.5, threshold, ha='center', va='center', fontsize=8)

    # Transitions with conditions
    arrow_props_up = dict(arrowstyle='->', color='#C62828', lw=2,
                          connectionstyle='arc3,rad=0.2')
    arrow_props_down = dict(arrowstyle='->', color='#2E7D32', lw=2,
                            connectionstyle='arc3,rad=-0.2')

    # T0 -> T1
    ax.annotate('', xy=(4.8, 6.3), xytext=(3.2, 6.3), arrowprops=arrow_props_up)
    ax.text(4, 7, r'$\Lambda > 2\times$', fontsize=8, ha='center', color='#C62828')
    # T1 -> T0
    ax.annotate('', xy=(3.2, 5.7), xytext=(4.8, 5.7), arrowprops=arrow_props_down)
    ax.text(4, 5.2, r'$\Lambda < 1.5\times$', fontsize=8, ha='center', color='#2E7D32')

    # T1 -> T2
    ax.annotate('', xy=(8.8, 6.3), xytext=(7.2, 6.3), arrowprops=arrow_props_up)
    ax.text(8, 7, r'$\Lambda > 5\times$', fontsize=8, ha='center', color='#C62828')
    # T2 -> T1
    ax.annotate('', xy=(7.2, 5.7), xytext=(8.8, 5.7), arrowprops=arrow_props_down)
    ax.text(8, 5.2, r'$\Lambda < 4\times$', fontsize=8, ha='center', color='#2E7D32')

    # T1 -> T3
    ax.annotate('', xy=(6.3, 3.2), xytext=(6.3, 4.8), arrowprops=arrow_props_up)
    ax.text(7, 4, r'$\Lambda > 10\times$' + '\n+ coherent', fontsize=8, ha='left', color='#C62828')
    # T3 -> T1
    ax.annotate('', xy=(5.7, 4.8), xytext=(5.7, 3.2), arrowprops=arrow_props_down)
    ax.text(4.5, 4, r'$\Lambda < 8\times$', fontsize=8, ha='right', color='#2E7D32')

    # T2 -> T3
    ax.annotate('', xy=(7.2, 2.3), xytext=(8.8, 5.0), arrowprops=arrow_props_up)
    ax.text(9, 3.5, r'$\Lambda > 10\times$', fontsize=8, color='#C62828')

    # Legend / explanation
    ax.add_patch(FancyBboxPatch((0.3, 0.2), 11.4, 1.3, boxstyle="round,pad=0.1",
                                 facecolor='#FAFAFA', edgecolor='#9E9E9E', linewidth=1))
    ax.text(6, 0.85,
            'Hysteresis prevents oscillation: upward thresholds are higher than downward thresholds.\n'
            'Spatial coherence required for TIER 3: anomaly must span multiple triangles (not isolated noise).',
            ha='center', va='center', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'alert_tiers.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: alert_tiers.png")


def create_validation_results():
    """Create validation results chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Data from validation
    earthquakes = ['Tohoku\n2011\nM9.0', 'Chile\n2010\nM8.8', 'Turkey\n2023\nM7.8',
                   'Ridgecrest\n2019\nM7.1', 'Morocco\n2023\nM6.8']
    amplification = [7999, 485, 1336, 5489, 2.8]
    lead_times = [143.5, 186.8, 139.5, 141.3, 208.6]
    success = [True, True, True, True, False]

    colors = ['#4CAF50' if s else '#F44336' for s in success]

    # Panel 1: Amplification
    ax1 = axes[0]
    bars1 = ax1.bar(earthquakes, amplification, color=colors, edgecolor='black', linewidth=1)
    ax1.set_yscale('log')
    ax1.axhline(y=5, color='#FF9800', linestyle='--', linewidth=2, label='Detection threshold (5×)')
    ax1.set_ylabel('Amplification (× baseline)', fontsize=11)
    ax1.set_title('(a) Lambda_geo Amplification Before Earthquakes', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars1, amplification):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height*1.2,
                f'{val:.0f}×' if val > 10 else f'{val:.1f}×',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Panel 2: Lead times
    ax2 = axes[1]
    bars2 = ax2.bar(earthquakes, lead_times, color=colors, edgecolor='black', linewidth=1)
    ax2.axhline(y=48, color='#2196F3', linestyle='--', linewidth=2, label='48-hour target')
    ax2.set_ylabel('Lead Time (hours)', fontsize=11)
    ax2.set_title('(b) Precursor Lead Time Before Mainshock', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars2, lead_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height+5,
                f'{val:.0f}h',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add success rate annotation
    fig.text(0.5, 0.02, 'Validation Results: 4/5 successful detections (80%)\n'
             'Green = Successful detection (>5× amplification)  |  Red = Not detected',
             ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(FIGURES_DIR / 'validation_results.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: validation_results.png")


def create_baseline_rolling():
    """Create rolling baseline computation diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Generate synthetic Lambda_geo time series
    np.random.seed(42)
    days = 150
    t = np.arange(days)

    # Base signal with seasonal variation
    seasonal = 0.02 * np.sin(2 * np.pi * t / 365)
    noise = 0.01 * np.random.randn(days)
    lambda_geo = 0.05 + seasonal + noise

    # Add anomaly around day 120
    anomaly_start = 115
    lambda_geo[anomaly_start:anomaly_start+10] += np.linspace(0, 0.3, 10)
    lambda_geo[anomaly_start+10:anomaly_start+20] += np.linspace(0.3, 0.1, 10)

    # Compute rolling baseline (90-day, 14-day exclusion)
    baseline = np.zeros(days)
    for i in range(90, days):
        window = lambda_geo[i-90:i-14]
        baseline[i] = np.median(window)
    baseline[:90] = np.nan

    # Plot
    ax.plot(t, lambda_geo, 'b-', linewidth=1.5, label=r'$\Lambda_{geo}$ time series', alpha=0.8)
    ax.plot(t, baseline, 'g-', linewidth=2, label='Rolling baseline (90d, 14d gap)')
    ax.fill_between(t, baseline * 0.5, baseline * 2, alpha=0.2, color='green',
                    label='Normal range (0.5-2× baseline)')

    # Mark regions
    ax.axvspan(90, 104, alpha=0.1, color='orange', label='Baseline window (example)')
    ax.axvspan(104, 118, alpha=0.1, color='red', label='14-day exclusion gap')
    ax.axvline(x=118, color='purple', linestyle='--', linewidth=2, label='Current day (example)')

    # Mark anomaly
    ax.annotate('Anomaly detected\n(7× baseline)', xy=(122, 0.35), xytext=(135, 0.38),
               fontsize=10, ha='center',
               arrowprops=dict(arrowstyle='->', color='red'))

    ax.set_xlabel('Days', fontsize=11)
    ax.set_ylabel(r'$\Lambda_{geo}$ value', fontsize=11)
    ax.set_title('Rolling Baseline Computation with Seasonal Detrending', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, days)
    ax.set_ylim(0, 0.45)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'rolling_baseline.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: rolling_baseline.png")


def create_region_map():
    """Create monitored regions map (simplified)."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # World map outline (simplified)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 75)

    # Region markers with data
    regions = [
        ('SoCal SAF\nMojave', -117, 34.5, 35, 56, '#4CAF50'),
        ('SoCal SAF\nCoachella', -116, 33.5, 36, 60, '#4CAF50'),
        ('NorCal\nHayward', -122, 37.5, 23, 37, '#4CAF50'),
        ('Cascadia', -124, 46, 30, 49, '#4CAF50'),
        ('Tokyo\nKanto', 139.5, 35.5, 41, 72, '#4CAF50'),
        ('Istanbul\nMarmara', 29, 40.5, 5, 4, '#FF9800'),  # Orange for sparse
    ]

    for name, lon, lat, stations, triangles, color in regions:
        ax.scatter(lon, lat, s=400, c=color, edgecolors='black', linewidths=2, zorder=5)
        ax.annotate(f'{name}\n({stations} sta, {triangles} tri)',
                   (lon, lat), xytext=(15, 15), textcoords='offset points',
                   fontsize=9, ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
                   arrowprops=dict(arrowstyle='->', color='gray'))

    # Add plate boundaries (simplified major faults)
    # San Andreas
    ax.plot([-125, -114], [40, 32], 'r-', linewidth=3, alpha=0.5, label='Major plate boundaries')
    # Cascadia
    ax.plot([-130, -122], [50, 40], 'r-', linewidth=3, alpha=0.5)
    # Japan Trench
    ax.plot([145, 140], [40, 30], 'r-', linewidth=3, alpha=0.5)
    # North Anatolian
    ax.plot([25, 40], [40, 40], 'r-', linewidth=3, alpha=0.5)

    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title('GeoSpec Monitored Regions (Current Pilot Network)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left')

    # Add data source note
    ax.text(0, -55, 'Data Source: Nevada Geodetic Laboratory (NGL) IGS20 Reference Frame\n'
            'Green = Dense coverage (>20 stations)  |  Orange = Sparse coverage (<10 stations)',
            ha='center', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'region_map.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: region_map.png")


def create_commutator_physics():
    """Create diagram explaining commutator physics."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel 1: Pure shear (commuting)
    ax1 = axes[0]
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)

    # Original square
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
    ax1.plot(square[:, 0], square[:, 1], 'b-', linewidth=2, label='Original')

    # Sheared square (E then E_dot same direction)
    shear1 = np.array([[1.2, 0], [0, 0.8]])
    sheared = square @ shear1.T
    ax1.plot(sheared[:, 0], sheared[:, 1], 'g--', linewidth=2, label='After strain')

    ax1.set_title('(a) Pure Shear\n(E and Ė aligned)', fontsize=11, fontweight='bold')
    ax1.text(0, -1.8, r'$[E, \dot{E}] = 0$' + '\nNo rotation of strain axes',
             ha='center', fontsize=10)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # Panel 2: Rotating strain (non-commuting)
    ax2 = axes[1]
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)

    ax2.plot(square[:, 0], square[:, 1], 'b-', linewidth=2, label='Original')

    # First strain
    shear_E = np.array([[1.3, 0.2], [0.2, 0.8]])
    sheared1 = square @ shear_E.T
    ax2.plot(sheared1[:, 0], sheared1[:, 1], 'orange', linewidth=2, linestyle='--', label='After E')

    # Then different strain rate
    shear_Edot = np.array([[0.9, -0.3], [-0.3, 1.1]])
    sheared2 = sheared1 @ shear_Edot.T
    ax2.plot(sheared2[:, 0], sheared2[:, 1], 'r-', linewidth=2, label='After Ė')

    ax2.set_title('(b) Rotating Strain\n(E and Ė misaligned)', fontsize=11, fontweight='bold')
    ax2.text(0, -1.8, r'$[E, \dot{E}] \neq 0$' + '\nStrain axes rotating',
             ha='center', fontsize=10)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_xlabel('x')

    # Panel 3: Physical interpretation
    ax3 = axes[2]
    ax3.axis('off')
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)

    ax3.set_title('(c) Physical Interpretation', fontsize=11, fontweight='bold')

    text = """
    The commutator [E, Ė] measures how much
    the principal strain directions rotate
    over time.

    In stable tectonic settings:
    • Strain accumulates steadily
    • Principal axes remain fixed
    • [E, Ė] ≈ 0

    Before earthquakes:
    • Stress redistributes
    • Strain directions shift
    • [E, Ė] increases → Λ_geo rises

    This is why Λ_geo serves as an
    earthquake precursor diagnostic:
    it detects the mechanical instability
    that precedes rupture.
    """

    ax3.text(5, 5, text, ha='center', va='center', fontsize=10,
             family='serif', linespacing=1.5,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5', edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'commutator_physics.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: commutator_physics.png")


if __name__ == '__main__':
    print("Generating figures for Lambda_geo technical paper...\n")

    create_system_architecture()
    create_lambda_geo_pipeline()
    create_triangulation_diagram()
    create_alert_tiers()
    create_validation_results()
    create_baseline_rolling()
    create_region_map()
    create_commutator_physics()

    print(f"\nAll figures saved to: {FIGURES_DIR}")
