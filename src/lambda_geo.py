#!/usr/bin/env python3
"""
lambda_geo.py
Core implementation of Λ_geo = ||[E, Ė]||_F diagnostic.

Based on the Navier-Stokes mathematical framework:
- Paper 1: Kinematic curvature definition
- Paper 3: Spectral lock hypothesis  
- paper_geometric_regularity.tex: Explicit formulas

Author: R.J. Mathews
Date: January 2026
"""

import numpy as np
from scipy import ndimage
from scipy.linalg import eigh
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import json


@dataclass
class LambdaGeoResult:
    """Container for Λ_geo analysis results."""
    # Time and space coordinates
    times: np.ndarray
    station_lats: np.ndarray
    station_lons: np.ndarray
    
    # Core diagnostics
    lambda_geo: np.ndarray              # Shape: (n_times, n_stations)
    lambda_geo_normalized: np.ndarray   # Normalized version
    
    # Spectral analysis
    eigenvalues: np.ndarray             # Shape: (n_times, n_stations, 3)
    spectral_gap_12: np.ndarray         # λ₁ - λ₂
    spectral_gap_13: np.ndarray         # λ₁ - λ₃
    
    # Derived quantities
    eigenframe_rotation_rate: np.ndarray
    principal_direction: np.ndarray     # e₁ vector field
    
    # Risk assessment
    risk_score: np.ndarray
    spatial_max_risk: np.ndarray        # Max over stations at each time
    
    # Metadata
    earthquake_info: Dict = field(default_factory=dict)
    computation_params: Dict = field(default_factory=dict)


class LambdaGeoAnalyzer:
    """
    Strain tensor commutator diagnostic for earthquake prediction.
    
    Implements Λ_geo = ||[E(t), Ė(t)]||_F where:
    - E(t) is the geodetic strain-rate tensor (3×3 symmetric)
    - Ė(t) = dE/dt is its time derivative
    - [A, B] = AB - BA is the matrix commutator
    
    Mathematical Foundation (from NS papers):
    -----------------------------------------
    The commutator measures eigenframe non-commutativity:
    
    ||[E, Ė]||_F² = 2 Σᵢ<ⱼ (λᵢ - λⱼ)² (Ė)ᵢⱼ²
    
    where (Ė)ᵢⱼ are off-diagonal components in the E-eigenbasis.
    
    Physical Interpretation:
    -----------------------
    - Λ_geo = 0: Principal strain directions stable ("locked" fault)
    - Λ_geo > 0: Eigenframe rotating (stress reorganizing)
    - High Λ_geo: Rapid rotation (regime transition imminent)
    """
    
    def __init__(self,
                 dt_hours: float = 1.0,
                 smoothing_window: int = 3,
                 derivative_method: str = 'central',
                 normalize: bool = True):
        """
        Initialize the Λ_geo analyzer.
        
        Args:
            dt_hours: Time step between strain tensor samples
            smoothing_window: Gaussian smoothing sigma (samples)
            derivative_method: 'central', 'forward', or 'savgol'
            normalize: Whether to compute normalized Λ_geo
        """
        self.dt = dt_hours
        self.smoothing_window = smoothing_window
        self.derivative_method = derivative_method
        self.normalize = normalize
    
    # =========================================================================
    # CORE COMPUTATION
    # =========================================================================
    
    def compute_time_derivative(self, E: np.ndarray) -> np.ndarray:
        """
        Compute Ė = dE/dt.
        
        For quasi-static tectonics, no material derivative needed.
        This is SIMPLER than CFD (no advection term).
        
        Args:
            E: Strain tensor time series, shape (n_times, ..., 3, 3)
            
        Returns:
            E_dot: Time derivative, same shape as E
        """
        E_dot = np.zeros_like(E)
        
        if self.derivative_method == 'central':
            # Central difference: (E[t+1] - E[t-1]) / 2dt
            E_dot[1:-1] = (E[2:] - E[:-2]) / (2 * self.dt)
            E_dot[0] = (E[1] - E[0]) / self.dt
            E_dot[-1] = (E[-1] - E[-2]) / self.dt
            
        elif self.derivative_method == 'forward':
            # Forward difference: (E[t+1] - E[t]) / dt
            E_dot[:-1] = (E[1:] - E[:-1]) / self.dt
            E_dot[-1] = E_dot[-2]
            
        elif self.derivative_method == 'savgol':
            # Savitzky-Golay filter for smoother derivative
            from scipy.signal import savgol_filter
            window = min(7, len(E) // 2 * 2 - 1)  # Must be odd
            if window >= 3:
                for i in range(3):
                    for j in range(3):
                        E_dot[:, :, i, j] = savgol_filter(
                            E[:, :, i, j], window, 2, deriv=1, delta=self.dt, axis=0
                        )
            else:
                # Fall back to central difference
                E_dot[1:-1] = (E[2:] - E[:-2]) / (2 * self.dt)
                E_dot[0] = (E[1] - E[0]) / self.dt
                E_dot[-1] = (E[-1] - E[-2]) / self.dt
        
        return E_dot
    
    def compute_commutator(self, E: np.ndarray, E_dot: np.ndarray) -> np.ndarray:
        """
        Compute [E, Ė] = E·Ė - Ė·E.
        
        This is THE fundamental operation from the umbrella patent.
        
        Args:
            E: Strain tensor, shape (..., 3, 3)
            E_dot: Time derivative, shape (..., 3, 3)
            
        Returns:
            Commutator [E, Ė], shape (..., 3, 3)
        """
        # Einstein summation for batch matrix multiplication
        # E @ E_dot: ...ij, ...jk -> ...ik
        term1 = np.einsum('...ij,...jk->...ik', E, E_dot)
        term2 = np.einsum('...ij,...jk->...ik', E_dot, E)
        return term1 - term2
    
    def compute_frobenius_norm(self, M: np.ndarray) -> np.ndarray:
        """
        Compute Frobenius norm ||M||_F = sqrt(Σᵢⱼ Mᵢⱼ²).
        
        Args:
            M: Matrix field, shape (..., 3, 3)
            
        Returns:
            Frobenius norm, shape (...)
        """
        return np.sqrt(np.einsum('...ij,...ij->...', M, M))
    
    def compute_lambda_geo(self, 
                           E: np.ndarray,
                           E_dot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Λ_geo = ||[E, Ė]||_F.
        
        This is the PRIMARY diagnostic (Umbrella Patent Claim 2).
        
        Args:
            E: Strain tensor field, shape (n_times, n_stations, 3, 3)
            E_dot: Time derivative, shape (n_times, n_stations, 3, 3)
            
        Returns:
            lambda_geo: Raw diagnostic
            lambda_geo_norm: Normalized diagnostic
        """
        # Compute commutator
        C = self.compute_commutator(E, E_dot)
        
        # Frobenius norm
        lambda_geo = self.compute_frobenius_norm(C)
        
        # Normalized version (scale-invariant)
        if self.normalize:
            norm_E = self.compute_frobenius_norm(E)
            norm_E_dot = self.compute_frobenius_norm(E_dot)
            lambda_geo_norm = lambda_geo / (norm_E * norm_E_dot + 1e-20)
        else:
            lambda_geo_norm = lambda_geo.copy()
        
        return lambda_geo, lambda_geo_norm
    
    # =========================================================================
    # SPECTRAL ANALYSIS
    # =========================================================================
    
    def compute_spectral_decomposition(self, 
                                       E: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues and eigenvectors of strain tensor field.
        
        From Paper 3, the spectral gap δ = λ₁ - λ₂ controls eigenframe
        sensitivity via the rotation lemma: |ė₁| ≤ C·Λ/δ
        
        Args:
            E: Strain tensor field, shape (n_times, n_stations, 3, 3)
            
        Returns:
            eigenvalues: shape (n_times, n_stations, 3), sorted descending
            eigenvectors: shape (n_times, n_stations, 3, 3), columns = eᵢ
        """
        n_times, n_stations = E.shape[:2]
        eigenvalues = np.zeros((n_times, n_stations, 3))
        eigenvectors = np.zeros((n_times, n_stations, 3, 3))
        
        for t in range(n_times):
            for s in range(n_stations):
                # Use eigh for symmetric matrices (numerically stable)
                vals, vecs = eigh(E[t, s])
                
                # Sort descending
                idx = np.argsort(vals)[::-1]
                eigenvalues[t, s] = vals[idx]
                eigenvectors[t, s] = vecs[:, idx]
        
        return eigenvalues, eigenvectors
    
    def compute_explicit_lambda_geo(self,
                                    eigenvalues: np.ndarray,
                                    E_dot: np.ndarray,
                                    eigenvectors: np.ndarray) -> np.ndarray:
        """
        Compute Λ_geo using explicit formula from Paper 3.
        
        ||[E, Ė]||_F² = 2 Σᵢ<ⱼ (λᵢ - λⱼ)² (Ė)ᵢⱼ²
        
        This formula reveals WHY Λ_geo works:
        - Off-diagonal Ė components = eigenframe rotation
        - Weighted by spectral gap squared
        
        Args:
            eigenvalues: E eigenvalues, shape (..., 3)
            E_dot: Time derivative in original basis, shape (..., 3, 3)
            eigenvectors: E eigenvectors, shape (..., 3, 3)
            
        Returns:
            lambda_geo: Computed via explicit formula
        """
        # Transform E_dot to E-eigenbasis
        # E_dot_eigen = V^T @ E_dot @ V
        E_dot_eigen = np.einsum('...ji,...jk,...kl->...il',
                                eigenvectors, E_dot, eigenvectors)
        
        # Off-diagonal components
        E_dot_12 = E_dot_eigen[..., 0, 1]
        E_dot_13 = E_dot_eigen[..., 0, 2]
        E_dot_23 = E_dot_eigen[..., 1, 2]
        
        # Eigenvalue differences
        lam1 = eigenvalues[..., 0]
        lam2 = eigenvalues[..., 1]
        lam3 = eigenvalues[..., 2]
        
        # Explicit formula
        lambda_geo_sq = 2 * (
            (lam1 - lam2)**2 * E_dot_12**2 +
            (lam1 - lam3)**2 * E_dot_13**2 +
            (lam2 - lam3)**2 * E_dot_23**2
        )
        
        return np.sqrt(np.maximum(lambda_geo_sq, 0))
    
    # =========================================================================
    # RISK ASSESSMENT
    # =========================================================================
    
    def compute_risk_score(self,
                           lambda_geo: np.ndarray,
                           lambda_geo_norm: np.ndarray,
                           spectral_gap: np.ndarray,
                           eigenframe_rotation: np.ndarray) -> np.ndarray:
        """
        Compute ensemble risk score from diagnostics.
        
        High risk when:
        - Λ_geo is elevated (eigenframe rotating)
        - Rotation rate is high (Λ_geo / δ large)
        - Spectral gap is reasonable (direction meaningful)
        
        Args:
            lambda_geo: Raw Λ_geo values
            lambda_geo_norm: Normalized Λ_geo
            spectral_gap: δ = λ₁ - λ₂
            eigenframe_rotation: Λ_geo / δ
            
        Returns:
            risk_score: Values in [0, 1]
        """
        def robust_zscore(x, axis=None):
            """Z-score robust to outliers."""
            median = np.nanmedian(x, axis=axis, keepdims=True)
            mad = np.nanmedian(np.abs(x - median), axis=axis, keepdims=True)
            mad = np.maximum(mad, 1e-10)
            return (x - median) / (1.4826 * mad)  # 1.4826 makes MAD consistent with std
        
        # Normalize each component
        z_lambda = robust_zscore(lambda_geo)
        z_lambda_norm = robust_zscore(lambda_geo_norm)
        z_rotation = robust_zscore(np.log1p(eigenframe_rotation))
        
        # Weighted combination
        risk_raw = (
            0.40 * z_lambda +
            0.30 * z_lambda_norm +
            0.30 * z_rotation
        )
        
        # Sigmoid to [0, 1]
        risk_score = 1 / (1 + np.exp(-risk_raw / 2))
        
        return risk_score
    
    # =========================================================================
    # MAIN ANALYSIS PIPELINE
    # =========================================================================
    
    def analyze(self,
                strain_tensors: np.ndarray,
                times: np.ndarray,
                station_lats: np.ndarray,
                station_lons: np.ndarray,
                earthquake_info: Optional[Dict] = None) -> LambdaGeoResult:
        """
        Run complete Λ_geo analysis pipeline.
        
        This is the main entry point for the sprint validation.
        
        Args:
            strain_tensors: Shape (n_times, n_stations, 3, 3)
            times: Time values (datetime strings or floats)
            station_lats: Station latitudes
            station_lons: Station longitudes
            earthquake_info: Optional earthquake metadata
            
        Returns:
            LambdaGeoResult with all computed fields
        """
        n_times, n_stations = strain_tensors.shape[:2]
        
        print(f"Analyzing strain tensor field...")
        print(f"  Shape: {strain_tensors.shape}")
        print(f"  Time steps: {n_times}")
        print(f"  Stations: {n_stations}")
        
        # Step 1: Time derivative
        print("  Computing time derivative...")
        E_dot = self.compute_time_derivative(strain_tensors)
        
        # Step 2: Core Λ_geo
        print("  Computing Λ_geo = ||[E, Ė]||_F...")
        lambda_geo, lambda_geo_norm = self.compute_lambda_geo(strain_tensors, E_dot)
        
        # Step 3: Spectral decomposition
        print("  Computing spectral decomposition...")
        eigenvalues, eigenvectors = self.compute_spectral_decomposition(strain_tensors)
        
        spectral_gap_12 = eigenvalues[..., 0] - eigenvalues[..., 1]
        spectral_gap_13 = eigenvalues[..., 0] - eigenvalues[..., 2]
        
        # Step 4: Eigenframe rotation rate
        print("  Computing eigenframe rotation rate...")
        safe_gap = np.maximum(np.abs(spectral_gap_12), 1e-20)
        eigenframe_rotation = lambda_geo / safe_gap
        
        # Step 5: Principal direction
        principal_direction = eigenvectors[..., :, 0]
        
        # Step 6: Risk scoring
        print("  Computing risk scores...")
        risk_score = self.compute_risk_score(
            lambda_geo, lambda_geo_norm, spectral_gap_12, eigenframe_rotation
        )
        
        # Spatial maximum risk at each time
        spatial_max_risk = np.max(risk_score, axis=1)
        
        # Optional smoothing
        if self.smoothing_window > 0:
            lambda_geo = ndimage.gaussian_filter1d(
                lambda_geo, self.smoothing_window, axis=0
            )
            risk_score = ndimage.gaussian_filter1d(
                risk_score, self.smoothing_window, axis=0
            )
            spatial_max_risk = ndimage.gaussian_filter1d(
                spatial_max_risk, self.smoothing_window
            )
        
        # Verify with explicit formula (sanity check)
        lambda_geo_explicit = self.compute_explicit_lambda_geo(
            eigenvalues, E_dot, eigenvectors
        )
        agreement = np.corrcoef(lambda_geo.flatten(), lambda_geo_explicit.flatten())[0, 1]
        print(f"  Formula verification: r = {agreement:.6f}")
        
        result = LambdaGeoResult(
            times=times,
            station_lats=station_lats,
            station_lons=station_lons,
            lambda_geo=lambda_geo,
            lambda_geo_normalized=lambda_geo_norm,
            eigenvalues=eigenvalues,
            spectral_gap_12=spectral_gap_12,
            spectral_gap_13=spectral_gap_13,
            eigenframe_rotation_rate=eigenframe_rotation,
            principal_direction=principal_direction,
            risk_score=risk_score,
            spatial_max_risk=spatial_max_risk,
            earthquake_info=earthquake_info or {},
            computation_params={
                'dt_hours': self.dt,
                'smoothing_window': self.smoothing_window,
                'derivative_method': self.derivative_method,
                'normalize': self.normalize
            }
        )
        
        print("  Analysis complete.")
        return result


def load_strain_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Load strain tensor data from NPZ file."""
    from pathlib import Path
    data = np.load(Path(filepath), allow_pickle=True)
    
    strain_tensors = data['strain_tensors']
    times = data['times']
    station_lats = data['station_lats']
    station_lons = data['station_lons']
    earthquake_info = json.loads(str(data['earthquake_info']))
    
    return strain_tensors, times, station_lats, station_lons, earthquake_info


if __name__ == "__main__":
    # Quick test
    print("Lambda_geo module loaded successfully")
    print("Use LambdaGeoAnalyzer.analyze() for full analysis")
