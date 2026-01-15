"""
coherence.py - Spatial Coherence Analysis

Ensures elevated Λ_geo signals are spatially coherent (not single-station noise).

Criteria:
- At least K adjacent cells/triangles exceed threshold
- Or at least N% of the grid exceeds threshold
"""

import numpy as np
from scipy.ndimage import label
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class CoherenceResult:
    """Result of spatial coherence check."""
    is_coherent: bool
    n_elevated: int
    total_cells: int
    fraction_elevated: float
    n_clusters: int
    max_cluster_size: int
    reason: str
    
    def to_dict(self) -> dict:
        return {
            'is_coherent': self.is_coherent,
            'n_elevated': self.n_elevated,
            'total_cells': self.total_cells,
            'fraction_elevated': self.fraction_elevated,
            'n_clusters': self.n_clusters,
            'max_cluster_size': self.max_cluster_size,
            'reason': self.reason,
        }


class SpatialCoherence:
    """
    Check spatial coherence of elevated Λ_geo regions.
    
    Prevents false alarms from single bad stations or grid artifacts.
    """
    
    def __init__(self,
                 min_cluster_size: int = 3,
                 min_fraction: float = 0.10,
                 require_both: bool = False):
        """
        Args:
            min_cluster_size: Minimum connected cells to be coherent
            min_fraction: Minimum fraction of grid to be coherent
            require_both: If True, require BOTH criteria; else either
        """
        self.min_cluster_size = min_cluster_size
        self.min_fraction = min_fraction
        self.require_both = require_both
    
    def check(self, 
              lambda_geo_grid: np.ndarray, 
              threshold: float) -> CoherenceResult:
        """
        Check if elevated region is spatially coherent.
        
        Args:
            lambda_geo_grid: 2D array of Λ_geo values
            threshold: Threshold value (e.g., 5 × baseline)
            
        Returns:
            CoherenceResult with coherence assessment
        """
        # Handle NaN
        grid = np.nan_to_num(lambda_geo_grid, nan=0.0)
        
        # Find elevated cells
        elevated = grid > threshold
        n_elevated = int(np.sum(elevated))
        total_cells = grid.size
        fraction = n_elevated / total_cells if total_cells > 0 else 0
        
        # Find connected components
        if n_elevated > 0:
            labeled, n_clusters = label(elevated)
            
            # Find size of each cluster
            cluster_sizes = []
            for i in range(1, n_clusters + 1):
                cluster_sizes.append(int(np.sum(labeled == i)))
            
            max_cluster = max(cluster_sizes) if cluster_sizes else 0
        else:
            n_clusters = 0
            max_cluster = 0
        
        # Check coherence criteria
        cluster_ok = max_cluster >= self.min_cluster_size
        fraction_ok = fraction >= self.min_fraction
        
        if self.require_both:
            is_coherent = cluster_ok and fraction_ok
        else:
            is_coherent = cluster_ok or fraction_ok
        
        # Build reason string
        if is_coherent:
            reasons = []
            if cluster_ok:
                reasons.append(f"cluster size {max_cluster} >= {self.min_cluster_size}")
            if fraction_ok:
                reasons.append(f"fraction {fraction:.1%} >= {self.min_fraction:.0%}")
            reason = "Coherent: " + " and ".join(reasons)
        else:
            reason = f"Not coherent: max cluster {max_cluster} < {self.min_cluster_size} and fraction {fraction:.1%} < {self.min_fraction:.0%}"
        
        return CoherenceResult(
            is_coherent=is_coherent,
            n_elevated=n_elevated,
            total_cells=total_cells,
            fraction_elevated=fraction,
            n_clusters=n_clusters,
            max_cluster_size=max_cluster,
            reason=reason,
        )


class TriangleCoherence:
    """
    Check coherence on Delaunay triangle graph (closer to physics).
    
    Uses triangle adjacency rather than grid cell adjacency.
    """
    
    def __init__(self, 
                 min_adjacent_triangles: int = 3,
                 min_fraction: float = 0.15):
        """
        Args:
            min_adjacent_triangles: Minimum connected triangles
            min_fraction: Minimum fraction of triangles elevated
        """
        self.min_adjacent_triangles = min_adjacent_triangles
        self.min_fraction = min_fraction
    
    def build_adjacency(self, triangles: np.ndarray) -> dict:
        """
        Build adjacency graph from triangle vertex indices.
        
        Args:
            triangles: Array of shape (n_triangles, 3) with vertex indices
            
        Returns:
            Dict mapping triangle index to list of adjacent triangle indices
        """
        n_tri = len(triangles)
        
        # Build edge -> triangle mapping
        edge_to_tri = {}
        for i, tri in enumerate(triangles):
            for j in range(3):
                edge = tuple(sorted([tri[j], tri[(j+1) % 3]]))
                if edge not in edge_to_tri:
                    edge_to_tri[edge] = []
                edge_to_tri[edge].append(i)
        
        # Build adjacency
        adjacency = {i: [] for i in range(n_tri)}
        for tri_list in edge_to_tri.values():
            if len(tri_list) == 2:
                i, j = tri_list
                adjacency[i].append(j)
                adjacency[j].append(i)
        
        return adjacency
    
    def check(self,
              triangle_values: np.ndarray,
              threshold: float,
              adjacency: dict) -> CoherenceResult:
        """
        Check coherence on triangle graph.
        
        Args:
            triangle_values: Λ_geo value per triangle
            threshold: Threshold value
            adjacency: Triangle adjacency dict
            
        Returns:
            CoherenceResult
        """
        n_tri = len(triangle_values)
        
        # Find elevated triangles
        elevated = triangle_values > threshold
        n_elevated = int(np.sum(elevated))
        fraction = n_elevated / n_tri if n_tri > 0 else 0
        
        # Find connected components via BFS
        visited = set()
        clusters = []
        
        for start in range(n_tri):
            if start in visited or not elevated[start]:
                continue
            
            # BFS from this triangle
            cluster = []
            queue = [start]
            
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                
                if elevated[node]:
                    cluster.append(node)
                    for neighbor in adjacency.get(node, []):
                        if neighbor not in visited:
                            queue.append(neighbor)
            
            if cluster:
                clusters.append(cluster)
        
        n_clusters = len(clusters)
        max_cluster = max(len(c) for c in clusters) if clusters else 0
        
        # Check criteria
        cluster_ok = max_cluster >= self.min_adjacent_triangles
        fraction_ok = fraction >= self.min_fraction
        
        is_coherent = cluster_ok or fraction_ok
        
        if is_coherent:
            reason = f"Coherent: cluster={max_cluster}, fraction={fraction:.0%}"
        else:
            reason = f"Not coherent: cluster={max_cluster} < {self.min_adjacent_triangles}"
        
        return CoherenceResult(
            is_coherent=is_coherent,
            n_elevated=n_elevated,
            total_cells=n_tri,
            fraction_elevated=fraction,
            n_clusters=n_clusters,
            max_cluster_size=max_cluster,
            reason=reason,
        )


# === Tests ===

def test_grid_coherence():
    """Test grid-based coherence detection."""
    coherence = SpatialCoherence(min_cluster_size=3, min_fraction=0.10)
    
    # Test 1: Single elevated cell (not coherent)
    grid1 = np.zeros((10, 10))
    grid1[5, 5] = 10.0
    result1 = coherence.check(grid1, threshold=5.0)
    assert not result1.is_coherent, "Single cell should not be coherent"
    print(f"Test 1 (single cell): {result1.reason}")
    
    # Test 2: Cluster of 4 cells (coherent)
    grid2 = np.zeros((10, 10))
    grid2[4:6, 4:6] = 10.0
    result2 = coherence.check(grid2, threshold=5.0)
    assert result2.is_coherent, "2x2 cluster should be coherent"
    print(f"Test 2 (2x2 cluster): {result2.reason}")
    
    # Test 3: Scattered cells (not coherent unless fraction high)
    grid3 = np.zeros((10, 10))
    grid3[0, 0] = 10.0
    grid3[9, 9] = 10.0
    grid3[0, 9] = 10.0
    result3 = coherence.check(grid3, threshold=5.0)
    assert not result3.is_coherent, "Scattered cells should not be coherent"
    print(f"Test 3 (scattered): {result3.reason}")
    
    # Test 4: High fraction (coherent)
    grid4 = np.random.rand(10, 10) * 10
    grid4[grid4 > 5] = 10.0
    grid4[grid4 <= 5] = 1.0
    # Make sure >10% elevated
    grid4[:2, :] = 10.0
    result4 = coherence.check(grid4, threshold=5.0)
    print(f"Test 4 (high fraction): {result4.reason}")
    
    print("✓ All coherence tests passed")


if __name__ == "__main__":
    test_grid_coherence()
