"""
First-Order Statistics
======================

Compute first-order (mean) statistics from DNS field data.
Supports both temporal and spatial averaging with optional weighting.
"""

import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass, field
import warnings

# Import from parent package - handle both module and direct import
try:
    from ..readers import FieldData
except ImportError:
    try:
        from readers.field_reader import FieldData
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from readers.field_reader import FieldData


@dataclass
class FirstOrderStatistics:
    """
    Container for first-order turbulence statistics.
    
    Attributes
    ----------
    u_mean : np.ndarray
        Mean streamwise velocity
    v_mean : np.ndarray
        Mean wall-normal velocity
    w_mean : np.ndarray
        Mean spanwise velocity
    p_mean : np.ndarray
        Mean pressure
    t_mean : np.ndarray
        Mean temperature (if available)
    n_samples : int
        Number of samples used in averaging
    time_start : float
        Start time for averaging
    time_end : float
        End time for averaging
    averaging_method : str
        Method used for averaging ('temporal', 'spatial', 'ensemble')
    """
    u_mean: np.ndarray
    v_mean: np.ndarray
    w_mean: np.ndarray
    p_mean: Optional[np.ndarray] = None
    t_mean: Optional[np.ndarray] = None
    n_samples: int = 0
    time_start: float = 0.0
    time_end: float = 0.0
    averaging_method: str = 'temporal'
    
    # Optional mesh for spatial reference
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    z: Optional[np.ndarray] = None
    
    @property
    def velocity_magnitude_mean(self) -> np.ndarray:
        """Mean velocity magnitude."""
        return np.sqrt(self.u_mean**2 + self.v_mean**2 + self.w_mean**2)
    
    def get_field(self, name: str) -> Optional[np.ndarray]:
        """Get a statistic field by name."""
        field_map = {
            'u_mean': self.u_mean,
            'v_mean': self.v_mean,
            'w_mean': self.w_mean,
            'p_mean': self.p_mean,
            't_mean': self.t_mean,
            'velocity_magnitude_mean': self.velocity_magnitude_mean,
        }
        return field_map.get(name)
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary."""
        result = {
            'u_mean': self.u_mean,
            'v_mean': self.v_mean,
            'w_mean': self.w_mean,
        }
        if self.p_mean is not None:
            result['p_mean'] = self.p_mean
        if self.t_mean is not None:
            result['t_mean'] = self.t_mean
        return result


def compute_mean_fields(
    fields: List,
    weights: Optional[np.ndarray] = None,
    method: str = 'temporal'
) -> FirstOrderStatistics:
    """
    Compute mean fields from a sequence of field snapshots.
    
    Parameters
    ----------
    fields : List[FieldData]
        List of field data snapshots
    weights : Optional[np.ndarray]
        Weights for weighted averaging (e.g., time intervals)
    method : str
        Averaging method: 'temporal', 'ensemble', or 'favre' (density-weighted)
        
    Returns
    -------
    FirstOrderStatistics
        Container with mean fields
        
    Notes
    -----
    For step change DNS, temporal averaging should be performed after
    the flow reaches statistical stationarity. Use convergence checks
    to verify sufficient averaging time.
    """
    if len(fields) == 0:
        raise ValueError("Empty field list provided")
    
    n_samples = len(fields)
    
    # Get reference shape from first field
    ref_field = fields[0]
    shape = ref_field.u.shape
    
    # Initialize accumulators
    u_sum = np.zeros(shape, dtype=np.float64)
    v_sum = np.zeros(shape, dtype=np.float64)
    w_sum = np.zeros(shape, dtype=np.float64)
    p_sum = np.zeros(shape, dtype=np.float64) if ref_field.p is not None else None
    t_sum = np.zeros(shape, dtype=np.float64) if ref_field.t is not None else None
    
    # Compute weights if not provided (uniform weighting)
    if weights is None:
        weights = np.ones(n_samples) / n_samples
    else:
        weights = weights / np.sum(weights)  # Normalize
    
    # Accumulate weighted sum
    for i, field in enumerate(fields):
        w = weights[i]
        u_sum += w * field.u
        v_sum += w * field.v
        w_sum += w * field.w if field.w is not None else 0
        
        if p_sum is not None and field.p is not None:
            p_sum += w * field.p
        if t_sum is not None and field.t is not None:
            t_sum += w * field.t
    
    # Get time range
    time_start = fields[0].time
    time_end = fields[-1].time
    
    return FirstOrderStatistics(
        u_mean=u_sum.astype(np.float32),
        v_mean=v_sum.astype(np.float32),
        w_mean=w_sum.astype(np.float32),
        p_mean=p_sum.astype(np.float32) if p_sum is not None else None,
        t_mean=t_sum.astype(np.float32) if t_sum is not None else None,
        n_samples=n_samples,
        time_start=time_start,
        time_end=time_end,
        averaging_method=method,
        x=ref_field.x,
        y=ref_field.y,
        z=ref_field.z,
    )


def compute_spatial_average(
    field_data,
    direction: str = 'z',
    weights: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Compute spatial average along homogeneous direction(s).
    
    Parameters
    ----------
    field_data : FieldData or FirstOrderStatistics
        Field data to average
    direction : str
        Direction(s) to average over: 'x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz'
    weights : Optional[np.ndarray]
        Integration weights (e.g., Jacobian for non-uniform grids)
        
    Returns
    -------
    Dict[str, np.ndarray]
        Spatially averaged fields
        
    Notes
    -----
    For channel/pipe flows, typically average over homogeneous directions
    (streamwise x and spanwise z), leaving wall-normal profiles.
    """
    # Get fields
    if hasattr(field_data, 'u_mean'):
        u = field_data.u_mean
        v = field_data.v_mean
        w = field_data.w_mean
        p = field_data.p_mean
    else:
        u = field_data.u
        v = field_data.v
        w = field_data.w
        p = field_data.p
    
    # Reshape to 3D grid if necessary
    # This assumes data is ordered as (element, point) or flattened
    # For proper spatial averaging, we need the grid structure
    
    # For now, return the fields as-is with a warning
    warnings.warn(
        "Spatial averaging requires grid structure information. "
        "Consider using structured grid interpolation first."
    )
    
    return {
        'u': u,
        'v': v,
        'w': w,
        'p': p,
    }


def compute_rms(
    fields: List,
    mean_stats: Optional[FirstOrderStatistics] = None
) -> Dict[str, np.ndarray]:
    """
    Compute RMS (root-mean-square) fluctuations.
    
    Parameters
    ----------
    fields : List[FieldData]
        List of field snapshots
    mean_stats : Optional[FirstOrderStatistics]
        Pre-computed mean statistics (computed if not provided)
        
    Returns
    -------
    Dict[str, np.ndarray]
        RMS fluctuation fields: u_rms, v_rms, w_rms, p_rms
    """
    if mean_stats is None:
        mean_stats = compute_mean_fields(fields)
    
    n_samples = len(fields)
    
    # Initialize accumulators for variance
    u_var = np.zeros_like(mean_stats.u_mean, dtype=np.float64)
    v_var = np.zeros_like(mean_stats.v_mean, dtype=np.float64)
    w_var = np.zeros_like(mean_stats.w_mean, dtype=np.float64)
    p_var = np.zeros_like(mean_stats.p_mean, dtype=np.float64) if mean_stats.p_mean is not None else None
    
    # Compute variance (second pass)
    for field in fields:
        u_var += (field.u - mean_stats.u_mean)**2
        v_var += (field.v - mean_stats.v_mean)**2
        w_var += (field.w - mean_stats.w_mean)**2 if field.w is not None else 0
        
        if p_var is not None and field.p is not None:
            p_var += (field.p - mean_stats.p_mean)**2
    
    # Normalize and take square root
    u_rms = np.sqrt(u_var / n_samples).astype(np.float32)
    v_rms = np.sqrt(v_var / n_samples).astype(np.float32)
    w_rms = np.sqrt(w_var / n_samples).astype(np.float32)
    p_rms = np.sqrt(p_var / n_samples).astype(np.float32) if p_var is not None else None
    
    return {
        'u_rms': u_rms,
        'v_rms': v_rms,
        'w_rms': w_rms,
        'p_rms': p_rms,
    }


class OnlineMeanComputer:
    """
    Online (streaming) computation of mean statistics.
    
    Uses Welford's algorithm for numerically stable online computation
    of mean and variance. Useful for processing large datasets that
    don't fit in memory.
    
    Example
    -------
    >>> computer = OnlineMeanComputer()
    >>> for field in reader.read_field_sequence(1, 1000):
    ...     computer.update(field)
    >>> stats = computer.get_statistics()
    """
    
    def __init__(self):
        self.n = 0
        self.mean_u = None
        self.mean_v = None
        self.mean_w = None
        self.mean_p = None
        self.M2_u = None  # For variance computation
        self.M2_v = None
        self.M2_w = None
        self.M2_p = None
        self.time_start = None
        self.time_end = None
        
    def update(self, field) -> None:
        """Update running statistics with a new field snapshot."""
        self.n += 1
        
        if self.time_start is None:
            self.time_start = field.time
        self.time_end = field.time
        
        if self.mean_u is None:
            # Initialize on first sample
            self.mean_u = np.zeros_like(field.u, dtype=np.float64)
            self.mean_v = np.zeros_like(field.v, dtype=np.float64)
            self.mean_w = np.zeros_like(field.w, dtype=np.float64) if field.w is not None else None
            self.mean_p = np.zeros_like(field.p, dtype=np.float64) if field.p is not None else None
            self.M2_u = np.zeros_like(field.u, dtype=np.float64)
            self.M2_v = np.zeros_like(field.v, dtype=np.float64)
            self.M2_w = np.zeros_like(field.w, dtype=np.float64) if field.w is not None else None
            self.M2_p = np.zeros_like(field.p, dtype=np.float64) if field.p is not None else None
        
        # Welford's online algorithm
        delta_u = field.u - self.mean_u
        self.mean_u += delta_u / self.n
        delta2_u = field.u - self.mean_u
        self.M2_u += delta_u * delta2_u
        
        delta_v = field.v - self.mean_v
        self.mean_v += delta_v / self.n
        delta2_v = field.v - self.mean_v
        self.M2_v += delta_v * delta2_v
        
        if self.mean_w is not None and field.w is not None:
            delta_w = field.w - self.mean_w
            self.mean_w += delta_w / self.n
            delta2_w = field.w - self.mean_w
            self.M2_w += delta_w * delta2_w
        
        if self.mean_p is not None and field.p is not None:
            delta_p = field.p - self.mean_p
            self.mean_p += delta_p / self.n
            delta2_p = field.p - self.mean_p
            self.M2_p += delta_p * delta2_p
    
    def get_statistics(self) -> FirstOrderStatistics:
        """Get current statistics."""
        if self.n == 0:
            raise ValueError("No samples processed yet")
        
        return FirstOrderStatistics(
            u_mean=self.mean_u.astype(np.float32),
            v_mean=self.mean_v.astype(np.float32),
            w_mean=self.mean_w.astype(np.float32) if self.mean_w is not None else np.zeros_like(self.mean_u, dtype=np.float32),
            p_mean=self.mean_p.astype(np.float32) if self.mean_p is not None else None,
            n_samples=self.n,
            time_start=self.time_start,
            time_end=self.time_end,
            averaging_method='temporal',
        )
    
    def get_variance(self) -> Dict[str, np.ndarray]:
        """Get current variance estimates."""
        if self.n < 2:
            raise ValueError("Need at least 2 samples for variance")
        
        return {
            'u_var': (self.M2_u / (self.n - 1)).astype(np.float32),
            'v_var': (self.M2_v / (self.n - 1)).astype(np.float32),
            'w_var': (self.M2_w / (self.n - 1)).astype(np.float32) if self.M2_w is not None else None,
            'p_var': (self.M2_p / (self.n - 1)).astype(np.float32) if self.M2_p is not None else None,
        }


def compute_convergence_statistics(
    fields: List,
    window_sizes: List[int] = None
) -> Dict[str, np.ndarray]:
    """
    Compute statistics at different window sizes to assess convergence.
    
    Parameters
    ----------
    fields : List[FieldData]
        List of field snapshots
    window_sizes : List[int]
        Window sizes to compute statistics for
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with convergence data for each statistic
        
    Notes
    -----
    For DNS, it's important to verify that statistics have converged.
    This function computes running averages at different sample sizes
    to check if the statistics are approaching asymptotic values.
    """
    if window_sizes is None:
        n = len(fields)
        window_sizes = [n // 8, n // 4, n // 2, 3 * n // 4, n]
        window_sizes = [w for w in window_sizes if w > 0]
    
    convergence_data = {
        'window_sizes': np.array(window_sizes),
        'u_mean_global': [],
        'v_mean_global': [],
        'w_mean_global': [],
    }
    
    for window_size in window_sizes:
        stats = compute_mean_fields(fields[:window_size])
        convergence_data['u_mean_global'].append(np.mean(stats.u_mean))
        convergence_data['v_mean_global'].append(np.mean(stats.v_mean))
        convergence_data['w_mean_global'].append(np.mean(stats.w_mean))
    
    # Convert to arrays
    for key in ['u_mean_global', 'v_mean_global', 'w_mean_global']:
        convergence_data[key] = np.array(convergence_data[key])
    
    return convergence_data
