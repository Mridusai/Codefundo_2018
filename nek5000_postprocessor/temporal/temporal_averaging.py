"""
Temporal Averaging
==================

Tools for computing time-averaged statistics from DNS field sequences.
Supports various averaging methods including:
- Simple time averaging
- Running (cumulative) averages
- Exponential moving averages
- Windowed statistics for non-stationary flows
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import warnings

try:
    from ..readers import FieldData
    from ..statistics import FirstOrderStatistics, SecondOrderStatistics
    from ..statistics.first_order import compute_mean_fields, OnlineMeanComputer
    from ..statistics.second_order import compute_reynolds_stresses, OnlineSecondOrderComputer
except ImportError:
    try:
        from readers.field_reader import FieldData
        from statistics.first_order import FirstOrderStatistics, compute_mean_fields, OnlineMeanComputer
        from statistics.second_order import SecondOrderStatistics, compute_reynolds_stresses, OnlineSecondOrderComputer
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from readers.field_reader import FieldData
        from statistics.first_order import FirstOrderStatistics, compute_mean_fields, OnlineMeanComputer
        from statistics.second_order import SecondOrderStatistics, compute_reynolds_stresses, OnlineSecondOrderComputer


@dataclass
class TemporalStatistics:
    """
    Container for time-dependent statistics.
    
    Stores statistics computed at multiple time points for
    tracking temporal evolution of averaged quantities.
    """
    times: np.ndarray
    u_mean: np.ndarray  # Shape: (n_times, n_points)
    v_mean: np.ndarray
    w_mean: np.ndarray
    p_mean: Optional[np.ndarray] = None
    
    # Second-order statistics
    uu: Optional[np.ndarray] = None
    vv: Optional[np.ndarray] = None
    ww: Optional[np.ndarray] = None
    uv: Optional[np.ndarray] = None
    tke: Optional[np.ndarray] = None
    
    # Metadata
    window_size: int = 0
    averaging_method: str = 'running'
    
    def get_at_time(self, t: float) -> Dict[str, np.ndarray]:
        """Get statistics at a specific time (interpolated if necessary)."""
        idx = np.searchsorted(self.times, t)
        idx = min(idx, len(self.times) - 1)
        
        result = {
            'u_mean': self.u_mean[idx],
            'v_mean': self.v_mean[idx],
            'w_mean': self.w_mean[idx],
        }
        
        if self.p_mean is not None:
            result['p_mean'] = self.p_mean[idx]
        if self.tke is not None:
            result['tke'] = self.tke[idx]
            
        return result
    
    def get_global_average(self, field_name: str) -> np.ndarray:
        """Get spatially-averaged time series."""
        field_data = getattr(self, field_name, None)
        if field_data is None:
            raise ValueError(f"Field {field_name} not available")
        
        # Average over spatial dimensions (axis=1 and beyond)
        axes = tuple(range(1, field_data.ndim))
        return np.mean(field_data, axis=axes)


class TemporalAverager:
    """
    Temporal averaging processor for DNS field sequences.
    
    Computes running averages and statistics over time, tracking
    the evolution of mean fields and turbulence quantities.
    
    Parameters
    ----------
    window_type : str
        Type of averaging window: 'cumulative', 'sliding', 'exponential'
    window_size : int
        Number of snapshots in sliding window (for 'sliding' type)
    alpha : float
        Smoothing factor for exponential averaging (for 'exponential' type)
        
    Example
    -------
    >>> averager = TemporalAverager(window_type='cumulative')
    >>> for field in field_sequence:
    ...     averager.update(field)
    ...     stats = averager.get_current_statistics()
    >>> evolution = averager.get_time_evolution()
    """
    
    def __init__(
        self,
        window_type: str = 'cumulative',
        window_size: int = 100,
        alpha: float = 0.1,
        compute_second_order: bool = True
    ):
        self.window_type = window_type
        self.window_size = window_size
        self.alpha = alpha
        self.compute_second_order = compute_second_order
        
        # Storage for running statistics
        self.n = 0
        self.times: List[float] = []
        
        # Running accumulators
        self.mean_u = None
        self.mean_v = None
        self.mean_w = None
        self.mean_p = None
        
        # For sliding window
        self._buffer: List = []
        
        # For second-order statistics
        if compute_second_order:
            self._second_order_computer = OnlineSecondOrderComputer()
        else:
            self._second_order_computer = None
        
        # History for time evolution
        self._history = {
            'times': [],
            'u_mean': [],
            'v_mean': [],
            'w_mean': [],
            'tke': [],
        }
        
    def update(self, field) -> None:
        """
        Update averages with a new field snapshot.
        
        Parameters
        ----------
        field : FieldData
            New field snapshot
        """
        self.n += 1
        self.times.append(field.time)
        
        if self.window_type == 'cumulative':
            self._update_cumulative(field)
        elif self.window_type == 'sliding':
            self._update_sliding(field)
        elif self.window_type == 'exponential':
            self._update_exponential(field)
        else:
            raise ValueError(f"Unknown window type: {self.window_type}")
        
        # Update history (store global averages for tracking)
        self._history['times'].append(field.time)
        self._history['u_mean'].append(np.mean(self.mean_u) if self.mean_u is not None else 0)
        self._history['v_mean'].append(np.mean(self.mean_v) if self.mean_v is not None else 0)
        self._history['w_mean'].append(np.mean(self.mean_w) if self.mean_w is not None else 0)
        
        # Update second-order statistics
        if self._second_order_computer is not None:
            self._second_order_computer.update(field)
            if self.n >= 2:
                stats = self._second_order_computer.get_statistics()
                self._history['tke'].append(np.mean(stats.tke))
            else:
                self._history['tke'].append(0)
    
    def _update_cumulative(self, field) -> None:
        """Update using cumulative (running) average."""
        if self.mean_u is None:
            self.mean_u = field.u.astype(np.float64).copy()
            self.mean_v = field.v.astype(np.float64).copy()
            self.mean_w = field.w.astype(np.float64).copy() if field.w is not None else np.zeros_like(field.u, dtype=np.float64)
            self.mean_p = field.p.astype(np.float64).copy() if field.p is not None else None
        else:
            n = self.n
            self.mean_u = self.mean_u + (field.u - self.mean_u) / n
            self.mean_v = self.mean_v + (field.v - self.mean_v) / n
            if field.w is not None:
                self.mean_w = self.mean_w + (field.w - self.mean_w) / n
            if field.p is not None and self.mean_p is not None:
                self.mean_p = self.mean_p + (field.p - self.mean_p) / n
    
    def _update_sliding(self, field) -> None:
        """Update using sliding window average."""
        self._buffer.append(field)
        
        # Maintain window size
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)
        
        # Recompute average over window
        n = len(self._buffer)
        self.mean_u = np.mean([f.u for f in self._buffer], axis=0)
        self.mean_v = np.mean([f.v for f in self._buffer], axis=0)
        self.mean_w = np.mean([f.w for f in self._buffer if f.w is not None], axis=0) if self._buffer[0].w is not None else None
        self.mean_p = np.mean([f.p for f in self._buffer if f.p is not None], axis=0) if self._buffer[0].p is not None else None
    
    def _update_exponential(self, field) -> None:
        """Update using exponential moving average."""
        alpha = self.alpha
        
        if self.mean_u is None:
            self.mean_u = field.u.astype(np.float64).copy()
            self.mean_v = field.v.astype(np.float64).copy()
            self.mean_w = field.w.astype(np.float64).copy() if field.w is not None else np.zeros_like(field.u, dtype=np.float64)
            self.mean_p = field.p.astype(np.float64).copy() if field.p is not None else None
        else:
            self.mean_u = alpha * field.u + (1 - alpha) * self.mean_u
            self.mean_v = alpha * field.v + (1 - alpha) * self.mean_v
            if field.w is not None:
                self.mean_w = alpha * field.w + (1 - alpha) * self.mean_w
            if field.p is not None and self.mean_p is not None:
                self.mean_p = alpha * field.p + (1 - alpha) * self.mean_p
    
    def get_current_statistics(self) -> Dict[str, np.ndarray]:
        """Get current averaged statistics."""
        result = {
            'u_mean': self.mean_u.astype(np.float32) if self.mean_u is not None else None,
            'v_mean': self.mean_v.astype(np.float32) if self.mean_v is not None else None,
            'w_mean': self.mean_w.astype(np.float32) if self.mean_w is not None else None,
            'p_mean': self.mean_p.astype(np.float32) if self.mean_p is not None else None,
            'n_samples': self.n,
        }
        
        if self._second_order_computer is not None and self.n >= 2:
            second_order = self._second_order_computer.get_statistics()
            result['uu'] = second_order.uu
            result['vv'] = second_order.vv
            result['ww'] = second_order.ww
            result['uv'] = second_order.uv
            result['tke'] = second_order.tke
        
        return result
    
    def get_time_evolution(self) -> TemporalStatistics:
        """
        Get time evolution of spatially-averaged statistics.
        
        Returns
        -------
        TemporalStatistics
            Container with statistics at each time step
        """
        return TemporalStatistics(
            times=np.array(self._history['times']),
            u_mean=np.array(self._history['u_mean']),
            v_mean=np.array(self._history['v_mean']),
            w_mean=np.array(self._history['w_mean']),
            tke=np.array(self._history['tke']) if self._history['tke'] else None,
            window_size=self.window_size,
            averaging_method=self.window_type,
        )
    
    def reset(self) -> None:
        """Reset all accumulators."""
        self.n = 0
        self.times = []
        self.mean_u = None
        self.mean_v = None
        self.mean_w = None
        self.mean_p = None
        self._buffer = []
        self._second_order_computer = OnlineSecondOrderComputer() if self.compute_second_order else None
        self._history = {
            'times': [],
            'u_mean': [],
            'v_mean': [],
            'w_mean': [],
            'tke': [],
        }


def running_average(
    fields: List,
    compute_second_order: bool = True,
    sample_interval: int = 1
) -> TemporalStatistics:
    """
    Compute running average over a field sequence.
    
    Parameters
    ----------
    fields : List[FieldData]
        List of field snapshots
    compute_second_order : bool
        Whether to compute second-order statistics
    sample_interval : int
        Store statistics every N snapshots
        
    Returns
    -------
    TemporalStatistics
        Time evolution of averaged statistics
    """
    averager = TemporalAverager(
        window_type='cumulative',
        compute_second_order=compute_second_order
    )
    
    for i, field in enumerate(fields):
        averager.update(field)
    
    return averager.get_time_evolution()


def exponential_moving_average(
    fields: List,
    alpha: float = 0.1,
    compute_second_order: bool = True
) -> TemporalStatistics:
    """
    Compute exponential moving average over a field sequence.
    
    Parameters
    ----------
    fields : List[FieldData]
        List of field snapshots
    alpha : float
        Smoothing factor (0 < alpha <= 1)
    compute_second_order : bool
        Whether to compute second-order statistics
        
    Returns
    -------
    TemporalStatistics
        Time evolution of averaged statistics
    """
    averager = TemporalAverager(
        window_type='exponential',
        alpha=alpha,
        compute_second_order=compute_second_order
    )
    
    for field in fields:
        averager.update(field)
    
    return averager.get_time_evolution()


def windowed_statistics(
    fields: List,
    window_size: int,
    overlap: float = 0.5,
    compute_second_order: bool = True
) -> List[Dict]:
    """
    Compute statistics over non-overlapping or overlapping windows.
    
    Useful for analyzing non-stationary flows where statistics
    vary significantly over time (e.g., during step change transients).
    
    Parameters
    ----------
    fields : List[FieldData]
        List of field snapshots
    window_size : int
        Number of snapshots per window
    overlap : float
        Fraction of overlap between windows (0 to 1)
    compute_second_order : bool
        Whether to compute second-order statistics
        
    Returns
    -------
    List[Dict]
        List of statistics dictionaries for each window
    """
    n_fields = len(fields)
    step = max(1, int(window_size * (1 - overlap)))
    
    windows = []
    
    for start in range(0, n_fields - window_size + 1, step):
        end = start + window_size
        window_fields = fields[start:end]
        
        # Compute statistics for this window
        try:
            from ..statistics.first_order import compute_mean_fields
            from ..statistics.second_order import compute_reynolds_stresses
        except ImportError:
            from statistics.first_order import compute_mean_fields
            from statistics.second_order import compute_reynolds_stresses
        
        mean_stats = compute_mean_fields(window_fields)
        
        window_result = {
            'time_start': window_fields[0].time,
            'time_end': window_fields[-1].time,
            'time_center': 0.5 * (window_fields[0].time + window_fields[-1].time),
            'n_samples': len(window_fields),
            'u_mean': mean_stats.u_mean,
            'v_mean': mean_stats.v_mean,
            'w_mean': mean_stats.w_mean,
            'p_mean': mean_stats.p_mean,
        }
        
        if compute_second_order:
            second_order = compute_reynolds_stresses(window_fields, mean_stats)
            window_result['uu'] = second_order.uu
            window_result['vv'] = second_order.vv
            window_result['ww'] = second_order.ww
            window_result['uv'] = second_order.uv
            window_result['tke'] = second_order.tke
        
        windows.append(window_result)
    
    return windows


def compute_ensemble_average(
    field_sequences: List[List],
    compute_second_order: bool = True
) -> Dict[str, np.ndarray]:
    """
    Compute ensemble average over multiple realizations.
    
    For DNS, ensemble averaging provides improved statistics when
    multiple independent simulations are available.
    
    Parameters
    ----------
    field_sequences : List[List[FieldData]]
        List of field sequences (one per realization)
    compute_second_order : bool
        Whether to compute second-order statistics
        
    Returns
    -------
    Dict[str, np.ndarray]
        Ensemble-averaged statistics
    """
    n_realizations = len(field_sequences)
    n_snapshots = min(len(seq) for seq in field_sequences)
    
    # Initialize accumulators
    ref_field = field_sequences[0][0]
    shape = ref_field.u.shape
    
    u_mean = np.zeros(shape, dtype=np.float64)
    v_mean = np.zeros(shape, dtype=np.float64)
    w_mean = np.zeros(shape, dtype=np.float64)
    
    # First pass: compute means
    for seq in field_sequences:
        for i in range(n_snapshots):
            u_mean += seq[i].u
            v_mean += seq[i].v
            if seq[i].w is not None:
                w_mean += seq[i].w
    
    total = n_realizations * n_snapshots
    u_mean /= total
    v_mean /= total
    w_mean /= total
    
    result = {
        'u_mean': u_mean.astype(np.float32),
        'v_mean': v_mean.astype(np.float32),
        'w_mean': w_mean.astype(np.float32),
        'n_realizations': n_realizations,
        'n_snapshots': n_snapshots,
    }
    
    if compute_second_order:
        # Second pass: compute variances
        uu = np.zeros(shape, dtype=np.float64)
        vv = np.zeros(shape, dtype=np.float64)
        ww = np.zeros(shape, dtype=np.float64)
        uv = np.zeros(shape, dtype=np.float64)
        
        for seq in field_sequences:
            for i in range(n_snapshots):
                u_prime = seq[i].u - u_mean
                v_prime = seq[i].v - v_mean
                w_prime = (seq[i].w - w_mean) if seq[i].w is not None else np.zeros_like(u_prime)
                
                uu += u_prime * u_prime
                vv += v_prime * v_prime
                ww += w_prime * w_prime
                uv += u_prime * v_prime
        
        uu /= total
        vv /= total
        ww /= total
        uv /= total
        
        result['uu'] = uu.astype(np.float32)
        result['vv'] = vv.astype(np.float32)
        result['ww'] = ww.astype(np.float32)
        result['uv'] = uv.astype(np.float32)
        result['tke'] = (0.5 * (uu + vv + ww)).astype(np.float32)
    
    return result


def compute_phase_average(
    fields: List,
    period: float,
    n_phases: int = 20,
    compute_second_order: bool = True
) -> List[Dict]:
    """
    Compute phase-averaged statistics for periodic flows.
    
    Parameters
    ----------
    fields : List[FieldData]
        List of field snapshots
    period : float
        Period of the oscillation
    n_phases : int
        Number of phase bins
    compute_second_order : bool
        Whether to compute second-order statistics
        
    Returns
    -------
    List[Dict]
        Statistics for each phase bin
    """
    phase_bins = [[] for _ in range(n_phases)]
    
    # Sort fields into phase bins
    for field in fields:
        phase = (field.time % period) / period  # Normalized phase [0, 1)
        bin_idx = int(phase * n_phases) % n_phases
        phase_bins[bin_idx].append(field)
    
    # Compute statistics for each phase
    results = []
    for i, bin_fields in enumerate(phase_bins):
        if len(bin_fields) == 0:
            continue
        
        phase_center = (i + 0.5) / n_phases
        
        try:
            from ..statistics.first_order import compute_mean_fields
            from ..statistics.second_order import compute_reynolds_stresses
        except ImportError:
            from statistics.first_order import compute_mean_fields
            from statistics.second_order import compute_reynolds_stresses
        
        mean_stats = compute_mean_fields(bin_fields)
        
        result = {
            'phase': phase_center,
            'n_samples': len(bin_fields),
            'u_mean': mean_stats.u_mean,
            'v_mean': mean_stats.v_mean,
            'w_mean': mean_stats.w_mean,
        }
        
        if compute_second_order and len(bin_fields) >= 2:
            second_order = compute_reynolds_stresses(bin_fields, mean_stats)
            result['tke'] = second_order.tke
        
        results.append(result)
    
    return results
