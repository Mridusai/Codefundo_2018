"""
Temporal Comparison
===================

Tools for comparing turbulence statistics across different time periods,
particularly useful for step change DNS analysis where conditions change
abruptly at a specific time.

Features:
- Before/after step change comparison
- Transient detection and duration estimation
- Relaxation time analysis
- Statistical significance testing
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
import warnings

try:
    from ..readers import FieldData
    from ..statistics import FirstOrderStatistics, SecondOrderStatistics
    from ..statistics.first_order import compute_mean_fields
    from ..statistics.second_order import compute_reynolds_stresses
except ImportError:
    try:
        from readers.field_reader import FieldData
        from statistics.first_order import FirstOrderStatistics, compute_mean_fields
        from statistics.second_order import SecondOrderStatistics, compute_reynolds_stresses
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from readers.field_reader import FieldData
        from statistics.first_order import FirstOrderStatistics, compute_mean_fields
        from statistics.second_order import SecondOrderStatistics, compute_reynolds_stresses


@dataclass
class TemporalComparison:
    """
    Container for temporal comparison results.
    
    Stores statistics from different time periods for comparison,
    along with derived quantities like changes and ratios.
    """
    # Period identifiers
    period_1_name: str = "before"
    period_2_name: str = "after"
    
    # Time ranges
    period_1_time: Tuple[float, float] = (0.0, 0.0)
    period_2_time: Tuple[float, float] = (0.0, 0.0)
    
    # First-order statistics
    u_mean_1: Optional[np.ndarray] = None
    v_mean_1: Optional[np.ndarray] = None
    w_mean_1: Optional[np.ndarray] = None
    u_mean_2: Optional[np.ndarray] = None
    v_mean_2: Optional[np.ndarray] = None
    w_mean_2: Optional[np.ndarray] = None
    
    # Second-order statistics
    tke_1: Optional[np.ndarray] = None
    tke_2: Optional[np.ndarray] = None
    uu_1: Optional[np.ndarray] = None
    uu_2: Optional[np.ndarray] = None
    vv_1: Optional[np.ndarray] = None
    vv_2: Optional[np.ndarray] = None
    ww_1: Optional[np.ndarray] = None
    ww_2: Optional[np.ndarray] = None
    uv_1: Optional[np.ndarray] = None
    uv_2: Optional[np.ndarray] = None
    
    # Sample counts
    n_samples_1: int = 0
    n_samples_2: int = 0
    
    @property
    def u_mean_change(self) -> Optional[np.ndarray]:
        """Absolute change in mean streamwise velocity."""
        if self.u_mean_1 is not None and self.u_mean_2 is not None:
            return self.u_mean_2 - self.u_mean_1
        return None
    
    @property
    def u_mean_ratio(self) -> Optional[np.ndarray]:
        """Ratio of mean streamwise velocity (after/before)."""
        if self.u_mean_1 is not None and self.u_mean_2 is not None:
            return np.where(np.abs(self.u_mean_1) > 1e-10,
                          self.u_mean_2 / self.u_mean_1,
                          np.nan)
        return None
    
    @property
    def tke_change(self) -> Optional[np.ndarray]:
        """Absolute change in TKE."""
        if self.tke_1 is not None and self.tke_2 is not None:
            return self.tke_2 - self.tke_1
        return None
    
    @property
    def tke_ratio(self) -> Optional[np.ndarray]:
        """Ratio of TKE (after/before)."""
        if self.tke_1 is not None and self.tke_2 is not None:
            return np.where(self.tke_1 > 1e-10,
                          self.tke_2 / self.tke_1,
                          np.nan)
        return None
    
    @property
    def uu_change(self) -> Optional[np.ndarray]:
        """Change in streamwise Reynolds stress."""
        if self.uu_1 is not None and self.uu_2 is not None:
            return self.uu_2 - self.uu_1
        return None
    
    @property
    def vv_change(self) -> Optional[np.ndarray]:
        """Change in wall-normal Reynolds stress."""
        if self.vv_1 is not None and self.vv_2 is not None:
            return self.vv_2 - self.vv_1
        return None
    
    @property
    def uv_change(self) -> Optional[np.ndarray]:
        """Change in Reynolds shear stress."""
        if self.uv_1 is not None and self.uv_2 is not None:
            return self.uv_2 - self.uv_1
        return None
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics (spatial averages of changes)."""
        summary = {
            'period_1': self.period_1_name,
            'period_2': self.period_2_name,
            'n_samples_1': self.n_samples_1,
            'n_samples_2': self.n_samples_2,
        }
        
        if self.u_mean_change is not None:
            summary['u_mean_change_avg'] = float(np.nanmean(self.u_mean_change))
            summary['u_mean_change_max'] = float(np.nanmax(np.abs(self.u_mean_change)))
        
        if self.tke_change is not None:
            summary['tke_change_avg'] = float(np.nanmean(self.tke_change))
            summary['tke_ratio_avg'] = float(np.nanmean(self.tke_ratio))
        
        return summary
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary of arrays."""
        result = {}
        
        for attr in ['u_mean_1', 'u_mean_2', 'v_mean_1', 'v_mean_2',
                     'tke_1', 'tke_2', 'uu_1', 'uu_2', 'vv_1', 'vv_2',
                     'uv_1', 'uv_2']:
            value = getattr(self, attr, None)
            if value is not None:
                result[attr] = value
        
        # Add derived quantities
        for attr in ['u_mean_change', 'u_mean_ratio', 'tke_change', 'tke_ratio']:
            value = getattr(self, attr, None)
            if value is not None:
                result[attr] = value
        
        return result


def compare_statistics(
    fields_before: List,
    fields_after: List,
    period_1_name: str = "before",
    period_2_name: str = "after"
) -> TemporalComparison:
    """
    Compare statistics between two time periods.
    
    Parameters
    ----------
    fields_before : List[FieldData]
        Field snapshots from first period (e.g., before step change)
    fields_after : List[FieldData]
        Field snapshots from second period (e.g., after step change)
    period_1_name : str
        Name for the first period
    period_2_name : str
        Name for the second period
        
    Returns
    -------
    TemporalComparison
        Comparison results with statistics from both periods
        
    Example
    -------
    >>> fields_before = reader.read_field_sequence(1, 100)
    >>> fields_after = reader.read_field_sequence(200, 300)
    >>> comparison = compare_statistics(fields_before, fields_after)
    >>> print(comparison.tke_ratio)  # TKE amplification factor
    """
    # Compute statistics for first period
    mean_1 = compute_mean_fields(fields_before)
    second_1 = compute_reynolds_stresses(fields_before, mean_1)
    
    # Compute statistics for second period
    mean_2 = compute_mean_fields(fields_after)
    second_2 = compute_reynolds_stresses(fields_after, mean_2)
    
    return TemporalComparison(
        period_1_name=period_1_name,
        period_2_name=period_2_name,
        period_1_time=(fields_before[0].time, fields_before[-1].time),
        period_2_time=(fields_after[0].time, fields_after[-1].time),
        u_mean_1=mean_1.u_mean,
        v_mean_1=mean_1.v_mean,
        w_mean_1=mean_1.w_mean,
        u_mean_2=mean_2.u_mean,
        v_mean_2=mean_2.v_mean,
        w_mean_2=mean_2.w_mean,
        tke_1=second_1.tke,
        tke_2=second_2.tke,
        uu_1=second_1.uu,
        uu_2=second_2.uu,
        vv_1=second_1.vv,
        vv_2=second_2.vv,
        ww_1=second_1.ww,
        ww_2=second_2.ww,
        uv_1=second_1.uv,
        uv_2=second_2.uv,
        n_samples_1=len(fields_before),
        n_samples_2=len(fields_after),
    )


@dataclass
class TimeEvolution:
    """Container for time evolution data."""
    times: np.ndarray
    values: np.ndarray
    field_name: str
    global_average: bool = True
    
    @property
    def derivative(self) -> np.ndarray:
        """Time derivative (finite difference)."""
        return np.gradient(self.values, self.times)
    
    def get_value_at_time(self, t: float) -> float:
        """Interpolate value at specific time."""
        return np.interp(t, self.times, self.values)


def compute_time_evolution(
    fields: List,
    field_name: str = 'tke',
    global_average: bool = True
) -> TimeEvolution:
    """
    Compute time evolution of a statistical quantity.
    
    Parameters
    ----------
    fields : List[FieldData]
        List of field snapshots
    field_name : str
        Name of field to track: 'tke', 'u_mean', 'v_mean', 'uu', etc.
    global_average : bool
        If True, compute spatial average at each time
        
    Returns
    -------
    TimeEvolution
        Time series of the quantity
    """
    times = []
    values = []
    
    # Track running statistics
    try:
        from ..statistics.second_order import OnlineSecondOrderComputer
    except ImportError:
        from statistics.second_order import OnlineSecondOrderComputer
    computer = OnlineSecondOrderComputer()
    
    for field in fields:
        times.append(field.time)
        computer.update(field)
        
        if computer.n >= 2:
            second_order = computer.get_statistics()
            first_order = computer.get_first_order()
            
            # Get requested field
            if field_name == 'tke':
                value = second_order.tke
            elif field_name == 'uu':
                value = second_order.uu
            elif field_name == 'vv':
                value = second_order.vv
            elif field_name == 'ww':
                value = second_order.ww
            elif field_name == 'uv':
                value = second_order.uv
            elif field_name == 'u_mean':
                value = first_order.u_mean
            elif field_name == 'v_mean':
                value = first_order.v_mean
            elif field_name == 'w_mean':
                value = first_order.w_mean
            else:
                raise ValueError(f"Unknown field: {field_name}")
            
            if global_average:
                values.append(np.mean(value))
            else:
                values.append(value.copy())
        else:
            # First sample - use instantaneous value
            if field_name in ['u_mean', 'tke']:
                values.append(0.0 if global_average else np.zeros_like(field.u))
            else:
                values.append(0.0 if global_average else np.zeros_like(field.u))
    
    return TimeEvolution(
        times=np.array(times),
        values=np.array(values),
        field_name=field_name,
        global_average=global_average,
    )


def detect_transient_duration(
    time_evolution: TimeEvolution,
    threshold: float = 0.05,
    method: str = 'derivative'
) -> Tuple[float, float]:
    """
    Detect the duration of a transient period.
    
    Parameters
    ----------
    time_evolution : TimeEvolution
        Time series of a statistical quantity
    threshold : float
        Threshold for detecting steady state
        For 'derivative': fraction of max derivative
        For 'asymptote': fraction of asymptotic value
    method : str
        Detection method: 'derivative' or 'asymptote'
        
    Returns
    -------
    Tuple[float, float]
        (transient_start, transient_end) times
        
    Notes
    -----
    For step change DNS, this function helps identify:
    - When the step change effect starts
    - When the flow reaches a new equilibrium state
    """
    times = time_evolution.times
    values = time_evolution.values
    
    if method == 'derivative':
        # Use time derivative to detect transient
        derivative = np.abs(time_evolution.derivative)
        max_deriv = np.max(derivative)
        
        if max_deriv < 1e-10:
            # No significant change
            return (times[0], times[0])
        
        # Find where derivative exceeds threshold
        transient_mask = derivative > threshold * max_deriv
        
        if not np.any(transient_mask):
            return (times[0], times[0])
        
        transient_indices = np.where(transient_mask)[0]
        start_idx = transient_indices[0]
        end_idx = transient_indices[-1]
        
        return (times[start_idx], times[end_idx])
    
    elif method == 'asymptote':
        # Compare to asymptotic value
        asymptotic_value = values[-1]  # Assume last values are steady
        
        if np.abs(asymptotic_value) < 1e-10:
            return (times[0], times[-1])
        
        # Find where values are within threshold of asymptote
        relative_diff = np.abs(values - asymptotic_value) / np.abs(asymptotic_value)
        steady_mask = relative_diff < threshold
        
        if np.all(steady_mask):
            return (times[0], times[0])
        
        # Find transition point
        steady_indices = np.where(steady_mask)[0]
        if len(steady_indices) > 0:
            end_idx = steady_indices[0]
        else:
            end_idx = len(times) - 1
        
        return (times[0], times[end_idx])
    
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_relaxation_time(
    time_evolution: TimeEvolution,
    method: str = 'exponential'
) -> float:
    """
    Estimate relaxation time for return to equilibrium.
    
    Parameters
    ----------
    time_evolution : TimeEvolution
        Time series of a statistical quantity
    method : str
        Method: 'exponential' (fit exponential decay) or 'e_folding'
        
    Returns
    -------
    float
        Estimated relaxation time scale
    """
    times = time_evolution.times
    values = time_evolution.values
    
    # Normalize values
    v_init = values[0]
    v_final = values[-1]
    
    if np.abs(v_final - v_init) < 1e-10:
        return 0.0  # No significant change
    
    # Normalized evolution
    v_norm = (values - v_init) / (v_final - v_init)
    
    if method == 'exponential':
        # Fit exponential: v_norm = 1 - exp(-t/tau)
        # ln(1 - v_norm) = -t/tau
        
        # Avoid log of zero
        v_norm_clipped = np.clip(v_norm, 1e-10, 1 - 1e-10)
        
        log_term = np.log(1 - v_norm_clipped)
        
        # Linear fit: log_term = -t/tau
        # Use least squares
        valid = np.isfinite(log_term)
        if np.sum(valid) < 2:
            return times[-1] - times[0]
        
        slope = np.polyfit(times[valid], log_term[valid], 1)[0]
        
        if slope >= 0:
            return times[-1] - times[0]  # Not decaying
        
        tau = -1.0 / slope
        return tau
    
    elif method == 'e_folding':
        # Find time to reach (1 - 1/e) ≈ 0.632 of final change
        target = 1 - np.exp(-1)  # ≈ 0.632
        
        idx = np.searchsorted(v_norm, target)
        if idx >= len(times):
            idx = len(times) - 1
        
        return times[idx] - times[0]
    
    else:
        raise ValueError(f"Unknown method: {method}")


class StepChangeAnalyzer:
    """
    Comprehensive analyzer for step change DNS simulations.
    
    Provides tools for analyzing turbulence response to step changes in:
    - Reynolds number
    - Wall temperature
    - Pressure gradient
    - Mass flow rate
    
    Parameters
    ----------
    step_time : float
        Time at which the step change occurs
    pre_step_window : float
        Duration of pre-step averaging window
    post_step_window : float
        Duration of post-step averaging window
        
    Example
    -------
    >>> analyzer = StepChangeAnalyzer(step_time=10.0)
    >>> for field in fields:
    ...     analyzer.add_field(field)
    >>> comparison = analyzer.compare_before_after()
    >>> transient = analyzer.analyze_transient()
    """
    
    def __init__(
        self,
        step_time: float,
        pre_step_window: float = None,
        post_step_window: float = None
    ):
        self.step_time = step_time
        self.pre_step_window = pre_step_window
        self.post_step_window = post_step_window
        
        self.fields_before: List = []
        self.fields_after: List = []
        self.fields_transient: List = []
        
    def add_field(self, field) -> None:
        """Add a field snapshot and classify by time period."""
        t = field.time
        
        if t < self.step_time:
            if self.pre_step_window is None or t >= self.step_time - self.pre_step_window:
                self.fields_before.append(field)
        else:
            # Categorize post-step fields
            if self.post_step_window is not None:
                if t <= self.step_time + self.post_step_window:
                    self.fields_transient.append(field)
                else:
                    self.fields_after.append(field)
            else:
                self.fields_after.append(field)
    
    def compare_before_after(self) -> TemporalComparison:
        """
        Compare pre-step and post-step statistics.
        
        Returns statistics from stable pre-step period compared to
        statistics from the new equilibrium after the step change.
        """
        if len(self.fields_before) == 0:
            raise ValueError("No pre-step fields available")
        if len(self.fields_after) == 0:
            raise ValueError("No post-step fields available")
        
        return compare_statistics(
            self.fields_before,
            self.fields_after,
            period_1_name="pre_step",
            period_2_name="post_step"
        )
    
    def analyze_transient(self) -> Dict:
        """
        Analyze the transient response after step change.
        
        Returns
        -------
        Dict
            Dictionary containing:
            - time_evolution: Time series of key quantities
            - transient_duration: Estimated transient duration
            - relaxation_time: Characteristic relaxation time
            - overshoot: Maximum overshoot (if any)
        """
        if len(self.fields_transient) == 0:
            all_post = self.fields_after if self.fields_after else []
            if len(all_post) == 0:
                raise ValueError("No post-step fields available")
            fields_for_analysis = all_post
        else:
            fields_for_analysis = self.fields_transient + self.fields_after
        
        # Compute time evolution of TKE
        tke_evolution = compute_time_evolution(fields_for_analysis, 'tke')
        
        # Detect transient duration
        transient_start, transient_end = detect_transient_duration(tke_evolution)
        
        # Estimate relaxation time
        relaxation_tau = compute_relaxation_time(tke_evolution)
        
        # Detect overshoot
        tke_values = tke_evolution.values
        if len(tke_values) > 1:
            final_value = tke_values[-1]
            max_value = np.max(tke_values)
            min_value = np.min(tke_values)
            
            if final_value > 0:
                overshoot = (max_value - final_value) / final_value
                undershoot = (final_value - min_value) / final_value
            else:
                overshoot = 0.0
                undershoot = 0.0
        else:
            overshoot = 0.0
            undershoot = 0.0
        
        return {
            'time_evolution': tke_evolution,
            'transient_start': transient_start,
            'transient_end': transient_end,
            'transient_duration': transient_end - transient_start,
            'relaxation_time': relaxation_tau,
            'overshoot': overshoot,
            'undershoot': undershoot,
        }
    
    def get_time_evolution(self, field_name: str = 'tke') -> TimeEvolution:
        """Get time evolution of a statistical quantity."""
        all_fields = self.fields_before + self.fields_transient + self.fields_after
        all_fields.sort(key=lambda f: f.time)
        
        return compute_time_evolution(all_fields, field_name)
    
    def get_summary(self) -> Dict:
        """Get summary of step change analysis."""
        comparison = self.compare_before_after() if self.fields_before and self.fields_after else None
        transient = self.analyze_transient() if (self.fields_transient or self.fields_after) else None
        
        summary = {
            'step_time': self.step_time,
            'n_fields_before': len(self.fields_before),
            'n_fields_transient': len(self.fields_transient),
            'n_fields_after': len(self.fields_after),
        }
        
        if comparison:
            summary.update(comparison.get_summary())
        
        if transient:
            summary['transient_duration'] = transient['transient_duration']
            summary['relaxation_time'] = transient['relaxation_time']
            summary['overshoot'] = transient['overshoot']
        
        return summary
