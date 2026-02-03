"""
Temporal Averaging and Comparison Module
=========================================

Provides utilities for:
- Temporal averaging over different time windows
- Running averages and exponential smoothing
- Temporal comparison of statistics before/after step change
- Convergence analysis of time-averaged statistics
"""

from .temporal_averaging import (
    TemporalAverager,
    running_average,
    exponential_moving_average,
    windowed_statistics,
)
from .temporal_comparison import (
    TemporalComparison,
    compare_statistics,
    compute_time_evolution,
    detect_transient_duration,
)

__all__ = [
    'TemporalAverager',
    'running_average',
    'exponential_moving_average',
    'windowed_statistics',
    'TemporalComparison',
    'compare_statistics',
    'compute_time_evolution',
    'detect_transient_duration',
]
