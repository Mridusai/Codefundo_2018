"""
Visualization Module
====================

Plotting and visualization tools for Nek5000 DNS post-processing results.

Features:
- Field contour and surface plots
- Statistics profiles
- Temporal evolution plots
- Before/after comparison plots
"""

from .plotting import (
    plot_field,
    plot_contour,
    plot_profile,
    plot_statistics,
    plot_temporal_comparison,
    plot_time_evolution,
    plot_reynolds_stresses,
    create_step_change_figure,
    save_figure,
)

__all__ = [
    'plot_field',
    'plot_contour',
    'plot_profile',
    'plot_statistics',
    'plot_temporal_comparison',
    'plot_time_evolution',
    'plot_reynolds_stresses',
    'create_step_change_figure',
    'save_figure',
]
