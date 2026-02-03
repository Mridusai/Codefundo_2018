"""
Nek5000 DNS Post-Processor
==========================

A comprehensive post-processing toolkit for Nek5000 Direct Numerical Simulations
of step change flows. Includes support for:

- Reading Nek5000 field files (.fld, f00001 format)
- First-order statistics (mean velocity, pressure)
- Second-order statistics (Reynolds stresses, TKE, turbulent dissipation)
- Temporal averaging and comparison utilities
- Visualization and plotting tools

Author: Mridusai
"""

__version__ = '1.0.0'
__author__ = 'Mridusai'

from .readers import Nek5000Reader, read_field_file, read_mesh
from .statistics import (
    FirstOrderStatistics,
    SecondOrderStatistics,
    compute_mean_fields,
    compute_reynolds_stresses,
    compute_tke,
)
from .temporal import (
    TemporalAverager,
    TemporalComparison,
    running_average,
)
from .visualization import (
    plot_field,
    plot_statistics,
    plot_temporal_comparison,
)

__all__ = [
    'Nek5000Reader',
    'read_field_file',
    'read_mesh',
    'FirstOrderStatistics',
    'SecondOrderStatistics',
    'compute_mean_fields',
    'compute_reynolds_stresses',
    'compute_tke',
    'TemporalAverager',
    'TemporalComparison',
    'running_average',
    'plot_field',
    'plot_statistics',
    'plot_temporal_comparison',
]
