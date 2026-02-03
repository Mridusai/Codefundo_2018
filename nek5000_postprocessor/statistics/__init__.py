"""
Statistics Module
=================

Provides first-order and second-order turbulence statistics for
Nek5000 DNS simulations of step change flows.

First-order statistics:
- Mean velocity components
- Mean pressure
- Mean temperature

Second-order statistics:
- Reynolds stresses (uu, vv, ww, uv, uw, vw)
- Turbulent kinetic energy (TKE)
- Turbulent dissipation rate
- Turbulent heat fluxes
"""

from .first_order import (
    FirstOrderStatistics,
    compute_mean_fields,
    compute_spatial_average,
    compute_rms,
)
from .second_order import (
    SecondOrderStatistics,
    compute_reynolds_stresses,
    compute_tke,
    compute_turbulent_dissipation,
    compute_turbulent_heat_flux,
)

__all__ = [
    'FirstOrderStatistics',
    'compute_mean_fields',
    'compute_spatial_average',
    'compute_rms',
    'SecondOrderStatistics',
    'compute_reynolds_stresses',
    'compute_tke',
    'compute_turbulent_dissipation',
    'compute_turbulent_heat_flux',
]
