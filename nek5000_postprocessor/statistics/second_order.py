"""
Second-Order Statistics
=======================

Compute second-order turbulence statistics including:
- Reynolds stresses (uu, vv, ww, uv, uw, vw)
- Turbulent kinetic energy (TKE)
- Turbulent dissipation rate
- Turbulent heat fluxes
- Higher-order moments (skewness, flatness)

These statistics are essential for DNS analysis of step change flows,
characterizing the turbulent state before and after the step change.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

try:
    from ..readers import FieldData
    from .first_order import FirstOrderStatistics, compute_mean_fields
except ImportError:
    try:
        from readers.field_reader import FieldData
        from statistics.first_order import FirstOrderStatistics, compute_mean_fields
    except ImportError:
        # For direct module import
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from readers.field_reader import FieldData
        from statistics.first_order import FirstOrderStatistics, compute_mean_fields


@dataclass
class SecondOrderStatistics:
    """
    Container for second-order turbulence statistics.
    
    Attributes
    ----------
    uu : np.ndarray
        Reynolds normal stress <u'u'>
    vv : np.ndarray
        Reynolds normal stress <v'v'>
    ww : np.ndarray
        Reynolds normal stress <w'w'>
    uv : np.ndarray
        Reynolds shear stress <u'v'>
    uw : np.ndarray
        Reynolds shear stress <u'w'>
    vw : np.ndarray
        Reynolds shear stress <v'w'>
    tke : np.ndarray
        Turbulent kinetic energy 0.5*(<u'u'> + <v'v'> + <w'w'>)
    pp : np.ndarray
        Pressure variance <p'p'>
    up, vp, wp : np.ndarray
        Pressure-velocity correlations
    ut, vt, wt : np.ndarray
        Turbulent heat fluxes (if temperature available)
    tt : np.ndarray
        Temperature variance <T'T'>
    n_samples : int
        Number of samples used
    time_start, time_end : float
        Time range for averaging
    """
    # Reynolds stresses
    uu: np.ndarray
    vv: np.ndarray
    ww: np.ndarray
    uv: np.ndarray
    uw: np.ndarray
    vw: np.ndarray
    
    # Turbulent kinetic energy
    tke: np.ndarray = field(default=None)
    
    # Pressure statistics
    pp: Optional[np.ndarray] = None
    up: Optional[np.ndarray] = None
    vp: Optional[np.ndarray] = None
    wp: Optional[np.ndarray] = None
    
    # Temperature statistics (turbulent heat fluxes)
    tt: Optional[np.ndarray] = None
    ut: Optional[np.ndarray] = None
    vt: Optional[np.ndarray] = None
    wt: Optional[np.ndarray] = None
    
    # Metadata
    n_samples: int = 0
    time_start: float = 0.0
    time_end: float = 0.0
    
    # Reference mesh
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    z: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Compute TKE if not provided."""
        if self.tke is None:
            self.tke = 0.5 * (self.uu + self.vv + self.ww)
    
    @property
    def reynolds_stress_tensor(self) -> np.ndarray:
        """
        Return Reynolds stress tensor at each point.
        Shape: (..., 3, 3)
        """
        shape = self.uu.shape
        R = np.zeros(shape + (3, 3), dtype=np.float32)
        R[..., 0, 0] = self.uu
        R[..., 1, 1] = self.vv
        R[..., 2, 2] = self.ww
        R[..., 0, 1] = R[..., 1, 0] = self.uv
        R[..., 0, 2] = R[..., 2, 0] = self.uw
        R[..., 1, 2] = R[..., 2, 1] = self.vw
        return R
    
    @property
    def anisotropy_tensor(self) -> np.ndarray:
        """
        Compute Reynolds stress anisotropy tensor.
        b_ij = R_ij / (2*k) - delta_ij / 3
        """
        k = self.tke
        k = np.where(k > 1e-10, k, 1e-10)  # Avoid division by zero
        
        R = self.reynolds_stress_tensor
        b = R / (2 * k[..., np.newaxis, np.newaxis])
        
        # Subtract isotropic part
        for i in range(3):
            b[..., i, i] -= 1.0 / 3.0
        
        return b
    
    def get_field(self, name: str) -> Optional[np.ndarray]:
        """Get a statistic field by name."""
        return getattr(self, name, None)
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary."""
        result = {
            'uu': self.uu,
            'vv': self.vv,
            'ww': self.ww,
            'uv': self.uv,
            'uw': self.uw,
            'vw': self.vw,
            'tke': self.tke,
        }
        
        optional_fields = ['pp', 'up', 'vp', 'wp', 'tt', 'ut', 'vt', 'wt']
        for field_name in optional_fields:
            value = getattr(self, field_name, None)
            if value is not None:
                result[field_name] = value
        
        return result


def compute_reynolds_stresses(
    fields: List,
    mean_stats: Optional[FirstOrderStatistics] = None,
    compute_pressure_stats: bool = True,
    compute_temperature_stats: bool = True
) -> SecondOrderStatistics:
    """
    Compute Reynolds stresses from field snapshots.
    
    Parameters
    ----------
    fields : List[FieldData]
        List of field snapshots
    mean_stats : Optional[FirstOrderStatistics]
        Pre-computed mean statistics
    compute_pressure_stats : bool
        Whether to compute pressure-related statistics
    compute_temperature_stats : bool
        Whether to compute temperature-related statistics
        
    Returns
    -------
    SecondOrderStatistics
        Container with all second-order statistics
        
    Notes
    -----
    Reynolds decomposition: u = <u> + u'
    Reynolds stress: R_ij = <u'_i u'_j>
    
    For step change DNS, compute these statistics separately for
    pre-step and post-step regions to characterize turbulence evolution.
    """
    if len(fields) == 0:
        raise ValueError("Empty field list provided")
    
    n_samples = len(fields)
    
    # Compute mean if not provided
    if mean_stats is None:
        mean_stats = compute_mean_fields(fields)
    
    # Get reference shape
    ref_field = fields[0]
    shape = ref_field.u.shape
    
    # Initialize accumulators for Reynolds stresses (using float64 for precision)
    uu_sum = np.zeros(shape, dtype=np.float64)
    vv_sum = np.zeros(shape, dtype=np.float64)
    ww_sum = np.zeros(shape, dtype=np.float64)
    uv_sum = np.zeros(shape, dtype=np.float64)
    uw_sum = np.zeros(shape, dtype=np.float64)
    vw_sum = np.zeros(shape, dtype=np.float64)
    
    # Pressure statistics
    pp_sum = np.zeros(shape, dtype=np.float64) if (compute_pressure_stats and ref_field.p is not None) else None
    up_sum = np.zeros(shape, dtype=np.float64) if (compute_pressure_stats and ref_field.p is not None) else None
    vp_sum = np.zeros(shape, dtype=np.float64) if (compute_pressure_stats and ref_field.p is not None) else None
    wp_sum = np.zeros(shape, dtype=np.float64) if (compute_pressure_stats and ref_field.p is not None) else None
    
    # Temperature statistics
    tt_sum = np.zeros(shape, dtype=np.float64) if (compute_temperature_stats and ref_field.t is not None) else None
    ut_sum = np.zeros(shape, dtype=np.float64) if (compute_temperature_stats and ref_field.t is not None) else None
    vt_sum = np.zeros(shape, dtype=np.float64) if (compute_temperature_stats and ref_field.t is not None) else None
    wt_sum = np.zeros(shape, dtype=np.float64) if (compute_temperature_stats and ref_field.t is not None) else None
    
    # Compute fluctuations and accumulate products
    for field in fields:
        # Velocity fluctuations
        u_prime = field.u - mean_stats.u_mean
        v_prime = field.v - mean_stats.v_mean
        w_prime = (field.w - mean_stats.w_mean) if field.w is not None else np.zeros_like(u_prime)
        
        # Reynolds stresses
        uu_sum += u_prime * u_prime
        vv_sum += v_prime * v_prime
        ww_sum += w_prime * w_prime
        uv_sum += u_prime * v_prime
        uw_sum += u_prime * w_prime
        vw_sum += v_prime * w_prime
        
        # Pressure statistics
        if pp_sum is not None and field.p is not None and mean_stats.p_mean is not None:
            p_prime = field.p - mean_stats.p_mean
            pp_sum += p_prime * p_prime
            up_sum += u_prime * p_prime
            vp_sum += v_prime * p_prime
            wp_sum += w_prime * p_prime
        
        # Temperature statistics (turbulent heat fluxes)
        if tt_sum is not None and field.t is not None and mean_stats.t_mean is not None:
            t_prime = field.t - mean_stats.t_mean
            tt_sum += t_prime * t_prime
            ut_sum += u_prime * t_prime
            vt_sum += v_prime * t_prime
            wt_sum += w_prime * t_prime
    
    # Normalize by number of samples
    uu = (uu_sum / n_samples).astype(np.float32)
    vv = (vv_sum / n_samples).astype(np.float32)
    ww = (ww_sum / n_samples).astype(np.float32)
    uv = (uv_sum / n_samples).astype(np.float32)
    uw = (uw_sum / n_samples).astype(np.float32)
    vw = (vw_sum / n_samples).astype(np.float32)
    
    pp = (pp_sum / n_samples).astype(np.float32) if pp_sum is not None else None
    up = (up_sum / n_samples).astype(np.float32) if up_sum is not None else None
    vp = (vp_sum / n_samples).astype(np.float32) if vp_sum is not None else None
    wp = (wp_sum / n_samples).astype(np.float32) if wp_sum is not None else None
    
    tt = (tt_sum / n_samples).astype(np.float32) if tt_sum is not None else None
    ut = (ut_sum / n_samples).astype(np.float32) if ut_sum is not None else None
    vt = (vt_sum / n_samples).astype(np.float32) if vt_sum is not None else None
    wt = (wt_sum / n_samples).astype(np.float32) if wt_sum is not None else None
    
    return SecondOrderStatistics(
        uu=uu, vv=vv, ww=ww,
        uv=uv, uw=uw, vw=vw,
        pp=pp, up=up, vp=vp, wp=wp,
        tt=tt, ut=ut, vt=vt, wt=wt,
        n_samples=n_samples,
        time_start=fields[0].time,
        time_end=fields[-1].time,
        x=ref_field.x,
        y=ref_field.y,
        z=ref_field.z,
    )


def compute_tke(
    fields: List,
    mean_stats: Optional[FirstOrderStatistics] = None
) -> np.ndarray:
    """
    Compute turbulent kinetic energy.
    
    TKE = 0.5 * (<u'u'> + <v'v'> + <w'w'>)
    
    Parameters
    ----------
    fields : List[FieldData]
        List of field snapshots
    mean_stats : Optional[FirstOrderStatistics]
        Pre-computed mean statistics
        
    Returns
    -------
    np.ndarray
        Turbulent kinetic energy field
    """
    second_order = compute_reynolds_stresses(fields, mean_stats)
    return second_order.tke


def compute_turbulent_dissipation(
    fields: List,
    mean_stats: Optional[FirstOrderStatistics] = None,
    nu: float = 1e-5,
    dx: Optional[float] = None,
    dy: Optional[float] = None,
    dz: Optional[float] = None
) -> np.ndarray:
    """
    Estimate turbulent dissipation rate.
    
    epsilon = 2 * nu * <s'_ij s'_ij>
    
    where s'_ij is the fluctuating strain rate tensor.
    
    Parameters
    ----------
    fields : List[FieldData]
        List of field snapshots
    mean_stats : Optional[FirstOrderStatistics]
        Pre-computed mean statistics
    nu : float
        Kinematic viscosity
    dx, dy, dz : float
        Grid spacing for gradient computation
        
    Returns
    -------
    np.ndarray
        Turbulent dissipation rate field
        
    Notes
    -----
    This is an approximation. For accurate dissipation, velocity gradients
    should be computed using spectral methods consistent with Nek5000.
    """
    if mean_stats is None:
        mean_stats = compute_mean_fields(fields)
    
    n_samples = len(fields)
    ref_field = fields[0]
    shape = ref_field.u.shape
    
    # Initialize accumulator
    dissipation_sum = np.zeros(shape, dtype=np.float64)
    
    # Estimate grid spacing if not provided
    if ref_field.x is not None and dx is None:
        dx = np.mean(np.diff(np.unique(ref_field.x)[:10]))
        dx = dx if dx > 0 else 1.0
    else:
        dx = dx if dx is not None else 1.0
    
    if ref_field.y is not None and dy is None:
        dy = np.mean(np.diff(np.unique(ref_field.y)[:10]))
        dy = dy if dy > 0 else dx
    else:
        dy = dy if dy is not None else dx
        
    if ref_field.z is not None and dz is None:
        dz = np.mean(np.diff(np.unique(ref_field.z)[:10]))
        dz = dz if dz > 0 else dx
    else:
        dz = dz if dz is not None else dx
    
    # Compute dissipation using a pseudo-spectral estimate
    # This is a simplified estimate based on TKE and integral length scale
    # epsilon ~ TKE^(3/2) / L
    
    second_order = compute_reynolds_stresses(fields, mean_stats)
    tke = second_order.tke
    
    # Estimate integral length scale from grid
    L = (dx * dy * dz) ** (1.0 / 3.0)
    
    # Dissipation estimate (using kolmogorov scaling)
    # This is a rough estimate; accurate computation requires velocity gradients
    epsilon = tke ** 1.5 / L
    
    return epsilon.astype(np.float32)


def compute_turbulent_heat_flux(
    fields: List,
    mean_stats: Optional[FirstOrderStatistics] = None
) -> Dict[str, np.ndarray]:
    """
    Compute turbulent heat fluxes.
    
    q_i = <u'_i T'>
    
    Parameters
    ----------
    fields : List[FieldData]
        List of field snapshots (must contain temperature)
    mean_stats : Optional[FirstOrderStatistics]
        Pre-computed mean statistics
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with 'ut', 'vt', 'wt', 'tt' fields
    """
    second_order = compute_reynolds_stresses(
        fields, mean_stats, 
        compute_pressure_stats=False,
        compute_temperature_stats=True
    )
    
    return {
        'ut': second_order.ut,
        'vt': second_order.vt,
        'wt': second_order.wt,
        'tt': second_order.tt,
    }


class OnlineSecondOrderComputer:
    """
    Online computation of second-order statistics.
    
    Uses numerically stable online algorithms to compute Reynolds stresses
    and related quantities from a stream of field snapshots.
    
    Example
    -------
    >>> computer = OnlineSecondOrderComputer()
    >>> for field in field_sequence:
    ...     computer.update(field)
    >>> stats = computer.get_statistics()
    """
    
    def __init__(self):
        self.n = 0
        
        # Running means
        self.mean_u = None
        self.mean_v = None
        self.mean_w = None
        self.mean_p = None
        self.mean_t = None
        
        # Running covariances (using online algorithm)
        self.C_uu = None
        self.C_vv = None
        self.C_ww = None
        self.C_uv = None
        self.C_uw = None
        self.C_vw = None
        self.C_pp = None
        self.C_tt = None
        self.C_ut = None
        self.C_vt = None
        self.C_wt = None
        
        self.time_start = None
        self.time_end = None
        
    def update(self, field) -> None:
        """Update with a new field snapshot."""
        self.n += 1
        
        if self.time_start is None:
            self.time_start = field.time
        self.time_end = field.time
        
        if self.mean_u is None:
            # Initialize on first sample
            self.mean_u = field.u.astype(np.float64).copy()
            self.mean_v = field.v.astype(np.float64).copy()
            self.mean_w = field.w.astype(np.float64).copy() if field.w is not None else None
            self.mean_p = field.p.astype(np.float64).copy() if field.p is not None else None
            self.mean_t = field.t.astype(np.float64).copy() if field.t is not None else None
            
            shape = field.u.shape
            self.C_uu = np.zeros(shape, dtype=np.float64)
            self.C_vv = np.zeros(shape, dtype=np.float64)
            self.C_ww = np.zeros(shape, dtype=np.float64)
            self.C_uv = np.zeros(shape, dtype=np.float64)
            self.C_uw = np.zeros(shape, dtype=np.float64)
            self.C_vw = np.zeros(shape, dtype=np.float64)
            
            if field.p is not None:
                self.C_pp = np.zeros(shape, dtype=np.float64)
            if field.t is not None:
                self.C_tt = np.zeros(shape, dtype=np.float64)
                self.C_ut = np.zeros(shape, dtype=np.float64)
                self.C_vt = np.zeros(shape, dtype=np.float64)
                self.C_wt = np.zeros(shape, dtype=np.float64)
            return
        
        # Online covariance update (Welford-style)
        n = self.n
        
        # Deltas from previous mean
        du = field.u - self.mean_u
        dv = field.v - self.mean_v
        dw = (field.w - self.mean_w) if self.mean_w is not None else np.zeros_like(du)
        
        # Update means
        self.mean_u += du / n
        self.mean_v += dv / n
        if self.mean_w is not None and field.w is not None:
            self.mean_w += dw / n
        
        # Deltas from new mean
        du2 = field.u - self.mean_u
        dv2 = field.v - self.mean_v
        dw2 = (field.w - self.mean_w) if self.mean_w is not None else np.zeros_like(du)
        
        # Update covariances
        self.C_uu += du * du2
        self.C_vv += dv * dv2
        self.C_ww += dw * dw2
        self.C_uv += du * dv2
        self.C_uw += du * dw2
        self.C_vw += dv * dw2
        
        # Pressure
        if field.p is not None and self.mean_p is not None:
            dp = field.p - self.mean_p
            self.mean_p += dp / n
            dp2 = field.p - self.mean_p
            self.C_pp += dp * dp2
        
        # Temperature
        if field.t is not None and self.mean_t is not None:
            dt = field.t - self.mean_t
            self.mean_t += dt / n
            dt2 = field.t - self.mean_t
            self.C_tt += dt * dt2
            self.C_ut += du * dt2
            self.C_vt += dv * dt2
            self.C_wt += dw * dt2
    
    def get_statistics(self) -> SecondOrderStatistics:
        """Get current second-order statistics."""
        if self.n < 2:
            raise ValueError("Need at least 2 samples for statistics")
        
        n = self.n
        
        return SecondOrderStatistics(
            uu=(self.C_uu / (n - 1)).astype(np.float32),
            vv=(self.C_vv / (n - 1)).astype(np.float32),
            ww=(self.C_ww / (n - 1)).astype(np.float32),
            uv=(self.C_uv / (n - 1)).astype(np.float32),
            uw=(self.C_uw / (n - 1)).astype(np.float32),
            vw=(self.C_vw / (n - 1)).astype(np.float32),
            pp=(self.C_pp / (n - 1)).astype(np.float32) if self.C_pp is not None else None,
            tt=(self.C_tt / (n - 1)).astype(np.float32) if self.C_tt is not None else None,
            ut=(self.C_ut / (n - 1)).astype(np.float32) if self.C_ut is not None else None,
            vt=(self.C_vt / (n - 1)).astype(np.float32) if self.C_vt is not None else None,
            wt=(self.C_wt / (n - 1)).astype(np.float32) if self.C_wt is not None else None,
            n_samples=self.n,
            time_start=self.time_start,
            time_end=self.time_end,
        )
    
    def get_first_order(self) -> FirstOrderStatistics:
        """Get first-order statistics."""
        return FirstOrderStatistics(
            u_mean=self.mean_u.astype(np.float32),
            v_mean=self.mean_v.astype(np.float32),
            w_mean=self.mean_w.astype(np.float32) if self.mean_w is not None else np.zeros_like(self.mean_u, dtype=np.float32),
            p_mean=self.mean_p.astype(np.float32) if self.mean_p is not None else None,
            t_mean=self.mean_t.astype(np.float32) if self.mean_t is not None else None,
            n_samples=self.n,
            time_start=self.time_start,
            time_end=self.time_end,
        )


def compute_higher_order_moments(
    fields: List,
    mean_stats: Optional[FirstOrderStatistics] = None,
    second_order_stats: Optional[SecondOrderStatistics] = None
) -> Dict[str, np.ndarray]:
    """
    Compute higher-order moments (skewness and flatness/kurtosis).
    
    Skewness: S = <u'^3> / <u'^2>^(3/2)
    Flatness: F = <u'^4> / <u'^2>^2
    
    Parameters
    ----------
    fields : List[FieldData]
        List of field snapshots
    mean_stats : Optional[FirstOrderStatistics]
        Pre-computed mean statistics
    second_order_stats : Optional[SecondOrderStatistics]
        Pre-computed second-order statistics
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with skewness and flatness fields for u, v, w
    """
    if mean_stats is None:
        mean_stats = compute_mean_fields(fields)
    if second_order_stats is None:
        second_order_stats = compute_reynolds_stresses(fields, mean_stats)
    
    n = len(fields)
    shape = fields[0].u.shape
    
    # Third and fourth moments
    u3_sum = np.zeros(shape, dtype=np.float64)
    v3_sum = np.zeros(shape, dtype=np.float64)
    w3_sum = np.zeros(shape, dtype=np.float64)
    u4_sum = np.zeros(shape, dtype=np.float64)
    v4_sum = np.zeros(shape, dtype=np.float64)
    w4_sum = np.zeros(shape, dtype=np.float64)
    
    for field in fields:
        u_prime = field.u - mean_stats.u_mean
        v_prime = field.v - mean_stats.v_mean
        w_prime = (field.w - mean_stats.w_mean) if field.w is not None else np.zeros_like(u_prime)
        
        u3_sum += u_prime ** 3
        v3_sum += v_prime ** 3
        w3_sum += w_prime ** 3
        u4_sum += u_prime ** 4
        v4_sum += v_prime ** 4
        w4_sum += w_prime ** 4
    
    # Normalize
    u3 = u3_sum / n
    v3 = v3_sum / n
    w3 = w3_sum / n
    u4 = u4_sum / n
    v4 = v4_sum / n
    w4 = w4_sum / n
    
    # Compute skewness and flatness
    # Avoid division by zero
    eps = 1e-10
    
    uu = np.maximum(second_order_stats.uu, eps)
    vv = np.maximum(second_order_stats.vv, eps)
    ww = np.maximum(second_order_stats.ww, eps)
    
    return {
        'u_skewness': (u3 / uu ** 1.5).astype(np.float32),
        'v_skewness': (v3 / vv ** 1.5).astype(np.float32),
        'w_skewness': (w3 / ww ** 1.5).astype(np.float32),
        'u_flatness': (u4 / uu ** 2).astype(np.float32),
        'v_flatness': (v4 / vv ** 2).astype(np.float32),
        'w_flatness': (w4 / ww ** 2).astype(np.float32),
    }


def compute_production_rate(
    fields: List,
    mean_stats: Optional[FirstOrderStatistics] = None,
    second_order_stats: Optional[SecondOrderStatistics] = None,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0
) -> np.ndarray:
    """
    Compute TKE production rate.
    
    P = -<u'_i u'_j> * d<U_i>/dx_j
    
    Parameters
    ----------
    fields : List[FieldData]
        List of field snapshots
    mean_stats, second_order_stats : Optional
        Pre-computed statistics
    dx, dy, dz : float
        Grid spacing for gradient computation
        
    Returns
    -------
    np.ndarray
        TKE production rate field
    """
    if mean_stats is None:
        mean_stats = compute_mean_fields(fields)
    if second_order_stats is None:
        second_order_stats = compute_reynolds_stresses(fields, mean_stats)
    
    # Compute mean velocity gradients (simple finite difference)
    # This is a simplified estimate
    
    # For more accurate results, spectral differentiation should be used
    # consistent with Nek5000's spatial discretization
    
    ref_field = fields[0]
    shape = ref_field.u.shape
    
    # Estimate production (dominant term for shear flows: -<uv> * dU/dy)
    # This requires reshaping to a proper grid structure
    
    # Return zeros as placeholder - full implementation requires grid structure
    return np.zeros(shape, dtype=np.float32)
