"""
Physics Utilities
=================

Physical quantities and scaling for turbulence analysis.
"""

import numpy as np
from typing import Dict, Optional, Tuple


def compute_friction_velocity(
    du_dy_wall: float,
    nu: float
) -> float:
    """
    Compute friction velocity from wall shear.
    
    u_tau = sqrt(tau_w / rho) = sqrt(nu * du/dy|_wall)
    
    Parameters
    ----------
    du_dy_wall : float
        Velocity gradient at the wall
    nu : float
        Kinematic viscosity
        
    Returns
    -------
    float
        Friction velocity u_tau
    """
    tau_w = nu * np.abs(du_dy_wall)
    return np.sqrt(tau_w)


def compute_wall_units(
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray = None,
    tke: np.ndarray = None,
    u_tau: float = 1.0,
    nu: float = 1e-5
) -> Dict[str, np.ndarray]:
    """
    Compute quantities in wall units (+ scaling).
    
    Parameters
    ----------
    y : np.ndarray
        Wall distance
    u : np.ndarray
        Mean streamwise velocity
    v : np.ndarray
        Mean wall-normal velocity
    tke : np.ndarray
        Turbulent kinetic energy
    u_tau : float
        Friction velocity
    nu : float
        Kinematic viscosity
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with plus-scaled quantities
    """
    # Viscous length scale
    delta_nu = nu / u_tau
    
    result = {
        'y_plus': y / delta_nu,
        'u_plus': u / u_tau,
    }
    
    if v is not None:
        result['v_plus'] = v / u_tau
    
    if tke is not None:
        result['tke_plus'] = tke / u_tau**2
    
    return result


def compute_bulk_velocity(
    u: np.ndarray,
    y: np.ndarray,
    half_height: float = 1.0
) -> float:
    """
    Compute bulk (average) velocity for channel flow.
    
    U_b = (1/2h) * integral(-h to h) U dy
    
    Parameters
    ----------
    u : np.ndarray
        Velocity profile
    y : np.ndarray
        Wall-normal coordinate
    half_height : float
        Channel half-height
        
    Returns
    -------
    float
        Bulk velocity
    """
    # Sort by y for integration
    sort_idx = np.argsort(y)
    y_sorted = y[sort_idx]
    u_sorted = u[sort_idx]
    
    # Trapezoidal integration
    U_b = np.trapz(u_sorted, y_sorted) / (2 * half_height)
    
    return U_b


def compute_reynolds_number(
    U_ref: float,
    L_ref: float,
    nu: float
) -> float:
    """
    Compute Reynolds number.
    
    Re = U_ref * L_ref / nu
    
    Parameters
    ----------
    U_ref : float
        Reference velocity (e.g., bulk velocity, centerline velocity)
    L_ref : float
        Reference length (e.g., half-height, diameter)
    nu : float
        Kinematic viscosity
        
    Returns
    -------
    float
        Reynolds number
    """
    return U_ref * L_ref / nu


def compute_friction_reynolds_number(
    u_tau: float,
    half_height: float,
    nu: float
) -> float:
    """
    Compute friction Reynolds number.
    
    Re_tau = u_tau * delta / nu
    
    Parameters
    ----------
    u_tau : float
        Friction velocity
    half_height : float
        Channel half-height (or pipe radius)
    nu : float
        Kinematic viscosity
        
    Returns
    -------
    float
        Friction Reynolds number
    """
    return u_tau * half_height / nu


def estimate_kolmogorov_scales(
    epsilon: float,
    nu: float
) -> Dict[str, float]:
    """
    Estimate Kolmogorov microscales.
    
    Parameters
    ----------
    epsilon : float
        Turbulent dissipation rate
    nu : float
        Kinematic viscosity
        
    Returns
    -------
    Dict[str, float]
        Dictionary with eta (length), tau (time), v (velocity)
    """
    if epsilon <= 0:
        return {'eta': np.inf, 'tau': np.inf, 'v': 0}
    
    eta = (nu**3 / epsilon)**0.25
    tau = (nu / epsilon)**0.5
    v = (nu * epsilon)**0.25
    
    return {
        'eta': eta,    # Kolmogorov length scale
        'tau': tau,    # Kolmogorov time scale
        'v': v,        # Kolmogorov velocity scale
    }


def estimate_taylor_microscale(
    tke: float,
    epsilon: float,
    nu: float
) -> float:
    """
    Estimate Taylor microscale.
    
    lambda = sqrt(10 * nu * k / epsilon)
    
    Parameters
    ----------
    tke : float
        Turbulent kinetic energy
    epsilon : float
        Turbulent dissipation rate
    nu : float
        Kinematic viscosity
        
    Returns
    -------
    float
        Taylor microscale
    """
    if epsilon <= 0:
        return np.inf
    
    return np.sqrt(10 * nu * tke / epsilon)


def compute_integral_length_scale(
    uu: np.ndarray,
    y: np.ndarray
) -> float:
    """
    Estimate integral length scale from Reynolds stress profile.
    
    L = integral(0 to inf) R_uu(r)/R_uu(0) dr
    
    This is a rough estimate based on the width of the 
    stress profile.
    
    Parameters
    ----------
    uu : np.ndarray
        Streamwise Reynolds stress profile
    y : np.ndarray
        Wall-normal coordinate
        
    Returns
    -------
    float
        Estimated integral length scale
    """
    # Sort by y
    sort_idx = np.argsort(y)
    y_sorted = y[sort_idx]
    uu_sorted = uu[sort_idx]
    
    # Find location of maximum
    max_idx = np.argmax(uu_sorted)
    uu_max = uu_sorted[max_idx]
    
    if uu_max <= 0:
        return 0.0
    
    # Find half-height locations (where uu = 0.5 * uu_max)
    above_half = uu_sorted >= 0.5 * uu_max
    
    if np.sum(above_half) < 2:
        return y_sorted[-1] - y_sorted[0]
    
    indices = np.where(above_half)[0]
    L = y_sorted[indices[-1]] - y_sorted[indices[0]]
    
    return L


def compute_turbulent_prandtl_number(
    nu_t: np.ndarray,
    alpha_t: np.ndarray
) -> np.ndarray:
    """
    Compute turbulent Prandtl number.
    
    Pr_t = nu_t / alpha_t
    
    Parameters
    ----------
    nu_t : np.ndarray
        Turbulent (eddy) viscosity
    alpha_t : np.ndarray
        Turbulent thermal diffusivity
        
    Returns
    -------
    np.ndarray
        Turbulent Prandtl number
    """
    return np.where(alpha_t > 1e-10, nu_t / alpha_t, np.nan)


def compute_eddy_viscosity(
    uv: np.ndarray,
    du_dy: np.ndarray
) -> np.ndarray:
    """
    Compute turbulent eddy viscosity from Boussinesq hypothesis.
    
    -<u'v'> = nu_t * dU/dy
    nu_t = -<u'v'> / (dU/dy)
    
    Parameters
    ----------
    uv : np.ndarray
        Reynolds shear stress <u'v'>
    du_dy : np.ndarray
        Mean velocity gradient
        
    Returns
    -------
    np.ndarray
        Eddy viscosity
    """
    return np.where(np.abs(du_dy) > 1e-10, -uv / du_dy, 0.0)


def compute_production_dissipation_ratio(
    production: np.ndarray,
    dissipation: np.ndarray
) -> np.ndarray:
    """
    Compute ratio of TKE production to dissipation.
    
    P/epsilon indicates local equilibrium (P/epsilon = 1 for equilibrium).
    
    Parameters
    ----------
    production : np.ndarray
        TKE production rate
    dissipation : np.ndarray
        TKE dissipation rate
        
    Returns
    -------
    np.ndarray
        P/epsilon ratio
    """
    return np.where(dissipation > 1e-10, production / dissipation, np.nan)


def law_of_the_wall(
    y_plus: np.ndarray,
    kappa: float = 0.41,
    B: float = 5.2
) -> np.ndarray:
    """
    Compute theoretical velocity profile using law of the wall.
    
    Viscous sublayer (y+ < 5): u+ = y+
    Log layer (y+ > 30): u+ = (1/kappa) * ln(y+) + B
    
    Parameters
    ----------
    y_plus : np.ndarray
        Wall distance in wall units
    kappa : float
        von Karman constant
    B : float
        Log-law intercept
        
    Returns
    -------
    np.ndarray
        Velocity in wall units (u+)
    """
    u_plus = np.zeros_like(y_plus)
    
    # Viscous sublayer
    viscous = y_plus < 5
    u_plus[viscous] = y_plus[viscous]
    
    # Buffer layer (smooth blend)
    buffer_layer = (y_plus >= 5) & (y_plus < 30)
    # Use Spalding's formula or similar for buffer layer
    # Simplified: linear interpolation
    y5 = 5.0
    y30 = 30.0
    u5 = 5.0
    u30 = (1/kappa) * np.log(30) + B
    
    y_buffer = y_plus[buffer_layer]
    u_plus[buffer_layer] = u5 + (u30 - u5) * (y_buffer - y5) / (y30 - y5)
    
    # Log layer
    log_layer = y_plus >= 30
    u_plus[log_layer] = (1/kappa) * np.log(y_plus[log_layer]) + B
    
    return u_plus


def compute_skin_friction_coefficient(
    u_tau: float,
    U_bulk: float
) -> float:
    """
    Compute skin friction coefficient.
    
    C_f = 2 * (u_tau / U_bulk)^2 = 2 * tau_w / (rho * U_bulk^2)
    
    Parameters
    ----------
    u_tau : float
        Friction velocity
    U_bulk : float
        Bulk velocity
        
    Returns
    -------
    float
        Skin friction coefficient
    """
    return 2 * (u_tau / U_bulk)**2


def estimate_pressure_gradient(
    u_tau: float,
    half_height: float
) -> float:
    """
    Estimate streamwise pressure gradient for channel flow.
    
    dP/dx = -tau_w / h = -rho * u_tau^2 / h
    
    Parameters
    ----------
    u_tau : float
        Friction velocity
    half_height : float
        Channel half-height
        
    Returns
    -------
    float
        Pressure gradient (per unit density)
    """
    return -u_tau**2 / half_height
