"""
Grid Utilities
==============

Grid manipulation and interpolation utilities for Nek5000 data.
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
import warnings


def interpolate_to_regular_grid(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    nx: int = 100,
    ny: int = 100,
    method: str = 'linear',
    fill_value: float = np.nan
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate scattered data to regular grid.
    
    Parameters
    ----------
    x, y : np.ndarray
        Coordinate arrays
    values : np.ndarray
        Field values
    nx, ny : int
        Number of points in regular grid
    method : str
        Interpolation method: 'linear', 'cubic', 'nearest'
    fill_value : float
        Value for points outside convex hull
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (X_grid, Y_grid, values_grid) - meshgrid arrays
    """
    try:
        from scipy.interpolate import griddata
    except ImportError:
        raise ImportError("scipy required for interpolation")
    
    # Flatten arrays
    x_flat = x.flatten()
    y_flat = y.flatten()
    v_flat = values.flatten()
    
    # Create regular grid
    x_min, x_max = np.min(x_flat), np.max(x_flat)
    y_min, y_max = np.min(y_flat), np.max(y_flat)
    
    xi = np.linspace(x_min, x_max, nx)
    yi = np.linspace(y_min, y_max, ny)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate
    Vi = griddata(
        (x_flat, y_flat), v_flat, (Xi, Yi),
        method=method,
        fill_value=fill_value
    )
    
    return Xi, Yi, Vi


def extract_profile(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    profile_type: str = 'vertical',
    location: float = None,
    n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract 1D profile from 2D field data.
    
    Parameters
    ----------
    x, y : np.ndarray
        Coordinate arrays
    values : np.ndarray
        Field values
    profile_type : str
        'vertical' (constant x) or 'horizontal' (constant y)
    location : float
        x-coordinate (for vertical) or y-coordinate (for horizontal)
    n_points : int
        Number of points in output profile
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (coordinate, values) along profile
    """
    x_flat = x.flatten()
    y_flat = y.flatten()
    v_flat = values.flatten()
    
    if profile_type == 'vertical':
        # Extract profile at constant x
        if location is None:
            location = 0.5 * (np.min(x_flat) + np.max(x_flat))
        
        # Find nearest x
        x_unique = np.unique(x_flat)
        x_profile = x_unique[np.argmin(np.abs(x_unique - location))]
        
        # Extract points near this x
        tolerance = (np.max(x_flat) - np.min(x_flat)) / 100
        mask = np.abs(x_flat - x_profile) < tolerance
        
        y_profile = y_flat[mask]
        v_profile = v_flat[mask]
        
        # Sort by y
        sort_idx = np.argsort(y_profile)
        y_profile = y_profile[sort_idx]
        v_profile = v_profile[sort_idx]
        
        return y_profile, v_profile
        
    elif profile_type == 'horizontal':
        # Extract profile at constant y
        if location is None:
            location = 0.5 * (np.min(y_flat) + np.max(y_flat))
        
        # Find nearest y
        y_unique = np.unique(y_flat)
        y_profile = y_unique[np.argmin(np.abs(y_unique - location))]
        
        # Extract points near this y
        tolerance = (np.max(y_flat) - np.min(y_flat)) / 100
        mask = np.abs(y_flat - y_profile) < tolerance
        
        x_profile = x_flat[mask]
        v_profile = v_flat[mask]
        
        # Sort by x
        sort_idx = np.argsort(x_profile)
        x_profile = x_profile[sort_idx]
        v_profile = v_profile[sort_idx]
        
        return x_profile, v_profile
        
    else:
        raise ValueError(f"Unknown profile type: {profile_type}")


def compute_wall_distance(
    y: np.ndarray,
    wall_location: float = 0.0,
    half_height: float = 1.0
) -> np.ndarray:
    """
    Compute wall distance for channel flow geometry.
    
    Parameters
    ----------
    y : np.ndarray
        Wall-normal coordinate
    wall_location : float
        y-coordinate of the wall
    half_height : float
        Channel half-height (for normalization)
        
    Returns
    -------
    np.ndarray
        Wall distance (y+) coordinate
    """
    y_wall = np.abs(y - wall_location)
    
    # Return normalized wall distance
    return y_wall / half_height


def average_over_homogeneous_directions(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    values: np.ndarray,
    directions: str = 'xz',
    n_bins_y: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Average field over homogeneous directions.
    
    For channel/pipe flows, average over streamwise (x) and spanwise (z)
    to get wall-normal profiles.
    
    Parameters
    ----------
    x, y, z : np.ndarray
        Coordinate arrays
    values : np.ndarray
        Field values
    directions : str
        Directions to average over: 'x', 'z', 'xz'
    n_bins_y : int
        Number of bins for wall-normal coordinate
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (y_bins, averaged_values)
    """
    x_flat = x.flatten()
    y_flat = y.flatten()
    v_flat = values.flatten()
    
    # Create y bins
    y_min, y_max = np.min(y_flat), np.max(y_flat)
    y_edges = np.linspace(y_min, y_max, n_bins_y + 1)
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    
    # Bin-average
    averaged_values = np.zeros(n_bins_y)
    counts = np.zeros(n_bins_y)
    
    for i in range(n_bins_y):
        mask = (y_flat >= y_edges[i]) & (y_flat < y_edges[i+1])
        if np.any(mask):
            averaged_values[i] = np.mean(v_flat[mask])
            counts[i] = np.sum(mask)
    
    return y_centers, averaged_values


def restructure_to_elements(
    data: np.ndarray,
    nelements: int,
    nx: int,
    ny: int,
    nz: int = 1
) -> np.ndarray:
    """
    Restructure flat array to element-based array.
    
    Parameters
    ----------
    data : np.ndarray
        Flat data array
    nelements : int
        Number of spectral elements
    nx, ny, nz : int
        Polynomial order in each direction
        
    Returns
    -------
    np.ndarray
        Reshaped array with shape (nelements, nz, ny, nx)
    """
    expected_size = nelements * nx * ny * nz
    
    if len(data.flatten()) != expected_size:
        warnings.warn(
            f"Data size {len(data.flatten())} doesn't match "
            f"expected {expected_size} = {nelements} x {nx} x {ny} x {nz}"
        )
        return data
    
    return data.reshape((nelements, nz, ny, nx))


def compute_element_centroids(
    x: np.ndarray,
    y: np.ndarray,
    z: Optional[np.ndarray],
    nelements: int,
    npoints_per_element: int
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Compute element centroids from coordinate arrays.
    
    Parameters
    ----------
    x, y, z : np.ndarray
        Coordinate arrays
    nelements : int
        Number of elements
    npoints_per_element : int
        Points per element
        
    Returns
    -------
    Tuple
        (x_centroids, y_centroids, z_centroids)
    """
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    x_centroids = np.zeros(nelements)
    y_centroids = np.zeros(nelements)
    z_centroids = np.zeros(nelements) if z is not None else None
    
    for i in range(nelements):
        start = i * npoints_per_element
        end = start + npoints_per_element
        
        x_centroids[i] = np.mean(x_flat[start:end])
        y_centroids[i] = np.mean(y_flat[start:end])
        
        if z is not None:
            z_flat = z.flatten()
            z_centroids[i] = np.mean(z_flat[start:end])
    
    return x_centroids, y_centroids, z_centroids


def create_spectral_element_grid(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    nelx: int,
    nely: int,
    polynomial_order: int = 7
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create spectral element grid with GLL points.
    
    Parameters
    ----------
    x_range, y_range : Tuple[float, float]
        Domain extent
    nelx, nely : int
        Number of elements in each direction
    polynomial_order : int
        Polynomial order (number of GLL points = order + 1)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (x, y) coordinate arrays
    """
    # GLL points on reference element [-1, 1]
    n = polynomial_order + 1
    
    # Chebyshev-Gauss-Lobatto points as approximation
    theta = np.pi * np.arange(n) / (n - 1)
    gll = np.cos(theta)[::-1]  # [-1, 1]
    
    # Element boundaries
    x_elem = np.linspace(x_range[0], x_range[1], nelx + 1)
    y_elem = np.linspace(y_range[0], y_range[1], nely + 1)
    
    # Build full grid
    x_list = []
    y_list = []
    
    for ey in range(nely):
        for ex in range(nelx):
            # Map GLL points to this element
            x_local = 0.5 * (x_elem[ex] + x_elem[ex+1]) + 0.5 * (x_elem[ex+1] - x_elem[ex]) * gll
            y_local = 0.5 * (y_elem[ey] + y_elem[ey+1]) + 0.5 * (y_elem[ey+1] - y_elem[ey]) * gll
            
            # Create tensor product grid
            X, Y = np.meshgrid(x_local, y_local)
            x_list.append(X.flatten())
            y_list.append(Y.flatten())
    
    x = np.concatenate(x_list)
    y = np.concatenate(y_list)
    
    return x, y
