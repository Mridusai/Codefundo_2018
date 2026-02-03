"""
Plotting Functions
==================

Visualization utilities for Nek5000 DNS post-processing results.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available. Plotting functions will not work.")


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")


# Default style settings
DEFAULT_FIGSIZE = (10, 8)
DEFAULT_DPI = 150
DEFAULT_CMAP = 'viridis'
TURBULENCE_CMAP = 'RdBu_r'
DIVERGING_CMAP = 'coolwarm'


def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    _check_matplotlib()
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.figsize': DEFAULT_FIGSIZE,
        'figure.dpi': DEFAULT_DPI,
        'lines.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def plot_field(
    field_data,
    field_name: str = 'u',
    ax: Optional['Axes'] = None,
    cmap: str = DEFAULT_CMAP,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    colorbar: bool = True,
    **kwargs
) -> Tuple['Figure', 'Axes']:
    """
    Plot a 2D field (e.g., velocity, pressure) as a filled contour.
    
    Parameters
    ----------
    field_data : FieldData or np.ndarray
        Field data to plot (uses x, y coordinates if FieldData)
    field_name : str
        Name of field to plot: 'u', 'v', 'w', 'p', 't'
    ax : Axes, optional
        Matplotlib axes to plot on
    cmap : str
        Colormap name
    vmin, vmax : float
        Color limits
    title : str
        Plot title
    colorbar : bool
        Whether to add colorbar
        
    Returns
    -------
    Tuple[Figure, Axes]
        Figure and axes objects
    """
    _check_matplotlib()
    
    # Get data
    if hasattr(field_data, 'get_field'):
        data = field_data.get_field(field_name)
        x = field_data.x
        y = field_data.y
    elif hasattr(field_data, field_name):
        data = getattr(field_data, field_name)
        x = getattr(field_data, 'x', None)
        y = getattr(field_data, 'y', None)
    else:
        data = field_data
        x = None
        y = None
    
    if data is None:
        raise ValueError(f"Field '{field_name}' not available")
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    else:
        fig = ax.get_figure()
    
    # Plot
    if x is not None and y is not None:
        # Scatter plot for unstructured data
        scatter = ax.scatter(x, y, c=data, cmap=cmap, vmin=vmin, vmax=vmax, 
                            s=1, **kwargs)
        if colorbar:
            plt.colorbar(scatter, ax=ax, label=field_name)
    else:
        # Assume 1D array - create index-based plot
        if data.ndim == 1:
            ax.plot(data, **kwargs)
            ax.set_xlabel('Point index')
            ax.set_ylabel(field_name)
        else:
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, 
                          aspect='auto', origin='lower', **kwargs)
            if colorbar:
                plt.colorbar(im, ax=ax, label=field_name)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'{field_name} field')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    return fig, ax


def plot_contour(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    ax: Optional['Axes'] = None,
    levels: int = 20,
    cmap: str = DEFAULT_CMAP,
    filled: bool = True,
    colorbar: bool = True,
    title: Optional[str] = None,
    **kwargs
) -> Tuple['Figure', 'Axes']:
    """
    Create contour plot from structured or scattered data.
    
    Parameters
    ----------
    x, y : np.ndarray
        Coordinate arrays
    z : np.ndarray
        Field values
    ax : Axes, optional
        Matplotlib axes
    levels : int
        Number of contour levels
    cmap : str
        Colormap
    filled : bool
        Whether to use filled contours
    colorbar : bool
        Whether to add colorbar
    title : str
        Plot title
        
    Returns
    -------
    Tuple[Figure, Axes]
    """
    _check_matplotlib()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    else:
        fig = ax.get_figure()
    
    # Try tricontour for scattered data
    try:
        from matplotlib.tri import Triangulation
        tri = Triangulation(x.flatten(), y.flatten())
        
        if filled:
            contour = ax.tricontourf(tri, z.flatten(), levels=levels, cmap=cmap, **kwargs)
        else:
            contour = ax.tricontour(tri, z.flatten(), levels=levels, cmap=cmap, **kwargs)
        
        if colorbar:
            plt.colorbar(contour, ax=ax)
            
    except Exception:
        # Fallback to scatter
        scatter = ax.scatter(x.flatten(), y.flatten(), c=z.flatten(), 
                            cmap=cmap, s=1, **kwargs)
        if colorbar:
            plt.colorbar(scatter, ax=ax)
    
    if title:
        ax.set_title(title)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal', 'box')
    
    return fig, ax


def plot_profile(
    y: np.ndarray,
    values: np.ndarray,
    ax: Optional['Axes'] = None,
    label: Optional[str] = None,
    xlabel: str = 'Value',
    ylabel: str = 'y',
    title: Optional[str] = None,
    normalize: bool = False,
    **kwargs
) -> Tuple['Figure', 'Axes']:
    """
    Plot 1D profile (e.g., velocity profile vs wall-normal distance).
    
    Parameters
    ----------
    y : np.ndarray
        Wall-normal coordinate
    values : np.ndarray
        Profile values
    ax : Axes, optional
        Matplotlib axes
    label : str
        Line label for legend
    xlabel, ylabel : str
        Axis labels
    title : str
        Plot title
    normalize : bool
        Whether to normalize values by maximum
        
    Returns
    -------
    Tuple[Figure, Axes]
    """
    _check_matplotlib()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()
    
    plot_values = values.copy()
    if normalize and np.max(np.abs(values)) > 0:
        plot_values = values / np.max(np.abs(values))
    
    ax.plot(plot_values, y, label=label, **kwargs)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if title:
        ax.set_title(title)
    
    if label:
        ax.legend()
    
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_statistics(
    stats,
    stat_names: List[str] = None,
    y_coordinate: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (12, 8),
    **kwargs
) -> 'Figure':
    """
    Plot multiple turbulence statistics profiles.
    
    Parameters
    ----------
    stats : FirstOrderStatistics or SecondOrderStatistics
        Statistics object
    stat_names : List[str]
        Names of statistics to plot
    y_coordinate : np.ndarray
        Wall-normal coordinate for profiles
    figsize : tuple
        Figure size
        
    Returns
    -------
    Figure
        Matplotlib figure
    """
    _check_matplotlib()
    
    if stat_names is None:
        # Default statistics to plot
        if hasattr(stats, 'uu'):
            stat_names = ['uu', 'vv', 'ww', 'uv', 'tke']
        else:
            stat_names = ['u_mean', 'v_mean', 'w_mean']
    
    n_plots = len(stat_names)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, stat_name in enumerate(stat_names):
        ax = axes[i]
        
        data = stats.get_field(stat_name) if hasattr(stats, 'get_field') else getattr(stats, stat_name, None)
        
        if data is None:
            ax.text(0.5, 0.5, f'{stat_name} not available', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        if y_coordinate is not None and len(y_coordinate) == len(data.flatten()):
            ax.plot(data.flatten(), y_coordinate.flatten(), **kwargs)
            ax.set_ylabel('y')
        else:
            ax.plot(data.flatten(), **kwargs)
            ax.set_ylabel(stat_name)
        
        ax.set_xlabel(stat_name)
        ax.set_title(f'{stat_name}')
        ax.grid(True, alpha=0.3)
    
    # Hide unused axes
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_temporal_comparison(
    comparison,
    field_names: List[str] = None,
    figsize: Tuple[int, int] = (14, 10),
    y_coordinate: Optional[np.ndarray] = None
) -> 'Figure':
    """
    Plot comparison of statistics between two time periods.
    
    Parameters
    ----------
    comparison : TemporalComparison
        Comparison results
    field_names : List[str]
        Fields to compare
    figsize : tuple
        Figure size
    y_coordinate : np.ndarray
        Wall-normal coordinate
        
    Returns
    -------
    Figure
    """
    _check_matplotlib()
    
    if field_names is None:
        field_names = ['u_mean', 'tke', 'uu', 'uv']
    
    n_fields = len(field_names)
    
    fig, axes = plt.subplots(n_fields, 3, figsize=figsize)
    
    for i, field_name in enumerate(field_names):
        # Get data for both periods
        data_1 = getattr(comparison, f'{field_name}_1', None)
        data_2 = getattr(comparison, f'{field_name}_2', None)
        
        if data_1 is None or data_2 is None:
            for j in range(3):
                axes[i, j].text(0.5, 0.5, f'{field_name} not available',
                               ha='center', va='center', transform=axes[i, j].transAxes)
            continue
        
        # Flatten for plotting
        d1 = data_1.flatten()
        d2 = data_2.flatten()
        
        if y_coordinate is not None:
            y = y_coordinate.flatten()
        else:
            y = np.arange(len(d1))
        
        # Plot period 1
        axes[i, 0].plot(d1, y, 'b-', label=comparison.period_1_name)
        axes[i, 0].set_title(f'{field_name} - {comparison.period_1_name}')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Plot period 2
        axes[i, 1].plot(d2, y, 'r-', label=comparison.period_2_name)
        axes[i, 1].set_title(f'{field_name} - {comparison.period_2_name}')
        axes[i, 1].grid(True, alpha=0.3)
        
        # Plot comparison (both on same axes)
        axes[i, 2].plot(d1, y, 'b-', label=comparison.period_1_name, alpha=0.7)
        axes[i, 2].plot(d2, y, 'r--', label=comparison.period_2_name, alpha=0.7)
        axes[i, 2].set_title(f'{field_name} - Comparison')
        axes[i, 2].legend(loc='best')
        axes[i, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_time_evolution(
    time_evolution,
    ax: Optional['Axes'] = None,
    normalize: bool = False,
    label: Optional[str] = None,
    title: Optional[str] = None,
    markers: List[float] = None,
    **kwargs
) -> Tuple['Figure', 'Axes']:
    """
    Plot time evolution of a statistical quantity.
    
    Parameters
    ----------
    time_evolution : TimeEvolution
        Time evolution data
    ax : Axes, optional
        Matplotlib axes
    normalize : bool
        Normalize by initial value
    label : str
        Line label
    title : str
        Plot title
    markers : List[float]
        Time markers to add (e.g., step change time)
        
    Returns
    -------
    Tuple[Figure, Axes]
    """
    _check_matplotlib()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()
    
    times = time_evolution.times
    values = time_evolution.values.copy()
    
    if normalize and len(values) > 0 and np.abs(values[0]) > 1e-10:
        values = values / values[0]
        ylabel = f'{time_evolution.field_name} (normalized)'
    else:
        ylabel = time_evolution.field_name
    
    line_label = label if label else time_evolution.field_name
    ax.plot(times, values, label=line_label, **kwargs)
    
    # Add time markers
    if markers:
        for t_marker in markers:
            ax.axvline(x=t_marker, color='k', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Time')
    ax.set_ylabel(ylabel)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Time evolution of {time_evolution.field_name}')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_reynolds_stresses(
    stats,
    y_coordinate: Optional[np.ndarray] = None,
    normalize_by_tke: bool = False,
    figsize: Tuple[int, int] = (12, 8)
) -> 'Figure':
    """
    Plot Reynolds stress components and TKE.
    
    Parameters
    ----------
    stats : SecondOrderStatistics
        Second-order statistics
    y_coordinate : np.ndarray
        Wall-normal coordinate
    normalize_by_tke : bool
        Normalize Reynolds stresses by 2*TKE (anisotropy)
    figsize : tuple
        Figure size
        
    Returns
    -------
    Figure
    """
    _check_matplotlib()
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    # Get data
    components = ['uu', 'vv', 'ww', 'uv', 'uw', 'vw']
    titles = ['$\\langle u\'u\' \\rangle$', '$\\langle v\'v\' \\rangle$',
              '$\\langle w\'w\' \\rangle$', '$\\langle u\'v\' \\rangle$',
              '$\\langle u\'w\' \\rangle$', '$\\langle v\'w\' \\rangle$']
    
    for i, (comp, title) in enumerate(zip(components, titles)):
        ax = axes[i]
        data = getattr(stats, comp, None)
        
        if data is None:
            ax.text(0.5, 0.5, f'{comp} not available',
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        d = data.flatten()
        
        if normalize_by_tke and stats.tke is not None:
            tke = stats.tke.flatten()
            tke = np.where(tke > 1e-10, tke, 1e-10)
            d = d / (2 * tke)
            title = f'{title} / 2k'
        
        if y_coordinate is not None:
            y = y_coordinate.flatten()
            ax.plot(d, y)
            ax.set_ylabel('y')
        else:
            ax.plot(d)
        
        ax.set_xlabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_step_change_figure(
    comparison,
    transient_analysis: Dict,
    step_time: float,
    figsize: Tuple[int, int] = (14, 12)
) -> 'Figure':
    """
    Create comprehensive figure for step change analysis.
    
    Parameters
    ----------
    comparison : TemporalComparison
        Before/after comparison
    transient_analysis : Dict
        Results from transient analysis
    step_time : float
        Time of step change
    figsize : tuple
        Figure size
        
    Returns
    -------
    Figure
    """
    _check_matplotlib()
    
    fig = plt.figure(figsize=figsize)
    
    # Layout: 2x3 grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Time evolution (top left, spans 2 columns)
    ax_evolution = fig.add_subplot(gs[0, :2])
    
    if 'time_evolution' in transient_analysis:
        evolution = transient_analysis['time_evolution']
        ax_evolution.plot(evolution.times, evolution.values, 'b-', linewidth=2)
        ax_evolution.axvline(x=step_time, color='r', linestyle='--', 
                            label='Step change', linewidth=2)
        
        # Mark transient region
        if 'transient_end' in transient_analysis:
            ax_evolution.axvspan(step_time, transient_analysis['transient_end'],
                               alpha=0.2, color='orange', label='Transient')
    
    ax_evolution.set_xlabel('Time')
    ax_evolution.set_ylabel('TKE')
    ax_evolution.set_title('Turbulent Kinetic Energy Evolution')
    ax_evolution.legend()
    ax_evolution.grid(True, alpha=0.3)
    
    # Summary statistics (top right)
    ax_summary = fig.add_subplot(gs[0, 2])
    ax_summary.axis('off')
    
    summary_text = [
        f"Step time: {step_time:.2f}",
        f"Samples (before): {comparison.n_samples_1}",
        f"Samples (after): {comparison.n_samples_2}",
    ]
    
    if 'transient_duration' in transient_analysis:
        summary_text.append(f"Transient duration: {transient_analysis['transient_duration']:.2f}")
    if 'relaxation_time' in transient_analysis:
        summary_text.append(f"Relaxation time: {transient_analysis['relaxation_time']:.2f}")
    if 'overshoot' in transient_analysis:
        summary_text.append(f"Overshoot: {transient_analysis['overshoot']*100:.1f}%")
    
    ax_summary.text(0.1, 0.9, '\n'.join(summary_text),
                   transform=ax_summary.transAxes, fontsize=12,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax_summary.set_title('Summary Statistics')
    
    # Mean velocity comparison (bottom left)
    ax_u = fig.add_subplot(gs[1, 0])
    if comparison.u_mean_1 is not None and comparison.u_mean_2 is not None:
        u1 = comparison.u_mean_1.flatten()
        u2 = comparison.u_mean_2.flatten()
        idx = np.arange(len(u1))
        
        ax_u.plot(u1, idx, 'b-', label='Before', alpha=0.7)
        ax_u.plot(u2, idx, 'r--', label='After', alpha=0.7)
    ax_u.set_xlabel('U mean')
    ax_u.set_ylabel('Index')
    ax_u.set_title('Mean Velocity')
    ax_u.legend()
    ax_u.grid(True, alpha=0.3)
    
    # TKE comparison (bottom middle)
    ax_tke = fig.add_subplot(gs[1, 1])
    if comparison.tke_1 is not None and comparison.tke_2 is not None:
        tke1 = comparison.tke_1.flatten()
        tke2 = comparison.tke_2.flatten()
        idx = np.arange(len(tke1))
        
        ax_tke.plot(tke1, idx, 'b-', label='Before', alpha=0.7)
        ax_tke.plot(tke2, idx, 'r--', label='After', alpha=0.7)
    ax_tke.set_xlabel('TKE')
    ax_tke.set_ylabel('Index')
    ax_tke.set_title('Turbulent Kinetic Energy')
    ax_tke.legend()
    ax_tke.grid(True, alpha=0.3)
    
    # Reynolds shear stress comparison (bottom right)
    ax_uv = fig.add_subplot(gs[1, 2])
    if comparison.uv_1 is not None and comparison.uv_2 is not None:
        uv1 = comparison.uv_1.flatten()
        uv2 = comparison.uv_2.flatten()
        idx = np.arange(len(uv1))
        
        ax_uv.plot(uv1, idx, 'b-', label='Before', alpha=0.7)
        ax_uv.plot(uv2, idx, 'r--', label='After', alpha=0.7)
    ax_uv.set_xlabel('$\\langle u\'v\' \\rangle$')
    ax_uv.set_ylabel('Index')
    ax_uv.set_title('Reynolds Shear Stress')
    ax_uv.legend()
    ax_uv.grid(True, alpha=0.3)
    
    fig.suptitle('Step Change DNS Analysis', fontsize=14, fontweight='bold')
    
    return fig


def save_figure(
    fig: 'Figure',
    filename: str,
    dpi: int = DEFAULT_DPI,
    formats: List[str] = None,
    **kwargs
) -> None:
    """
    Save figure to file(s).
    
    Parameters
    ----------
    fig : Figure
        Matplotlib figure
    filename : str
        Base filename (without extension)
    dpi : int
        Resolution
    formats : List[str]
        File formats (default: ['png', 'pdf'])
    """
    _check_matplotlib()
    
    if formats is None:
        formats = ['png']
    
    for fmt in formats:
        full_filename = f"{filename}.{fmt}"
        fig.savefig(full_filename, dpi=dpi, bbox_inches='tight', **kwargs)
        print(f"Saved: {full_filename}")


def create_animation_frames(
    fields: List,
    field_name: str = 'u',
    output_dir: str = './frames',
    **plot_kwargs
) -> List[str]:
    """
    Create animation frames from field sequence.
    
    Parameters
    ----------
    fields : List[FieldData]
        List of field snapshots
    field_name : str
        Field to animate
    output_dir : str
        Directory for frame images
    **plot_kwargs
        Additional arguments for plot_field
        
    Returns
    -------
    List[str]
        List of frame filenames
    """
    _check_matplotlib()
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    filenames = []
    
    # Get global color limits
    all_data = np.concatenate([f.get_field(field_name).flatten() for f in fields])
    vmin = np.percentile(all_data, 2)
    vmax = np.percentile(all_data, 98)
    
    for i, field in enumerate(fields):
        fig, ax = plot_field(field, field_name, vmin=vmin, vmax=vmax, **plot_kwargs)
        ax.set_title(f'{field_name} at t = {field.time:.3f}')
        
        filename = os.path.join(output_dir, f'frame_{i:05d}.png')
        fig.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        filenames.append(filename)
    
    return filenames
