#!/usr/bin/env python3
"""
Demonstration of Nek5000 Post-Processor
========================================

This example uses synthetic data to demonstrate the post-processor capabilities.
Run this to verify the installation and see example outputs.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path for proper imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now import modules
from readers.field_reader import FieldData
from statistics.first_order import (
    FirstOrderStatistics,
    compute_mean_fields,
    compute_rms,
    OnlineMeanComputer,
)
from statistics.second_order import (
    SecondOrderStatistics,
    compute_reynolds_stresses,
    compute_tke,
    OnlineSecondOrderComputer,
)
from temporal.temporal_averaging import (
    TemporalAverager,
    running_average,
    windowed_statistics,
)
from temporal.temporal_comparison import (
    compare_statistics,
    StepChangeAnalyzer,
    compute_time_evolution,
)


def create_synthetic_channel_field(
    time: float,
    nx: int = 32,
    ny: int = 32,
    Re_tau: float = 180.0,
    turbulence_intensity: float = 0.1
) -> FieldData:
    """
    Create a synthetic channel flow field for testing.
    
    Uses a mean profile approximating turbulent channel flow
    with added synthetic turbulent fluctuations.
    """
    # Create grid
    x = np.linspace(0, 4*np.pi, nx)
    y = np.linspace(-1, 1, ny)  # Wall-normal, normalized by half-height
    X, Y = np.meshgrid(x, y)
    
    x_flat = X.flatten()
    y_flat = Y.flatten()
    
    n_points = nx * ny
    
    # Mean velocity profile (approximate law of the wall)
    # Using a simple parabolic profile + log-layer correction
    y_wall = 1 - np.abs(y_flat)  # Distance from nearest wall
    u_mean = (1 - y_flat**2) * 15.0  # Parabolic base
    
    # Add turbulent fluctuations
    np.random.seed(int(time * 1000) % 2**31)
    
    # Turbulent fluctuations (simplified)
    u_fluct = turbulence_intensity * u_mean.max() * (
        np.sin(2*x_flat + time) * np.exp(-y_wall*2) +
        0.5 * np.random.randn(n_points) * (1 - y_flat**2)
    )
    
    v_fluct = 0.3 * turbulence_intensity * u_mean.max() * (
        np.cos(3*x_flat + time) * np.exp(-y_wall*2) +
        0.3 * np.random.randn(n_points) * (1 - y_flat**2)
    )
    
    w_fluct = 0.5 * turbulence_intensity * u_mean.max() * (
        np.sin(4*x_flat - time) * np.exp(-y_wall*2) +
        0.4 * np.random.randn(n_points) * (1 - y_flat**2)
    )
    
    # Combine mean and fluctuations
    u = u_mean + u_fluct
    v = v_fluct  # Mean v = 0 for channel flow
    w = w_fluct  # Mean w = 0
    
    # Pressure (with some fluctuations)
    p = -0.5 * (u_fluct**2 + v_fluct**2 + w_fluct**2) + 0.01 * np.random.randn(n_points)
    
    return FieldData(
        time=time,
        timestep=int(time * 100),
        nelements=nx * ny // 64,  # Approximate
        nx=8, ny=8, nz=1,
        ndim=2,
        x=x_flat.astype(np.float32),
        y=y_flat.astype(np.float32),
        z=None,
        u=u.astype(np.float32),
        v=v.astype(np.float32),
        w=w.astype(np.float32),
        p=p.astype(np.float32),
        t=None,
    )


def create_step_change_field(
    time: float,
    step_time: float = 5.0,
    nx: int = 32,
    ny: int = 32,
) -> FieldData:
    """
    Create synthetic field with step change in turbulence intensity.
    
    Before step_time: low turbulence intensity
    After step_time: high turbulence intensity (simulating acceleration)
    """
    if time < step_time:
        # Pre-step: equilibrium state
        intensity = 0.05
    else:
        # Post-step: increased turbulence with transient
        t_rel = time - step_time
        # Exponential relaxation to new equilibrium
        tau = 2.0  # Relaxation time
        intensity_final = 0.15
        intensity_init = 0.05
        intensity = intensity_final - (intensity_final - intensity_init) * np.exp(-t_rel / tau)
        # Add overshoot
        if t_rel < 3.0:
            intensity += 0.02 * np.sin(np.pi * t_rel / 3.0)
    
    return create_synthetic_channel_field(time, nx, ny, turbulence_intensity=intensity)


def demo_basic_statistics():
    """Demonstrate basic first and second-order statistics."""
    print("=" * 60)
    print("Demo: Basic Turbulence Statistics")
    print("=" * 60)
    
    # Generate synthetic field sequence
    n_snapshots = 50
    times = np.linspace(0, 5, n_snapshots)
    fields = [create_synthetic_channel_field(t) for t in times]
    
    print(f"\nGenerated {n_snapshots} synthetic field snapshots")
    print(f"Grid size: {fields[0].u.shape}")
    print(f"Time range: [{times[0]:.2f}, {times[-1]:.2f}]")
    
    # Compute first-order statistics
    print("\n--- First-Order Statistics ---")
    mean_stats = compute_mean_fields(fields)
    
    print(f"Mean U: min={mean_stats.u_mean.min():.4f}, max={mean_stats.u_mean.max():.4f}")
    print(f"Mean V: min={mean_stats.v_mean.min():.4f}, max={mean_stats.v_mean.max():.4f}")
    
    # Compute RMS fluctuations
    rms = compute_rms(fields, mean_stats)
    print(f"u_rms: max={rms['u_rms'].max():.4f}")
    print(f"v_rms: max={rms['v_rms'].max():.4f}")
    
    # Compute second-order statistics
    print("\n--- Second-Order Statistics ---")
    second_order = compute_reynolds_stresses(fields, mean_stats)
    
    print(f"<u'u'>: max={second_order.uu.max():.6f}")
    print(f"<v'v'>: max={second_order.vv.max():.6f}")
    print(f"<u'v'>: min={second_order.uv.min():.6f}, max={second_order.uv.max():.6f}")
    print(f"TKE: mean={second_order.tke.mean():.6f}, max={second_order.tke.max():.6f}")
    
    return mean_stats, second_order


def demo_online_computation():
    """Demonstrate online (streaming) statistics computation."""
    print("\n" + "=" * 60)
    print("Demo: Online Statistics Computation")
    print("=" * 60)
    
    # Initialize online computers
    first_order_computer = OnlineMeanComputer()
    second_order_computer = OnlineSecondOrderComputer()
    
    # Process fields one at a time (memory efficient)
    n_snapshots = 100
    print(f"\nProcessing {n_snapshots} snapshots online...")
    
    for i in range(n_snapshots):
        field = create_synthetic_channel_field(i * 0.1)
        first_order_computer.update(field)
        second_order_computer.update(field)
        
        # Print progress every 25 snapshots
        if (i + 1) % 25 == 0:
            stats = second_order_computer.get_statistics()
            print(f"  After {i+1} samples: TKE_mean = {stats.tke.mean():.6f}")
    
    # Get final statistics
    mean_stats = first_order_computer.get_statistics()
    second_stats = second_order_computer.get_statistics()
    
    print(f"\nFinal statistics (n={second_stats.n_samples}):")
    print(f"  Mean U: [{mean_stats.u_mean.min():.4f}, {mean_stats.u_mean.max():.4f}]")
    print(f"  TKE: mean={second_stats.tke.mean():.6f}")


def demo_temporal_averaging():
    """Demonstrate temporal averaging methods."""
    print("\n" + "=" * 60)
    print("Demo: Temporal Averaging Methods")
    print("=" * 60)
    
    # Generate fields
    n_snapshots = 100
    times = np.linspace(0, 10, n_snapshots)
    fields = [create_synthetic_channel_field(t) for t in times]
    
    # Running (cumulative) average
    print("\n--- Running Average ---")
    averager_cumulative = TemporalAverager(window_type='cumulative')
    for field in fields:
        averager_cumulative.update(field)
    
    evolution = averager_cumulative.get_time_evolution()
    print(f"Time points: {len(evolution.times)}")
    print(f"TKE convergence: start={evolution.tke[0]:.6f}, end={evolution.tke[-1]:.6f}")
    
    # Exponential moving average
    print("\n--- Exponential Moving Average (alpha=0.1) ---")
    averager_ema = TemporalAverager(window_type='exponential', alpha=0.1)
    for field in fields:
        averager_ema.update(field)
    
    current_stats = averager_ema.get_current_statistics()
    print(f"Final mean U: max={current_stats['u_mean'].max():.4f}")
    
    # Windowed statistics
    print("\n--- Windowed Statistics (window=20, overlap=0.5) ---")
    windows = windowed_statistics(fields, window_size=20, overlap=0.5)
    print(f"Number of windows: {len(windows)}")
    for i, w in enumerate(windows[:3]):  # Print first 3
        print(f"  Window {i}: t=[{w['time_start']:.2f}, {w['time_end']:.2f}], "
              f"TKE_mean={w['tke'].mean():.6f}")


def demo_step_change_analysis():
    """Demonstrate step change analysis capabilities."""
    print("\n" + "=" * 60)
    print("Demo: Step Change Analysis")
    print("=" * 60)
    
    # Generate step change data
    step_time = 5.0
    n_snapshots = 100
    times = np.linspace(0, 15, n_snapshots)
    fields = [create_step_change_field(t, step_time=step_time) for t in times]
    
    print(f"\nGenerated step change data:")
    print(f"  Step time: {step_time}")
    print(f"  Total snapshots: {n_snapshots}")
    
    # Initialize analyzer
    analyzer = StepChangeAnalyzer(step_time=step_time)
    for field in fields:
        analyzer.add_field(field)
    
    print(f"  Pre-step fields: {len(analyzer.fields_before)}")
    print(f"  Post-step fields: {len(analyzer.fields_after)}")
    
    # Compare before/after
    print("\n--- Before/After Comparison ---")
    comparison = analyzer.compare_before_after()
    
    print(f"Period 1 ({comparison.period_1_name}): n={comparison.n_samples_1}")
    print(f"Period 2 ({comparison.period_2_name}): n={comparison.n_samples_2}")
    
    if comparison.tke_1 is not None and comparison.tke_2 is not None:
        tke_before = comparison.tke_1.mean()
        tke_after = comparison.tke_2.mean()
        tke_ratio = tke_after / tke_before if tke_before > 0 else np.nan
        print(f"TKE before: {tke_before:.6f}")
        print(f"TKE after: {tke_after:.6f}")
        print(f"TKE ratio: {tke_ratio:.3f}")
    
    # Analyze transient
    print("\n--- Transient Analysis ---")
    try:
        transient = analyzer.analyze_transient()
        print(f"Transient duration: {transient['transient_duration']:.4f}")
        print(f"Relaxation time: {transient['relaxation_time']:.4f}")
        print(f"Overshoot: {transient['overshoot']*100:.1f}%")
    except ValueError as e:
        print(f"Could not analyze transient: {e}")
    
    # Get summary
    print("\n--- Summary ---")
    summary = analyzer.get_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6g}")
        else:
            print(f"  {key}: {value}")
    
    return analyzer


def demo_field_data_operations():
    """Demonstrate FieldData operations."""
    print("\n" + "=" * 60)
    print("Demo: FieldData Operations")
    print("=" * 60)
    
    field = create_synthetic_channel_field(time=1.0)
    
    print(f"\nFieldData properties:")
    print(f"  Time: {field.time}")
    print(f"  Timestep: {field.timestep}")
    print(f"  Dimensions: {field.ndim}D")
    print(f"  Total points: {field.total_points}")
    
    # Velocity magnitude
    vmag = field.get_velocity_magnitude()
    print(f"\nVelocity magnitude: min={vmag.min():.4f}, max={vmag.max():.4f}")
    
    # Get specific fields
    print("\nAccessible fields:")
    for name in ['u', 'v', 'w', 'p', 'velocity_magnitude']:
        data = field.get_field(name)
        if data is not None:
            print(f"  {name}: shape={data.shape}, dtype={data.dtype}")


def main():
    """Run all demonstrations."""
    print("\n" + "#" * 60)
    print("#" + " " * 18 + "NEK5000 POST-PROCESSOR DEMO" + " " * 13 + "#")
    print("#" * 60 + "\n")
    
    # Run demos
    demo_field_data_operations()
    demo_basic_statistics()
    demo_online_computation()
    demo_temporal_averaging()
    demo_step_change_analysis()
    
    print("\n" + "#" * 60)
    print("#" + " " * 22 + "DEMO COMPLETE" + " " * 23 + "#")
    print("#" * 60 + "\n")


if __name__ == '__main__':
    main()
