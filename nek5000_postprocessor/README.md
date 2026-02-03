# Nek5000 DNS Post-Processor

A comprehensive Python-based post-processing toolkit for Nek5000 Direct Numerical Simulation (DNS) of step change flows.

## Features

### Core Capabilities

- **Field File Reader**: Read Nek5000 binary field files (.fld, f00001 format)
- **First-Order Statistics**: Mean velocity, pressure, and temperature fields
- **Second-Order Statistics**: Reynolds stresses, turbulent kinetic energy (TKE), turbulent dissipation
- **Temporal Analysis**: Running averages, windowed statistics, step change comparisons
- **Visualization**: Contour plots, profile plots, temporal evolution plots

### Step Change Analysis

Specialized tools for analyzing DNS of step change flows:
- Before/after comparison of turbulence statistics
- Transient detection and duration estimation
- Relaxation time analysis
- TKE amplification factor computation

## Installation

### Requirements

- Python 3.8+
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Matplotlib >= 3.4.0
- h5py >= 3.3.0 (optional, for HDF5 export)
- pandas >= 1.3.0 (optional, for data analysis)
- PyYAML >= 5.4.0 (for configuration files)
- tqdm >= 4.62.0 (for progress bars)
- click >= 8.0.0 (optional, for CLI)
- numba >= 0.54.0 (optional, for acceleration)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Command Line Usage

```bash
# Basic usage
python postprocess.py --case channel --data-dir ./output --start 1 --end 1000

# With step change analysis
python postprocess.py --case dns --data-dir ./data --start 100 --end 500 --step-change 200

# Generate default config file
python postprocess.py --generate-config

# Use configuration file
python postprocess.py --config config.yaml
```

### Python API Usage

```python
from nek5000_postprocessor import (
    Nek5000Reader,
    compute_mean_fields,
    compute_reynolds_stresses,
    TemporalComparison,
)

# Read field files
reader = Nek5000Reader('channel', './data/')
fields = reader.read_field_sequence(1, 100)

# Compute statistics
mean_stats = compute_mean_fields(fields)
second_order = compute_reynolds_stresses(fields, mean_stats)

# Print TKE
print(f"Maximum TKE: {second_order.tke.max()}")
```

## Module Overview

### readers

Read Nek5000 binary output files.

```python
from nek5000_postprocessor.readers import Nek5000Reader, FieldData

# Initialize reader
reader = Nek5000Reader(
    case_name='channel',
    data_dir='./output/',
    precision='single'  # or 'double'
)

# Read single field
field = reader.read_field(timestep=100)

# Read sequence
fields = reader.read_field_sequence(
    start_timestep=1,
    end_timestep=1000,
    step=10
)

# Access field data
print(f"Time: {field.time}")
print(f"Velocity shape: {field.u.shape}")
print(f"Coordinates: x={field.x}, y={field.y}")
```

### statistics

Compute first-order and second-order turbulence statistics.

```python
from nek5000_postprocessor.statistics import (
    FirstOrderStatistics,
    SecondOrderStatistics,
    compute_mean_fields,
    compute_reynolds_stresses,
    compute_tke,
)
from nek5000_postprocessor.statistics.first_order import (
    OnlineMeanComputer,
    compute_rms,
    compute_convergence_statistics,
)
from nek5000_postprocessor.statistics.second_order import (
    OnlineSecondOrderComputer,
    compute_turbulent_heat_flux,
    compute_higher_order_moments,
)

# Compute mean fields
mean_stats = compute_mean_fields(fields)
print(f"Mean U range: [{mean_stats.u_mean.min()}, {mean_stats.u_mean.max()}]")

# Compute Reynolds stresses
reynolds = compute_reynolds_stresses(fields, mean_stats)
print(f"Reynolds stress <uu>: {reynolds.uu}")
print(f"TKE: {reynolds.tke}")

# Online computation (memory efficient)
computer = OnlineSecondOrderComputer()
for field in fields:
    computer.update(field)
stats = computer.get_statistics()

# Higher-order moments
moments = compute_higher_order_moments(fields)
print(f"Velocity skewness: {moments['u_skewness']}")
print(f"Velocity flatness: {moments['u_flatness']}")

# RMS fluctuations
rms = compute_rms(fields, mean_stats)
print(f"u_rms: {rms['u_rms'].max()}")
```

### temporal

Temporal averaging and comparison tools for step change analysis.

```python
from nek5000_postprocessor.temporal import (
    TemporalAverager,
    TemporalComparison,
    running_average,
    compare_statistics,
    StepChangeAnalyzer,
    compute_time_evolution,
    detect_transient_duration,
)

# Running average
evolution = running_average(fields)
print(f"TKE evolution: {evolution.tke}")

# Step change analysis
analyzer = StepChangeAnalyzer(step_time=10.0)
for field in fields:
    analyzer.add_field(field)

# Compare before/after
comparison = analyzer.compare_before_after()
print(f"TKE ratio (after/before): {comparison.tke_ratio.mean()}")

# Analyze transient
transient = analyzer.analyze_transient()
print(f"Transient duration: {transient['transient_duration']}")
print(f"Relaxation time: {transient['relaxation_time']}")

# Windowed statistics for non-stationary flows
windows = windowed_statistics(
    fields,
    window_size=50,
    overlap=0.5
)
for w in windows:
    print(f"Window [{w['time_start']:.2f}, {w['time_end']:.2f}]: TKE={w['tke'].mean()}")
```

### visualization

Plotting tools for DNS results.

```python
from nek5000_postprocessor.visualization import (
    plot_field,
    plot_statistics,
    plot_temporal_comparison,
    plot_time_evolution,
    plot_reynolds_stresses,
    create_step_change_figure,
    save_figure,
)

# Plot velocity field
fig, ax = plot_field(field, 'u', cmap='RdBu_r')
save_figure(fig, 'velocity_field', formats=['png', 'pdf'])

# Plot statistics profiles
fig = plot_statistics(reynolds, stat_names=['uu', 'vv', 'uv', 'tke'])

# Plot temporal comparison
fig = plot_temporal_comparison(comparison)

# Plot time evolution
fig, ax = plot_time_evolution(
    evolution,
    markers=[10.0],  # Mark step change time
    normalize=True
)

# Comprehensive step change figure
fig = create_step_change_figure(
    comparison,
    transient_analysis,
    step_time=10.0
)
```

### utils

Utility functions for I/O, grid operations, and physics calculations.

```python
from nek5000_postprocessor.utils import (
    save_statistics,
    load_statistics,
    export_to_vtk,
    export_to_csv,
    interpolate_to_regular_grid,
    extract_profile,
    compute_friction_velocity,
    compute_wall_units,
    compute_reynolds_number,
)

# Save/load statistics
save_statistics(reynolds, 'reynolds_stresses.npz')
data = load_statistics('reynolds_stresses.npz')

# Export to VTK for ParaView
export_to_vtk(field, 'field.vtk')

# Export to CSV
export_to_csv(reynolds, 'statistics.csv')

# Interpolate to regular grid
X, Y, U = interpolate_to_regular_grid(field.x, field.y, field.u)

# Extract wall-normal profile
y_prof, u_prof = extract_profile(field.x, field.y, field.u, 'vertical')

# Wall units scaling
u_tau = compute_friction_velocity(du_dy_wall, nu)
scaled = compute_wall_units(y, u, u_tau=u_tau, nu=nu)
print(f"y+ range: {scaled['y_plus'].min()} to {scaled['y_plus'].max()}")

# Reynolds number
Re = compute_reynolds_number(U_bulk, half_height, nu)
Re_tau = compute_friction_reynolds_number(u_tau, half_height, nu)
```

## Configuration File

Create a YAML configuration file for reproducible post-processing:

```yaml
# postprocessor_config.yaml
case_name: channel
data_dir: ./output
output_dir: ./postprocessing_results

timesteps:
  start: 1
  end: 1000
  step: 1

precision: single

statistics:
  compute_first_order: true
  compute_second_order: true
  compute_higher_order: false

temporal:
  enable: true
  step_change_time: 50.0  # Set to null if no step change
  window_type: cumulative  # cumulative, sliding, or exponential

output:
  save_statistics: true
  export_vtk: false
  export_csv: true
  formats:
    - npz
    - hdf5

physics:
  nu: 1.0e-5
  half_height: 1.0
```

## Second-Order Statistics Details

The post-processor computes the following second-order statistics:

### Reynolds Stresses

| Symbol | Description | Formula |
|--------|-------------|---------|
| `uu` | Streamwise normal stress | `<u'u'>` |
| `vv` | Wall-normal normal stress | `<v'v'>` |
| `ww` | Spanwise normal stress | `<w'w'>` |
| `uv` | Reynolds shear stress | `<u'v'>` |
| `uw` | Reynolds shear stress | `<u'w'>` |
| `vw` | Reynolds shear stress | `<v'w'>` |

### Derived Quantities

| Symbol | Description | Formula |
|--------|-------------|---------|
| `tke` | Turbulent Kinetic Energy | `0.5 * (uu + vv + ww)` |
| `pp` | Pressure variance | `<p'p'>` |
| `ut`, `vt`, `wt` | Turbulent heat flux | `<u'T'>`, `<v'T'>`, `<w'T'>` |
| `tt` | Temperature variance | `<T'T'>` |

### Higher-Order Moments

| Symbol | Description | Formula |
|--------|-------------|---------|
| Skewness | Third moment | `<u'^3> / <u'^2>^{3/2}` |
| Flatness | Fourth moment | `<u'^4> / <u'^2>^2` |

## Step Change Analysis

For DNS simulations with step changes in flow conditions (e.g., Reynolds number, wall temperature):

```python
from nek5000_postprocessor.temporal import StepChangeAnalyzer

# Initialize analyzer
analyzer = StepChangeAnalyzer(
    step_time=100.0,              # Time of step change
    pre_step_window=50.0,         # Averaging window before step
    post_step_window=100.0        # Duration to consider as transient
)

# Add field snapshots
for field in all_fields:
    analyzer.add_field(field)

# Get before/after comparison
comparison = analyzer.compare_before_after()

# Analyze transient response
transient = analyzer.analyze_transient()

# Key metrics
print(f"TKE amplification: {comparison.tke_ratio.mean():.2f}")
print(f"Transient duration: {transient['transient_duration']:.2f}")
print(f"Relaxation time: {transient['relaxation_time']:.2f}")
print(f"Overshoot: {transient['overshoot']*100:.1f}%")

# Summary
summary = analyzer.get_summary()
```

## Output Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| NumPy | `.npz` | Compressed NumPy arrays (default) |
| HDF5 | `.h5` | Hierarchical Data Format (requires h5py) |
| CSV | `.csv` | Comma-separated values |
| VTK | `.vtk` | Visualization Toolkit (for ParaView) |

## Performance Tips

1. **Memory Management**: Use `OnlineMeanComputer` or `OnlineSecondOrderComputer` for large datasets
2. **Parallel Reading**: Read multiple files in parallel when possible
3. **Mesh Caching**: First call to `read_field()` caches mesh for subsequent reads
4. **Windowed Statistics**: Use `windowed_statistics()` for very long time series

## License

MIT License

## Contributing

Contributions are welcome! Please submit pull requests or open issues for:
- Bug fixes
- New features
- Documentation improvements
- Performance optimizations

## References

- Nek5000 Documentation: https://nek5000.github.io/NekDoc/
- Pope, S.B. (2000). Turbulent Flows. Cambridge University Press.
- Moser, R.D., Kim, J., & Mansour, N.N. (1999). DNS of turbulent channel flow up to Re_τ=590. Physics of Fluids.
