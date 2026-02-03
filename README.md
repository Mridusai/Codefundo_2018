# Nek5000 DNS Post-Processor

A comprehensive Python-based post-processing toolkit for Nek5000 Direct Numerical Simulations (DNS) of step change flows, providing full support for computing and analyzing first and second-order turbulence statistics with temporal comparison capabilities.

## Overview

This post-processor is designed for analyzing DNS data from Nek5000 simulations, with particular focus on:
- **Step change flows**: Simulations where flow conditions change abruptly
- **Turbulence statistics**: Both first-order (mean) and second-order (Reynolds stresses)
- **Temporal evolution**: Track how statistics evolve over time

## Features

| Feature | Description |
|---------|-------------|
| **Field Reader** | Read Nek5000 binary output files (.fld, f00001) |
| **First-Order Stats** | Mean velocity, pressure, temperature |
| **Second-Order Stats** | Reynolds stresses, TKE, turbulent dissipation |
| **Temporal Analysis** | Running averages, windowed statistics, step change comparison |
| **Visualization** | Contour plots, profiles, temporal evolution |
| **Export** | VTK (ParaView), HDF5, CSV, NumPy formats |

## Quick Start

```bash
# Install dependencies
cd nek5000_postprocessor
pip install -r requirements.txt

# Run the demo with synthetic data
python examples/demo_synthetic_data.py

# Post-process actual Nek5000 data
python postprocess.py --case channel --data-dir ./output --start 1 --end 1000

# For step change analysis
python postprocess.py --case dns --data-dir ./data --start 100 --end 500 --step-change 200
```

## Project Structure

```
nek5000_postprocessor/
├── __init__.py              # Package initialization
├── postprocess.py           # Main CLI script
├── requirements.txt         # Python dependencies
├── README.md               # Detailed documentation
├── example_config.yaml     # Example configuration
│
├── readers/                # Field file readers
│   ├── __init__.py
│   └── field_reader.py     # Nek5000 binary file reader
│
├── statistics/             # Turbulence statistics
│   ├── __init__.py
│   ├── first_order.py      # Mean statistics
│   └── second_order.py     # Reynolds stresses, TKE
│
├── temporal/               # Temporal analysis
│   ├── __init__.py
│   ├── temporal_averaging.py   # Time averaging methods
│   └── temporal_comparison.py  # Step change analysis
│
├── visualization/          # Plotting tools
│   ├── __init__.py
│   └── plotting.py         # Matplotlib-based plotting
│
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── io.py              # File I/O
│   ├── grid.py            # Grid operations
│   └── physics.py         # Physical quantities
│
└── examples/              # Example scripts
    ├── __init__.py
    └── demo_synthetic_data.py  # Demo with synthetic data
```

## Usage Examples

### Basic Statistics Computation

```python
from nek5000_postprocessor import (
    Nek5000Reader,
    compute_mean_fields,
    compute_reynolds_stresses,
)

# Read Nek5000 field files
reader = Nek5000Reader('channel', './data/')
fields = reader.read_field_sequence(1, 100)

# Compute first-order statistics (mean fields)
mean_stats = compute_mean_fields(fields)
print(f"Mean velocity: {mean_stats.u_mean.max()}")

# Compute second-order statistics (Reynolds stresses)
reynolds = compute_reynolds_stresses(fields, mean_stats)
print(f"TKE: {reynolds.tke.mean()}")
print(f"Reynolds shear stress <uv>: {reynolds.uv.max()}")
```

### Step Change Analysis

```python
from nek5000_postprocessor.temporal import StepChangeAnalyzer

# Analyze step change (e.g., Reynolds number change at t=100)
analyzer = StepChangeAnalyzer(step_time=100.0)
for field in all_fields:
    analyzer.add_field(field)

# Compare before/after statistics
comparison = analyzer.compare_before_after()
print(f"TKE amplification: {comparison.tke_ratio.mean():.2f}x")

# Analyze transient response
transient = analyzer.analyze_transient()
print(f"Relaxation time: {transient['relaxation_time']}")
```

## Second-Order Statistics

The post-processor computes the following turbulence statistics:

| Statistic | Symbol | Description |
|-----------|--------|-------------|
| Reynolds normal stress | `<u'u'>`, `<v'v'>`, `<w'w'>` | Velocity variances |
| Reynolds shear stress | `<u'v'>`, `<u'w'>`, `<v'w'>` | Velocity covariances |
| Turbulent Kinetic Energy | TKE | `0.5 * (uu + vv + ww)` |
| Pressure variance | `<p'p'>` | Pressure fluctuation variance |
| Turbulent heat flux | `<u'T'>`, `<v'T'>` | Temperature-velocity correlations |

## Configuration

Create a YAML configuration file for reproducible post-processing:

```yaml
case_name: channel
data_dir: ./output
output_dir: ./results

timesteps:
  start: 1
  end: 1000
  step: 1

statistics:
  compute_first_order: true
  compute_second_order: true

temporal:
  step_change_time: 500.0  # Set to null if no step change
```

## License

MIT License

## Author

Mridusai
