# Nek5000 DNS Post-Processor with Second-Order Statistics

A comprehensive post-processing toolkit for Nek5000 Direct Numerical Simulation (DNS) data with specialized features for analyzing step change dynamics, calculating second-order statistics, and performing temporal comparisons.

## Features

### Core Capabilities

- **Nek5000 Binary File Reading**: Native support for Nek5000 field files (.fld, .f0xxxx format)
- **Second-Order Statistics**: 
  - Mean, variance, standard deviation, RMS
  - Reynolds stress tensor components (u'u', v'v', w'w', u'v', u'w', v'w')
  - Turbulent kinetic energy (TKE)
  - Higher-order moments (skewness, kurtosis)
- **Temporal Analysis**:
  - Time window-based statistics
  - Temporal evolution tracking
  - Step change response characterization
  - Settling time detection
- **Advanced Diagnostics**:
  - Turbulence intensity profiles
  - Statistical distributions
  - Spatial and temporal averaging
- **Publication-Quality Visualization**:
  - Comprehensive plotting suite
  - Multi-panel comparative plots
  - Statistical distribution visualizations

## Installation

### Requirements

- Python 3.7+
- NumPy >= 1.21.0
- Matplotlib >= 3.4.0
- SciPy >= 1.7.0
- h5py >= 3.1.0

### Setup

```bash
# Clone or download the repository
cd /path/to/postprocessor

# Install dependencies
pip install -r requirements.txt

# Make scripts executable
chmod +x nek5000_postprocessor.py
chmod +x visualize_results.py
chmod +x example_usage.py
chmod +x generate_test_data.py
```

## Quick Start

### 1. Basic Usage

```bash
python nek5000_postprocessor.py <case_name> \
  --data-dir ./field_data \
  --output-dir ./results \
  --variables u v w p
```

### 2. With Time Windows (for step change analysis)

Create a configuration file `time_windows.json`:

```json
{
  "windows": [
    {"start": 0.0, "end": 10.0, "label": "pre_step"},
    {"start": 10.0, "end": 20.0, "label": "transition"},
    {"start": 20.0, "end": 40.0, "label": "settling"},
    {"start": 40.0, "end": 100.0, "label": "quasi_steady"}
  ]
}
```

Then run:

```bash
python nek5000_postprocessor.py step_dns \
  --data-dir ./field_data \
  --output-dir ./results \
  --time-windows time_windows.json \
  --variables u v w p T
```

### 3. Generate Visualizations

```bash
python visualize_results.py --results-dir ./results --plot-type all
```

## Detailed Usage

### Main Post-Processor Script

**Command-line interface:**

```bash
python nek5000_postprocessor.py [OPTIONS] case_name
```

**Options:**
- `case_name`: Case name (prefix of field files, e.g., "dns" for "dns0.f00001")
- `--data-dir DIR`: Directory containing field files (default: current directory)
- `--output-dir DIR`: Output directory for results (default: ./results)
- `--start-time T`: Start time for analysis
- `--end-time T`: End time for analysis
- `--variables VAR [VAR ...]`: Variables to analyze (default: u v w p)
- `--time-windows FILE`: JSON file with time window definitions

**Example:**

```bash
python nek5000_postprocessor.py channel_flow \
  --data-dir /data/nek5000/runs/channel_flow/ \
  --output-dir ./channel_flow_results \
  --start-time 50.0 \
  --end-time 200.0 \
  --variables u v w p T \
  --time-windows windows_config.json
```

### Python API Usage

```python
from nek5000_postprocessor import PostProcessor

# Initialize
pp = PostProcessor(
    case_name='step_dns',
    data_dir='./field_data',
    output_dir='./results'
)

# Load data
pp.load_time_range(start_time=0.0, end_time=100.0)

# Compute global statistics
global_stats = pp.compute_global_statistics(['u', 'v', 'w', 'p'])

# Analyze temporal evolution
evolution = pp.analyze_temporal_evolution(['u', 'v', 'w'])

# Define and analyze time windows
time_windows = [
    (0.0, 10.0, "baseline"),
    (10.0, 30.0, "transient"),
    (30.0, 100.0, "steady")
]
pp.compute_windowed_statistics(time_windows, ['u', 'v', 'w', 'p'])

# Compare windows
comparison = pp.compare_time_windows(['u', 'v', 'w'])

# Analyze step response
response = pp.analyze_step_response(['u', 'v'])

# Generate visualizations
pp.generate_plots(global_stats)

# Generate report
pp.generate_report()
```

## Output Files

### Statistics Files

**HDF5 Format (`*.h5`):**
- `global_statistics.h5`: Complete statistics for entire time range
- `statistics_<label>.h5`: Statistics for specific time windows

Contains datasets:
- `/mean/{variable}`: Mean field for each variable
- `/variance/{variable}`: Variance field
- `/std_dev/{variable}`: Standard deviation field
- `/rms/{variable}`: RMS field
- `/skewness/{variable}`: Skewness field
- `/kurtosis/{variable}`: Kurtosis field
- `/reynolds_stress/{component}`: Reynolds stress components

**JSON Format:**
- `global_statistics_summary.json`: Summary statistics
- `temporal_evolution.json`: Time series of spatial averages
- `window_comparison.json`: Comparisons between time windows
- `step_response_analysis.json`: Step response characteristics

### Visualization Files

All plots saved as high-resolution PNG files (300 DPI):
- `temporal_evolution.png`: Time evolution of spatial averages
- `statistics_profiles.png`: Spatial profiles of mean and fluctuations
- `reynolds_stress.png`: Reynolds stress components
- `window_comparison.png`: Bar charts comparing time windows
- `figures/` subdirectory with advanced visualizations

### Reports

- `analysis_report.txt`: Comprehensive text report with all key metrics

## Statistics Calculated

### First-Order Statistics
- **Mean**: ⟨u⟩, ⟨v⟩, ⟨w⟩, ⟨p⟩
- **Variance**: σ²
- **Standard Deviation**: σ
- **RMS**: √(⟨u²⟩)

### Second-Order Statistics (Reynolds Stress Tensor)
- **Normal Stresses**: ⟨u'u'⟩, ⟨v'v'⟩, ⟨w'w'⟩
- **Shear Stresses**: ⟨u'v'⟩, ⟨u'w'⟩, ⟨v'w'⟩
- **Turbulent Kinetic Energy**: TKE = 0.5(⟨u'u'⟩ + ⟨v'v'⟩ + ⟨w'w'⟩)

### Higher-Order Statistics
- **Skewness**: S = ⟨u'³⟩/σ³
- **Kurtosis**: K = ⟨u'⁴⟩/σ⁴

### Derived Quantities
- **Turbulence Intensity**: TI = σ/|U| × 100%
- **Reynolds Number**: Based on local quantities

## Step Change Analysis

The post-processor includes specialized tools for analyzing DNS of step changes:

### Response Characterization

1. **Step Magnitude**: Change in mean value
2. **Overshoot**: Maximum deviation from final value
3. **Settling Time**: Time to reach within threshold of final value
4. **Transition Dynamics**: Temporal evolution through step change

### Time Window Strategy

For optimal step change analysis, define windows covering:

1. **Baseline**: Pre-step steady state
2. **Initiation**: Step change onset
3. **Transient**: Dynamic response period
4. **Settling**: Approach to new steady state
5. **Quasi-Steady**: New equilibrium

Example configuration:

```json
{
  "windows": [
    {"start": 0.0, "end": 5.0, "label": "baseline", 
     "description": "Steady state before step"},
    {"start": 5.0, "end": 10.0, "label": "initiation",
     "description": "Step change initiation"},
    {"start": 10.0, "end": 25.0, "label": "early_transient",
     "description": "Early transient response"},
    {"start": 25.0, "end": 50.0, "label": "mid_transient",
     "description": "Mid transient development"},
    {"start": 50.0, "end": 100.0, "label": "settling",
     "description": "Settling to new state"},
    {"start": 100.0, "end": 200.0, "label": "quasi_steady",
     "description": "New quasi-steady state"}
  ]
}
```

## Advanced Visualization

### Visualization Script

```bash
python visualize_results.py --results-dir ./results --plot-type [all|temporal|reynolds|tke|comparison|distributions|turbulence]
```

**Plot Types:**
- `all`: Generate all available plots
- `temporal`: Temporal evolution plots
- `reynolds`: Reynolds stress components
- `tke`: Turbulent kinetic energy analysis
- `comparison`: Window comparison plots
- `distributions`: Statistical distributions (skewness, kurtosis)
- `turbulence`: Turbulence intensity profiles

### Customization

Modify plot parameters in `visualize_results.py`:

```python
# Publication-quality settings
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
```

## Testing with Synthetic Data

Generate synthetic DNS data with step change for testing:

```bash
python generate_test_data.py \
  --output-dir ./test_data \
  --case-name test_dns \
  --n-elements 50 \
  --grid-size 8 \
  --t-start 0.0 \
  --t-end 50.0 \
  --dt 0.5 \
  --step-time 10.0
```

Then analyze:

```bash
python nek5000_postprocessor.py test_dns \
  --data-dir ./test_data \
  --output-dir ./test_results \
  --variables u v w p T
```

## Example Workflow

Complete example provided in `example_usage.py`:

```bash
# Run complete example workflow
python example_usage.py --mode default

# Or use configuration file
python example_usage.py --mode config
```

The example demonstrates:
1. Loading field files
2. Computing global statistics
3. Analyzing temporal evolution
4. Windowed statistics for step change
5. Time window comparisons
6. Step response characterization
7. Visualization generation
8. Report generation

## File Format Support

### Nek5000 Binary Format

The post-processor reads standard Nek5000 binary field files with:
- Header: 132 bytes
- Word size: 4 bytes (float) or 8 bytes (double)
- Data layout: coordinates (x, y, z), velocity (u, v, w), pressure (p), temperature (T)
- Elements: Spectral element data structure

### Expected File Naming

Files should follow Nek5000 convention:
- `{case_name}0.f00001`
- `{case_name}0.f00002`
- etc.

## Performance Considerations

### Memory Usage

- Each field file is loaded into memory
- For large datasets, process in batches using time ranges
- Statistics are computed incrementally

### Optimization Tips

```python
# Load specific time range
pp.load_time_range(start_time=50.0, end_time=100.0)

# Analyze subset of variables
pp.compute_global_statistics(['u', 'p'])  # Instead of all variables

# Limit spatial sampling for very large grids
# (modify reader to subsample if needed)
```

## Troubleshooting

### Common Issues

**Problem**: "No field files found"
- **Solution**: Check case name matches file prefix
- Ensure files match pattern: `{case_name}0.f*`
- Verify data directory path

**Problem**: "Could not read field file"
- **Solution**: Verify file format (binary vs ASCII)
- Check word size (4-byte vs 8-byte)
- Ensure file is complete (not corrupted)

**Problem**: "Insufficient data for statistics"
- **Solution**: Ensure multiple time samples available
- Check time window definitions don't exclude all data
- Verify start_time and end_time settings

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Citation

If you use this post-processor in your research, please cite:

```
DNS Post-Processor for Nek5000
Second-Order Statistics and Temporal Analysis Tools
[Your Institution/Name], 2026
```

## Contributing

Contributions welcome! Areas for enhancement:
- Additional file format support
- More statistical measures
- Advanced visualization options
- Parallel processing for large datasets
- Spatial filtering and interpolation

## License

[Specify your license here]

## Contact

[Your contact information]

## References

1. Nek5000 Documentation: https://nek5000.mcs.anl.gov/
2. Fischer, P. F., et al. "Nek5000: Open source spectral element CFD solver"
3. Pope, S. B. "Turbulent Flows" Cambridge University Press, 2000
4. Reynolds Stress and DNS: Moin, P. & Mahesh, K. "Direct Numerical Simulation: A Tool in Turbulence Research"

## Version History

- **v1.0.0** (2026-02): Initial release
  - Complete Nek5000 binary reader
  - Second-order statistics calculation
  - Temporal comparison tools
  - Step change analysis
  - Comprehensive visualization suite
