# Quick Start Guide - Nek5000 DNS Post-Processor

Get started with the Nek5000 DNS post-processor in 5 minutes!

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Option 1: Test with Synthetic Data (Recommended for First Run)

Perfect for testing the post-processor without real DNS data.

### Step 1: Generate Test Data

```bash
python generate_test_data.py \
  --output-dir ./test_data \
  --case-name demo \
  --t-end 50.0 \
  --step-time 15.0
```

This creates ~100 field files with a step change at t=15.0

### Step 2: Run Post-Processor

```bash
python nek5000_postprocessor.py demo \
  --data-dir ./test_data \
  --output-dir ./demo_results \
  --variables u v w p T
```

### Step 3: View Results

```bash
# Check the analysis report
cat ./demo_results/analysis_report.txt

# Generate advanced visualizations
python visualize_results.py --results-dir ./demo_results
```

### Step 4: View Plots

Open the generated PNG files in `./demo_results/` and `./demo_results/figures/`

---

## Option 2: Analyze Real Nek5000 Data

### For Simple Analysis (No Time Windows)

```bash
python nek5000_postprocessor.py <your_case_name> \
  --data-dir /path/to/nek5000/output \
  --output-dir ./results \
  --variables u v w p
```

### For Step Change Analysis (With Time Windows)

1. **Create time windows configuration** (`my_windows.json`):

```json
{
  "windows": [
    {"start": 0.0, "end": 10.0, "label": "before_step"},
    {"start": 10.0, "end": 20.0, "label": "transition"},
    {"start": 20.0, "end": 50.0, "label": "after_step"}
  ]
}
```

2. **Run analysis**:

```bash
python nek5000_postprocessor.py <your_case_name> \
  --data-dir /path/to/nek5000/output \
  --output-dir ./results \
  --time-windows my_windows.json \
  --variables u v w p T
```

---

## Option 3: Use the Example Script

The example script demonstrates the complete workflow:

```bash
python example_usage.py
```

Or with custom configuration:

```bash
# Edit config_example.json first, then:
python example_usage.py --mode config
```

---

## Understanding the Output

### Key Output Files

| File | Description |
|------|-------------|
| `global_statistics.h5` | Complete statistics (all spatial points) |
| `global_statistics_summary.json` | Summary of global statistics |
| `temporal_evolution.json` | Time series of spatial averages |
| `window_comparison.json` | Comparisons between time windows |
| `step_response_analysis.json` | Step change characteristics |
| `analysis_report.txt` | Human-readable comprehensive report |
| `*.png` | Visualization plots |
| `figures/*.png` | Advanced visualizations |

### Key Statistics Available

- **First-order**: Mean, variance, standard deviation, RMS
- **Second-order**: Reynolds stress (u'u', v'v', w'w', u'v', u'w', v'w'), TKE
- **Higher-order**: Skewness, kurtosis
- **Derived**: Turbulence intensity, step response metrics

---

## Common Use Cases

### 1. Quick Statistics Over Time Range

```bash
python nek5000_postprocessor.py mycaseSTART_TIME \
  --data-dir ./data \
  --start-time 50.0 \
  --end-time 100.0 \
  --variables u v w
```

### 2. Analyze Specific Variables Only

```bash
# Only analyze streamwise velocity and pressure
python nek5000_postprocessor.py mycase \
  --data-dir ./data \
  --variables u p
```

### 3. Generate Only Specific Plots

```bash
# Generate only Reynolds stress plots
python visualize_results.py \
  --results-dir ./results \
  --plot-type reynolds
```

Plot types: `temporal`, `reynolds`, `tke`, `comparison`, `distributions`, `turbulence`, `all`

---

## Validation

Run the test suite to verify everything works:

```bash
python test_postprocessor.py
```

This will:
1. Generate synthetic test data
2. Run all post-processor functions
3. Validate outputs
4. Generate test visualizations

---

## Troubleshooting

### "No field files found"

**Problem**: The post-processor can't find your Nek5000 output files.

**Solutions**:
- Check that `case_name` matches your file prefix
  - If files are `channel0.f00001`, use `case_name = "channel"`
- Verify `--data-dir` points to the correct directory
- Check file naming: expected pattern is `{case_name}0.f*`

### "Could not read field file"

**Problem**: File format not recognized or corrupted.

**Solutions**:
- Ensure files are in Nek5000 binary format
- Check that files are complete (not partially written)
- Verify word size (4-byte float or 8-byte double)

### "Insufficient samples"

**Problem**: Not enough time samples for statistics.

**Solutions**:
- Check `--start-time` and `--end-time` aren't too restrictive
- Verify field files exist in the time range
- Need at least 2-3 samples for meaningful statistics

### Memory issues with large datasets

**Solutions**:
- Process in batches using `--start-time` and `--end-time`
- Analyze fewer variables at once
- Reduce spatial sampling (modify reader code)

---

## Python API Quick Reference

```python
from nek5000_postprocessor import PostProcessor

# Initialize
pp = PostProcessor('mycase', data_dir='./data', output_dir='./results')

# Load data
pp.load_time_range(start_time=0.0, end_time=100.0)

# Compute statistics
stats = pp.compute_global_statistics(['u', 'v', 'w', 'p'])

# Temporal evolution
evolution = pp.analyze_temporal_evolution(['u', 'v', 'w'])

# Window analysis
windows = [(0, 10, 'pre'), (10, 30, 'during'), (30, 100, 'post')]
pp.compute_windowed_statistics(windows, ['u', 'v', 'w'])
comparison = pp.compare_time_windows(['u', 'v', 'w'])

# Step response
response = pp.analyze_step_response(['u', 'v'])

# Generate outputs
pp.generate_plots(stats)
pp.generate_report()
```

---

## What's Next?

### For Basic Analysis
1. Run with your data
2. Check `analysis_report.txt`
3. View the PNG plots
4. Examine JSON files for specific values

### For Detailed Analysis
1. Define appropriate time windows for your step change
2. Run with `--time-windows` option
3. Use `visualize_results.py` for publication-quality figures
4. Extract specific statistics from HDF5 files using Python:

```python
import h5py
import numpy as np

with h5py.File('results/global_statistics.h5', 'r') as f:
    mean_u = f['mean']['u'][:]
    tke = f['reynolds_stress']['TKE'][:]
    
    print(f"Mean u: {np.mean(mean_u):.6e}")
    print(f"Mean TKE: {np.mean(tke):.6e}")
```

### For Custom Analysis
- Modify `nek5000_postprocessor.py` to add custom statistics
- Extend `visualize_results.py` for custom plots
- Use the Statistics Calculator and Temporal Comparator classes directly

---

## Need Help?

1. **Check the full documentation**: See `README_POSTPROCESSOR.md`
2. **Run the test suite**: `python test_postprocessor.py`
3. **Try with test data**: Use `generate_test_data.py` first
4. **Check example**: Review `example_usage.py` for complete workflow

---

## Tips for Best Results

1. **Choose appropriate time windows** based on your physical problem
2. **Include enough samples** in each window (at least 10-20)
3. **Verify steady state** before and after step change
4. **Use consistent time steps** for smoother temporal evolution
5. **Check statistics validity** using the analysis report

---

## One-Line Demo

```bash
python generate_test_data.py && python nek5000_postprocessor.py test_dns --data-dir ./test_data --variables u v w p && python visualize_results.py --results-dir ./results
```

This generates test data, analyzes it, and creates all visualizations!
