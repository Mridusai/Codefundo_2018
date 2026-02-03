# Nek5000 DNS Post-Processor

This post-processor reads Nek5000 field files and computes:

- First-order statistics (time-averaged mean fields)
- Second-order statistics (variance, RMS, Reynolds covariances)
- Temporal comparisons between averaged fields (windowed mean deltas)
- Spatial mean time series for convergence checks

The implementation uses `pymech` for Nek5000 binary files and can also read
pre-converted NPZ snapshots.

## Install

```bash
python3 -m pip install -r requirements.txt
```

Optional:

- Install `h5py` if you want to extend the writers to HDF5.

## Basic Usage

```bash
python3 postprocess_dns.py \
  --data-dir /path/to/nek/run \
  --pattern "*.f*" \
  --time-weighted \
  --window baseline:50:100 \
  --window post:100:150 \
  --compare post:baseline \
  --output-dir postprocess_out
```

### Common Flags

- `--variables u,v,w,p,t,s01` to restrict which fields are processed.
- `--time-weighted` for trapezoidal weights using snapshot times.
- `--window label:start:end` for windowed stats (start/end may be empty).
- `--compare label:reference` for temporal mean comparisons.
- `--covariance-pairs uv,uw,vw` to configure Reynolds covariances.
- `--no-coords` to skip coordinate output.

## Output Layout

```
postprocess_out/
  stats.npz
  metadata.json
  timeseries.json
  windows/
    baseline.npz
    post.npz
  comparisons/
    post_vs_baseline.npz
```

### NPZ Contents

`stats.npz` and window files contain:

- `coord_x`, `coord_y`, `coord_z` (if enabled)
- `mean_u`, `mean_v`, `mean_w`, `mean_p`, `mean_t`, `mean_s01`, ...
- `variance_u`, `variance_v`, ...
- `rms_u`, `rms_v`, ...
- `cov_uv`, `cov_uw`, `cov_vw` (if requested)
- `tke` (0.5 * (u'2 + v'2 + w'2), if u/v/w are available)
- `count`, `weight_sum`

Comparison files contain:

- `delta_mean_u`, `delta_mean_v`, ...
- `percent_mean_u`, `percent_mean_v`, ...

`timeseries.json` contains snapshot times and spatial means for each variable.

## Notes and Assumptions

- Mesh topology must be consistent across snapshots (shape checks enforced).
- Spatial averages are unweighted (simple arithmetic mean).
- Time-weighted statistics use trapezoidal weights based on file times.
- Windowed comparisons are based on windowed mean fields.

## NPZ Snapshot Format (Optional)

For `--format npz`, each file should include:

- `time`
- `u`, `v`, `w`, `p`, `t`, `s01`, ... (as needed)
- `x`, `y`, `z` (optional)
