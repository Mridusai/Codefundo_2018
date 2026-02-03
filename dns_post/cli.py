from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from dns_post.io import (
    collect_file_info,
    discover_files,
    normalize_variable_list,
    normalize_variable_name,
    read_snapshot,
)
from dns_post.models import ComparisonSpec, TimeWindow
from dns_post.output import write_comparisons, write_metadata, write_stats, write_timeseries
from dns_post.stats import StatsAccumulator, compute_time_weights, finalize_stats


def parse_window(spec: str) -> TimeWindow:
    parts = spec.split(":")
    if len(parts) < 2 or len(parts) > 3:
        raise ValueError(
            "Window spec must be label:start:end (start or end may be empty)."
        )
    label = parts[0].strip()
    if not label:
        raise ValueError("Window label must be non-empty.")

    start = _parse_optional_float(parts[1])
    end = _parse_optional_float(parts[2]) if len(parts) == 3 else None
    return TimeWindow(label=label, start=start, end=end)


def parse_comparison(spec: str) -> ComparisonSpec:
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError("Comparison spec must be label:reference.")
    label = parts[0].strip()
    reference = parts[1].strip()
    if not label or not reference:
        raise ValueError("Comparison labels must be non-empty.")
    return ComparisonSpec(label=label, reference=reference)


def parse_covariance_pairs(spec: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if not spec:
        return pairs
    for item in spec.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if ":" in token:
            left, right = token.split(":", 1)
        elif len(token) == 2:
            left, right = token[0], token[1]
        else:
            raise ValueError(
                f"Invalid covariance pair '{token}'. Use uv or u:v syntax."
            )
        pairs.append((normalize_variable_name(left), normalize_variable_name(right)))
    return pairs


def _parse_optional_float(text: str) -> Optional[float]:
    if text is None:
        return None
    stripped = text.strip()
    if stripped == "" or stripped.lower() == "none":
        return None
    return float(stripped)


def _weighted_average(values: Sequence[float], weights: Sequence[float]) -> float:
    total_weight = float(sum(weights))
    if total_weight <= 0.0:
        return float("nan")
    return float(sum(val * wt for val, wt in zip(values, weights)) / total_weight)


def _window_indices(times: Sequence[float], window: TimeWindow) -> List[int]:
    return [i for i, t in enumerate(times) if window.contains(t)]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process Nek5000 DNS fields with first/second-order statistics "
            "and temporal comparisons."
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Field file directory.")
    parser.add_argument("--pattern", default="*.f*", help="Glob pattern for field files.")
    parser.add_argument(
        "--recursive", action="store_true", help="Search for files recursively."
    )
    parser.add_argument(
        "--format", choices=("nek", "npz"), default="nek", help="Input format."
    )
    parser.add_argument(
        "--variables",
        default="",
        help="Comma-separated variables (u,v,w,p,t,s01...). Default: all.",
    )
    parser.add_argument(
        "--no-coords",
        dest="include_coords",
        action="store_false",
        help="Skip writing coordinates.",
    )
    parser.set_defaults(include_coords=True)
    parser.add_argument("--t-start", type=float, default=None, help="Start time.")
    parser.add_argument("--t-end", type=float, default=None, help="End time.")
    parser.add_argument(
        "--time-weighted",
        action="store_true",
        help="Use trapezoidal time weighting.",
    )
    parser.add_argument(
        "--window",
        action="append",
        default=[],
        help="Time window label:start:end (start or end may be empty).",
    )
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        help="Comparison spec label:reference.",
    )
    parser.add_argument(
        "--covariance-pairs",
        default="uv,uw,vw",
        help="Covariance pairs (uv,uw or u:v syntax).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-12,
        help="Epsilon for percent difference denominators.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("postprocess_out"),
        help="Output directory.",
    )
    parser.add_argument(
        "--no-compress",
        dest="compress",
        action="store_false",
        help="Disable NPZ compression.",
    )
    parser.set_defaults(compress=True)
    parser.add_argument(
        "--check-coords",
        action="store_true",
        help="Verify coordinates are identical across snapshots.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    data_dir = args.data_dir.expanduser()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    paths = discover_files(data_dir, args.pattern, args.recursive)
    if not paths:
        raise FileNotFoundError(
            f"No files found under {data_dir} with pattern '{args.pattern}'."
        )

    variables = normalize_variable_list(args.variables.split(","))
    covariance_pairs = parse_covariance_pairs(args.covariance_pairs)

    infos = collect_file_info(paths, args.format)
    infos.sort(key=lambda info: (info.time, info.path.name))

    if args.t_start is not None:
        infos = [info for info in infos if info.time >= args.t_start]
    if args.t_end is not None:
        infos = [info for info in infos if info.time <= args.t_end]

    if not infos:
        raise ValueError("No files left after applying time filters.")

    windows = [parse_window(spec) for spec in args.window]
    if args.compare and not windows:
        raise ValueError("Comparisons require at least one time window.")

    if windows:
        labels = [window.label for window in windows]
        if len(labels) != len(set(labels)):
            raise ValueError("Window labels must be unique.")

    times = [info.time for info in infos]
    if args.time_weighted:
        weights = compute_time_weights(times)
    else:
        weights = [1.0] * len(times)

    global_acc: Optional[StatsAccumulator] = None
    window_accs: Dict[str, StatsAccumulator] = {}
    spatial_mean: Dict[str, List[float]] = {}
    timeseries_weights: List[float] = []
    output_coords: Optional[Dict[str, np.ndarray]] = None
    reference_shapes: Dict[str, Tuple[int, ...]] = {}
    reference_coords: Optional[Dict[str, np.ndarray]] = None

    for info, weight in zip(infos, weights):
        snapshot = read_snapshot(info.path, args.format, variables, args.include_coords)
        fields = snapshot.fields

        if global_acc is None:
            if variables is not None:
                missing = sorted(set(variables) - set(fields))
                if missing:
                    raise ValueError(
                        "Requested variables missing from first snapshot: "
                        + ", ".join(missing)
                    )
            global_acc = StatsAccumulator.from_fields(fields, covariance_pairs)
            if windows:
                window_accs = {
                    window.label: StatsAccumulator.from_fields(fields, covariance_pairs)
                    for window in windows
                }
            spatial_mean = {name: [] for name in fields}
            reference_shapes = {name: values.shape for name, values in fields.items()}
            if args.include_coords:
                output_coords = snapshot.coords
                reference_coords = snapshot.coords
        else:
            for name, values in fields.items():
                if name not in reference_shapes:
                    raise ValueError(f"Unexpected variable '{name}' in {info.path}.")
                if values.shape != reference_shapes[name]:
                    raise ValueError(
                        f"Shape mismatch for '{name}' in {info.path}: "
                        f"{values.shape} != {reference_shapes[name]}"
                    )

        if args.check_coords and args.include_coords and reference_coords:
            for name, coords in snapshot.coords.items():
                if name not in reference_coords:
                    raise ValueError(f"Unexpected coord '{name}' in {info.path}.")
                if not np.allclose(coords, reference_coords[name]):
                    raise ValueError(f"Coordinate mismatch for '{name}' in {info.path}.")

        global_acc.update(fields, weight)
        for window in windows:
            if window.contains(snapshot.time):
                window_accs[window.label].update(fields, weight)

        for name, values in fields.items():
            spatial_mean[name].append(float(np.mean(values)))
        timeseries_weights.append(weight)

    if global_acc is None:
        raise RuntimeError("No data processed.")

    global_stats = finalize_stats(global_acc)
    window_stats = {label: finalize_stats(acc) for label, acc in window_accs.items()}

    comparisons: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    compare_specs = [parse_comparison(spec) for spec in args.compare]
    if compare_specs:
        for spec in compare_specs:
            if spec.label not in window_stats or spec.reference not in window_stats:
                raise ValueError(
                    f"Comparison '{spec.label}:{spec.reference}' "
                    "refers to missing window."
                )
            mean_target = window_stats[spec.label]["mean"]
            mean_ref = window_stats[spec.reference]["mean"]
            delta = {}
            percent = {}
            for name in mean_target:
                if name not in mean_ref:
                    continue
                diff = mean_target[name] - mean_ref[name]
                delta[name] = diff
                percent[name] = diff / (np.abs(mean_ref[name]) + args.epsilon) * 100.0
            key = f"{spec.label}_vs_{spec.reference}"
            comparisons[key] = {"delta_mean": delta, "percent_mean": percent}

    timeseries: Dict[str, object] = {
        "times": times,
        "weights": timeseries_weights,
        "spatial_mean": spatial_mean,
    }

    if windows:
        window_series = {}
        for window in windows:
            indices = _window_indices(times, window)
            if not indices:
                continue
            window_weights = [timeseries_weights[i] for i in indices]
            window_means = {}
            for name, series in spatial_mean.items():
                window_values = [series[i] for i in indices]
                window_means[name] = _weighted_average(window_values, window_weights)
            window_series[window.label] = window_means
        timeseries["window_spatial_mean"] = window_series

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    write_stats(
        output_dir,
        global_stats,
        output_coords,
        filename="stats.npz",
        compress=args.compress,
    )
    if window_stats:
        window_dir = output_dir / "windows"
        for label, stats in window_stats.items():
            write_stats(
                window_dir,
                stats,
                output_coords,
                filename=f"{label}.npz",
                compress=args.compress,
            )
    if comparisons:
        write_comparisons(output_dir / "comparisons", comparisons, compress=args.compress)

    metadata = {
        "data_dir": str(data_dir),
        "pattern": args.pattern,
        "format": args.format,
        "time_weighted": args.time_weighted,
        "t_start": args.t_start,
        "t_end": args.t_end,
        "variables": sorted(reference_shapes.keys()),
        "covariance_pairs": covariance_pairs,
        "window_defs": [
            {"label": window.label, "start": window.start, "end": window.end}
            for window in windows
        ],
        "comparisons": [
            {"label": spec.label, "reference": spec.reference}
            for spec in compare_specs
        ],
        "count": global_stats.get("count"),
        "weight_sum": global_stats.get("weight_sum"),
        "files": [str(info.path) for info in infos],
    }
    write_metadata(output_dir / "metadata.json", metadata)
    write_timeseries(output_dir / "timeseries.json", timeseries)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
