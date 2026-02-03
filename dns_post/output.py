from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np


def save_npz(path: Path, arrays: Mapping[str, np.ndarray], compress: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def build_stats_payload(
    stats: Mapping[str, object],
    coords: Optional[Mapping[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    payload: Dict[str, np.ndarray] = {}
    if coords:
        for name, values in coords.items():
            payload[f"coord_{name}"] = values

    mean = stats.get("mean", {})
    variance = stats.get("variance", {})
    rms = stats.get("rms", {})
    covariance = stats.get("covariance", {})

    for name, values in mean.items():
        payload[f"mean_{name}"] = values
    for name, values in variance.items():
        payload[f"variance_{name}"] = values
    for name, values in rms.items():
        payload[f"rms_{name}"] = values
    for name, values in covariance.items():
        payload[f"cov_{name}"] = values

    if stats.get("tke") is not None:
        payload["tke"] = stats["tke"]

    payload["count"] = np.asarray(stats.get("count", 0))
    payload["weight_sum"] = np.asarray(stats.get("weight_sum", 0.0))
    return payload


def write_stats(
    output_dir: Path,
    stats: Mapping[str, object],
    coords: Optional[Mapping[str, np.ndarray]],
    filename: str,
    compress: bool,
) -> Path:
    payload = build_stats_payload(stats, coords)
    path = output_dir / filename
    save_npz(path, payload, compress=compress)
    return path


def write_comparisons(
    output_dir: Path,
    comparisons: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    compress: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, comparison in comparisons.items():
        payload: Dict[str, np.ndarray] = {}
        for var, values in comparison.get("delta_mean", {}).items():
            payload[f"delta_mean_{var}"] = values
        for var, values in comparison.get("percent_mean", {}).items():
            payload[f"percent_mean_{var}"] = values
        save_npz(output_dir / f"{name}.npz", payload, compress=compress)


def write_metadata(path: Path, metadata: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def write_timeseries(path: Path, timeseries: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(timeseries, handle, indent=2, sort_keys=True)
