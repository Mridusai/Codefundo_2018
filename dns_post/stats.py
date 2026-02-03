from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np


class WeightedMean:
    def __init__(self, shape: Tuple[int, ...], dtype: np.dtype) -> None:
        self.mean = np.zeros(shape, dtype=dtype)
        self.weight_sum = 0.0

    def update(self, values: np.ndarray, weight: float) -> None:
        if weight <= 0.0:
            return
        new_weight = self.weight_sum + weight
        delta = values - self.mean
        self.mean += (weight / new_weight) * delta
        self.weight_sum = new_weight


class WeightedRunningStats:
    def __init__(self, shape: Tuple[int, ...], dtype: np.dtype) -> None:
        self.mean = np.zeros(shape, dtype=dtype)
        self.M2 = np.zeros(shape, dtype=dtype)
        self.weight_sum = 0.0

    def update(self, values: np.ndarray, weight: float) -> None:
        if weight <= 0.0:
            return
        new_weight = self.weight_sum + weight
        delta = values - self.mean
        self.mean += (weight / new_weight) * delta
        self.M2 += weight * delta * (values - self.mean)
        self.weight_sum = new_weight

    def variance(self) -> np.ndarray:
        if self.weight_sum <= 0.0:
            return np.zeros_like(self.mean)
        return self.M2 / self.weight_sum


@dataclass
class StatsAccumulator:
    stats: Dict[str, WeightedRunningStats]
    mean_products: Dict[Tuple[str, str], WeightedMean]
    count: int = 0
    weight_sum: float = 0.0

    @classmethod
    def from_fields(
        cls,
        fields: Mapping[str, np.ndarray],
        covariance_pairs: Iterable[Tuple[str, str]],
    ) -> "StatsAccumulator":
        stats = {
            name: WeightedRunningStats(values.shape, values.dtype)
            for name, values in fields.items()
        }
        mean_products: Dict[Tuple[str, str], WeightedMean] = {}
        for var_a, var_b in covariance_pairs:
            if var_a in fields and var_b in fields:
                mean_products[(var_a, var_b)] = WeightedMean(
                    fields[var_a].shape, fields[var_a].dtype
                )
        return cls(stats=stats, mean_products=mean_products)

    def update(self, fields: Mapping[str, np.ndarray], weight: float) -> None:
        if weight <= 0.0:
            return
        self.count += 1
        self.weight_sum += weight
        for name, values in fields.items():
            self.stats[name].update(values, weight)
        for (var_a, var_b), mean_prod in self.mean_products.items():
            if var_a in fields and var_b in fields:
                mean_prod.update(fields[var_a] * fields[var_b], weight)


def finalize_stats(accumulator: StatsAccumulator) -> Dict[str, object]:
    mean = {name: stat.mean.copy() for name, stat in accumulator.stats.items()}
    variance = {name: stat.variance() for name, stat in accumulator.stats.items()}
    rms = {name: np.sqrt(var) for name, var in variance.items()}
    covariance = {}
    for (var_a, var_b), mean_prod in accumulator.mean_products.items():
        if var_a in mean and var_b in mean:
            key = f"{var_a}{var_b}"
            covariance[key] = mean_prod.mean - mean[var_a] * mean[var_b]

    tke = None
    if all(name in variance for name in ("u", "v", "w")):
        tke = 0.5 * (variance["u"] + variance["v"] + variance["w"])

    return {
        "mean": mean,
        "variance": variance,
        "rms": rms,
        "covariance": covariance,
        "tke": tke,
        "count": accumulator.count,
        "weight_sum": accumulator.weight_sum,
    }


def compute_time_weights(times: List[float]) -> List[float]:
    if not times:
        return []
    if len(times) == 1:
        return [1.0]
    times_arr = np.asarray(times, dtype=float)
    weights = np.zeros_like(times_arr)
    weights[0] = 0.5 * (times_arr[1] - times_arr[0])
    weights[-1] = 0.5 * (times_arr[-1] - times_arr[-2])
    weights[1:-1] = 0.5 * (times_arr[2:] - times_arr[:-2])
    return weights.tolist()
