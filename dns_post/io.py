from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from pymech.neksuite import readnek
    from pymech.neksuite.field import read_header
except ImportError:  # pragma: no cover - optional dependency
    readnek = None
    read_header = None


@dataclass
class Snapshot:
    time: float
    fields: Dict[str, np.ndarray]
    coords: Dict[str, np.ndarray]
    metadata: Dict[str, object]


@dataclass(frozen=True)
class FileInfo:
    path: Path
    time: float


def discover_files(data_dir: Path, pattern: str, recursive: bool) -> List[Path]:
    if recursive:
        paths = list(data_dir.rglob(pattern))
    else:
        paths = list(data_dir.glob(pattern))
    return [path for path in paths if path.is_file()]


def normalize_variable_name(name: str) -> str:
    text = name.strip().lower()
    if text in ("ux", "u"):
        return "u"
    if text in ("uy", "v"):
        return "v"
    if text in ("uz", "w"):
        return "w"
    if text in ("p", "pressure"):
        return "p"
    if text in ("t", "temp", "temperature"):
        return "t"
    if text.startswith("s") and text[1:].isdigit():
        return f"s{int(text[1:]):02d}"
    return text


def normalize_variable_list(variables: Optional[Iterable[str]]) -> Optional[Set[str]]:
    if not variables:
        return None
    normalized = {normalize_variable_name(var) for var in variables if var.strip()}
    if not normalized or "all" in normalized:
        return None
    return normalized


def collect_file_info(paths: Sequence[Path], file_format: str) -> List[FileInfo]:
    infos = []
    for path in paths:
        time_value = read_time_from_file(path, file_format)
        infos.append(FileInfo(path=path, time=float(time_value)))
    return infos


def read_time_from_file(path: Path, file_format: str) -> float:
    if file_format == "nek":
        if read_header is None:
            raise ImportError("pymech is required to read Nek5000 field files.")
        header = read_header(path)
        return float(header.time)
    if file_format == "npz":
        with np.load(path) as data:
            if "time" not in data:
                raise KeyError(f"NPZ file missing 'time' entry: {path}")
            return float(data["time"])
    raise ValueError(f"Unsupported format: {file_format}")


def read_snapshot(
    path: Path,
    file_format: str,
    variables: Optional[Set[str]],
    include_coords: bool,
) -> Snapshot:
    if file_format == "nek":
        return read_nek_snapshot(path, variables, include_coords)
    if file_format == "npz":
        return read_npz_snapshot(path, variables, include_coords)
    raise ValueError(f"Unsupported format: {file_format}")


def read_nek_snapshot(
    path: Path,
    variables: Optional[Set[str]],
    include_coords: bool,
) -> Snapshot:
    if readnek is None or read_header is None:
        raise ImportError("pymech is required to read Nek5000 field files.")

    header = read_header(path)
    skip_vars = _build_skip_vars(header, variables, include_coords)
    data = readnek(str(path), dtype="float64", skip_vars=tuple(skip_vars))

    fields, coords = _extract_nek_fields(data, header, variables, include_coords)
    metadata = {
        "istep": int(header.istep),
        "ndim": int(header.nb_dims),
        "orders": tuple(int(value) for value in header.orders),
        "nel": int(header.nb_elems),
        "variables": header.variables,
    }
    return Snapshot(time=float(header.time), fields=fields, coords=coords, metadata=metadata)


def read_npz_snapshot(
    path: Path,
    variables: Optional[Set[str]],
    include_coords: bool,
) -> Snapshot:
    with np.load(path) as data:
        if "time" not in data:
            raise KeyError(f"NPZ file missing 'time' entry: {path}")
        time_value = float(data["time"])

        fields: Dict[str, np.ndarray] = {}
        coords: Dict[str, np.ndarray] = {}
        for key in data.files:
            if key == "time":
                continue
            normalized = normalize_variable_name(key)
            if normalized in {"u", "v", "w", "p", "t"} or normalized.startswith("s"):
                if variables is None or normalized in variables:
                    fields[normalized] = np.asarray(data[key])

            if include_coords:
                if key in {"x", "y", "z"}:
                    coords[key] = np.asarray(data[key])
                if key in {"coord_x", "coord_y", "coord_z"}:
                    coords[key.replace("coord_", "")] = np.asarray(data[key])

        metadata = {"source": "npz"}
        return Snapshot(time=time_value, fields=fields, coords=coords, metadata=metadata)


def _build_skip_vars(header, variables: Optional[Set[str]], include_coords: bool) -> List[str]:
    skip_vars: List[str] = []
    if not include_coords:
        skip_vars.extend(["x", "y", "z"])

    if variables is None:
        return skip_vars

    if "u" not in variables:
        skip_vars.append("ux")
    if "v" not in variables:
        skip_vars.append("uy")
    if "w" not in variables:
        skip_vars.append("uz")

    if "p" not in variables:
        skip_vars.append("pressure")
    if "t" not in variables:
        skip_vars.append("temperature")

    nb_scalars = int(header.nb_vars[4]) if header.nb_vars else 0
    scalar_names = [f"s{i:02d}" for i in range(1, nb_scalars + 1)]
    requested_scalars = {name for name in variables if name.startswith("s")}
    if requested_scalars:
        skip_vars.extend([name for name in scalar_names if name not in requested_scalars])
    else:
        skip_vars.extend(scalar_names)

    return skip_vars


def _extract_nek_fields(
    data,
    header,
    variables: Optional[Set[str]],
    include_coords: bool,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    elements = data.elem
    nb_geo = int(header.nb_vars[0]) if header.nb_vars else 0
    nb_vel = int(header.nb_vars[1]) if header.nb_vars else 0
    nb_pres = int(header.nb_vars[2]) if header.nb_vars else 0
    nb_temp = int(header.nb_vars[3]) if header.nb_vars else 0
    nb_scalars = int(header.nb_vars[4]) if header.nb_vars else 0

    fields: Dict[str, np.ndarray] = {}
    coords: Dict[str, np.ndarray] = {}

    if include_coords and nb_geo:
        coord_names = ("x", "y", "z")
        for idx in range(min(nb_geo, 3)):
            coords[coord_names[idx]] = _stack_component(elements, "pos", idx)

    vel_names = ("u", "v", "w")
    for idx in range(min(nb_vel, 3)):
        name = vel_names[idx]
        if variables is None or name in variables:
            fields[name] = _stack_component(elements, "vel", idx)

    if nb_pres and (variables is None or "p" in variables):
        fields["p"] = _stack_component(elements, "pres", 0)

    if nb_temp and (variables is None or "t" in variables):
        fields["t"] = _stack_component(elements, "temp", 0)

    if nb_scalars:
        for idx in range(nb_scalars):
            name = f"s{idx + 1:02d}"
            if variables is None or name in variables:
                fields[name] = _stack_component(elements, "scal", idx)

    return fields, coords


def _stack_component(elements, attr: str, idx: int) -> np.ndarray:
    return np.stack([getattr(elem, attr)[idx] for elem in elements], axis=0)
