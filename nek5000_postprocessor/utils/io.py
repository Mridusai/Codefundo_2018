"""
I/O Utilities
=============

File input/output utilities for saving and loading post-processing results.
"""

import numpy as np
from typing import Dict, Optional, Union
from pathlib import Path
import json
import warnings


def save_statistics(
    stats,
    filename: str,
    format: str = 'npz',
    compress: bool = True,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save statistics to file.
    
    Parameters
    ----------
    stats : FirstOrderStatistics, SecondOrderStatistics, or Dict
        Statistics to save
    filename : str
        Output filename
    format : str
        File format: 'npz', 'hdf5', or 'json'
    compress : bool
        Whether to compress (for npz format)
    metadata : Dict
        Additional metadata to include
    """
    # Convert stats to dictionary if needed
    if hasattr(stats, 'to_dict'):
        data = stats.to_dict()
    elif isinstance(stats, dict):
        data = stats
    else:
        raise TypeError(f"Cannot save stats of type {type(stats)}")
    
    # Add metadata
    if metadata:
        data['_metadata'] = metadata
    
    # Add type information
    data['_stats_type'] = type(stats).__name__ if hasattr(stats, '__class__') else 'dict'
    
    # Add attributes that aren't arrays
    for attr in ['n_samples', 'time_start', 'time_end', 'averaging_method']:
        if hasattr(stats, attr):
            data[f'_attr_{attr}'] = getattr(stats, attr)
    
    if format == 'npz':
        if compress:
            np.savez_compressed(filename, **data)
        else:
            np.savez(filename, **data)
        print(f"Saved statistics to {filename}")
        
    elif format == 'hdf5':
        try:
            import h5py
            with h5py.File(filename, 'w') as f:
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        f.create_dataset(key, data=value, compression='gzip' if compress else None)
                    elif isinstance(value, (int, float, str)):
                        f.attrs[key] = value
                    elif isinstance(value, dict):
                        # Store metadata as JSON string
                        f.attrs[key] = json.dumps(value)
            print(f"Saved statistics to {filename}")
        except ImportError:
            raise ImportError("h5py required for HDF5 format. Install with: pip install h5py")
            
    elif format == 'json':
        # Convert arrays to lists for JSON
        json_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            else:
                json_data[key] = value
        
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Saved statistics to {filename}")
        
    else:
        raise ValueError(f"Unknown format: {format}")


def load_statistics(filename: str, format: str = None) -> Dict[str, np.ndarray]:
    """
    Load statistics from file.
    
    Parameters
    ----------
    filename : str
        Input filename
    format : str
        File format (auto-detected if None)
        
    Returns
    -------
    Dict[str, np.ndarray]
        Loaded statistics
    """
    filepath = Path(filename)
    
    if format is None:
        suffix = filepath.suffix.lower()
        if suffix in ['.npz']:
            format = 'npz'
        elif suffix in ['.h5', '.hdf5']:
            format = 'hdf5'
        elif suffix in ['.json']:
            format = 'json'
        else:
            format = 'npz'  # Default
    
    if format == 'npz':
        data = dict(np.load(filename, allow_pickle=True))
        print(f"Loaded statistics from {filename}")
        return data
        
    elif format == 'hdf5':
        try:
            import h5py
            data = {}
            with h5py.File(filename, 'r') as f:
                for key in f.keys():
                    data[key] = f[key][:]
                for key in f.attrs.keys():
                    value = f.attrs[key]
                    try:
                        data[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        data[key] = value
            print(f"Loaded statistics from {filename}")
            return data
        except ImportError:
            raise ImportError("h5py required for HDF5 format")
            
    elif format == 'json':
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to arrays
        for key, value in data.items():
            if isinstance(value, list):
                data[key] = np.array(value)
        
        print(f"Loaded statistics from {filename}")
        return data
        
    else:
        raise ValueError(f"Unknown format: {format}")


def export_to_vtk(
    field_data,
    filename: str,
    field_names: Optional[list] = None
) -> None:
    """
    Export field data to VTK format for visualization in ParaView.
    
    Parameters
    ----------
    field_data : FieldData
        Field data with coordinates
    filename : str
        Output VTK filename
    field_names : list
        Fields to export (default: all available)
    """
    if field_data.x is None or field_data.y is None:
        raise ValueError("Coordinates required for VTK export")
    
    x = field_data.x.flatten()
    y = field_data.y.flatten()
    z = field_data.z.flatten() if field_data.z is not None else np.zeros_like(x)
    
    n_points = len(x)
    
    # Determine fields to export
    if field_names is None:
        field_names = []
        for name in ['u', 'v', 'w', 'p', 't']:
            if hasattr(field_data, name) and getattr(field_data, name) is not None:
                field_names.append(name)
    
    with open(filename, 'w') as f:
        # VTK header
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Nek5000 field data\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        
        # Points
        f.write(f"POINTS {n_points} float\n")
        for i in range(n_points):
            f.write(f"{x[i]} {y[i]} {z[i]}\n")
        
        # Cells (vertices)
        f.write(f"CELLS {n_points} {2*n_points}\n")
        for i in range(n_points):
            f.write(f"1 {i}\n")
        
        f.write(f"CELL_TYPES {n_points}\n")
        for i in range(n_points):
            f.write("1\n")  # VTK_VERTEX
        
        # Point data
        f.write(f"POINT_DATA {n_points}\n")
        
        for name in field_names:
            data = getattr(field_data, name, None)
            if data is not None:
                data = data.flatten()
                f.write(f"SCALARS {name} float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for val in data:
                    f.write(f"{val}\n")
    
    print(f"Exported to VTK: {filename}")


def export_to_hdf5(
    data: Dict[str, np.ndarray],
    filename: str,
    metadata: Optional[Dict] = None,
    compression: str = 'gzip'
) -> None:
    """
    Export data to HDF5 format.
    
    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Data arrays to export
    filename : str
        Output filename
    metadata : Dict
        Metadata to include
    compression : str
        Compression algorithm
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required for HDF5 export")
    
    with h5py.File(filename, 'w') as f:
        # Save arrays
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                f.create_dataset(key, data=value, compression=compression)
        
        # Save metadata
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (int, float, str)):
                    f.attrs[key] = value
                else:
                    f.attrs[key] = json.dumps(value)
    
    print(f"Exported to HDF5: {filename}")


def export_to_csv(
    stats,
    filename: str,
    y_coordinate: Optional[np.ndarray] = None,
    fields: Optional[list] = None
) -> None:
    """
    Export statistics to CSV format.
    
    Parameters
    ----------
    stats : FirstOrderStatistics or SecondOrderStatistics
        Statistics to export
    filename : str
        Output filename
    y_coordinate : np.ndarray
        Wall-normal coordinate
    fields : list
        Fields to include
    """
    if fields is None:
        if hasattr(stats, 'uu'):
            fields = ['uu', 'vv', 'ww', 'uv', 'tke']
        else:
            fields = ['u_mean', 'v_mean', 'w_mean']
    
    # Build data arrays
    data = {}
    
    if y_coordinate is not None:
        data['y'] = y_coordinate.flatten()
    
    for name in fields:
        value = stats.get_field(name) if hasattr(stats, 'get_field') else getattr(stats, name, None)
        if value is not None:
            data[name] = value.flatten()
    
    # Find common length
    lengths = [len(v) for v in data.values()]
    if len(set(lengths)) > 1:
        warnings.warn("Arrays have different lengths. Truncating to shortest.")
        min_len = min(lengths)
        data = {k: v[:min_len] for k, v in data.items()}
    
    # Write CSV
    with open(filename, 'w') as f:
        # Header
        f.write(','.join(data.keys()) + '\n')
        
        # Data rows
        n_rows = len(list(data.values())[0])
        for i in range(n_rows):
            row = [str(data[k][i]) for k in data.keys()]
            f.write(','.join(row) + '\n')
    
    print(f"Exported to CSV: {filename}")


def create_xdmf_file(
    hdf5_filename: str,
    xdmf_filename: str,
    field_names: list,
    n_points: int,
    time: float = 0.0
) -> None:
    """
    Create XDMF file for HDF5 data (for ParaView visualization).
    
    Parameters
    ----------
    hdf5_filename : str
        Associated HDF5 file
    xdmf_filename : str
        Output XDMF filename
    field_names : list
        Names of fields in HDF5
    n_points : int
        Number of data points
    time : float
        Time value for this snapshot
    """
    xdmf_content = f'''<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="nek5000_data" GridType="Uniform">
      <Time Value="{time}" />
      <Topology TopologyType="Polyvertex" NumberOfElements="{n_points}" />
      <Geometry GeometryType="XYZ">
        <DataItem Dimensions="{n_points} 3" NumberType="Float" Precision="4" Format="HDF">
          {hdf5_filename}:/coordinates
        </DataItem>
      </Geometry>
'''
    
    for name in field_names:
        xdmf_content += f'''      <Attribute Name="{name}" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{n_points}" NumberType="Float" Precision="4" Format="HDF">
          {hdf5_filename}:/{name}
        </DataItem>
      </Attribute>
'''
    
    xdmf_content += '''    </Grid>
  </Domain>
</Xdmf>
'''
    
    with open(xdmf_filename, 'w') as f:
        f.write(xdmf_content)
    
    print(f"Created XDMF file: {xdmf_filename}")
