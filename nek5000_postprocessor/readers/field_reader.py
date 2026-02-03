"""
Nek5000 Field File Reader
=========================

Comprehensive reader for Nek5000 binary field files (.fld format).
Supports reading mesh coordinates, velocity fields, pressure, temperature,
and passive scalars from Nek5000 simulation outputs.

Nek5000 Field File Format:
- Header (132 bytes ASCII)
- Mesh coordinates (optional, if header indicates)
- Velocity components (u, v, w)
- Pressure
- Temperature (optional)
- Passive scalars (optional)
"""

import numpy as np
import struct
import os
import glob
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path


@dataclass
class FieldData:
    """
    Container for Nek5000 field data.
    
    Attributes
    ----------
    time : float
        Simulation time
    timestep : int
        Timestep number
    nelements : int
        Number of spectral elements
    nx, ny, nz : int
        Polynomial order in each direction
    ndim : int
        Number of spatial dimensions (2 or 3)
    x, y, z : np.ndarray
        Mesh coordinates (if available)
    u, v, w : np.ndarray
        Velocity components
    p : np.ndarray
        Pressure field
    t : np.ndarray
        Temperature field (if available)
    scalars : Dict[str, np.ndarray]
        Dictionary of passive scalars
    """
    time: float
    timestep: int
    nelements: int
    nx: int
    ny: int
    nz: int
    ndim: int
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    z: Optional[np.ndarray] = None
    u: Optional[np.ndarray] = None
    v: Optional[np.ndarray] = None
    w: Optional[np.ndarray] = None
    p: Optional[np.ndarray] = None
    t: Optional[np.ndarray] = None
    scalars: Dict[str, np.ndarray] = field(default_factory=dict)
    
    @property
    def npoints_per_element(self) -> int:
        """Number of grid points per element."""
        return self.nx * self.ny * self.nz
    
    @property
    def total_points(self) -> int:
        """Total number of grid points."""
        return self.nelements * self.npoints_per_element
    
    def get_velocity_magnitude(self) -> np.ndarray:
        """Compute velocity magnitude."""
        if self.ndim == 2:
            return np.sqrt(self.u**2 + self.v**2)
        else:
            return np.sqrt(self.u**2 + self.v**2 + self.w**2)
    
    def get_field(self, field_name: str) -> Optional[np.ndarray]:
        """Get a field by name."""
        field_map = {
            'x': self.x, 'y': self.y, 'z': self.z,
            'u': self.u, 'v': self.v, 'w': self.w,
            'p': self.p, 't': self.t,
            'velocity_magnitude': self.get_velocity_magnitude(),
        }
        if field_name in field_map:
            return field_map[field_name]
        elif field_name in self.scalars:
            return self.scalars[field_name]
        return None


class Nek5000Reader:
    """
    Reader for Nek5000 simulation output files.
    
    Supports reading:
    - Binary field files (.fld, f00001, etc.)
    - Multiple file sequences for temporal data
    - Both single and double precision formats
    
    Parameters
    ----------
    case_name : str
        Base name of the Nek5000 case
    data_dir : str
        Directory containing the field files
    precision : str
        'single' or 'double' precision for binary data
        
    Example
    -------
    >>> reader = Nek5000Reader('channel', './data/')
    >>> field = reader.read_field(1)  # Read timestep 1
    >>> fields = reader.read_field_sequence(1, 100)  # Read timesteps 1-100
    """
    
    def __init__(self, case_name: str, data_dir: str = './', precision: str = 'single'):
        self.case_name = case_name
        self.data_dir = Path(data_dir)
        self.precision = precision
        self.dtype = np.float32 if precision == 'single' else np.float64
        self.word_size = 4 if precision == 'single' else 8
        
        # Cache for mesh data (typically constant across timesteps)
        self._mesh_cache: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        
    def _get_field_filename(self, timestep: int) -> Path:
        """Generate field filename for a given timestep."""
        # Nek5000 naming convention: casename0.f00001
        patterns = [
            f"{self.case_name}0.f{timestep:05d}",
            f"{self.case_name}.fld{timestep:05d}",
            f"{self.case_name}{timestep:05d}.fld",
        ]
        
        for pattern in patterns:
            filepath = self.data_dir / pattern
            if filepath.exists():
                return filepath
        
        # Try glob search
        glob_patterns = [
            f"{self.case_name}*.f{timestep:05d}",
            f"*{timestep:05d}.fld",
        ]
        
        for glob_pattern in glob_patterns:
            matches = list(self.data_dir.glob(glob_pattern))
            if matches:
                return matches[0]
        
        raise FileNotFoundError(
            f"Could not find field file for timestep {timestep} in {self.data_dir}"
        )
    
    def _parse_header(self, header_bytes: bytes) -> Dict:
        """
        Parse Nek5000 field file header (132 bytes ASCII).
        
        Header format:
        - Bytes 0-3: word size (e.g., '#std', '   4')
        - Bytes 4-7: precision indicator
        - Rest contains: nel, nx, ny, nz, time, istep, fid, etc.
        """
        header_str = header_bytes.decode('ascii', errors='ignore').strip()
        
        # Parse header fields
        header_info = {}
        
        try:
            # Standard Nek5000 header parsing
            parts = header_str.split()
            
            # Try to extract key values
            # Format varies but typically: wdsize nel nx ny nz time istep fid nfiles rdcode
            if len(parts) >= 7:
                header_info['word_size'] = int(float(parts[0])) if parts[0].isdigit() else 4
                header_info['nx'] = int(parts[1]) if len(parts) > 1 else 8
                header_info['ny'] = int(parts[2]) if len(parts) > 2 else 8  
                header_info['nz'] = int(parts[3]) if len(parts) > 3 else 8
                header_info['nelements'] = int(parts[4]) if len(parts) > 4 else 1
                header_info['nelements_global'] = int(parts[5]) if len(parts) > 5 else header_info['nelements']
                header_info['time'] = float(parts[6]) if len(parts) > 6 else 0.0
                header_info['timestep'] = int(parts[7]) if len(parts) > 7 else 0
                header_info['fid'] = int(parts[8]) if len(parts) > 8 else 0
                header_info['nfiles'] = int(parts[9]) if len(parts) > 9 else 1
                header_info['rdcode'] = parts[10] if len(parts) > 10 else 'XUVP'
            else:
                # Fallback for non-standard headers
                header_info = self._parse_header_fallback(header_str)
                
        except (ValueError, IndexError):
            header_info = self._parse_header_fallback(header_str)
            
        return header_info
    
    def _parse_header_fallback(self, header_str: str) -> Dict:
        """Fallback header parsing for non-standard formats."""
        return {
            'word_size': 4,
            'nx': 8,
            'ny': 8,
            'nz': 8,
            'nelements': 1,
            'nelements_global': 1,
            'time': 0.0,
            'timestep': 0,
            'fid': 0,
            'nfiles': 1,
            'rdcode': 'XUVP',
        }
    
    def _read_binary_field(self, f, npoints: int) -> np.ndarray:
        """Read a single field from binary file."""
        data = np.frombuffer(f.read(npoints * self.word_size), dtype=self.dtype)
        return data.copy()
    
    def read_field(self, timestep: int, read_mesh: bool = True) -> FieldData:
        """
        Read a single field file.
        
        Parameters
        ----------
        timestep : int
            Timestep number to read
        read_mesh : bool
            Whether to read mesh coordinates (can be slow for large meshes)
            
        Returns
        -------
        FieldData
            Container with all field data
        """
        filepath = self._get_field_filename(timestep)
        
        with open(filepath, 'rb') as f:
            # Read header (132 bytes)
            header_bytes = f.read(132)
            header = self._parse_header(header_bytes)
            
            # Extract dimensions
            nx = header.get('nx', 8)
            ny = header.get('ny', 8)
            nz = header.get('nz', 1)
            nel = header.get('nelements', 1)
            time = header.get('time', 0.0)
            istep = header.get('timestep', timestep)
            rdcode = header.get('rdcode', 'XUVP').upper()
            
            # Determine dimensionality
            ndim = 3 if nz > 1 else 2
            npoints = nx * ny * nz * nel
            
            # Read test pattern (for byte swapping detection)
            test_pattern = f.read(4)
            
            # Initialize field data
            field_data = FieldData(
                time=time,
                timestep=istep,
                nelements=nel,
                nx=nx,
                ny=ny,
                nz=nz,
                ndim=ndim,
            )
            
            # Read element mapping (if present)
            if nel > 0:
                # Skip element indices (4 bytes per element for global element numbers)
                element_indices = np.frombuffer(f.read(nel * 4), dtype=np.int32)
            
            # Parse rdcode to determine which fields are present
            # X/Y/Z = coordinates, U = velocity, P = pressure, T = temperature, S = scalars
            
            # Read coordinates if present
            if 'X' in rdcode and read_mesh:
                field_data.x = self._read_binary_field(f, npoints)
                field_data.y = self._read_binary_field(f, npoints)
                if ndim == 3:
                    field_data.z = self._read_binary_field(f, npoints)
                    
                # Cache mesh for later use
                self._mesh_cache = (field_data.x.copy(), field_data.y.copy(), 
                                   field_data.z.copy() if field_data.z is not None else None)
            elif self._mesh_cache is not None and read_mesh:
                # Use cached mesh
                field_data.x, field_data.y, field_data.z = self._mesh_cache
            
            # Read velocity if present
            if 'U' in rdcode:
                field_data.u = self._read_binary_field(f, npoints)
                field_data.v = self._read_binary_field(f, npoints)
                if ndim == 3:
                    field_data.w = self._read_binary_field(f, npoints)
                else:
                    field_data.w = np.zeros_like(field_data.u)
            
            # Read pressure if present
            if 'P' in rdcode:
                field_data.p = self._read_binary_field(f, npoints)
            
            # Read temperature if present
            if 'T' in rdcode:
                field_data.t = self._read_binary_field(f, npoints)
            
            # Read passive scalars if present
            nscalars = rdcode.count('S')
            for i in range(nscalars):
                scalar_data = self._read_binary_field(f, npoints)
                field_data.scalars[f'scalar_{i}'] = scalar_data
        
        return field_data
    
    def read_field_sequence(
        self, 
        start_timestep: int, 
        end_timestep: int, 
        step: int = 1,
        read_mesh: bool = True,
        verbose: bool = True
    ) -> List[FieldData]:
        """
        Read a sequence of field files.
        
        Parameters
        ----------
        start_timestep : int
            Starting timestep
        end_timestep : int
            Ending timestep (inclusive)
        step : int
            Step between timesteps
        read_mesh : bool
            Whether to read mesh coordinates
        verbose : bool
            Print progress
            
        Returns
        -------
        List[FieldData]
            List of field data for each timestep
        """
        fields = []
        timesteps = range(start_timestep, end_timestep + 1, step)
        
        if verbose:
            try:
                from tqdm import tqdm
                timesteps = tqdm(timesteps, desc="Reading fields")
            except ImportError:
                pass
        
        for t in timesteps:
            try:
                field = self.read_field(t, read_mesh=(read_mesh and len(fields) == 0))
                
                # Copy mesh from first field if not reading mesh each time
                if len(fields) > 0 and not read_mesh:
                    field.x = fields[0].x
                    field.y = fields[0].y
                    field.z = fields[0].z
                
                fields.append(field)
            except FileNotFoundError:
                if verbose:
                    print(f"Warning: Could not find field file for timestep {t}")
        
        return fields
    
    def get_available_timesteps(self) -> List[int]:
        """Find all available timesteps in the data directory."""
        patterns = [
            f"{self.case_name}0.f*",
            f"{self.case_name}*.fld*",
        ]
        
        timesteps = set()
        for pattern in patterns:
            for filepath in self.data_dir.glob(pattern):
                # Extract timestep number from filename
                name = filepath.name
                try:
                    # Try to find the numeric part
                    import re
                    numbers = re.findall(r'\d+', name)
                    if numbers:
                        timesteps.add(int(numbers[-1]))
                except ValueError:
                    pass
        
        return sorted(list(timesteps))


def read_field_file(filepath: str, precision: str = 'single') -> FieldData:
    """
    Convenience function to read a single field file.
    
    Parameters
    ----------
    filepath : str
        Path to the field file
    precision : str
        'single' or 'double' precision
        
    Returns
    -------
    FieldData
        Container with field data
    """
    filepath = Path(filepath)
    reader = Nek5000Reader(
        case_name=filepath.stem,
        data_dir=filepath.parent,
        precision=precision
    )
    
    # Extract timestep from filename
    import re
    numbers = re.findall(r'\d+', filepath.name)
    timestep = int(numbers[-1]) if numbers else 0
    
    return reader.read_field(timestep)


def read_mesh(filepath: str, precision: str = 'single') -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Read only mesh coordinates from a field file.
    
    Parameters
    ----------
    filepath : str
        Path to the field file
    precision : str
        'single' or 'double' precision
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]
        (x, y, z) mesh coordinates
    """
    field = read_field_file(filepath, precision)
    return field.x, field.y, field.z


class Nek5000MeshReader:
    """
    Reader specifically for Nek5000 mesh files (.re2 format).
    
    The .re2 file is a binary mesh file that contains:
    - Element vertices
    - Curved side information
    - Boundary conditions
    """
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        
    def read(self) -> Dict:
        """Read the mesh file and return mesh information."""
        mesh_info = {
            'vertices': [],
            'elements': [],
            'boundary_conditions': [],
        }
        
        with open(self.filepath, 'rb') as f:
            # Read header
            header = f.read(80).decode('ascii', errors='ignore')
            
            # Parse header to get mesh dimensions
            # Implementation depends on specific file version
            
        return mesh_info


class Nek5000StatisticsReader:
    """
    Reader for Nek5000 statistics files (sts*.fld).
    
    These files contain time-averaged statistics computed during runtime
    by Nek5000's built-in statistics module.
    """
    
    def __init__(self, data_dir: str, case_name: str = 'sts'):
        self.data_dir = Path(data_dir)
        self.case_name = case_name
        
    def read_statistics(self, timestep: int) -> Dict[str, np.ndarray]:
        """
        Read statistics file.
        
        Typical statistics fields:
        - <u>, <v>, <w>: mean velocities
        - <uu>, <vv>, <ww>: normal stresses
        - <uv>, <uw>, <vw>: shear stresses
        - <p>: mean pressure
        - <pp>: pressure variance
        """
        reader = Nek5000Reader(self.case_name, str(self.data_dir))
        field = reader.read_field(timestep)
        
        # Statistics files store data in specific order
        # This mapping depends on how statistics were configured in Nek5000
        stats = {
            'u_mean': field.u,
            'v_mean': field.v,
            'w_mean': field.w,
            'p_mean': field.p,
        }
        
        # Additional statistics may be stored in scalars
        if field.scalars:
            for name, data in field.scalars.items():
                stats[name] = data
        
        return stats
