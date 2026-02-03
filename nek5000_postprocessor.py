#!/usr/bin/env python3
"""
Complete Post-Processor for Nek5000 DNS Simulations
Calculates second-order statistics and temporal comparisons
for averaged variables from step change simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import FortranFile
import struct
import os
import glob
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json
import h5py


@dataclass
class SimulationData:
    """Container for simulation field data"""
    time: float
    velocity_x: np.ndarray
    velocity_y: np.ndarray
    velocity_z: np.ndarray
    pressure: np.ndarray
    temperature: Optional[np.ndarray] = None
    coordinates: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None


@dataclass
class StatisticsResult:
    """Container for statistical results"""
    mean: Dict[str, np.ndarray]
    variance: Dict[str, np.ndarray]
    std_dev: Dict[str, np.ndarray]
    rms: Dict[str, np.ndarray]
    reynolds_stress: Dict[str, np.ndarray]
    skewness: Dict[str, np.ndarray]
    kurtosis: Dict[str, np.ndarray]
    time_range: Tuple[float, float]
    n_samples: int


class Nek5000Reader:
    """Reader for Nek5000 binary output files (.fld, .f0xxxx)"""
    
    def __init__(self, case_name: str, data_dir: str = '.'):
        self.case_name = case_name
        self.data_dir = data_dir
        
    def read_field_file(self, filename: str) -> SimulationData:
        """
        Read a single Nek5000 field file
        Supports both ASCII and binary formats
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found")
        
        # Try to detect file format
        with open(filepath, 'rb') as f:
            header = f.read(132)
            
        # Check if binary format
        if b'#std' in header[:4] or b'#std' in header[:10]:
            return self._read_binary_field(filepath)
        else:
            return self._read_ascii_field(filepath)
    
    def _read_binary_field(self, filepath: str) -> SimulationData:
        """Read binary format Nek5000 field file"""
        with open(filepath, 'rb') as f:
            # Read header
            header = f.read(132).decode('utf-8', errors='ignore')
            
            # Parse header
            wdsizo = struct.unpack('i', f.read(4))[0]
            
            # Read number of elements
            nelgt = struct.unpack('i', f.read(4))[0]
            nelgv = struct.unpack('i', f.read(4))[0]
            
            # Read time
            time = struct.unpack('d', f.read(8))[0]
            
            # Read element dimensions
            nx, ny, nz = struct.unpack('iii', f.read(12))
            
            # Read element IDs
            nelo = struct.unpack('i', f.read(4))[0]
            
            # Initialize arrays
            nel = nelgv
            nxyz = nx * ny * nz
            
            # Read coordinate data
            x = np.zeros((nel, nxyz))
            y = np.zeros((nel, nxyz))
            z = np.zeros((nel, nxyz))
            
            for e in range(nel):
                for i in range(nxyz):
                    if wdsizo == 4:
                        x[e, i] = struct.unpack('f', f.read(4))[0]
                    else:
                        x[e, i] = struct.unpack('d', f.read(8))[0]
                        
            for e in range(nel):
                for i in range(nxyz):
                    if wdsizo == 4:
                        y[e, i] = struct.unpack('f', f.read(4))[0]
                    else:
                        y[e, i] = struct.unpack('d', f.read(8))[0]
                        
            if nz > 1:
                for e in range(nel):
                    for i in range(nxyz):
                        if wdsizo == 4:
                            z[e, i] = struct.unpack('f', f.read(4))[0]
                        else:
                            z[e, i] = struct.unpack('d', f.read(8))[0]
            
            # Read velocity fields
            u = np.zeros((nel, nxyz))
            v = np.zeros((nel, nxyz))
            w = np.zeros((nel, nxyz))
            
            for e in range(nel):
                for i in range(nxyz):
                    if wdsizo == 4:
                        u[e, i] = struct.unpack('f', f.read(4))[0]
                    else:
                        u[e, i] = struct.unpack('d', f.read(8))[0]
                        
            for e in range(nel):
                for i in range(nxyz):
                    if wdsizo == 4:
                        v[e, i] = struct.unpack('f', f.read(4))[0]
                    else:
                        v[e, i] = struct.unpack('d', f.read(8))[0]
                        
            if nz > 1:
                for e in range(nel):
                    for i in range(nxyz):
                        if wdsizo == 4:
                            w[e, i] = struct.unpack('f', f.read(4))[0]
                        else:
                            w[e, i] = struct.unpack('d', f.read(8))[0]
            
            # Read pressure
            p = np.zeros((nel, nxyz))
            for e in range(nel):
                for i in range(nxyz):
                    if wdsizo == 4:
                        p[e, i] = struct.unpack('f', f.read(4))[0]
                    else:
                        p[e, i] = struct.unpack('d', f.read(8))[0]
            
            # Try to read temperature if available
            try:
                T = np.zeros((nel, nxyz))
                for e in range(nel):
                    for i in range(nxyz):
                        if wdsizo == 4:
                            T[e, i] = struct.unpack('f', f.read(4))[0]
                        else:
                            T[e, i] = struct.unpack('d', f.read(8))[0]
            except:
                T = None
        
        # Flatten arrays
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = z.flatten()
        u_flat = u.flatten()
        v_flat = v.flatten()
        w_flat = w.flatten()
        p_flat = p.flatten()
        T_flat = T.flatten() if T is not None else None
        
        return SimulationData(
            time=time,
            velocity_x=u_flat,
            velocity_y=v_flat,
            velocity_z=w_flat,
            pressure=p_flat,
            temperature=T_flat,
            coordinates=(x_flat, y_flat, z_flat)
        )
    
    def _read_ascii_field(self, filepath: str) -> SimulationData:
        """Read ASCII format Nek5000 field file (simplified)"""
        # This is a simplified reader - actual implementation depends on format
        raise NotImplementedError("ASCII reader requires specific format details")
    
    def get_field_files(self, pattern: str = None) -> List[str]:
        """Get list of field files matching pattern"""
        if pattern is None:
            pattern = f"{self.case_name}0.f*"
        
        files = sorted(glob.glob(os.path.join(self.data_dir, pattern)))
        return files


class StatisticsCalculator:
    """Calculate first and second-order statistics from DNS data"""
    
    def __init__(self):
        self.data_samples: List[SimulationData] = []
        
    def add_sample(self, data: SimulationData):
        """Add a time sample to the statistics calculator"""
        self.data_samples.append(data)
    
    def calculate_statistics(self, variables: List[str] = None) -> StatisticsResult:
        """
        Calculate comprehensive statistics from collected samples
        
        Args:
            variables: List of variables to compute statistics for
                      ['u', 'v', 'w', 'p', 'T']
        
        Returns:
            StatisticsResult object containing all statistics
        """
        if not self.data_samples:
            raise ValueError("No data samples available for statistics calculation")
        
        if variables is None:
            variables = ['u', 'v', 'w', 'p']
        
        n_samples = len(self.data_samples)
        n_points = len(self.data_samples[0].velocity_x)
        
        # Initialize arrays for statistics
        means = {}
        variances = {}
        std_devs = {}
        rms_values = {}
        skewness = {}
        kurtosis = {}
        
        # Variable mapping
        var_map = {
            'u': 'velocity_x',
            'v': 'velocity_y',
            'w': 'velocity_z',
            'p': 'pressure',
            'T': 'temperature'
        }
        
        print(f"Calculating statistics from {n_samples} samples...")
        
        for var in variables:
            if var not in var_map:
                continue
                
            attr_name = var_map[var]
            
            # Collect all samples for this variable
            samples = np.zeros((n_samples, n_points))
            for i, data in enumerate(self.data_samples):
                field = getattr(data, attr_name)
                if field is None:
                    continue
                samples[i, :] = field
            
            # Calculate first-order statistics
            mean = np.mean(samples, axis=0)
            variance = np.var(samples, axis=0)
            std_dev = np.std(samples, axis=0)
            rms = np.sqrt(np.mean(samples**2, axis=0))
            
            # Calculate higher-order statistics
            # Fluctuations
            fluctuations = samples - mean
            
            # Third moment (skewness)
            skew = np.mean(fluctuations**3, axis=0) / (std_dev**3 + 1e-10)
            
            # Fourth moment (kurtosis)
            kurt = np.mean(fluctuations**4, axis=0) / (std_dev**4 + 1e-10)
            
            means[var] = mean
            variances[var] = variance
            std_devs[var] = std_dev
            rms_values[var] = rms
            skewness[var] = skew
            kurtosis[var] = kurt
        
        # Calculate Reynolds stresses (second-order correlations)
        reynolds_stress = self._calculate_reynolds_stress()
        
        # Get time range
        times = [data.time for data in self.data_samples]
        time_range = (min(times), max(times))
        
        return StatisticsResult(
            mean=means,
            variance=variances,
            std_dev=std_devs,
            rms=rms_values,
            reynolds_stress=reynolds_stress,
            skewness=skewness,
            kurtosis=kurtosis,
            time_range=time_range,
            n_samples=n_samples
        )
    
    def _calculate_reynolds_stress(self) -> Dict[str, np.ndarray]:
        """Calculate Reynolds stress tensor components"""
        n_samples = len(self.data_samples)
        n_points = len(self.data_samples[0].velocity_x)
        
        # Collect velocity samples
        u_samples = np.array([data.velocity_x for data in self.data_samples])
        v_samples = np.array([data.velocity_y for data in self.data_samples])
        w_samples = np.array([data.velocity_z for data in self.data_samples])
        
        # Calculate means
        u_mean = np.mean(u_samples, axis=0)
        v_mean = np.mean(v_samples, axis=0)
        w_mean = np.mean(w_samples, axis=0)
        
        # Calculate fluctuations
        u_prime = u_samples - u_mean
        v_prime = v_samples - v_mean
        w_prime = w_samples - w_mean
        
        # Calculate Reynolds stress components
        reynolds_stress = {
            "u'u'": np.mean(u_prime * u_prime, axis=0),
            "v'v'": np.mean(v_prime * v_prime, axis=0),
            "w'w'": np.mean(w_prime * w_prime, axis=0),
            "u'v'": np.mean(u_prime * v_prime, axis=0),
            "u'w'": np.mean(u_prime * w_prime, axis=0),
            "v'w'": np.mean(v_prime * w_prime, axis=0),
        }
        
        # Calculate turbulent kinetic energy
        reynolds_stress['TKE'] = 0.5 * (reynolds_stress["u'u'"] + 
                                        reynolds_stress["v'v'"] + 
                                        reynolds_stress["w'w'"])
        
        return reynolds_stress
    
    def calculate_spatial_average(self, variable: str) -> np.ndarray:
        """Calculate spatial average of a variable over time"""
        var_map = {
            'u': 'velocity_x',
            'v': 'velocity_y',
            'w': 'velocity_z',
            'p': 'pressure',
            'T': 'temperature'
        }
        
        if variable not in var_map:
            raise ValueError(f"Unknown variable: {variable}")
        
        attr_name = var_map[variable]
        
        spatial_avgs = []
        times = []
        
        for data in self.data_samples:
            field = getattr(data, attr_name)
            if field is not None:
                spatial_avgs.append(np.mean(field))
                times.append(data.time)
        
        return np.array(times), np.array(spatial_avgs)


class TemporalComparator:
    """Compare temporal evolution of averaged variables"""
    
    def __init__(self):
        self.time_windows: Dict[str, StatisticsResult] = {}
        self.window_labels: List[str] = []
        
    def add_time_window(self, label: str, stats: StatisticsResult):
        """Add statistics from a specific time window"""
        self.time_windows[label] = stats
        self.window_labels.append(label)
    
    def compare_means(self, variable: str, spatial_location: int = None) -> Dict:
        """
        Compare mean values across time windows
        
        Args:
            variable: Variable to compare ('u', 'v', 'w', 'p', 'T')
            spatial_location: Specific point index, or None for spatial average
        """
        comparison = {
            'labels': self.window_labels,
            'values': [],
            'relative_change': []
        }
        
        for label in self.window_labels:
            stats = self.time_windows[label]
            if variable not in stats.mean:
                continue
                
            if spatial_location is not None:
                value = stats.mean[variable][spatial_location]
            else:
                value = np.mean(stats.mean[variable])
            
            comparison['values'].append(value)
        
        # Calculate relative changes
        if len(comparison['values']) > 1:
            baseline = comparison['values'][0]
            for val in comparison['values']:
                rel_change = (val - baseline) / (abs(baseline) + 1e-10) * 100
                comparison['relative_change'].append(rel_change)
        
        return comparison
    
    def compare_variances(self, variable: str, spatial_location: int = None) -> Dict:
        """Compare variance/turbulence intensity across time windows"""
        comparison = {
            'labels': self.window_labels,
            'variance': [],
            'turbulence_intensity': []
        }
        
        for label in self.window_labels:
            stats = self.time_windows[label]
            if variable not in stats.variance:
                continue
                
            if spatial_location is not None:
                var = stats.variance[variable][spatial_location]
                mean = stats.mean[variable][spatial_location]
            else:
                var = np.mean(stats.variance[variable])
                mean = np.mean(stats.mean[variable])
            
            comparison['variance'].append(var)
            
            # Turbulence intensity
            ti = np.sqrt(var) / (abs(mean) + 1e-10)
            comparison['turbulence_intensity'].append(ti)
        
        return comparison
    
    def compare_reynolds_stress(self, component: str = 'TKE', 
                                spatial_location: int = None) -> Dict:
        """Compare Reynolds stress components across time windows"""
        comparison = {
            'labels': self.window_labels,
            'values': []
        }
        
        for label in self.window_labels:
            stats = self.time_windows[label]
            if component not in stats.reynolds_stress:
                continue
                
            if spatial_location is not None:
                value = stats.reynolds_stress[component][spatial_location]
            else:
                value = np.mean(stats.reynolds_stress[component])
            
            comparison['values'].append(value)
        
        return comparison
    
    def detect_step_change_response(self, variable: str, 
                                   threshold: float = 0.05) -> Dict:
        """
        Detect and characterize response to step change
        
        Args:
            variable: Variable to analyze
            threshold: Threshold for detecting settling (relative change)
        
        Returns:
            Dictionary with response characteristics
        """
        if len(self.window_labels) < 2:
            return {"error": "Need at least 2 time windows"}
        
        comparison = self.compare_means(variable)
        values = comparison['values']
        
        if len(values) < 2:
            return {"error": "Insufficient data"}
        
        # Calculate response metrics
        initial = values[0]
        final = values[-1]
        step_magnitude = final - initial
        
        # Find settling time (when change is within threshold)
        settled_idx = None
        for i in range(1, len(values)):
            rel_change = abs(values[i] - final) / (abs(final) + 1e-10)
            if rel_change < threshold:
                settled_idx = i
                break
        
        # Find overshoot
        if step_magnitude > 0:
            overshoot = max(values) - final
        else:
            overshoot = final - min(values)
        
        overshoot_percent = overshoot / (abs(step_magnitude) + 1e-10) * 100
        
        return {
            'initial_value': initial,
            'final_value': final,
            'step_magnitude': step_magnitude,
            'overshoot': overshoot,
            'overshoot_percent': overshoot_percent,
            'settled_window_index': settled_idx,
            'settling_label': self.window_labels[settled_idx] if settled_idx else None
        }


class PostProcessor:
    """Main post-processor class coordinating all operations"""
    
    def __init__(self, case_name: str, data_dir: str = '.', output_dir: str = './results'):
        self.case_name = case_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.reader = Nek5000Reader(case_name, data_dir)
        self.stats_calc = StatisticsCalculator()
        self.temporal_comp = TemporalComparator()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_time_range(self, start_time: float = None, end_time: float = None,
                       file_pattern: str = None):
        """Load all field files in a time range"""
        files = self.reader.get_field_files(file_pattern)
        
        print(f"Found {len(files)} field files")
        
        loaded_count = 0
        for ffile in files:
            try:
                data = self.reader.read_field_file(ffile)
                
                # Check time range
                if start_time is not None and data.time < start_time:
                    continue
                if end_time is not None and data.time > end_time:
                    continue
                
                self.stats_calc.add_sample(data)
                loaded_count += 1
                
                if loaded_count % 10 == 0:
                    print(f"Loaded {loaded_count} files...")
                    
            except Exception as e:
                print(f"Warning: Could not read {ffile}: {e}")
                continue
        
        print(f"Successfully loaded {loaded_count} field files")
        return loaded_count
    
    def compute_global_statistics(self, variables: List[str] = None) -> StatisticsResult:
        """Compute statistics over all loaded data"""
        print("\nComputing global statistics...")
        stats = self.stats_calc.calculate_statistics(variables)
        
        # Save statistics to file
        self._save_statistics(stats, 'global_statistics')
        
        return stats
    
    def compute_windowed_statistics(self, time_windows: List[Tuple[float, float, str]],
                                   variables: List[str] = None):
        """
        Compute statistics for multiple time windows
        
        Args:
            time_windows: List of (start_time, end_time, label) tuples
            variables: Variables to analyze
        """
        for start_time, end_time, label in time_windows:
            print(f"\nProcessing time window: {label} ({start_time} - {end_time})")
            
            # Filter samples for this window
            window_calc = StatisticsCalculator()
            for data in self.stats_calc.data_samples:
                if start_time <= data.time <= end_time:
                    window_calc.add_sample(data)
            
            if len(window_calc.data_samples) == 0:
                print(f"Warning: No samples in window {label}")
                continue
            
            # Calculate statistics
            stats = window_calc.calculate_statistics(variables)
            
            # Add to temporal comparator
            self.temporal_comp.add_time_window(label, stats)
            
            # Save windowed statistics
            self._save_statistics(stats, f'statistics_{label}')
    
    def analyze_temporal_evolution(self, variables: List[str] = ['u', 'v', 'w', 'p']):
        """Analyze temporal evolution of spatial averages"""
        print("\nAnalyzing temporal evolution...")
        
        evolution_data = {}
        
        for var in variables:
            try:
                times, spatial_avgs = self.stats_calc.calculate_spatial_average(var)
                evolution_data[var] = {
                    'time': times.tolist(),
                    'spatial_average': spatial_avgs.tolist()
                }
            except Exception as e:
                print(f"Warning: Could not analyze {var}: {e}")
        
        # Save temporal evolution data
        output_file = os.path.join(self.output_dir, 'temporal_evolution.json')
        with open(output_file, 'w') as f:
            json.dump(evolution_data, f, indent=2)
        
        print(f"Saved temporal evolution to {output_file}")
        
        return evolution_data
    
    def compare_time_windows(self, variables: List[str] = ['u', 'v', 'w', 'p']):
        """Compare statistics across time windows"""
        print("\nComparing time windows...")
        
        comparison_results = {}
        
        for var in variables:
            comparison_results[var] = {
                'mean_comparison': self.temporal_comp.compare_means(var),
                'variance_comparison': self.temporal_comp.compare_variances(var),
            }
        
        # Compare Reynolds stresses
        comparison_results['reynolds_stress'] = {
            'TKE': self.temporal_comp.compare_reynolds_stress('TKE'),
            "u'u'": self.temporal_comp.compare_reynolds_stress("u'u'"),
            "v'v'": self.temporal_comp.compare_reynolds_stress("v'v'"),
            "w'w'": self.temporal_comp.compare_reynolds_stress("w'w'"),
            "u'v'": self.temporal_comp.compare_reynolds_stress("u'v'"),
        }
        
        # Save comparison results
        output_file = os.path.join(self.output_dir, 'window_comparison.json')
        with open(output_file, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        print(f"Saved window comparison to {output_file}")
        
        return comparison_results
    
    def analyze_step_response(self, variables: List[str] = ['u', 'v', 'w']):
        """Analyze response to step change"""
        print("\nAnalyzing step change response...")
        
        response_analysis = {}
        
        for var in variables:
            response = self.temporal_comp.detect_step_change_response(var)
            response_analysis[var] = response
        
        # Save response analysis
        output_file = os.path.join(self.output_dir, 'step_response_analysis.json')
        with open(output_file, 'w') as f:
            json.dump(response_analysis, f, indent=2)
        
        print(f"Saved step response analysis to {output_file}")
        
        return response_analysis
    
    def _save_statistics(self, stats: StatisticsResult, prefix: str):
        """Save statistics to HDF5 and JSON files"""
        # Save to HDF5 (for large arrays)
        h5_file = os.path.join(self.output_dir, f'{prefix}.h5')
        with h5py.File(h5_file, 'w') as f:
            # Save means
            mean_grp = f.create_group('mean')
            for var, data in stats.mean.items():
                mean_grp.create_dataset(var, data=data)
            
            # Save variances
            var_grp = f.create_group('variance')
            for var, data in stats.variance.items():
                var_grp.create_dataset(var, data=data)
            
            # Save standard deviations
            std_grp = f.create_group('std_dev')
            for var, data in stats.std_dev.items():
                std_grp.create_dataset(var, data=data)
            
            # Save RMS
            rms_grp = f.create_group('rms')
            for var, data in stats.rms.items():
                rms_grp.create_dataset(var, data=data)
            
            # Save skewness
            skew_grp = f.create_group('skewness')
            for var, data in stats.skewness.items():
                skew_grp.create_dataset(var, data=data)
            
            # Save kurtosis
            kurt_grp = f.create_group('kurtosis')
            for var, data in stats.kurtosis.items():
                kurt_grp.create_dataset(var, data=data)
            
            # Save Reynolds stress
            rey_grp = f.create_group('reynolds_stress')
            for component, data in stats.reynolds_stress.items():
                rey_grp.create_dataset(component, data=data)
            
            # Save metadata
            f.attrs['time_start'] = stats.time_range[0]
            f.attrs['time_end'] = stats.time_range[1]
            f.attrs['n_samples'] = stats.n_samples
        
        print(f"Saved statistics to {h5_file}")
        
        # Save summary to JSON
        summary = {
            'time_range': stats.time_range,
            'n_samples': stats.n_samples,
            'variables': list(stats.mean.keys()),
            'spatial_averages': {
                'mean': {var: float(np.mean(data)) for var, data in stats.mean.items()},
                'variance': {var: float(np.mean(data)) for var, data in stats.variance.items()},
                'rms': {var: float(np.mean(data)) for var, data in stats.rms.items()},
            },
            'reynolds_stress_avg': {
                comp: float(np.mean(data)) for comp, data in stats.reynolds_stress.items()
            }
        }
        
        json_file = os.path.join(self.output_dir, f'{prefix}_summary.json')
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Saved summary to {json_file}")
    
    def generate_plots(self, stats: StatisticsResult = None):
        """Generate visualization plots"""
        print("\nGenerating plots...")
        
        if stats is None and len(self.stats_calc.data_samples) > 0:
            stats = self.stats_calc.calculate_statistics()
        
        if stats is None:
            print("No statistics available for plotting")
            return
        
        # Plot temporal evolution
        self._plot_temporal_evolution()
        
        # Plot statistics profiles (if spatial data available)
        if stats.mean:
            self._plot_statistics_profiles(stats)
        
        # Plot Reynolds stress
        self._plot_reynolds_stress(stats)
        
        # Plot window comparisons
        if len(self.temporal_comp.window_labels) > 1:
            self._plot_window_comparisons()
    
    def _plot_temporal_evolution(self):
        """Plot temporal evolution of spatially averaged variables"""
        variables = ['u', 'v', 'w', 'p']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, var in enumerate(variables):
            try:
                times, spatial_avgs = self.stats_calc.calculate_spatial_average(var)
                axes[idx].plot(times, spatial_avgs, 'b-', linewidth=1.5)
                axes[idx].set_xlabel('Time', fontsize=12)
                axes[idx].set_ylabel(f'<{var}>', fontsize=12)
                axes[idx].set_title(f'Temporal Evolution of {var}', fontsize=13)
                axes[idx].grid(True, alpha=0.3)
            except Exception as e:
                print(f"Could not plot {var}: {e}")
        
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, 'temporal_evolution.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved temporal evolution plot to {output_file}")
    
    def _plot_statistics_profiles(self, stats: StatisticsResult):
        """Plot spatial profiles of statistics"""
        # This is a simplified version - actual implementation would need spatial coordinates
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        variables = ['u', 'v', 'w', 'p']
        
        for idx, var in enumerate(variables[:4]):
            if var not in stats.mean:
                continue
            
            mean = stats.mean[var]
            std = stats.std_dev[var]
            
            x = np.arange(len(mean))
            
            axes[idx].plot(x, mean, 'b-', label='Mean', linewidth=1.5)
            axes[idx].fill_between(x, mean - std, mean + std, 
                                  alpha=0.3, label='±1 std dev')
            axes[idx].set_xlabel('Point Index', fontsize=12)
            axes[idx].set_ylabel(var, fontsize=12)
            axes[idx].set_title(f'Statistics Profile: {var}', fontsize=13)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, 'statistics_profiles.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved statistics profiles to {output_file}")
    
    def _plot_reynolds_stress(self, stats: StatisticsResult):
        """Plot Reynolds stress components"""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        components = ["u'u'", "v'v'", "w'w'", "u'v'", "u'w'", "v'w'"]
        
        for idx, comp in enumerate(components):
            if comp not in stats.reynolds_stress:
                continue
            
            data = stats.reynolds_stress[comp]
            x = np.arange(len(data))
            
            axes[idx].plot(x, data, 'r-', linewidth=1.5)
            axes[idx].set_xlabel('Point Index', fontsize=12)
            axes[idx].set_ylabel(comp, fontsize=12)
            axes[idx].set_title(f'Reynolds Stress: {comp}', fontsize=13)
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, 'reynolds_stress.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved Reynolds stress plot to {output_file}")
    
    def _plot_window_comparisons(self):
        """Plot comparisons across time windows"""
        variables = ['u', 'v', 'w', 'p']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, var in enumerate(variables):
            # Compare means
            comp_mean = self.temporal_comp.compare_means(var)
            
            if comp_mean['values']:
                x_pos = np.arange(len(comp_mean['labels']))
                axes[idx].bar(x_pos, comp_mean['values'], alpha=0.7)
                axes[idx].set_xticks(x_pos)
                axes[idx].set_xticklabels(comp_mean['labels'], rotation=45)
                axes[idx].set_ylabel(f'Mean {var}', fontsize=12)
                axes[idx].set_title(f'Window Comparison: {var}', fontsize=13)
                axes[idx].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, 'window_comparison.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved window comparison plot to {output_file}")
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        report_file = os.path.join(self.output_dir, 'analysis_report.txt')
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("NEK5000 DNS POST-PROCESSING REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Case Name: {self.case_name}\n")
            f.write(f"Data Directory: {self.data_dir}\n")
            f.write(f"Output Directory: {self.output_dir}\n")
            f.write(f"Total Samples: {len(self.stats_calc.data_samples)}\n\n")
            
            if self.stats_calc.data_samples:
                times = [d.time for d in self.stats_calc.data_samples]
                f.write(f"Time Range: {min(times):.6f} - {max(times):.6f}\n")
                f.write(f"Time Step (avg): {np.mean(np.diff(times)):.6f}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("TIME WINDOW ANALYSIS\n")
            f.write("-" * 80 + "\n\n")
            
            for label in self.temporal_comp.window_labels:
                stats = self.temporal_comp.time_windows[label]
                f.write(f"\nWindow: {label}\n")
                f.write(f"  Time Range: {stats.time_range[0]:.6f} - {stats.time_range[1]:.6f}\n")
                f.write(f"  Samples: {stats.n_samples}\n")
                f.write(f"  Variables: {', '.join(stats.mean.keys())}\n")
                
                f.write(f"\n  Spatial Averages:\n")
                for var in stats.mean.keys():
                    mean_avg = np.mean(stats.mean[var])
                    std_avg = np.mean(stats.std_dev[var])
                    f.write(f"    {var}: mean = {mean_avg:.6e}, std = {std_avg:.6e}\n")
                
                if 'TKE' in stats.reynolds_stress:
                    tke_avg = np.mean(stats.reynolds_stress['TKE'])
                    f.write(f"  TKE (avg): {tke_avg:.6e}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("Analysis complete. See additional output files for detailed results.\n")
            f.write("=" * 80 + "\n")
        
        print(f"\nGenerated comprehensive report: {report_file}")


def main():
    """Example usage of the post-processor"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Nek5000 DNS Post-Processor with Second-Order Statistics'
    )
    parser.add_argument('case_name', help='Case name (prefix of field files)')
    parser.add_argument('--data-dir', default='.', help='Directory containing field files')
    parser.add_argument('--output-dir', default='./results', help='Output directory')
    parser.add_argument('--start-time', type=float, help='Start time for analysis')
    parser.add_argument('--end-time', type=float, help='End time for analysis')
    parser.add_argument('--variables', nargs='+', default=['u', 'v', 'w', 'p'],
                       help='Variables to analyze')
    parser.add_argument('--time-windows', type=str,
                       help='JSON file with time window definitions')
    
    args = parser.parse_args()
    
    # Initialize post-processor
    pp = PostProcessor(args.case_name, args.data_dir, args.output_dir)
    
    # Load data
    print("Loading field files...")
    pp.load_time_range(args.start_time, args.end_time)
    
    # Compute global statistics
    global_stats = pp.compute_global_statistics(args.variables)
    
    # Analyze temporal evolution
    pp.analyze_temporal_evolution(args.variables)
    
    # Process time windows if specified
    if args.time_windows:
        with open(args.time_windows, 'r') as f:
            windows_config = json.load(f)
        
        time_windows = [
            (w['start'], w['end'], w['label']) 
            for w in windows_config['windows']
        ]
        
        pp.compute_windowed_statistics(time_windows, args.variables)
        pp.compare_time_windows(args.variables)
        pp.analyze_step_response(args.variables)
    
    # Generate plots
    pp.generate_plots(global_stats)
    
    # Generate report
    pp.generate_report()
    
    print("\n" + "=" * 80)
    print("POST-PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
