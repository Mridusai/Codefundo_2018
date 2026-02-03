#!/usr/bin/env python3
"""
Nek5000 DNS Post-Processor
==========================

Command-line interface for post-processing Nek5000 DNS simulations.

Usage:
    python postprocess.py --case channel --data-dir ./output --start 1 --end 1000
    python postprocess.py --config config.yaml
"""

import argparse
import yaml
import sys
import os
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from readers import Nek5000Reader, FieldData
from statistics import (
    FirstOrderStatistics, SecondOrderStatistics,
    compute_mean_fields, compute_reynolds_stresses, compute_tke
)
from statistics.first_order import OnlineMeanComputer, compute_rms
from statistics.second_order import OnlineSecondOrderComputer, compute_higher_order_moments
from temporal import (
    TemporalAverager, TemporalComparison,
    running_average, windowed_statistics,
    compare_statistics, StepChangeAnalyzer
)
from utils.io import save_statistics, export_to_csv, export_to_vtk
from utils.physics import (
    compute_friction_velocity, compute_wall_units,
    compute_bulk_velocity, compute_reynolds_number
)


def load_config(config_file: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def create_default_config() -> Dict:
    """Create default configuration."""
    return {
        'case_name': 'nek5000',
        'data_dir': './',
        'output_dir': './postprocessing_results',
        'timesteps': {
            'start': 1,
            'end': 100,
            'step': 1,
        },
        'precision': 'single',
        'statistics': {
            'compute_first_order': True,
            'compute_second_order': True,
            'compute_higher_order': False,
        },
        'temporal': {
            'enable': True,
            'step_change_time': None,
            'window_type': 'cumulative',
        },
        'output': {
            'save_statistics': True,
            'export_vtk': False,
            'export_csv': True,
            'formats': ['npz'],
        },
        'physics': {
            'nu': 1e-5,
            'half_height': 1.0,
        }
    }


class Nek5000PostProcessor:
    """
    Main post-processor class for Nek5000 DNS simulations.
    
    Parameters
    ----------
    config : Dict
        Configuration dictionary
        
    Example
    -------
    >>> config = load_config('config.yaml')
    >>> processor = Nek5000PostProcessor(config)
    >>> processor.run()
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.reader = Nek5000Reader(
            case_name=config['case_name'],
            data_dir=config['data_dir'],
            precision=config.get('precision', 'single')
        )
        
        # Create output directory
        self.output_dir = Path(config.get('output_dir', './postprocessing_results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.first_order_stats: Optional[FirstOrderStatistics] = None
        self.second_order_stats: Optional[SecondOrderStatistics] = None
        self.temporal_comparison: Optional[TemporalComparison] = None
        self.fields: List[FieldData] = []
        
    def run(self) -> Dict:
        """
        Run the complete post-processing pipeline.
        
        Returns
        -------
        Dict
            Summary of results
        """
        print("=" * 60)
        print("Nek5000 DNS Post-Processor")
        print("=" * 60)
        
        # Step 1: Read field data
        print("\n[1/5] Reading field data...")
        self._read_fields()
        
        # Step 2: Compute first-order statistics
        if self.config['statistics'].get('compute_first_order', True):
            print("\n[2/5] Computing first-order statistics...")
            self._compute_first_order_statistics()
        
        # Step 3: Compute second-order statistics
        if self.config['statistics'].get('compute_second_order', True):
            print("\n[3/5] Computing second-order statistics...")
            self._compute_second_order_statistics()
        
        # Step 4: Temporal analysis
        if self.config['temporal'].get('enable', True):
            print("\n[4/5] Performing temporal analysis...")
            self._temporal_analysis()
        
        # Step 5: Save results
        print("\n[5/5] Saving results...")
        self._save_results()
        
        # Generate summary
        summary = self._generate_summary()
        
        print("\n" + "=" * 60)
        print("Post-processing complete!")
        print("=" * 60)
        
        return summary
    
    def _read_fields(self) -> None:
        """Read field files."""
        timesteps = self.config['timesteps']
        start = timesteps['start']
        end = timesteps['end']
        step = timesteps.get('step', 1)
        
        print(f"  Reading timesteps {start} to {end} (step={step})...")
        
        self.fields = self.reader.read_field_sequence(
            start_timestep=start,
            end_timestep=end,
            step=step,
            read_mesh=True,
            verbose=True
        )
        
        if len(self.fields) == 0:
            raise RuntimeError("No field files found!")
        
        print(f"  Loaded {len(self.fields)} field snapshots")
        print(f"  Time range: {self.fields[0].time:.4f} to {self.fields[-1].time:.4f}")
        print(f"  Grid points: {self.fields[0].total_points}")
    
    def _compute_first_order_statistics(self) -> None:
        """Compute first-order (mean) statistics."""
        print("  Computing mean velocity and pressure fields...")
        
        self.first_order_stats = compute_mean_fields(self.fields)
        
        print(f"  Samples used: {self.first_order_stats.n_samples}")
        print(f"  Mean u: min={np.min(self.first_order_stats.u_mean):.4f}, "
              f"max={np.max(self.first_order_stats.u_mean):.4f}")
        
        # Compute RMS fluctuations
        print("  Computing RMS fluctuations...")
        rms = compute_rms(self.fields, self.first_order_stats)
        print(f"  u_rms: max={np.max(rms['u_rms']):.4f}")
    
    def _compute_second_order_statistics(self) -> None:
        """Compute second-order (Reynolds stress) statistics."""
        print("  Computing Reynolds stresses and TKE...")
        
        self.second_order_stats = compute_reynolds_stresses(
            self.fields,
            self.first_order_stats
        )
        
        print(f"  TKE: min={np.min(self.second_order_stats.tke):.6f}, "
              f"max={np.max(self.second_order_stats.tke):.6f}")
        print(f"  <u'u'>: max={np.max(self.second_order_stats.uu):.6f}")
        print(f"  <v'v'>: max={np.max(self.second_order_stats.vv):.6f}")
        print(f"  <u'v'>: min={np.min(self.second_order_stats.uv):.6f}, "
              f"max={np.max(self.second_order_stats.uv):.6f}")
        
        # Higher-order moments if requested
        if self.config['statistics'].get('compute_higher_order', False):
            print("  Computing higher-order moments (skewness, flatness)...")
            higher_order = compute_higher_order_moments(
                self.fields,
                self.first_order_stats,
                self.second_order_stats
            )
            # Store for later
            self.higher_order_stats = higher_order
    
    def _temporal_analysis(self) -> None:
        """Perform temporal analysis."""
        temporal_config = self.config['temporal']
        step_time = temporal_config.get('step_change_time')
        
        if step_time is not None:
            print(f"  Analyzing step change at t = {step_time}...")
            
            # Use StepChangeAnalyzer
            analyzer = StepChangeAnalyzer(step_time=step_time)
            for field in self.fields:
                analyzer.add_field(field)
            
            # Get comparison
            try:
                self.temporal_comparison = analyzer.compare_before_after()
                transient = analyzer.analyze_transient()
                
                print(f"  Pre-step samples: {self.temporal_comparison.n_samples_1}")
                print(f"  Post-step samples: {self.temporal_comparison.n_samples_2}")
                
                if self.temporal_comparison.tke_ratio is not None:
                    tke_ratio = np.nanmean(self.temporal_comparison.tke_ratio)
                    print(f"  TKE ratio (after/before): {tke_ratio:.3f}")
                
                if 'transient_duration' in transient:
                    print(f"  Transient duration: {transient['transient_duration']:.4f}")
                if 'relaxation_time' in transient:
                    print(f"  Relaxation time: {transient['relaxation_time']:.4f}")
                
                self.transient_analysis = transient
                
            except ValueError as e:
                print(f"  Warning: Could not complete step change analysis: {e}")
        
        else:
            print("  No step change time specified, computing running averages...")
            
            # Compute running average evolution
            evolution = running_average(self.fields, compute_second_order=True)
            
            print(f"  Time evolution computed for {len(evolution.times)} time points")
            
            self.temporal_evolution = evolution
    
    def _save_results(self) -> None:
        """Save results to files."""
        output_config = self.config['output']
        formats = output_config.get('formats', ['npz'])
        
        # Save first-order statistics
        if self.first_order_stats is not None and output_config.get('save_statistics', True):
            for fmt in formats:
                filename = self.output_dir / f'first_order_stats.{fmt}'
                save_statistics(self.first_order_stats, str(filename), format=fmt)
        
        # Save second-order statistics
        if self.second_order_stats is not None and output_config.get('save_statistics', True):
            for fmt in formats:
                filename = self.output_dir / f'second_order_stats.{fmt}'
                save_statistics(self.second_order_stats, str(filename), format=fmt)
        
        # Export to CSV
        if output_config.get('export_csv', True) and self.second_order_stats is not None:
            filename = self.output_dir / 'statistics.csv'
            export_to_csv(
                self.second_order_stats,
                str(filename),
                y_coordinate=self.fields[0].y if self.fields else None
            )
        
        # Export to VTK
        if output_config.get('export_vtk', False) and len(self.fields) > 0:
            filename = self.output_dir / 'mean_field.vtk'
            # Create a synthetic FieldData with mean values
            mean_field = self.fields[0]  # Use as template
            mean_field.u = self.first_order_stats.u_mean
            mean_field.v = self.first_order_stats.v_mean
            mean_field.w = self.first_order_stats.w_mean
            export_to_vtk(mean_field, str(filename))
        
        # Save temporal comparison if available
        if hasattr(self, 'temporal_comparison') and self.temporal_comparison is not None:
            filename = self.output_dir / 'temporal_comparison.npz'
            data = self.temporal_comparison.to_dict()
            data['period_1_name'] = self.temporal_comparison.period_1_name
            data['period_2_name'] = self.temporal_comparison.period_2_name
            np.savez_compressed(str(filename), **data)
            print(f"  Saved temporal comparison to {filename}")
    
    def _generate_summary(self) -> Dict:
        """Generate summary of results."""
        summary = {
            'case_name': self.config['case_name'],
            'n_snapshots': len(self.fields),
            'time_range': [self.fields[0].time, self.fields[-1].time] if self.fields else [0, 0],
            'total_points': self.fields[0].total_points if self.fields else 0,
        }
        
        if self.first_order_stats is not None:
            summary['u_mean_max'] = float(np.max(self.first_order_stats.u_mean))
            summary['u_mean_min'] = float(np.min(self.first_order_stats.u_mean))
        
        if self.second_order_stats is not None:
            summary['tke_max'] = float(np.max(self.second_order_stats.tke))
            summary['tke_mean'] = float(np.mean(self.second_order_stats.tke))
        
        if hasattr(self, 'temporal_comparison') and self.temporal_comparison is not None:
            summary.update(self.temporal_comparison.get_summary())
        
        # Print summary
        print("\nSummary:")
        print("-" * 40)
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6g}")
            else:
                print(f"  {key}: {value}")
        
        # Save summary to JSON
        import json
        summary_file = self.output_dir / 'summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSummary saved to: {summary_file}")
        
        return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Nek5000 DNS Post-Processor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --case channel --data-dir ./output --start 1 --end 1000
  %(prog)s --config config.yaml
  %(prog)s --case dns --data-dir ./data --start 100 --end 500 --step-change 200
        """
    )
    
    # Configuration file
    parser.add_argument(
        '--config', '-c',
        help='Path to YAML configuration file'
    )
    
    # Command-line options (override config file)
    parser.add_argument(
        '--case',
        help='Case name (base name of Nek5000 files)'
    )
    parser.add_argument(
        '--data-dir',
        help='Directory containing field files'
    )
    parser.add_argument(
        '--output-dir',
        help='Directory for output files'
    )
    parser.add_argument(
        '--start', type=int,
        help='Starting timestep'
    )
    parser.add_argument(
        '--end', type=int,
        help='Ending timestep'
    )
    parser.add_argument(
        '--step', type=int, default=1,
        help='Timestep interval'
    )
    parser.add_argument(
        '--step-change', type=float,
        help='Time of step change for temporal analysis'
    )
    parser.add_argument(
        '--precision', choices=['single', 'double'], default='single',
        help='Data precision'
    )
    parser.add_argument(
        '--no-second-order', action='store_true',
        help='Skip second-order statistics'
    )
    parser.add_argument(
        '--higher-order', action='store_true',
        help='Compute higher-order moments'
    )
    parser.add_argument(
        '--export-vtk', action='store_true',
        help='Export results to VTK format'
    )
    parser.add_argument(
        '--generate-config', action='store_true',
        help='Generate default configuration file and exit'
    )
    
    args = parser.parse_args()
    
    # Generate default config if requested
    if args.generate_config:
        config = create_default_config()
        with open('postprocessor_config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print("Generated default configuration: postprocessor_config.yaml")
        return
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config()
    
    # Override with command-line arguments
    if args.case:
        config['case_name'] = args.case
    if args.data_dir:
        config['data_dir'] = args.data_dir
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.start:
        config['timesteps']['start'] = args.start
    if args.end:
        config['timesteps']['end'] = args.end
    if args.step:
        config['timesteps']['step'] = args.step
    if args.step_change:
        config['temporal']['step_change_time'] = args.step_change
    if args.precision:
        config['precision'] = args.precision
    if args.no_second_order:
        config['statistics']['compute_second_order'] = False
    if args.higher_order:
        config['statistics']['compute_higher_order'] = True
    if args.export_vtk:
        config['output']['export_vtk'] = True
    
    # Validate required parameters
    if not config.get('case_name'):
        parser.error("Case name is required (--case or in config file)")
    
    # Run post-processor
    try:
        processor = Nek5000PostProcessor(config)
        summary = processor.run()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
