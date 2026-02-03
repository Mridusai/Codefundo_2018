#!/usr/bin/env python3
"""
Utility scripts for common post-processing tasks
Convenient wrappers and helper functions
"""

import os
import sys
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def extract_statistics_summary(results_dir: str = './results', 
                               output_format: str = 'txt') -> Dict:
    """
    Extract and print summary statistics from HDF5 files
    
    Args:
        results_dir: Directory containing results
        output_format: 'txt', 'json', or 'csv'
    
    Returns:
        Dictionary with summary statistics
    """
    h5_file = os.path.join(results_dir, 'global_statistics.h5')
    
    if not os.path.exists(h5_file):
        print(f"Error: {h5_file} not found")
        return {}
    
    summary = {}
    
    with h5py.File(h5_file, 'r') as f:
        # Extract metadata
        summary['metadata'] = {
            'time_start': f.attrs.get('time_start', 0.0),
            'time_end': f.attrs.get('time_end', 0.0),
            'n_samples': f.attrs.get('n_samples', 0)
        }
        
        # Extract spatial averages
        summary['mean'] = {}
        summary['variance'] = {}
        summary['turbulence_intensity'] = {}
        
        if 'mean' in f:
            for var in f['mean'].keys():
                mean_data = f['mean'][var][:]
                summary['mean'][var] = {
                    'spatial_avg': float(np.mean(mean_data)),
                    'min': float(np.min(mean_data)),
                    'max': float(np.max(mean_data)),
                    'std': float(np.std(mean_data))
                }
        
        if 'variance' in f:
            for var in f['variance'].keys():
                var_data = f['variance'][var][:]
                summary['variance'][var] = {
                    'spatial_avg': float(np.mean(var_data)),
                    'min': float(np.min(var_data)),
                    'max': float(np.max(var_data))
                }
                
                # Compute turbulence intensity
                if var in summary['mean']:
                    mean_val = abs(summary['mean'][var]['spatial_avg'])
                    if mean_val > 1e-10:
                        ti = np.sqrt(summary['variance'][var]['spatial_avg']) / mean_val * 100
                        summary['turbulence_intensity'][var] = ti
        
        # Extract Reynolds stress
        summary['reynolds_stress'] = {}
        if 'reynolds_stress' in f:
            for comp in f['reynolds_stress'].keys():
                data = f['reynolds_stress'][comp][:]
                summary['reynolds_stress'][comp] = {
                    'spatial_avg': float(np.mean(data)),
                    'min': float(np.min(data)),
                    'max': float(np.max(data)),
                    'std': float(np.std(data))
                }
    
    # Output
    if output_format == 'txt':
        print_summary_text(summary)
    elif output_format == 'json':
        print(json.dumps(summary, indent=2))
    elif output_format == 'csv':
        print_summary_csv(summary)
    
    return summary


def print_summary_text(summary: Dict):
    """Print summary in text format"""
    print("\n" + "="*80)
    print("STATISTICS SUMMARY")
    print("="*80)
    
    print("\nMetadata:")
    print(f"  Time range: {summary['metadata']['time_start']:.4f} - {summary['metadata']['time_end']:.4f}")
    print(f"  Samples: {summary['metadata']['n_samples']}")
    
    print("\nMean Values (spatial averages):")
    for var, stats in summary['mean'].items():
        print(f"  {var}: {stats['spatial_avg']:.6e} (min={stats['min']:.6e}, max={stats['max']:.6e})")
    
    print("\nTurbulence Intensity:")
    for var, ti in summary['turbulence_intensity'].items():
        print(f"  {var}: {ti:.2f}%")
    
    print("\nReynolds Stress (spatial averages):")
    for comp, stats in summary['reynolds_stress'].items():
        print(f"  {comp}: {stats['spatial_avg']:.6e}")
    
    print("="*80 + "\n")


def print_summary_csv(summary: Dict):
    """Print summary in CSV format"""
    print("Variable,Mean,Variance,Turbulence_Intensity")
    for var in summary['mean'].keys():
        mean = summary['mean'][var]['spatial_avg']
        var_val = summary['variance'].get(var, {}).get('spatial_avg', 0)
        ti = summary['turbulence_intensity'].get(var, 0)
        print(f"{var},{mean},{var_val},{ti}")


def compare_cases(cases: List[Dict], output_file: Optional[str] = None):
    """
    Compare statistics from multiple cases/runs
    
    Args:
        cases: List of dictionaries with 'label' and 'results_dir'
        output_file: Optional output file for comparison plot
    
    Example:
        cases = [
            {'label': 'Re=1000', 'results_dir': './results_re1000'},
            {'label': 'Re=2000', 'results_dir': './results_re2000'}
        ]
        compare_cases(cases, 'case_comparison.png')
    """
    summaries = {}
    
    for case in cases:
        label = case['label']
        results_dir = case['results_dir']
        
        print(f"Loading case: {label}")
        summaries[label] = extract_statistics_summary(results_dir, output_format='none')
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    variables = ['u', 'v', 'w', 'p']
    
    for idx, var in enumerate(variables):
        labels = []
        mean_values = []
        ti_values = []
        
        for label, summary in summaries.items():
            if var in summary['mean']:
                labels.append(label)
                mean_values.append(summary['mean'][var]['spatial_avg'])
                ti_values.append(summary['turbulence_intensity'].get(var, 0))
        
        if labels:
            x_pos = np.arange(len(labels))
            
            # Plot mean values
            bars = axes[idx].bar(x_pos, mean_values, alpha=0.7, label='Mean')
            axes[idx].set_xticks(x_pos)
            axes[idx].set_xticklabels(labels, rotation=45, ha='right')
            axes[idx].set_ylabel(f'Mean {var}')
            axes[idx].set_title(f'Comparison: {var}')
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # Add TI as text
            for i, (bar, ti) in enumerate(zip(bars, ti_values)):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'TI={ti:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved to: {output_file}")
    else:
        plt.show()


def extract_time_series(results_dir: str = './results',
                       variable: str = 'u',
                       output_file: Optional[str] = None) -> tuple:
    """
    Extract time series data for a specific variable
    
    Args:
        results_dir: Directory containing results
        variable: Variable to extract
        output_file: Optional CSV output file
    
    Returns:
        Tuple of (times, values) arrays
    """
    json_file = os.path.join(results_dir, 'temporal_evolution.json')
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if variable not in data:
        print(f"Variable {variable} not found")
        return None, None
    
    times = np.array(data[variable]['time'])
    values = np.array(data[variable]['spatial_average'])
    
    if output_file:
        np.savetxt(output_file, np.column_stack([times, values]),
                  header=f'Time,{variable}', delimiter=',', comments='')
        print(f"Time series saved to: {output_file}")
    
    return times, values


def plot_custom_comparison(results_dir: str, 
                          windows: List[str],
                          variables: List[str] = ['u', 'v', 'w'],
                          output_file: str = 'custom_comparison.png'):
    """
    Create custom comparison plot for specific windows and variables
    
    Args:
        results_dir: Results directory
        windows: List of window labels to compare
        variables: Variables to plot
        output_file: Output filename
    """
    comp_file = os.path.join(results_dir, 'window_comparison.json')
    
    with open(comp_file, 'r') as f:
        comp_data = json.load(f)
    
    n_vars = len(variables)
    fig, axes = plt.subplots(1, n_vars, figsize=(5*n_vars, 5))
    
    if n_vars == 1:
        axes = [axes]
    
    for idx, var in enumerate(variables):
        if var not in comp_data:
            continue
        
        mean_comp = comp_data[var]['mean_comparison']
        labels = mean_comp['labels']
        values = mean_comp['values']
        
        # Filter to requested windows
        filtered_labels = []
        filtered_values = []
        for label, val in zip(labels, values):
            if label in windows:
                filtered_labels.append(label)
                filtered_values.append(val)
        
        if filtered_labels:
            x_pos = np.arange(len(filtered_labels))
            axes[idx].bar(x_pos, filtered_values, alpha=0.7)
            axes[idx].set_xticks(x_pos)
            axes[idx].set_xticklabels(filtered_labels, rotation=45, ha='right')
            axes[idx].set_ylabel(f'{var}')
            axes[idx].set_title(f'Window Comparison: {var}')
            axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Custom comparison saved to: {output_file}")


def export_to_paraview(results_dir: str, 
                       statistic: str = 'mean',
                       variables: List[str] = ['u', 'v', 'w']):
    """
    Export statistics to format suitable for ParaView visualization
    
    Args:
        results_dir: Results directory
        statistic: Which statistic to export ('mean', 'variance', etc.)
        variables: Variables to export
    
    Note: This creates a simplified export. For full ParaView support,
          coordinates would need to be properly structured.
    """
    print("ParaView export functionality - To be implemented")
    print("For now, use the HDF5 files directly with h5py")
    print("Example:")
    print("  import h5py")
    print("  with h5py.File('global_statistics.h5', 'r') as f:")
    print("      mean_u = f['mean']['u'][:]")


def main():
    """Command-line interface for utility functions"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Utility functions for post-processing')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Extract summary
    extract_parser = subparsers.add_parser('summary', 
                                          help='Extract statistics summary')
    extract_parser.add_argument('--results-dir', default='./results')
    extract_parser.add_argument('--format', choices=['txt', 'json', 'csv'], 
                               default='txt')
    
    # Extract time series
    series_parser = subparsers.add_parser('timeseries',
                                         help='Extract time series')
    series_parser.add_argument('--results-dir', default='./results')
    series_parser.add_argument('--variable', default='u')
    series_parser.add_argument('--output', help='Output CSV file')
    
    # Custom comparison
    comp_parser = subparsers.add_parser('compare',
                                       help='Custom window comparison')
    comp_parser.add_argument('--results-dir', default='./results')
    comp_parser.add_argument('--windows', nargs='+', required=True)
    comp_parser.add_argument('--variables', nargs='+', default=['u', 'v', 'w'])
    comp_parser.add_argument('--output', default='custom_comparison.png')
    
    args = parser.parse_args()
    
    if args.command == 'summary':
        extract_statistics_summary(args.results_dir, args.format)
    
    elif args.command == 'timeseries':
        times, values = extract_time_series(args.results_dir, 
                                           args.variable, 
                                           args.output)
        if times is not None:
            print(f"Extracted {len(times)} time points for variable '{args.variable}'")
    
    elif args.command == 'compare':
        plot_custom_comparison(args.results_dir, 
                             args.windows, 
                             args.variables,
                             args.output)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
