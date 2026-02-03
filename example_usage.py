#!/usr/bin/env python3
"""
Example usage script for Nek5000 DNS post-processor
Demonstrates complete workflow for step change analysis
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nek5000_postprocessor import PostProcessor
import json


def analyze_step_change_dns():
    """
    Complete example workflow for analyzing DNS of step change
    """
    
    print("="*80)
    print("NEK5000 DNS STEP CHANGE ANALYSIS - EXAMPLE WORKFLOW")
    print("="*80 + "\n")
    
    # Configuration
    case_name = "step_dns"
    data_dir = "./field_data"
    output_dir = "./results"
    
    # Variables to analyze
    variables = ['u', 'v', 'w', 'p']
    
    # Define time windows for step change analysis
    # Adjust these based on your simulation
    time_windows = [
        (0.0, 5.0, "baseline"),          # Steady state before step
        (5.0, 10.0, "step_initiation"),  # Step change begins
        (10.0, 20.0, "transient_1"),     # Early transient response
        (20.0, 40.0, "transient_2"),     # Mid transient
        (40.0, 80.0, "settling"),        # Settling period
        (80.0, 150.0, "new_steady"),     # New steady state
    ]
    
    # Initialize post-processor
    print(f"Initializing post-processor for case: {case_name}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}\n")
    
    pp = PostProcessor(case_name, data_dir, output_dir)
    
    # Step 1: Load all field data
    print("\nSTEP 1: Loading field data...")
    print("-" * 80)
    n_loaded = pp.load_time_range(start_time=None, end_time=None)
    
    if n_loaded == 0:
        print("\nWARNING: No field files found!")
        print("Expected file pattern: {}_0.f*".format(case_name))
        print("Please ensure field files are in the correct location.")
        return
    
    # Step 2: Compute global statistics
    print("\nSTEP 2: Computing global statistics...")
    print("-" * 80)
    global_stats = pp.compute_global_statistics(variables)
    
    print("\nGlobal Statistics Summary:")
    print(f"  Time range: {global_stats.time_range[0]:.4f} to {global_stats.time_range[1]:.4f}")
    print(f"  Number of samples: {global_stats.n_samples}")
    print(f"  Variables analyzed: {', '.join(global_stats.mean.keys())}")
    
    # Step 3: Analyze temporal evolution
    print("\nSTEP 3: Analyzing temporal evolution...")
    print("-" * 80)
    evolution_data = pp.analyze_temporal_evolution(variables)
    
    # Step 4: Compute windowed statistics for step change analysis
    print("\nSTEP 4: Computing windowed statistics...")
    print("-" * 80)
    pp.compute_windowed_statistics(time_windows, variables)
    
    # Step 5: Compare time windows
    print("\nSTEP 5: Comparing time windows...")
    print("-" * 80)
    comparison = pp.compare_time_windows(variables)
    
    print("\nMean velocity comparison across windows:")
    for var in ['u', 'v', 'w']:
        if var in comparison:
            mean_vals = comparison[var]['mean_comparison']['values']
            labels = comparison[var]['mean_comparison']['labels']
            print(f"\n  {var}:")
            for label, val in zip(labels, mean_vals):
                print(f"    {label:20s}: {val:.6e}")
    
    # Step 6: Analyze step response
    print("\nSTEP 6: Analyzing step response characteristics...")
    print("-" * 80)
    response = pp.analyze_step_response(variables)
    
    print("\nStep Response Analysis:")
    for var in variables:
        if var in response and 'error' not in response[var]:
            r = response[var]
            print(f"\n  {var}:")
            print(f"    Initial value:      {r['initial_value']:.6e}")
            print(f"    Final value:        {r['final_value']:.6e}")
            print(f"    Step magnitude:     {r['step_magnitude']:.6e}")
            print(f"    Overshoot:          {r['overshoot']:.6e} ({r['overshoot_percent']:.2f}%)")
            if r['settling_label']:
                print(f"    Settled at window:  {r['settling_label']}")
    
    # Step 7: Generate visualization plots
    print("\nSTEP 7: Generating visualization plots...")
    print("-" * 80)
    pp.generate_plots(global_stats)
    
    # Step 8: Generate comprehensive report
    print("\nSTEP 8: Generating comprehensive report...")
    print("-" * 80)
    pp.generate_report()
    
    # Print summary of second-order statistics
    print("\n" + "="*80)
    print("SECOND-ORDER STATISTICS SUMMARY")
    print("="*80)
    
    print("\nReynolds Stress Components (spatial averages):")
    for comp, data in global_stats.reynolds_stress.items():
        import numpy as np
        mean_val = np.mean(data)
        std_val = np.std(data)
        print(f"  {comp:8s}: mean = {mean_val:.6e}, std = {std_val:.6e}")
    
    print("\nTurbulence Intensities:")
    for var in ['u', 'v', 'w']:
        if var in global_stats.mean and var in global_stats.std_dev:
            import numpy as np
            mean = np.abs(np.mean(global_stats.mean[var])) + 1e-10
            std = np.mean(global_stats.std_dev[var])
            ti = std / mean * 100
            print(f"  {var}: {ti:.2f}%")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - global_statistics.h5 (full statistics data)")
    print("  - global_statistics_summary.json (summary)")
    print("  - temporal_evolution.json (time series data)")
    print("  - window_comparison.json (window comparisons)")
    print("  - step_response_analysis.json (step response)")
    print("  - analysis_report.txt (comprehensive report)")
    print("  - *.png (visualization plots)")
    
    print("\nTo generate advanced visualizations, run:")
    print(f"  python visualize_results.py --results-dir {output_dir}")
    

def quick_analysis_from_config():
    """
    Run analysis using configuration file
    """
    config_file = "config_example.json"
    
    if not os.path.exists(config_file):
        print(f"Configuration file {config_file} not found!")
        print("Please create a configuration file or use analyze_step_change_dns()")
        return
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Extract configuration
    case_name = config['case_name']
    data_dir = config.get('data_directory', '.')
    output_dir = config.get('output_directory', './results')
    variables = config['analysis_parameters'].get('variables', ['u', 'v', 'w', 'p'])
    
    # Time windows
    time_windows = [
        (w['start'], w['end'], w['label'])
        for w in config['windows']
    ]
    
    # Initialize and run
    pp = PostProcessor(case_name, data_dir, output_dir)
    
    print(f"Loading data for case: {case_name}")
    n_loaded = pp.load_time_range()
    
    if n_loaded > 0:
        pp.compute_global_statistics(variables)
        pp.analyze_temporal_evolution(variables)
        pp.compute_windowed_statistics(time_windows, variables)
        pp.compare_time_windows(variables)
        pp.analyze_step_response(variables)
        pp.generate_plots()
        pp.generate_report()
        
        print(f"\nAnalysis complete! Results in: {output_dir}")
    else:
        print("No data files found. Please check configuration.")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Example Nek5000 DNS analysis')
    parser.add_argument('--mode', choices=['default', 'config'], default='default',
                       help='Analysis mode: default or config-based')
    
    args = parser.parse_args()
    
    if args.mode == 'config':
        quick_analysis_from_config()
    else:
        analyze_step_change_dns()
