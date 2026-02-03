#!/usr/bin/env python3
"""
Advanced visualization tool for Nek5000 DNS post-processing results
Creates publication-quality plots for second-order statistics and temporal comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import h5py
import json
import os
import argparse
from typing import Dict, List


class ResultVisualizer:
    """Advanced visualization for DNS post-processing results"""
    
    def __init__(self, results_dir: str = './results'):
        self.results_dir = results_dir
        self.figures_dir = os.path.join(results_dir, 'figures')
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Set publication-quality defaults
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 13
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    
    def load_statistics(self, filename: str) -> Dict:
        """Load statistics from HDF5 file"""
        filepath = os.path.join(self.results_dir, filename)
        
        stats = {
            'mean': {},
            'variance': {},
            'std_dev': {},
            'rms': {},
            'skewness': {},
            'kurtosis': {},
            'reynolds_stress': {}
        }
        
        with h5py.File(filepath, 'r') as f:
            # Load metadata
            stats['time_start'] = f.attrs.get('time_start', 0.0)
            stats['time_end'] = f.attrs.get('time_end', 0.0)
            stats['n_samples'] = f.attrs.get('n_samples', 0)
            
            # Load data groups
            for group_name in ['mean', 'variance', 'std_dev', 'rms', 'skewness', 'kurtosis']:
                if group_name in f:
                    for var in f[group_name].keys():
                        stats[group_name][var] = f[group_name][var][:]
            
            # Load Reynolds stress
            if 'reynolds_stress' in f:
                for comp in f['reynolds_stress'].keys():
                    stats['reynolds_stress'][comp] = f['reynolds_stress'][comp][:]
        
        return stats
    
    def load_temporal_evolution(self) -> Dict:
        """Load temporal evolution data"""
        filepath = os.path.join(self.results_dir, 'temporal_evolution.json')
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return data
    
    def load_window_comparison(self) -> Dict:
        """Load window comparison data"""
        filepath = os.path.join(self.results_dir, 'window_comparison.json')
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return data
    
    def plot_comprehensive_temporal_evolution(self):
        """Create comprehensive temporal evolution plots"""
        print("Creating comprehensive temporal evolution plots...")
        
        data = self.load_temporal_evolution()
        
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        variables = ['u', 'v', 'w', 'p']
        titles = ['Streamwise Velocity (u)', 'Cross-stream Velocity (v)', 
                 'Spanwise Velocity (w)', 'Pressure (p)']
        
        for idx, (var, title) in enumerate(zip(variables, titles)):
            if var not in data:
                continue
            
            times = np.array(data[var]['time'])
            values = np.array(data[var]['spatial_average'])
            
            if idx < 4:
                ax = fig.add_subplot(gs[idx // 2, idx % 2])
            else:
                continue
            
            # Plot evolution
            ax.plot(times, values, 'b-', linewidth=1.5, label='Spatial Average')
            
            # Add moving average
            if len(values) > 10:
                window = min(len(values) // 10, 50)
                moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
                times_avg = times[window-1:]
                ax.plot(times_avg, moving_avg, 'r--', linewidth=2, 
                       label=f'Moving Avg (N={window})')
            
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel(f'$\\langle {var} \\rangle$', fontsize=12)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best')
        
        # Add overall title
        fig.suptitle('Temporal Evolution of Spatial Averages', 
                    fontsize=15, fontweight='bold', y=0.995)
        
        output_file = os.path.join(self.figures_dir, 'temporal_evolution_comprehensive.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved to {output_file}")
    
    def plot_reynolds_stress_evolution(self):
        """Plot Reynolds stress tensor components"""
        print("Creating Reynolds stress plots...")
        
        try:
            stats = self.load_statistics('global_statistics.h5')
        except FileNotFoundError:
            print("Global statistics file not found, skipping Reynolds stress plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        components = ["u'u'", "v'v'", "w'w'", "u'v'", "u'w'", "v'w'"]
        titles = [
            "Normal Stress: $\\langle u'u' \\rangle$",
            "Normal Stress: $\\langle v'v' \\rangle$",
            "Normal Stress: $\\langle w'w' \\rangle$",
            "Shear Stress: $\\langle u'v' \\rangle$",
            "Shear Stress: $\\langle u'w' \\rangle$",
            "Shear Stress: $\\langle v'w' \\rangle$"
        ]
        
        for idx, (comp, title) in enumerate(zip(components, titles)):
            if comp not in stats['reynolds_stress']:
                continue
            
            data = stats['reynolds_stress'][comp]
            x = np.arange(len(data))
            
            # Plot with gradient coloring
            scatter = axes[idx].scatter(x, data, c=data, cmap='viridis', 
                                       s=1, alpha=0.6)
            
            # Add mean line
            mean_val = np.mean(data)
            axes[idx].axhline(mean_val, color='r', linestyle='--', 
                            linewidth=2, label=f'Mean = {mean_val:.4e}')
            
            axes[idx].set_xlabel('Spatial Point Index', fontsize=11)
            axes[idx].set_ylabel(comp, fontsize=11)
            axes[idx].set_title(title, fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3, linestyle='--')
            axes[idx].legend(loc='best')
            
            # Add colorbar
            plt.colorbar(scatter, ax=axes[idx], label='Magnitude')
        
        fig.suptitle('Reynolds Stress Tensor Components', 
                    fontsize=15, fontweight='bold')
        plt.tight_layout()
        
        output_file = os.path.join(self.figures_dir, 'reynolds_stress_detailed.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved to {output_file}")
    
    def plot_turbulent_kinetic_energy(self):
        """Plot turbulent kinetic energy distribution"""
        print("Creating TKE distribution plot...")
        
        try:
            stats = self.load_statistics('global_statistics.h5')
        except FileNotFoundError:
            print("Global statistics file not found, skipping TKE plot")
            return
        
        if 'TKE' not in stats['reynolds_stress']:
            print("TKE data not available")
            return
        
        tke = stats['reynolds_stress']['TKE']
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Spatial distribution
        x = np.arange(len(tke))
        axes[0].plot(x, tke, 'b-', linewidth=1.5)
        axes[0].fill_between(x, 0, tke, alpha=0.3)
        axes[0].set_xlabel('Spatial Point Index', fontsize=12)
        axes[0].set_ylabel('TKE', fontsize=12)
        axes[0].set_title('TKE Spatial Distribution', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Histogram
        axes[1].hist(tke, bins=50, alpha=0.7, edgecolor='black')
        axes[1].axvline(np.mean(tke), color='r', linestyle='--', 
                       linewidth=2, label=f'Mean = {np.mean(tke):.4e}')
        axes[1].axvline(np.median(tke), color='g', linestyle='--', 
                       linewidth=2, label=f'Median = {np.median(tke):.4e}')
        axes[1].set_xlabel('TKE', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('TKE Distribution', fontsize=13, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Log-scale distribution
        axes[2].plot(x, tke, 'b-', linewidth=1.5)
        axes[2].set_xlabel('Spatial Point Index', fontsize=12)
        axes[2].set_ylabel('TKE (log scale)', fontsize=12)
        axes[2].set_yscale('log')
        axes[2].set_title('TKE (Log Scale)', fontsize=13, fontweight='bold')
        axes[2].grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        
        output_file = os.path.join(self.figures_dir, 'tke_analysis.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved to {output_file}")
    
    def plot_window_comparison_detailed(self):
        """Create detailed window comparison plots"""
        print("Creating detailed window comparison plots...")
        
        try:
            comp_data = self.load_window_comparison()
        except FileNotFoundError:
            print("Window comparison file not found, skipping")
            return
        
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
        
        variables = ['u', 'v', 'w', 'p']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for idx, var in enumerate(variables):
            if var not in comp_data:
                continue
            
            # Mean comparison
            ax1 = fig.add_subplot(gs[idx // 2, idx % 2])
            
            mean_comp = comp_data[var]['mean_comparison']
            labels = mean_comp['labels']
            values = mean_comp['values']
            
            x_pos = np.arange(len(labels))
            bars = ax1.bar(x_pos, values, color=colors[idx], alpha=0.7, 
                          edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.4e}', ha='center', va='bottom', fontsize=9)
            
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(labels, rotation=45, ha='right')
            ax1.set_ylabel(f'Mean {var}', fontsize=12)
            ax1.set_title(f'Mean {var} Across Time Windows', 
                         fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')
        
        # Reynolds stress comparison
        ax_rey = fig.add_subplot(gs[2, :])
        
        if 'reynolds_stress' in comp_data:
            rey_data = comp_data['reynolds_stress']
            components = ['TKE', "u'u'", "v'v'", "w'w'"]
            
            x_pos = np.arange(len(components))
            width = 0.15
            
            colors_rey = plt.cm.Set3(np.linspace(0, 1, 10))
            
            for i, label in enumerate(rey_data['TKE']['labels']):
                values = []
                for comp in components:
                    if comp in rey_data and i < len(rey_data[comp]['values']):
                        values.append(rey_data[comp]['values'][i])
                    else:
                        values.append(0)
                
                offset = (i - len(rey_data['TKE']['labels'])/2) * width
                ax_rey.bar(x_pos + offset, values, width, 
                          label=label, color=colors_rey[i], 
                          alpha=0.8, edgecolor='black')
            
            ax_rey.set_xticks(x_pos)
            ax_rey.set_xticklabels(components, fontsize=11)
            ax_rey.set_ylabel('Magnitude', fontsize=12)
            ax_rey.set_title('Reynolds Stress Components Across Time Windows', 
                           fontsize=13, fontweight='bold')
            ax_rey.legend(loc='upper right', ncol=3)
            ax_rey.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle('Detailed Window Comparison Analysis', 
                    fontsize=15, fontweight='bold', y=0.995)
        
        output_file = os.path.join(self.figures_dir, 'window_comparison_detailed.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved to {output_file}")
    
    def plot_statistics_distributions(self):
        """Plot statistical distributions (PDF, skewness, kurtosis)"""
        print("Creating statistical distribution plots...")
        
        try:
            stats = self.load_statistics('global_statistics.h5')
        except FileNotFoundError:
            print("Global statistics file not found, skipping")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        variables = ['u', 'v', 'w']
        
        for idx, var in enumerate(variables):
            if var not in stats['mean']:
                continue
            
            # Skewness plot
            if var in stats['skewness']:
                skew = stats['skewness'][var]
                x = np.arange(len(skew))
                
                axes[0, idx].scatter(x, skew, c=skew, cmap='RdBu_r', 
                                   s=2, alpha=0.6, vmin=-3, vmax=3)
                axes[0, idx].axhline(0, color='k', linestyle='-', linewidth=1)
                axes[0, idx].axhline(np.mean(skew), color='r', linestyle='--', 
                                   linewidth=2, label=f'Mean = {np.mean(skew):.3f}')
                axes[0, idx].set_xlabel('Spatial Point Index', fontsize=11)
                axes[0, idx].set_ylabel('Skewness', fontsize=11)
                axes[0, idx].set_title(f'Skewness: {var}', fontsize=12, fontweight='bold')
                axes[0, idx].legend()
                axes[0, idx].grid(True, alpha=0.3)
            
            # Kurtosis plot
            if var in stats['kurtosis']:
                kurt = stats['kurtosis'][var]
                x = np.arange(len(kurt))
                
                axes[1, idx].scatter(x, kurt, c=kurt, cmap='plasma', 
                                   s=2, alpha=0.6)
                axes[1, idx].axhline(3, color='r', linestyle='--', 
                                   linewidth=2, label='Gaussian (3.0)')
                axes[1, idx].axhline(np.mean(kurt), color='g', linestyle='--', 
                                   linewidth=2, label=f'Mean = {np.mean(kurt):.3f}')
                axes[1, idx].set_xlabel('Spatial Point Index', fontsize=11)
                axes[1, idx].set_ylabel('Kurtosis', fontsize=11)
                axes[1, idx].set_title(f'Kurtosis: {var}', fontsize=12, fontweight='bold')
                axes[1, idx].legend()
                axes[1, idx].grid(True, alpha=0.3)
        
        fig.suptitle('Higher-Order Statistical Moments', 
                    fontsize=15, fontweight='bold')
        plt.tight_layout()
        
        output_file = os.path.join(self.figures_dir, 'statistical_distributions.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved to {output_file}")
    
    def plot_turbulence_intensity(self):
        """Plot turbulence intensity profiles"""
        print("Creating turbulence intensity plots...")
        
        try:
            stats = self.load_statistics('global_statistics.h5')
        except FileNotFoundError:
            print("Global statistics file not found, skipping")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        variables = ['u', 'v', 'w']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for idx, (var, color) in enumerate(zip(variables, colors)):
            if var not in stats['mean'] or var not in stats['std_dev']:
                continue
            
            mean = np.abs(stats['mean'][var]) + 1e-10
            std = stats['std_dev'][var]
            ti = std / mean * 100  # Turbulence intensity in percentage
            
            x = np.arange(len(ti))
            
            # Turbulence intensity profile
            axes[idx].plot(x, ti, color=color, linewidth=1.5, label=f'{var}')
            axes[idx].fill_between(x, 0, ti, alpha=0.3, color=color)
            axes[idx].axhline(np.mean(ti), color='r', linestyle='--', 
                            linewidth=2, label=f'Mean = {np.mean(ti):.2f}%')
            axes[idx].set_xlabel('Spatial Point Index', fontsize=12)
            axes[idx].set_ylabel('Turbulence Intensity (%)', fontsize=12)
            axes[idx].set_title(f'Turbulence Intensity: {var}', 
                              fontsize=13, fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        # Combined plot
        for var, color in zip(variables, colors):
            if var not in stats['mean'] or var not in stats['std_dev']:
                continue
            
            mean = np.abs(stats['mean'][var]) + 1e-10
            std = stats['std_dev'][var]
            ti = std / mean * 100
            
            x = np.arange(len(ti))
            axes[3].plot(x, ti, color=color, linewidth=1.5, 
                        label=f'{var} (mean={np.mean(ti):.2f}%)', alpha=0.7)
        
        axes[3].set_xlabel('Spatial Point Index', fontsize=12)
        axes[3].set_ylabel('Turbulence Intensity (%)', fontsize=12)
        axes[3].set_title('Combined Turbulence Intensity', 
                         fontsize=13, fontweight='bold')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        fig.suptitle('Turbulence Intensity Analysis', 
                    fontsize=15, fontweight='bold')
        plt.tight_layout()
        
        output_file = os.path.join(self.figures_dir, 'turbulence_intensity.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved to {output_file}")
    
    def generate_all_plots(self):
        """Generate all available plots"""
        print("\n" + "="*80)
        print("GENERATING ALL VISUALIZATION PLOTS")
        print("="*80 + "\n")
        
        self.plot_comprehensive_temporal_evolution()
        self.plot_reynolds_stress_evolution()
        self.plot_turbulent_kinetic_energy()
        self.plot_window_comparison_detailed()
        self.plot_statistics_distributions()
        self.plot_turbulence_intensity()
        
        print("\n" + "="*80)
        print("VISUALIZATION COMPLETE")
        print("="*80)
        print(f"All figures saved to: {self.figures_dir}\n")


def main():
    """Main visualization script"""
    parser = argparse.ArgumentParser(
        description='Visualize Nek5000 DNS post-processing results'
    )
    parser.add_argument('--results-dir', default='./results',
                       help='Directory containing post-processing results')
    parser.add_argument('--plot-type', choices=['all', 'temporal', 'reynolds', 
                                                 'tke', 'comparison', 'distributions',
                                                 'turbulence'],
                       default='all', help='Type of plots to generate')
    
    args = parser.parse_args()
    
    viz = ResultVisualizer(args.results_dir)
    
    if args.plot_type == 'all':
        viz.generate_all_plots()
    elif args.plot_type == 'temporal':
        viz.plot_comprehensive_temporal_evolution()
    elif args.plot_type == 'reynolds':
        viz.plot_reynolds_stress_evolution()
    elif args.plot_type == 'tke':
        viz.plot_turbulent_kinetic_energy()
    elif args.plot_type == 'comparison':
        viz.plot_window_comparison_detailed()
    elif args.plot_type == 'distributions':
        viz.plot_statistics_distributions()
    elif args.plot_type == 'turbulence':
        viz.plot_turbulence_intensity()


if __name__ == '__main__':
    main()
