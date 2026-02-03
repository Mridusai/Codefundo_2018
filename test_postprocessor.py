#!/usr/bin/env python3
"""
Test suite for Nek5000 DNS post-processor
Validates functionality using synthetic test data
"""

import sys
import os
import shutil
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_test_data import SyntheticDNSGenerator
from nek5000_postprocessor import PostProcessor, StatisticsCalculator, TemporalComparator
from visualize_results import ResultVisualizer


class PostProcessorTester:
    """Test suite for post-processor functionality"""
    
    def __init__(self, test_dir: str = './test_validation'):
        self.test_dir = test_dir
        self.data_dir = os.path.join(test_dir, 'data')
        self.results_dir = os.path.join(test_dir, 'results')
        self.case_name = 'test_case'
        
        # Create test directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.tests_passed = 0
        self.tests_failed = 0
    
    def setup(self):
        """Generate test data"""
        print("\n" + "="*80)
        print("SETTING UP TEST ENVIRONMENT")
        print("="*80 + "\n")
        
        print("Generating synthetic DNS test data...")
        generator = SyntheticDNSGenerator(nx=4, ny=4, nz=4, nel=20)
        
        generator.generate_step_change_dataset(
            output_dir=self.data_dir,
            case_name=self.case_name,
            t_start=0.0,
            t_end=30.0,
            dt=1.0,
            step_time=10.0
        )
        
        print("\nTest environment setup complete!\n")
    
    def test_file_reading(self):
        """Test 1: File reading functionality"""
        print("\n" + "-"*80)
        print("TEST 1: File Reading")
        print("-"*80)
        
        try:
            from nek5000_postprocessor import Nek5000Reader
            
            reader = Nek5000Reader(self.case_name, self.data_dir)
            files = reader.get_field_files()
            
            assert len(files) > 0, "No files found"
            print(f"✓ Found {len(files)} field files")
            
            # Try reading first file
            data = reader.read_field_file(os.path.basename(files[0]))
            
            assert data.velocity_x is not None, "Velocity data not loaded"
            assert data.pressure is not None, "Pressure data not loaded"
            assert len(data.velocity_x) > 0, "Empty velocity array"
            
            print(f"✓ Successfully read field file")
            print(f"  Time: {data.time:.4f}")
            print(f"  Data points: {len(data.velocity_x)}")
            
            self.tests_passed += 1
            return True
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            self.tests_failed += 1
            return False
    
    def test_statistics_calculation(self):
        """Test 2: Statistics calculation"""
        print("\n" + "-"*80)
        print("TEST 2: Statistics Calculation")
        print("-"*80)
        
        try:
            pp = PostProcessor(self.case_name, self.data_dir, self.results_dir)
            
            # Load data
            n_loaded = pp.load_time_range(start_time=0.0, end_time=15.0)
            assert n_loaded > 5, f"Insufficient data loaded: {n_loaded}"
            print(f"✓ Loaded {n_loaded} time samples")
            
            # Compute statistics
            stats = pp.compute_global_statistics(['u', 'v', 'w', 'p'])
            
            assert 'u' in stats.mean, "Mean velocity not calculated"
            assert 'u' in stats.variance, "Variance not calculated"
            assert 'u' in stats.std_dev, "Standard deviation not calculated"
            
            print(f"✓ Statistics calculated successfully")
            print(f"  Variables: {', '.join(stats.mean.keys())}")
            print(f"  Time range: {stats.time_range[0]:.2f} - {stats.time_range[1]:.2f}")
            print(f"  Samples: {stats.n_samples}")
            
            # Check Reynolds stress
            assert 'TKE' in stats.reynolds_stress, "TKE not calculated"
            assert "u'u'" in stats.reynolds_stress, "Reynolds stress not calculated"
            
            print(f"✓ Reynolds stress calculated")
            print(f"  Components: {', '.join(stats.reynolds_stress.keys())}")
            
            # Check values are reasonable
            mean_u = np.mean(stats.mean['u'])
            assert not np.isnan(mean_u), "NaN in mean velocity"
            assert not np.isinf(mean_u), "Inf in mean velocity"
            
            print(f"✓ Statistics values are valid")
            print(f"  Mean u: {mean_u:.4e}")
            print(f"  Mean TKE: {np.mean(stats.reynolds_stress['TKE']):.4e}")
            
            self.tests_passed += 1
            return True
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            self.tests_failed += 1
            return False
    
    def test_temporal_analysis(self):
        """Test 3: Temporal analysis"""
        print("\n" + "-"*80)
        print("TEST 3: Temporal Analysis")
        print("-"*80)
        
        try:
            pp = PostProcessor(self.case_name, self.data_dir, self.results_dir)
            pp.load_time_range()
            
            # Analyze temporal evolution
            evolution = pp.analyze_temporal_evolution(['u', 'v', 'w'])
            
            assert 'u' in evolution, "Velocity evolution not computed"
            assert 'time' in evolution['u'], "Time data missing"
            assert 'spatial_average' in evolution['u'], "Spatial average missing"
            
            times = np.array(evolution['u']['time'])
            values = np.array(evolution['u']['spatial_average'])
            
            assert len(times) == len(values), "Mismatched time and value arrays"
            assert len(times) > 0, "Empty temporal data"
            
            print(f"✓ Temporal evolution computed")
            print(f"  Time points: {len(times)}")
            print(f"  Variables: {', '.join(evolution.keys())}")
            
            # Check temporal trend (should show step change)
            if len(values) > 10:
                early_mean = np.mean(values[:5])
                late_mean = np.mean(values[-5:])
                change_percent = abs(late_mean - early_mean) / (abs(early_mean) + 1e-10) * 100
                
                print(f"✓ Detected temporal change: {change_percent:.2f}%")
            
            self.tests_passed += 1
            return True
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            self.tests_failed += 1
            return False
    
    def test_windowed_statistics(self):
        """Test 4: Windowed statistics and comparison"""
        print("\n" + "-"*80)
        print("TEST 4: Windowed Statistics")
        print("-"*80)
        
        try:
            pp = PostProcessor(self.case_name, self.data_dir, self.results_dir)
            pp.load_time_range()
            
            # Define time windows
            time_windows = [
                (0.0, 8.0, "pre_step"),
                (12.0, 20.0, "post_step"),
                (20.0, 30.0, "settled")
            ]
            
            # Compute windowed statistics
            pp.compute_windowed_statistics(time_windows, ['u', 'v', 'w'])
            
            assert len(pp.temporal_comp.window_labels) == 3, "Windows not created"
            print(f"✓ Computed statistics for {len(time_windows)} windows")
            
            # Compare windows
            comparison = pp.compare_time_windows(['u', 'v'])
            
            assert 'u' in comparison, "Comparison data missing"
            assert 'mean_comparison' in comparison['u'], "Mean comparison missing"
            
            mean_comp = comparison['u']['mean_comparison']
            assert len(mean_comp['values']) == 3, "Incorrect number of window values"
            
            print(f"✓ Window comparison computed")
            print(f"  Windows: {', '.join(mean_comp['labels'])}")
            
            # Check for step change detection
            if len(mean_comp['values']) >= 2:
                change = abs(mean_comp['values'][1] - mean_comp['values'][0])
                print(f"✓ Step magnitude detected: {change:.4e}")
            
            self.tests_passed += 1
            return True
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            self.tests_failed += 1
            return False
    
    def test_step_response_analysis(self):
        """Test 5: Step response characterization"""
        print("\n" + "-"*80)
        print("TEST 5: Step Response Analysis")
        print("-"*80)
        
        try:
            pp = PostProcessor(self.case_name, self.data_dir, self.results_dir)
            pp.load_time_range()
            
            # Setup windows for step analysis
            time_windows = [
                (0.0, 8.0, "baseline"),
                (8.0, 12.0, "transition"),
                (12.0, 20.0, "response"),
                (20.0, 30.0, "steady")
            ]
            
            pp.compute_windowed_statistics(time_windows, ['u', 'v', 'w'])
            
            # Analyze step response
            response = pp.analyze_step_response(['u'])
            
            assert 'u' in response, "Response analysis missing"
            assert 'initial_value' in response['u'], "Initial value missing"
            assert 'final_value' in response['u'], "Final value missing"
            assert 'step_magnitude' in response['u'], "Step magnitude missing"
            
            r = response['u']
            print(f"✓ Step response characterized")
            print(f"  Initial value: {r['initial_value']:.4e}")
            print(f"  Final value: {r['final_value']:.4e}")
            print(f"  Step magnitude: {r['step_magnitude']:.4e}")
            print(f"  Overshoot: {r['overshoot']:.4e}")
            
            # Verify step was detected
            assert abs(r['step_magnitude']) > 1e-10, "No step detected"
            print(f"✓ Step change successfully detected")
            
            self.tests_passed += 1
            return True
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            self.tests_failed += 1
            return False
    
    def test_output_files(self):
        """Test 6: Output file generation"""
        print("\n" + "-"*80)
        print("TEST 6: Output File Generation")
        print("-"*80)
        
        try:
            pp = PostProcessor(self.case_name, self.data_dir, self.results_dir)
            pp.load_time_range()
            
            stats = pp.compute_global_statistics(['u', 'v', 'w', 'p'])
            pp.analyze_temporal_evolution(['u', 'v', 'w'])
            
            # Check for output files
            expected_files = [
                'global_statistics.h5',
                'global_statistics_summary.json',
                'temporal_evolution.json'
            ]
            
            for filename in expected_files:
                filepath = os.path.join(self.results_dir, filename)
                assert os.path.exists(filepath), f"Missing file: {filename}"
                assert os.path.getsize(filepath) > 0, f"Empty file: {filename}"
                print(f"✓ Found {filename}")
            
            # Generate plots
            pp.generate_plots(stats)
            
            # Check for plot files
            plot_files = [f for f in os.listdir(self.results_dir) if f.endswith('.png')]
            assert len(plot_files) > 0, "No plots generated"
            print(f"✓ Generated {len(plot_files)} plots")
            
            # Generate report
            pp.generate_report()
            report_file = os.path.join(self.results_dir, 'analysis_report.txt')
            assert os.path.exists(report_file), "Report not generated"
            print(f"✓ Generated analysis report")
            
            self.tests_passed += 1
            return True
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            self.tests_failed += 1
            return False
    
    def test_visualization(self):
        """Test 7: Advanced visualization"""
        print("\n" + "-"*80)
        print("TEST 7: Advanced Visualization")
        print("-"*80)
        
        try:
            # Run basic analysis first
            pp = PostProcessor(self.case_name, self.data_dir, self.results_dir)
            pp.load_time_range()
            pp.compute_global_statistics(['u', 'v', 'w', 'p'])
            pp.analyze_temporal_evolution(['u', 'v', 'w'])
            
            time_windows = [
                (0.0, 8.0, "pre"),
                (12.0, 30.0, "post")
            ]
            pp.compute_windowed_statistics(time_windows, ['u', 'v', 'w'])
            pp.compare_time_windows(['u', 'v', 'w'])
            
            # Test visualization
            viz = ResultVisualizer(self.results_dir)
            
            # Generate individual plots
            viz.plot_comprehensive_temporal_evolution()
            viz.plot_reynolds_stress_evolution()
            viz.plot_turbulent_kinetic_energy()
            viz.plot_window_comparison_detailed()
            
            # Check for figures directory
            figures_dir = os.path.join(self.results_dir, 'figures')
            assert os.path.exists(figures_dir), "Figures directory not created"
            
            # Count generated figures
            figure_files = [f for f in os.listdir(figures_dir) if f.endswith('.png')]
            print(f"✓ Generated {len(figure_files)} advanced visualizations")
            
            for fig in figure_files:
                print(f"  - {fig}")
            
            self.tests_passed += 1
            return True
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            self.tests_failed += 1
            return False
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("\n" + "="*80)
        print("NEK5000 POST-PROCESSOR TEST SUITE")
        print("="*80)
        
        # Setup
        self.setup()
        
        # Run tests
        tests = [
            self.test_file_reading,
            self.test_statistics_calculation,
            self.test_temporal_analysis,
            self.test_windowed_statistics,
            self.test_step_response_analysis,
            self.test_output_files,
            self.test_visualization
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                print(f"\n✗ Unexpected error in {test.__name__}: {e}")
                self.tests_failed += 1
        
        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Tests passed: {self.tests_passed}")
        print(f"Tests failed: {self.tests_failed}")
        print(f"Total tests:  {self.tests_passed + self.tests_failed}")
        
        if self.tests_failed == 0:
            print("\n✓ ALL TESTS PASSED!")
            print("="*80 + "\n")
            return True
        else:
            print(f"\n✗ {self.tests_failed} TEST(S) FAILED")
            print("="*80 + "\n")
            return False
    
    def cleanup(self):
        """Clean up test files"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            print(f"Cleaned up test directory: {self.test_dir}")


def main():
    """Run test suite"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Nek5000 post-processor')
    parser.add_argument('--test-dir', default='./test_validation',
                       help='Directory for test files')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up test files after completion')
    parser.add_argument('--keep-on-failure', action='store_true',
                       help='Keep test files if tests fail')
    
    args = parser.parse_args()
    
    tester = PostProcessorTester(args.test_dir)
    
    try:
        success = tester.run_all_tests()
        
        if args.cleanup and (success or not args.keep_on_failure):
            tester.cleanup()
        else:
            print(f"\nTest files preserved in: {args.test_dir}")
            print("To clean up manually, run:")
            print(f"  rm -rf {args.test_dir}")
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
