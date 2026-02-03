#!/usr/bin/env python3
"""
Generate synthetic test data for Nek5000 post-processor validation
Creates realistic DNS-like data with step change characteristics
"""

import numpy as np
import struct
import os
from typing import Tuple


class SyntheticDNSGenerator:
    """Generate synthetic DNS data with step change"""
    
    def __init__(self, nx: int = 8, ny: int = 8, nz: int = 8, nel: int = 100):
        """
        Initialize synthetic data generator
        
        Args:
            nx, ny, nz: Grid points per element in each direction
            nel: Number of elements
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.nel = nel
        self.nxyz = nx * ny * nz
        
        # Generate spatial coordinates
        self._generate_coordinates()
        
    def _generate_coordinates(self):
        """Generate spatial coordinates"""
        self.x = np.random.uniform(0, 10, (self.nel, self.nxyz))
        self.y = np.random.uniform(0, 2, (self.nel, self.nxyz))
        self.z = np.random.uniform(0, 2, (self.nel, self.nxyz))
        
        # Sort to make more realistic
        for e in range(self.nel):
            idx = np.argsort(self.x[e, :])
            self.x[e, :] = self.x[e, idx]
    
    def generate_step_change_field(self, time: float, 
                                   step_time: float = 10.0,
                                   transition_width: float = 2.0) -> Tuple:
        """
        Generate velocity and pressure fields with step change
        
        Args:
            time: Current simulation time
            step_time: Time when step change occurs
            transition_width: Width of transition region
        
        Returns:
            Tuple of (u, v, w, p, T) arrays
        """
        # Base flow parameters
        U_base = 1.0
        Re_tau = 180.0
        nu = 0.001
        
        # Step change magnitude
        if time < step_time:
            step_factor = 0.0
        elif time > step_time + transition_width:
            step_factor = 1.0
        else:
            # Smooth transition (tanh profile)
            t_norm = (time - step_time) / transition_width
            step_factor = 0.5 * (1 + np.tanh(4 * (t_norm - 0.5)))
        
        # New steady state is 30% higher velocity
        U_new = U_base * (1.0 + 0.3 * step_factor)
        
        # Generate turbulent fluctuations
        turbulence_intensity = 0.05 + 0.02 * step_factor  # Increases during transition
        
        # Mean flow (streamwise velocity)
        u_mean = np.zeros((self.nel, self.nxyz))
        for e in range(self.nel):
            # Log-law profile in y-direction
            y_norm = self.y[e, :] / 2.0  # Normalize by channel half-height
            u_mean[e, :] = U_new * (1.0 - y_norm**2)  # Parabolic profile
        
        # Add turbulent fluctuations
        u_turb = np.random.normal(0, turbulence_intensity * U_new, (self.nel, self.nxyz))
        v_turb = np.random.normal(0, turbulence_intensity * U_new * 0.5, (self.nel, self.nxyz))
        w_turb = np.random.normal(0, turbulence_intensity * U_new * 0.5, (self.nel, self.nxyz))
        
        # Add coherent structures (simplified)
        freq = 2 * np.pi / 5.0  # Characteristic frequency
        for e in range(self.nel):
            phase = self.x[e, :] * 0.5 + time * 0.1
            u_turb[e, :] += 0.02 * U_new * np.sin(freq * phase)
            v_turb[e, :] += 0.01 * U_new * np.cos(freq * phase)
        
        # Total velocity
        u = u_mean + u_turb
        v = v_turb
        w = w_turb
        
        # Pressure (from simplified momentum balance)
        p = np.zeros((self.nel, self.nxyz))
        for e in range(self.nel):
            # Pressure gradient balances mean flow
            p[e, :] = -self.x[e, :] * 0.1 * U_new**2
            # Add pressure fluctuations correlated with velocity
            p[e, :] += -0.5 * (u_turb[e, :]**2 + v_turb[e, :]**2 + w_turb[e, :]**2)
        
        # Temperature (passive scalar with step change)
        T = np.ones((self.nel, self.nxyz)) * (300.0 + 10.0 * step_factor)
        T += np.random.normal(0, 0.5, (self.nel, self.nxyz))
        
        return u, v, w, p, T
    
    def write_field_file(self, filename: str, time: float, 
                        step_time: float = 10.0):
        """
        Write a synthetic field file in Nek5000-like binary format
        
        Args:
            filename: Output filename
            time: Simulation time
            step_time: Time of step change
        """
        u, v, w, p, T = self.generate_step_change_field(time, step_time)
        
        with open(filename, 'wb') as f:
            # Write header
            header = b'#std 4  ' + b' ' * 124
            f.write(header)
            
            # Write word size
            f.write(struct.pack('i', 8))  # 8 bytes = double precision
            
            # Write element counts
            f.write(struct.pack('i', self.nel))
            f.write(struct.pack('i', self.nel))
            
            # Write time
            f.write(struct.pack('d', time))
            
            # Write grid dimensions
            f.write(struct.pack('iii', self.nx, self.ny, self.nz))
            
            # Write number of elements
            f.write(struct.pack('i', self.nel))
            
            # Write coordinates
            for e in range(self.nel):
                for i in range(self.nxyz):
                    f.write(struct.pack('d', self.x[e, i]))
            
            for e in range(self.nel):
                for i in range(self.nxyz):
                    f.write(struct.pack('d', self.y[e, i]))
            
            for e in range(self.nel):
                for i in range(self.nxyz):
                    f.write(struct.pack('d', self.z[e, i]))
            
            # Write velocity fields
            for e in range(self.nel):
                for i in range(self.nxyz):
                    f.write(struct.pack('d', u[e, i]))
            
            for e in range(self.nel):
                for i in range(self.nxyz):
                    f.write(struct.pack('d', v[e, i]))
            
            for e in range(self.nel):
                for i in range(self.nxyz):
                    f.write(struct.pack('d', w[e, i]))
            
            # Write pressure
            for e in range(self.nel):
                for i in range(self.nxyz):
                    f.write(struct.pack('d', p[e, i]))
            
            # Write temperature
            for e in range(self.nel):
                for i in range(self.nxyz):
                    f.write(struct.pack('d', T[e, i]))
    
    def generate_step_change_dataset(self, output_dir: str, 
                                    case_name: str = "test_dns",
                                    t_start: float = 0.0,
                                    t_end: float = 50.0,
                                    dt: float = 0.5,
                                    step_time: float = 10.0):
        """
        Generate a complete dataset with step change
        
        Args:
            output_dir: Directory to save files
            case_name: Case name prefix
            t_start, t_end: Time range
            dt: Time step between snapshots
            step_time: Time when step change occurs
        """
        os.makedirs(output_dir, exist_ok=True)
        
        times = np.arange(t_start, t_end + dt, dt)
        
        print(f"Generating {len(times)} field files...")
        print(f"Step change at t = {step_time}")
        print(f"Output directory: {output_dir}")
        
        for idx, time in enumerate(times):
            filename = os.path.join(output_dir, f"{case_name}0.f{idx:05d}")
            self.write_field_file(filename, time, step_time)
            
            if (idx + 1) % 10 == 0:
                print(f"  Generated {idx + 1}/{len(times)} files...")
        
        print(f"\nComplete! Generated {len(times)} files in {output_dir}")
        print(f"File pattern: {case_name}0.f*")


def main():
    """Generate test dataset"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate synthetic DNS test data with step change'
    )
    parser.add_argument('--output-dir', default='./test_data',
                       help='Output directory for test data')
    parser.add_argument('--case-name', default='test_dns',
                       help='Case name prefix')
    parser.add_argument('--n-elements', type=int, default=50,
                       help='Number of spectral elements')
    parser.add_argument('--grid-size', type=int, default=8,
                       help='Grid points per element (nx=ny=nz)')
    parser.add_argument('--t-start', type=float, default=0.0,
                       help='Start time')
    parser.add_argument('--t-end', type=float, default=50.0,
                       help='End time')
    parser.add_argument('--dt', type=float, default=0.5,
                       help='Time step between snapshots')
    parser.add_argument('--step-time', type=float, default=10.0,
                       help='Time of step change')
    
    args = parser.parse_args()
    
    print("="*80)
    print("SYNTHETIC DNS DATA GENERATOR")
    print("="*80 + "\n")
    
    print("Configuration:")
    print(f"  Case name: {args.case_name}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Elements: {args.n_elements}")
    print(f"  Grid points: {args.grid_size}³ per element")
    print(f"  Time range: {args.t_start} to {args.t_end}")
    print(f"  Time step: {args.dt}")
    print(f"  Step change time: {args.step_time}\n")
    
    # Generate data
    generator = SyntheticDNSGenerator(
        nx=args.grid_size,
        ny=args.grid_size,
        nz=args.grid_size,
        nel=args.n_elements
    )
    
    generator.generate_step_change_dataset(
        output_dir=args.output_dir,
        case_name=args.case_name,
        t_start=args.t_start,
        t_end=args.t_end,
        dt=args.dt,
        step_time=args.step_time
    )
    
    print("\n" + "="*80)
    print("To analyze this data, run:")
    print(f"  python nek5000_postprocessor.py {args.case_name} \\")
    print(f"    --data-dir {args.output_dir} \\")
    print(f"    --output-dir ./results \\")
    print(f"    --variables u v w p T")
    print("="*80)


if __name__ == '__main__':
    main()
