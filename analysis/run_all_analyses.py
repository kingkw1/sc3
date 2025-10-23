"""
Master Analysis Runner
======================

This script runs all analysis modules in sequence and generates a complete
set of performance plots for the DTRA briefing.

Usage:
    python run_all_analyses.py [--sims N] [--output DIR]

Options:
    --sims N: Number of simulations to analyze (default: 10)
    --output DIR: Output directory for plots (default: ./outputs)
"""

import os
import sys
import argparse
from datetime import datetime

def run_analysis(script_name, description):
    """Run a single analysis script."""
    print("\n" + "=" * 70)
    print(f"Running: {description}")
    print("=" * 70)
    
    try:
        # Import and run the script
        import importlib.util
        spec = importlib.util.spec_from_file_location("module", script_name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        print(f"✓ Completed: {description}")
        return True
    except Exception as e:
        print(f"✗ Error in {description}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run all SC3 data analyses')
    parser.add_argument('--sims', type=int, default=10, 
                       help='Number of simulations to analyze (default: 10)')
    parser.add_argument('--output', type=str, default='./outputs',
                       help='Output directory for plots (default: ./outputs)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 70)
    print("SC3 DATA ANALYSIS - MASTER RUNNER")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {args.output}")
    print(f"Number of simulations: {args.sims}")
    print("=" * 70)
    
    # Define all analyses
    analyses = [
        ("1_casualties_over_time.py", "Casualties Over Time Analysis"),
        ("2_force_strength_evolution.py", "Force Strength Evolution Analysis"),
        ("3_combat_intensity_heatmap.py", "Combat Intensity Heatmap Analysis"),
        ("4_unit_survival_analysis.py", "Unit Survival Analysis"),
        ("5_coa_comparison.py", "COA Comparison Analysis"),
    ]
    
    results = []
    
    # Run each analysis
    for script, description in analyses:
        success = run_analysis(script, description)
        results.append((description, success))
    
    # Print summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    for description, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {description}")
    
    total = len(results)
    successful = sum(1 for _, s in results if s)
    
    print("\n" + "-" * 70)
    print(f"Total analyses: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print("-" * 70)
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutput files saved to: {os.path.abspath(args.output)}")
    
    # List output files
    if os.path.exists(args.output):
        output_files = [f for f in os.listdir(args.output) if f.endswith('.png')]
        if output_files:
            print(f"\nGenerated {len(output_files)} plot(s):")
            for f in sorted(output_files):
                print(f"  - {f}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    return successful == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
