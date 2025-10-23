"""
Quick Demo - Single Simulation Analysis
========================================

This script analyzes just ONE simulation and creates a simple plot.
Use this to verify everything works before running full analyses.

Runtime: ~30 seconds
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

# Configuration
DATA_DIR = '../data/sim'
OUTPUT_DIR = './outputs'

def quick_demo():
    """Run a quick demonstration analysis on one simulation."""
    
    print("=" * 60)
    print("QUICK DEMO - Single Simulation Analysis")
    print("=" * 60)
    
    # Find first simulation
    sim_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    if not sim_dirs:
        print("ERROR: No simulation directories found!")
        return
    
    sim_id = sim_dirs[0]
    sim_path = os.path.join(DATA_DIR, sim_id)
    
    print(f"\nAnalyzing simulation: {sim_id[:16]}...")
    
    # Load data
    print("Loading data files...")
    with open(os.path.join(sim_path, 'damage_events.json')) as f:
        damage_events = json.load(f)
    
    with open(os.path.join(sim_path, 'config.json')) as f:
        config = json.load(f)
    
    print(f"  - Loaded {len(damage_events):,} damage events")
    print(f"  - Scenario: {config.get('name', 'Unknown')}")
    print(f"  - Max time: {config.get('max_scenario_time', 0)/3600:.1f} hours")
    print(f"  - Entities: {len(config.get('entities', []))}")
    
    # Simple analysis: damage over time
    print("\nAnalyzing damage over time...")
    
    # Bin by hour
    max_time = config.get('max_scenario_time', 345600)
    hours = np.arange(0, max_time/3600 + 1, 1)
    damage_per_hour = np.zeros(len(hours))
    
    for event in damage_events:
        hour = int(event['timestamp'] / 3600)
        if hour < len(hours):
            damage_per_hour[hour] += event['damage']
    
    # Cumulative damage
    cumulative_damage = np.cumsum(damage_per_hour)
    
    # Create plot
    print("Creating plot...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Damage per hour
    ax1.bar(hours, damage_per_hour, color='orange', alpha=0.7, edgecolor='darkorange')
    ax1.set_xlabel('Simulation Time (hours)', fontsize=12)
    ax1.set_ylabel('Damage Dealt per Hour', fontsize=12)
    ax1.set_title(f'Combat Damage Over Time\n{config.get("name", "Unknown")}', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    total_damage = damage_per_hour.sum()
    peak_hour = hours[np.argmax(damage_per_hour)]
    ax1.text(0.02, 0.98, 
             f'Total Damage: {total_damage:.1f}\nPeak: Hour {peak_hour:.0f}\nEvents: {len(damage_events):,}',
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Plot 2: Cumulative damage
    ax2.plot(hours, cumulative_damage, 'r-', linewidth=2.5)
    ax2.fill_between(hours, 0, cumulative_damage, alpha=0.3, color='red')
    ax2.set_xlabel('Simulation Time (hours)', fontsize=12)
    ax2.set_ylabel('Cumulative Damage', fontsize=12)
    ax2.set_title('Cumulative Combat Damage', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'demo_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ SUCCESS!")
    print(f"\nPlot saved to: {os.path.abspath(output_path)}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Simulation: {config.get('name', 'Unknown')}")
    print(f"Total Damage Events: {len(damage_events):,}")
    print(f"Total Damage Dealt: {total_damage:.1f}")
    print(f"Peak Combat Hour: {peak_hour:.0f}")
    print(f"Combat Duration: {hours[np.where(damage_per_hour > 0)[0][-1]]:.0f} hours" 
          if np.any(damage_per_hour > 0) else "Unknown")
    print("=" * 60)
    
    print("\n✓ Demo complete! If this worked, you're ready to run full analyses.")
    print("\nNext steps:")
    print("  python run_all_analyses.py --sims 10")

if __name__ == '__main__':
    try:
        quick_demo()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        print("\nTroubleshooting:")
        print("  1. Run: python test_setup.py")
        print("  2. Check that you're in the analysis/ directory")
        print("  3. Ensure data files are in ../data/sim/")
