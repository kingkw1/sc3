"""
Casualties Over Time Analysis
==============================

This script analyzes and plots casualty rates throughout the simulation timeline.
It shows how combat power degrades over time for both Red and Blue forces.

Outputs:
    - casualties_over_time.png: Line plot showing cumulative casualties
    - casualty_rate.png: Plot showing rate of casualties per hour

Metrics:
    - Cumulative damage dealt over simulation time
    - Casualties aggregated by faction (Red vs Blue)
    - Attrition rates at different phases of combat
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Configuration
DATA_DIR = '../data/sim'
OUTPUT_DIR = './outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_simulation_data(sim_id):
    """Load damage events and config for a simulation."""
    sim_path = os.path.join(DATA_DIR, sim_id)
    
    with open(os.path.join(sim_path, 'damage_events.json')) as f:
        damage_events = json.load(f)
    
    with open(os.path.join(sim_path, 'config.json')) as f:
        config = json.load(f)
    
    with open(os.path.join(sim_path, 'external_id_to_internal_id.json')) as f:
        id_map = json.load(f)
    
    return damage_events, config, id_map

def get_entity_faction(entity_id, config, id_map):
    """Determine which faction an entity belongs to."""
    # Create reverse map
    internal_to_external = {v: k for k, v in id_map.items()}
    
    if entity_id not in internal_to_external:
        return 'Unknown'
    
    external_id = internal_to_external[entity_id]
    
    # Search for entity in config
    for entity in config.get('entities', []):
        if entity['type'] == 'Unit':
            if entity['data']['header']['id'] == external_id:
                faction_id = entity['data'].get('faction', '')
                # Look up faction name
                for faction in config.get('factions', []):
                    if faction['header']['id'] == faction_id:
                        return faction['header']['name']
    
    return 'Unknown'

def analyze_casualties(sim_id):
    """Analyze casualties over time for a single simulation."""
    damage_events, config, id_map = load_simulation_data(sim_id)
    
    # Group damage by time and faction
    time_buckets = defaultdict(lambda: {'Blue Force': 0, 'Red Force': 0, 'Unknown': 0})
    
    for event in damage_events:
        timestamp = event['timestamp']
        damage = event['damage']
        target_id = event['target']
        
        # Determine target's faction
        faction = get_entity_faction(target_id, config, id_map)
        
        # Bucket by hour
        hour = int(timestamp / 3600)
        time_buckets[hour][faction] += damage
    
    return time_buckets

def plot_casualties_over_time(sim_id, sim_name='Operation Tropic Tortoise'):
    """Create casualties over time plot."""
    print(f"Analyzing simulation: {sim_id}")
    time_buckets = analyze_casualties(sim_id)
    
    # Convert to cumulative arrays
    max_hour = max(time_buckets.keys()) if time_buckets else 0
    hours = np.arange(0, max_hour + 1)
    
    blue_casualties = np.zeros(len(hours))
    red_casualties = np.zeros(len(hours))
    
    cumulative_blue = 0
    cumulative_red = 0
    
    for i, hour in enumerate(hours):
        cumulative_blue += time_buckets[hour]['Blue Force']
        cumulative_red += time_buckets[hour]['Red Force']
        blue_casualties[i] = cumulative_blue
        red_casualties[i] = cumulative_red
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Cumulative casualties
    ax1.plot(hours, blue_casualties, 'b-', linewidth=2, label='Blue Force Casualties')
    ax1.plot(hours, red_casualties, 'r-', linewidth=2, label='Red Force Casualties')
    ax1.fill_between(hours, 0, blue_casualties, alpha=0.3, color='blue')
    ax1.fill_between(hours, 0, red_casualties, alpha=0.3, color='red')
    
    ax1.set_xlabel('Simulation Time (hours)', fontsize=12)
    ax1.set_ylabel('Cumulative Combat Power Lost', fontsize=12)
    ax1.set_title(f'Cumulative Casualties Over Time\n{sim_name}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Add total casualties annotation
    total_blue = blue_casualties[-1] if len(blue_casualties) > 0 else 0
    total_red = red_casualties[-1] if len(red_casualties) > 0 else 0
    ax1.text(0.02, 0.98, 
             f'Total Blue Casualties: {total_blue:.1f}\nTotal Red Casualties: {total_red:.1f}',
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Casualty rate (per hour)
    blue_rate = np.diff(blue_casualties, prepend=0)
    red_rate = np.diff(red_casualties, prepend=0)
    
    ax2.bar(hours - 0.2, blue_rate, width=0.4, color='blue', alpha=0.7, label='Blue Force')
    ax2.bar(hours + 0.2, red_rate, width=0.4, color='red', alpha=0.7, label='Red Force')
    
    ax2.set_xlabel('Simulation Time (hours)', fontsize=12)
    ax2.set_ylabel('Combat Power Lost per Hour', fontsize=12)
    ax2.set_title('Casualty Rate Over Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, f'casualties_over_time_{sim_id[:8]}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    return total_blue, total_red

def analyze_multiple_simulations(num_sims=10):
    """Analyze multiple simulations and create summary statistics."""
    sim_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    sim_dirs = sim_dirs[:num_sims]
    
    results = []
    for sim_id in sim_dirs:
        try:
            blue_cas, red_cas = plot_casualties_over_time(sim_id)
            results.append({
                'sim_id': sim_id,
                'blue_casualties': blue_cas,
                'red_casualties': red_cas,
                'total_casualties': blue_cas + red_cas,
                'blue_red_ratio': blue_cas / red_cas if red_cas > 0 else float('inf')
            })
        except Exception as e:
            print(f"Error processing {sim_id}: {e}")
    
    # Create summary plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sim_numbers = range(len(results))
    blue_vals = [r['blue_casualties'] for r in results]
    red_vals = [r['red_casualties'] for r in results]
    
    x = np.arange(len(results))
    width = 0.35
    
    ax.bar(x - width/2, blue_vals, width, label='Blue Force', color='blue', alpha=0.7)
    ax.bar(x + width/2, red_vals, width, label='Red Force', color='red', alpha=0.7)
    
    ax.set_xlabel('Simulation Run', fontsize=12)
    ax.set_ylabel('Total Casualties (Combat Power Lost)', fontsize=12)
    ax.set_title(f'Casualties Across {len(results)} Simulation Runs', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Run {i+1}' for i in range(len(results))], rotation=45)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add average lines
    avg_blue = np.mean(blue_vals)
    avg_red = np.mean(red_vals)
    ax.axhline(avg_blue, color='blue', linestyle='--', alpha=0.5, label=f'Avg Blue: {avg_blue:.1f}')
    ax.axhline(avg_red, color='red', linestyle='--', alpha=0.5, label=f'Avg Red: {avg_red:.1f}')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'casualties_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved summary: {output_path}")
    
    return results

if __name__ == '__main__':
    print("=" * 60)
    print("CASUALTIES OVER TIME ANALYSIS")
    print("=" * 60)
    
    # Analyze first simulation in detail
    sim_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    if sim_dirs:
        first_sim = sim_dirs[0]
        print(f"\nDetailed analysis of first simulation...")
        plot_casualties_over_time(first_sim)
    
    # Analyze multiple simulations
    print(f"\n\nAnalyzing multiple simulations...")
    results = analyze_multiple_simulations(num_sims=10)
    
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Simulations analyzed: {len(results)}")
    print(f"Average Blue casualties: {np.mean([r['blue_casualties'] for r in results]):.1f}")
    print(f"Average Red casualties: {np.mean([r['red_casualties'] for r in results]):.1f}")
    print(f"Average casualty ratio (Blue/Red): {np.mean([r['blue_red_ratio'] for r in results if r['blue_red_ratio'] != float('inf')]):.2f}")
    print("=" * 60)
