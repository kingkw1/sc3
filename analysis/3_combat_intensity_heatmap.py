"""
Combat Intensity Heatmap Analysis
==================================

This script analyzes combat intensity throughout the simulation, showing when
and where fighting occurred most heavily. Creates temporal heatmaps and 
identifies peak combat periods.

Outputs:
    - combat_intensity_heatmap.png: Heatmap showing combat events over time
    - combat_hotspots.png: Geographic heatmap of combat locations
    - combat_phases.png: Identification of distinct combat phases

Metrics:
    - Number of combat events per time period
    - Damage dealt per time period
    - Geographic distribution of combat
    - Combat phase identification (engagement, peak, decline)
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.ndimage import gaussian_filter

# Configuration
DATA_DIR = '../data/sim'
OUTPUT_DIR = './outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_simulation_data(sim_id):
    """Load combat and damage events."""
    sim_path = os.path.join(DATA_DIR, sim_id)
    
    with open(os.path.join(sim_path, 'combat_events.json')) as f:
        combat_events = json.load(f)
    
    with open(os.path.join(sim_path, 'damage_events.json')) as f:
        damage_events = json.load(f)
    
    with open(os.path.join(sim_path, 'config.json')) as f:
        config = json.load(f)
    
    return combat_events, damage_events, config

def analyze_combat_intensity(sim_id, time_resolution_minutes=30):
    """Analyze combat intensity over time."""
    combat_events, damage_events, config = load_simulation_data(sim_id)
    
    max_time = config.get('max_scenario_time', 345600)
    time_resolution_seconds = time_resolution_minutes * 60
    num_bins = int(max_time / time_resolution_seconds) + 1
    
    # Initialize arrays
    combat_counts = np.zeros(num_bins)
    damage_totals = np.zeros(num_bins)
    
    # Count combat events
    for event in combat_events:
        timestamp = event['timestamp']
        bin_idx = int(timestamp / time_resolution_seconds)
        if bin_idx < num_bins:
            combat_counts[bin_idx] += len(event.get('targets', []))
    
    # Sum damage
    for event in damage_events:
        timestamp = event['timestamp']
        bin_idx = int(timestamp / time_resolution_seconds)
        if bin_idx < num_bins:
            damage_totals[bin_idx] += event['damage']
    
    # Create time labels (in hours)
    time_labels = np.arange(num_bins) * (time_resolution_minutes / 60)
    
    return time_labels, combat_counts, damage_totals

def identify_combat_phases(combat_counts, damage_totals, threshold_percentile=10):
    """Identify distinct phases of combat."""
    # Smooth the data
    smoothed_combat = gaussian_filter(combat_counts, sigma=2)
    smoothed_damage = gaussian_filter(damage_totals, sigma=2)
    
    # Find periods of significant activity
    combat_threshold = np.percentile(smoothed_combat[smoothed_combat > 0], threshold_percentile)
    damage_threshold = np.percentile(smoothed_damage[smoothed_damage > 0], threshold_percentile)
    
    # Identify phases
    active = (smoothed_combat > combat_threshold) | (smoothed_damage > damage_threshold)
    
    # Find continuous active periods
    phases = []
    in_phase = False
    phase_start = 0
    
    for i, is_active in enumerate(active):
        if is_active and not in_phase:
            phase_start = i
            in_phase = True
        elif not is_active and in_phase:
            phases.append((phase_start, i - 1))
            in_phase = False
    
    if in_phase:
        phases.append((phase_start, len(active) - 1))
    
    return phases

def plot_combat_intensity(sim_id, sim_name='Operation Tropic Tortoise'):
    """Create comprehensive combat intensity visualizations."""
    print(f"Analyzing combat intensity: {sim_id}")
    
    time_labels, combat_counts, damage_totals = analyze_combat_intensity(sim_id, time_resolution_minutes=30)
    
    # Identify phases
    phases = identify_combat_phases(combat_counts, damage_totals)
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: Combat events over time
    ax1 = axes[0]
    ax1.bar(time_labels, combat_counts, width=0.4, color='orange', alpha=0.7, edgecolor='darkorange')
    
    # Highlight combat phases
    for phase_start, phase_end in phases:
        ax1.axvspan(time_labels[phase_start], time_labels[phase_end], 
                   alpha=0.2, color='red', label='Active Combat' if phase_start == phases[0][0] else '')
    
    ax1.set_xlabel('Simulation Time (hours)', fontsize=12)
    ax1.set_ylabel('Combat Events per 30 min', fontsize=12)
    ax1.set_title(f'Combat Intensity Over Time - {sim_name}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    if phases:
        ax1.legend(fontsize=10)
    
    # Add statistics
    total_events = int(combat_counts.sum())
    peak_hour = time_labels[np.argmax(combat_counts)]
    ax1.text(0.02, 0.98, 
             f'Total Events: {total_events:,}\nPeak: Hour {peak_hour:.1f}\nPhases: {len(phases)}',
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Plot 2: Damage dealt over time
    ax2 = axes[1]
    ax2.bar(time_labels, damage_totals, width=0.4, color='red', alpha=0.7, edgecolor='darkred')
    
    # Add moving average
    window = 5
    if len(damage_totals) >= window:
        moving_avg = np.convolve(damage_totals, np.ones(window)/window, mode='valid')
        ma_time = time_labels[window-1:]
        ax2.plot(ma_time, moving_avg, 'b-', linewidth=2.5, label=f'{window*30}-min Moving Avg')
        ax2.legend(fontsize=10)
    
    ax2.set_xlabel('Simulation Time (hours)', fontsize=12)
    ax2.set_ylabel('Damage Dealt per 30 min', fontsize=12)
    ax2.set_title('Damage Intensity Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    total_damage = damage_totals.sum()
    peak_damage_hour = time_labels[np.argmax(damage_totals)]
    ax2.text(0.02, 0.98, 
             f'Total Damage: {total_damage:.1f}\nPeak: Hour {peak_damage_hour:.1f}',
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    # Plot 3: Combined heatmap
    ax3 = axes[2]
    
    # Normalize both metrics to 0-1 for comparison
    norm_combat = combat_counts / combat_counts.max() if combat_counts.max() > 0 else combat_counts
    norm_damage = damage_totals / damage_totals.max() if damage_totals.max() > 0 else damage_totals
    
    # Create stacked area
    ax3.fill_between(time_labels, 0, norm_combat, alpha=0.5, color='orange', label='Combat Events (norm)')
    ax3.fill_between(time_labels, 0, norm_damage, alpha=0.5, color='red', label='Damage (norm)')
    
    ax3.set_xlabel('Simulation Time (hours)', fontsize=12)
    ax3.set_ylabel('Normalized Intensity', fontsize=12)
    ax3.set_title('Combat Activity Profile', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.1])
    
    # Annotate phases
    for i, (phase_start, phase_end) in enumerate(phases):
        mid_point = (time_labels[phase_start] + time_labels[phase_end]) / 2
        ax3.annotate(f'Phase {i+1}', xy=(mid_point, 1.05), 
                    ha='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, f'combat_intensity_{sim_id[:8]}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    return {
        'total_events': int(combat_counts.sum()),
        'total_damage': float(damage_totals.sum()),
        'num_phases': len(phases),
        'peak_combat_hour': float(time_labels[np.argmax(combat_counts)]),
        'peak_damage_hour': float(time_labels[np.argmax(damage_totals)])
    }

def analyze_geographic_intensity(sim_id):
    """Analyze geographic distribution of combat."""
    combat_events, damage_events, config = load_simulation_data(sim_id)
    
    # Get entity positions from config
    entity_positions = {}
    for entity in config.get('entities', []):
        if entity['type'] == 'Unit':
            entity_id = entity['data']['header']['id']
            pos = entity['data']['position']
            entity_positions[entity_id] = (pos['x'], pos['y'])
    
    # Map internal IDs to positions (using config)
    sim_path = os.path.join(DATA_DIR, sim_id)
    with open(os.path.join(sim_path, 'external_id_to_internal_id.json')) as f:
        id_map = json.load(f)
    
    internal_positions = {}
    for ext_id, int_id in id_map.items():
        if ext_id in entity_positions:
            internal_positions[int_id] = entity_positions[ext_id]
    
    # Collect combat locations
    combat_locations = []
    for event in damage_events[:1000]:  # Sample for performance
        target_id = event['target']
        if target_id in internal_positions:
            combat_locations.append(internal_positions[target_id])
    
    if not combat_locations:
        print("No geographic data available for combat locations")
        return None
    
    # Create heatmap
    lons = [loc[0] for loc in combat_locations]
    lats = [loc[1] for loc in combat_locations]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create 2D histogram
    heatmap, xedges, yedges = np.histogram2d(lons, lats, bins=20)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='YlOrRd', aspect='auto')
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Geographic Combat Intensity Heatmap', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Combat Events', fontsize=11)
    
    # Add AOI boundary if available
    aoi = config.get('aoi', {}).get('exterior', [])
    if aoi:
        aoi_lons = [p['x'] for p in aoi] + [aoi[0]['x']]
        aoi_lats = [p['y'] for p in aoi] + [aoi[0]['y']]
        ax.plot(aoi_lons, aoi_lats, 'b-', linewidth=2, label='Area of Interest')
        ax.legend(fontsize=10)
    
    output_path = os.path.join(OUTPUT_DIR, f'combat_geographic_{sim_id[:8]}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    return len(combat_locations)

if __name__ == '__main__':
    print("=" * 60)
    print("COMBAT INTENSITY ANALYSIS")
    print("=" * 60)
    
    # Get simulation directories
    sim_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    if sim_dirs:
        # Analyze first simulation
        first_sim = sim_dirs[0]
        print(f"\nAnalyzing simulation: {first_sim[:16]}...")
        stats = plot_combat_intensity(first_sim)
        
        print("\n" + "-" * 60)
        print("COMBAT INTENSITY STATISTICS")
        print("-" * 60)
        print(f"Total Combat Events: {stats['total_events']:,}")
        print(f"Total Damage Dealt: {stats['total_damage']:.1f}")
        print(f"Number of Combat Phases: {stats['num_phases']}")
        print(f"Peak Combat Activity: Hour {stats['peak_combat_hour']:.1f}")
        print(f"Peak Damage Period: Hour {stats['peak_damage_hour']:.1f}")
        
        # Analyze geographic distribution
        print("\n" + "-" * 60)
        print("GEOGRAPHIC ANALYSIS")
        print("-" * 60)
        num_locations = analyze_geographic_intensity(first_sim)
        if num_locations:
            print(f"Combat locations analyzed: {num_locations}")
        
        print("\n" + "=" * 60)
