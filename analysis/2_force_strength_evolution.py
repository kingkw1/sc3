"""
Force Strength Evolution Analysis
==================================

This script tracks how combat power evolves for Red and Blue forces throughout
the simulation. Shows the balance of power over time and identifies critical
turning points in the engagement.

Outputs:
    - force_strength_evolution.png: Line plot of remaining combat power over time
    - force_ratio_over_time.png: Blue/Red force ratio throughout simulation
    - power_balance.png: Stacked area chart showing force composition

Metrics:
    - Remaining combat power at each timestep
    - Force ratio (Blue/Red) evolution
    - Rate of combat power degradation
    - Identification of decisive moments (when ratio changes significantly)
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
    """Load all necessary data for a simulation."""
    sim_path = os.path.join(DATA_DIR, sim_id)
    
    with open(os.path.join(sim_path, 'damage_events.json')) as f:
        damage_events = json.load(f)
    
    with open(os.path.join(sim_path, 'config.json')) as f:
        config = json.load(f)
    
    with open(os.path.join(sim_path, 'external_id_to_internal_id.json')) as f:
        id_map = json.load(f)
    
    with open(os.path.join(sim_path, 'results.json')) as f:
        results = json.load(f)
    
    return damage_events, config, id_map, results

def get_initial_combat_power(config, id_map):
    """Extract initial combat power for all entities by faction."""
    blue_force_id = None
    red_force_id = None
    
    # Get faction IDs
    for faction in config.get('factions', []):
        if 'Blue' in faction['header']['name']:
            blue_force_id = faction['header']['id']
        elif 'Red' in faction['header']['name']:
            red_force_id = faction['header']['id']
    
    blue_power = 0
    red_power = 0
    entity_faction = {}  # Map internal ID to faction
    
    for entity in config.get('entities', []):
        if entity['type'] == 'Unit':
            external_id = entity['data']['header']['id']
            internal_id = id_map.get(external_id)
            
            if internal_id:
                faction_id = entity['data'].get('faction')
                initial_power = entity['data'].get('combat_power', {}).get('initial', 0)
                
                if faction_id == blue_force_id:
                    blue_power += initial_power
                    entity_faction[internal_id] = 'Blue'
                elif faction_id == red_force_id:
                    red_power += initial_power
                    entity_faction[internal_id] = 'Red'
    
    return blue_power, red_power, entity_faction

def calculate_force_strength_over_time(sim_id):
    """Calculate remaining combat power at each timestep."""
    damage_events, config, id_map, results = load_simulation_data(sim_id)
    
    # Get initial values
    initial_blue, initial_red, entity_faction = get_initial_combat_power(config, id_map)
    
    # Create time series
    max_time = config.get('max_scenario_time', 345600)
    timestep = config.get('timestep', 60)
    time_points = np.arange(0, max_time + timestep, timestep)
    
    blue_strength = np.full(len(time_points), initial_blue, dtype=float)
    red_strength = np.full(len(time_points), initial_red, dtype=float)
    
    # Process damage events
    for event in damage_events:
        timestamp = event['timestamp']
        damage = event['damage']
        target_id = event['target']
        
        # Find the time index
        time_idx = int(timestamp / timestep)
        if time_idx < len(time_points):
            # Determine which force was damaged
            faction = entity_faction.get(target_id, 'Unknown')
            
            if faction == 'Blue':
                # Apply damage from this point forward
                blue_strength[time_idx:] -= damage
            elif faction == 'Red':
                red_strength[time_idx:] -= damage
    
    # Ensure non-negative values
    blue_strength = np.maximum(blue_strength, 0)
    red_strength = np.maximum(red_strength, 0)
    
    # Convert time to hours
    time_hours = time_points / 3600
    
    return time_hours, blue_strength, red_strength, initial_blue, initial_red

def plot_force_strength_evolution(sim_id, sim_name='Operation Tropic Tortoise'):
    """Create comprehensive force strength plots."""
    print(f"Analyzing force strength evolution: {sim_id}")
    
    time_hours, blue_strength, red_strength, initial_blue, initial_red = \
        calculate_force_strength_over_time(sim_id)
    
    # Create multi-panel figure
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Force strength over time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_hours, blue_strength, 'b-', linewidth=2.5, label='Blue Force')
    ax1.plot(time_hours, red_strength, 'r-', linewidth=2.5, label='Red Force')
    ax1.fill_between(time_hours, 0, blue_strength, alpha=0.2, color='blue')
    ax1.fill_between(time_hours, 0, red_strength, alpha=0.2, color='red')
    
    ax1.set_xlabel('Simulation Time (hours)', fontsize=12)
    ax1.set_ylabel('Remaining Combat Power', fontsize=12)
    ax1.set_title(f'Force Strength Evolution - {sim_name}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Add initial strength annotations
    ax1.text(0.02, 0.98, 
             f'Initial Blue: {initial_blue:.1f}\nInitial Red: {initial_red:.1f}',
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Plot 2: Force ratio over time
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Calculate ratio (avoid division by zero)
    ratio = np.divide(blue_strength, red_strength, 
                      out=np.zeros_like(blue_strength), 
                      where=red_strength!=0)
    
    ax2.plot(time_hours, ratio, 'g-', linewidth=2)
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Equal Strength')
    ax2.fill_between(time_hours, 1, ratio, where=(ratio >= 1), 
                     alpha=0.3, color='blue', label='Blue Advantage')
    ax2.fill_between(time_hours, ratio, 1, where=(ratio < 1), 
                     alpha=0.3, color='red', label='Red Advantage')
    
    ax2.set_xlabel('Simulation Time (hours)', fontsize=11)
    ax2.set_ylabel('Force Ratio (Blue/Red)', fontsize=11)
    ax2.set_title('Balance of Power', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    # Plot 3: Percentage of initial strength
    ax3 = fig.add_subplot(gs[1, 1])
    
    blue_percent = (blue_strength / initial_blue * 100) if initial_blue > 0 else np.zeros_like(blue_strength)
    red_percent = (red_strength / initial_red * 100) if initial_red > 0 else np.zeros_like(red_strength)
    
    ax3.plot(time_hours, blue_percent, 'b-', linewidth=2, label='Blue Force')
    ax3.plot(time_hours, red_percent, 'r-', linewidth=2, label='Red Force')
    
    ax3.set_xlabel('Simulation Time (hours)', fontsize=11)
    ax3.set_ylabel('Strength (% of Initial)', fontsize=11)
    ax3.set_title('Attrition Rate', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 105])
    
    # Plot 4: Stacked area chart
    ax4 = fig.add_subplot(gs[2, :])
    
    ax4.fill_between(time_hours, 0, blue_strength, alpha=0.6, color='blue', label='Blue Force')
    ax4.fill_between(time_hours, blue_strength, blue_strength + red_strength, 
                     alpha=0.6, color='red', label='Red Force')
    
    ax4.set_xlabel('Simulation Time (hours)', fontsize=12)
    ax4.set_ylabel('Total Combat Power', fontsize=12)
    ax4.set_title('Combined Force Composition', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, f'force_strength_{sim_id[:8]}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Calculate summary statistics
    final_blue = blue_strength[-1]
    final_red = red_strength[-1]
    blue_loss = initial_blue - final_blue
    red_loss = initial_red - final_red
    
    return {
        'initial_blue': initial_blue,
        'initial_red': initial_red,
        'final_blue': final_blue,
        'final_red': final_red,
        'blue_loss_pct': (blue_loss / initial_blue * 100) if initial_blue > 0 else 0,
        'red_loss_pct': (red_loss / initial_red * 100) if initial_red > 0 else 0
    }

def analyze_decisive_moments(sim_id):
    """Identify key moments where force balance shifted."""
    time_hours, blue_strength, red_strength, initial_blue, initial_red = \
        calculate_force_strength_over_time(sim_id)
    
    # Calculate ratio
    ratio = np.divide(blue_strength, red_strength, 
                      out=np.ones_like(blue_strength), 
                      where=red_strength!=0)
    
    # Find significant changes (derivative)
    ratio_change = np.abs(np.diff(ratio))
    
    # Find top 5 moments of greatest change
    top_indices = np.argsort(ratio_change)[-5:][::-1]
    
    decisive_moments = []
    for idx in top_indices:
        if ratio_change[idx] > 0.01:  # Threshold for significance
            decisive_moments.append({
                'time_hours': time_hours[idx],
                'blue_strength': blue_strength[idx],
                'red_strength': red_strength[idx],
                'ratio_change': ratio_change[idx]
            })
    
    return decisive_moments

if __name__ == '__main__':
    print("=" * 60)
    print("FORCE STRENGTH EVOLUTION ANALYSIS")
    print("=" * 60)
    
    # Get simulation directories
    sim_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    if sim_dirs:
        # Analyze first simulation in detail
        first_sim = sim_dirs[0]
        print(f"\nAnalyzing simulation: {first_sim[:16]}...")
        stats = plot_force_strength_evolution(first_sim)
        
        print("\n" + "-" * 60)
        print("SUMMARY STATISTICS")
        print("-" * 60)
        print(f"Initial Blue Force: {stats['initial_blue']:.1f}")
        print(f"Initial Red Force: {stats['initial_red']:.1f}")
        print(f"Final Blue Force: {stats['final_blue']:.1f}")
        print(f"Final Red Force: {stats['final_red']:.1f}")
        print(f"Blue Force Losses: {stats['blue_loss_pct']:.1f}%")
        print(f"Red Force Losses: {stats['red_loss_pct']:.1f}%")
        
        # Analyze decisive moments
        print("\n" + "-" * 60)
        print("DECISIVE MOMENTS")
        print("-" * 60)
        moments = analyze_decisive_moments(first_sim)
        for i, moment in enumerate(moments, 1):
            print(f"{i}. Hour {moment['time_hours']:.1f}: "
                  f"Blue={moment['blue_strength']:.1f}, "
                  f"Red={moment['red_strength']:.1f}")
        
        print("\n" + "=" * 60)
