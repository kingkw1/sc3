"""
COA Comparison Analysis
=======================

This script compares outcomes across different Courses of Action (COAs) to
identify which strategies led to better performance. Links web sessions to
simulation outcomes.

Outputs:
    - coa_performance_comparison.png: Comparison of outcomes across COAs
    - coa_development_time.png: Time spent developing each COA
    - outcome_metrics.png: Multiple performance metrics per COA

Metrics:
    - Casualties by COA
    - Final force ratios
    - Simulation duration to combat end
    - COA development time correlation
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime

# Configuration
SIM_DIR = '../data/sim'
WEB_DIR = '../data/web'
OUTPUT_DIR = './outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_coa_ids_from_web():
    """Extract COA IDs and development times from web data."""
    coa_data = {}
    
    web_sessions = ['TS-ARL3538', 'TS-ARL3542', 'TS-ARL3543']
    
    for session in web_sessions:
        web_file = os.path.join(WEB_DIR, session, f'{session}.jsonl')
        
        if not os.path.exists(web_file):
            continue
        
        print(f"Processing web session: {session}")
        
        coa_times = defaultdict(list)
        
        with open(web_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    coa_id = data.get('coa_id')
                    timestamp = data.get('timeStamp')
                    
                    if coa_id and timestamp:
                        coa_times[coa_id].append(timestamp)
                except:
                    continue
        
        # Calculate development time for each COA
        for coa_id, timestamps in coa_times.items():
            if timestamps:
                start_time = min(timestamps)
                end_time = max(timestamps)
                duration_minutes = (end_time - start_time) / 1000 / 60
                
                coa_data[coa_id] = {
                    'session': session,
                    'development_time_minutes': duration_minutes,
                    'start_time': start_time,
                    'end_time': end_time,
                    'interactions': len(timestamps)
                }
    
    return coa_data

def analyze_simulation_outcomes():
    """Analyze outcomes for all simulations."""
    sim_dirs = [d for d in os.listdir(SIM_DIR) if os.path.isdir(os.path.join(SIM_DIR, d))]
    
    outcomes = []
    
    for i, sim_id in enumerate(sim_dirs):
        if i >= 30:  # Limit for performance
            break
        
        try:
            sim_path = os.path.join(SIM_DIR, sim_id)
            
            # Load damage events
            with open(os.path.join(sim_path, 'damage_events.json')) as f:
                damage_events = json.load(f)
            
            # Load config
            with open(os.path.join(sim_path, 'config.json')) as f:
                config = json.load(f)
            
            # Load ID map
            with open(os.path.join(sim_path, 'external_id_to_internal_id.json')) as f:
                id_map = json.load(f)
            
            # Load results
            with open(os.path.join(sim_path, 'results.json')) as f:
                results = json.load(f)
            
            # Calculate metrics
            blue_faction_id = None
            red_faction_id = None
            
            for faction in config.get('factions', []):
                if 'Blue' in faction['header']['name']:
                    blue_faction_id = faction['header']['id']
                elif 'Red' in faction['header']['name']:
                    red_faction_id = faction['header']['id']
            
            # Initial combat power
            blue_initial = 0
            red_initial = 0
            entity_factions = {}
            
            for entity in config.get('entities', []):
                if entity['type'] == 'Unit':
                    ext_id = entity['data']['header']['id']
                    int_id = id_map.get(ext_id)
                    faction_id = entity['data'].get('faction')
                    initial_cp = entity['data'].get('combat_power', {}).get('initial', 0)
                    
                    if faction_id == blue_faction_id:
                        blue_initial += initial_cp
                        if int_id:
                            entity_factions[int_id] = 'Blue'
                    elif faction_id == red_faction_id:
                        red_initial += initial_cp
                        if int_id:
                            entity_factions[int_id] = 'Red'
            
            # Calculate casualties
            blue_casualties = 0
            red_casualties = 0
            
            for event in damage_events:
                target_id = event['target']
                damage = event['damage']
                
                if target_id in entity_factions:
                    if entity_factions[target_id] == 'Blue':
                        blue_casualties += damage
                    elif entity_factions[target_id] == 'Red':
                        red_casualties += damage
            
            # Last combat event time
            last_combat_time = max([e['timestamp'] for e in damage_events]) / 3600 if damage_events else 0
            
            outcomes.append({
                'sim_id': sim_id,
                'blue_initial': blue_initial,
                'red_initial': red_initial,
                'blue_casualties': blue_casualties,
                'red_casualties': red_casualties,
                'blue_final': blue_initial - blue_casualties,
                'red_final': red_initial - red_casualties,
                'last_combat_hour': last_combat_time,
                'total_casualties': blue_casualties + red_casualties,
                'casualty_ratio': blue_casualties / red_casualties if red_casualties > 0 else 0
            })
            
        except Exception as e:
            print(f"Error processing {sim_id}: {e}")
            continue
    
    return outcomes

def plot_simulation_comparison():
    """Create comprehensive comparison plots."""
    print("Analyzing all simulations...")
    outcomes = analyze_simulation_outcomes()
    
    print(f"Analyzed {len(outcomes)} simulations")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Casualties comparison
    ax1 = fig.add_subplot(gs[0, :])
    
    x = np.arange(len(outcomes))
    width = 0.35
    
    blue_cas = [o['blue_casualties'] for o in outcomes]
    red_cas = [o['red_casualties'] for o in outcomes]
    
    ax1.bar(x - width/2, blue_cas, width, label='Blue Casualties', color='blue', alpha=0.7)
    ax1.bar(x + width/2, red_cas, width, label='Red Casualties', color='red', alpha=0.7)
    
    # Add averages
    avg_blue = np.mean(blue_cas)
    avg_red = np.mean(red_cas)
    ax1.axhline(avg_blue, color='blue', linestyle='--', alpha=0.5, linewidth=2)
    ax1.axhline(avg_red, color='red', linestyle='--', alpha=0.5, linewidth=2)
    
    ax1.set_xlabel('Simulation Run', fontsize=12)
    ax1.set_ylabel('Casualties (Combat Power Lost)', fontsize=12)
    ax1.set_title('Casualties Across All Simulations', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add statistics box
    ax1.text(0.02, 0.98, 
             f'Avg Blue: {avg_blue:.1f}\nAvg Red: {avg_red:.1f}\n'
             f'Std Blue: {np.std(blue_cas):.1f}\nStd Red: {np.std(red_cas):.1f}',
             transform=ax1.transAxes, fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Plot 2: Final force ratios
    ax2 = fig.add_subplot(gs[1, 0])
    
    final_ratios = [o['blue_final'] / o['red_final'] if o['red_final'] > 0 else 0 
                    for o in outcomes]
    
    ax2.hist(final_ratios, bins=20, color='green', alpha=0.7, edgecolor='darkgreen')
    ax2.axvline(1.0, color='black', linestyle='--', linewidth=2, label='Equal Strength')
    ax2.axvline(np.median(final_ratios), color='red', linestyle='-', linewidth=2, 
                label=f'Median: {np.median(final_ratios):.2f}')
    
    ax2.set_xlabel('Final Force Ratio (Blue/Red)', fontsize=11)
    ax2.set_ylabel('Number of Simulations', fontsize=11)
    ax2.set_title('Distribution of Final Force Ratios', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Combat duration
    ax3 = fig.add_subplot(gs[1, 1])
    
    combat_hours = [o['last_combat_hour'] for o in outcomes]
    
    ax3.hist(combat_hours, bins=20, color='orange', alpha=0.7, edgecolor='darkorange')
    ax3.axvline(np.mean(combat_hours), color='red', linestyle='-', linewidth=2,
                label=f'Mean: {np.mean(combat_hours):.1f}h')
    
    ax3.set_xlabel('Last Combat Event (hours)', fontsize=11)
    ax3.set_ylabel('Number of Simulations', fontsize=11)
    ax3.set_title('Combat Duration Distribution', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Casualty ratio distribution
    ax4 = fig.add_subplot(gs[2, 0])
    
    casualty_ratios = [o['casualty_ratio'] for o in outcomes if o['casualty_ratio'] > 0]
    
    ax4.hist(casualty_ratios, bins=20, color='purple', alpha=0.7, edgecolor='indigo')
    ax4.axvline(1.0, color='black', linestyle='--', linewidth=2, label='Equal Casualties')
    ax4.axvline(np.median(casualty_ratios), color='red', linestyle='-', linewidth=2,
                label=f'Median: {np.median(casualty_ratios):.2f}')
    
    ax4.set_xlabel('Casualty Ratio (Blue/Red)', fontsize=11)
    ax4.set_ylabel('Number of Simulations', fontsize=11)
    ax4.set_title('Casualty Ratio Distribution', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Performance scatter
    ax5 = fig.add_subplot(gs[2, 1])
    
    ax5.scatter(blue_cas, red_cas, alpha=0.6, s=100, c=combat_hours, 
                cmap='viridis', edgecolors='black', linewidth=0.5)
    
    # Add diagonal line (equal casualties)
    max_cas = max(max(blue_cas), max(red_cas))
    ax5.plot([0, max_cas], [0, max_cas], 'k--', alpha=0.3, linewidth=2, label='Equal Casualties')
    
    ax5.set_xlabel('Blue Casualties', fontsize=11)
    ax5.set_ylabel('Red Casualties', fontsize=11)
    ax5.set_title('Casualty Correlation', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(ax5.collections[0], ax=ax5)
    cbar.set_label('Combat Duration (hours)', fontsize=9)
    
    plt.suptitle('Simulation Outcomes Comparison', fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, 'simulation_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    return outcomes

def plot_coa_analysis():
    """Analyze COA development patterns."""
    print("\nAnalyzing COA development...")
    coa_data = extract_coa_ids_from_web()
    
    if not coa_data:
        print("No COA data found in web sessions")
        return
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Development time per COA
    coa_ids = list(coa_data.keys())
    dev_times = [coa_data[coa]['development_time_minutes'] for coa in coa_ids]
    sessions = [coa_data[coa]['session'] for coa in coa_ids]
    
    colors = {'TS-ARL3538': 'blue', 'TS-ARL3542': 'red', 'TS-ARL3543': 'green'}
    bar_colors = [colors.get(s, 'gray') for s in sessions]
    
    x_pos = np.arange(len(coa_ids))
    ax1.bar(x_pos, dev_times, color=bar_colors, alpha=0.7, edgecolor='black')
    
    ax1.set_xlabel('COA Number', fontsize=12)
    ax1.set_ylabel('Development Time (minutes)', fontsize=12)
    ax1.set_title('COA Development Time', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'COA {i+1}' for i in range(len(coa_ids))], rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[s], alpha=0.7, label=s) 
                      for s in colors.keys()]
    ax1.legend(handles=legend_elements, fontsize=9)
    
    # Plot 2: Interactions per COA
    interactions = [coa_data[coa]['interactions'] for coa in coa_ids]
    
    ax2.bar(x_pos, interactions, color=bar_colors, alpha=0.7, edgecolor='black')
    
    ax2.set_xlabel('COA Number', fontsize=12)
    ax2.set_ylabel('Number of Interactions', fontsize=12)
    ax2.set_title('User Interactions per COA', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'COA {i+1}' for i in range(len(coa_ids))], rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, 'coa_development.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print summary
    print("\n" + "-" * 60)
    print("COA DEVELOPMENT SUMMARY")
    print("-" * 60)
    print(f"Total COAs: {len(coa_data)}")
    print(f"Average development time: {np.mean(dev_times):.1f} minutes")
    print(f"Average interactions: {np.mean(interactions):.0f}")
    
    for i, coa_id in enumerate(coa_ids):
        print(f"\nCOA {i+1} ({coa_data[coa_id]['session']}):")
        print(f"  Development time: {coa_data[coa_id]['development_time_minutes']:.1f} min")
        print(f"  Interactions: {coa_data[coa_id]['interactions']:,}")

if __name__ == '__main__':
    print("=" * 60)
    print("COA COMPARISON ANALYSIS")
    print("=" * 60)
    
    # Analyze simulations
    outcomes = plot_simulation_comparison()
    
    # Analyze COA development
    plot_coa_analysis()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
