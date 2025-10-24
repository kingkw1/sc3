"""
Completion Time vs Performance Analysis
========================================

This script analyzes whether longer mission completion times correlate with
better performance outcomes:
1. Kill ratio (enemy casualties vs friendly)
2. Blue Force survival
3. Mission success
4. Battle complexity

Answers: Do COAs that take longer perform better?

Outputs:
    - completion_time_vs_performance.png: Correlation analysis
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Configuration
DATA_DIR = '../data/sim'
OUTPUT_DIR = './outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_simulation_performance_and_time():
    """Extract both completion time and performance metrics."""
    sim_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    results = []
    for sim_id in sim_dirs[:30]:
        sim_path = os.path.join(DATA_DIR, sim_id)
        
        try:
            with open(os.path.join(sim_path, 'config.json')) as f:
                config = json.load(f)
            
            with open(os.path.join(sim_path, 'damage_events.json')) as f:
                damage_events = json.load(f)
            
            with open(os.path.join(sim_path, 'external_id_to_internal_id.json')) as f:
                id_map = json.load(f)
            
            # Get faction IDs
            blue_faction_id = None
            red_faction_id = None
            
            for faction in config.get('factions', []):
                if 'Blue' in faction['header']['name']:
                    blue_faction_id = faction['header']['id']
                elif 'Red' in faction['header']['name']:
                    red_faction_id = faction['header']['id']
            
            # Map entities to factions and get initial strengths
            entity_factions = {}
            blue_initial = 0
            red_initial = 0
            
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
            last_combat_time = 0
            
            for event in damage_events:
                target_id = event['target']
                damage = event['damage']
                timestamp = event['timestamp']
                
                if timestamp > last_combat_time:
                    last_combat_time = timestamp
                
                if target_id in entity_factions:
                    if entity_factions[target_id] == 'Blue':
                        blue_casualties += damage
                    elif entity_factions[target_id] == 'Red':
                        red_casualties += damage
            
            # Calculate completion time
            completion_time_hours = last_combat_time / 3600 if last_combat_time > 0 else 0
            
            # Calculate metrics
            kill_ratio = red_casualties / blue_casualties if blue_casualties > 0 else float('inf')
            blue_survival_rate = ((blue_initial - blue_casualties) / blue_initial * 100) if blue_initial > 0 else 0
            red_survival_rate = ((red_initial - red_casualties) / red_initial * 100) if red_initial > 0 else 0
            
            # Final force ratio
            blue_final = blue_initial - blue_casualties
            red_final = red_initial - red_casualties
            force_ratio_final = blue_final / red_final if red_final > 0 else float('inf')
            
            # Battle complexity (total events, total damage)
            battle_complexity = len(damage_events)
            total_damage = blue_casualties + red_casualties
            
            # Get COA tasks
            total_tasks = 0
            for entity in config.get('entities', []):
                if entity['type'] == 'Unit':
                    faction_id = entity['data'].get('faction')
                    if faction_id == blue_faction_id:
                        tasks = entity['data'].get('tasks', [])
                        total_tasks += len(tasks)
            
            results.append({
                'sim_id': sim_id,
                'completion_time_hours': completion_time_hours,
                'kill_ratio': kill_ratio if kill_ratio != float('inf') else 10.0,  # Cap for visualization
                'blue_casualties': blue_casualties,
                'red_casualties': red_casualties,
                'blue_survival_rate': blue_survival_rate,
                'red_survival_rate': red_survival_rate,
                'force_ratio_final': force_ratio_final if force_ratio_final != float('inf') else 10.0,
                'battle_complexity': battle_complexity,
                'total_damage': total_damage,
                'total_tasks': total_tasks,
                'won': force_ratio_final > 1.0
            })
            
        except Exception as e:
            print(f"Error loading {sim_id}: {e}")
    
    return results

def plot_completion_vs_performance(results):
    """Create correlation plots showing if longer missions perform better."""
    print(f"Creating completion time vs performance analysis...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35)
    
    # Extract data
    completion_times = np.array([r['completion_time_hours'] for r in results])
    kill_ratios = np.array([r['kill_ratio'] for r in results])
    blue_survival = np.array([r['blue_survival_rate'] for r in results])
    battle_complexity = np.array([r['battle_complexity'] for r in results])
    total_tasks = np.array([r['total_tasks'] for r in results])
    won = np.array([r['won'] for r in results])
    
    # Plot 1: Completion Time vs Kill Ratio
    ax1 = fig.add_subplot(gs[0, 0])
    
    colors = ['green' if w else 'red' for w in won]
    scatter1 = ax1.scatter(completion_times, kill_ratios, s=150, c=colors, 
                          alpha=0.6, edgecolors='black', linewidth=1.5)
    
    # Add trend line
    if len(completion_times) > 1:
        z = np.polyfit(completion_times, kill_ratios, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(completion_times.min(), completion_times.max(), 100)
        ax1.plot(x_trend, p(x_trend), "b--", linewidth=2.5, alpha=0.7,
                label=f'Trend: {"Improving ✓" if z[0] > 0 else "Declining"}')
        
        # Calculate correlation
        corr = np.corrcoef(completion_times, kill_ratios)[0, 1]
        
        # Add correlation text
        corr_text = f'Correlation: {corr:.3f}\n'
        if abs(corr) < 0.3:
            corr_text += 'Weak relationship'
        elif abs(corr) < 0.7:
            corr_text += 'Moderate relationship'
        else:
            corr_text += 'Strong relationship'
        
        ax1.text(0.02, 0.98, corr_text, transform=ax1.transAxes,
                fontsize=11, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.set_xlabel('Mission Completion Time (Hours)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Kill Ratio (Red/Blue)', fontsize=11, fontweight='bold')
    ax1.set_title('Completion Time vs Enemy Casualties Ratio\n(Higher Kill Ratio = Better Performance)', 
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Completion Time vs Blue Survival
    ax2 = fig.add_subplot(gs[0, 1])
    
    scatter2 = ax2.scatter(completion_times, blue_survival, s=150, c=colors, 
                          alpha=0.6, edgecolors='black', linewidth=1.5)
    
    # Add trend line
    if len(completion_times) > 1:
        z = np.polyfit(completion_times, blue_survival, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(completion_times.min(), completion_times.max(), 100)
        ax2.plot(x_trend, p(x_trend), "b--", linewidth=2.5, alpha=0.7,
                label=f'Trend: {"Better ✓" if z[0] > 0 else "Worse"}')
        
        # Calculate correlation
        corr = np.corrcoef(completion_times, blue_survival)[0, 1]
        
        corr_text = f'Correlation: {corr:.3f}\n'
        if abs(corr) < 0.3:
            corr_text += 'Weak relationship'
        elif abs(corr) < 0.7:
            corr_text += 'Moderate relationship'
        else:
            corr_text += 'Strong relationship'
        
        ax2.text(0.02, 0.02, corr_text, transform=ax2.transAxes,
                fontsize=11, fontweight='bold', verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    ax2.set_xlabel('Mission Completion Time (Hours)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Blue Force Survival Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Completion Time vs Friendly Survival\n(Higher = Fewer Friendly Casualties)', 
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    
    # Plot 3: Completion Time vs Battle Complexity
    ax3 = fig.add_subplot(gs[1, 0])
    
    scatter3 = ax3.scatter(completion_times, battle_complexity, s=150, c=colors, 
                          alpha=0.6, edgecolors='black', linewidth=1.5)
    
    # Add trend line
    if len(completion_times) > 1:
        z = np.polyfit(completion_times, battle_complexity, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(completion_times.min(), completion_times.max(), 100)
        ax3.plot(x_trend, p(x_trend), "b--", linewidth=2.5, alpha=0.7,
                label=f'Trend: {"More Complex ↑" if z[0] > 0 else "Less Complex ↓"}')
        
        # Calculate correlation
        corr = np.corrcoef(completion_times, battle_complexity)[0, 1]
        
        corr_text = f'Correlation: {corr:.3f}\n'
        if abs(corr) < 0.3:
            corr_text += 'Weak relationship'
        elif abs(corr) < 0.7:
            corr_text += 'Moderate relationship'
        else:
            corr_text += 'Strong relationship'
        
        ax3.text(0.02, 0.98, corr_text, transform=ax3.transAxes,
                fontsize=11, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    ax3.set_xlabel('Mission Completion Time (Hours)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Battle Complexity (Combat Events)', fontsize=11, fontweight='bold')
    ax3.set_title('Completion Time vs Battle Complexity\n(More Events = More Complex Battle)', 
                  fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary - Performance Grouped by Completion Time
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Bin by completion time (fast, medium, slow)
    time_bins = np.percentile(completion_times, [0, 33, 67, 100])
    bin_labels = ['Fast\n(Quick Wins)', 'Medium', 'Slow\n(Long Battles)']
    
    fast_mask = completion_times <= time_bins[1]
    medium_mask = (completion_times > time_bins[1]) & (completion_times <= time_bins[2])
    slow_mask = completion_times > time_bins[2]
    
    avg_kill_ratios = [
        np.mean(kill_ratios[fast_mask]),
        np.mean(kill_ratios[medium_mask]),
        np.mean(kill_ratios[slow_mask])
    ]
    
    avg_survival = [
        np.mean(blue_survival[fast_mask]),
        np.mean(blue_survival[medium_mask]),
        np.mean(blue_survival[slow_mask])
    ]
    
    x = np.arange(len(bin_labels))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, avg_kill_ratios, width, label='Avg Kill Ratio',
                   color='steelblue', alpha=0.7, edgecolor='darkblue', linewidth=1.5)
    
    # Add second y-axis for survival
    ax4_twin = ax4.twinx()
    bars2 = ax4_twin.bar(x + width/2, avg_survival, width, label='Avg Blue Survival %',
                        color='green', alpha=0.7, edgecolor='darkgreen', linewidth=1.5)
    
    ax4.set_xlabel('Mission Duration Category', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Average Kill Ratio', fontsize=11, fontweight='bold', color='steelblue')
    ax4_twin.set_ylabel('Average Blue Survival (%)', fontsize=11, fontweight='bold', color='green')
    ax4.set_title('Performance by Mission Duration\n(Do Longer Missions Perform Better?)', 
                  fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(bin_labels, fontsize=10)
    ax4.tick_params(axis='y', labelcolor='steelblue')
    ax4_twin.tick_params(axis='y', labelcolor='green')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    
    # Add insight text
    best_category = bin_labels[np.argmax(avg_kill_ratios)]
    insight_text = f'Best Kill Ratio: {best_category}'
    ax4.text(0.98, 0.02, insight_text, transform=ax4.transAxes,
            fontsize=10, fontweight='bold', verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.suptitle('Mission Completion Time vs Performance: Does Taking Longer = Better Results?', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_path = os.path.join(OUTPUT_DIR, 'completion_time_vs_performance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

if __name__ == '__main__':
    print("=" * 70)
    print("COMPLETION TIME VS PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    # Analyze all simulations
    print("\nAnalyzing simulations...")
    results = analyze_simulation_performance_and_time()
    print(f"Analyzed {len(results)} simulations")
    
    # Generate plot
    plot_completion_vs_performance(results)
    
    # Print correlation summary
    print("\n" + "=" * 70)
    print("CORRELATION SUMMARY")
    print("=" * 70)
    
    completion_times = np.array([r['completion_time_hours'] for r in results])
    kill_ratios = np.array([r['kill_ratio'] for r in results])
    blue_survival = np.array([r['blue_survival_rate'] for r in results])
    battle_complexity = np.array([r['battle_complexity'] for r in results])
    
    corr_kill = np.corrcoef(completion_times, kill_ratios)[0, 1]
    corr_survival = np.corrcoef(completion_times, blue_survival)[0, 1]
    corr_complexity = np.corrcoef(completion_times, battle_complexity)[0, 1]
    
    print(f"\nCompletion Time vs Kill Ratio: {corr_kill:.3f}")
    print(f"Completion Time vs Blue Survival: {corr_survival:.3f}")
    print(f"Completion Time vs Battle Complexity: {corr_complexity:.3f}")
    
    print("\nInterpretation:")
    if corr_kill > 0.3:
        print("  ✓ Longer missions tend to have BETTER kill ratios")
    elif corr_kill < -0.3:
        print("  ✗ Longer missions tend to have WORSE kill ratios")
    else:
        print("  ≈ Mission duration doesn't strongly affect kill ratio")
    
    if corr_survival > 0.3:
        print("  ✓ Longer missions have BETTER blue survival")
    elif corr_survival < -0.3:
        print("  ✗ Longer missions have WORSE blue survival")
    else:
        print("  ≈ Mission duration doesn't strongly affect survival")
    
    print("=" * 70)
