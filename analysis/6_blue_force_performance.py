"""
Blue Force Performance Analysis
================================

This script focuses on Blue Force (us) performance metrics showing:
1. Kill ratio (Red casualties / Blue casualties) - higher is better
2. Blue Force effectiveness over simulation runs
3. Identification of "winning" vs "losing" scenarios
4. Performance trends

Outputs:
    - kill_ratio_analysis.png: Kill ratio metrics showing Blue Force effectiveness
    - performance_trends.png: Are we getting better over time?
    - blue_force_summary.png: Overall Blue Force performance dashboard
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

def load_and_analyze_simulation(sim_id):
    """Load simulation data and calculate Blue Force metrics."""
    sim_path = os.path.join(DATA_DIR, sim_id)
    
    try:
        with open(os.path.join(sim_path, 'damage_events.json')) as f:
            damage_events = json.load(f)
        
        with open(os.path.join(sim_path, 'config.json')) as f:
            config = json.load(f)
        
        with open(os.path.join(sim_path, 'external_id_to_internal_id.json')) as f:
            id_map = json.load(f)
        
        with open(os.path.join(sim_path, 'results.json')) as f:
            results = json.load(f)
    except Exception as e:
        print(f"Error loading {sim_id}: {e}")
        return None
    
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
    blue_units = 0
    red_units = 0
    
    for entity in config.get('entities', []):
        if entity['type'] == 'Unit':
            ext_id = entity['data']['header']['id']
            int_id = id_map.get(ext_id)
            faction_id = entity['data'].get('faction')
            initial_cp = entity['data'].get('combat_power', {}).get('initial', 0)
            
            if faction_id == blue_faction_id:
                blue_initial += initial_cp
                blue_units += 1
                if int_id:
                    entity_factions[int_id] = 'Blue'
            elif faction_id == red_faction_id:
                red_initial += initial_cp
                red_units += 1
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
    
    # Calculate metrics
    kill_ratio = red_casualties / blue_casualties if blue_casualties > 0 else float('inf')
    blue_survival_rate = ((blue_initial - blue_casualties) / blue_initial * 100) if blue_initial > 0 else 0
    red_survival_rate = ((red_initial - red_casualties) / red_initial * 100) if red_initial > 0 else 0
    
    # Blue Force "won" if they have better final force ratio
    blue_final = blue_initial - blue_casualties
    red_final = red_initial - red_casualties
    force_ratio_final = blue_final / red_final if red_final > 0 else float('inf')
    won = force_ratio_final > 1.0
    
    return {
        'sim_id': sim_id,
        'timestamp': results.get('end_time', 0),
        'blue_initial': blue_initial,
        'red_initial': red_initial,
        'blue_units': blue_units,
        'red_units': red_units,
        'blue_casualties': blue_casualties,
        'red_casualties': red_casualties,
        'blue_final': blue_final,
        'red_final': red_final,
        'kill_ratio': kill_ratio,
        'blue_survival_rate': blue_survival_rate,
        'red_survival_rate': red_survival_rate,
        'force_ratio_final': force_ratio_final,
        'combat_duration_hours': last_combat_time / 3600,
        'won': won,
        'victory_margin': force_ratio_final - 1.0
    }

def analyze_all_simulations(num_sims=30):
    """Analyze all simulations for Blue Force performance."""
    sim_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    sim_dirs = sim_dirs[:num_sims]
    
    results = []
    for sim_id in sim_dirs:
        result = load_and_analyze_simulation(sim_id)
        if result:
            results.append(result)
    
    # Sort by timestamp to see performance over time
    results.sort(key=lambda x: x['timestamp'])
    
    return results

def plot_kill_ratio_analysis(results):
    """Create kill ratio focused plots showing Blue Force effectiveness."""
    print(f"Creating kill ratio analysis for {len(results)} simulations...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Extract data - filter out invalid entries consistently
    valid_results = [r for r in results if r['kill_ratio'] != float('inf')]
    kill_ratios = [r['kill_ratio'] for r in valid_results]
    blue_cas = [r['blue_casualties'] for r in valid_results]
    red_cas = [r['red_casualties'] for r in valid_results]
    won = [r['won'] for r in valid_results]
    
    # Plot 1: Kill Ratio over simulation runs
    ax1 = fig.add_subplot(gs[0, :])
    
    x = np.arange(len(valid_results))
    colors = ['green' if r['won'] else 'red' for r in valid_results]
    
    bars = ax1.bar(x, kill_ratios, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Equal Casualties (1:1)')
    
    # Add average line
    avg_kill_ratio = np.mean(kill_ratios)
    ax1.axhline(y=avg_kill_ratio, color='blue', linestyle='-', linewidth=2.5, 
                label=f'Average Kill Ratio: {avg_kill_ratio:.2f}:1')
    
    ax1.set_xlabel('Simulation Run Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Kill Ratio (Red/Blue)', fontsize=12, fontweight='bold')
    ax1.set_title('Blue Force Kill Ratio Across Simulations\n(Higher is Better - Green = Blue Won)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add text box with statistics
    wins = sum(won)
    win_rate = wins / len(valid_results) * 100
    ax1.text(0.02, 0.98, 
             f'Win Rate: {win_rate:.0f}% ({wins}/{len(valid_results)})\n'
             f'Avg Kill Ratio: {avg_kill_ratio:.2f}:1\n'
             f'Best: {max(kill_ratios):.2f}:1\n'
             f'Worst: {min(kill_ratios):.2f}:1',
             transform=ax1.transAxes, fontsize=11, fontweight='bold',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Plot 2: Kill Ratio Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    
    ax2.hist(kill_ratios, bins=15, color='blue', alpha=0.7, edgecolor='darkblue', linewidth=1.5)
    ax2.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Equal (1:1)')
    ax2.axvline(avg_kill_ratio, color='green', linestyle='-', linewidth=2.5, 
                label=f'Average: {avg_kill_ratio:.2f}:1')
    
    ax2.set_xlabel('Kill Ratio (Red/Blue)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Simulations', fontsize=11, fontweight='bold')
    ax2.set_title('Kill Ratio Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Blue vs Red Casualties with kill ratio indicated
    ax3 = fig.add_subplot(gs[1, 1])
    
    scatter = ax3.scatter(blue_cas, red_cas, c=kill_ratios, s=150, 
                         cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidth=1)
    
    # Add equal casualties line
    max_cas = max(max(blue_cas), max(red_cas))
    ax3.plot([0, max_cas], [0, max_cas], 'k--', linewidth=2, alpha=0.5, label='Equal Casualties')
    
    # Add 2:1 and 3:1 lines
    ax3.plot([0, max_cas], [0, max_cas*2], 'g--', linewidth=1.5, alpha=0.5, label='2:1 Kill Ratio')
    ax3.plot([0, max_cas], [0, max_cas*3], 'g:', linewidth=1.5, alpha=0.5, label='3:1 Kill Ratio')
    
    ax3.set_xlabel('Blue Force Casualties', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Red Force Casualties', fontsize=11, fontweight='bold')
    ax3.set_title('Casualty Correlation\n(Above diagonal = Blue winning)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Kill Ratio', fontsize=10, fontweight='bold')
    
    # Plot 4: Win/Loss Summary
    ax4 = fig.add_subplot(gs[2, 0])
    
    win_counts = [sum(won), len(valid_results) - sum(won)]
    colors_pie = ['green', 'red']
    explode = (0.1, 0)
    
    wedges, texts, autotexts = ax4.pie(win_counts, explode=explode, labels=['Blue Won', 'Red Won'],
                                        colors=colors_pie, autopct='%1.1f%%',
                                        shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    
    ax4.set_title(f'Blue Force Win Rate\n({wins} wins out of {len(valid_results)} simulations)', 
                  fontsize=12, fontweight='bold')
    
    # Plot 5: Final Force Ratio
    ax5 = fig.add_subplot(gs[2, 1])
    
    final_ratios = [r['force_ratio_final'] for r in valid_results if r['force_ratio_final'] != float('inf')]
    
    ax5.hist(final_ratios, bins=15, color='purple', alpha=0.7, edgecolor='indigo', linewidth=1.5)
    ax5.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Equal Strength')
    ax5.axvline(np.median(final_ratios), color='green', linestyle='-', linewidth=2.5,
                label=f'Median: {np.median(final_ratios):.2f}')
    
    ax5.set_xlabel('Final Force Ratio (Blue/Red)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Number of Simulations', fontsize=11, fontweight='bold')
    ax5.set_title('Final Force Balance\n(>1.0 = Blue has more remaining)', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Blue Force Performance Analysis - Kill Ratios', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    output_path = os.path.join(OUTPUT_DIR, 'kill_ratio_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_performance_trends(results):
    """Show if Blue Force performance is improving over time."""
    print(f"Creating performance trends analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x = np.arange(len(results))
    
    # Plot 1: Kill Ratio Trend
    ax1 = axes[0, 0]
    kill_ratios = [r['kill_ratio'] for r in results if r['kill_ratio'] != float('inf')]
    
    ax1.plot(x[:len(kill_ratios)], kill_ratios, 'bo-', linewidth=2, markersize=8, alpha=0.6)
    
    # Add trend line
    if len(kill_ratios) > 1:
        z = np.polyfit(x[:len(kill_ratios)], kill_ratios, 1)
        p = np.poly1d(z)
        ax1.plot(x[:len(kill_ratios)], p(x[:len(kill_ratios)]), "r--", linewidth=3, 
                label=f'Trend: {"Improving" if z[0] > 0 else "Declining"}')
        
        # Add trend info
        trend_text = f'Slope: {z[0]:.4f}\n{"✓ Improving!" if z[0] > 0 else "⚠ Declining"}'
        ax1.text(0.98, 0.02, trend_text, transform=ax1.transAxes,
                fontsize=11, fontweight='bold',
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgreen' if z[0] > 0 else 'lightcoral', alpha=0.8))
    
    ax1.axhline(1.0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Simulation Run (Chronological)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Kill Ratio (Red/Blue)', fontsize=11, fontweight='bold')
    ax1.set_title('Kill Ratio Trend Over Time', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Blue Survival Rate Trend
    ax2 = axes[0, 1]
    blue_survival = [r['blue_survival_rate'] for r in results]
    
    ax2.plot(x, blue_survival, 'go-', linewidth=2, markersize=8, alpha=0.6)
    
    # Add trend line
    if len(blue_survival) > 1:
        z = np.polyfit(x, blue_survival, 1)
        p = np.poly1d(z)
        ax2.plot(x, p(x), "r--", linewidth=3, 
                label=f'Trend: {"Improving" if z[0] > 0 else "Declining"}')
    
    ax2.set_xlabel('Simulation Run (Chronological)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Blue Survival Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Blue Force Survival Trend', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    
    # Plot 3: Victory Margin Trend
    ax3 = axes[1, 0]
    victory_margins = [r['victory_margin'] for r in results if r['force_ratio_final'] != float('inf')]
    
    colors = ['green' if m > 0 else 'red' for m in victory_margins]
    ax3.bar(x[:len(victory_margins)], victory_margins, color=colors, alpha=0.7, edgecolor='black')
    ax3.axhline(0, color='black', linestyle='-', linewidth=2)
    
    # Add trend line
    if len(victory_margins) > 1:
        z = np.polyfit(x[:len(victory_margins)], victory_margins, 1)
        p = np.poly1d(z)
        ax3.plot(x[:len(victory_margins)], p(x[:len(victory_margins)]), "b--", linewidth=3, 
                label=f'Trend: {"Improving" if z[0] > 0 else "Declining"}')
    
    ax3.set_xlabel('Simulation Run (Chronological)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Victory Margin (Final Force Ratio - 1)', fontsize=11, fontweight='bold')
    ax3.set_title('Victory Margin Trend\n(Positive = Blue Won)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Combat Duration Trend
    ax4 = axes[1, 1]
    durations = [r['combat_duration_hours'] for r in results]
    
    ax4.plot(x, durations, 'mo-', linewidth=2, markersize=8, alpha=0.6)
    
    # Add trend line
    if len(durations) > 1:
        z = np.polyfit(x, durations, 1)
        p = np.poly1d(z)
        ax4.plot(x, p(x), "r--", linewidth=3, label='Trend')
    
    ax4.set_xlabel('Simulation Run (Chronological)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Combat Duration (hours)', fontsize=11, fontweight='bold')
    ax4.set_title('Combat Duration Trend\n(Shorter may indicate decisive victory)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Blue Force Performance Trends - Are We Improving?', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'performance_trends.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def create_summary_dashboard(results):
    """Create overall Blue Force performance summary."""
    print(f"Creating Blue Force summary dashboard...")
    
    # Calculate statistics
    kill_ratios = [r['kill_ratio'] for r in results if r['kill_ratio'] != float('inf')]
    wins = sum(r['won'] for r in results)
    win_rate = wins / len(results) * 100
    avg_kill_ratio = np.mean(kill_ratios)
    best_kill_ratio = max(kill_ratios)
    worst_kill_ratio = min(kill_ratios)
    avg_blue_survival = np.mean([r['blue_survival_rate'] for r in results])
    avg_red_survival = np.mean([r['red_survival_rate'] for r in results])
    
    # Create dashboard
    fig = plt.figure(figsize=(14, 8))
    
    # Large title
    fig.text(0.5, 0.95, 'BLUE FORCE PERFORMANCE SUMMARY', 
             ha='center', fontsize=20, fontweight='bold')
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4, top=0.88, bottom=0.05)
    
    # Key metrics as large text
    metrics = [
        ('Win Rate', f'{win_rate:.0f}%', 'green' if win_rate > 50 else 'red'),
        ('Avg Kill Ratio', f'{avg_kill_ratio:.2f}:1', 'green' if avg_kill_ratio > 1 else 'red'),
        ('Best Kill Ratio', f'{best_kill_ratio:.2f}:1', 'darkgreen'),
        ('Blue Survival', f'{avg_blue_survival:.0f}%', 'blue'),
        ('Red Survival', f'{avg_red_survival:.0f}%', 'red'),
        ('Simulations', f'{len(results)}', 'black'),
    ]
    
    for i, (label, value, color) in enumerate(metrics):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        ax.axis('off')
        
        # Draw box
        ax.add_patch(plt.Rectangle((0.1, 0.1), 0.8, 0.8, 
                                   facecolor=color, alpha=0.2, edgecolor=color, linewidth=3))
        
        # Add text
        ax.text(0.5, 0.65, value, ha='center', va='center', 
               fontsize=28, fontweight='bold', color=color)
        ax.text(0.5, 0.3, label, ha='center', va='center', 
               fontsize=14, fontweight='bold', color='black')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    # Add interpretation text at bottom
    interpretation = (
        f"INTERPRETATION:\n"
        f"• Blue Force wins {win_rate:.0f}% of simulations\n"
        f"• Average kill ratio of {avg_kill_ratio:.2f}:1 means we eliminate {avg_kill_ratio:.1f} red units for every blue loss\n"
        f"• Blue Force retains {avg_blue_survival:.0f}% combat power on average\n"
        f"• {'✓ STRONG PERFORMANCE' if win_rate > 50 and avg_kill_ratio > 1.5 else '⚠ ROOM FOR IMPROVEMENT'}"
    )
    
    fig.text(0.5, 0.02, interpretation, ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=1))
    
    output_path = os.path.join(OUTPUT_DIR, 'blue_force_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

if __name__ == '__main__':
    print("=" * 70)
    print("BLUE FORCE PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    # Analyze all available simulations
    print("\nAnalyzing all simulations...")
    results = analyze_all_simulations(num_sims=30)
    
    print(f"Loaded {len(results)} simulations")
    
    # Generate plots
    plot_kill_ratio_analysis(results)
    plot_performance_trends(results)
    create_summary_dashboard(results)
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("BLUE FORCE SUMMARY STATISTICS")
    print("=" * 70)
    
    kill_ratios = [r['kill_ratio'] for r in results if r['kill_ratio'] != float('inf')]
    wins = sum(r['won'] for r in results)
    
    print(f"Total Simulations: {len(results)}")
    print(f"Blue Wins: {wins} ({wins/len(results)*100:.1f}%)")
    print(f"Red Wins: {len(results)-wins} ({(len(results)-wins)/len(results)*100:.1f}%)")
    print(f"\nKill Ratio Statistics:")
    print(f"  Average: {np.mean(kill_ratios):.2f}:1")
    print(f"  Median: {np.median(kill_ratios):.2f}:1")
    print(f"  Best: {max(kill_ratios):.2f}:1")
    print(f"  Worst: {min(kill_ratios):.2f}:1")
    print(f"\nSurvival Rates:")
    print(f"  Blue Force: {np.mean([r['blue_survival_rate'] for r in results]):.1f}%")
    print(f"  Red Force: {np.mean([r['red_survival_rate'] for r in results]):.1f}%")
    print("=" * 70)
