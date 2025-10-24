"""
COA Complexity and Efficiency Analysis
=======================================

This script analyzes Course of Action (COA) characteristics:
1. Number of tasks/steps in each COA
2. Number of units/assets assigned tasks
3. COA complexity vs. performance correlation
4. Development time vs. complexity

Outputs:
    - coa_complexity.png: COA complexity metrics
    - coa_efficiency.png: Complexity vs. performance correlation
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Configuration
DATA_DIR = '../data/sim'
WEB_DIR = '../data/web'
OUTPUT_DIR = './outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_coa_from_config(sim_id):
    """Extract COA characteristics from simulation config."""
    sim_path = os.path.join(DATA_DIR, sim_id)
    
    try:
        with open(os.path.join(sim_path, 'config.json')) as f:
            config = json.load(f)
        
        with open(os.path.join(sim_path, 'damage_events.json')) as f:
            damage_events = json.load(f)
        
        with open(os.path.join(sim_path, 'external_id_to_internal_id.json')) as f:
            id_map = json.load(f)
    except Exception as e:
        print(f"Error loading {sim_id}: {e}")
        return None
    
    # Get Blue Force faction ID
    blue_faction_id = None
    for faction in config.get('factions', []):
        if 'Blue' in faction['header']['name']:
            blue_faction_id = faction['header']['id']
            break
    
    # Analyze Blue Force units and their tasks
    blue_units = []
    total_tasks = 0
    units_with_tasks = 0
    task_types = defaultdict(int)
    
    for entity in config.get('entities', []):
        if entity['type'] == 'Unit':
            faction_id = entity['data'].get('faction')
            
            if faction_id == blue_faction_id:
                unit_data = {
                    'id': entity['data']['header']['id'],
                    'name': entity['data']['header']['name'],
                    'tasks': entity['data'].get('tasks', []),
                    'num_tasks': len(entity['data'].get('tasks', [])),
                    'combat_power': entity['data'].get('combat_power', {}).get('initial', 0)
                }
                
                blue_units.append(unit_data)
                
                if unit_data['num_tasks'] > 0:
                    units_with_tasks += 1
                    total_tasks += unit_data['num_tasks']
                    
                    # Count task types
                    for task in unit_data['tasks']:
                        task_type = task.get('type', 'Unknown')
                        task_types[task_type] += 1
    
    # Calculate casualties for outcome
    blue_casualties = 0
    red_casualties = 0
    entity_factions = {}
    
    # Map factions
    red_faction_id = None
    for faction in config.get('factions', []):
        if 'Red' in faction['header']['name']:
            red_faction_id = faction['header']['id']
            break
    
    for entity in config.get('entities', []):
        if entity['type'] == 'Unit':
            ext_id = entity['data']['header']['id']
            int_id = id_map.get(ext_id)
            faction_id = entity['data'].get('faction')
            
            if int_id:
                if faction_id == blue_faction_id:
                    entity_factions[int_id] = 'Blue'
                elif faction_id == red_faction_id:
                    entity_factions[int_id] = 'Red'
    
    for event in damage_events:
        target_id = event['target']
        damage = event['damage']
        
        if target_id in entity_factions:
            if entity_factions[target_id] == 'Blue':
                blue_casualties += damage
            elif entity_factions[target_id] == 'Red':
                red_casualties += damage
    
    kill_ratio = red_casualties / blue_casualties if blue_casualties > 0 else 0
    
    return {
        'sim_id': sim_id,
        'sim_name': config.get('name', 'Unknown'),
        'total_blue_units': len(blue_units),
        'units_with_tasks': units_with_tasks,
        'total_tasks': total_tasks,
        'avg_tasks_per_unit': total_tasks / len(blue_units) if blue_units else 0,
        'task_utilization': units_with_tasks / len(blue_units) * 100 if blue_units else 0,
        'task_types': dict(task_types),
        'num_task_types': len(task_types),
        'kill_ratio': kill_ratio,
        'blue_casualties': blue_casualties,
        'red_casualties': red_casualties,
    }

def analyze_web_coa_data():
    """Extract COA data from web interaction logs."""
    coa_data = {}
    
    web_sessions = ['TS-ARL3538', 'TS-ARL3542', 'TS-ARL3543']
    
    for session in web_sessions:
        web_file = os.path.join(WEB_DIR, session, f'{session}.jsonl')
        
        if not os.path.exists(web_file):
            continue
        
        print(f"Processing web session: {session}")
        
        coa_times = defaultdict(list)
        coa_actions = defaultdict(lambda: defaultdict(int))
        
        with open(web_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    coa_id = data.get('coa_id')
                    timestamp = data.get('timeStamp')
                    action_type = data.get('actionType', 'unknown')
                    
                    if coa_id and timestamp:
                        coa_times[coa_id].append(timestamp)
                        coa_actions[coa_id][action_type] += 1
                except:
                    continue
        
        # Calculate metrics for each COA
        for coa_id, timestamps in coa_times.items():
            if timestamps:
                start_time = min(timestamps)
                end_time = max(timestamps)
                duration_minutes = (end_time - start_time) / 1000 / 60
                
                # Count meaningful actions (not just mouse moves)
                meaningful_actions = sum(count for action, count in coa_actions[coa_id].items() 
                                        if action not in ['pointermove', 'pointerover', 'pointerout'])
                
                coa_data[coa_id] = {
                    'session': session,
                    'development_time_minutes': duration_minutes,
                    'total_interactions': len(timestamps),
                    'meaningful_actions': meaningful_actions,
                    'actions_per_minute': meaningful_actions / duration_minutes if duration_minutes > 0 else 0,
                }
    
    return coa_data

def plot_coa_complexity(sim_results, coa_data):
    """Plot COA complexity analysis."""
    print(f"Creating COA complexity analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Task Distribution
    ax1 = axes[0, 0]
    
    total_tasks = [r['total_tasks'] for r in sim_results]
    units_with_tasks = [r['units_with_tasks'] for r in sim_results]
    
    x = np.arange(len(sim_results))
    width = 0.35
    
    ax1.bar(x - width/2, total_tasks, width, label='Total Tasks', color='blue', alpha=0.7)
    ax1.bar(x + width/2, units_with_tasks, width, label='Units with Tasks', color='green', alpha=0.7)
    
    ax1.set_xlabel('Simulation Run', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax1.set_title('COA Complexity: Tasks per Simulation', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add average lines
    ax1.axhline(np.mean(total_tasks), color='blue', linestyle='--', alpha=0.5,
                label=f'Avg Tasks: {np.mean(total_tasks):.1f}')
    ax1.axhline(np.mean(units_with_tasks), color='green', linestyle='--', alpha=0.5,
                label=f'Avg Units: {np.mean(units_with_tasks):.1f}')
    
    # Plot 2: Task Utilization
    ax2 = axes[0, 1]
    
    utilization = [r['task_utilization'] for r in sim_results]
    
    ax2.bar(x, utilization, color='orange', alpha=0.7, edgecolor='darkorange', linewidth=1.5)
    ax2.axhline(np.mean(utilization), color='red', linestyle='--', linewidth=2,
                label=f'Average: {np.mean(utilization):.1f}%')
    
    ax2.set_xlabel('Simulation Run', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Task Utilization (%)', fontsize=11, fontweight='bold')
    ax2.set_title('% of Blue Units with Assigned Tasks', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 105])
    
    # Plot 3: Tasks vs Performance
    ax3 = axes[1, 0]
    
    kill_ratios = [r['kill_ratio'] for r in sim_results]
    
    scatter = ax3.scatter(total_tasks, kill_ratios, s=150, alpha=0.6, 
                         c=utilization, cmap='viridis', edgecolors='black', linewidth=1.5)
    
    # Add trend line
    if len(total_tasks) > 1:
        z = np.polyfit(total_tasks, kill_ratios, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(total_tasks), max(total_tasks), 100)
        ax3.plot(x_line, p(x_line), "r--", linewidth=2.5, 
                label=f'Trend: {"Positive" if z[0] > 0 else "Negative"}')
    
    ax3.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='Equal Casualties')
    
    ax3.set_xlabel('Total Tasks in COA', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Kill Ratio (Red/Blue)', fontsize=11, fontweight='bold')
    ax3.set_title('COA Complexity vs Performance\n(Does more planning = better results?)', 
                  fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Task Utilization %', fontsize=10)
    
    # Plot 4: Development Time (if available)
    ax4 = axes[1, 1]
    
    if coa_data:
        dev_times = [coa['development_time_minutes'] for coa in coa_data.values()]
        meaningful_actions = [coa['meaningful_actions'] for coa in coa_data.values()]
        
        ax4.scatter(dev_times, meaningful_actions, s=150, alpha=0.7, 
                   c=range(len(dev_times)), cmap='coolwarm', edgecolors='black', linewidth=1.5)
        
        ax4.set_xlabel('Development Time (minutes)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Meaningful Actions', fontsize=11, fontweight='bold')
        ax4.set_title('COA Development Effort\n(Time vs Actions)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add correlation info
        if len(dev_times) > 1:
            corr = np.corrcoef(dev_times, meaningful_actions)[0, 1]
            ax4.text(0.05, 0.95, f'Correlation: {corr:.2f}',
                    transform=ax4.transAxes, fontsize=10, fontweight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    else:
        ax4.text(0.5, 0.5, 'No Web Data Available', ha='center', va='center',
                fontsize=14, fontweight='bold')
        ax4.axis('off')
    
    plt.suptitle('COA Complexity Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'coa_complexity.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_coa_efficiency(sim_results, coa_data):
    """Plot COA efficiency - complexity vs outcomes."""
    print(f"Creating COA efficiency analysis...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.4)
    
    # Extract metrics
    total_tasks = np.array([r['total_tasks'] for r in sim_results])
    utilization = np.array([r['task_utilization'] for r in sim_results])
    kill_ratios = np.array([r['kill_ratio'] for r in sim_results])
    avg_tasks_per_unit = np.array([r['avg_tasks_per_unit'] for r in sim_results])
    
    # Plot 1: Task Efficiency
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Define efficiency as kill_ratio / avg_tasks_per_unit (more kills with fewer tasks = efficient)
    efficiency = kill_ratios / (avg_tasks_per_unit + 0.1)  # Add small constant to avoid div by zero
    
    x = np.arange(len(sim_results))
    colors = ['green' if e > np.median(efficiency) else 'orange' for e in efficiency]
    
    ax1.bar(x, efficiency, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.axhline(np.median(efficiency), color='red', linestyle='--', linewidth=2,
                label=f'Median Efficiency: {np.median(efficiency):.2f}')
    
    ax1.set_xlabel('Simulation Run', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Efficiency Score\n(Kill Ratio / Avg Tasks per Unit)', fontsize=11, fontweight='bold')
    ax1.set_title('COA Efficiency: Performance per Planning Effort\n(Higher = Better)', 
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Summary Statistics
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    stats_text = (
        f"COA STATISTICS\n"
        f"{'='*30}\n\n"
        f"Avg Tasks per COA: {np.mean(total_tasks):.1f}\n"
        f"Avg Task Utilization: {np.mean(utilization):.1f}%\n"
        f"Avg Tasks per Unit: {np.mean(avg_tasks_per_unit):.2f}\n\n"
        f"Performance:\n"
        f"Avg Kill Ratio: {np.mean(kill_ratios):.2f}:1\n"
        f"Avg Efficiency: {np.mean(efficiency):.2f}\n\n"
        f"Insight:\n"
        f"{'More tasks ≠ better' if np.corrcoef(total_tasks, kill_ratios)[0,1] < 0.3 else 'More tasks = better'}\n"
        f"performance"
    )
    
    ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=1))
    
    # Plot 3: Complexity vs Kill Ratio (detailed)
    ax3 = fig.add_subplot(gs[1, 0])
    
    ax3.scatter(total_tasks, kill_ratios, s=100, alpha=0.6, edgecolors='black', linewidth=1)
    
    # Quadrant lines
    ax3.axvline(np.median(total_tasks), color='gray', linestyle='--', alpha=0.5)
    ax3.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    
    # Label quadrants
    ax3.text(0.25, 0.95, 'Simple Plan\nGood Result', transform=ax3.transAxes,
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax3.text(0.75, 0.95, 'Complex Plan\nGood Result', transform=ax3.transAxes,
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    ax3.set_xlabel('Total Tasks', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Kill Ratio', fontsize=10, fontweight='bold')
    ax3.set_title('Complexity vs Outcome', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Utilization vs Kill Ratio
    ax4 = fig.add_subplot(gs[1, 1])
    
    ax4.scatter(utilization, kill_ratios, s=100, alpha=0.6, edgecolors='black', linewidth=1)
    
    ax4.axvline(np.median(utilization), color='gray', linestyle='--', alpha=0.5)
    ax4.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    
    ax4.set_xlabel('Task Utilization (%)', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Kill Ratio', fontsize=10, fontweight='bold')
    ax4.set_title('Unit Utilization vs Outcome', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Efficiency Distribution
    ax5 = fig.add_subplot(gs[1, 2])
    
    ax5.hist(efficiency, bins=10, color='purple', alpha=0.7, edgecolor='indigo', linewidth=1.5)
    ax5.axvline(np.median(efficiency), color='red', linestyle='--', linewidth=2,
                label=f'Median: {np.median(efficiency):.2f}')
    
    ax5.set_xlabel('Efficiency Score', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Count', fontsize=10, fontweight='bold')
    ax5.set_title('Efficiency Distribution', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('COA Efficiency Analysis - Planning vs Performance', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_path = os.path.join(OUTPUT_DIR, 'coa_efficiency.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

if __name__ == '__main__':
    print("=" * 70)
    print("COA COMPLEXITY & EFFICIENCY ANALYSIS")
    print("=" * 70)
    
    # Analyze simulations
    print("\nAnalyzing simulation COAs...")
    sim_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    sim_results = []
    for i, sim_id in enumerate(sim_dirs[:30]):
        result = analyze_coa_from_config(sim_id)
        if result:
            sim_results.append(result)
    
    print(f"Analyzed {len(sim_results)} simulations")
    
    # Analyze web data
    print("\nAnalyzing web interaction data...")
    coa_data = analyze_web_coa_data()
    print(f"Found {len(coa_data)} COAs in web data")
    
    # Generate plots
    plot_coa_complexity(sim_results, coa_data)
    plot_coa_efficiency(sim_results, coa_data)
    
    # Print summary
    print("\n" + "=" * 70)
    print("COA COMPLEXITY SUMMARY")
    print("=" * 70)
    print(f"Average tasks per COA: {np.mean([r['total_tasks'] for r in sim_results]):.1f}")
    print(f"Average units with tasks: {np.mean([r['units_with_tasks'] for r in sim_results]):.1f}")
    print(f"Average task utilization: {np.mean([r['task_utilization'] for r in sim_results]):.1f}%")
    print(f"Average tasks per unit: {np.mean([r['avg_tasks_per_unit'] for r in sim_results]):.2f}")
    
    if coa_data:
        print(f"\nWeb Data:")
        print(f"Average development time: {np.mean([c['development_time_minutes'] for c in coa_data.values()]):.1f} min")
        print(f"Average meaningful actions: {np.mean([c['meaningful_actions'] for c in coa_data.values()]):.0f}")
    
    print("=" * 70)
