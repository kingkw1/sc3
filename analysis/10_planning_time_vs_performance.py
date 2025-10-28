"""
Planning Time vs Performance Analysis
======================================

This script analyzes whether more planning time (human refinement) leads to
better COA performance:
1. Extract COA development times from web logs
2. Link COAs to their simulation runs
3. Analyze performance (kill ratio, survival, etc.) by planning time
4. Answer: Does spending more time planning improve results?

Outputs:
    - planning_time_vs_performance.png: Does human refinement help?
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Configuration
DATA_DIR = '../data/sim'
WEB_DIR = '../data/web'
OUTPUT_DIR = './outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_coa_development_times():
    """Extract COA development times from web logs."""
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

def analyze_coa_performance(coa_id):
    """Analyze performance for all simulations of a specific COA."""
    sim_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    sim_results = []
    
    for sim_id in sim_dirs:
        sim_path = os.path.join(DATA_DIR, sim_id)
        
        try:
            with open(os.path.join(sim_path, 'config.json')) as f:
                config = json.load(f)
            
            # Check if this simulation uses this COA
            tags = config.get('tags', {})
            if tags.get('coa_id') != coa_id:
                continue
            
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
            
            # Map entities and calculate initial strengths
            entity_factions = {}
            blue_initial = 0
            red_initial = 0
            total_tasks = 0
            
            for entity in config.get('entities', []):
                if entity['type'] == 'Unit':
                    ext_id = entity['data']['header']['id']
                    int_id = id_map.get(ext_id)
                    faction_id = entity['data'].get('faction')
                    initial_cp = entity['data'].get('combat_power', {}).get('initial', 0)
                    
                    if faction_id == blue_faction_id:
                        blue_initial += initial_cp
                        total_tasks += len(entity['data'].get('tasks', []))
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
            
            # Calculate metrics
            kill_ratio = red_casualties / blue_casualties if blue_casualties > 0 else 10.0
            blue_survival = ((blue_initial - blue_casualties) / blue_initial * 100) if blue_initial > 0 else 0
            completion_time = last_combat_time / 3600
            
            sim_results.append({
                'sim_id': sim_id,
                'kill_ratio': kill_ratio,
                'blue_survival': blue_survival,
                'completion_time': completion_time,
                'blue_casualties': blue_casualties,
                'red_casualties': red_casualties,
                'total_tasks': total_tasks
            })
            
        except Exception as e:
            continue
    
    # Return average performance for this COA
    if sim_results:
        return {
            'num_sims': len(sim_results),
            'avg_kill_ratio': np.mean([r['kill_ratio'] for r in sim_results]),
            'avg_blue_survival': np.mean([r['blue_survival'] for r in sim_results]),
            'avg_completion_time': np.mean([r['completion_time'] for r in sim_results]),
            'avg_blue_casualties': np.mean([r['blue_casualties'] for r in sim_results]),
            'avg_red_casualties': np.mean([r['red_casualties'] for r in sim_results]),
            'avg_total_tasks': np.mean([r['total_tasks'] for r in sim_results])
        }
    
    return None

def plot_planning_vs_performance(coa_dev_times, coa_performance):
    """Create plots showing if more planning time improves performance."""
    print(f"Creating planning time vs performance analysis...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.40)
    
    # Extract data
    planning_times = []
    kill_ratios = []
    blue_survivals = []
    completion_times = []
    num_sims = []
    coa_names = []
    
    for i, (coa_id, dev_data) in enumerate(sorted(coa_dev_times.items(), key=lambda x: x[1]['development_time_minutes'])):
        if coa_id in coa_performance:
            perf = coa_performance[coa_id]
            planning_times.append(dev_data['development_time_minutes'])
            kill_ratios.append(perf['avg_kill_ratio'])
            blue_survivals.append(perf['avg_blue_survival'])
            completion_times.append(perf['avg_completion_time'])
            num_sims.append(perf['num_sims'])
            coa_names.append(f"COA {i+1}")
    
    planning_times = np.array(planning_times)
    kill_ratios = np.array(kill_ratios)
    blue_survivals = np.array(blue_survivals)
    completion_times = np.array(completion_times)
    num_sims = np.array(num_sims)
    
    print(f"Analyzing {len(planning_times)} COAs with simulation data")
    
    # Plot 1: Planning Time vs Kill Ratio
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Size by number of sims
    sizes = num_sims * 100
    scatter1 = ax1.scatter(planning_times, kill_ratios, s=sizes, alpha=0.6, 
                          c=range(len(planning_times)), cmap='viridis',
                          edgecolors='black', linewidth=2)
    
    # Add labels
    for i, txt in enumerate(coa_names):
        ax1.annotate(f'{txt}\n({num_sims[i]} sims)', 
                    (planning_times[i], kill_ratios[i]),
                    fontsize=10, ha='center', fontweight='bold')
    
    # Add trend line
    if len(planning_times) > 2:
        z = np.polyfit(planning_times, kill_ratios, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(planning_times.min(), planning_times.max(), 100)
        ax1.plot(x_trend, p(x_trend), "r--", linewidth=3, alpha=0.7,
                label=f'Trend: {"Improving ✓" if z[0] > 0 else "Declining ✗"}')
        
        # Calculate correlation
        corr = np.corrcoef(planning_times, kill_ratios)[0, 1]
        
        interpretation = "STRONG" if abs(corr) > 0.7 else "MODERATE" if abs(corr) > 0.3 else "WEAK"
        direction = "POSITIVE" if corr > 0 else "NEGATIVE"
        
        corr_text = f'Correlation: {corr:.3f}\n{interpretation} {direction}'
        color = 'lightgreen' if corr > 0.3 else 'lightcoral' if corr < -0.3 else 'lightyellow'
        
        ax1.text(0.02, 0.98, corr_text, transform=ax1.transAxes,
                fontsize=13, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    
    ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('COA Planning Time (Minutes)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Average Kill Ratio (Red/Blue)', fontsize=14, fontweight='bold')
    ax1.set_title('Planning Time vs Kill Ratio\n(Does More Planning = Better Results?)', 
                  fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=11)
    
    # Plot 2: Planning Time vs Blue Survival
    ax2 = fig.add_subplot(gs[0, 1])
    
    scatter2 = ax2.scatter(planning_times, blue_survivals, s=sizes, alpha=0.6,
                          c=range(len(planning_times)), cmap='viridis',
                          edgecolors='black', linewidth=2)
    
    for i, txt in enumerate(coa_names):
        ax2.annotate(f'{txt}', (planning_times[i], blue_survivals[i]),
                    fontsize=10, ha='center', fontweight='bold')
    
    if len(planning_times) > 2:
        z = np.polyfit(planning_times, blue_survivals, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(planning_times.min(), planning_times.max(), 100)
        ax2.plot(x_trend, p(x_trend), "r--", linewidth=3, alpha=0.7,
                label=f'Trend: {"Better ✓" if z[0] > 0 else "Worse ✗"}')
        
        corr = np.corrcoef(planning_times, blue_survivals)[0, 1]
        interpretation = "STRONG" if abs(corr) > 0.7 else "MODERATE" if abs(corr) > 0.3 else "WEAK"
        direction = "POSITIVE" if corr > 0 else "NEGATIVE"
        
        corr_text = f'Correlation: {corr:.3f}\n{interpretation} {direction}'
        color = 'lightgreen' if corr > 0.3 else 'lightcoral' if corr < -0.3 else 'lightyellow'
        
        ax2.text(0.02, 0.02, corr_text, transform=ax2.transAxes,
                fontsize=13, fontweight='bold', verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    
    ax2.set_xlabel('COA Planning Time (Minutes)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Blue Survival (%)', fontsize=14, fontweight='bold')
    ax2.set_title('Planning Time vs Friendly Survival\n(Does Human Refinement Reduce Casualties?)', 
                  fontsize=15, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    ax2.tick_params(axis='both', which='major', labelsize=11)
    
    # Plot 3: Performance Comparison
    ax3 = fig.add_subplot(gs[1, :])
    
    x = np.arange(len(coa_names))
    width = 0.25
    
    # Normalize for comparison
    norm_kill = (kill_ratios - kill_ratios.min()) / (kill_ratios.max() - kill_ratios.min()) * 100
    norm_survival = blue_survivals
    norm_time = planning_times / planning_times.max() * 100
    
    bars1 = ax3.bar(x - width, norm_kill, width, label='Kill Ratio (normalized)',
                   color='steelblue', alpha=0.7, edgecolor='darkblue', linewidth=1.5)
    bars2 = ax3.bar(x, norm_survival, width, label='Blue Survival %',
                   color='green', alpha=0.7, edgecolor='darkgreen', linewidth=1.5)
    bars3 = ax3.bar(x + width, norm_time, width, label='Planning Time (% of max)',
                   color='orange', alpha=0.7, edgecolor='darkorange', linewidth=1.5)
    
    ax3.set_xlabel('COA (Ordered by Planning Time: Low → High)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Performance Metrics (Normalized)', fontsize=14, fontweight='bold')
    ax3.set_title('COA Performance Comparison: Early vs. Late COAs\n(Left = Less Planning, Right = More Planning)', 
                  fontsize=15, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(coa_names, fontsize=11)
    ax3.legend(fontsize=13, loc='upper left')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 110])
    ax3.tick_params(axis='both', which='major', labelsize=11)
    
    # Add insight text
    early_avg_kr = np.mean(kill_ratios[:len(kill_ratios)//2])
    late_avg_kr = np.mean(kill_ratios[len(kill_ratios)//2:])
    
    if late_avg_kr > early_avg_kr * 1.1:
        insight = "Later COAs perform BETTER ✓\n(Human+AI improves)"
        color = 'lightgreen'
    elif early_avg_kr > late_avg_kr * 1.1:
        insight = "Early COAs perform BETTER\n(AI alone may be sufficient)"
        color = 'lightcoral'
    else:
        insight = "Mixed results\n(No clear trend)"
        color = 'lightyellow'
    
    ax3.text(0.98, 0.98, insight, transform=ax3.transAxes,
            fontsize=14, fontweight='bold', verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    
    plt.suptitle('Planning Time vs Performance: Does Human Refinement Improve COAs?', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_path = os.path.join(OUTPUT_DIR, 'planning_time_vs_performance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

if __name__ == '__main__':
    print("=" * 70)
    print("PLANNING TIME VS PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    # Extract COA development times
    print("\nExtracting COA development times from web logs...")
    coa_dev_times = extract_coa_development_times()
    print(f"Found {len(coa_dev_times)} COAs")
    
    # Analyze performance for each COA
    print("\nAnalyzing performance for each COA...")
    coa_performance = {}
    for coa_id in coa_dev_times.keys():
        perf = analyze_coa_performance(coa_id)
        if perf:
            coa_performance[coa_id] = perf
            print(f"  COA {coa_id[:20]}...: {perf['num_sims']} sims, avg kill ratio: {perf['avg_kill_ratio']:.2f}")
    
    print(f"\nFound simulation data for {len(coa_performance)} COAs")
    
    # Generate plot
    if coa_performance:
        plot_planning_vs_performance(coa_dev_times, coa_performance)
        
        # Print summary
        print("\n" + "=" * 70)
        print("ANALYSIS SUMMARY")
        print("=" * 70)
        
        planning_times = [coa_dev_times[cid]['development_time_minutes'] for cid in coa_performance.keys()]
        kill_ratios = [coa_performance[cid]['avg_kill_ratio'] for cid in coa_performance.keys()]
        
        if len(planning_times) > 2:
            corr = np.corrcoef(planning_times, kill_ratios)[0, 1]
            print(f"\nPlanning Time vs Kill Ratio Correlation: {corr:.3f}")
            
            if corr > 0.3:
                print("  ✓ MORE planning time correlates with BETTER performance")
                print("    → Human refinement improves COAs (Human+AI > AI alone)")
            elif corr < -0.3:
                print("  ✗ MORE planning time correlates with WORSE performance")
                print("    → AI-generated COAs may be better (AI alone may be sufficient)")
            else:
                print("  ≈ NO strong correlation between planning time and performance")
                print("    → Mixed bag - quality varies regardless of planning time")
    else:
        print("\nNo COAs with simulation data found!")
    
    print("=" * 70)
