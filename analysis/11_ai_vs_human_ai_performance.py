"""
AI vs Human+AI Performance Analysis
====================================

This script analyzes the iterative refinement process:
1. AI creates initial COA (first simulation = baseline)
2. Human refines COA (subsequent simulations show improvements)
3. Compare first vs. last iteration to measure human contribution

Answers: What does AI bring vs. what does Human bring?

Outputs:
    - ai_vs_human_ai_performance.png: AI baseline vs Human+AI final performance
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime

# Configuration
DATA_DIR = '../data/sim'
OUTPUT_DIR = './outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_sim_performance(sim_id):
    """Analyze performance metrics for a single simulation."""
    try:
        sim_path = f'{DATA_DIR}/{sim_id}'
        
        with open(f'{sim_path}/config.json') as f:
            config = json.load(f)
        with open(f'{sim_path}/damage_events.json') as f:
            damage_events = json.load(f)
        with open(f'{sim_path}/external_id_to_internal_id.json') as f:
            id_map = json.load(f)
        with open(f'{sim_path}/results.json') as f:
            results = json.load(f)
        
        # Get factions
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
        
        # Calculate metrics
        kill_ratio = red_casualties / blue_casualties if blue_casualties > 0 else 10.0
        blue_survival = ((blue_initial - blue_casualties) / blue_initial * 100) if blue_initial > 0 else 0
        completion_time = last_combat_time / 3600
        
        return {
            'end_time': results.get('end_time', 0),
            'kill_ratio': kill_ratio,
            'blue_survival': blue_survival,
            'blue_casualties': blue_casualties,
            'red_casualties': red_casualties,
            'completion_time': completion_time
        }
    except Exception as e:
        return None

def collect_coa_iterations():
    """Collect all simulations grouped by COA ID with chronological ordering."""
    sim_dirs = os.listdir(DATA_DIR)
    coa_sims = defaultdict(list)
    
    for sim_id in sim_dirs:
        try:
            with open(f'{DATA_DIR}/{sim_id}/config.json') as f:
                config = json.load(f)
                tags = config.get('tags', {})
                coa_id = tags.get('coa_id')
                
                if coa_id:
                    perf = analyze_sim_performance(sim_id)
                    if perf:
                        perf['sim_id'] = sim_id
                        coa_sims[coa_id].append(perf)
        except:
            continue
    
    # Sort each COA's simulations chronologically
    for coa_id in coa_sims:
        coa_sims[coa_id].sort(key=lambda x: x['end_time'])
    
    return coa_sims

def analyze_ai_vs_human(coa_sims):
    """Compare first iteration (AI) vs last iteration (Human+AI) for each COA."""
    results = []
    
    for coa_id, sims in coa_sims.items():
        if len(sims) >= 2:  # Need at least 2 iterations to compare
            first = sims[0]  # AI baseline
            last = sims[-1]   # Final human-refined version
            
            time_span = (last['end_time'] - first['end_time']) / 60  # minutes
            
            results.append({
                'coa_id': coa_id,
                'num_iterations': len(sims),
                'ai_kill_ratio': first['kill_ratio'],
                'final_kill_ratio': last['kill_ratio'],
                'ai_blue_survival': first['blue_survival'],
                'final_blue_survival': last['blue_survival'],
                'improvement_kr': last['kill_ratio'] - first['kill_ratio'],
                'improvement_pct': ((last['kill_ratio'] / first['kill_ratio']) - 1) * 100 if first['kill_ratio'] > 0 else 0,
                'refinement_time_min': time_span,
                'all_iterations': sims
            })
    
    return results

def plot_ai_vs_human_ai(analysis_results):
    """Create comprehensive AI vs Human+AI comparison plots."""
    print(f"Creating AI vs Human+AI performance comparison...")
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.40)
    
    # Extract data
    coa_names = [f"COA {i+1}" for i in range(len(analysis_results))]
    ai_kr = np.array([r['ai_kill_ratio'] for r in analysis_results])
    final_kr = np.array([r['final_kill_ratio'] for r in analysis_results])
    improvements = np.array([r['improvement_kr'] for r in analysis_results])
    improvement_pct = np.array([r['improvement_pct'] for r in analysis_results])
    num_iterations = np.array([r['num_iterations'] for r in analysis_results])
    
    # Plot 1: AI Baseline vs Human+AI Final
    ax1 = fig.add_subplot(gs[0, :])
    
    x = np.arange(len(coa_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, ai_kr, width, label='AI Baseline (First Iteration)',
                   color='steelblue', alpha=0.7, edgecolor='darkblue', linewidth=2)
    bars2 = ax1.bar(x + width/2, final_kr, width, label='Human+AI Final (Last Iteration)',
                   color='green', alpha=0.7, edgecolor='darkgreen', linewidth=2)
    
    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax1.text(bar1.get_x() + bar1.get_width()/2, height1, f'{height1:.2f}',
                ha='center', va='bottom', fontsize=10)
        ax1.text(bar2.get_x() + bar2.get_width()/2, height2, f'{height2:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax1.set_xlabel('Course of Action', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Kill Ratio (Red/Blue)', fontsize=14, fontweight='bold')
    ax1.set_title('AI Baseline vs Human+AI Final Performance\n(What Does Human Refinement Add?)', 
                  fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(coa_names, fontsize=11)
    ax1.legend(fontsize=13, loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='both', which='major', labelsize=11)
    
    # Add summary stats
    avg_ai = np.mean(ai_kr)
    avg_final = np.mean(final_kr)
    avg_improvement = np.mean(improvement_pct)

    stats_text = f'AI Baseline Avg: {avg_ai:.2f}\nHuman+AI Avg: {avg_final:.2f}\nAvg Improvement: {avg_improvement:+.1f}%'
    color = 'lightgreen' if avg_improvement > 10 else 'lightyellow' if avg_improvement > 0 else 'lightcoral'
    ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes,
            fontsize=13, fontweight='bold', verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    
    # Plot 2: Improvement Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax2.barh(coa_names, improvements, color=colors, alpha=0.7, 
                    edgecolor='black', linewidth=1.5)
    
    ax2.axvline(0, color='black', linestyle='-', linewidth=2)
    ax2.axvline(np.mean(improvements), color='blue', linestyle='--', linewidth=2.5,
                label=f'Average: {np.mean(improvements):+.2f}')
    
    ax2.set_xlabel('Kill Ratio Improvement', fontsize=13, fontweight='bold')
    ax2.set_ylabel('COA', fontsize=13, fontweight='bold')
    ax2.set_title('Human Contribution to Performance\n(Positive = Improvement)', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.tick_params(axis='both', which='major', labelsize=11)
    
    # Plot 3: Improvement vs Number of Iterations
    ax3 = fig.add_subplot(gs[1, 1])
    
    scatter = ax3.scatter(num_iterations, improvement_pct, s=200, alpha=0.6,
                         c=improvement_pct, cmap='RdYlGn', vmin=-50, vmax=200,
                         edgecolors='black', linewidth=2)
    
    # Add trend line
    if len(num_iterations) > 2:
        z = np.polyfit(num_iterations, improvement_pct, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(num_iterations.min(), num_iterations.max(), 100)
        ax3.plot(x_trend, p(x_trend), "b--", linewidth=3, alpha=0.7,
                label=f'Trend: {"More iterations → Better" if z[0] > 0 else "Fewer iterations → Better"}')
    
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Number of Refinement Iterations', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Performance Improvement (%)', fontsize=13, fontweight='bold')
    ax3.set_title('Does More Refinement Help?\n(Iterations vs Improvement)', 
                  fontsize=14, fontweight='bold')
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=11)
    
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Improvement %', fontsize=12)
    cbar.ax.tick_params(labelsize=11)
    
    # Plot 4: Example Iteration Progression (best COA)
    ax4 = fig.add_subplot(gs[2, :])
    
    # Find COA with most iterations
    best_coa = max(analysis_results, key=lambda x: x['num_iterations'])
    iterations = best_coa['all_iterations']
    
    iter_nums = list(range(1, len(iterations) + 1))
    iter_kr = [it['kill_ratio'] for it in iterations]
    
    ax4.plot(iter_nums, iter_kr, 'o-', linewidth=3, markersize=10, color='darkgreen',
            markerfacecolor='lightgreen', markeredgewidth=2, markeredgecolor='darkgreen')
    
    # Highlight first and last
    ax4.plot(1, iter_kr[0], 'o', markersize=15, color='steelblue', 
            label=f'AI Baseline: {iter_kr[0]:.2f}', zorder=5)
    ax4.plot(len(iter_kr), iter_kr[-1], 'o', markersize=15, color='green',
            label=f'Final Human+AI: {iter_kr[-1]:.2f}', zorder=5)
    
    # Add iteration labels
    for i, (x, y) in enumerate(zip(iter_nums, iter_kr), 1):
        ax4.annotate(f'{y:.2f}', (x, y), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=10, fontweight='bold')
    
    ax4.axhline(iter_kr[0], color='steelblue', linestyle='--', alpha=0.3, linewidth=1.5)
    ax4.set_xlabel('Iteration Number (Time →)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Kill Ratio', fontsize=14, fontweight='bold')
    ax4.set_title(f'Example: COA Refinement Process ({len(iterations)} iterations)\n'
                  f'AI → Human Refinement → Final ({iter_kr[-1]/iter_kr[0]:.1f}x improvement)', 
                  fontsize=15, fontweight='bold')
    ax4.legend(fontsize=13, loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(iter_nums)
    ax4.tick_params(axis='both', which='major', labelsize=11)
    
    # Add insight
    if iter_kr[-1] > iter_kr[0] * 1.5:
        insight = f'Human+AI is {iter_kr[-1]/iter_kr[0]:.1f}x better ✓'
        color = 'lightgreen'
    elif iter_kr[-1] > iter_kr[0]:
        insight = 'Modest improvement'
        color = 'lightyellow'
    else:
        insight = 'AI baseline was better'
        color = 'lightcoral'
    
    ax4.text(0.98, 0.02, insight, transform=ax4.transAxes,
            fontsize=14, fontweight='bold', verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    
    plt.suptitle('AI vs Human+AI: What Does Each Bring to the Table?', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_path = os.path.join(OUTPUT_DIR, 'ai_vs_human_ai_performance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

if __name__ == '__main__':
    print("=" * 70)
    print("AI vs HUMAN+AI PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    # Collect all COA iterations
    print("\nCollecting COA iteration data...")
    coa_sims = collect_coa_iterations()
    print(f"Found {len(coa_sims)} unique COAs")
    
    # Filter to COAs with multiple iterations
    multi_iter_coas = {k: v for k, v in coa_sims.items() if len(v) >= 2}
    print(f"COAs with 2+ iterations (refinement): {len(multi_iter_coas)}")
    
    # Analyze AI vs Human contribution
    print("\nAnalyzing AI baseline vs Human+AI final performance...")
    analysis_results = analyze_ai_vs_human(coa_sims)
    
    # Sort by number of iterations for display
    analysis_results.sort(key=lambda x: x['num_iterations'], reverse=True)
    
    # Print summary
    print(f"\n{'COA':<6} {'Iters':<7} {'AI KR':<8} {'Final KR':<10} {'Change':<12} {'%'}")
    print("-" * 60)
    for i, r in enumerate(analysis_results, 1):
        print(f"COA {i:<3} {r['num_iterations']:<7} {r['ai_kill_ratio']:<8.2f} "
              f"{r['final_kill_ratio']:<10.2f} {r['improvement_kr']:<+12.2f} "
              f"{r['improvement_pct']:+.0f}%")
    
    # Generate plot
    if analysis_results:
        plot_ai_vs_human_ai(analysis_results)
        
        # Print conclusions
        print("\n" + "=" * 70)
        print("CONCLUSIONS")
        print("=" * 70)
        
        avg_ai = np.mean([r['ai_kill_ratio'] for r in analysis_results])
        avg_final = np.mean([r['final_kill_ratio'] for r in analysis_results])
        avg_improvement = np.mean([r['improvement_pct'] for r in analysis_results])
        
        positive_improvements = sum(1 for r in analysis_results if r['improvement_kr'] > 0)
        
        print(f"\nAI Baseline Performance: {avg_ai:.2f} average kill ratio")
        print(f"Human+AI Final Performance: {avg_final:.2f} average kill ratio")
        print(f"Average Improvement: {avg_improvement:+.1f}%")
        print(f"COAs that improved: {positive_improvements}/{len(analysis_results)} ({positive_improvements/len(analysis_results)*100:.0f}%)")
        
        print("\nWhat AI brings: Solid baseline (~{:.2f} kill ratio)".format(avg_ai))
        print("What Human brings: {:.1f}% average improvement through refinement".format(avg_improvement))
        
        if avg_improvement > 20:
            print("\n✓ Human+AI is SIGNIFICANTLY better than AI alone")
        elif avg_improvement > 5:
            print("\n✓ Human+AI is moderately better than AI alone")
        else:
            print("\n≈ Human refinement has minimal impact")
        
        print("=" * 70)
    else:
        print("\nNo COAs with multiple iterations found!")
