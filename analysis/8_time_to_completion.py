"""
Time to Completion Analysis
============================

This script analyzes both:
1. Simulation time to completion (operational speed - in-game time)
2. COA development time (planning effort - real time spent building COAs)

Shows the relationship between planning complexity and operational execution time.

Outputs:
    - time_to_completion.png: Combined analysis of both metrics
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

# Configuration
DATA_DIR = '../data/sim'
WEB_DIR = '../data/web'
OUTPUT_DIR = './outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_simulation_completion_times():
    """Extract completion times from simulations."""
    sim_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    results = []
    for sim_id in sim_dirs[:30]:
        sim_path = os.path.join(DATA_DIR, sim_id)
        
        try:
            with open(os.path.join(sim_path, 'config.json')) as f:
                config = json.load(f)
            
            with open(os.path.join(sim_path, 'damage_events.json')) as f:
                damage_events = json.load(f)
            
            # Get simulation time parameters
            max_scenario_time = config.get('max_scenario_time', 345600)  # Default 96 hours
            timestep = config.get('timestep', 60)
            
            # Get actual completion time from last event
            if damage_events:
                last_event_time = max(event['timestamp'] for event in damage_events)
                completion_time_hours = last_event_time / 3600  # Convert to hours
            else:
                # No combat - use max scenario time
                completion_time_hours = max_scenario_time / 3600
            
            # Get COA name if available
            coa_name = config.get('name', 'Unknown')
            
            results.append({
                'sim_id': sim_id,
                'coa_name': coa_name,
                'completion_time_hours': completion_time_hours,
                'max_scenario_time_hours': max_scenario_time / 3600,
                'timestep': timestep
            })
        except Exception as e:
            print(f"Error loading {sim_id}: {e}")
    
    # Sort by completion time
    results.sort(key=lambda x: x['completion_time_hours'])
    
    return results

def analyze_coa_development_times():
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

def plot_time_to_completion(sim_results, coa_dev_times):
    """Create comprehensive time to completion analysis."""
    print(f"Creating time to completion analysis...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35)
    
    # Extract data
    completion_times = [r['completion_time_hours'] for r in sim_results]
    x_sims = np.arange(len(sim_results))
    
    # Plot 1: Simulation Completion Times
    ax1 = fig.add_subplot(gs[0, :])
    
    bars = ax1.bar(x_sims, completion_times, color='steelblue', alpha=0.7, 
                   edgecolor='darkblue', linewidth=1.5)
    
    avg_completion = np.mean(completion_times)
    ax1.axhline(avg_completion, color='red', linestyle='--', linewidth=2.5,
                label=f'Average: {avg_completion:.1f} hours')
    
    # Add trend line
    if len(completion_times) > 1:
        z = np.polyfit(x_sims, completion_times, 1)
        p = np.poly1d(z)
        ax1.plot(x_sims, p(x_sims), "g--", linewidth=3, alpha=0.7,
                label=f'Trend: {"Faster ✓" if z[0] < 0 else "Slower"}')
    
    ax1.set_xlabel('Simulation Run (Chronological Order)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time to Completion (Hours)', fontsize=12, fontweight='bold')
    ax1.set_title('Operational Speed: In-Game Time to Complete Mission\n(Lower = Faster)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add stats box
    stats_text = (f'Min: {min(completion_times):.1f} hrs\n'
                 f'Max: {max(completion_times):.1f} hrs\n'
                 f'Median: {np.median(completion_times):.1f} hrs\n'
                 f'Std Dev: {np.std(completion_times):.1f} hrs')
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Plot 2: Completion Time Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    
    ax2.hist(completion_times, bins=15, color='steelblue', alpha=0.7, 
            edgecolor='darkblue', linewidth=1.5)
    ax2.axvline(avg_completion, color='red', linestyle='--', linewidth=2.5,
                label=f'Average: {avg_completion:.1f} hrs')
    ax2.axvline(np.median(completion_times), color='green', linestyle='-', linewidth=2.5,
                label=f'Median: {np.median(completion_times):.1f} hrs')
    
    ax2.set_xlabel('Time to Completion (Hours)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Simulations', fontsize=11, fontweight='bold')
    ax2.set_title('Distribution of Completion Times', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: COA Development Time
    ax3 = fig.add_subplot(gs[1, 1])
    
    if coa_dev_times:
        coa_names = list(coa_dev_times.keys())
        dev_times = [coa_dev_times[name]['development_time_minutes'] for name in coa_names]
        
        bars = ax3.barh(range(len(coa_names)), dev_times, color='coral', 
                       alpha=0.7, edgecolor='darkred', linewidth=1.5)
        
        avg_dev = np.mean(dev_times)
        ax3.axvline(avg_dev, color='blue', linestyle='--', linewidth=2.5,
                   label=f'Average: {avg_dev:.1f} min')
        
        ax3.set_yticks(range(len(coa_names)))
        ax3.set_yticklabels([f'COA {i+1}' for i in range(len(coa_names))], fontsize=10)
        ax3.set_xlabel('Development Time (Minutes)', fontsize=11, fontweight='bold')
        ax3.set_title('Planning Effort: COA Development Time\n(Real Time Spent Building)', 
                     fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Add stats
        stats_text = (f'Total COAs: {len(coa_names)}\n'
                     f'Min: {min(dev_times):.1f} min\n'
                     f'Max: {max(dev_times):.1f} min')
        ax3.text(0.98, 0.02, stats_text, transform=ax3.transAxes,
                fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    else:
        ax3.text(0.5, 0.5, 'No COA development data available', 
                transform=ax3.transAxes, ha='center', va='center',
                fontsize=12, style='italic')
        ax3.set_title('Planning Effort: COA Development Time', 
                     fontsize=12, fontweight='bold')
    
    plt.suptitle('Time to Completion: Operational Speed vs Planning Effort', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_path = os.path.join(OUTPUT_DIR, 'time_to_completion.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

if __name__ == '__main__':
    print("=" * 70)
    print("TIME TO COMPLETION ANALYSIS")
    print("=" * 70)
    
    # Analyze simulation completion times
    print("\nAnalyzing simulation completion times...")
    sim_results = analyze_simulation_completion_times()
    print(f"Analyzed {len(sim_results)} simulations")
    
    # Analyze COA development times
    print("\nAnalyzing COA development times...")
    coa_dev_times = analyze_coa_development_times()
    print(f"Found {len(coa_dev_times)} COAs with development data")
    
    # Generate plot
    plot_time_to_completion(sim_results, coa_dev_times)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TIME TO COMPLETION SUMMARY")
    print("=" * 70)
    
    completion_times = [r['completion_time_hours'] for r in sim_results]
    print(f"\nSimulation Completion Times (Operational Speed):")
    print(f"  Average: {np.mean(completion_times):.1f} hours")
    print(f"  Median: {np.median(completion_times):.1f} hours")
    print(f"  Range: {min(completion_times):.1f} - {max(completion_times):.1f} hours")
    
    if coa_dev_times:
        dev_times = [data['development_time_minutes'] for data in coa_dev_times.values()]
        print(f"\nCOA Development Times (Planning Effort):")
        print(f"  Average: {np.mean(dev_times):.1f} minutes")
        print(f"  Median: {np.median(dev_times):.1f} minutes")
        print(f"  Range: {min(dev_times):.1f} - {max(dev_times):.1f} minutes")
    
    print("=" * 70)
