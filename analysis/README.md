# SC3 Performance Analysis Scripts

## Overview

This directory contains Python scripts for analyzing SC3 simulation data and generating performance plots for the DTRA briefing. Each script focuses on a specific aspect of simulation performance.

## Prerequisites

```bash
pip install numpy matplotlib scipy
```

## Scripts

### 1. Casualties Over Time (`1_casualties_over_time.py`)

**Purpose:** Analyzes and visualizes casualty accumulation throughout simulation runs.

**What it shows:**
- Cumulative combat power lost over time for Red and Blue forces
- Casualty rate (losses per hour) throughout the simulation
- Comparison across multiple simulation runs
- Statistical summaries (total casualties, averages, standard deviations)

**Outputs:**
- `casualties_over_time_*.png` - Individual simulation casualty plots
- `casualties_summary.png` - Comparison across multiple simulations

**Key Insights:**
- When did most casualties occur?
- Which force suffered more losses?
- How consistent are casualty patterns across simulations?
- What is the casualty rate over time (early vs. late combat)?

**Usage:**
```bash
cd analysis
python 1_casualties_over_time.py
```

---

### 2. Force Strength Evolution (`2_force_strength_evolution.py`)

**Purpose:** Tracks remaining combat power for both forces throughout the simulation.

**What it shows:**
- Remaining combat power over time (not cumulative losses, but what's left)
- Force ratio (Blue/Red) evolution - who has the advantage?
- Percentage of initial strength remaining
- Stacked area chart showing total force composition
- Identification of decisive moments (when balance shifts dramatically)

**Outputs:**
- `force_strength_*.png` - Multi-panel plot showing:
  - Force strength over time
  - Force ratio (balance of power)
  - Attrition rate (% of initial strength)
  - Combined force composition

**Key Insights:**
- When does one force gain/lose advantage?
- How quickly do forces degrade?
- What's the final force ratio?
- When do decisive moments occur in combat?

**Usage:**
```bash
python 2_force_strength_evolution.py
```

---

### 3. Combat Intensity Heatmap (`3_combat_intensity_heatmap.py`)

**Purpose:** Analyzes when and where combat was most intense.

**What it shows:**
- Number of combat events per time period (30-minute bins)
- Damage dealt per time period
- Identification of distinct combat phases (engagement, peak, decline)
- Geographic heatmap of combat locations
- Moving average to identify trends

**Outputs:**
- `combat_intensity_*.png` - Three-panel plot showing:
  - Combat events over time
  - Damage dealt over time
  - Combined activity profile
- `combat_geographic_*.png` - Geographic heatmap of combat hotspots

**Key Insights:**
- When was combat most intense?
- How many distinct combat phases occurred?
- Where (geographically) did most fighting occur?
- Are there lulls in combat? Sustained engagements?

**Usage:**
```bash
python 3_combat_intensity_heatmap.py
```

---

### 4. Unit Survival Analysis (`4_unit_survival_analysis.py`)

**Purpose:** Analyzes which units survive and for how long.

**What it shows:**
- Survival probability curves (similar to Kaplan-Meier)
- Number of units remaining over time
- Distribution of elimination times
- Survival rates by unit size/type
- Mean time to elimination

**Outputs:**
- `survival_analysis_*.png` - Four-panel plot showing:
  - Survival rate percentage over time
  - Absolute number of units remaining
  - Histogram of when units were eliminated
  - Survival rates by unit characteristics

**Key Insights:**
- What percentage of each force survives?
- When do most eliminations occur?
- Which unit types survive best?
- What's the mean survival time for eliminated units?

**Usage:**
```bash
python 4_unit_survival_analysis.py
```

---

### 5. COA Comparison Analysis (`5_coa_comparison.py`)

**Purpose:** Compares outcomes across different simulations and COAs.

**What it shows:**
- Casualties across all analyzed simulations
- Distribution of final force ratios (who won?)
- Distribution of combat durations (how long did fighting last?)
- Casualty ratio distribution (Blue/Red loss comparison)
- Correlation between Blue and Red casualties
- COA development time from web interaction logs
- Number of user interactions per COA

**Outputs:**
- `simulation_comparison.png` - Five-panel comparison showing:
  - Casualties across all simulations with averages
  - Distribution of final force ratios
  - Combat duration distribution
  - Casualty ratio distribution
  - Scatter plot of Blue vs. Red casualties
- `coa_development.png` - COA development patterns:
  - Time spent developing each COA
  - Number of interactions per COA

**Key Insights:**
- How consistent are simulation outcomes?
- What's the typical final force ratio?
- How long does combat typically last?
- Is there a relationship between COA development time and outcomes?
- Which simulations were outliers?

**Usage:**
```bash
python 5_coa_comparison.py
```

---

### Master Runner (`run_all_analyses.py`)

**Purpose:** Runs all analysis scripts in sequence.

**Usage:**
```bash
python run_all_analyses.py [--sims 10] [--output ./outputs]
```

**Options:**
- `--sims N`: Number of simulations to analyze (default: 10)
- `--output DIR`: Output directory for plots (default: ./outputs)

This will generate all plots automatically and provide a summary of successful analyses.

---

## Output Directory Structure

After running the analyses, the `outputs/` directory will contain:

```
outputs/
├── casualties_over_time_*.png      # Individual simulation casualty plots
├── casualties_summary.png          # Multi-simulation casualty comparison
├── force_strength_*.png           # Force evolution analysis
├── combat_intensity_*.png         # Combat timing analysis
├── combat_geographic_*.png        # Geographic combat distribution
├── survival_analysis_*.png        # Unit survival plots
├── simulation_comparison.png      # Cross-simulation comparison
└── coa_development.png           # COA development patterns
```

## Recommended Workflow

### For Quick Analysis (10 simulations):
```bash
cd analysis
python run_all_analyses.py --sims 10
```

### For Detailed Single Simulation:
```bash
# Run each script individually to see detailed console output
python 1_casualties_over_time.py
python 2_force_strength_evolution.py
python 3_combat_intensity_heatmap.py
python 4_unit_survival_analysis.py
python 5_coa_comparison.py
```

### For Full Dataset (all 85 simulations):
```bash
python run_all_analyses.py --sims 85
```
**Note:** This will take longer but provides comprehensive statistics.

## Performance Metrics Summary

Each script calculates specific metrics useful for the DTRA briefing:

### Casualty Metrics
- Total casualties (Blue and Red)
- Casualty rate per hour
- Cumulative losses over time
- Blue/Red casualty ratio

### Force Effectiveness
- Final force ratio (who has advantage at end?)
- Rate of combat power degradation
- Survival percentages
- Time to eliminate units

### Combat Characteristics
- Combat duration (when does fighting stop?)
- Combat intensity (events per time period)
- Number of combat phases
- Geographic distribution

### COA Analysis
- Development time per COA
- User interaction patterns
- Outcome consistency
- Success metrics

## Tips for DTRA Briefing

1. **Start with overview plots**: Use `simulation_comparison.png` to show overall patterns
2. **Dive into specifics**: Use individual simulation plots to highlight interesting cases
3. **Show trends**: Force strength evolution clearly shows "who's winning"
4. **Combat intensity**: Heatmaps show when fighting occurred (useful for operational analysis)
5. **Survival analysis**: Good for unit-level effectiveness discussions

## Customization

Each script can be modified to:
- Change time resolution (currently 30-60 minute bins)
- Adjust color schemes
- Add additional metrics
- Filter by specific unit types or geographic regions
- Export data to CSV for further analysis

## Troubleshooting

**Issue:** `ModuleNotFoundError: No module named 'matplotlib'`
**Solution:** `pip install matplotlib numpy scipy`

**Issue:** "No simulation directories found"
**Solution:** Ensure you're running from the `analysis/` directory and the data is in `../data/sim/`

**Issue:** Memory errors with large datasets
**Solution:** Reduce number of simulations analyzed or process in batches

## Data Requirements

Scripts expect the following directory structure:
```
../data/
├── sim/
│   ├── [sim-uuid-1]/
│   │   ├── combat_events.json
│   │   ├── damage_events.json
│   │   ├── config.json
│   │   ├── results.json
│   │   └── *_id_to_*_id.json
│   └── [sim-uuid-2]/
│       └── ...
└── web/
    ├── TS-ARL3538/
    │   └── TS-ARL3538.jsonl
    ├── TS-ARL3542/
    │   └── TS-ARL3542.jsonl
    └── TS-ARL3543/
        └── TS-ARL3543.jsonl
```

## Contact

For questions about the analysis scripts or modifications needed for the briefing, contact Kevin or Mike.

---

**Last Updated:** October 23, 2025  
**For:** DTRA Briefing, November 5, 2025  
**System:** SC3 (Scalable Command and Control for Coalition operations)
