# SC3 Performance Analysis - Quick Start Guide

## For: DTRA Briefing (November 5, 2025)

### What We Have

**Data Collection:** May 2021 at ARL  
**Participants:** Mission Command Battle Labs personnel  
**Scenario:** "Operation Tropic Tortoise" - Red vs. Blue force engagement  
**Simulations:** 85 simulation runs  
**COAs Developed:** 9 distinct Courses of Action across 3 user sessions  

### Data Summary

- **163 military units** per simulation (Red and Blue forces)
- **96-hour simulated scenarios** (4 days of "in-game" time)
- **~6 minutes actual runtime** per simulation (1000x timescale)
- **~17,000 damage events** per simulation
- **~67 units survive** on average (96 eliminated)

### Analysis Scripts Created

Five Python scripts that generate publication-quality plots:

1. **Casualties Over Time** - Shows when and how casualties accumulate
2. **Force Strength Evolution** - Tracks remaining combat power and who has advantage
3. **Combat Intensity Heatmap** - Identifies when/where combat was most intense
4. **Unit Survival Analysis** - Which units survive and for how long
5. **COA Comparison** - Compares outcomes across different simulations

### Quick Start (10 minutes)

```bash
# 1. Install dependencies
cd analysis
pip install -r requirements.txt

# 2. Test setup
python test_setup.py

# 3. Run all analyses
python run_all_analyses.py --sims 10

# Plots will be in: analysis/outputs/
```

### What the Plots Show

**For "Performance" (as requested):**

1. **Casualty trends over time**
   - When do most losses occur?
   - Early engagement vs. sustained combat
   - Red vs. Blue losses

2. **Force effectiveness**
   - Which force maintains advantage?
   - Rate of combat power degradation
   - Final force ratios (who "wins")

3. **Combat characteristics**
   - Intensity over time (peak engagement periods)
   - Duration until combat cessation
   - Geographic hotspots

4. **COA quality**
   - Outcome consistency across simulations
   - Development time vs. results
   - User interaction patterns

### Key Findings (Example - based on initial exploration)

- Average **~214 total combat power lost** per simulation
- Combat typically lasts **~70 hours** of simulated time
- **~59% of units eliminated** on average
- **Distinct combat phases** identifiable (engagement, peak, decline)
- Users spent **57-154 minutes** per session developing COAs

### Files Generated

Each plot is high-resolution (300 DPI) suitable for briefings:

- `casualties_over_time_*.png` - Individual and summary casualty plots
- `force_strength_*.png` - Multi-panel force evolution analysis
- `combat_intensity_*.png` - Temporal combat patterns
- `combat_geographic_*.png` - Spatial combat distribution
- `survival_analysis_*.png` - Unit survival curves
- `simulation_comparison.png` - Cross-simulation comparisons
- `coa_development.png` - COA development patterns

### Recommended Briefing Flow

1. **Overview**: Show `simulation_comparison.png` (big picture)
2. **Casualties**: Show `casualties_over_time_*.png` (what you asked for specifically)
3. **Force Evolution**: Show `force_strength_*.png` (performance over time)
4. **Combat Timing**: Show `combat_intensity_*.png` (operational insights)
5. **COA Analysis**: Show `coa_development.png` (link to user sessions)

### Customization Options

Each script can be modified for:
- Different time resolutions
- Specific unit type analysis
- Geographic filtering
- Custom metrics
- Export to CSV for additional analysis

### Next Steps

1. **Run the analyses** (10 minutes)
2. **Review the plots** (30 minutes)
3. **Select best plots for briefing** (varies)
4. **Customize if needed** (optional)

### Timeline

- **Today (Oct 23):** Run analyses, review outputs
- **Oct 24-31:** Refine plots, add customizations if needed
- **Nov 1-4:** Prepare briefing materials
- **Nov 5:** DTRA Briefing/Demonstration

### Questions?

All scripts include detailed documentation:
- `analysis/README.md` - Complete guide for all scripts
- Each script has inline comments and docstrings
- `test_setup.py` - Verifies everything works before running

### Contact

- Kevin: Analysis implementation
- Mike: Coordination/review
- Stephen: Data provider, domain expertise

---

**Status:** Ready to run  
**Estimated Time:** 10-30 minutes for full analysis  
**Output:** ~10-15 high-quality performance plots
