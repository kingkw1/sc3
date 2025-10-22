# SC3 Data Analysis - ARL Mission Command Battle Labs

## Overview

This dataset contains simulation results and user interaction data from the SC3 (Scalable Command and Control for Coalition operations) system demonstration conducted at Army Research Laboratory (ARL) in May 2021. The demonstration involved personnel from Mission Command Battle Labs who used the SC3 tool to develop and evaluate Courses of Action (COAs) for red/blue force military scenarios.

**Collection Period:** May 2021  
**Purpose:** Chem-bio stakeholder briefing/demonstration for DTRA  
**Briefing Date:** November 5, 2025  

## Dataset Structure

The dataset is organized into three main directories:

```
data/
├── sim/          # Simulation results (85 simulation runs)
├── web/          # User interaction logs (3 user sessions)
└── compressed/   # Original compressed data files
```

## Simulation Data (`sim/`)

### Overview
- **Total Simulation Runs:** 85 unique simulation executions
- **Scenario Name:** Operation Tropic Tortoise
- **Geographic Area:** Southern California region (coordinates approximately -117.6 to -116.3 longitude, 34.2 to 35.1 latitude)
- **Duration:** 96 hours (345,600 seconds) per simulation
- **Timestep:** 60 seconds

### Directory Structure

Each simulation run is stored in a directory named with a unique UUID. Each directory contains:

#### 1. `config.json` (14,812 lines)
Complete simulation configuration including:
- **Scenario metadata:** ID, name, max_scenario_time, timestep, timescale
- **Area of Interest (AOI):** Geographic boundaries defining the operational area
- **Factions:** 
  - Blue Force (ID: e54bfe7a-7a43-4ebf-887a-55c5dd4947c6)
  - Red Force (ID: 70865c57-86f4-42e9-abe1-f1a7fdaf0ed0)
  - Stance: Hostile to each other
- **Entities:** 163 military units per simulation
  - Unit types include: Field Artillery, Infantry Brigades, Mechanized units
  - Properties: position, combat_power (initial & max), combat_range, sensor_range
  - Metadata tags: SIDC codes, Faction, Function, Status, Modifier/Size
  - Combat ranges typically 5,000m, sensor ranges 7,500m
  - Max velocities around 3.0 m/s
  - Initial combat power values: typically 3-150 units

#### 2. `combat_events.json` (~30,863 lines)
Records of all combat engagements:
```json
{
  "combatant": 17179869884,    // Entity ID engaging
  "targets": [17179869811],     // Target entity ID(s)
  "timestamp": 60               // Time in seconds
}
```

#### 3. `damage_events.json` (~102,476 lines)
Detailed damage assessments for each combat event:
```json
{
  "damage": 0.0978,             // Damage value
  "source": 17179869884,        // Attacking entity ID
  "target": 17179869811,        // Receiving entity ID
  "timestamp": 60               // Time in seconds
}
```
- **Typical simulation:** ~17,000 damage events
- **Time range:** 60 seconds to ~71 hours
- **Total damage per simulation:** ~200-300 combat power units

#### 4. `results.json` (~2,020 lines)
Final state of simulation:
```json
{
  "end_time": 1747754689.332,
  "entities": {
    "entity-uuid": {
      "combat_power": 7.7,      // Remaining combat power
      "name": "entity-uuid",
      "position": {             // Final position
        "x": -116.612,
        "y": 34.491
      }
    }
  }
}
```
- **Entities at end:** ~67 entities (compared to 163 initial)
- **Interpretation:** ~96 entities eliminated/destroyed during simulation

#### 5. `external_id_to_internal_id.json` & `internal_id_to_extneral_id.json`
Mapping between human-readable UUIDs and internal numeric entity IDs:
```json
{
  "entity-uuid": 17179869499,   // Maps UUID to internal ID
  ...
}
```

## Web Interaction Data (`web/`)

### Overview
- **Total Sessions:** 3 user sessions (TS-ARL3538, TS-ARL3542, TS-ARL3543)
- **Format:** JSONL (JSON Lines) - one JSON object per line
- **Total Events:** Over 1 million user interaction events

### Session Details

| Session ID | Duration | Total Events | COAs Created | Event Types |
|------------|----------|--------------|--------------|-------------|
| TS-ARL3538 | 64.0 min | 409,811 | 4 | Mouse, keyboard, UI interactions |
| TS-ARL3542 | 154.3 min | 319,725 | 4 | Mouse, keyboard, UI interactions |
| TS-ARL3543 | 57.3 min | 316,879 | 1 | Mouse, keyboard, UI interactions |

**Total COAs Developed:** 9 courses of action across all sessions

### Event Structure
Each line in the JSONL file represents a single user interaction:
```json
{
  "currentPage": "http://10.0.0.140:3000/TS-ARL3538",
  "coa_id": "17281576-9056-8220-1406-701293056666",  // null or COA UUID
  "actionType": "pointermove",                        // Event type
  "data": { ... },                                    // Event-specific data
  "timeStamp": 1747753778205                          // Unix timestamp (ms)
}
```

### Common Action Types
1. **pointermove** (~70% of events): Mouse cursor movement
2. **pointerover/pointerout** (~20%): Mouse hover events
3. **click**: Mouse clicks
4. **keydown/keyup**: Keyboard input
5. **selectionchange**: Text selection
6. **animationstart/animationend**: UI animations

## Analysis Potential

### Performance Metrics

Based on the data structure, the following analyses are possible:

#### 1. **Casualties Over Time**
- Track combat_power degradation through damage_events
- Calculate entity elimination rate (163 start → ~67 end)
- Compare casualties between Red and Blue forces
- Analyze attrition rates across different unit types

#### 2. **Combat Intensity**
- Events per time period
- Geographic hotspots of combat activity
- Peak engagement times
- Damage rates over time

#### 3. **Force Effectiveness**
- Kill ratios (damage dealt vs. received)
- Survival rates by unit type
- Combat power preservation
- Tactical advantage metrics

#### 4. **COA Development Patterns**
- Time spent developing each COA
- Number of iterations per session
- User interaction patterns
- Interface usage statistics

#### 5. **Simulation Comparisons**
- Compare outcomes across 85 simulation runs
- Identify successful vs. unsuccessful strategies
- Statistical variation in results
- Scenario sensitivity analysis

### Key Performance Indicators (KPIs)

1. **Total Casualties:** Sum of combat_power lost across all entities
2. **Casualty Rate:** Casualties per hour of simulation time
3. **Force Ratio Evolution:** Blue vs. Red combat power over time
4. **Engagement Density:** Combat events per geographic area
5. **Mission Duration:** Time until combat cessation
6. **Survival Rate:** Percentage of units surviving to simulation end
7. **COA Quality:** Correlation between user session duration and simulation outcomes

## Data Files

### MATLAB Helper Function

The `readData.m` file provides a MATLAB function to read and structure the data:

```matlab
[sim_data, web_data] = readData(fpath)
```

**Parameters:**
- `fpath`: Base path to data directory (e.g., '/home/kevin/Documents/sc3/data/')

**Returns:**
- `sim_data`: Struct array containing all simulation runs
- `web_data`: Struct array containing all web session data

**Structure:**
- Each element has a `name` field (simulation ID or session ID)
- Simulation fields: `combat_events`, `config`, `damage_events`, `external_id_to_internal_id`, `internal_id_to_extneral_id`, `results`
- Web fields: `data` (array of interaction events)

## Technical Details

### Entity ID Format
- **External IDs:** UUID format (e.g., "cd0d9891-84ff-47de-a407-c32671401482")
- **Internal IDs:** Large integers (e.g., 17179869884)
- Use mapping files to convert between formats

### Coordinate System
- **Geographic:** Longitude/Latitude (WGS84)
- **Bounding Box:** 
  - Longitude: -117.575 to -116.331
  - Latitude: 34.248 to 35.068
  - Approximate area: 45km × 90km

### Time Representation
- **Simulation time:** Seconds from simulation start (0 to 345,600)
- **Wall clock time:** Unix timestamps in milliseconds
- **Conversion:** Simulation uses 60-second timesteps

## Recommended Analysis Tools

- **Python:** pandas, matplotlib, seaborn, geopandas (for spatial analysis)
- **MATLAB:** Built-in plotting, statistics toolbox
- **R:** ggplot2, dplyr, tidyr
- **GIS Tools:** QGIS for geographic visualization

## Questions for Analysis

As per the briefing requirement, focus on:

1. **Performance visualization:** How do forces perform over time?
2. **Casualty trends:** When and where do casualties occur?
3. **Force effectiveness:** Which COAs result in better outcomes?
4. **Time-based metrics:** How does combat intensity evolve?
5. **Comparative analysis:** What patterns emerge across multiple simulation runs?

## Contact

For questions about this dataset, contact Stephen (original data provider).

---

**Last Updated:** October 22, 2025  
**Data Collection:** May 2021  
**System:** SC3 (Scalable Command and Control for Coalition operations)  
**Location:** Army Research Laboratory (ARL)