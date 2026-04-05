# Phase 1 — Data Dictionary

## File: `phase1_dataset.csv`

### Structure

This dataset uses a **wide format** with a multi-level header:

- **Row 1 (Header):** Room identifiers — format: `room_{number}_{zone}` (e.g., `room_1001_bedroom`, `room_1021_living_room`)
- **Row 2 (Sub-header):** Sensor/datapoint type for each column
- **Row 3 onwards:** Timestamped sensor readings at **5-minute intervals**

The first column is the **timestamp** (`datetime` in row 2, formatted as `YYYY-MM-DD HH:MM:SS`).

### Room Naming Convention

| Pattern | Description |
|---------|-------------|
| `room_XXXX_bedroom` | Standard guest room (single-zone) |
| `room_XXXX_living_room` | Living area of a suite (multi-zone rooms like suites have both `bedroom` and `living_room` entries) |

Rooms span **Floors 8–24** (room numbers 8XX through 24XX). Not all room numbers are sequential — some numbers are skipped (e.g., no room 13XX floor).

### Sensor Types (Datapoints)

Each room has a subset of the following sensors. **Not all rooms have all sensors** — this is realistic; some rooms may be missing humidity or CO2 sensors due to installation scope.

| Datapoint | Type | Unit | Description |
|-----------|------|------|-------------|
| `motion` | Integer | — | Motion detection event count within the 5-minute window. `0` = no motion detected. Values ≥ `1` indicate motion events. |
| `presence_state` | Integer | — | **This is your target variable (ground truth).** Occupancy state of the room. See values below. |
| `room_temperature` | Float | °C | Indoor air temperature measured by the room sensor. |
| `relative_humidity` | Float | % | Indoor relative humidity. |
| `co2` | Float | ppm | Indoor CO2 concentration. |

### `presence_state` Values (Target Variable)

| Value | Meaning |
|-------|---------|
| `0` | **Unoccupied** — No guest presence detected |
| `1` | **Occupied** — Guest is present in the room |
| `3` | **CO2 sensor disconnected** — CO2 data unavailable, other sensors still active |
| `4` | **Motion sensor disconnected** — Motion data unavailable, other sensors still active |

For your occupancy detection model, you may choose to simplify this into a binary classification (`1` as "occupied", `0` as "unoccupied"). Values `3` and `4` indicate sensor disconnections — how you handle these is part of the task. Justify your approach.

### Data Characteristics

- **Time range:** ~2 weeks of continuous data
- **Sampling interval:** 5 minutes
- **Total rooms:** 400+ rooms across Floors 8–24
- **Missing data:** Some cells are empty — this reflects real-world sensor gaps (offline sensors, communication dropouts). Handling missing data is part of the task.
- **Data source:** Derived from real AltoTech hotel deployments (anonymized and modified for this assignment).

### Tips

- The wide format requires reshaping — consider melting/unpivoting into a long format (room, timestamp, sensor_type, value) for easier analysis.
- Some rooms have both `bedroom` and `living_room` zones — these are suites. You may want to combine or analyze them separately.
- Pay attention to sensor availability per room — not every room has all five sensor types.
