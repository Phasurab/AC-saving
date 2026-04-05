"""
Room Gateway — Routes each room to the correct model + threshold
=================================================================
Classifies rooms into 3 segments:
  - "regular"        → standard rooms, all sensors functional
  - "suite"          → multi-zone suites (living room + bedroom)
  - "missing_sensor" → rooms with >50% Presence code 3/4 (broken Motion)
"""

import pandas as pd
from typing import Dict, Tuple

# ─── Hardcoded from segment analysis ─────────────────────────────────────────
# These 4 rooms have >50% of their Presence readings as code 3 or 4
MISSING_SENSOR_ROOMS = frozenset([
    "room_1002_bedroom",   # 86.4% code 3/4 (Motion disconnected)
    "room_1005_bedroom",   # 67.8% code 3/4
    "room_1602_bedroom",   # 55.9% code 3/4
    "room_1032_bedroom",   # 50.3% code 3/4
])

# Motion-derived features to zero out for missing sensor rooms
MOTION_FEATURES = [
    "Motion", "motion_binary", "motion_roll_max_6", "motion_cumsum_12",
    "steps_since_motion", "steps_since_motion_causal",
    "motion_streak", "CO2_x_motion", "CO2_rising_while_motion",
    "CO2_falling_no_motion", "motion_but_low_CO2", "high_CO2_no_motion",
    "suite_any_motion",
]


def classify_room(room_area: str, is_suite: int) -> str:
    """Classify a single room into its segment.

    Args:
        room_area: Room identifier (e.g. "room_1002_bedroom")
        is_suite: 1 if room is part of a suite, 0 otherwise

    Returns:
        "missing_sensor", "suite", or "regular"
    """
    if room_area in MISSING_SENSOR_ROOMS:
        return "missing_sensor"
    elif is_suite == 1:
        return "suite"
    else:
        return "regular"


def classify_all_rooms(df: pd.DataFrame) -> Dict[str, str]:
    """Classify all rooms in the dataset.

    Returns:
        Dict mapping room_area → segment name
    """
    room_map = {}
    for room in df["room_area"].unique():
        is_suite = df.loc[df["room_area"] == room, "is_suite"].iloc[0]
        room_map[room] = classify_room(room, int(is_suite))

    counts = {}
    for seg in room_map.values():
        counts[seg] = counts.get(seg, 0) + 1
    print(f"Room classification: {counts}")

    return room_map


def get_segment_rooms(room_map: Dict[str, str], segment: str):
    """Return list of room_areas belonging to a segment."""
    return [r for r, s in room_map.items() if s == segment]


def zero_motion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Zero out motion-derived features for missing-sensor rooms.
    LightGBM uses -999 as the missing sentinel."""
    df = df.copy()
    for col in MOTION_FEATURES:
        if col in df.columns:
            df[col] = -999.0
    return df
