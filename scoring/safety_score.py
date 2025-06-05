import pandas as pd
from typing import List, Tuple, Dict

def compute_safety_score(vehicle_count: int, pedestrian_count: int,
                         animal_count: int, pothole_detected: bool = False) -> int:
    """
    Compute road safety score based on object counts and pothole presence.
    Score ranges from 0 (safest) to 10 (most dangerous).
    """
    score = 0
    score += min(vehicle_count // 5, 4)        # Max 4 points for vehicles
    score += min(pedestrian_count // 2, 3)     # Max 3 points for pedestrians
    score += min(animal_count, 2)              # Max 2 points for animals
    if pothole_detected:
        score += 1                             # 1 point for pothole
    return min(score, 10)                      # Cap at 10


def analyze_frame_detections(tracks: List[Tuple], pothole_status: bool = False) -> Dict[str, int]:
    """
    Count different object types in the current frame.
    Returns dictionary with counts for each category.
    """
    counts = {
        'vehicle': 0,
        'pedestrian': 0,
        'animal': 0,
        'pothole': int(pothole_status)
    }

    for track in tracks:
        _, _, _, _, _, cls_name = track
        if cls_name in counts:
            counts[cls_name] += 1

    return counts


def generate_segment_report(frame_stats: List[Dict], fps: float, segment_size: float = 5.0) -> pd.DataFrame:
    """
    Generate a report DataFrame with segment-based analysis.
    Each segment is 'segment_size' seconds long.
    """
    df = pd.DataFrame(frame_stats)

    # Ensure required columns
    required_cols = ['vehicle', 'pedestrian', 'animal', 'pothole', 'timestamp']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0  # Fill missing with zero

    df['segment'] = (df['timestamp'] // segment_size).astype(int)

    segment_df = df.groupby('segment').agg({
        'vehicle': 'max',
        'pedestrian': 'max',
        'animal': 'max',
        'pothole': 'max',
        'timestamp': 'first'
    }).reset_index()

    segment_df['score'] = segment_df.apply(
        lambda x: compute_safety_score(
            int(x['vehicle']),
            int(x['pedestrian']),
            int(x['animal']),
            bool(x['pothole'])
        ),
        axis=1
    )

    return segment_df
