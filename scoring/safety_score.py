from typing import List, Tuple, Dict
import pandas as pd
import numpy as np

def compute_safety_score(vehicle_count: int, pedestrian_count: int, animal_count: int) -> int:
    """
    Compute road safety score based on object counts.
    Score ranges from 0 (safest) to 10 (most dangerous).
    """
    score = 0
    score += min(vehicle_count // 5, 4)  # Max 4 points for vehicles
    score += min(pedestrian_count // 2, 3)  # Max 3 points for pedestrians
    score += min(animal_count, 3)  # Max 3 points for animals
    return min(score, 10)  # Cap at 10

def analyze_frame_detections(tracks: List[Tuple]) -> Dict[str, int]:
    """
    Count different object types in current frame.
    Returns dictionary with counts for each category.
    """
    counts = {
        'vehicle': 0,
        'pedestrian': 0,
        'animal': 0
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
    
    # Calculate segment numbers
    df['segment'] = (df['timestamp'] // segment_size).astype(int)
    
    # Group by segment and get max counts
    segment_df = df.groupby('segment').agg({
        'vehicle': 'max',
        'pedestrian': 'max',
        'animal': 'max',
        'timestamp': 'first'
    }).reset_index()
    
    # Calculate safety score for each segment
    segment_df['score'] = segment_df.apply(
        lambda x: compute_safety_score(x['vehicle'], x['pedestrian'], x['animal']), 
        axis=1
    )
    
    return segment_df