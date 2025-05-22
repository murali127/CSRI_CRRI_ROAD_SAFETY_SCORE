from typing import List, Tuple, Dict
import pandas as pd

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

def generate_report(frame_stats: List[Dict]) -> pd.DataFrame:
    """
    Generate a report DataFrame from frame statistics.
    """
    df = pd.DataFrame(frame_stats)
    df['safety_score'] = df.apply(
        lambda x: compute_safety_score(x['vehicle'], x['pedestrian'], x['animal']), 
        axis=1
    )
    return df