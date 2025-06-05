import pandas as pd
import numpy as np
from typing import List, Tuple, Dict

def compute_safety_score(vehicle_count: int, pedestrian_count: int, 
                        animal_count: int, pothole_detected: bool = False) -> int:
    """Compute safety score with safe defaults"""
    try:
        vehicle_count = int(vehicle_count) if vehicle_count else 0
        pedestrian_count = int(pedestrian_count) if pedestrian_count else 0
        animal_count = int(animal_count) if animal_count else 0
        pothole_detected = bool(pothole_detected)
        
        score = 0
        score += min(vehicle_count // 5, 4)
        score += min(pedestrian_count // 2, 3)
        score += min(animal_count, 3)
        if pothole_detected:
            score += 2
        return min(max(score, 0), 10)  # Ensure score is between 0-10
    except Exception:
        return 0

def analyze_frame_detections(tracks: List[Tuple], pothole_status: bool = False) -> Dict[str, int]:
    """Count different object types in current frame"""
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
    """Generate segment-based report with robust error handling"""
    try:
        if not frame_stats:
            return pd.DataFrame()
            
        df = pd.DataFrame(frame_stats)
        if df.empty:
            return pd.DataFrame()
        
        # Ensure required columns exist
        required_cols = ['vehicle', 'pedestrian', 'animal', 'pothole', 'frame']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Calculate timestamp if not present
        if 'timestamp' not in df.columns:
            df['timestamp'] = df['frame'] / fps if fps > 0 else 0
        
        # Group by segment
        segment_df = df.groupby((df['timestamp'] // segment_size).astype(int)).agg({
            'vehicle': 'max',
            'pedestrian': 'max',
            'animal': 'max',
            'pothole': 'max',
            'timestamp': 'first'
        }).reset_index().rename(columns={'index': 'segment'})
        
        # Calculate safety score
        if not segment_df.empty:
            segment_df['score'] = segment_df.apply(
                lambda x: compute_safety_score(
                    x['vehicle'],
                    x['pedestrian'],
                    x['animal'],
                    x['pothole']
                ),
                axis=1
            )
        
        return segment_df
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        return pd.DataFrame()