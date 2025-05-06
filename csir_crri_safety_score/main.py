from pathlib import Path
import cv2
import os
import sys
from typing import Dict, List
from tracking.tracker import RoadSafetyTracker
from scoring.scorer import SafetyScorer
from utils.visualizer import Visualizer
from collections import defaultdict

class bcolors:
    """Color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def process_video(input_path: str, output_dir: str, segment_duration: int = 5) -> List[Dict]:
    """Process video in segments with enhanced scoring and preview
    
    Args:
        input_path: Path to input video file
        output_dir: Directory to save outputs
        segment_duration: Duration of each segment in seconds
        
    Returns:
        List of segment scores with metadata
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    tracker = RoadSafetyTracker()
    scorer = SafetyScorer()
    visualizer = Visualizer()
    
    # Open video file
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError("Invalid video FPS")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Prepare output video
    output_path = os.path.join(output_dir, 'output_annotated.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Process frames
    frames_per_segment = int(fps * segment_duration)
    current_segment = 1
    segment_events = defaultdict(int)
    frame_count = 0
    segment_scores = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        tracked_objects = tracker.track(frame)
        frame_events = scorer.calculate_frame_score(
            tracked_objects, frame_width, frame_height, frame_count
        )
        
        # Update segment events
        for event in frame_events:
            segment_events[event] += 1
            
        # Visualize frame
        frame = visualizer.draw_segment_info(
            frame, current_segment, segment_duration, dict(segment_events)
        )
        
        # Handle segment transition
        if frame_count > 0 and frame_count % frames_per_segment == 0:
            segment_score = min(10, max(segment_events.values(), default=0))
            segment_scores.append({
                'Segment': current_segment,
                'Start_Time': (frame_count - frames_per_segment) / fps,
                'End_Time': frame_count / fps,
                'Events': dict(segment_events),
                'Score': segment_score
            })
            current_segment += 1
            segment_events.clear()
            
        out.write(frame)
        frame_count += 1
        
        # Print progress
        if frame_count % 30 == 0:
            print(f"\rProcessing frame {frame_count}", end="")
            
    # Handle last segment
    if segment_events:
        segment_score = min(10, max(segment_events.values(), default=0))
        segment_scores.append({
            'Segment': current_segment,
            'Start_Time': (frame_count - (frame_count % frames_per_segment)) / fps,
            'End_Time': frame_count / fps,
            'Events': dict(segment_events),
            'Score': segment_score
        })
    
    cap.release()
    out.release()
    print(f"\n{bcolors.OKGREEN}Processing complete!{bcolors.ENDC}")
    
    return segment_scores

def save_segment_report(segment_scores: List[Dict], output_dir: str) -> None:
    """Save segment analysis to CSV"""
    import pandas as pd
    df = pd.DataFrame(segment_scores)
    output_path = os.path.join(output_dir, 'segment_report.csv')
    df.to_csv(output_path, index=False)
    print(f"{bcolors.OKBLUE}Saved report to: {output_path}{bcolors.ENDC}")

def save_visualization(segment_scores: List[Dict], output_dir: str, segment_duration: int) -> None:
    """Generate and save visualization plots"""
    import matplotlib.pyplot as plt
    
    scores = [s['Score'] for s in segment_scores]
    segments = [s['Segment'] for s in segment_scores]
    
    plt.figure(figsize=(12, 6))
    plt.plot(segments, scores, marker='o', linewidth=2)
    plt.xlabel(f"Segment Number ({segment_duration}s each)")
    plt.ylabel("Safety Score (0-10)")
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(output_dir, 'safety_visualization.png')
    plt.savefig(output_path)
    print(f"{bcolors.OKBLUE}Saved visualization to: {output_path}{bcolors.ENDC}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"{bcolors.FAIL}Usage: python main.py <video_path> [segment_duration]{bcolors.ENDC}")
        sys.exit(1)
        
    input_path = sys.argv[1]
    segment_duration = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    output_dir = "output"
    
    try:
        segment_scores = process_video(input_path, output_dir, segment_duration)
        save_segment_report(segment_scores, output_dir)
        save_visualization(segment_scores, output_dir, segment_duration)
    except Exception as e:
        print(f"{bcolors.FAIL}Error: {str(e)}{bcolors.ENDC}")
        sys.exit(1)