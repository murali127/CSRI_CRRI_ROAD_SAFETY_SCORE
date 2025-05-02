import argparse
import os
import time
import cv2
import pandas as pd
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

def process_video(input_path, output_dir, segment_duration=5):
    """Process video in segments with enhanced scoring and preview"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    tracker = RoadSafetyTracker()
    visualizer = Visualizer()
    
    # Open video file
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Default value
        print(f"{bcolors.WARNING}Warning: Using default FPS: {fps}{bcolors.ENDC}")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Prepare output video
    output_path = os.path.join(output_dir, 'output_annotated.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Segment processing variables
    frames_per_segment = int(fps * segment_duration)
    current_segment = 1
    segment_data = []
    segment_events = defaultdict(int)
    frame_count = 0
    segment_scores = []
    
    print(f"{bcolors.HEADER}\nProcessing video in {segment_duration}-second segments...{bcolors.ENDC}")
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        tracked_objects = tracker.track(frame)
        
        # Segment scoring logic
        if frame_count % frames_per_segment == 0 and frame_count > 0:
            # Calculate segment score (using max event score in segment)
            segment_score = min(10, max(segment_events.values(), default=0))
            segment_scores.append({
                'Segment': current_segment,
                'Start_Frame': frame_count - frames_per_segment,
                'End_Frame': frame_count - 1,
                'Events': ", ".join([f"{k}×{v}" for k,v in segment_events.items()]),
                'Score': segment_score
            })
            
            # Reset for next segment
            current_segment += 1
            segment_events = defaultdict(int)
        
        # Score current frame (returns events and their weights)
        _, events = SafetyScorer().calculate_frame_score(
            tracked_objects, 
            frame_width, 
            frame_height, 
            frame_count
        )
        
        # Aggregate events for segment
        for event in events:
            if "Pedestrian" in event:
                segment_events["Pedestrian"] += 3
            elif "Pothole" in event:
                segment_events["Pothole"] += 2
            elif "Lane" in event:
                segment_events["Lane Departure"] += 3
            elif "Vehicle" in event:
                segment_events["High Density"] += 2
        
        # Visualize with segment info
        frame = visualizer.draw_objects(frame, tracked_objects)
        frame = visualizer.draw_segment_info(
            frame,
            current_segment,
            segment_duration,
            segment_events
        )
        
        # Write to output video
        out.write(frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            print(f"{bcolors.OKBLUE}Processed {frame_count} frames ({elapsed:.2f}s elapsed){bcolors.ENDC}")
    
    # Handle last segment
    if segment_events:
        segment_score = min(10, max(segment_events.values(), default=0))
        segment_scores.append({
            'Segment': current_segment,
            'Start_Frame': frame_count - (frame_count % frames_per_segment),
            'End_Frame': frame_count - 1,
            'Events': ", ".join([f"{k}×{v}" for k,v in segment_events.items()]),
            'Score': segment_score
        })
    
    # Clean up
    cap.release()
    out.release()
    
    # Save reports
    save_segment_report(segment_scores, output_dir)
    save_visualization(segment_scores, output_dir, segment_duration)
    
    print(f"{bcolors.HEADER}\n=== Processing Complete ==={bcolors.ENDC}")
    print(f"{bcolors.OKGREEN}Results saved to {output_dir}{bcolors.ENDC}")
    print(f"{bcolors.OKGREEN}Segment scores saved to {os.path.join(output_dir, 'segment_scores.csv')}{bcolors.ENDC}")

    return segment_scores, os.path.abspath(output_path)

def save_segment_report(segment_scores, output_dir):
    """Save segment scores to CSV"""
    df = pd.DataFrame(segment_scores)
    report_path = os.path.join(output_dir, 'segment_scores.csv')
    df.to_csv(report_path, index=False)

def save_visualization(segment_scores, output_dir, segment_duration):
    """Generate and save visualization of segment scores"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    segments = [f"Seg {x['Segment']}" for x in segment_scores]
    scores = [x['Score'] for x in segment_scores]
    
    bars = plt.bar(segments, scores, color=['red' if s >= 7 else 'orange' if s >=4 else 'green' for s in scores])
    plt.xlabel(f'Segments ({segment_duration} seconds each)')
    plt.ylabel('Safety Score (0-10)')
    plt.title('Road Safety Segment Scores')
    plt.ylim(0, 10)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')
    
    plot_path = os.path.join(output_dir, 'segment_scores.png')
    plt.savefig(plot_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Road Safety Scoring System")
    parser.add_argument('--input', type=str, required=True, help="Path to input video file")
    parser.add_argument('--output', type=str, default='data/outputs', help="Output directory")
    parser.add_argument('--segment', type=int, default=5, help="Segment duration in seconds (default: 5)")
    
    args = parser.parse_args()
    process_video(args.input, args.output, args.segment)