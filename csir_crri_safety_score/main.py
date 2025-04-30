import argparse
import os
import time
import cv2
from tracking.tracker import RoadSafetyTracker
from scoring.scorer import SafetyScorer
from utils.visualizer import Visualizer
from utils.video_utils import extract_frames

class bcolors:
    """Color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def process_video(input_path, output_dir):
    """Process video through full pipeline with detailed terminal output"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create frames directory if it doesn't exist
    frames_dir = os.path.join('data', 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    # Initialize components
    tracker = RoadSafetyTracker()
    scorer = SafetyScorer()
    visualizer = Visualizer()
    
    # Extract frames first
    print(f"{bcolors.HEADER}\nExtracting frames from video...{bcolors.ENDC}")
    fps, total_frames, _ = extract_frames(input_path, frames_dir)
    if fps <= 0:
        fps = 30  # Default value if FPS can't be determined
        print(f"{bcolors.WARNING}Warning: Couldn't determine FPS, using default {fps}{bcolors.ENDC}")
    
    # Prepare output video
    output_path = os.path.join(output_dir, 'output.mp4')
    first_frame = cv2.imread(os.path.join(frames_dir, os.listdir(frames_dir)[0]))
    frame_height, frame_width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    print(f"{bcolors.HEADER}\nProcessing frames...{bcolors.ENDC}")
    frame_count = 0
    start_time = time.time()
    
    # Process each frame in the frames directory
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        
        # Process frame with frame number
        tracked_objects = tracker.track(frame)
        score, events = scorer.calculate_frame_score(tracked_objects, frame_width, frame_height, frame_count)
        
        # Visualize
        frame = visualizer.draw_objects(frame, tracked_objects)
        frame = visualizer.draw_score(frame, score, events)
        
        # Write to output video
        out.write(frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            print(f"{bcolors.OKBLUE}Processed {frame_count}/{total_frames} frames ({elapsed:.2f}s elapsed){bcolors.ENDC}")
    
    # Clean up
    out.release()
    
    # Generate segment scores
    segment_scores = scorer.get_segment_score(fps=fps)
    
    # Save score report
    save_score_report(segment_scores, output_dir)
    
    # Print final summary
    print(f"{bcolors.HEADER}\n=== Processing Complete ==={bcolors.ENDC}")
    print(f"{bcolors.OKGREEN}Results saved to {output_dir}{bcolors.ENDC}")
    print(f"{bcolors.OKGREEN}Average safety score: {sum(scorer.frame_scores)/len(scorer.frame_scores):.1f}/10{bcolors.ENDC}")
    # Save detection logs
    # Add this right before the final print statements in process_video()
    scorer.save_detection_log(output_dir)
    # Print risk factor summary
    print(f"{bcolors.HEADER}\n=== Risk Factor Summary ==={bcolors.ENDC}")
    scorer.print_summary()

def save_score_report(segment_scores, output_dir):
    """Save score report to CSV"""
    import pandas as pd
    
    report_path = os.path.join(output_dir, 'safety_scores.csv')
    df = pd.DataFrame(segment_scores)
    df.to_csv(report_path, index=False)
    print(f"{bcolors.OKGREEN}Score report saved to {report_path}{bcolors.ENDC}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Road Safety Scoring System")
    parser.add_argument('--input', type=str, required=True, help="Path to input video file")
    parser.add_argument('--output', type=str, default='data/outputs', help="Output directory")
    
    args = parser.parse_args()
    process_video(args.input, args.output)