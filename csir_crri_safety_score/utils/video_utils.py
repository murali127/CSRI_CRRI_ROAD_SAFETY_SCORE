import cv2
import os
from pathlib import Path

def extract_frames(video_path: str, output_dir: str, frame_interval: int = 1) -> tuple:
    """
    Extract frames from video at specified interval
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        frame_interval: Extract every nth frame
        
    Returns:
        Tuple of (fps, total_frames, extracted_frames)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear existing frames
    for f in output_dir.glob("*.jpg"):
        f.unlink()
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError("Invalid video FPS")
        
    frame_count = 0
    extracted_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            frame_path = output_dir / f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            extracted_count += 1
            
        frame_count += 1
    
    cap.release()
    return fps, frame_count, extracted_count

def create_video_from_frames(
    frame_dir: str, 
    output_path: str, 
    fps: float,
    include_annotations: bool = True
) -> None:
    """
    Create video from processed frames
    
    Args:
        frame_dir: Directory containing frames
        output_path: Output video path
        fps: Frames per second
        include_annotations: Whether to include visualization annotations
    """
    frame_dir = Path(frame_dir)
    frame_files = sorted(list(frame_dir.glob("*.jpg")))
    if not frame_files:
        raise ValueError("No frames found in directory")
    
    first_frame = cv2.imread(str(frame_files[0]))
    height, width = first_frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_file in frame_files:
        frame = cv2.imread(str(frame_file))
        out.write(frame)
    
    out.release()