import cv2
import os

def extract_frames(video_path, output_dir, frame_interval=1):
    """Extract frames from video at specified interval"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear existing frames in the directory
    for f in os.listdir(output_dir):
        if f.endswith('.jpg'):
            os.remove(os.path.join(output_dir, f))
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    extracted_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            extracted_count += 1
            
        frame_count += 1
    
    cap.release()
    return fps, frame_count, extracted_count

def create_video_from_frames(frame_dir, output_path, fps):
    """Create video from processed frames"""
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    if not frame_files:
        raise ValueError("No frames found in directory")
    
    # Get frame dimensions from first frame
    first_frame = cv2.imread(os.path.join(frame_dir, frame_files[0]))
    height, width, _ = first_frame.shape
    
    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames to video
    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(frame_dir, frame_file))
        out.write(frame)
    
    out.release()