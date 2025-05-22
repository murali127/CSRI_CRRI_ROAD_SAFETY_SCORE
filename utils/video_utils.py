import cv2
import numpy as np
import logging
logger = logging.getLogger(__name__)
from typing import Tuple, Optional
from .config import COLORS

def read_video(video_path: str):
    """Read video file and return video capture object"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error opening video file: {video_path}")
        return None
    return cap

def get_video_properties(cap) -> Tuple[int, int, int, float]:
    """Get video properties (width, height, frame count, fps)"""
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return width, height, frame_count, fps

def initialize_video_writer(output_path: str, width: int, height: int, fps: float):
    """Initialize video writer for output"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def draw_objects(frame, detections, tracks=None):
    """Draw detected objects and tracks on frame"""
    for det in detections:
        x1, y1, x2, y2, conf, cls_id, cls_name = det
        color = COLORS.get(cls_name, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Draw label
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    if tracks:
        for track in tracks:
            x1, y1, x2, y2, track_id, cls_name = track
            color = COLORS.get(cls_name, (255, 255, 255))
            
            # Draw bounding box with track ID
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{cls_name} ID:{track_id}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

def draw_safety_score(frame, score):
    """Draw safety score on frame"""
    score_text = f"Safety Score: {score}/10"
    cv2.putText(frame, score_text, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame