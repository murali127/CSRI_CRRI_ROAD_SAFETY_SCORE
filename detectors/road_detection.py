# detectors/road_detection.py
import cv2
import numpy as np
from typing import Tuple, Optional
from utils.config import ROAD_DETECTION

class RoadDetector:
    def __init__(self):
        self.road_width_history = []
        self.stable_road_width = None
    
    def detect_road_edges(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect road edges using combined edge detection and color masking"""
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for road-like colors
        mask = cv2.inRange(hsv, ROAD_DETECTION['ROAD_COLOR_LOWER'], ROAD_DETECTION['ROAD_COLOR_UPPER'])
        
        # Find edges
        edges = cv2.Canny(mask, ROAD_DETECTION['CANNY_THRESH1'], 
                         ROAD_DETECTION['CANNY_THRESH2'])
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, ROAD_DETECTION['HOUGH_THRESH'],
                               minLineLength=ROAD_DETECTION['MIN_LINE_LENGTH'],
                               maxLineGap=ROAD_DETECTION['MAX_LINE_GAP'])
        
        if lines is not None:
            leftmost, rightmost = frame.shape[1], 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) > 20:  # Filter near-horizontal lines
                    x = min(x1, x2)
                    if x < leftmost:
                        leftmost = x
                    x = max(x1, x2)
                    if x > rightmost:
                        rightmost = x
            
            if rightmost - leftmost > frame.shape[1]//2:  # Sanity check
                self.road_width_history.append((leftmost, rightmost))
                if len(self.road_width_history) > 10:
                    self.road_width_history.pop(0)
                return leftmost, rightmost
        
        return None
    
    def get_stable_road_width(self, frame_width: int) -> Tuple[int, int]:
        """Get smoothed road width from history"""
        if not self.road_width_history:
            return 50, frame_width - 50  # Default margins
        
        lefts, rights = zip(*self.road_width_history)
        avg_left = int(sum(lefts)/len(lefts))
        avg_right = int(sum(rights)/len(rights))
        
        # Apply some constraints
        margin = frame_width // 10
        avg_left = max(margin, min(avg_left, frame_width - margin))
        avg_right = max(margin, min(avg_right, frame_width - margin))
        
        return avg_left, avg_right