import cv2
import numpy as np
from typing import List, Dict, Any

class Visualizer:
    def __init__(self):
        self.colors = {
            'person': (0, 255, 0),    # green
            'car': (255, 0, 0),       # blue
            'bus': (0, 0, 255),       # red
            'truck': (255, 165, 0),   # orange
            'motorcycle': (255, 255, 0), # yellow
            'bicycle': (0, 255, 255), # cyan
            'pothole': (255, 0, 255)  # magenta
        }
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
    
    def draw_objects(self, frame, tracked_objects: List[Dict[str, Any]]):
        """Draw bounding boxes and labels on frame"""
        for obj in tracked_objects:
            bbox = obj['bbox']
            class_name = obj['class_name']
            track_id = obj.get('track_id', -1)
            
            color = self.colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label
            label = f"{class_name} #{track_id}"
            (label_width, label_height), _ = cv2.getTextSize(
                label, self.font, self.font_scale, self.thickness)
            
            # Label background
            cv2.rectangle(
                frame, 
                (bbox[0], bbox[1] - label_height - 10),
                (bbox[0] + label_width, bbox[1] - 10),
                color, -1
            )
            
            # Label text
            cv2.putText(
                frame, label, 
                (bbox[0], bbox[1] - 10), 
                self.font, self.font_scale, (0, 0, 0), self.thickness
            )
        
        return frame
    
    def draw_segment_info(self, frame, segment_num: int, segment_duration: int, events: Dict[str, int]):
        """Draw segment information on frame"""
        # Segment header
        cv2.putText(
            frame, f"Segment {segment_num} ({segment_duration}s)", 
            (10, 30), self.font, 0.7, (255, 255, 255), self.thickness
        )
        
        # Event summary
        y_offset = 60
        for event, count in events.items():
            text = f"{event}: {count}"
            cv2.putText(
                frame, text, 
                (10, y_offset), 
                self.font, 0.6, (255, 255, 255), self.thickness-1
            )
            y_offset += 25
        
        return frame