import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        self.colors = {
            'person': (0, 255, 0),  # green
            'car': (0, 0, 255),     # red
            'bus': (255, 0, 0),     # blue
            'truck': (255, 165, 0), # orange
            'motorcycle': (255, 255, 0), # yellow
            'bicycle': (0, 255, 255),   # cyan
            'traffic light': (255, 0, 255) # magenta
        }
    
    def draw_objects(self, frame, tracked_objects):
        """Draw bounding boxes and labels on frame"""
        for obj in tracked_objects:
            bbox = obj['bbox']
            class_name = obj['class_name']
            track_id = obj.get('track_id', -1)
            
            color = self.colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label background
            label = f"{class_name} {track_id}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (bbox[0], bbox[1] - label_height - 5), 
                          (bbox[0] + label_width, bbox[1] - 5), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (bbox[0], bbox[1] - 7), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return frame
    
    def draw_score(self, frame, score, events):
        """Draw safety score and events on frame"""
        # Draw score bar
        score_width = 200
        score_height = 20
        margin = 10
        
        # Background
        cv2.rectangle(frame, (margin, margin), 
                      (margin + score_width, margin + score_height + 30 + len(events)*15), 
                      (0, 0, 0), -1)
        
        # Score bar
        filled_width = int(score_width * (score / 10))
        cv2.rectangle(frame, (margin, margin), 
                      (margin + filled_width, margin + score_height), 
                      (0, 0, 255) if score > 5 else (0, 255, 0), -1)
        cv2.rectangle(frame, (margin, margin), 
                      (margin + score_width, margin + score_height), 
                      (255, 255, 255), 1)
        
        # Score text
        cv2.putText(frame, f"Safety Score: {score:.1f}/10", 
                    (margin + 5, margin + score_height - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Events text
        for i, event in enumerate(events):
            cv2.putText(frame, f"- {event}", 
                        (margin + 5, margin + score_height + 25 + i*15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame