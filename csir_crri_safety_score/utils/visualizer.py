import cv2
import numpy as np
from collections import defaultdict

class Visualizer:
    def __init__(self):
        """Initialize visualizer for road safety objects"""
        # Color scheme for different classes
        self.colors = {
            'person': (0, 255, 0),     # Green
            'bicycle': (102, 255, 102), # Light Green
            'car': (255, 0, 0),        # Blue
            'motorcycle': (255, 128, 0), # Light Blue
            'bus': (0, 0, 255),        # Red
            'truck': (255, 0, 255),    # Purple
            'traffic light': (0, 255, 255), # Yellow
            'stop sign': (255, 255, 0), # Cyan
            'pothole': (128, 0, 128),  # Purple
            'crack': (180, 180, 180),  # Gray
            'accident': (0, 0, 255),   # Bright Red
            'roadwork': (255, 165, 0), # Orange
            'pedestrian_crossing': (255, 192, 203), # Pink
            'road_sign': (255, 255, 255) # White
        }
        
        # Default color for unlisted classes
        self.default_color = (200, 200, 200)  # Light Gray
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 1
        
        # Current segment being processed
        self.current_segment = 1
        
        # Risk color mapping (for score visualization)
        self.risk_colors = {
            'low': (0, 255, 0),      # Green
            'medium': (0, 255, 255),  # Yellow
            'high': (0, 0, 255)       # Red
        }

    def draw_objects(self, frame, tracked_objects):
        """
        Draw tracked objects on frame
        
        Args:
            frame: OpenCV image
            tracked_objects: List of tracked objects with metadata
            
        Returns:
            Annotated frame
        """
        # Count objects per class (for statistics)
        class_counts = defaultdict(int)
        accident_detected = False
        for obj in tracked_objects:
            class_counts[obj['class_name']] += 1

        # Draw each object
        for obj in tracked_objects:
            bbox = obj['bbox']
            class_name = obj['class_name']
            
            # Get color (use default if not in map)
            color = self.colors.get(class_name, self.default_color)
            
            # Draw bounding box
            cv2.rectangle(frame, 
                         (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), 
                         color, 2)
            
            # Create label with class name only
            label = f"{class_name}"
            
            # Add 'NEW' flag if this is a new detection
            if obj.get('is_new', False):
                label += " (NEW)"
                
            # Add velocity if available
            if obj.get('velocity') is not None:
                label += f" {obj['velocity']:.1f}px"
                
            # Add direction if available
            if obj.get('direction') is not None:
                label += f" {obj['direction']}"
            
            # Draw background rectangle for text
            (text_width, text_height), _ = cv2.getTextSize(
                label, self.font, self.font_scale, self.font_thickness
            )
            cv2.rectangle(
                frame, 
                (bbox[0], bbox[1] - text_height - 10), 
                (bbox[0] + text_width + 10, bbox[1]), 
                color, 
                -1  # Filled rectangle
            )
            
            # Draw text
            cv2.putText(
                frame, 
                label, 
                (bbox[0] + 5, bbox[1] - 5), 
                self.font, 
                self.font_scale, 
                (0, 0, 0),  # Black text
                self.font_thickness
            )
            
            # Special handling for accidents
            if obj['class_name'] == 'accident':
                accident_detected = True
                # Draw thicker box with warning stripes
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 4)
                
                # Add warning text
                warning_text = "⚠️ ACCIDENT DETECTED ⚠️"
                text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                cv2.putText(frame, warning_text, 
                           (frame.shape[1]//2 - text_size[0]//2, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # Add flashing border if accident detected
        if accident_detected:
            # Create red border
            border_thickness = 20
            frame[:border_thickness, :] = [0, 0, 255]  # Top
            frame[-border_thickness:, :] = [0, 0, 255]  # Bottom
            frame[:, :border_thickness] = [0, 0, 255]  # Left
            frame[:, -border_thickness:] = [0, 0, 255]  # Right
        
        # Draw object counts in top-left corner
        y_offset = 30
        cv2.putText(
            frame, 
            f"Segment {self.current_segment}", 
            (10, y_offset), 
            self.font, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
        # Draw each class count
        for cls, count in class_counts.items():
            y_offset += 25
            color = self.colors.get(cls, self.default_color)
            cv2.putText(
                frame, 
                f"{cls}: {count}", 
                (10, y_offset), 
                self.font, 
                0.6, 
                color, 
                2
            )

        return frame

    def update_segment(self, segment_id):
        """Update current segment number"""
        self.current_segment = segment_id
        
    def draw_segment_info(self, frame, segment_num, segment_duration, events):
        """
        Draw segment information overlay
        
        Args:
            frame: Image to draw on
            segment_num: Current segment number
            segment_duration: Duration in seconds
            events: Dict of detected events with counts
            
        Returns:
            Annotated frame
        """
        # Calculate total risk score (sum of all event weights)
        total_score = sum(events.values())
        risk_level = "Low"
        color = self.risk_colors['low']
        
        if total_score >= 7:
            risk_level = "High"
            color = self.risk_colors['high']
        elif total_score >= 4:
            risk_level = "Medium"
            color = self.risk_colors['medium']
        
        # Draw segment header with risk level
        header_text = f"Segment {segment_num} ({segment_duration}s) - {risk_level} Risk"
        cv2.putText(
            frame, 
            header_text,
            (10, 30), 
            self.font, 
            0.7, 
            color, 
            2
        )
        
        # Draw semi-transparent overlay for risk score
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (10, 40),
            (10 + int(min(10, total_score) * 30), 60),
            color,
            -1
        )
        
        # Apply overlay with transparency
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw risk score text
        cv2.putText(
            frame, 
            f"Score: {min(10, total_score)}",
            (10, 55), 
            self.font, 
            0.6, 
            (255, 255, 255), 
            1
        )
        
        # Draw events list
        y_offset = 80
        for event, count in events.items():
            text = f"{event}: {count}"
            cv2.putText(
                frame, 
                text,
                (10, y_offset), 
                self.font, 
                0.6, 
                (255, 255, 255), 
                2
            )
            y_offset += 25
        
        return frame
        
    def draw_safety_heatmap(self, frame, risk_map):
        """
        Draw safety heatmap overlay
        
        Args:
            frame: Image to draw on
            risk_map: 2D numpy array with risk values
            
        Returns:
            Frame with heatmap overlay
        """
        # This would be implemented for visual risk zones
        # Not currently used in the main system
        pass

    def draw_labels(self, frame, class_name, x, y):
        """
        Draw labels on the frame
        
        Args:
            frame: Image to draw on
            class_name: Name of the class
            x: X coordinate for the label
            y: Y coordinate for the label
            
        Returns:
            Frame with labels
        """
        label = f"{class_name}"  # Only display the class name
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return frame