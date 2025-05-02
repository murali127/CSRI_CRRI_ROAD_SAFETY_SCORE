from deep_sort_realtime.deepsort_tracker import DeepSort
from detection.detector import RoadSafetyDetector
import numpy as np

class RoadSafetyTracker:
    def __init__(self):
        self.detector = RoadSafetyDetector()
        # Enhanced tracker configuration
        self.tracker = DeepSort(
            max_age=30,  # Keep tracks longer
            n_init=3,    # Require more detections before confirming track
            nn_budget=100,
            max_cosine_distance=0.2,  # Stricter matching
            override_track_class=None
        )
        self.track_history = {}  # To maintain counts across frames
    
    def track(self, frame):
        detections = self.detector.detect(frame)
        
        # Convert detections to DeepSort format
        bboxes = []
        confidences = []
        class_names = []
        
        for det in detections:
            bbox = det['bbox']
            bboxes.append([bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]])  # Convert to [x,y,w,h]
            confidences.append(det['confidence'])
            class_names.append(det['class_name'])
        
        # Update tracker
        tracks = self.tracker.update_tracks(
            list(zip(bboxes, confidences, class_names)),
            frame=frame
        )
        
        # Update track history and maintain consistent counts
        current_ids = set()
        tracked_objects = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            class_name = track.det_class
            bbox = track.to_ltrb()  # [x1,y1,x2,y2]
            
            # Update count logic
            if track_id not in self.track_history:
                self.track_history[track_id] = {
                    'class': class_name,
                    'first_frame': True,
                    'counted': False
                }
            
            current_ids.add(track_id)
            tracked_objects.append({
                'track_id': track_id,
                'class_name': class_name,
                'bbox': [int(x) for x in bbox],
                'count_this': self.track_history[track_id]['first_frame']
            })
            
            self.track_history[track_id]['first_frame'] = False
        
        # Clean up old tracks
        for track_id in list(self.track_history.keys()):
            if track_id not in current_ids:
                del self.track_history[track_id]
        
        return tracked_objects