from deep_sort_realtime.deepsort_tracker import DeepSort
from detection.detector import RoadSafetyDetector

class RoadSafetyTracker:
    def __init__(self):
        self.detector = RoadSafetyDetector()
        self.tracker = DeepSort(max_age=30)
    
    def track(self, frame):
        detections = self.detector.detect(frame)
        
        # Convert detections to DeepSort format
        ds_detections = []
        for det in detections:
            bbox = det['bbox']
            ds_detections.append(([bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]], 
                                 det['confidence'], 
                                 det['class_name']))
        
        # Update tracker
        tracks = self.tracker.update_tracks(ds_detections, frame=frame)
        
        # Format results
        tracked_objects = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            bbox = track.to_ltrb()
            tracked_objects.append({
                'track_id': track.track_id,
                'class_name': track.det_class,
                'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            })
        
        return tracked_objects