import numpy as np
from collections import defaultdict
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    print("ERROR: deep_sort_realtime not installed. Install with: pip install deep-sort-realtime")

# Import detector with error handling
try:
    from detection.detector import RoadSafetyDetector
except ImportError as e:
    print(f"ERROR: Could not import detector: {e}")
    print("Make sure detector.py is in the correct location")

class RoadSafetyTracker:
    def __init__(self):
        """Initialize tracker with detector and DeepSORT"""
        try:
            self.detector = RoadSafetyDetector()
            
            # Initialize DeepSORT tracker with good parameters for road scenes
            self.tracker = DeepSort(
                max_age=30,  # How many frames a track can be missing before dying
                n_init=3,    # How many detections needed to confirm track
                max_cosine_distance=0.2,  # Threshold for matching identity
                nn_budget=100,  # Max size for appearance buffer
                override_track_class=None
            )
            
            # Track object appearances per segment
            self.segment_objects = defaultdict(set)  # {segment_id: set(track_ids)}
            self.current_segment = 1
            
            # Store velocities and directions
            self.track_history = {}  # {track_id: {'positions': [], 'timestamps': []}}
            
            # Initialize successfully
            self.initialized = True
            
        except Exception as e:
            print(f"ERROR initializing tracker: {e}")
            self.initialized = False

    def reset_segment(self):
        """Start a new segment"""
        self.current_segment += 1

    def track(self, frame, segment_transition=False):
        """
        Track objects in frame
        
        Args:
            frame: BGR image
            segment_transition: Flag if this is a new segment
            
        Returns:
            List of tracked objects with IDs and metadata
        """
        if not hasattr(self, 'initialized') or not self.initialized:
            print("ERROR: Tracker not properly initialized")
            return []
            
        if segment_transition:
            self.reset_segment()

        # Detect objects using detector
        try:
            detections = self.detector.detect(frame)
        except Exception as e:
            print(f"ERROR in detection: {e}")
            return []
            
        # Format for DeepSORT (bbox, confidence, class)
        bboxes = []
        confidences = []
        class_names = []

        for det in detections:
            try:
                bbox = det['bbox']
                # DeepSORT expects [x,y,w,h], not [x1,y1,x2,y2]
                bboxes.append([bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]])
                confidences.append(det['confidence'])
                class_names.append(det['class_name'])
            except (KeyError, IndexError) as e:
                print(f"ERROR in detection format: {e}")
                continue

        # Update tracker with detections
        try:
            tracks = self.tracker.update_tracks(
                list(zip(bboxes, confidences, class_names)),
                frame=frame
            )
        except Exception as e:
            print(f"ERROR in tracker update: {e}")
            return []

        # Process tracks and return standardized output
        tracked_objects = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            class_name = track.det_class
            
            # Convert DeepSORT's xyah bbox format to [x1,y1,x2,y2]
            bbox = track.to_ltrb()

            # Check if object is new to this segment
            is_new = track_id not in self.segment_objects[self.current_segment]
            self.segment_objects[self.current_segment].add(track_id)
            
            # Calculate velocity if we have history
            velocity = None
            direction = None
            
            if track_id in self.track_history:
                history = self.track_history[track_id]
                if len(history['positions']) > 1:
                    # Calculate simple velocity vector
                    last_pos = history['positions'][-1]
                    current_pos = [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]
                    
                    # Movement vector
                    dx = current_pos[0] - last_pos[0]
                    dy = current_pos[1] - last_pos[1]
                    
                    # Magnitude (pixels per frame)
                    velocity = np.sqrt(dx*dx + dy*dy)
                    
                    # Direction (approximate cardinal)
                    if abs(dx) > abs(dy):
                        direction = "East" if dx > 0 else "West"
                    else:
                        direction = "South" if dy > 0 else "North"
                        
                    # Update position history
                    history['positions'].append(current_pos)
                    
                    # Keep history limited
                    if len(history['positions']) > 30:
                        history['positions'].pop(0)
            else:
                # Initialize history for new track
                self.track_history[track_id] = {
                    'positions': [[(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]],
                    'timestamps': [0]  # Would use actual frame time in real system
                }
            
            # Add to tracked objects
            tracked_objects.append({
                'track_id': track_id,
                'class_name': class_name,
                'bbox': [int(x) for x in bbox],
                'is_new': is_new,
                'velocity': velocity,
                'direction': direction
            })

        return tracked_objects