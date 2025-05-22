import numpy as np
import logging
logger = logging.getLogger(__name__)
from typing import List, Tuple
from utils.config import TRACKING_THRESHOLD

class BYTETracker:
    def __init__(self, track_thresh: float = 0.5, match_thresh: float = 0.8):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.tracked_objects = {}
        self.next_id = 1
    
    def update(self, detections: List[Tuple]) -> List[Tuple]:
        """Update tracker with new detections"""
        active_tracks = []
        
        # First pass: update existing tracks with high confidence detections
        for det in detections:
            x1, y1, x2, y2, conf, cls_id, cls_name = det
            if conf < self.track_thresh:
                continue
                
            matched = False
            for obj_id, obj in self.tracked_objects.items():
                if obj['cls_name'] != cls_name:
                    continue
                    
                # Simple IoU matching (can be replaced with more sophisticated metric)
                iou = self._calculate_iou(det, obj['last_detection'])
                if iou > self.match_thresh:
                    obj['last_detection'] = det
                    obj['misses'] = 0
                    active_tracks.append((*det[:4], obj_id, cls_name))
                    matched = True
                    break
            
            # If no match found, create new track
            if not matched:
                self.tracked_objects[self.next_id] = {
                    'last_detection': det,
                    'cls_name': cls_name,
                    'misses': 0
                }
                active_tracks.append((*det[:4], self.next_id, cls_name))
                self.next_id += 1
        
        # Second pass: try to match low confidence detections
        for det in detections:
            x1, y1, x2, y2, conf, cls_id, cls_name = det
            if conf >= self.track_thresh:
                continue
                
            best_match = None
            best_iou = 0
            for obj_id, obj in self.tracked_objects.items():
                if obj['cls_name'] != cls_name:
                    continue
                    
                iou = self._calculate_iou(det, obj['last_detection'])
                if iou > best_iou and iou > self.match_thresh:
                    best_iou = iou
                    best_match = obj_id
            
            if best_match is not None:
                self.tracked_objects[best_match]['last_detection'] = det
                active_tracks.append((*det[:4], best_match, cls_name))
        
        # Remove lost tracks
        lost_ids = []
        for obj_id, obj in self.tracked_objects.items():
            if obj_id not in [t[4] for t in active_tracks]:
                obj['misses'] += 1
                if obj['misses'] > 5:  # Remove after 5 consecutive misses
                    lost_ids.append(obj_id)
        
        for obj_id in lost_ids:
            del self.tracked_objects[obj_id]
        
        return active_tracks
    
    def _calculate_iou(self, det1, det2):
        """Calculate Intersection over Union between two detections"""
        box1 = det1[:4]
        box2 = det2[:4]
        
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0