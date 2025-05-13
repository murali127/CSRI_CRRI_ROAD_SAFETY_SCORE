import numpy as np
from collections import defaultdict

class SafetyScorer:
    def __init__(self):
        # Risk weights for different events (higher = more risky)
        self.risk_factors = {
            # Person-related risks
            'pedestrian_proximity': {'threshold': 2.0, 'score': 3},
            'cyclist_proximity': {'threshold': 2.0, 'score': 3},
            
            # Road hazards
            'pothole': {'score': 2},
            'crack': {'score': 1},
            'speed_bump': {'score': 1},
            
            # Traffic conditions
            'vehicle_density': {'threshold': 5, 'score': 2},
            'lane_departure': {'score': 3},
            
            # Traffic control
            'traffic_light_red': {'score': 2},
            'stop_sign_ignored': {'score': 3},
            
            # Special cases
            'accident': {'score': 10},  # Maximum risk
            'roadwork': {'score': 4},
            'pedestrian_crossing': {'threshold': 2.0, 'score': 3}
        }
        
        # Time-based risk tracking
        self.historical_events = defaultdict(list)
        self.max_history = 30  # frames
        
        # Add accident to high risk events
        self.high_risk_event_weight = 10  # Highest possible score impact
        
    def calculate_frame_score(self, tracked_objects, frame_width, frame_height, frame_number):
        """Calculate safety score for current frame based on detected objects"""
        score = 0
        frame_events = []
        
        # Add accident detection logic
        events = []
        for obj in tracked_objects:
            # Check for accident detection
            if obj['class_name'] == 'accident':
                events.append(f"CRITICAL: Accident detected at frame {frame_number}")
                # Immediately assign maximum risk score for accidents
                return self.high_risk_event_weight, events
        
        # Classify objects by type
        vehicles = []
        pedestrians = []
        cyclists = []
        potholes = []
        traffic_lights = []
        stop_signs = []
        road_signs = []
        accidents = []
        roadworks = []
        pedestrian_crossings = []
        
        # Filter only new detections for scoring
        for obj in tracked_objects:
            if not obj['is_new']:
                continue
                
            if obj['class_name'] in ['car', 'bus', 'truck', 'motorcycle']:
                vehicles.append(obj)
            elif obj['class_name'] == 'person':
                pedestrians.append(obj)
            elif obj['class_name'] == 'bicycle':
                cyclists.append(obj)
            elif obj['class_name'] == 'pothole':
                potholes.append(obj)
            elif obj['class_name'] == 'traffic light':
                traffic_lights.append(obj)
            elif obj['class_name'] == 'stop sign':
                stop_signs.append(obj)
            elif obj['class_name'] == 'road_sign':
                road_signs.append(obj)
            elif obj['class_name'] == 'accident':
                accidents.append(obj)
            elif obj['class_name'] == 'roadwork':
                roadworks.append(obj)
            elif obj['class_name'] == 'pedestrian_crossing':
                pedestrian_crossings.append(obj)

        # ========== RISK SCORING LOGIC ==========
        
        # 1. Accident detection (highest priority)
        if accidents:
            score += self.risk_factors['accident']['score']
            frame_events.append(f"ACCIDENT DETECTED!")
        
        # 2. Pedestrian proximity checks
        for ped in pedestrians:
            for veh in vehicles:
                distance = self._calculate_bbox_distance(
                    ped['bbox'], veh['bbox'], frame_width, frame_height
                )
                if distance < self.risk_factors['pedestrian_proximity']['threshold']:
                    score += self.risk_factors['pedestrian_proximity']['score']
                    frame_events.append(
                        f"Pedestrian #{ped['track_id']} near Vehicle #{veh['track_id']} ({distance:.1f}m)"
                    )
        
        # 3. Cyclist proximity checks
        for cyc in cyclists:
            for veh in vehicles:
                distance = self._calculate_bbox_distance(
                    cyc['bbox'], veh['bbox'], frame_width, frame_height
                )
                if distance < self.risk_factors['cyclist_proximity']['threshold']:
                    score += self.risk_factors['cyclist_proximity']['score']
                    frame_events.append(
                        f"Cyclist #{cyc['track_id']} near Vehicle #{veh['track_id']} ({distance:.1f}m)"
                    )
        
        # 4. Vehicle density
        if len(vehicles) > self.risk_factors['vehicle_density']['threshold']:
            score += self.risk_factors['vehicle_density']['score']
            frame_events.append(f"High vehicle density: {len(vehicles)} vehicles")

        # 5. Road hazards
        if potholes:
            score += self.risk_factors['pothole']['score'] * len(potholes)
            frame_events.extend([f"Pothole #{p['track_id']}" for p in potholes])
            
        # 6. Roadwork zones
        if roadworks:
            score += self.risk_factors['roadwork']['score']
            frame_events.append(f"Roadwork zone detected")
            
        # 7. Pedestrian crossing zone
        for crossing in pedestrian_crossings:
            # Check if vehicles are near pedestrian crossings
            for veh in vehicles:
                distance = self._calculate_bbox_distance(
                    crossing['bbox'], veh['bbox'], frame_width, frame_height
                )
                if distance < self.risk_factors['pedestrian_crossing']['threshold']:
                    score += self.risk_factors['pedestrian_crossing']['score']
                    frame_events.append(f"Vehicle near pedestrian crossing")
                    break

        # 8. Traffic signals/signs
        # Note: In a real implementation, this would need light state detection
        # and vehicle movement analysis to determine if signals are being obeyed
        if traffic_lights and vehicles:
            # Just mark presence for now
            frame_events.append(f"Traffic light with {len(vehicles)} vehicles")
            
        if stop_signs and vehicles:
            # Just mark presence for now
            frame_events.append(f"Stop sign with {len(vehicles)} vehicles")
            
        # Update historical events (for persistent tracking)
        self._update_history(frame_number, frame_events)
        
        # Cap score at 10
        return min(10, score), frame_events
        
    def _calculate_bbox_distance(self, bbox1, bbox2, frame_width, frame_height):
        """Calculate normalized distance between bounding boxes (0-1 scale)"""
        # Get centers of bounding boxes
        cx1 = (bbox1[0] + bbox1[2]) / 2
        cy1 = (bbox1[1] + bbox1[3]) / 2
        cx2 = (bbox2[0] + bbox2[2]) / 2
        cy2 = (bbox2[1] + bbox2[3]) / 2
        
        # Calculate Euclidean distance normalized by frame dimensions
        return np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2) / np.sqrt(frame_width**2 + frame_height**2)
        
    def _update_history(self, frame_number, events):
        """Update history of events for temporal analysis"""
        self.historical_events[frame_number] = events
        
        # Clean up old events
        keys_to_remove = []
        for key in self.historical_events:
            if frame_number - key > self.max_history:
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            del self.historical_events[key]