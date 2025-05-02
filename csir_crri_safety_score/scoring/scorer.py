import numpy as np

class SafetyScorer:
    def __init__(self):
        self.risk_factors = {
            'pedestrian_proximity': {'threshold': 2.0, 'score': 3},
            'pothole': {'score': 2},
            'vehicle_density': {'threshold': 5, 'score': 2},
            'lane_departure': {'score': 3},
            'signal_violation': {'score': 2}
        }
        self.frame_scores = []
        self.event_log = []
    
    def calculate_frame_score(self, tracked_objects, frame_width, frame_height, frame_number):
        """Calculate safety score for a single frame with detailed terminal output"""
        score = 0
        frame_events = []
        
        vehicles = [obj for obj in tracked_objects if obj['class_name'] in ['car', 'bus', 'truck', 'motorcycle']]
        pedestrians = [obj for obj in tracked_objects if obj['class_name'] == 'person']
        potholes = [obj for obj in tracked_objects if obj['class_name'] == 'pothole']
        
        # Print basic detection info
        print(f"\nFrame {frame_number} Detections:")
        print(f"- Vehicles: {len(vehicles)}")
        print(f"- Pedestrians: {len(pedestrians)}")
        if potholes:
            print(f"- Potholes: {len(potholes)}")

        # Pedestrian proximity check
        for ped in pedestrians:
            for veh in vehicles:
                distance = self._calculate_bbox_distance(ped['bbox'], veh['bbox'], frame_width, frame_height)
                if distance < self.risk_factors['pedestrian_proximity']['threshold']:
                    score += self.risk_factors['pedestrian_proximity']['score']
                    event_msg = f"! Pedestrian too close to vehicle (distance: {distance:.1f}m)"
                    frame_events.append(event_msg)
                    print(event_msg)
        
        # Vehicle density check
        if len(vehicles) > self.risk_factors['vehicle_density']['threshold']:
            score += self.risk_factors['vehicle_density']['score']
            event_msg = f"! High vehicle density ({len(vehicles)} vehicles)"
            frame_events.append(event_msg)
            print(event_msg)
        
        # Pothole detection
        if potholes:
            score += self.risk_factors['pothole']['score'] * len(potholes)
            event_msg = f"! Pothole detected on road"
            frame_events.append(event_msg)
            print(event_msg)
        
        # Normalize score
        score = min(10, score)
        print(f"=> Frame Safety Score: {score}/10")
        
        self.frame_scores.append(score)
        if frame_events:
            self.event_log.append({'frame': frame_number, 'events': frame_events, 'score': score})
        
        return score, frame_events
    
    def _calculate_bbox_distance(self, bbox1, bbox2, frame_width, frame_height):
        """Approximate distance between two objects (simplified for demo)"""
        # Get centers of bounding boxes
        cx1 = (bbox1[0] + bbox1[2]) / 2
        cy1 = (bbox1[1] + bbox1[3]) / 2
        cx2 = (bbox2[0] + bbox2[2]) / 2
        cy2 = (bbox2[1] + bbox2[3]) / 2
        
        # Calculate normalized distance (0-1)
        distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
        normalized_distance = distance / np.sqrt(frame_width**2 + frame_height**2)
        
        # Convert to approximate meters
        return normalized_distance * 20
    
    def get_segment_score(self, segment_length=5, fps=30):
        """Calculate average score for video segments"""
        frames_per_segment = int(segment_length * fps)
        segment_scores = []
        
        for i in range(0, len(self.frame_scores), frames_per_segment):
            segment = self.frame_scores[i:i+frames_per_segment]
            avg_score = sum(segment) / len(segment) if segment else 0
            segment_scores.append({
                'start_frame': i,
                'end_frame': i + len(segment) - 1,
                'start_time': i / fps,
                'end_time': (i + len(segment) - 1) / fps,
                'score': avg_score
            })
        
        return segment_scores
    
    def print_summary(self):
        """Print summary of detected risks"""
        total_risks = {
            'pedestrian_proximity': 0,
            'vehicle_density': 0,
            'potholes': 0,
            'lane_departure': 0
        }
        
        for entry in self.event_log:
            for event in entry['events']:
                if 'Pedestrian too close' in event:
                    total_risks['pedestrian_proximity'] += 1
                elif 'High vehicle density' in event:
                    total_risks['vehicle_density'] += 1
                elif 'Pothole detected' in event:
                    total_risks['potholes'] += 1
                elif 'Lane departure' in event:
                    total_risks['lane_departure'] += 1
        
        print("\n=== Risk Factor Summary ===")
        print(f"Pedestrian proximity incidents: {total_risks['pedestrian_proximity']}")
        print(f"High vehicle density incidents: {total_risks['vehicle_density']}")
        print(f"Pothole detections: {total_risks['potholes']}")
        print(f"Lane departure incidents: {total_risks['lane_departure']}")
    def save_detection_log(self, output_dir):
        """Save detailed detection log to CSV"""
        import pandas as pd
        import os
        
        log_path = os.path.join(output_dir, "detection_log.csv")
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert event log to dataframe
        log_data = []
        for entry in self.event_log:
            for event in entry['events']:
                log_data.append({
                    'frame_number': entry['frame'],
                    'event': event,
                    'score': entry['score'],
                    'timestamp': entry['frame'] / 30  # Assuming 30 FPS
                })
        
        # Add frames without events
        for i, score in enumerate(self.frame_scores):
            if not any(entry['frame'] == i for entry in self.event_log):
                log_data.append({
                    'frame_number': i,
                    'event': "No risks detected",
                    'score': score,
                    'timestamp': i / 30
                })
        
        pd.DataFrame(log_data).to_csv(log_path, index=False)