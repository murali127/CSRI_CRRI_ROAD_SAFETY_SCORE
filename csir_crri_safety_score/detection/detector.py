from ultralytics import YOLO
import cv2
import numpy as np

class RoadSafetyDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        self.classes_of_interest = {
            0: 'person',  # pedestrian
            1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck',  # vehicles
            9: 'traffic light',  # traffic signals
        }
        
    def detect(self, frame):
        """Detect objects in a single frame"""
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                if class_id in self.classes_of_interest:
                    detections.append({
                        'class_id': class_id,
                        'class_name': self.classes_of_interest[class_id],
                        'confidence': float(box.conf),
                        'bbox': box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    })
        
        return detections