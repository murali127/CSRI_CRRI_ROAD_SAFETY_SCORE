import os
import torch
from ultralytics import YOLO
import cv2
import numpy as np

class RoadSafetyDetector:
    def __init__(self, model_path=None):
        """Initialize the detector with YOLO model
        
        Args:
            model_path: Path to custom model. If None, uses yolov8n.pt
        """
        # Use custom model if provided, otherwise use YOLOv8 nano
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            self.model = YOLO("yolov8n.pt")
            
        # Define class mappings (standard COCO classes + custom)
        self.class_map = {
            0: 'person',        # COCO classes
            1: 'bicycle',
            2: 'car', 
            3: 'motorcycle',
            4: 'airplane',
            5: 'bus', 
            6: 'train',
            7: 'truck',
            8: 'boat',
            9: 'traffic light',
            10: 'fire hydrant',
            11: 'stop sign',
            12: 'parking meter',
            13: 'bench',
            # ... (other COCO classes)
            # Custom classes for road safety
            80: 'pothole',
            81: 'crack',
            82: 'speed_bump',
            83: 'road_sign',
            84: 'accident',
            85: 'traffic_cone',
            86: 'roadwork',
            87: 'pedestrian_crossing',
            # Specific road signs
            90: 'speed_limit',
            91: 'no_entry',
            92: 'yield',
            93: 'no_parking',
            94: 'warning',
            95: 'direction',
            96: 'regulatory'
        }
        
        # Define which classes to detect (filter others)
        self.target_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
            'traffic light', 'stop sign', 'pothole', 'crack', 'speed_bump',
            'road_sign', 'accident', 'traffic_cone', 'roadwork', 'pedestrian_crossing',
            'speed_limit', 'no_entry', 'yield', 'no_parking', 'warning', 'direction', 'regulatory'
        ]

        # Add specialized accident detector (optional)
        try:
            self.accident_model = YOLO("accident_model.pt") if os.path.exists("accident_model.pt") else None
        except Exception as e:
            print(f"Warning: Could not load accident model: {e}")
            self.accident_model = None
            
        # Add specialized road sign detector
        try:
            self.road_sign_model = YOLO("road_signs_model.pt") if os.path.exists("road_signs_model.pt") else None
        except Exception as e:
            print(f"Warning: Could not load road sign model: {e}")
            self.road_sign_model = None

    def detect(self, frame):
        """
        Detect objects in a frame
        
        Args:
            frame: OpenCV BGR image
            
        Returns:
            List of detections with class, confidence, and bbox
        """
        # Detect with YOLO
        results = self.model(frame, verbose=False)
        
        # Process results
        detections = []
        if results and len(results) > 0:
            # Get boxes, confidence scores, and class ids
            result = results[0]  # First image result
            
            # Convert to numpy for easier handling if not already
            if hasattr(result.boxes, 'cpu'):
                boxes = result.boxes.cpu().numpy()
                
                # Extract boxes, scores, and class ids
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    # Map class id to name using our mapping
                    class_name = self.class_map.get(cls_id, 'unknown')
                    
                    # Only include target classes with confidence > 0.35
                    if class_name in self.target_classes and conf > 0.35:
                        detections.append({
                            'class_name': class_name,
                            'confidence': conf,
                            'bbox': [x1, y1, x2, y2]
                        })
        
        # Add specialized accident detection with higher confidence
        if self.accident_model:
            accident_results = self.accident_model(frame, verbose=False)
            if accident_results and len(accident_results) > 0:
                accident_result = accident_results[0]
                if hasattr(accident_result.boxes, 'cpu'):
                    boxes = accident_result.boxes.cpu().numpy()
                    for box in boxes:
                        if box.conf > 0.6:  # Higher threshold for accidents
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            detections.append({
                                'class_name': 'accident',
                                'confidence': float(box.conf),
                                'bbox': [x1, y1, x2, y2]
                            })
        
        # Add specialized road sign detection
        if self.road_sign_model:
            sign_results = self.road_sign_model(frame, verbose=False)
            if sign_results and len(sign_results) > 0:
                sign_result = sign_results[0]
                if hasattr(sign_result.boxes, 'cpu'):
                    boxes = sign_result.boxes.cpu().numpy()
                    for box in boxes:
                        if box.conf > 0.5:  # Threshold for road signs
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cls_id = int(box.cls[0])
                            
                            # Get specific road sign type
                            if cls_id == 0:
                                sign_type = 'speed_limit'
                            elif cls_id == 1:
                                sign_type = 'stop_sign'
                            elif cls_id == 2:
                                sign_type = 'yield'
                            elif cls_id == 3:
                                sign_type = 'no_entry'
                            elif cls_id == 4:
                                sign_type = 'warning'
                            else:
                                sign_type = 'road_sign'  # Generic name if not recognized
                                
                            detections.append({
                                'class_name': sign_type,
                                'confidence': float(box.conf),
                                'bbox': [x1, y1, x2, y2]
                            })
        
        return detections

    def detect_with_nms(self, frame, conf_threshold=0.35, iou_threshold=0.5):
        """
        Detect objects with custom NMS implementation
        Useful when model doesn't handle NMS internally
        """
        # Standard detection
        detections = self.detect(frame)
        
        # Apply custom NMS if needed
        # This is already handled by YOLO internally, but can be customized
        
        return detections