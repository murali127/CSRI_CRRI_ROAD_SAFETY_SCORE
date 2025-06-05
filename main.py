import cv2
import time
import logging
import pandas as pd  # Add this import at the top
from typing import Optional, Dict, List
from pathlib import Path
from detectors.yolox_inference import YOLOXDetector
from detectors.pothole_detector import PotholeDetector
from trackers.bytetrack import BYTETracker
from scoring.safety_score import compute_safety_score, analyze_frame_detections, generate_segment_report
from utils.video_utils import read_video, get_video_properties, initialize_video_writer, draw_objects, draw_safety_score
from utils.config import COLORS, POTHOLEDETECTION, create_roi_mask

logger = logging.getLogger(__name__)

class RoadSafetyScorer:
    def __init__(self, model_path: str = "yolox_s.pth", device: str = "cuda", segment_size: float = 5.0):
        self.detector = YOLOXDetector(model_path, device)
        self.tracker = BYTETracker()
        
        # Initialize pothole detector
        try:
            self.pothole_detector = PotholeDetector(
                model_path=POTHOLEDETECTION['MODEL_PATH'],
                input_size=POTHOLEDETECTION['INPUT_SIZE'],
                threshold=POTHOLEDETECTION['THRESHOLD']
            )
        except Exception as e:
            logger.error(f"Pothole detector initialization failed: {str(e)}")
            self.pothole_detector = None
        
        self.frame_stats = []
        self.segment_size = segment_size

    def process_video(self, input_path: str, output_path: str) -> dict:
        """Process video with improved score tracking"""
        result = {
            "output_video": output_path,
            "report": pd.DataFrame(),
            "average_score": 0.0,
            "processing_time": 0.0,
            "segment_size": self.segment_size,
            "frame_stats": [],
            "error": None
        }

        try:
            cap = read_video(input_path)
            if not cap.isOpened():
                result["error"] = "Could not open video file"
                return result

            width, height, frame_count, fps = get_video_properties(cap)
            out_writer = initialize_video_writer(output_path, width, height, fps)
            
            start_time = time.time()
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                detections = self.detector.detect(frame)
                tracks = self.tracker.update(detections)

                # Pothole detection
                pothole_status = False
                if self.pothole_detector:
                    try:
                        pothole_label, pothole_prob = self.pothole_detector.predict(frame)
                        pothole_status = pothole_label is not None and pothole_label == 1
                    except Exception as e:
                        print(f"Pothole detection error: {str(e)}")

                # Get counts and score
                counts = analyze_frame_detections(tracks, pothole_status)
                score = compute_safety_score(
                    counts['vehicle'],
                    counts['pedestrian'],
                    counts['animal'],
                    counts['pothole']
                )

                # Store frame stats with timestamp
                self.frame_stats.append({
                    "frame": frame_idx,
                    "vehicle": int(counts['vehicle']),
                    "pedestrian": int(counts['pedestrian']),
                    "animal": int(counts['animal']),
                    "pothole": int(pothole_status),
                    "score": float(score),
                    "timestamp": frame_idx / fps if fps > 0 else frame_idx / 30  # Fallback to 30fps
                })

                # Visualization
                frame = draw_objects(frame, detections, tracks)
                frame = draw_safety_score(frame, score)
                if pothole_status:
                    cv2.putText(frame, "POTHOLE DETECTED", (width//2, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['pothole'], 2)
                out_writer.write(frame)
                frame_idx += 1

            # Generate final report
            if self.frame_stats:
                result["report"] = generate_segment_report(self.frame_stats, fps, self.segment_size)
                if not result["report"].empty:
                    result["average_score"] = result["report"]["score"].mean()
                result["frame_stats"] = self.frame_stats

            result["processing_time"] = time.time() - start_time
            return result

        except Exception as e:
            result["error"] = str(e)
            return result