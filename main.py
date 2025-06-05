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
        """Process video and return analysis results"""
        # Initialize result with all required keys
        result = {
            "output_video": output_path,
            "report": pd.DataFrame(),  # Initialize empty DataFrame
            "average_score": 0.0,      # Default value
            "processing_time": 0.0,
            "segment_size": self.segment_size,
            "frame_stats": [],
            "error": None
        }

        try:
            cap = read_video(input_path)
            if cap is None:
                result["error"] = "Could not open video file"
                return result

            width, height, frame_count, fps = get_video_properties(cap)
            out_writer = initialize_video_writer(output_path, width, height, fps)
            roi_mask = create_roi_mask(width, height)

            frame_idx = 0
            start_time = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
                detections = self.detector.detect(masked_frame)
                tracks = self.tracker.update(detections)

                # Pothole detection
                pothole_status = False
                if self.pothole_detector:
                    try:
                        pothole_label, pothole_prob = self.pothole_detector.predict(frame)
                        pothole_status = pothole_label == 1 if pothole_label is not None else False
                        
                        if pothole_status:
                            cv2.putText(frame, f"Pothole ({pothole_prob:.2f})", 
                                        (width - 300, 40), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                        COLORS['pothole'], 2)
                    except Exception as e:
                        logger.warning(f"Pothole detection failed: {str(e)}")

                # Update stats and draw
                counts = analyze_frame_detections(tracks, pothole_status)
                score = compute_safety_score(
                    counts['vehicle'],
                    counts['pedestrian'],
                    counts['animal'],
                    counts.get('pothole', False)
                )

                self.frame_stats.append({
                    "frame": frame_idx,
                    "vehicle": counts['vehicle'],
                    "pedestrian": counts['pedestrian'],
                    "animal": counts['animal'],
                    "pothole": int(pothole_status),
                    "score": score,
                    "timestamp": frame_idx / fps
                })

                frame = draw_objects(frame, detections, tracks)
                frame = draw_safety_score(frame, score)
                out_writer.write(frame)
                frame_idx += 1

            cap.release()
            out_writer.release()

            # Generate final report
            if self.frame_stats:
                result["report"] = generate_segment_report(self.frame_stats, fps, self.segment_size)
                if not result["report"].empty and 'score' in result["report"]:
                    result["average_score"] = float(result["report"]["score"].mean())
                result["frame_stats"] = self.frame_stats

            result["processing_time"] = time.time() - start_time
            return result

        except Exception as e:
            result["error"] = str(e)
            return result