import cv2
import time
import logging
logger = logging.getLogger(__name__)
from typing import Optional
from pathlib import Path
from detectors.yolox_inference import YOLOXDetector
from trackers.bytetrack import BYTETracker
from scoring.safety_score import compute_safety_score, analyze_frame_detections, generate_report
from utils.video_utils import read_video, get_video_properties, initialize_video_writer, draw_objects, draw_safety_score
from utils.config import create_roi_mask

class RoadSafetyScorer:
    def __init__(self, model_path: str = "yolox_s.pth", device: str = "cuda"):
        self.detector = YOLOXDetector(model_path, device)
        self.tracker = BYTETracker()
        self.frame_stats = []
    
    def process_video(self, input_path: str, output_path: str) -> dict:
        """Process video and generate safety analysis"""
        cap = read_video(input_path)
        if cap is None:
            return {"error": "Could not open video file"}
        
        width, height, frame_count, fps = get_video_properties(cap)
        out_writer = initialize_video_writer(output_path, width, height, fps)
        roi_mask = create_roi_mask(width, height)
        
        logger.info(f"Processing video: {input_path}")
        logger.info(f"Resolution: {width}x{height}, Frames: {frame_count}, FPS: {fps}")
        
        frame_idx = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply ROI mask
            masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
            
            # Detect objects
            detections = self.detector.detect(masked_frame)
            
            # Track objects
            tracks = self.tracker.update(detections)
            
            # Analyze frame
            counts = analyze_frame_detections(tracks)
            score = compute_safety_score(counts['vehicle'], counts['pedestrian'], counts['animal'])
            
            # Store frame stats
            self.frame_stats.append({
                "frame": frame_idx,
                "vehicle": counts['vehicle'],
                "pedestrian": counts['pedestrian'],
                "animal": counts['animal'],
                "score": score,
                "timestamp": frame_idx / fps
            })
            
            # Draw annotations
            annotated_frame = draw_objects(frame, detections, tracks)
            annotated_frame = draw_safety_score(annotated_frame, score)
            
            # Write frame
            out_writer.write(annotated_frame)
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx}/{frame_count} frames")
        
        cap.release()
        out_writer.release()
        
        processing_time = time.time() - start_time
        logger.info(f"Finished processing in {processing_time:.2f} seconds")
        
        # Generate report
        report = generate_report(self.frame_stats)
        avg_score = report['score'].mean()
        
        return {
            "output_video": output_path,
            "report": report,
            "average_score": avg_score,
            "processing_time": processing_time
        }

if __name__ == "__main__":
    scorer = RoadSafetyScorer()
    result = scorer.process_video("input/sample_video.mp4", "output/annotated_output.mp4")
    result['report'].to_csv("output/report.csv", index=False)