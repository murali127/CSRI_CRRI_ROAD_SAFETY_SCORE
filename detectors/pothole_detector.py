import cv2
import numpy as np
from typing import Tuple, Optional
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

class PotholeDetector:
    def __init__(self, model_path: str="pothole.h5", input_size: int = 300, threshold: float = 0.9):
        self.input_size = input_size
        self.threshold = threshold
        
        try:
            self.model = tf.keras.models.load_model(model_path, compile=False)
            self.model.compile()  # Recompile with default settings
            logger.info("Pothole model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load pothole model: {str(e)}")
            raise

    def predict(self, frame: np.ndarray) -> Tuple[Optional[int], float]:
        """Predict pothole presence in frame"""
        try:
            if len(frame.shape) == 3:  # Convert to grayscale if color
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            frame = cv2.resize(frame, (self.input_size, self.input_size))
            frame = frame.reshape(1, self.input_size, self.input_size, 1)
            frame = frame.astype('float32') / 255.0
            
            predictions = self.model.predict(frame, verbose=0)
            max_prob = np.max(predictions)
            
            if max_prob >= self.threshold:
                return np.argmax(predictions), max_prob
            return None, max_prob
            
        except Exception as e:
            logger.error(f"Pothole prediction failed: {str(e)}")
            return None, 0.0