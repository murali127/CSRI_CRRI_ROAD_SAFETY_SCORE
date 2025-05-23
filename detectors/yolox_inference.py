import torch
import cv2
import numpy as np
import logging
logger = logging.getLogger(__name__)
from yolox.utils import postprocess
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import get_model_info
from yolox.data.datasets import COCO_CLASSES
from typing import List, Tuple, Optional
from utils.config import CLASS_IDS, DETECTION_THRESHOLD

class YOLOXDetector:
    def __init__(self, model_path: str = "yolox_s.pth", device: str = "cuda"):
        self.device = device
        self.model = self._load_model(model_path)
        self.class_names = COCO_CLASSES
        self.cls_id_to_name = {i: name for i, name in enumerate(self.class_names)}
        
# yolox_inference.py
    def _load_model(self, model_path: str):
        """Load YOLOX model"""
        exp = get_exp(None, "yolox-s")
        model = exp.get_model()
        logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    
    # Modified CUDA handling
        if self.device == "cuda" and torch.cuda.is_available():
            model.cuda()
        elif self.device == "cuda":
            logger.warning("CUDA not available, using CPU instead")
            self.device = "cpu"
    
        model.eval()
    
        logger.info("Loading checkpoint from {}".format(model_path))
        ckpt = torch.load(model_path, map_location=self.device)
        model.load_state_dict(ckpt["model"])
        logger.info("Loaded checkpoint successfully.")
    
        return model
    def detect(self, img: np.ndarray) -> List[Tuple]:
        """Detect objects in image"""
        img_info = {"id": 0}
        img_info["file_name"] = None
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        
        # Preprocess image
        img, ratio = preproc(img, (640, 640))
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float()
        
        if self.device == "cuda":
            img = img.cuda()
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(
                outputs, 
                len(self.class_names), 
                conf_thre=DETECTION_THRESHOLD,
                nms_thre=0.45
            )
        
        # Process detections
        detections = []
        if outputs[0] is not None:
            outputs = outputs[0].cpu().numpy()
            bboxes = outputs[:, 0:4]
            bboxes /= img_info["ratio"]
            cls_ids = outputs[:, 6]
            scores = outputs[:, 4] * outputs[:, 5]
            
            for i in range(len(bboxes)):
                bbox = bboxes[i]
                cls_id = int(cls_ids[i])
                score = scores[i]
                
                # Filter by class
                cls_name = self.cls_id_to_name[cls_id]
                for category, ids in CLASS_IDS.items():
                    if cls_id in ids:
                        detections.append((
                            bbox[0], bbox[1], bbox[2], bbox[3], 
                            score, cls_id, category
                        ))
                        break
        
        return detections