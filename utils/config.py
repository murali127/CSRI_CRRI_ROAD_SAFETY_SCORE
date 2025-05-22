import numpy as np

# Color mappings for different classes
COLORS = {
    'vehicle': (0, 255, 0),      # Green
    'pedestrian': (255, 0, 0),    # Red
    'animal': (0, 0, 255),        # Blue
    'traffic_sign': (255, 255, 0) # Yellow
}

# Class IDs mapping (based on COCO dataset)
CLASS_IDS = {
    'vehicle': [2, 3, 5, 7],      # car, motorcycle, bus, truck
    'pedestrian': [0, 1],          # person
    'animal': [15, 16, 17, 18, 19, 20, 21, 22, 23]  # various animals
}

# Detection thresholds
DETECTION_THRESHOLD = 0.5
TRACKING_THRESHOLD = 0.3

# ROI (Region of Interest) mask - can be customized
def create_roi_mask(width, height):
    """Create a mask for region of interest (entire frame by default)"""
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:, :] = 255  # Entire frame is ROI
    return mask