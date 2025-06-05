import numpy as np

# Detection parameters
# In utils/config.py, make sure you have:
COLORS = {
    'vehicle': (0, 255, 0),      # Green
    'pedestrian': (255, 0, 0),   # Red
    'animal': (0, 0, 255),       # Blue
    'road_box': (255, 255, 0),   # Yellow
    'distance_line': (0, 255, 255), # Cyan
    'pothole': (255, 0, 255)     # Purple
}
CLASS_IDS = {
    'vehicle': [2, 3, 5, 7, 1],  # car, motorcycle, bus, truck, bicycle
    'pedestrian': [0],            # person
    'animal': list(range(15, 24)) # various animals
}


# Add pothole detection parameters
POTHOLEDETECTION = {
    'MODEL_PATH': 'pothole.h5',
    'INPUT_SIZE': 300,
    'THRESHOLD': 0.9
}
# config.py - Add these new parameters
# utils/config.py
# Detection thresholds
DETECTION_THRESHOLD = 0.5
TRACKING_THRESHOLD = 0.3

# Road detection parameters
ROAD_BOX_HEIGHT_RATIO = 0.4  # Height of road box as ratio of frame height
MIN_VEHICLE_DISTANCE = 50    # Minimum safe distance between vehicles (pixels)
# ROI (Region of Interest) mask - can be customized
def create_roi_mask(width, height):
    """Create a mask for region of interest (entire frame by default)"""
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:, :] = 255  # Entire frame is ROI
    return mask