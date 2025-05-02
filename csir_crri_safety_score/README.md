# CSIR-CRRI Road Safety Scoring System

This project provides an automated system for analyzing dashcam videos to assess road safety. It detects risk factors such as pedestrian proximity, vehicle density, potholes, and lane departures, and generates a safety score timeline, detailed detection logs, and annotated videos.

## Features

- **Object Detection & Tracking:** Identifies vehicles, pedestrians, and potholes in video frames.
- **Safety Scoring:** Calculates safety scores per frame and for video segments.
- **Event Logging:** Logs detected risk events with timestamps.
- **Streamlit Web App:** User-friendly interface for uploading videos, viewing results, and downloading reports.
- **Downloadable Outputs:** Annotated video, detection log (`detection_log.csv`), and score report (`safety_scores.csv`).

## Project Structure

csir_crri_safety_score/
├── main.py                 # 🔁 Entry point: loads video, runs detection, scoring, outputs results
├── requirements.txt        # 📦 Python dependencies
├── yolov8n.pt              # 🧠 YOLOv8 model weights (nano version, replaceable)

├── app/
│   └── streamlit_app.py    # 🌐 Web UI using Streamlit

├── data/
│   ├── raw/                # 🎥 Uploaded/input videos (MP4, AVI)
│   └── outputs/            # 📤 Processed videos, risk scores, CSVs

├── detection/
│   └── detector.py         # 🔍 Object detection logic (YOLOv8 inference)

├── tracking/
│   └── tracker.py          # 🔄 Object tracking logic (e.g., DeepSORT or ByteTrack)

├── scoring/
│   └── scorer.py           # 📊 Safety index calculator + rule-based event logger

├── utils/
│   ├── video_utils.py      # 🎞️ Frame extraction, FPS, resizing, etc.
│   └── visualizer.py       # 🖼️ Draw boxes, scores, annotations on video frames


## Quick Start

1. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

2. **Run the Streamlit app:**
    ```sh
    streamlit run app/streamlit_app.py
    ```

3. **Upload a dashcam video** via the web interface and click "Analyze Video".

4. **Download results**: Annotated video, detection log, and safety score report.

## Outputs

- **Annotated Video:** Video with bounding boxes, scores, and risk events.
- **detection_log.csv:** Frame-by-frame log of detected risks.
- **safety_scores.csv:** Segment-wise safety scores.

## Customization

- **Detection Model:** Replace `yolov8n.pt` with a different YOLOv8 model if needed.
- **Risk Factors:** Modify [`scoring/scorer.py`](scoring/scorer.py) to adjust risk definitions or scoring logic.

## Requirements

- Python 3.8+
- See [`requirements.txt`](requirements.txt) for all dependencies.

## Acknowledgements

- Developed by CSIR-CRRI.
- Uses [YOLOv8](https://github.com/ultralytics/ultralytics) for object detection.

---
For questions or contributions, please open an issue or pull request.