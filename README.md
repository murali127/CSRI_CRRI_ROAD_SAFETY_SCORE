📘 README.md – CSIR-CRRI Road Safety Scoring System
markdown

# 🚗 CSIR-CRRI Road Safety Scoring System

A smart computer vision system that analyzes dashcam videos and assigns a **road safety score (0–10)** to each road segment based on real-time object detection and event risk factors. Developed for the CSIR-CRRI initiative.

---

## 🎯 Project Objective

To automatically compute a **safety score** using YOLOv8 and tracking algorithms by identifying:
- Pedestrians near the vehicle
- Potholes or road hazards
- Lane departures
- Dense traffic

This system helps assess driving risk for each video segment and can support future applications like navigation safety, traffic auditing, and insurance tech.

---

## 🧱 Tech Stack

| Layer          | Tools Used                           |
|----------------|--------------------------------------|
| Object Detection | YOLOv8 (Ultralytics)               |
| Object Tracking  | DeepSORT / ByteTrack               |
| Video Processing | OpenCV                             |
| Backend Logic    | Python, NumPy, Pandas              |
| Frontend UI      | Streamlit                          |
| Visualization    | Matplotlib / Plotly                |

---

---

### 📁 Folder Structure

```yaml
csir_crri_safety_score/
├── data/
│   ├── raw/               # Input dashcam videos (MP4)
│   ├── frames/            # Temporary frames (in-memory or saved)
│   └── outputs/           # Annotated videos, CSV reports
│
├── detection/
│   └── detector.py        # YOLOv8 detection logic
│
├── tracking/
│   └── tracker.py         # Object tracking with DeepSORT or ByteTrack
│
├── scoring/
│   └── scorer.py          # Segment-wise risk scoring logic (0–10)
│
├── utils/
│   ├── video_utils.py     # Frame extraction, in-memory processing, overlays
│   └── config.py          # Risk event weights and thresholds
│
├── app/
│   └── streamlit_app.py   # Streamlit UI for dashboard and video upload
│
├── .streamlit/
│   └── config.toml        # Streamlit app theme and layout settings
│
├── models/
│   └── yolo/              # YOLOv8 weights (e.g., yolov8n.pt, yolov8s.pt)
│
├── main.py                # Main CLI runner for full pipeline execution
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation and setup guide
```

---



---

## ⚙️ How It Works

1. 📥 Upload an MP4 dashcam video via the dashboard or CLI.
2. 🧠 YOLOv8 detects objects like pedestrians, vehicles, potholes.
3. 🔄 DeepSORT or ByteTrack tracks objects across time.
4. ⏱️ Every 5-second segment is analyzed for risk events.
5. 🧮 A rule-based engine calculates a **0–10 safety score** per segment.
6. 🎥 A processed video and CSV report are generated.
7. 📊 Everything is visualized in a **Streamlit dashboard**.

---

---

## 🚀 Getting Started

### 1. 🔧 Install Dependencies

```bash
git clone https://github.com/yourusername/csir_crri_safety_score.git
cd csir_crri_safety_score
pip install -r requirements.txt
```

---

### 2. 📂 Add Your Dashcam Video

Place your `.mp4` dashcam video inside the `data/raw/` folder.

---

### 3. ▶️ Run from Terminal

```bash
python main.py --input data/raw/your_video.mp4
```

---

### 4. 🌐 Or Launch the Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Sample Output

| Segment | Start | End   | Risk Events              | Score |
| ------- | ----- | ----- | ------------------------ | ----- |
| 1       | 00:00 | 00:05 | Pedestrian ahead         | 3.0   |
| 2       | 00:05 | 00:10 | Lane departure + pothole | 7.0   |
| 3       | 00:10 | 00:15 | Clear road               | 0.0   |

---

### ✅ Final Output Includes:

* 🎥 Processed video with risk overlays
* 📄 CSV file of segment scores
* 📊 Interactive score chart in Streamlit UI

---

## 📦 Requirements

Make sure the following dependencies are installed (already listed in `requirements.txt`):

* Python 3.8+
* YOLOv8 (via `ultralytics`)
* OpenCV
* NumPy, Pandas
* Streamlit
* Matplotlib or Plotly

Install them all using:

```bash
pip install -r requirements.txt
```

---

## 📁 .gitignore Tips

To avoid committing large generated files, add the following to your `.gitignore`:

```
data/outputs/
data/frames/
models/yolo/*.pt
*.log
*.csv
__pycache__/
```

---

## 💡 Future Enhancements

* ✅ Integrate ML-based safety score prediction (replace heuristics)
* 🌙 Add weather/night driving risk detection
* 🚀 Deploy on edge devices for real-time use

---

## 🤝 Contributing

Pull requests are welcome! Please:

* Follow standard Python coding practices
* Keep functions and modules modular
* Document your changes in the README if needed

---

# Contributors 🙋🏽

<p align="center">
  <a href="https://github.com/VIZAGBOYS/CSRI_CRRI_ROAD_SAFETY_SCORE">
    <img src="https://img.shields.io/badge/GitHub-CSRI_CRRI_ROAD_SAFETY_SCORE-blue?logo=github" alt="Repo Badge" />
  </a>
</p>

<p align="center">
  <a href="https://github.com/VIZAGBOYS/CSRI_CRRI_ROAD_SAFETY_SCORE/graphs/contributors">
    <img src="https://api.vaunt.dev/v1/github/entities/VIZAGBOYS/repositories/CSRI_CRRI_ROAD_SAFETY_SCORE/contributors?format=svg&limit=100" width="700" height="250" />
  </a>
</p>


<br>
  
## Thank you for contributing 💗 
We truly appreciate your time and effort to help improve our project. Happy coding! 🚀

Thankyou


---

## 🙏 Acknowledgements

* CSIR-CRRI for the problem statement and research inspiration
* Ultralytics for the YOLOv8 model
* OpenCV, Streamlit, and the open-source community ❤️

## 🧾 License

MIT License © 2025 Murali Paila

---
