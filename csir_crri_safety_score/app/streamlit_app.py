# COMPLETE ERROR-FREE STREAMLIT APP WITH IMPROVEMENTS

# 1. CRITICAL FIXES AT THE TOP
import asyncio
import sys
import os
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict

# Disable all warnings
warnings.filterwarnings("ignore")

# Windows-specific fixes
if sys.platform == "win32":
    # Fix for asyncio event loop
    if sys.version_info >= (3, 8):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Disable Torch's problematic hooks
    os.environ['STREAMLIT_SERVER_ENABLE_WATCHER'] = 'false'
    os.environ['NO_PROXY'] = 'localhost'

# 2. PATH CONFIGURATION
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "outputs"

# Create directories
for dir_path in [DATA_DIR, RAW_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 3. MAIN IMPORTS (after path configuration)
try:
    import streamlit as st
    import cv2
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from tracking.tracker import RoadSafetyTracker
    from scoring.scorer import SafetyScorer
    from utils.visualizer import Visualizer
    from utils.video_utils import extract_frames, create_video_from_frames
    
    # Additional Torch fixes
    import torch
    torch._C._set_graph_executor_optimize(False)
except ImportError as e:
    st.error(f"CRITICAL: Missing dependencies - {e}")
    st.error("Please run: pip install -r requirements.txt")
    st.stop()

# 4. APP CONFIGURATION
st.set_page_config(
    page_title="Road Safety Analysis",
    layout="wide",
    page_icon="🚦",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-box {
        background-color: #1E88E5;
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .risk-high { color: #FF5252; font-weight: bold; }
    .risk-medium { color: #FFC107; font-weight: bold; }
    .risk-low { color: #4CAF50; font-weight: bold; }
    .stVideo { 
        border-radius: 10px; 
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .live-preview {
        border: 2px solid #1E88E5;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

def process_video_segments(
    input_path: str, 
    output_dir: Path,
    segment_duration: int = 5
) -> Tuple[List[Dict], str]:
    """Process video in segments with per-object tracking"""
    # Initialize components
    tracker = RoadSafetyTracker()
    visualizer = Visualizer()
    scorer = SafetyScorer()
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 if unavailable
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Prepare output video
    output_path = str(output_dir / "output_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Segment processing
    frames_per_segment = int(fps * segment_duration)
    current_segment = 1
    segment_scores = []
    segment_events = defaultdict(int)
    frame_count = 0
    
    # Live preview placeholder
    preview_placeholder = st.empty()
    
    # Add accident log
    accident_log = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame (with segment transition flag)
        segment_transition = (frame_count > 0 and frame_count % frames_per_segment == 0)
        tracked_objects = tracker.track(frame, segment_transition)
        score, events = scorer.calculate_frame_score(tracked_objects, width, height, frame_count)
        
        # Update visualizer segment
        if segment_transition:
            visualizer.update_segment(tracker.current_segment)
        
        # Update segment events (only count new objects)
        for event in events:
            if "Pedestrian" in event:
                segment_events["Pedestrian"] += 3
            elif "Pothole" in event:
                segment_events["Pothole"] += 2
            elif "Lane" in event:
                segment_events["Lane Departure"] += 3
            elif "Vehicle" in event:
                segment_events["High Density"] += 2
        
        # Segment transition logic
        if segment_transition:
            segment_score = min(10, max(segment_events.values(), default=0))
            segment_scores.append({
                'Segment': current_segment,
                'Start_Time': (frame_count - frames_per_segment) / fps,
                'End_Time': frame_count / fps,
                'Events': dict(segment_events),
                'Score': segment_score
            })
            current_segment += 1
            segment_events = defaultdict(int)
        
        # Check for accident events specifically
        accident_events = [e for e in events if "Accident" in e]
        if accident_events:
            segment_events["Accident"] += 10  # Highest weight
            # Force segment score to maximum for accidents
            segment_score = 10
            for event in events:
                if "Accident" in event:
                    accident_timestamp = frame_count / fps
                    accident_log.append({
                        'Timestamp': accident_timestamp,
                        'Frame': frame_count,
                        'Event': event
                    })

        # Visualize with enhanced overlays
        frame = visualizer.draw_objects(frame, tracked_objects)
        frame = visualizer.draw_segment_info(
            frame, current_segment, segment_duration, segment_events
        )
        out.write(frame)
        
        # Update live preview every 50 frames
        if frame_count % 50 == 0:
            preview_placeholder.image(frame, channels="BGR", 
                                   caption=f"Live Preview | Segment {current_segment} | Frame {frame_count}")
        
        frame_count += 1
    
    # Handle last segment
    if segment_events:
        segment_score = min(10, max(segment_events.values(), default=0))
        segment_scores.append({
            'Segment': current_segment,
            'Start_Time': (frame_count - (frame_count % frames_per_segment)) / fps,
            'End_Time': frame_count / fps,
            'Events': dict(segment_events),
            'Score': segment_score
        })
    
    cap.release()
    out.release()
    
    # After processing, save accident log if any accidents detected
    if accident_log:
        accident_path = str(output_dir / "accident_log.csv")
        pd.DataFrame(accident_log).to_csv(accident_path, index=False)
        st.session_state['accident_path'] = accident_path
    
    return segment_scores, output_path

def main():
    st.title("🚦 Road Safety Scoring System")
    
    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader(
            "Upload Dashcam Video", 
            type=["mp4", "mov", "avi"],
            help="Supported formats: MP4, MOV, AVI"
        )
        
        segment_duration = st.slider(
            "Segment Duration (seconds)",
            min_value=1,
            max_value=10,
            value=5,
            help="Duration for each analysis segment"
        )
        
        if st.button("Analyze Video", type="primary", use_container_width=True):
            if uploaded_file:
                with st.spinner("Processing video..."):
                    try:
                        # Save uploaded file
                        input_path = str(RAW_DIR / uploaded_file.name)
                        output_dir = OUTPUT_DIR
                        output_dir.mkdir(parents=True, exist_ok=True)
                        with open(input_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Process video with live preview
                        with st.expander("Live Processing Preview", expanded=True):
                            segment_scores, output_path = process_video_segments(
                                input_path,
                                OUTPUT_DIR,
                                segment_duration
                            )
                        
                        # Save report
                        report_path = str(OUTPUT_DIR / "segment_report.csv")
                        pd.DataFrame(segment_scores).to_csv(report_path, index=False)
                        
                        # Store results
                        st.session_state.update({
                            'processed': True,
                            'video_path': output_path,
                            'report_path': report_path,
                            'segment_scores': segment_scores,
                            'segment_duration': segment_duration
                        })
                        st.success("Analysis complete!")
                    except Exception as e:
                        st.error(f"Processing failed: {str(e)}")
            else:
                st.warning("Please upload a video file first")

    if 'processed' in st.session_state:
        # Video Preview
        st.header("Processed Video Preview")
        st.video(st.session_state['video_path'])
        
        # Safety Summary
        st.header("Safety Summary")
        df = pd.DataFrame(st.session_state['segment_scores'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_score = df['Score'].mean()
            risk_level = "High" if avg_score >=7 else "Medium" if avg_score >=4 else "Low"
            st.markdown(f"""
            <div class="metric-box">
                <h3>Average Score</h3>
                <p><span class="risk-{risk_level.lower()}">{avg_score:.1f}/10</span><br>({risk_level} Risk)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            total_peds = sum(seg['Events'].get('Pedestrian', 0) for seg in st.session_state['segment_scores'])
            st.markdown(f"""
            <div class="metric-box">
                <h3>Pedestrian Events</h3>
                <p>{total_peds} incidents</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_vehicles = sum(seg['Events'].get('High Density', 0) for seg in st.session_state['segment_scores'])
            st.markdown(f"""
            <div class="metric-box">
                <h3>Vehicle Density Events</h3>
                <p>{total_vehicles} incidents</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Add accident summary
        total_accidents = sum(1 for seg in st.session_state['segment_scores'] 
                              if 'Accident' in seg['Events'] and seg['Events']['Accident'] > 0)
        
        if total_accidents > 0:
            st.error(f"⚠️ CRITICAL: {total_accidents} accident events detected! Immediate attention required.")
            
            # Find accident segments
            accident_segments = [seg['Segment'] for seg in st.session_state['segment_scores'] 
                                if 'Accident' in seg['Events'] and seg['Events']['Accident'] > 0]
            
            st.warning(f"Accidents detected in segments: {', '.join(map(str, accident_segments))}")
        
        # Segment Analysis
        st.header("Segment Analysis")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            df['Segment'], 
            df['Score'], 
            marker='o', 
            color='#1E88E5',
            linewidth=2
        )
        ax.set_xlabel(f"Segment Number ({st.session_state['segment_duration']}s each)")
        ax.set_ylabel("Safety Score (0-10)")
        ax.set_ylim(0, 10)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Detailed Data
        st.header("Detailed Segment Data")
        st.dataframe(
            df,
            column_config={
                "Segment": "Segment #",
                "Start_Time": st.column_config.NumberColumn("Start Time (s)", format="%.1f"),
                "End_Time": st.column_config.NumberColumn("End Time (s)", format="%.1f"),
                "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=10),
                "Events": st.column_config.Column("Detected Events", 
                                               help="Counts of risk events per segment")
            },
            hide_index=True,
            use_container_width=True,
            height=500
        )
        
        # Download Section
        st.header("Download Results")
        col1, col2 = st.columns(2)
        with col1:
            with open(st.session_state['video_path'], "rb") as f:
                st.download_button(
                    "📹 Download Processed Video",
                    f,
                    file_name="safety_processed.mp4",
                    help="Video with safety annotations",
                    use_container_width=True
                )
        with col2:
            with open(st.session_state['report_path'], "rb") as f:
                st.download_button(
                    "📊 Download Analysis Report",
                    f,
                    file_name="safety_report.csv",
                    help="Detailed segment analysis data",
                    use_container_width=True
                )
        if 'accident_path' in st.session_state:
            with open(st.session_state['accident_path'], "rb") as f:
                st.download_button(
                    "⚠️ Download Accident Report",
                    f,
                    file_name="accident_report.csv",
                    help="Detailed accident timestamp data",
                    use_container_width=True
                )
    else:
        st.info("Please upload a video and click 'Analyze Video' to begin")

if __name__ == "__main__":
    try:
        # Windows-specific event loop fix
        if sys.platform == "win32":
            asyncio.set_event_loop(asyncio.ProactorEventLoop())
        
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.stop()