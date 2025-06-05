import streamlit as st
from PIL import Image, ImageOps
import os
from pathlib import Path
import pandas as pd
import time
import sys
import tempfile
import torch
import cv2
from typing import Optional

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Now import modules
from main import RoadSafetyScorer
from utils.video_utils import read_video, get_video_properties

# Page configuration
st.set_page_config(
    page_title="Road Safety Scoring System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stApp {
            background-size: 60vw;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            opacity: 0.95;
        }
        .sidebar .sidebar-content {
            background-color: #ffffff;
            border-right: 1px solid #e1e4e8;
        }
        .stButton>button {
            background-color: #4a6fa5;
            color: white;
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 500;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #3a5a8a;
            transform: scale(1.02);
        }
        .metric-card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }
        .score-display {
            font-size: 2rem;
            font-weight: bold;
            text-align: center;
        }
        .safe { color: #2ecc71; }
        .moderate { color: #f39c12; }
        .danger { color: #e74c3c; }
        .report-table {
            font-size: 0.9rem;
        }
    </style>
""", unsafe_allow_html=True)

def get_score_class(score):
    """Return CSS class based on safety score"""
    if score >= 7: return "safe"
    elif score >= 4: return "moderate"
    return "danger"

# App header
def render_header():
    header = st.container()
    with header:
        col1, col2, col3 = st.columns([1, 5, 1])
        with col1:
            try:
                logo_path = Path(__file__).parent / "logo.png"
                if logo_path.exists():
                    logo = Image.open(logo_path)
                    padded_logo = ImageOps.expand(logo, border=16, fill=(255, 255, 255))
                    st.image(padded_logo, width=150)
            except Exception:
                pass

        with col2:
            st.title("Road Safety Scoring System")
            st.markdown("""
                <div style="color: #4a4a4a; font-size: 16px; margin-bottom: 2rem;">
                    Upload dashcam footage to analyze road safety conditions. 
                    The system detects vehicles, pedestrians, animals, and potholes to compute a safety score (1-10).
                </div>
            """, unsafe_allow_html=True)

        with col3:
            try:
                right_logo_path = Path(__file__).parent / "CSIR.png"
                if right_logo_path.exists():
                    right_logo = Image.open(right_logo_path)
                    padded_right_logo = ImageOps.expand(right_logo, border=16, fill=(255, 255, 255))
                    st.image(padded_right_logo, width=150)
            except Exception:
                pass

# Sidebar
def render_sidebar():
    with st.sidebar:
        st.markdown("### Upload Video")
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"], key="file_uploader")

        st.markdown("---")
        st.markdown("### Analysis Settings")

        segment_size = st.slider(
            "Segment Size (seconds)",
            min_value=1.0,
            max_value=30.0,
            value=5.0,
            step=0.5,
            help="Duration of each analysis segment"
        )

        device_options = ["CPU", "GPU"] if torch.cuda.is_available() else ["CPU"]
        selected_device = st.selectbox(
            "Processing Device",
            device_options,
            index=0,
            help="Select GPU for faster processing if available"
        )
        processing_device = "cuda" if selected_device == "GPU" else "cpu"

        st.markdown("---")
        st.markdown("""
            <div style="font-size: 14px; color: #4a4a4a;">
                <strong>Detection Models:</strong>
                <ul>
                    <li>YOLOX for object detection</li>
                    <li>ByteTrack for object tracking</li>
                    <li>CNN for pothole detection</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

        return uploaded_file, segment_size, processing_device

# Main analysis function
def analyze_video(uploaded_file, segment_size, processing_device):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
        tmp_input.write(uploaded_file.read())
        input_path = tmp_input.name

    output_dir = os.path.join(ROOT_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"annotated_{uploaded_file.name}")

    try:
        cap = read_video(input_path)
        if cap is None:
            st.error("Error: Could not read the uploaded video file.")
            return None
        
        width, height, frame_count, fps = get_video_properties(cap)
        cap.release()

        with st.spinner("🚀 Processing video... This may take a few minutes..."):
            if 'analysis_results' not in st.session_state:
                st.session_state.analysis_results = None

            scorer = RoadSafetyScorer(
                device=processing_device,
                segment_size=segment_size
            )
            start_time = time.time()
            result = scorer.process_video(input_path, output_path)
            result["processing_time"] = time.time() - start_time
            st.session_state.analysis_results = result

        return result

    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
        return None
    finally:
        try:
            os.unlink(input_path)
        except Exception:
            pass

# Results display
def render_results(result, uploaded_file):  # Add uploaded_file as parameter
    st.markdown("## 📊 Analysis Results")
    
    # Score summary cards
    avg_score = result.get('average_score', 0)
    score_class = get_score_class(avg_score)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 1rem; color: #7f8c8d;">Average Safety Score</div>
                <div class="score-display {score_class}">{avg_score:.1f}<span style="font-size: 1rem;">/10</span></div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 1rem; color: #7f8c8d;">Processing Time</div>
                <div style="font-size: 1.5rem; font-weight: bold; text-align: center;">
                    {result.get('processing_time', 0):.2f}s
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 1rem; color: #7f8c8d;">Video Duration</div>
                <div style="font-size: 1.5rem; font-weight: bold; text-align: center;">
                    {(result.get('frame_stats', [{}])[-1].get('timestamp', 0)):.1f}s
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Video preview and report
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎬 Annotated Video Preview")
        st.video(result["output_video"])
        
        with open(result["output_video"], "rb") as f:
            st.download_button(
                label="📹 Download Annotated Video",
                data=f,
                file_name=f"annotated_{uploaded_file.name}",
                mime="video/mp4",
                use_container_width=True
            )
    
    with col2:
        st.markdown("### 📈 Safety Score Timeline")
        report_df = result.get('report', pd.DataFrame())
        
        if not report_df.empty:
            report_df['time'] = pd.to_datetime(report_df['timestamp'], unit='s').dt.strftime('%M:%S')
            st.line_chart(report_df.set_index('time')['score'])
            
            st.markdown("### 📋 Detailed Report")
            st.dataframe(
                report_df[['time', 'vehicle', 'pedestrian', 'animal', 'pothole', 'score']],
                height=300,
                use_container_width=True
            )
            
            csv = report_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Download Full Report (CSV)",
                data=csv,
                file_name="safety_report.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.warning("No report data available")

# In your main function, update the call to render_results:
def main():
    render_header()
    uploaded_file, segment_size, processing_device = render_sidebar()
    
    if uploaded_file is not None:
        st.markdown("### 🎥 Video Preview")
        st.video(uploaded_file)
        
        if st.button("🚀 Analyze Video", use_container_width=True):
            result = analyze_video(uploaded_file, segment_size, processing_device)
            
            if result and not result.get('error'):
                st.success("✅ Analysis completed successfully!")
                render_results(result, uploaded_file)  # Pass uploaded_file here
            elif result and result.get('error'):
                st.error(f"❌ Analysis failed: {result['error']}")
    
    else:
        st.info("ℹ️ Please upload a video file to get started.")

if __name__ == "__main__":
    main()