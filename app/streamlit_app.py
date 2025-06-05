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
            background-color: rgba(248, 249, 250, 0.9);
        }
        .stApp {
            background-image: url('app\\CSIR.png');
            background-size: 60vw;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-opacity: 0.1;
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
        .stMetric {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# App header
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
            else:
                st.warning("Logo image not found")
        except Exception as e:
            st.warning(f"Could not load logo: {str(e)}")

    with col2:
        st.title("Road Safety Scoring System")

    with col3:
        try:
            right_logo_path = Path(__file__).parent / "CSIR.png"
            if right_logo_path.exists():
                right_logo = Image.open(right_logo_path)
                padded_right_logo = ImageOps.expand(right_logo, border=16, fill=(255, 255, 255))
                st.image(padded_right_logo, width=150)
            else:
                st.warning("Right logo image not found")
        except Exception as e:
            st.warning(f"Could not load right logo: {str(e)}")

    st.markdown("""
        <div style="color: #4a4a4a; font-size: 16px; margin-bottom: 2rem;">
            Upload dashcam footage to analyze road safety conditions. 
            The system detects and tracks vehicles, pedestrians, and animals to compute a safety score (1-10).
        </div>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Upload Video")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

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

    device_options = ["cpu", "gpu"]
    default_index = 1 if torch.cuda.is_available() else 0
    selected_device = st.selectbox(
        "Processing Device",
        device_options,
        index=default_index,
        help="Select 'gpu' for GPU acceleration if available"
    )
    processing_device = "cuda" if selected_device == "gpu" else "cpu"

    st.markdown("---")
    st.markdown("""
        <div style="font-size: 14px; color: #4a4a4a;">
            This system uses:
            <ul>
                <li>YOLOX for object detection</li>
                <li>ByteTrack for object tracking</li>
                <li>Custom logic for safety scoring</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# Main container
main_container = st.container()
with main_container:
    if uploaded_file is not None:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
            tmp_input.write(uploaded_file.read())
            input_path = tmp_input.name

        output_dir = os.path.join(ROOT_DIR, "output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"annotated_{uploaded_file.name}")

        # Display video info
        try:
            cap = read_video(input_path)
            if cap is None:
                st.error("Error: Could not read the uploaded video file.")
            else:
                width, height, frame_count, fps = get_video_properties(cap)
                cap.release()

                st.markdown("### Video Analysis Preview")
                info_col, preview_col = st.columns([1, 1.5])

                with info_col:
                    with st.expander("📊 Video Information", expanded=True):
                        st.markdown(f"""
                            **Resolution:** {width}x{height}  
                            **Duration:** {frame_count/fps:.2f} seconds  
                            **Frame Count:** {frame_count}  
                            **FPS:** {fps:.2f}  
                            **Segment Size:** {segment_size} seconds  
                            **Device:** {processing_device.upper()}
                        """)

                with preview_col:
                    st.video(input_path)

                # ⬇️ NEW Analyze Button with Updated Results Section
                st.markdown("---")
                if st.button("🚀 Analyze Video", use_container_width=True):
                    with st.spinner("Processing video..."):
                        try:
                            scorer = RoadSafetyScorer(
                                device=processing_device,
                                segment_size=segment_size
                            )
                            start_time = time.time()
                            result = scorer.process_video(input_path, output_path)
                            result["processing_time"] = time.time() - start_time

                            if result.get("error"):
                                st.error(f"Error: {result['error']}")
                            else:
                                st.success("Analysis completed!")

                                st.markdown("## Results Overview")

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Average Score", f"{result.get('average_score', 0):.1f}/10")
                                with col2:
                                    st.metric("Processing Time", f"{result.get('processing_time', 0):.2f}s")
                                with col3:
                                    st.metric("Segment Size", f"{result.get('segment_size', segment_size)}s")

                                # Report and graph
                                report_df = result.get("report", pd.DataFrame())
                                if not report_df.empty:
                                    st.markdown("### Detailed Report")
                                    st.dataframe(report_df)

                                    st.markdown("### Safety Score Timeline")
                                    st.line_chart(report_df.set_index('timestamp')['score'])

                                # Annotated video and downloads
                                st.markdown("### Annotated Video")
                                st.video(output_path)

                                st.download_button(
                                    label="📥 Download Annotated Video",
                                    data=open(output_path, "rb").read(),
                                    file_name=f"annotated_{uploaded_file.name}",
                                    mime="video/mp4",
                                    use_container_width=True
                                )

                                if not report_df.empty:
                                    st.download_button(
                                        label="📊 Download Report (CSV)",
                                        data=report_df.to_csv(index=False).encode("utf-8"),
                                        file_name="safety_report.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                        except Exception as e:
                            st.error(f"Processing failed: {str(e)}")
                            st.exception(e)

                # Cleanup
                try:
                    os.unlink(input_path)
                except Exception as e:
                    st.warning(f"Could not delete temporary file: {str(e)}")
        except Exception as e:
            st.error(f"❌ Error processing video: {str(e)}")
            st.exception(e)
    else:
        st.info("ℹ️ Please upload a video file to get started.")
