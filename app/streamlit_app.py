import streamlit as st
from PIL import Image
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
from PIL import ImageOps

# Page configuration
st.set_page_config(
    page_title="Road Safety Scoring System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
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
        .stProgress>div>div>div {
            background-color: #4a6fa5;
        }
        .stMetric {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stDataFrame {
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stAlert {
            border-radius: 8px;
        }
        .header {
            display: flex;
            align-items: center;
            margin-bottom: 1.5rem;
        }
        .title-container {
            margin-left: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# App header with logo and title
header = st.container()
with header:
    col1, col2, col3 = st.columns([1, 5, 1])
    with col1:
        # Left logo
        try:
            logo_path = Path(__file__).parent / "logo.png"
            if logo_path.exists():
            logo = Image.open(logo_path)
            padded_logo = ImageOps.expand(logo, border=16, fill=(255, 255, 255))
            st.image(padded_logo, width=80)
            else:
            st.warning("Logo image not found")
        except Exception as e:
            st.warning(f"Could not load logo: {str(e)}")

        with col2:
        st.title("Road Safety Scoring System")

        with col3:
        # Right logo
        try:
            right_logo_path = Path(__file__).parent / "csir-india.jpg"
            if right_logo_path.exists():
            right_logo = Image.open(right_logo_path)
            padded_right_logo = ImageOps.expand(right_logo, border=16, fill=(255, 255, 255))
            st.image(padded_right_logo, width=80)
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

# Sidebar for upload and settings
with st.sidebar:
    st.markdown("""
        <div style="font-size: 18px; font-weight: 600; margin-bottom: 1rem;">
            Upload Video
        </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a video file", 
        type=["mp4", "avi", "mov"],
        key="video_uploader",
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("""
        <div style="font-size: 18px; font-weight: 600; margin-bottom: 1rem;">
            Analysis Settings
        </div>
    """, unsafe_allow_html=True)
    
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
    # Map 'gpu' to 'cuda' for internal use
    processing_device = "cuda" if selected_device == "gpu" else "cpu"
    
    st.markdown("---")
    st.markdown("""
        <div style="font-size: 18px; font-weight: 600; margin-bottom: 1rem;">
            About
        </div>
        <div style="font-size: 14px; color: #4a4a4a;">
            This system uses:
            <ul>
                <li>YOLOX for object detection</li>
                <li>ByteTrack for object tracking</li>
                <li>Custom logic for safety scoring</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# Main content area
main_container = st.container()
with main_container:
    if uploaded_file is not None:
        # Create temporary files
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
                            <div style="margin-bottom: 1rem;">
                                <span style="color: #6c757d;">Resolution:</span> 
                                <span style="font-weight: 500;">{width}x{height}</span>
                            </div>
                            <div style="margin-bottom: 1rem;">
                                <span style="color: #6c757d;">Duration:</span> 
                                <span style="font-weight: 500;">{frame_count/fps:.2f} seconds</span>
                            </div>
                            <div style="margin-bottom: 1rem;">
                                <span style="color: #6c757d;">Frame count:</span> 
                                <span style="font-weight: 500;">{frame_count}</span>
                            </div>
                            <div style="margin-bottom: 1rem;">
                                <span style="color: #6c757d;">FPS:</span> 
                                <span style="font-weight: 500;">{fps:.2f}</span>
                            </div>
                            <div style="margin-bottom: 1rem;">
                                <span style="color: #6c757d;">Segment size:</span> 
                                <span style="font-weight: 500;">{segment_size} seconds</span>
                            </div>
                            <div>
                                <span style="color: #6c757d;">Processing device:</span> 
                                <span style="font-weight: 500;">{processing_device.upper()}</span>
                            </div>
                        """, unsafe_allow_html=True)
                
                with preview_col:
                    st.video(input_path)
                
                # Process video button
                st.markdown("---")
                if st.button("🚀 Analyze Video", use_container_width=True):
                    with st.spinner("Processing video... This may take a while depending on video length."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Initialize scorer with user settings
                        try:
                            scorer = RoadSafetyScorer(
                                device=processing_device,
                                segment_size=segment_size
                            )
                            
                            # Process video
                            start_time = time.time()
                            result = scorer.process_video(input_path, output_path)
                            processing_time = time.time() - start_time
                            
                            # Display results
                            st.success(f"✅ Analysis completed in {processing_time:.2f} seconds!")
                            
                            # Results section
                            st.markdown("## Results Overview")
                            
                            # Metrics row
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            with metric_col1:
                                st.metric(
                                    label="Average Safety Score",
                                    value=f"{result['average_score']:.1f}/10",
                                    help="Overall safety score for the entire video"
                                )
                            with metric_col2:
                                st.metric(
                                    label="Max Vehicles in Segment",
                                    value=result['report']['vehicle'].max(),
                                    help="Highest number of vehicles detected in any segment"
                                )
                            with metric_col3:
                                st.metric(
                                    label="Max Pedestrians in Segment",
                                    value=result['report']['pedestrian'].max(),
                                    help="Highest number of pedestrians detected in any segment"
                                )
                            
                            # Annotated video
                            st.markdown("### Annotated Video Preview")
                            st.video(output_path)
                            
                            # Detailed report
                            st.markdown("### Detailed Safety Analysis")
                            with st.expander("📈 View Full Report Data", expanded=True):
                                st.dataframe(result['report'])
                            
                            # Visualization
                            st.markdown("### Safety Score Timeline")
                            st.line_chart(
                                result['report'].set_index('timestamp')['score'],
                                use_container_width=True
                            )
                            
                            # Download section
                            st.markdown("---")
                            st.markdown("### Download Results")
                            dl_col1, dl_col2 = st.columns(2)
                            with dl_col1:
                                with open(output_path, "rb") as f:
                                    st.download_button(
                                        label="📥 Download Annotated Video",
                                        data=f,
                                        file_name=f"annotated_{uploaded_file.name}",
                                        mime="video/mp4",
                                        use_container_width=True
                                    )
                            
                            with dl_col2:
                                csv = result['report'].to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📊 Download Report (CSV)",
                                    data=csv,
                                    file_name="safety_report.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                        
                        except Exception as e:
                            st.error(f"❌ An error occurred during processing: {str(e)}")
                            st.exception(e)
                
                # Clean up
                try:
                    os.unlink(input_path)
                except Exception as e:
                    st.warning(f"⚠️ Could not delete temporary file: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Error processing video: {str(e)}")
            st.exception(e)
    else:
        st.info("ℹ️ Please upload a video file to get started.")