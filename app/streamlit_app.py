import streamlit as st
import pandas as pd
import time
import os
import sys
from pathlib import Path
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
    layout="wide"
)

# App title and description
st.title("🚦 Road Safety Scoring System")
st.markdown("""
    Upload dashcam footage to analyze road safety conditions. 
    The system detects and tracks vehicles, pedestrians, and animals to compute a safety score (1-10).
""")

# Sidebar for upload and settings
with st.sidebar:
    st.header("Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a video file", 
        type=["mp4", "avi", "mov"],
        key="video_uploader"
    )
    
    st.header("Analysis Settings")
    segment_size = st.slider(
        "Segment Size (seconds)",
        min_value=1.0,
        max_value=30.0,
        value=5.0,
        step=0.5,
        help="Duration of each analysis segment"
    )
    
    processing_device = st.selectbox(
        "Processing Device",
        ["cpu", "cuda"],
        index=1 if torch.cuda.is_available() else 0,
        help="Select 'cuda' for GPU acceleration if available"
    )
    
    st.header("About")
    st.markdown("""
    This system uses:
    - YOLOX for object detection
    - ByteTrack for object tracking
    - Custom logic for safety scoring
    """)

# Main content area
if uploaded_file is not None:
    # Create temporary files
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
        tmp_input.write(uploaded_file.read())
        input_path = tmp_input.name
    
    output_dir = os.path.join(ROOT_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"annotated_{uploaded_file.name}")
    
    # Display video info
    cap = read_video(input_path)
    if cap is None:
        st.error("Error: Could not read the uploaded video file.")
    else:
        width, height, frame_count, fps = get_video_properties(cap)
        cap.release()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Video Information")
            st.write(f"Resolution: {width}x{height}")
            st.write(f"Duration: {frame_count/fps:.2f} seconds")
            st.write(f"Frame count: {frame_count}")
            st.write(f"FPS: {fps:.2f}")
            st.write(f"Segment size: {segment_size} seconds")
            st.write(f"Processing device: {processing_device.upper()}")
        
        with col2:
            st.subheader("Preview")
            st.video(input_path)
        
        # Process video
        if st.button("Analyze Video"):
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
                    st.success(f"Analysis completed in {processing_time:.2f} seconds!")
                    
                    # Show annotated video
                    st.subheader("Annotated Video")
                    st.video(output_path)
                    
                    # Show report
                    st.subheader("Safety Analysis Report")
                    
                    # Overall stats
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Average Safety Score", f"{result['average_score']:.1f}/10")
                    col2.metric("Max Vehicles in Segment", result['report']['vehicle'].max())
                    col3.metric("Max Pedestrians in Segment", result['report']['pedestrian'].max())
                    
                    # Detailed segment data
                    st.dataframe(result['report'])
                    
                    # Plot safety score over time
                    st.subheader("Safety Score Over Time")
                    st.line_chart(result['report'].set_index('timestamp')['score'])
                    
                    # Download options
                    st.subheader("Download Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        with open(output_path, "rb") as f:
                            st.download_button(
                                label="Download Annotated Video",
                                data=f,
                                file_name=f"annotated_{uploaded_file.name}",
                                mime="video/mp4"
                            )
                    
                    with col2:
                        csv = result['report'].to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Report (CSV)",
                            data=csv,
                            file_name="safety_report.csv",
                            mime="text/csv"
                        )
                
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")
                    st.exception(e)
        
        # Clean up
        try:
            os.unlink(input_path)
        except:
            pass
else:
    st.info("Please upload a video file to get started.")