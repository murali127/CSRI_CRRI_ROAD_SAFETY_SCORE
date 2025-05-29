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

# Page configuration
st.set_page_config(
    page_title="Road Safety Analytics Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .block-container {
            padding-top: 2rem;
            background: white;
            margin: 2rem;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        
        .sidebar .sidebar-content {
            background: #1a1d29;
            color: white;
            border: none;
        }
        
        .sidebar .sidebar-content * {
            color: white !important;
        }
        
        .sidebar .stSelectbox > div > div {
            background-color: #2d3748;
            color: white;
            border: 1px solid #4a5568;
        }
        
        .sidebar .stSlider > div > div > div {
            color: white;
        }
        
        .sidebar .stFileUploader > div {
            background-color: #2d3748;
            border: 2px dashed #4a5568;
            border-radius: 12px;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 12px 24px;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        }
        
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #667eea, #764ba2);
        }
        
        .stMetric {
            background: white;
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.08);
            border: 1px solid #f1f5f9;
            transition: all 0.3s ease;
        }
        
        .stMetric:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(0,0,0,0.12);
        }
        
        .stDataFrame {
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0,0,0,0.08);
            border: 1px solid #f1f5f9;
        }
        
        .stAlert {
            border-radius: 12px;
            border: none;
            font-family: 'Inter', sans-serif;
        }
        
        .professional-header {
            background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%);
            padding: 2rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            color: white;
            position: relative;
            overflow: hidden;
        }
        
        .professional-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="1"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
            opacity: 0.3;
        }
        
        .professional-header > * {
            position: relative;
            z-index: 1;
        }
        
        .header-title {
            font-family: 'Inter', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            background: linear-gradient(135deg, #ffffff 0%, #e2e8f0 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .header-subtitle {
            font-family: 'Inter', sans-serif;
            font-size: 1.1rem;
            font-weight: 400;
            margin-top: 0.5rem;
            opacity: 0.9;
            line-height: 1.6;
        }
        
        .section-header {
            font-family: 'Inter', sans-serif;
            font-size: 1.5rem;
            font-weight: 600;
            color: #1e293b;
            margin: 2rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e2e8f0;
        }
        
        .info-card {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid #e2e8f0;
            font-family: 'Inter', sans-serif;
        }
        
        .info-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 0;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .info-item:last-child {
            border-bottom: none;
        }
        
        .info-label {
            color: #64748b;
            font-weight: 500;
        }
        
        .info-value {
            color: #1e293b;
            font-weight: 600;
        }
        
        .sidebar-section {
            margin: 2rem 0;
            padding: 1rem 0;
            border-bottom: 1px solid #374151;
        }
        
        .sidebar-title {
            font-family: 'Inter', sans-serif;
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #f9fafb !important;
        }
        
        .about-text {
            font-family: 'Inter', sans-serif;
            font-size: 0.9rem;
            line-height: 1.6;
            color: #d1d5db !important;
        }
        
        .about-list {
            margin: 1rem 0;
            padding-left: 1rem;
        }
        
        .about-list li {
            margin: 0.5rem 0;
            color: #d1d5db !important;
        }
        
        .metric-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }
        
        .chart-container {
            background: white;
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.08);
            border: 1px solid #f1f5f9;
        }
        
        .download-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin: 2rem 0;
        }
        
        .success-message {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 12px;
            font-weight: 500;
            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
        }
        
        .error-message {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 12px;
            font-weight: 500;
            box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
        }
        
        .video-container {
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0,0,0,0.12);
        }
        
        .expander-container {
            background: white;
            border-radius: 16px;
            border: 1px solid #f1f5f9;
            box-shadow: 0 4px 16px rgba(0,0,0,0.04);
        }
    </style>
""", unsafe_allow_html=True)

# Professional header
st.markdown("""
    <div class="professional-header">
        <h1 class="header-title">Road Safety Analytics Platform</h1>
        <p class="header-subtitle">
            Advanced AI-powered dashcam analysis system for comprehensive road safety assessment. 
            Upload your footage to receive detailed safety insights and risk analytics powered by cutting-edge computer vision technology.
        </p>
    </div>
""", unsafe_allow_html=True)

# Professional sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-title">📁 Upload Video File</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Select dashcam footage", 
        type=["mp4", "avi", "mov"],
        key="video_uploader",
        label_visibility="collapsed"
    )
    
    st.markdown('<div class="sidebar-section"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">⚙️ Analysis Configuration</div>', unsafe_allow_html=True)
    
    segment_size = st.slider(
        "Analysis Segment Duration (seconds)",
        min_value=1.0,
        max_value=30.0,
        value=5.0,
        step=0.5,
        help="Duration of each analysis segment for detailed scoring"
    )
    
    processing_device = st.selectbox(
        "Computational Processing Unit",
        ["cpu", "cuda"],
        index=1 if torch.cuda.is_available() else 0,
        format_func=lambda x: "GPU" if x == "cuda" else "CPU",
        help="Select GPU for acceleration when available"
    )
    
    st.markdown('<div class="sidebar-section"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">🔬 Technology Stack</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="about-text">
            Our platform leverages state-of-the-art AI technologies:
            <ul class="about-list">
                <li><strong>YOLOX</strong> - Advanced object detection</li>
                <li><strong>ByteTrack</strong> - Multi-object tracking</li>
                <li><strong>Custom ML</strong> - Safety scoring algorithms</li>
                <li><strong>Computer Vision</strong> - Real-time analysis</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# Main content area
if uploaded_file is not None:
    # Create temporary files
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
        tmp_input.write(uploaded_file.read())
        input_path = tmp_input.name
    
    output_dir = os.path.join(ROOT_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"analyzed_{uploaded_file.name}")
    
    # Display video analysis preview
    try:
        cap = read_video(input_path)
        if cap is None:
            st.markdown('<div class="error-message">❌ Error: Unable to process the uploaded video file. Please ensure the file is not corrupted.</div>', unsafe_allow_html=True)
        else:
            width, height, frame_count, fps = get_video_properties(cap)
            cap.release()
            
            st.markdown('<h2 class="section-header">📊 Video Analysis Dashboard</h2>', unsafe_allow_html=True)
            
            analysis_col1, analysis_col2 = st.columns([1, 1.5])
            
            with analysis_col1:
                st.markdown("""
                    <div class="info-card">
                        <h3 style="margin-top: 0; color: #1e293b; font-family: 'Inter', sans-serif; font-weight: 600;">Video Properties</h3>
                        <div class="info-item">
                            <span class="info-label">Resolution</span>
                            <span class="info-value">{width} × {height}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Duration</span>
                            <span class="info-value">{duration:.2f} seconds</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Total Frames</span>
                            <span class="info-value">{frame_count:,}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Frame Rate</span>
                            <span class="info-value">{fps:.1f} FPS</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Segment Size</span>
                            <span class="info-value">{segment_size} seconds</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Processing Unit</span>
                            <span class="info-value">{processing_device.upper()}</span>
                        </div>
                    </div>
                """.format(
                    width=width,
                    height=height,
                    duration=frame_count/fps,
                    frame_count=frame_count,
                    fps=fps,
                    segment_size=segment_size,
                    processing_device=processing_device
                ), unsafe_allow_html=True)
            
            with analysis_col2:
                st.markdown('<div class="video-container">', unsafe_allow_html=True)
                st.video(input_path)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Professional analysis button
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Initialize Safety Analysis", use_container_width=True):
                with st.spinner("Processing video analytics... Please wait while our AI analyzes your footage."):
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
                        
                        # Professional success message
                        st.markdown(f"""
                            <div class="success-message">
                                ✅ Analysis completed successfully in {processing_time:.2f} seconds! 
                                Your road safety assessment is ready for review.
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Results overview section
                        st.markdown('<h2 class="section-header">📈 Safety Assessment Results</h2>', unsafe_allow_html=True)
                        
                        # Professional metrics display
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        with metric_col1:
                            st.metric(
                                label="Overall Safety Rating",
                                value=f"{result['average_score']:.1f}/10",
                                help="Comprehensive safety score across entire video duration"
                            )
                        with metric_col2:
                            st.metric(
                                label="Peak Vehicle Density",
                                value=result['report']['vehicle'].max(),
                                help="Maximum vehicles detected simultaneously in any segment"
                            )
                        with metric_col3:
                            st.metric(
                                label="Peak Pedestrian Activity",
                                value=result['report']['pedestrian'].max(),
                                help="Maximum pedestrians detected simultaneously in any segment"
                            )
                        
                        # Annotated video results
                        st.markdown('<h2 class="section-header">🎥 Annotated Analysis Video</h2>', unsafe_allow_html=True)
                        st.markdown('<div class="video-container">', unsafe_allow_html=True)
                        st.video(output_path)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Detailed analytics report
                        st.markdown('<h2 class="section-header">📋 Comprehensive Analytics Report</h2>', unsafe_allow_html=True)
                        with st.expander("📊 View Detailed Analysis Data", expanded=True):
                            st.markdown('<div class="expander-container">', unsafe_allow_html=True)
                            st.dataframe(result['report'], use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Professional visualization
                        st.markdown('<h2 class="section-header">📉 Safety Score Timeline</h2>', unsafe_allow_html=True)
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.line_chart(
                            result['report'].set_index('timestamp')['score'],
                            use_container_width=True
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Professional download section
                        st.markdown('<h2 class="section-header">💾 Export Results</h2>', unsafe_allow_html=True)
                        dl_col1, dl_col2 = st.columns(2)
                        with dl_col1:
                            with open(output_path, "rb") as f:
                                st.download_button(
                                    label="📥 Download Annotated Video",
                                    data=f,
                                    file_name=f"safety_analysis_{uploaded_file.name}",
                                    mime="video/mp4",
                                    use_container_width=True
                                )
                        
                        with dl_col2:
                            csv = result['report'].to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📊 Download Analytics Report (CSV)",
                                data=csv,
                                file_name="road_safety_analytics_report.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                    
                    except Exception as e:
                        st.markdown(f"""
                            <div class="error-message">
                                ❌ Processing Error: {str(e)}
                                <br><small>Please try again or contact support if the issue persists.</small>
                            </div>
                        """, unsafe_allow_html=True)
                        st.exception(e)
            
            # Clean up temporary files
            try:
                os.unlink(input_path)
            except Exception as e:
                st.warning(f"⚠️ Temporary file cleanup warning: {str(e)}")
    
    except Exception as e:
        st.markdown(f"""
            <div class="error-message">
                ❌ Video Processing Error: {str(e)}
                <br><small>Please ensure your video file is valid and try again.</small>
            </div>
        """, unsafe_allow_html=True)
        st.exception(e)
else:
    st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border-radius: 20px; margin: 2rem 0;">
            <h3 style="color: #64748b; font-family: 'Inter', sans-serif; font-weight: 500; margin-bottom: 1rem;">
                Ready to Analyze Your Dashcam Footage
            </h3>
            <p style="color: #94a3b8; font-family: 'Inter', sans-serif; font-size: 1.1rem; line-height: 1.6;">
                Upload your video file using the sidebar to begin comprehensive road safety analysis. 
                Our AI will detect vehicles, pedestrians, and assess overall safety conditions.
            </p>
        </div>
    """, unsafe_allow_html=True)