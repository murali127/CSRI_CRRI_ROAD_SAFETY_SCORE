# COMPLETELY FIXED VERSION WITH ALL IMPROVEMENTS

# 1. CRITICAL FIXES AT THE TOP
import asyncio
import sys
import os
from pathlib import Path

# Windows-specific fixes
if sys.platform == "win32":
    # Fix for asyncio event loop
    if sys.version_info >= (3, 8) and sys.version_info < (3, 9):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Disable problematic file watcher
    os.environ['STREAMLIT_SERVER_ENABLE_STATIC_FILE_HANDLING'] = 'false'

# 2. PATH CONFIGURATION WITH FALLBACKS
try:
    PROJECT_ROOT = Path(__file__).parent.parent
    # Try project data directory first
    DATA_DIR = PROJECT_ROOT / "data"
    DATA_DIR.mkdir(exist_ok=True)
    
    # Test if we can write to project directory
    test_file = DATA_DIR / "permission_test.txt"
    test_file.write_text("test")
    test_file.unlink()
except Exception:
    # Fallback to user directory if project directory fails
    DATA_DIR = Path.home() / "csir_crri_data"
    DATA_DIR.mkdir(exist_ok=True)

# Define subdirectories
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "outputs"

# 3. IMPORTS WITH PROPER ERROR HANDLING
try:
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Add project root to Python path
    sys.path.append(str(PROJECT_ROOT))
    
    from main import process_video
    from utils.video_utils import extract_frames
except ImportError as e:
    print(f"CRITICAL: Missing dependencies - {e}")
    print("Please run: pip install -r requirements.txt")
    sys.exit(1)

# 4. STREAMLIT APP CONFIGURATION
st.set_page_config(
    page_title="Road Safety Analysis",
    layout="wide",
    page_icon="🚦",
    initial_sidebar_state="expanded"
)

# Custom CSS for better display
st.markdown("""
<style>
    .stDataFrame {
        width: 100%;
    }
    .metric-box {
        border-left: 5px solid #4e79a7;
        padding: 10px;
        margin-bottom: 10px;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    .risk-high {
        color: #d62728;
        font-weight: bold;
    }
    .risk-medium {
        color: #ff7f0e;
        font-weight: bold;
    }
    .risk-low {
        color: #2ca02c;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    [data-testid="stFileUploader"] {
        padding: 10px;
        background: #f0f2f6;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# 5. UTILITY FUNCTIONS
def ensure_directory(path: Path) -> bool:
    """Ensure directory exists and is writable"""
    try:
        path.mkdir(parents=True, exist_ok=True)
        test_file = path / "permission_test.txt"
        test_file.write_text("test")
        test_file.unlink()
        return True
    except Exception as e:
        st.error(f"⚠️ Directory error: {path} - {str(e)}")
        st.error("Please check permissions or run as administrator")
        return False

def save_uploaded_file(uploaded_file, target_path: Path) -> bool:
    """Save uploaded file with chunked writing"""
    try:
        CHUNK_SIZE = 4096 * 1024  # 4MB chunks for large files
        
        with st.spinner(f"Saving {uploaded_file.name}..."):
            with open(target_path, "wb") as f:
                while chunk := uploaded_file.read(CHUNK_SIZE):
                    f.write(chunk)
        return True
    except Exception as e:
        st.error(f"Failed to save file: {str(e)}")
        return False

# 6. MAIN APP CODE
st.title("🚦 Road Safety Scoring System")

# Sidebar for inputs
with st.sidebar:
    st.header("📤 Video Input")
    uploaded_file = st.file_uploader(
        "Upload Dashcam Video",
        type=["mp4", "mov", "avi"],
        help="Upload a video file for safety analysis"
    )
    
    if uploaded_file:
        # Create timestamped filename to avoid conflicts
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = f"video_{timestamp}{Path(uploaded_file.name).suffix}"
        input_path = RAW_DIR / safe_name
        
        # Ensure directory exists
        if not ensure_directory(RAW_DIR):
            st.stop()
            
        # Save file with progress
        if save_uploaded_file(uploaded_file, input_path):
            st.success(f"✅ File saved to: {input_path}")
            
            # Process button
            if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
                with st.spinner("Processing video (this may take several minutes)..."):
                    if not ensure_directory(OUTPUT_DIR):
                        st.stop()
                    
                    try:
                        process_video(input_path, OUTPUT_DIR)
                        st.session_state['processed'] = True
                        st.session_state['video_path'] = OUTPUT_DIR / "output.mp4"
                        st.session_state['report_path'] = OUTPUT_DIR / "safety_scores.csv"
                        st.session_state['log_path'] = OUTPUT_DIR / "detection_log.csv"
                        st.success("🎉 Analysis complete!")
                    except Exception as e:
                        st.error(f"Processing failed: {str(e)}")
                        st.error("Check console for detailed error message")

# Main display area
if 'processed' in st.session_state and st.session_state['processed']:
    # Video display
    st.header("🎥 Processed Video Preview")
    try:
        video_file = open(st.session_state['video_path'], 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
    except Exception as e:
        st.error(f"Couldn't load video: {str(e)}")

    # Load data
    try:
        report_df = pd.read_csv(st.session_state['report_path'])
        detection_log = pd.read_csv(st.session_state['log_path'])
    except Exception as e:
        st.error(f"Couldn't load results: {str(e)}")
        st.stop()
    
    # Overall metrics
    st.header("📊 Safety Summary")
    col1, col2, col3 = st.columns(3)
    
    avg_score = report_df['score'].mean()
    if avg_score >= 7:
        risk_class = "risk-high"
        risk_level = "High Risk"
    elif avg_score >= 4:
        risk_class = "risk-medium"
        risk_level = "Medium Risk"
    else:
        risk_class = "risk-low"
        risk_level = "Low Risk"
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <h3>Average Safety Score</h3>
            <p><span class="{risk_class}">{avg_score:.1f}/10</span> ({risk_level})</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        pedestrian_risks = len(detection_log[detection_log['event'].str.contains("Pedestrian", na=False)])
        st.markdown(f"""
        <div class="metric-box">
            <h3>Pedestrian Proximity</h3>
            <p>{pedestrian_risks} incidents</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        vehicle_density = len(detection_log[detection_log['event'].str.contains("High vehicle density", na=False)])
        st.markdown(f"""
        <div class="metric-box">
            <h3>Vehicle Density</h3>
            <p>{vehicle_density} incidents</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Score timeline
    st.header("📈 Safety Score Over Time")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(report_df['start_time'], report_df['score'], marker='o', markersize=3, color='#4e79a7')
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Safety Score")
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Detailed detections
    st.header("📝 Detailed Detection Log")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        risk_filter = st.multiselect(
            "Filter by risk type",
            options=["Pedestrian", "Vehicle density", "Pothole", "No risks"],
            default=["Pedestrian", "Vehicle density", "Pothole"]
        )
    
    with col2:
        score_range = st.slider(
            "Filter by score range",
            0, 10, (0, 10)
        )
    
    # Apply filters
    filtered_log = detection_log.copy()
    if risk_filter:
        condition = False
        if "Pedestrian" in risk_filter:
            condition |= filtered_log['event'].str.contains("Pedestrian", na=False)
        if "Vehicle density" in risk_filter:
            condition |= filtered_log['event'].str.contains("High vehicle density", na=False)
        if "Pothole" in risk_filter:
            condition |= filtered_log['event'].str.contains("Pothole", na=False)
        if "No risks" in risk_filter:
            condition |= filtered_log['event'].str.contains("No risks", na=False)
        filtered_log = filtered_log[condition]
    
    filtered_log = filtered_log[
        (filtered_log['score'] >= score_range[0]) & 
        (filtered_log['score'] <= score_range[1])
    ]
    
    # Show filtered data
    st.dataframe(
        filtered_log,
        column_config={
            "frame_number": "Frame #",
            "event": "Risk Event",
            "score": st.column_config.ProgressColumn(
                "Score",
                help="Safety score for this event",
                format="%.1f",
                min_value=0,
                max_value=10,
            ),
            "timestamp": st.column_config.NumberColumn("Timestamp (s)", format="%.1f")
        },
        hide_index=True,
        use_container_width=True,
        height=500
    )
    
    # Download section
    st.header("💾 Download Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        try:
            with open(st.session_state['video_path'], "rb") as f:
                st.download_button(
                    "📹 Download Processed Video",
                    f,
                    file_name="safety_processed.mp4",
                    help="Download the annotated video with safety scores",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Couldn't prepare video download: {str(e)}")
    
    with col2:
        try:
            with open(st.session_state['report_path'], "rb") as f:
                st.download_button(
                    "📊 Download Score Report",
                    f,
                    file_name="safety_scores.csv",
                    help="Download the detailed score timeline",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Couldn't prepare report download: {str(e)}")
    
    with col3:
        try:
            with open(st.session_state['log_path'], "rb") as f:
                st.download_button(
                    "📋 Download Detection Log",
                    f,
                    file_name="detection_log.csv",
                    help="Download the complete detection log",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Couldn't prepare log download: {str(e)}")

elif uploaded_file and 'processed' not in st.session_state:
    st.info("ℹ️ Click 'Analyze Video' to process the uploaded file")
else:
    st.info("ℹ️ Please upload a dashcam video to begin analysis")