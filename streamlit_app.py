import streamlit as st
from pathlib import Path
import tempfile
import time
import os
import cv2
import numpy as np
from collections import deque
import math
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO
import requests
import imageio

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="CourtVision AI", layout="wide", initial_sidebar_state="collapsed")

# ---------------------------
# PRO UI/CSS
# ---------------------------
# This is the "sexy AF" part. We're injecting custom CSS for a pro look.
st.markdown("""
<style>
/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

/* Set global font */
html, body, [class*="st-"], [class*="css-"] {
    font-family: 'Inter', sans-serif;
}

/* Base theme: Dark */
:root {
    --primary-color: #FF7F00; /* Basketball Orange */
    --background-color: #0E1117; /* Streamlit Dark BG */
    --secondary-background-color: #1A1E29; /* Slightly Lighter Dark */
    --text-color: #FAFAFA;
    --secondary-text-color: #ADB5BD;
    --card-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

/* Main app background */
[data-testid="stAppViewContainer"] {
    background-color: var(--background-color);
    color: var(--text-color);
}

/* Remove Streamlit header/footer */
header {visibility: hidden;}
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}

/* Custom Title */
h1 {
    color: var(--primary-color);
    font-weight: 700;
    font-size: 2.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--primary-color);
}

h3 {
    color: var(--text-color);
    font-weight: 600;
}

h5 {
    color: var(--primary-color);
    font-weight: 600;
    margin-bottom: 0.5rem;
}

/* Style columns as "cards" */
[data-testid="column"] {
    background-color: var(--secondary-background-color);
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: var(--card-shadow);
    border: 1px solid #2C334A;
    transition: all 0.3s ease-in-out;
}

[data-testid="column"]:hover {
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    border-color: #4A557A;
}

/* Main "Analyze" Button */
[data-testid="stButton"] button {
    background-color: var(--primary-color);
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    font-size: 1.1rem;
    width: 100%;
    transition: all 0.3s ease;
}

[data-testid="stButton"] button:hover {
    background-color: #E67300;
    box-shadow: 0 4px 15px rgba(255, 127, 0, 0.3);
}

[data-testid="stButton"] button:active {
    background-color: #CC6600;
}

/* Download Button */
[data-testid="stDownloadButton"] button {
    background-color: #2C334A;
    color: var(--text-color);
    border: 1px solid var(--primary-color);
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: 600;
    transition: all 0.3s ease;
}

[data-testid="stDownloadButton"] button:hover {
    background-color: var(--primary-color);
    color: #FFFFFF;
}

/* Metric Cards */
[data-testid="stMetric"] {
    background-color: #2C334A;
    border-radius: 8px;
    padding: 1rem;
    border-left: 5px solid var(--primary-color);
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
}

[data-testid="stMetricLabel"] {
    color: var(--secondary-text-color);
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: var(--text-color);
    font-size: 2rem;
    font-weight: 700;
}

/* File Uploader */
[data-testid="stFileUploader"] {
    background-color: #2C334A;
    border: 2px dashed #4A557A;
    border-radius: 8px;
    padding: 1rem;
}

[data-testid="stFileUploader"] label {
    color: var(--text-color);
}

/* Checkbox */
[data-testid="stCheckbox"] {
    color: var(--secondary-text-color);
}

/* Selectbox */
[data-testid="stSelectbox"] {
    color: var(--secondary-text-color);
}

/* Separator */
hr {
    border-top: 1px solid #2C334A;
    margin: 1.5rem 0;
}

/* Video element */
video {
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# Helper utils
# ---------------------------
def init_kalman():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([0,0,0,0])
    kf.F = np.array([[1,0,1,0],
                     [0,1,0,1],
                     [0,0,1,0],
                     [0,0,0,1]])
    kf.H = np.array([[1,0,0,0],
                     [0,1,0,0]])
    kf.P *= 1000
    kf.R *= 5
    kf.Q *= 0.01
    return kf

def draw_text_with_outline(img, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX,
                           font_scale=0.6, text_color=(0,255,255), thickness=2,
                           outline_color=(0,0,0), outline_thickness=3):
    x, y = pos
    cv2.putText(img, text, (x,y), font, font_scale, outline_color, outline_thickness, cv2.LINE_AA)
    cv2.putText(img, text, (x,y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return img

COCO_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (0,5),(0,6),(5,7),(7,9),
    (6,8),(8,10),(5,6),(5,11),
    (6,12),(11,12),(11,13),(13,15),
    (12,14),(14,16)
]

def draw_skeleton(frame, kpts_xy, color=(0,255,0)):
    if kpts_xy is None:
        return frame
    h,w = frame.shape[:2]
    pts = np.asarray(kpts_xy, dtype=np.float32)
    pts_i = np.clip(pts,0,[w-1,h-1]).astype(int)
    for i,j in COCO_SKELETON:
        if i<pts_i.shape[0] and j<pts_i.shape[0]:
            cv2.line(frame,tuple(pts_i[i]),tuple(pts_i[j]),color,2,lineType=cv2.LINE_AA)
    for (x,y) in pts_i:
        cv2.circle(frame,(x,y),3,color,-1,lineType=cv2.LINE_AA)
    return frame

def draw_dynamic_parabola(points, frame, color=(255,0,255), window=15):
    if len(points) < 5:
        return frame
    pts = np.array(points[-window:])
    xs, ys = pts[:,0], pts[:,1]
    try:
        coeffs = np.polyfit(xs, ys, 2)
    except Exception:
        return frame
    poly = np.poly1d(coeffs)
    x_start = int(xs[-1])
    x_end   = min(x_start + 150, frame.shape[1])
    for x in range(x_start, x_end, 5):
        y = int(poly(x))
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            cv2.circle(frame, (x,y), 3, color, -1)
    return frame

# ---------------------------
# Video Processor
# ---------------------------
class VideoProcessor:
    def __init__(self, detector_path, pose_path=None, device="cpu",
                 ball_class_id=0, player_class_id=4, pixel_to_meter=0.01):
        self.device = device
        
        # --- 1. Load Detector Model (Custom/Tracking) ---
        with st.spinner("📥 Downloading/Loading Detector model..."):
            final_det_path = detector_path
            
            # Check if path is a URL and needs downloading
            if detector_path.startswith("http"):
                tmp_model_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pt").name
                try:
                    with open(tmp_model_path, "wb") as f:
                        f.write(requests.get(detector_path).content)
                    final_det_path = tmp_model_path
                except Exception as e:
                    st.error(f"Error downloading detector model: {e}")
                    raise
            # If it's a model name or a local file path, YOLO handles it directly.
            
            self.detector = YOLO(final_det_path)
            try:
                self.detector.to(device)
            except Exception:
                pass
        st.toast("✅ Detector model loaded successfully.", icon="🏀")

        # --- 2. Load Pose Model ---
        self.pose = None
        if pose_path:
            with st.spinner(f"📥 Loading pose model: {Path(pose_path).name if '/' in pose_path else pose_path}..."):
                
                final_pose_path = pose_path
                # YOLO handles local files, model names (like yolov8n-pose.pt), or external URLs
                try:
                    self.pose = YOLO(final_pose_path)
                    self.pose.to(device)
                    st.toast("✅ Pose model loaded.", icon="🤸")
                except Exception as e:
                    st.warning(f"Failed to load pose model at {final_pose_path}. Pose estimation will be skipped. Error: {e}")
                    self.pose = None

        self.BALL_CLASS_ID = ball_class_id
        self.PLAYER_CLASS_ID = player_class_id
        self.PIXEL_TO_METER = pixel_to_meter
        self.kalman = init_kalman()

    def process(self, input_path, output_path, progress_callback=None):
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError("Failed to open video")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        try:
            # Use imageio writer as it's generally more reliable in various environments
            writer = imageio.get_writer(str(output_path), fps=fps, codec='libx264', quality=8, pixelformat='yuv420p')
        except Exception as e:
            # Fallback suggestion if imageio fails (less likely to be needed now)
            raise RuntimeError(f"Failed to initialize imageio writer. Error: {e}")

        ball_positions = deque(maxlen=50)
        player_trajectories = {}
        player_stats = {}

        idx = 0
        t0 = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated_frame = frame.copy()

            # --- 1. Detection and Tracking ---
            det_results = self.detector.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
            if len(det_results) > 0:
                res = det_results[0]
                for box in res.boxes:
                    try:
                        xyxy = box.xyxy.cpu().numpy().reshape(-1)
                        x1,y1,x2,y2 = map(int, xyxy)
                    except Exception:
                        continue
                    cls_id = int(box.cls)
                    tid = int(box.id) if getattr(box, "id", None) is not None else 0

                    if cls_id == self.BALL_CLASS_ID:
                        # Ball tracking and Kalman filtering
                        cx,cy = (x1+x2)//2, (y1+y2)//2
                        cv2.rectangle(annotated_frame,(x1,y1),(x2,y2),(0,0,255),2)
                        draw_text_with_outline(annotated_frame, "Ball", (x1,y1-5), text_color=(0,0,255), thickness=2)
                        z = np.array([cx,cy])
                        self.kalman.predict()
                        self.kalman.update(z)
                        kx, ky = int(self.kalman.x[0]), int(self.kalman.x[1])
                        ball_positions.append([kx,ky])
                        cv2.circle(annotated_frame,(kx,ky),5,(255,255,0),-1)

                    elif cls_id == self.PLAYER_CLASS_ID:
                        # Player tracking and stats calculation
                        cx,cy = (x1+x2)//2, (y1+y2)//2
                        player_trajectories.setdefault(tid, deque(maxlen=1000)).append((cx,cy))
                        traj = player_trajectories[tid]
                        if len(traj) > 1:
                            dist = sum(math.sqrt((traj[i][0]-traj[i-1][0])**2 + (traj[i][1]-traj[i-1][1])**2)
                                       for i in range(1,len(traj)))
                            dist_m = dist * self.PIXEL_TO_METER
                            if len(traj) > 2:
                                dx = traj[-1][0]-traj[-2][0]
                                dy = traj[-1][1]-traj[-2][1]
                            else:
                                dx, dy = 0, 0
                            speed_mps = math.sqrt(dx**2 + dy**2) * self.PIXEL_TO_METER * fps
                            player_stats[tid] = {"distance": dist_m, "speed": speed_mps}

                        cv2.rectangle(annotated_frame,(x1,y1),(x2,y2),(255,0,0),2)
                        stat = player_stats.get(tid,{"distance":0,"speed":0})
                        text = f"ID:{tid} | Spd:{stat['speed']:.2f} m/s | Dist:{stat['distance']:.1f} m"
                        draw_text_with_outline(annotated_frame, text, (x1, y1-5))

            # Draw Ball Trajectory
            annotated_frame = draw_dynamic_parabola(list(ball_positions), annotated_frame)

            # --- 2. Pose Estimation (using separate model on whole frame) ---
            if self.pose is not None:
                # This runs the pose model on the entire frame, generating keypoints for all detected people.
                pose_results = self.pose(frame, verbose=False)
                if len(pose_results) > 0 and hasattr(pose_results[0], "keypoints"):
                    kpts_all = pose_results[0].keypoints.xy.cpu().numpy()
                    for xy in kpts_all:
                        # Draw the skeleton on the annotated frame
                        annotated_frame = draw_skeleton(annotated_frame, xy, color=(0,255,0))

            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            writer.append_data(frame_rgb)
            
            idx += 1
            if progress_callback and frame_count > 0:
                progress_callback(idx / frame_count)

        cap.release()
        writer.close()

        if idx == 0:
            raise RuntimeError("Video processing complete, but zero frames were processed. Input video might be empty or corrupt.")

        return {"frames": idx, "fps": fps, "elapsed_s": time.time() - t0, "player_stats": player_stats}

# ---------------------------
# NEW Streamlit UI
# ---------------------------
st.title("🏀 CourtVision: AI Basketball Analytics")
st.markdown("Upload a basketball video to get AI-powered tracking, stats, and predictions.")

# Model Constants
HUGGINGFACE_DETECTOR_URL = "https://huggingface.co/matheshvishnu/courtvision-best-pt/resolve/main/best.pt"
DEFAULT_POSE_MODEL_NAME = "yolov8n-pose.pt" 

# --- Main Layout: Two Columns for Settings ---
st.markdown("### 1. Upload & Configure")
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("##### Upload Video")
    uploaded_video = st.file_uploader("Upload a video (mp4, mov, avi, mkv)", type=["mp4","mov","avi","mkv"], label_visibility="collapsed")
    
    if uploaded_video:
        st.video(uploaded_video)
        st.caption("Original video preview.")

with col2:
    st.markdown("##### Model & Device")
    
    st.markdown("###### Detector (Ball & Player Tracking)")
    use_hf_model = st.checkbox("Use Hugging Face hosted detector model", value=True)
    
    # --- CHANGE: Make uploader always visible ---
    uploaded_model = st.file_uploader("...or Upload custom detector `.pt`", type=["pt"])
    
    st.markdown("---") # Visual separator
    
    st.markdown("###### Keypoint Estimator")
    
    # New checkbox for default pose model
    use_default_pose = st.checkbox("Use Default Keypoint Model (External Source)", value=False, 
                                   help="Enables a general-purpose keypoint estimation model, downloaded automatically by Ultralytics.")
    
    # Original uploader block, now for custom files only
    uploaded_pose = st.file_uploader("...or Upload custom keypoint model (`.pt`)", type=["pt"])
    
    device = st.selectbox("Device", ["cpu", "mps", "cuda"], index=0, help="Select 'mps' for M1/M2/M3 Mac, 'cuda' for NVIDIA GPU.")

st.markdown("---")

# --- Main Action Button ---
if st.button("🚀 Start Analysis", use_container_width=True):

    if uploaded_video:
        # --- Start processing logic ---
        with st.spinner("Starting analysis... Please wait. This may take a few minutes."):
            tmp_dir = Path(tempfile.gettempdir()) / "courtvision"
            tmp_dir.mkdir(exist_ok=True)
            in_path = tmp_dir / f"in_{int(time.time())}.mp4"
            with open(in_path, "wb") as f:
                f.write(uploaded_video.getbuffer())

            # --- Detector Path Logic ---
            detector_path = None
            if use_hf_model:
                # Priority 1: Use Hugging Face Model if checked
                detector_path = HUGGINGFACE_DETECTOR_URL
            elif uploaded_model:
                # Priority 2: Use uploaded model if HF is unchecked and a file is provided
                detector_tmp_path = tmp_dir / f"model_{uploaded_model.name}"
                with open(detector_tmp_path, "wb") as f:
                    f.write(uploaded_model.getbuffer())
                detector_path = str(detector_tmp_path)
            
            if not detector_path:
                st.error("Please select the Hugging Face model or upload a custom file.")
                st.stop()

            # --- Pose Path Logic ---
            pose_path = None
            if use_default_pose:
                # Use the reliable model name, which handles download/cache internally
                pose_path = DEFAULT_POSE_MODEL_NAME
            elif uploaded_pose:
                # Save uploaded file locally
                pose_tmp = tmp_dir / f"pose_{uploaded_pose.name}"
                with open(pose_tmp, "wb") as f:
                    f.write(uploaded_pose.getbuffer())
                pose_path = str(pose_tmp)
            
            if not pose_path:
                st.info("Keypoint estimation is disabled.")


            out_path = tmp_dir / f"out_{int(time.time())}.mp4"
            prog = st.progress(0.0)
            status = st.empty()

            def cb(frac):
                prog.progress(min(1.0, frac))
                status.text(f"Processing: {frac*100:.1f}%")

            try:
                processor = VideoProcessor(detector_path=detector_path, pose_path=pose_path, device=device)
                meta = processor.process(str(in_path), str(out_path), progress_callback=cb)
                prog.progress(1.0)
                status.empty()
                st.toast(f"✅ Analysis complete!", icon="🎉")

                # --- NEW: Professional Results Section ---
                st.markdown("---")
                st.markdown("### 2. Analysis Results")
                
                res_col1, res_col2 = st.columns([2, 1])
                
                with res_col1:
                    st.markdown("##### Processed Video")
                    if os.path.exists(out_path):
                        with open(out_path, "rb") as f:
                            video_bytes = f.read()
                        st.video(video_bytes, format="video/mp4")
                        st.download_button("⬇️ Download Processed Video", video_bytes, file_name="courtvision_output.mp4", mime="video/mp4")
                    else:
                        st.error("⚠️ Output file not found. Please try again.")
                
                with res_col2:
                    st.markdown("##### Key Metrics")
                    
                    # Calculate new metrics from returned data
                    player_stats = meta.get('player_stats', {})
                    total_players = len(player_stats)
                    
                    all_speeds = []
                    if total_players > 0:
                        for stat in player_stats.values():
                            if stat['speed'] > 0: # Only average actual movement
                                all_speeds.append(stat['speed'])
                    
                    avg_speed = np.mean(all_speeds) if all_speeds else 0
                    
                    # Display metrics
                    st.metric("Processing Time", f"{meta['elapsed_s']:.1f} s")
                    st.metric("Total Frames Processed", f"{meta['frames']}")
                    st.metric("Players Tracked", f"{total_players}")
                    st.metric("Avg. Player Speed", f"{avg_speed:.2f} m/s")

            except Exception as e:
                st.error(f"Processing failed: {e}")
                st.exception(e)
        # --- End processing logic ---

    else:
        st.warning("Please upload a video first.")