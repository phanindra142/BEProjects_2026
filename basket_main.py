from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque
import math
from filterpy.kalman import KalmanFilter

# ------------------------------
# Load YOLO models
# ------------------------------
det_model = YOLO("/Users/vmathesh/basketball/best.pt")  # custom trained model
det_model.to("mps")

pose_model = YOLO("yolov8s-pose.pt")
pose_model.to("mps")

# ------------------------------
# Config
# ------------------------------
BALL_CLASS_ID = 0
PLAYER_CLASS_ID = 4
PIXEL_TO_METER = 0.01  # scaling factor

ball_positions = deque(maxlen=50)
player_trajectories = {}   # {track_id: deque of positions}
player_stats = {}          # {track_id: {"distance": float, "speed": float}}

# ------------------------------
# Kalman Filter for ball
# ------------------------------
def init_kalman():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([0,0,0,0])  # [x, y, vx, vy]
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

kalman = init_kalman()

# ------------------------------
# Draw Text with Outline
# ------------------------------
def draw_text_with_outline(img, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX,
                           font_scale=0.6, text_color=(0,255,255), thickness=2,
                           outline_color=(0,0,0), outline_thickness=3):
    x, y = pos
    # Outline (black)
    cv2.putText(img, text, (x,y), font, font_scale, outline_color, outline_thickness, cv2.LINE_AA)
    # Main text (yellow)
    cv2.putText(img, text, (x,y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return img

# ------------------------------
# Draw Dynamic Parabola Trajectory
# ------------------------------
def draw_dynamic_parabola(points, frame, color=(255,0,255), window=15):
    if len(points) < 5:
        return frame
    
    # Recent N points only
    pts = np.array(points[-window:])
    xs, ys = pts[:,0], pts[:,1]

    # Fit parabola
    coeffs = np.polyfit(xs, ys, 2)
    poly = np.poly1d(coeffs)

    # Draw trajectory forward from last ball position
    x_start = int(xs[-1])
    x_end   = min(x_start + 150, frame.shape[1])  
    for x in range(x_start, x_end, 5):
        y = int(poly(x))
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            cv2.circle(frame, (x,y), 3, color, -1)  # magenta

    return frame

# ------------------------------
# Pose Skeleton
# ------------------------------
COCO_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (0,5),(0,6),(5,7),(7,9),
    (6,8),(8,10),(5,6),(5,11),
    (6,12),(11,12),(11,13),(13,15),
    (12,14),(14,16)
]

def draw_skeleton(frame, kpts_xy, color=(0,255,0)):
    if kpts_xy is None: return frame
    h,w = frame.shape[:2]
    pts = np.asarray(kpts_xy, dtype=np.float32)
    pts_i = np.clip(pts,0,[w-1,h-1]).astype(int)
    for i,j in COCO_SKELETON:
        if i<pts_i.shape[0] and j<pts_i.shape[0]:
            cv2.line(frame,tuple(pts_i[i]),tuple(pts_i[j]),color,2,lineType=cv2.LINE_AA)
    for (x,y) in pts_i:
        cv2.circle(frame,(x,y),3,color,-1,lineType=cv2.LINE_AA)
    return frame

# ------------------------------
# Video Processing
# ------------------------------
video_path = "/Users/vmathesh/Downloads/video_1.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

while True:
    ret, frame = cap.read()
    if not ret: break
    H,W = frame.shape[:2]
    annotated_frame = frame.copy()

    # --------------------------
    # Detection + Tracking
    # --------------------------
    det_results = det_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
    balls, players = [], []

    if len(det_results) > 0:
        res = det_results[0]
        for box in res.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy.cpu().numpy().reshape(-1))
            cls_id = int(box.cls)
            tid = int(box.id) if box.id is not None else 0

            # ------------------ Ball ------------------
            if cls_id == BALL_CLASS_ID:
                balls.append(((x1,y1,x2,y2),tid))
                cx,cy = (x1+x2)//2, (y1+y2)//2
                cv2.rectangle(annotated_frame,(x1,y1),(x2,y2),(0,0,255),2)  # red
                draw_text_with_outline(annotated_frame, "Ball", (x1,y1-5),
                                       text_color=(0,0,255), thickness=2)

                # Kalman update
                z = np.array([cx,cy])
                kalman.predict()
                kalman.update(z)
                kx, ky = int(kalman.x[0]), int(kalman.x[1])
                ball_positions.append([kx,ky])
                cv2.circle(annotated_frame,(kx,ky),5,(255,255,0),-1)  # cyan

            # ------------------ Player ------------------
            elif cls_id == PLAYER_CLASS_ID:
                players.append((x1,y1,x2,y2,tid))
                cx,cy = (x1+x2)//2, (y1+y2)//2

                # Update trajectory
                player_trajectories.setdefault(tid, deque(maxlen=1000)).append((cx,cy))

                # Compute distance & speed
                traj = player_trajectories[tid]
                if len(traj) > 1:
                    dist = sum(math.sqrt((traj[i][0]-traj[i-1][0])**2 + (traj[i][1]-traj[i-1][1])**2)
                               for i in range(1,len(traj)))
                    dist_m = dist * PIXEL_TO_METER
                    dx = traj[-1][0]-traj[-2][0]
                    dy = traj[-1][1]-traj[-2][1]
                    speed_mps = math.sqrt(dx**2 + dy**2) * PIXEL_TO_METER * fps
                    player_stats[tid] = {"distance": dist_m, "speed": speed_mps}

                # Draw box + stats
                cv2.rectangle(annotated_frame,(x1,y1),(x2,y2),(255,0,0),2)  # blue
                stat = player_stats.get(tid,{"distance":0,"speed":0})
                text = f"ID:{tid} | Spd:{stat['speed']:.2f} m/s | Dist:{stat['distance']:.1f} m"
                draw_text_with_outline(annotated_frame, text, (x1,y1-5))

    # --------------------------
    # Ball Trajectory (Dynamic Parabola)
    # --------------------------
    annotated_frame = draw_dynamic_parabola(list(ball_positions), annotated_frame)

    # --------------------------
    # Pose Estimation
    # --------------------------
    pose_results = pose_model(frame)
    if len(pose_results) > 0 and hasattr(pose_results[0], "keypoints"):
        kpts_all = pose_results[0].keypoints.xy.cpu().numpy()
        for xy in kpts_all:
            annotated_frame = draw_skeleton(annotated_frame, xy, color=(0,255,0))  # green

    # --------------------------
    # Display
    # --------------------------
    cv2.imshow("Basketball Analytics", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()