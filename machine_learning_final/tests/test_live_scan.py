import cv2
import mediapipe as mp
import joblib
import numpy as np
import pandas as pd
import time
import os
import math
import warnings

# Suppress MediaPipe/protobuf warnings for clean output
warnings.filterwarnings("ignore")
# Disable TensorFlow logging to reduce console clutter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ==========================================
# 1. SETUP & LOAD MODEL
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir, '../models/model.pkl')

print(f"📂 Model Path: {MODEL_PATH}")
print("⏳ Loading AI Models...")

try:
    # Load the model artifact containing the pipeline and metadata
    artifact = joblib.load(MODEL_PATH)

    # Load from pipeline structure (matches current model.pkl)
    pipeline = artifact['pipeline']
    label_encoder = artifact['label_encoder']
    class_names = label_encoder.classes_
    feature_names = artifact['feature_names']

    print("✅ Model Loaded Successfully.")
    print(f"   CV Accuracy : {artifact['cv_accuracy']*100:.2f}%")
    print(f"   Classes     : {list(class_names)}")
    print(f"   Features    : {feature_names}")

except Exception as e:
    print(f"❌ Error Load Model: {e}")
    exit()

# ==========================================
# 2. INIT MEDIAPIPE
# ==========================================
# Initialize the high-fidelity 3D face mesh solution
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, # Enable detailed tracking around eyes and lips
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================================
# 3. FEATURE EXTRACTION
# ==========================================

def extract_features(landmarks, w, h):
    """
    Objective: Extract 8 geometric features from MediaPipe landmarks.
    Parameter: landmarks - MediaPipe landmark list, w/h - image dimensions.
    Return: Dictionary of 8 features in correct order, or None if extraction fails.
    """
    coords = np.array([(p.x * w, p.y * h) for p in landmarks], dtype=np.float32)

    def dist(idx1, idx2):
        return np.linalg.norm(coords[idx1] - coords[idx2])

    # Calculate angles using cosine rule
    def angle(idx1, idx2, idx3):
        a = dist(idx2, idx3)
        b = dist(idx1, idx2)
        c = dist(idx1, idx3)
        if a * b == 0:
            return 0.0
        val = (a**2 + b**2 - c**2) / (2 * a * b)
        return np.degrees(np.arccos(np.clip(val, -1.0, 1.0)))

    try:
        face_length = dist(10, 152)
        face_width = dist(234, 454)
        jaw_width = dist(58, 288)
        chin_width = dist(172, 397)
        forehead_width = dist(103, 332)

        if face_width == 0 or jaw_width == 0:
            return None

        face_oval_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
        oval_points = coords[face_oval_indices].astype(np.int32)

        area = cv2.contourArea(oval_points)
        perimeter = cv2.arcLength(oval_points, True)

        if area == 0 or perimeter == 0:
            return None

        hull = cv2.convexHull(oval_points)
        hull_area = cv2.contourArea(hull)
        x, y, wr, hr = cv2.boundingRect(oval_points)

        # Return features in same order as training CSV
        return {
            'ratio_len_width': face_length / face_width,
            'ratio_jaw_cheek': jaw_width / face_width,
            'ratio_forehead_jaw': forehead_width / jaw_width,
            'avg_jaw_angle': (angle(93, 58, 152) + angle(323, 288, 152)) / 2,
            'ratio_chin_jaw': chin_width / jaw_width,
            'circularity': (4 * np.pi * area) / (perimeter ** 2),
            'solidity': area / float(hull_area) if hull_area > 0 else 0,
            'extent': area / float(wr * hr) if (wr * hr) > 0 else 0
        }

    except Exception:
        return None


def predict_face_shape(features_dict):
    """
    Objective: Run ML pipeline prediction from extracted features.
    Parameter: features_dict - dictionary of 8 geometric features.
    Return: Tuple (shape_label, confidence_percent) or (None, 0) if fails.
    """
    try:
        # Convert to DataFrame with correct column order (matches training)
        X_input = pd.DataFrame([features_dict], columns=feature_names)

        # Pipeline handles poly → scale → select → predict automatically
        pred_idx = pipeline.predict(X_input)[0]
        prob = np.max(pipeline.predict_proba(X_input))

        shape_label = class_names[pred_idx]
        return shape_label, prob * 100

    except Exception as e:
        print(f"⚠️ Prediction error: {e}")
        return None, 0


def analyze_skintone(image, landmarks, w, h):
    """
    Objective: Detect skin tone by analyzing cheek ROIs in LAB color space.
    Parameter: image - BGR numpy array, landmarks - MediaPipe landmarks, w/h - dimensions.
    Return: Skintone string ('Fair Light', 'Medium Tan', or 'Deep').
    """
    try:
        rois = [[330, 347, 346, 352], [101, 118, 117, 123]]
        mean_colors = []

        for roi in rois:
            points = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in roi])
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [points], 255)
            mean_val = cv2.mean(image, mask=mask)[:3]
            mean_colors.append(mean_val)

        avg_bgr = np.mean(mean_colors, axis=0)
        lab = cv2.cvtColor(np.uint8([[avg_bgr]]), cv2.COLOR_BGR2LAB)[0][0]
        L = lab[0]

        if L > 165:
            return "Fair Light"
        elif L > 105:
            return "Medium Tan"
        else:
            return "Deep"

    except Exception:
        return "Unknown"

# ==========================================
# 4. CAMERA SETUP
# ==========================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# State Machine
STATE_SCANNING = 0
STATE_RESULT_LIVE = 1
current_state = STATE_SCANNING

stable_frames = 0
REQUIRED_FRAMES = 5
final_shape = ""
final_tone = ""
process_time_ms = 0.0
prev_time = 0
t_result_start = 0

print("\n🚀 Camera Started. Look at the camera and hold still...")
print("   Press [Q] to quit\n")

# ==========================================
# 5. MAIN LOOP
# ==========================================
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # FPS Counter
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    # ==========================================
    # STATE: SCANNING
    # ==========================================
    if current_state == STATE_SCANNING:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        status_text = "Looking for face..."
        color_status = (0, 0, 255)  # Red

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Check face is centered (not tilted)
            nose_x = landmarks[1].x
            left_cheek_x = landmarks[234].x
            right_cheek_x = landmarks[454].x
            dist_L = nose_x - left_cheek_x
            dist_R = right_cheek_x - nose_x
            ratio = dist_L / dist_R if dist_R != 0 else 0
            is_centered = 0.7 < ratio < 1.3

            if is_centered:
                stable_frames += 1
                progress = min(stable_frames / REQUIRED_FRAMES, 1.0)

                # Progress bar at bottom
                bar_w = int(w * progress)
                cv2.rectangle(frame, (0, h - 10), (bar_w, h), (0, 255, 0), -1)
                status_text = f"ANALYZING... {int(progress * 100)}%"
                color_status = (0, 255, 0)  # Green

                if stable_frames >= REQUIRED_FRAMES:
                    t_start_proc = time.perf_counter()

                    # Extract features
                    features_dict = extract_features(landmarks, w, h)

                    if features_dict is not None:
                        # ✅ Predict using pipeline
                        shape_label, confidence = predict_face_shape(features_dict)

                        if shape_label is not None:
                            final_shape = f"{shape_label} ({confidence:.0f}%)"
                        else:
                            final_shape = "Unknown"
                    else:
                        final_shape = "Unknown"

                    # Skintone
                    final_tone = analyze_skintone(frame, landmarks, w, h)

                    # Measure total processing time
                    process_time_ms = (time.perf_counter() - t_start_proc) * 1000

                    print(f"✅ Result     : {final_shape}")
                    print(f"   Skin Tone  : {final_tone}")
                    print(f"   Proc Time  : {process_time_ms:.2f} ms")

                    # Transition to result state
                    current_state = STATE_RESULT_LIVE
                    t_result_start = time.time()

            else:
                stable_frames = 0
                status_text = "Please Look Straight"
                color_status = (0, 165, 255)  # Orange

        cv2.putText(frame, status_text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color_status, 2)

    # ==========================================
    # STATE: RESULT (with auto-rescan)
    # ==========================================
    elif current_state == STATE_RESULT_LIVE:
        # Semi-transparent black overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (460, 190), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        cv2.putText(frame, "ANALYSIS COMPLETE", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(frame, f"SHAPE : {final_shape}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"SKIN  : {final_tone}", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, f"Speed : {process_time_ms:.1f} ms", (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Countdown timer
        elapsed = time.time() - t_result_start
        remaining = max(0, 2.0 - elapsed)
        cv2.putText(frame, f"Auto-refresh in: {remaining:.1f}s", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        # Auto-rescan after 1.5 seconds
        if elapsed >= 1.5:
            current_state = STATE_SCANNING
            stable_frames = 0

    # FPS display
    cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Serona AI - Face Shape Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n👋 Camera closed. Goodbye!")