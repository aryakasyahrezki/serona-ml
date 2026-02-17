import os
import cv2
import joblib
import uvicorn
import numpy as np
import mediapipe as mp
import gc
import time 
import warnings
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Serona AI API")

# ==========================================
# 0. SILENCE WARNINGS (Clean Logs for Azure)
# ==========================================
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# ==========================================
# 1. SECURITY: CORS CONFIGURATION
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. LOAD MODEL & ARTIFACTS
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, '../models/model.pkl')

try:
    artifact = joblib.load(model_path)
    
    # ✅ NEW: Load from pipeline structure
    pipeline = artifact['pipeline']
    label_encoder = artifact['label_encoder']
    class_names = label_encoder.classes_
    feature_names = artifact['feature_names']
    
    print("="*60)
    print("✅ Serona AI Model Loaded Successfully")
    print(f"   Model: Logistic Regression with RFECV")
    print(f"   Classes: {list(class_names)}")
    print(f"   Features: {len(feature_names)}")
    print(f"   Random Seed: {artifact['random_seed']}")
    print(f"   CV Accuracy: {artifact['cv_accuracy']*100:.2f}%")
    print(f"   Selected Features: {artifact['metadata']['n_features_selected']}")
    print("="*60)
    
except Exception as e:
    print(f"❌ Critical Error: {e}")
    exit()

# Init MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5
)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def get_distance(p1, p2):
    """
    Objective: Calculate Euclidean distance between two landmarks
    Parameter: p1, p2 - MediaPipe landmarks with .x and .y attributes
    Return: Float distance value
    """
    return np.hypot((p1.x - p2.x), (p1.y - p2.y))

def get_angle(p1, p2, p3):
    """
    Objective: Calculate angle at p2 formed by p1-p2-p3
    Parameter: p1, p2, p3 - Three MediaPipe landmarks
    Return: Angle in degrees at p2
    """
    a = get_distance(p2, p3)
    b = get_distance(p1, p2)
    c = get_distance(p1, p3)
    
    if a * b == 0: 
        return 0.0
    
    cos_angle = (a**2 + b**2 - c**2) / (2 * a * b)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

def extract_features(landmarks, image_shape):
    """
    Objective: Extract 8 geometric features matching training data format
    Parameter: landmarks - MediaPipe face mesh landmark list, image_shape - (height, width, channels)
    Return: Dictionary with 8 features in correct order, or None if extraction fails
    """
    try:
        h, w, c = image_shape
        
        # Key landmarks
        pt_top = landmarks[10]
        pt_bottom = landmarks[152]
        pt_cheek_L = landmarks[234]
        pt_cheek_R = landmarks[454]
        pt_jaw_L = landmarks[58]
        pt_jaw_R = landmarks[288]
        pt_ear_L = landmarks[93]
        pt_ear_R = landmarks[323]
        pt_chin_L = landmarks[172]
        pt_chin_R = landmarks[397]
        pt_forehead_L = landmarks[103]
        pt_forehead_R = landmarks[332]
        
        # Distance measurements
        face_length = get_distance(pt_top, pt_bottom)
        face_width = get_distance(pt_cheek_L, pt_cheek_R)
        jaw_width = get_distance(pt_jaw_L, pt_jaw_R)
        chin_width = get_distance(pt_chin_L, pt_chin_R)
        forehead_width = get_distance(pt_forehead_L, pt_forehead_R)
        
        if face_width == 0 or jaw_width == 0:
            return None
        
        # Ratios
        ratio_len_width = face_length / face_width
        ratio_jaw_cheek = jaw_width / face_width
        ratio_forehead_jaw = forehead_width / jaw_width
        ratio_chin_jaw = chin_width / jaw_width
        
        # Jaw angle
        angle_L = get_angle(pt_ear_L, pt_jaw_L, pt_bottom)
        angle_R = get_angle(pt_ear_R, pt_jaw_R, pt_bottom)
        avg_jaw_angle = (angle_L + angle_R) / 2
        
        # OpenCV features
        face_oval_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
        
        oval_points = []
        for idx in face_oval_indices:
            pt = landmarks[idx]
            oval_points.append([int(pt.x * w), int(pt.y * h)])
        
        oval_contour = np.array(oval_points).reshape((-1, 1, 2))
        
        area = cv2.contourArea(oval_contour)
        perimeter = cv2.arcLength(oval_contour, True)
        
        if area == 0 or perimeter == 0:
            return None
        
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        
        hull = cv2.convexHull(oval_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / float(hull_area) if hull_area > 0 else 0
        
        x, y, w_rect, h_rect = cv2.boundingRect(oval_contour)
        rect_area = w_rect * h_rect
        extent = area / float(rect_area) if rect_area > 0 else 0
        
        # Return in EXACT order as training CSV
        return {
            'ratio_len_width': ratio_len_width,
            'ratio_jaw_cheek': ratio_jaw_cheek,
            'ratio_forehead_jaw': ratio_forehead_jaw,
            'avg_jaw_angle': avg_jaw_angle,
            'ratio_chin_jaw': ratio_chin_jaw,
            'circularity': circularity,
            'solidity': solidity,
            'extent': extent
        }
        
    except Exception:
        return None

def analyze_skintone(image, landmarks, w, h):
    """
    Objective: Detect skin tone category by analyzing specific cheek ROIs in the LAB color space
    Parameter: image - BGR numpy array, landmarks - MediaPipe landmarks, w/h - image dimensions
    Return: String representation of skintone (Fair Light, Medium Tan, or Deep)
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
        
        if L > 165: return "Fair Light"
        elif L > 105: return "Medium Tan"
        else: return "Deep"
    except Exception:
        return "Unknown"

# ==========================================
# 4. ENDPOINTS
# ==========================================

@app.get("/")
def home():
    """
    Objective: Health check endpoint to verify server status
    Parameter: None
    Return: JSON indicating server is online
    """
    return {"status": "online", "service": "Serona AI", "location": "Cloud"}

@app.post("/predict")
async def predict_face(file: UploadFile = File(...)):
    """
    Objective: Predict face shape and skin tone from uploaded image
    Parameter: file - UploadFile containing face image
    Return: JSON with EXACT same format as old version for Android compatibility
    """
    # ✅ START: Total request timer
    request_start_time = time.perf_counter()
    
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid file type"})

    img = None
    try:
        # ===== TIMING CHECKPOINT 1: File Reading =====
        t1 = time.perf_counter()
        contents = await file.read()
        t2 = time.perf_counter()
        file_read_ms = (t2 - t1) * 1000
        
        # ===== TIMING CHECKPOINT 2: Image Decoding =====
        t3 = time.perf_counter()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None: 
            raise ValueError("Could not decode image")
            
        h, w, _ = img.shape
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t4 = time.perf_counter()
        decode_ms = (t4 - t3) * 1000
        
        # ===== TIMING CHECKPOINT 3: MediaPipe Processing =====
        t5 = time.perf_counter()
        results = face_mesh.process(rgb_img)
        t6 = time.perf_counter()
        mediapipe_ms = (t6 - t5) * 1000

        if not results.multi_face_landmarks:
            return {"status": "failed", "message": "No face detected"}

        landmarks = results.multi_face_landmarks[0].landmark
        
        # ===== TIMING CHECKPOINT 4: Feature Extraction =====
        t7 = time.perf_counter()
        features_dict = extract_features(landmarks, img.shape)
        t8 = time.perf_counter()
        feature_extraction_ms = (t8 - t7) * 1000
        
        if features_dict is None: 
            return {"status": "failed", "message": "Alignment error"}

        # Convert to DataFrame for pipeline
        import pandas as pd
        X_input = pd.DataFrame([features_dict], columns=feature_names)

        # ===== TIMING CHECKPOINT 5: ML Inference =====
        t_start = time.perf_counter()
        pred_idx = pipeline.predict(X_input)[0]
        prob = np.max(pipeline.predict_proba(X_input))
        t_end = time.perf_counter()
        inference_ms = (t_end - t_start) * 1000

        final_shape = class_names[pred_idx]
        
        # ===== TIMING CHECKPOINT 6: Skintone Analysis =====
        t9 = time.perf_counter()
        final_tone = analyze_skintone(img, landmarks, w, h)
        t10 = time.perf_counter()
        skintone_ms = (t10 - t9) * 1000
        
        # ===== TOTAL REQUEST TIME =====
        request_end_time = time.perf_counter()
        total_request_ms = (request_end_time - request_start_time) * 1000

        # ===== DETAILED LOGGING =====
        print("="*60)
        print(f"[REQUEST BREAKDOWN]")
        print(f"  1. File Read        : {file_read_ms:6.2f} ms ({file_read_ms/total_request_ms*100:5.1f}%)")
        print(f"  2. Image Decode     : {decode_ms:6.2f} ms ({decode_ms/total_request_ms*100:5.1f}%)")
        print(f"  3. MediaPipe        : {mediapipe_ms:6.2f} ms ({mediapipe_ms/total_request_ms*100:5.1f}%)  ⚠️")
        print(f"  4. Feature Extract  : {feature_extraction_ms:6.2f} ms ({feature_extraction_ms/total_request_ms*100:5.1f}%)")
        print(f"  5. ML Inference     : {inference_ms:6.2f} ms ({inference_ms/total_request_ms*100:5.1f}%)")
        print(f"  6. Skintone Analysis: {skintone_ms:6.2f} ms ({skintone_ms/total_request_ms*100:5.1f}%)")
        print(f"  " + "-"*56)
        print(f"  TOTAL REQUEST TIME  : {total_request_ms:6.2f} ms")
        print(f"  Result              : {final_shape} ({prob*100:.0f}%)")
        print("="*60)

        # ✅ RETURN with timing info
        return {
            "status": "success",
            "shape": f"{final_shape} ({prob*100:.0f}%)",
            "skintone": final_tone,
            "server_inference_ms": str(round(inference_ms, 2)),
            "total_request_ms": str(round(total_request_ms, 2)),  # ← New
            "breakdown": {  # ← New (optional, for debugging)
                "file_read_ms": round(file_read_ms, 2),
                "decode_ms": round(decode_ms, 2),
                "mediapipe_ms": round(mediapipe_ms, 2),
                "feature_extraction_ms": round(feature_extraction_ms, 2),
                "ml_inference_ms": round(inference_ms, 2),
                "skintone_ms": round(skintone_ms, 2)
            }
        }

    except Exception as e:
        print(f"[ERROR]: {str(e)}")
        return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})
    
    finally:
        if img is not None: 
            del img
        await file.close()
        gc.collect()
        
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)