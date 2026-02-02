import os
import cv2
import joblib
import uvicorn
import numpy as np
import mediapipe as mp
import gc  # Garbage Collector untuk pembersihan memori
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="Serona AI API")

# ==========================================
# 1. LOAD MODEL & ARTIFACTS
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, '../models/final_model.pkl')

try:
    artifact = joblib.load(model_path)
    model = artifact['model']
    poly = artifact['poly']
    scaler = artifact['scaler']
    selector = artifact['selector']
    class_names = artifact['class_names']
    feature_names_original = artifact['features_used']
    print("✅ Serona AI Model Loaded Successfully")
except Exception as e:
    print(f"❌ Critical Error: {e}")
    exit()

# Init MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def extract_features_optimized(landmarks, w, h):
    coords = np.array([(p.x * w, p.y * h) for p in landmarks], dtype=np.float32)

    def dist(idx1, idx2):
        return np.linalg.norm(coords[idx1] - coords[idx2])

    def angle(idx1, idx2, idx3):
        a = dist(idx2, idx3)
        b = dist(idx1, idx2)
        c = dist(idx1, idx3)
        if a * b == 0: return 0.0
        val = (a**2 + b**2 - c**2) / (2 * a * b)
        return np.degrees(np.arccos(np.clip(val, -1.0, 1.0)))

    try:
        face_length = dist(10, 152)
        face_width = dist(234, 454)
        jaw_width = dist(58, 288)
        chin_width = dist(172, 397)
        forehead_width = dist(103, 332)

        face_oval_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
                             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        oval_points = coords[face_oval_indices].astype(np.int32)
        
        area = cv2.contourArea(oval_points)
        perimeter = cv2.arcLength(oval_points, True)
        hull = cv2.convexHull(oval_points)
        x, y, wr, hr = cv2.boundingRect(oval_points)
        
        raw_feats = {
            'circularity': (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0,
            'solidity': area / float(cv2.contourArea(hull)) if cv2.contourArea(hull) > 0 else 0,
            'extent': area / float(wr * hr) if (wr * hr) > 0 else 0,
            'ratio_len_width': face_length / face_width if face_width > 0 else 0,
            'ratio_jaw_cheek': jaw_width / face_width if face_width > 0 else 0,
            'ratio_forehead_jaw': forehead_width / jaw_width if jaw_width > 0 else 0,
            'ratio_chin_jaw': chin_width / jaw_width if jaw_width > 0 else 0,
            'avg_jaw_angle': (angle(93, 58, 152) + angle(323, 288, 152)) / 2
        }
        
        input_values = [raw_feats[name] for name in feature_names_original]
        return np.array([input_values])
    except Exception:
        return None

def analyze_skintone(image, landmarks, w, h):
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
    else: return "Dark"

@app.get("/")
def home():
    return {"message": "Serona AI Server is Online", "region": "Localhost"}

@app.post("/predict")
async def predict_face(file: UploadFile = File(...)):
    img = None
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image")
            
        h, w, _ = img.shape
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_img)

        if not results.multi_face_landmarks:
            return JSONResponse(status_code=200, content={"status": "failed", "message": "No face detected"})

        landmarks = results.multi_face_landmarks[0].landmark
        X_raw = extract_features_optimized(landmarks, w, h)
        
        if X_raw is None:
            return {"status": "failed", "message": "Alignment error"}

        X_poly = poly.transform(X_raw)
        X_scaled = scaler.transform(X_poly)
        X_sel = selector.transform(X_scaled)
        
        pred_idx = model.predict(X_sel)[0]
        prob = np.max(model.predict_proba(X_sel))
        final_shape = class_names[pred_idx]
        final_tone = analyze_skintone(img, landmarks, w, h)

        return {
            "status": "success",
            "shape": f"{final_shape} ({prob*100:.0f}%)",
            "skintone": final_tone
        }

    except Exception as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})
    
    finally:
        if img is not None:
            del img # Hapus gambar dari memori
        await file.close()
        gc.collect() # Paksa pembersihan RAM

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)