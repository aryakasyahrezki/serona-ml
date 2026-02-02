import cv2
import mediapipe as mp
import csv
import math
import os
import glob
import numpy as np

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def get_distance(p1, p2):
    return math.hypot((p1.x - p2.x), (p1.y - p2.y))

def get_angle(p1, p2, p3):
    # P1 (Telinga) -> P2 (Sudut Rahang) -> P3 (Dagu)
    a = get_distance(p2, p3)
    b = get_distance(p1, p2)
    c = get_distance(p1, p3)
    
    if a * b == 0: return 0
    
    cos_angle = (a**2 + b**2 - c**2) / (2 * a * b)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))

def extract_enhanced_features(landmarks, image_shape):
    try:
        h, w, c = image_shape
        
        # --- TITIK KUNCI MEDIAPIPE (NORMALIZED) ---
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

        # --- PART A: FITUR JARAK & RASIO (MANUAL) ---
        face_length = get_distance(pt_top, pt_bottom)
        face_width = get_distance(pt_cheek_L, pt_cheek_R)
        jaw_width = get_distance(pt_jaw_L, pt_jaw_R)
        chin_width = get_distance(pt_chin_L, pt_chin_R)
        forehead_width = get_distance(landmarks[103], landmarks[332])

        if face_width == 0 or jaw_width == 0: return None

        # Rasio
        ratio_len_width = face_length / face_width
        ratio_jaw_cheek = jaw_width / face_width
        ratio_forehead_jaw = forehead_width / jaw_width
        ratio_chin_jaw = chin_width / jaw_width
        
        # Sudut Rahang
        angle_L = get_angle(pt_ear_L, pt_jaw_L, pt_bottom)
        angle_R = get_angle(pt_ear_R, pt_jaw_R, pt_bottom)
        avg_jaw_angle = (angle_L + angle_R) / 2

        # --- PART B: FITUR GEOMETRI (OPENCV STYLE) ---
        # Kita ambil titik-titik pembentuk kontur wajah (Face Oval)
        # MediaPipe Face Oval Indices
        face_oval_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
        
        # Convert ke Pixel Coordinates untuk dihitung OpenCV
        oval_points = []
        for idx in face_oval_indices:
            pt = landmarks[idx]
            oval_points.append([int(pt.x * w), int(pt.y * h)])
        
        oval_contour = np.array(oval_points).reshape((-1, 1, 2))
        
        # 1. Hitung Area & Perimeter dari kontur wajah
        area = cv2.contourArea(oval_contour)
        perimeter = cv2.arcLength(oval_contour, True)
        
        if area == 0: return None

        # 2. Circularity (Kebulatan)
        # Round face -> Mendekati 0.8-0.9, Square/Heart -> Lebih rendah
        circularity = (4 * np.pi * area) / (perimeter ** 2)

        # 3. Solidity & Convex Hull
        # Heart face punya solidity rendah karena dagu lancip & dahi lebar (cekungan)
        hull = cv2.convexHull(oval_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / float(hull_area) if hull_area > 0 else 0

        # 4. Extent (Kepadatan dalam Bounding Box)
        # Square face -> Extent tinggi (mengisi kotak)
        x, y, w_rect, h_rect = cv2.boundingRect(oval_contour)
        rect_area = w_rect * h_rect
        extent = area / float(rect_area) if rect_area > 0 else 0

        return [
            ratio_len_width, 
            ratio_jaw_cheek, 
            ratio_forehead_jaw, 
            avg_jaw_angle,   
            ratio_chin_jaw,
            circularity, # <-- Fitur Baru
            solidity,    # <-- Fitur Baru
            extent       # <-- Fitur Baru
        ]
        
    except Exception as e:
        print(f"Error extraction: {e}")
        return None

# ==========================================
# 2. MAIN EXECUTION
# ==========================================

INPUT_DATASET_PATH = '../data/raw_data_30s_cropped'   
OUTPUT_CSV_PATH = '../data/processed_data/data_30s_cropped.csv' 
LABELS = ['Heart', 'Oblong', 'Oval', 'Round', 'Square'] 

mp_face_mesh = mp.solutions.face_mesh

# Header CSV (Update dengan kolom baru)
header = [
    'label', 
    'ratio_len_width', 
    'ratio_jaw_cheek', 
    'ratio_forehead_jaw', 
    'avg_jaw_angle', 
    'ratio_chin_jaw',
    'circularity', 
    'solidity', 
    'extent'
]

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Buat folder jika belum ada
os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)

with open(OUTPUT_CSV_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True, 
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        
        print(f"Processing High-Level Features to {OUTPUT_CSV_PATH}...")
        total_count = 0
        
        for label in LABELS:
            label_path = os.path.join(INPUT_DATASET_PATH, label)
            if not os.path.isdir(label_path): continue

            # Ambil semua jenis gambar
            files = []
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
                files.extend(glob.glob(os.path.join(label_path, ext)))
            files = list(set(files))
            
            print(f" -> Processing class '{label}' ({len(files)} images)...")
            
            label_count = 0
            for img_path in files:
                image = cv2.imread(img_path)
                if image is None: continue
                
                # Convert ke RGB untuk MediaPipe
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image_rgb)
                
                if results.multi_face_landmarks:
                    # Kita butuh image.shape untuk denormalisasi koordinat
                    feats = extract_enhanced_features(results.multi_face_landmarks[0].landmark, image.shape)
                    
                    if feats:
                        writer.writerow([label] + feats)
                        label_count += 1
                        total_count += 1
            
            print(f"    Saved {label_count} rows.")

        print(f"Done! Total {total_count} data processed.")