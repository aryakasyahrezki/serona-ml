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
    """
    Objective: Calculate Euclidean distance between two MediaPipe landmarks
    Parameter:
        - p1: First landmark object with .x and .y attributes (normalized coordinates)
        - p2: Second landmark object with .x and .y attributes (normalized coordinates)
    Return: Float distance value between the two points
    """
    return math.hypot((p1.x - p2.x), (p1.y - p2.y))


def get_angle(p1, p2, p3):
    """
    Objective: Calculate angle at point p2 formed by three points (p1-p2-p3) using cosine rule
    Parameter:
        - p1: First landmark (e.g., ear point)
        - p2: Vertex landmark where angle is measured (e.g., jaw corner)
        - p3: Third landmark (e.g., chin point)
    Return: Angle in degrees at p2, range [0, 180]
    """
    a = get_distance(p2, p3)
    b = get_distance(p1, p2)
    c = get_distance(p1, p3)

    if a * b == 0:
        return 0

    cos_angle = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
    cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to valid range
    return math.degrees(math.acos(cos_angle))


def extract_enhanced_features(landmarks, image_shape):
    """
    Objective: Extract 8 geometric features from MediaPipe facial landmarks for face shape classification
    Parameter:
        - landmarks: MediaPipe face mesh landmark list (468 3D points with normalized coordinates)
        - image_shape: Tuple (height, width, channels) of the input image for denormalization
    Return: List of 8 float feature values, or None if extraction fails
    """
    try:
        h, w, c = image_shape

        # --- TITIK KUNCI MEDIAPIPE (NORMALIZED) ---
        pt_top = landmarks[10]  # Top of head
        pt_bottom = landmarks[152]  # Chin tip
        pt_cheek_L = landmarks[234]  # Left cheekbone
        pt_cheek_R = landmarks[454]  # Right cheekbone
        pt_jaw_L = landmarks[58]  # Left jaw corner
        pt_jaw_R = landmarks[288]  # Right jaw corner
        pt_ear_L = landmarks[93]  # Left ear region
        pt_ear_R = landmarks[323]  # Right ear region
        pt_chin_L = landmarks[172]  # Left chin edge
        pt_chin_R = landmarks[397]  # Right chin edge

        # --- PART A: FITUR JARAK & RASIO (MANUAL) ---
        face_length = get_distance(pt_top, pt_bottom)
        face_width = get_distance(pt_cheek_L, pt_cheek_R)
        jaw_width = get_distance(pt_jaw_L, pt_jaw_R)
        chin_width = get_distance(pt_chin_L, pt_chin_R)
        forehead_width = get_distance(landmarks[103], landmarks[332])

        if face_width == 0 or jaw_width == 0:
            return None

        # Rasio geometris wajah
        ratio_len_width = face_length / face_width
        ratio_jaw_cheek = jaw_width / face_width
        ratio_forehead_jaw = forehead_width / jaw_width
        ratio_chin_jaw = chin_width / jaw_width

        # Sudut Rahang (Jawline angle)
        angle_L = get_angle(pt_ear_L, pt_jaw_L, pt_bottom)
        angle_R = get_angle(pt_ear_R, pt_jaw_R, pt_bottom)
        avg_jaw_angle = (angle_L + angle_R) / 2

        # --- PART B: FITUR GEOMETRI (OPENCV STYLE) ---
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

        if area == 0:
            return None

        circularity = (4 * np.pi * area) / (perimeter ** 2)

        hull = cv2.convexHull(oval_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / float(hull_area) if hull_area > 0 else 0

        x, y, w_rect, h_rect = cv2.boundingRect(oval_contour)
        rect_area = w_rect * h_rect
        extent = area / float(rect_area) if rect_area > 0 else 0

        return [
            ratio_len_width,  # 1. Face length to width ratio
            ratio_jaw_cheek,  # 2. Jaw to cheek width ratio
            ratio_forehead_jaw,  # 3. Forehead to jaw width ratio
            avg_jaw_angle,  # 4. Average jawline angle
            ratio_chin_jaw,  # 5. Chin to jaw width ratio
            circularity,  # 6. Face circularity (how round)
            solidity,  # 7. Face solidity (convex hull ratio)
            extent  # 8. Face extent in bounding box
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

os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)

print("=" * 60)
print("FEATURE EXTRACTION FOR FACE SHAPE CLASSIFICATION")
print("=" * 60)
print(f"Input: {INPUT_DATASET_PATH}")
print(f"Output: {OUTPUT_CSV_PATH}")
print(f"Features: {len(header) - 1}")
print("=" * 60)

with open(OUTPUT_CSV_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        total_count = 0
        failed_count = 0

        for label in LABELS:
            label_path = os.path.join(INPUT_DATASET_PATH, label)
            if not os.path.isdir(label_path):
                print(f"⚠️  Warning: Directory not found: {label_path}")
                continue

            files = []
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
                files.extend(glob.glob(os.path.join(label_path, ext)))
            files = list(set(files))

            print(f"\n📁 Processing class '{label}' ({len(files)} images)...")

            label_count = 0
            label_failed = 0

            for img_path in files:
                image = cv2.imread(img_path)
                if image is None:
                    label_failed += 1
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image_rgb)

                if results.multi_face_landmarks:
                    feats = extract_enhanced_features(
                        results.multi_face_landmarks[0].landmark,
                        image.shape
                    )

                    if feats:
                        writer.writerow([label] + feats)
                        label_count += 1
                        total_count += 1
                    else:
                        label_failed += 1
                else:
                    label_failed += 1

            success_rate = (label_count / len(files)) * 100 if len(files) > 0 else 0
            print(f"   ✓ Saved: {label_count} | ✗ Failed: {label_failed} | Success: {success_rate:.1f}%")
            failed_count += label_failed

        print("\n" + "=" * 60)
        print("✅ COMPLETED!")
        print(f"   Total processed: {total_count}")
        print(f"   Failed: {failed_count}")
        print(f"   Output file: {OUTPUT_CSV_PATH}")
        print("=" * 60)
