"""
test_robustness.py

Objective: Test model robustness by evaluating performance on augmented/perturbed images.
           Measures how well the model handles real-world variations in input quality.

Note: This test uses OpenCV and MediaPipe to process images and extract features,
      then tests if the model prediction stays consistent under various conditions.
"""

import os
import sys
import cv2
import math
import joblib
import numpy as np
import pandas as pd
import warnings
from pathlib import Path

# Suppress MediaPipe/protobuf deprecation warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = '../models/model.pkl'
TEST_DATA_DIR = '../data/raw_data_30s_cropped'
LABELS = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
IMAGES_PER_CLASS = 5  # Test 5 images per class (25 total)

# ==========================================
# MEDIAPIPE SETUP
# ==========================================
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️  MediaPipe not available - using feature perturbation instead")

# ==========================================
# FEATURE EXTRACTION (same as process_raw_data.py)
# ==========================================

def get_distance(p1, p2):
    """
    Objective: Calculate Euclidean distance between two MediaPipe landmarks
    Parameter: p1, p2 - MediaPipe landmarks with .x and .y attributes
    Return: Float distance value
    """
    return math.hypot((p1.x - p2.x), (p1.y - p2.y))

def get_angle(p1, p2, p3):
    """
    Objective: Calculate angle at p2 formed by three landmarks
    Parameter: p1, p2, p3 - Three MediaPipe landmarks
    Return: Angle in degrees at p2
    """
    a = get_distance(p2, p3)
    b = get_distance(p1, p2)
    c = get_distance(p1, p3)
    if a * b == 0:
        return 0
    cos_angle = (a**2 + b**2 - c**2) / (2 * a * b)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))

def extract_features(landmarks, image_shape):
    """
    Objective: Extract 8 geometric features from MediaPipe face mesh landmarks
    Parameter: landmarks - MediaPipe landmark list, image_shape - (h, w, c)
    Return: Dictionary with 8 features, or None if extraction fails
    """
    try:
        h, w, c = image_shape

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

        face_length = get_distance(pt_top, pt_bottom)
        face_width = get_distance(pt_cheek_L, pt_cheek_R)
        jaw_width = get_distance(pt_jaw_L, pt_jaw_R)
        chin_width = get_distance(pt_chin_L, pt_chin_R)
        forehead_width = get_distance(pt_forehead_L, pt_forehead_R)

        if face_width == 0 or jaw_width == 0:
            return None

        ratio_len_width = face_length / face_width
        ratio_jaw_cheek = jaw_width / face_width
        ratio_forehead_jaw = forehead_width / jaw_width
        ratio_chin_jaw = chin_width / jaw_width

        angle_L = get_angle(pt_ear_L, pt_jaw_L, pt_bottom)
        angle_R = get_angle(pt_ear_R, pt_jaw_R, pt_bottom)
        avg_jaw_angle = (angle_L + angle_R) / 2

        face_oval_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
        oval_points = [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in face_oval_indices]
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

# ==========================================
# IMAGE PERTURBATION FUNCTIONS
# ==========================================

def apply_brightness(image, factor):
    """
    Objective: Adjust image brightness
    Parameter: image - BGR numpy array, factor - float (>1 = brighter, <1 = darker)
    Return: Brightness-adjusted image
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_blur(image, kernel_size):
    """
    Objective: Apply Gaussian blur to simulate out-of-focus images
    Parameter: image - BGR numpy array, kernel_size - int (blur intensity)
    Return: Blurred image
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_noise(image, noise_level):
    """
    Objective: Add Gaussian noise to simulate poor camera quality
    Parameter: image - BGR numpy array, noise_level - int (noise intensity)
    Return: Noisy image
    """
    noise = np.random.normal(0, noise_level, image.shape).astype(np.int16)
    noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy

def apply_rotation(image, angle):
    """
    Objective: Rotate image by given angle to simulate head tilt
    Parameter: image - BGR numpy array, angle - float in degrees
    Return: Rotated image
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def apply_resize(image, scale):
    """
    Objective: Resize image to simulate different camera resolutions
    Parameter: image - BGR numpy array, scale - float (0.5 = half, 2.0 = double)
    Return: Resized image (original dimensions preserved via interpolation)
    """
    h, w = image.shape[:2]
    small = cv2.resize(image, (int(w * scale), int(h * scale)))
    return cv2.resize(small, (w, h))

# ==========================================
# INFERENCE FUNCTION
# ==========================================

def predict_from_image(image, pipeline, feature_names, face_mesh_instance):
    """
    Objective: Run complete inference pipeline on an image
    Parameter:
        - image: BGR numpy array
        - pipeline: trained sklearn pipeline
        - feature_names: list of feature names
        - face_mesh_instance: MediaPipe FaceMesh instance
    Return: Tuple (prediction_label, confidence) or (None, 0) if failed
    """
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh_instance.process(rgb_img)

    if not results.multi_face_landmarks:
        return None, 0

    landmarks = results.multi_face_landmarks[0].landmark
    features = extract_features(landmarks, image.shape)

    if features is None:
        return None, 0

    X_input = pd.DataFrame([features], columns=feature_names)
    pred_idx = pipeline.predict(X_input)[0]
    confidence = np.max(pipeline.predict_proba(X_input))

    return pred_idx, confidence

# ==========================================
# MAIN ROBUSTNESS TEST
# ==========================================

def main():
    print("=" * 65)
    print("MODEL ROBUSTNESS TESTING")
    print("=" * 65)

    if not MEDIAPIPE_AVAILABLE:
        print("❌ MediaPipe required for full robustness testing")
        print("   Run: pip install mediapipe")
        sys.exit(1)

    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at: {MODEL_PATH}")
        sys.exit(1)

    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact['pipeline']
    label_encoder = artifact['label_encoder']
    feature_names = artifact['feature_names']

    print(f"\n✅ Model loaded (CV accuracy: {artifact['cv_accuracy']*100:.2f}%)")

    # Define perturbation tests
    perturbations = {
        'Original (Baseline)':    lambda img: img,
        'Bright (+30%)':          lambda img: apply_brightness(img, 1.3),
        'Dark (-30%)':            lambda img: apply_brightness(img, 0.7),
        'Slight Blur (3x3)':      lambda img: apply_blur(img, 3),
        'Heavy Blur (7x7)':       lambda img: apply_blur(img, 7),
        'Low Noise (σ=10)':       lambda img: apply_noise(img, 10),
        'High Noise (σ=25)':      lambda img: apply_noise(img, 25),
        'Rotate +5°':             lambda img: apply_rotation(img, 5),
        'Rotate -5°':             lambda img: apply_rotation(img, -5),
        'Rotate +10°':            lambda img: apply_rotation(img, 10),
        'Low Resolution (50%)':   lambda img: apply_resize(img, 0.5),
    }

    # Collect test images
    print(f"\n📁 Collecting test images...")
    test_images = []

    for label in LABELS:
        label_path = Path(TEST_DATA_DIR) / label
        if not label_path.exists():
            print(f"   ⚠️  Folder not found: {label_path}")
            continue

        images = list(label_path.glob("*.jpg")) + \
                 list(label_path.glob("*.jpeg")) + \
                 list(label_path.glob("*.png"))
        images = images[:IMAGES_PER_CLASS]

        for img_path in images:
            test_images.append((str(img_path), label))

        print(f"   ✅ {label}: {len(images)} images")

    if not test_images:
        print("❌ No test images found!")
        sys.exit(1)

    print(f"\n📊 Total test images: {len(test_images)}")
    print(f"   Perturbation types: {len(perturbations)}")
    print(f"   Total tests: {len(test_images) * len(perturbations)}")

    # Run tests with MediaPipe
    results_summary = {}

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        for perturb_name, perturb_fn in perturbations.items():
            print(f"\n🔄 Testing: {perturb_name}...")

            correct = 0
            total = 0
            failed = 0
            confidences = []

            for img_path, true_label in test_images:
                # Load image
                image = cv2.imread(img_path)
                if image is None:
                    failed += 1
                    continue

                # Apply perturbation
                perturbed = perturb_fn(image)

                # Predict
                pred_idx, confidence = predict_from_image(
                    perturbed, pipeline, feature_names, face_mesh
                )

                if pred_idx is None:
                    failed += 1
                    continue

                pred_label = label_encoder.classes_[pred_idx]
                total += 1
                confidences.append(confidence)

                if pred_label == true_label:
                    correct += 1

            if total > 0:
                accuracy = correct / total * 100
                avg_confidence = np.mean(confidences) * 100
                results_summary[perturb_name] = {
                    'accuracy': accuracy,
                    'correct': correct,
                    'total': total,
                    'failed': failed,
                    'avg_confidence': avg_confidence
                }

                status = "✅" if accuracy >= 65 else "⚠️" if accuracy >= 50 else "❌"
                print(f"   {status} Accuracy: {accuracy:.1f}% ({correct}/{total}) | "
                      f"Confidence: {avg_confidence:.1f}% | Failed: {failed}")
            else:
                results_summary[perturb_name] = {
                    'accuracy': 0, 'correct': 0,
                    'total': 0, 'failed': failed,
                    'avg_confidence': 0
                }
                print(f"   ❌ All predictions failed!")

    # ==========================================
    # RESULTS SUMMARY
    # ==========================================
    print(f"\n{'='*65}")
    print(f"ROBUSTNESS TEST RESULTS SUMMARY")
    print(f"{'='*65}")

    baseline_acc = results_summary.get('Original (Baseline)', {}).get('accuracy', 0)

    print(f"\n{'Perturbation':<30} {'Accuracy':>10} {'vs Baseline':>12} {'Avg Conf':>10}")
    print(f"{'-'*65}")

    for name, result in results_summary.items():
        acc = result['accuracy']
        conf = result['avg_confidence']
        diff = acc - baseline_acc
        diff_str = f"+{diff:.1f}%" if diff >= 0 else f"{diff:.1f}%"
        status = "✅" if acc >= 65 else "⚠️" if acc >= 50 else "❌"
        print(f"{status} {name:<28} {acc:>9.1f}% {diff_str:>12} {conf:>9.1f}%")

    # ==========================================
    # ROBUSTNESS SCORE
    # ==========================================
    print(f"\n📊 ROBUSTNESS ANALYSIS:")

    accuracies = [r['accuracy'] for r in results_summary.values()]
    mean_acc = np.mean(accuracies)
    min_acc = np.min(accuracies)
    std_acc = np.std(accuracies)
    drop_from_baseline = baseline_acc - min_acc

    print(f"   Baseline Accuracy   : {baseline_acc:.1f}%")
    print(f"   Mean Accuracy       : {mean_acc:.1f}%")
    print(f"   Min Accuracy        : {min_acc:.1f}%")
    print(f"   Std Dev             : {std_acc:.1f}%")
    print(f"   Max Drop            : {drop_from_baseline:.1f}% (vs baseline)")

    # Overall robustness rating
    if drop_from_baseline < 5:
        rating = "🟢 HIGHLY ROBUST"
        note = "Model maintains performance under various conditions"
    elif drop_from_baseline < 15:
        rating = "🟡 MODERATELY ROBUST"
        note = "Minor performance drop under challenging conditions"
    elif drop_from_baseline < 30:
        rating = "🟠 SOMEWHAT ROBUST"
        note = "Noticeable performance drop - consider preprocessing"
    else:
        rating = "🔴 NOT ROBUST"
        note = "Significant performance drop - model needs improvement"

    print(f"\n🏆 ROBUSTNESS RATING: {rating}")
    print(f"   {note}")

    # ==========================================
    # RECOMMENDATIONS
    # ==========================================
    print(f"\n💡 RECOMMENDATIONS:")

    for name, result in results_summary.items():
        acc = result['accuracy']
        drop = baseline_acc - acc
        if drop > 15:
            print(f"   ⚠️  {name}: {drop:.1f}% drop → Consider adding preprocessing")

    print(f"\n{'='*65}")
    print(f"✅ ROBUSTNESS TESTING COMPLETE")
    print(f"   Overall Rating: {rating}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()