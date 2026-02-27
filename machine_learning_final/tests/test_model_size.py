"""
test_model_size.py

Objective: Analyze the size of the saved ML model artifact (.pkl file)
           and report detailed breakdown of its components.
"""

import os
import sys
import joblib
import pickle
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = '../models/model.pkl'

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_file_size(path):
    """
    Objective: Get file size in multiple units
    Parameter: path - file path string
    Return: Dictionary with size in bytes, KB, MB
    """
    size_bytes = os.path.getsize(path)
    return {
        'bytes': size_bytes,
        'kb': size_bytes / 1024,
        'mb': size_bytes / (1024 * 1024)
    }

def get_object_size(obj):
    """
    Objective: Get in-memory size of a Python object
    Parameter: obj - any Python object
    Return: Dictionary with size in bytes, KB, MB
    """
    size_bytes = sys.getsizeof(pickle.dumps(obj))
    return {
        'bytes': size_bytes,
        'kb': size_bytes / 1024,
        'mb': size_bytes / (1024 * 1024)
    }

def format_size(size_dict):
    """
    Objective: Format size dictionary into readable string
    Parameter: size_dict - dict with bytes, kb, mb keys
    Return: Formatted string
    """
    if size_dict['mb'] >= 1:
        return f"{size_dict['mb']:.2f} MB"
    elif size_dict['kb'] >= 1:
        return f"{size_dict['kb']:.2f} KB"
    else:
        return f"{size_dict['bytes']} bytes"

# ==========================================
# MAIN ANALYSIS
# ==========================================

def main():
    print("=" * 60)
    print("MODEL SIZE ANALYSIS")
    print("=" * 60)

    # Check model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at: {MODEL_PATH}")
        print("   Please make sure model.pkl exists in the models/ folder")
        sys.exit(1)

    # ==========================================
    # 1. FILE SIZE ON DISK
    # ==========================================
    file_size = get_file_size(MODEL_PATH)

    print(f"\n📁 FILE SIZE ON DISK:")
    print(f"   Path  : {os.path.abspath(MODEL_PATH)}")
    print(f"   Size  : {format_size(file_size)}")
    print(f"   Exact : {file_size['bytes']:,} bytes")

    # ==========================================
    # 2. LOAD MODEL & METADATA
    # ==========================================
    print(f"\n⏳ Loading model...")
    artifact = joblib.load(MODEL_PATH)

    print(f"\n📊 MODEL METADATA:")
    print(f"   CV F1-Macro      : {artifact['metrics']['cv_f1_macro'] * 100:.2f}%")
    print(f"   CV Std Dev       : {artifact.get('cv_std', 0) * 100:.2f}%")
    print(f"   Random Seed      : {artifact['random_seed']}")
    print(f"   Original Features: {artifact['metadata']['n_features_original']}")
    print(f"   Selected Features: {artifact['metadata']['n_features_selected']}")
    print(f"   Classes          : {artifact['metadata']['classes']}")
    print(f"   Training Samples : {artifact['metadata']['n_samples']}")

    # ==========================================
    # 3. COMPONENT SIZE BREAKDOWN
    # ==========================================
    # identifies which part of the pipeline consumes the most memory
    print(f"\n🔍 COMPONENT SIZE BREAKDOWN (in-memory):")
    print(f"   {'Component':<25} {'Size':>10}")
    print(f"   {'-'*35}")

    total_component_size = 0
    components = {
        'pipeline (full)': artifact['pipeline'],
        'label_encoder': artifact['label_encoder'],
    }

    # Break down pipeline components
    pipeline = artifact['pipeline']
    pipeline_components = {
        'poly_features': pipeline.named_steps['poly'],
        'scaler': pipeline.named_steps['scaler'],
        'rfecv_selector': pipeline.named_steps['rfecv'],
    }

    all_components = {**components, **pipeline_components}

    for name, obj in all_components.items():
        size = get_object_size(obj)
        total_component_size += size['bytes']
        print(f"   {name:<25} {format_size(size):>10}")

    print(f"   {'-'*35}")
    total_size = {'bytes': total_component_size, 'kb': total_component_size/1024, 'mb': total_component_size/(1024*1024)}
    print(f"   {'TOTAL (in-memory)':<25} {format_size(total_size):>10}")

    # ==========================================
    # 4. PIPELINE DETAILS
    # ==========================================
    print(f"\n⚙️  PIPELINE CONFIGURATION:")
    print(f"   Step 1 - PolynomialFeatures:")
    poly = pipeline.named_steps['poly']
    print(f"     Degree         : {poly.degree}")
    print(f"     Include Bias   : {poly.include_bias}")
    print(f"     Input Features : {artifact['metadata']['n_features_original']}")
    print(f"     Output Features: {len(poly.get_feature_names_out())}")

    print(f"   Step 2 - StandardScaler:")
    scaler = pipeline.named_steps['scaler']
    print(f"     Mean Values    : {len(scaler.mean_)} features scaled")

    print(f"   Step 3 - RFECV:")
    rfecv = pipeline.named_steps['rfecv']
    print(f"     Selected       : {rfecv.n_features_} features")
    print(f"     CV Folds       : 5")

    # ==========================================
    # 5. MODEL COMPLEXITY
    # ==========================================
    print(f"\n🧮 MODEL COMPLEXITY:")

    # Count model parameters (logistic regression coefficients)
    lr_model = rfecv.estimator_
    n_params = lr_model.coef_.size + lr_model.intercept_.size
    print(f"   Algorithm        : Logistic Regression")
    print(f"   Parameters       : {n_params} (coefficients + intercepts)")
    print(f"   Regularization   : {lr_model.penalty} (C={lr_model.C})")
    print(f"   Classes          : {len(lr_model.classes_)}")

    # ==========================================
    # 6. DEPLOYMENT SUITABILITY
    # ==========================================
    print(f"\n🚀 DEPLOYMENT SUITABILITY:")

    file_mb = file_size['mb']

    if file_mb < 1:
        rating = "🟢 EXCELLENT"
        note = "Very lightweight - perfect for mobile/cloud deployment"
    elif file_mb < 5:
        rating = "🟡 GOOD"
        note = "Lightweight - suitable for most deployments"
    elif file_mb < 20:
        rating = "🟠 ACCEPTABLE"
        note = "Moderate size - may need optimization for mobile"
    else:
        rating = "🔴 HEAVY"
        note = "Large model - consider compression or optimization"

    print(f"   Model Size Rating: {rating}")
    print(f"   Note             : {note}")
    print(f"   File Size        : {format_size(file_size)}")
    # Estimate runtime RAM needs based on serialized size
    print(f"   RAM Required     : ~{format_size({'bytes': total_component_size * 3, 'kb': total_component_size * 3 / 1024, 'mb': total_component_size * 3 / (1024*1024)})} (estimated 3x file size)")

    # ==========================================
    # 7. COMPARISON WITH ALTERNATIVES
    # ==========================================
    print(f"\n📊 CONTEXT - TYPICAL MODEL SIZES:")
    print(f"   {'Model Type':<30} {'Typical Size':>12}")
    print(f"   {'-'*44}")
    print(f"   {'Your Model (LogReg + RFECV)':<30} {format_size(file_size):>12} ✅")
    print(f"   {'Random Forest (100 trees)':<30} {'5-50 MB':>12}")
    print(f"   {'XGBoost':<30} {'1-20 MB':>12}")
    print(f"   {'Small Neural Network':<30} {'10-100 MB':>12}")
    print(f"   {'MobileNet (image)':<30} {'~16 MB':>12}")
    print(f"   {'ResNet50 (image)':<30} {'~100 MB':>12}")

    print(f"\n{'='*60}")
    print(f"✅ ANALYSIS COMPLETE")
    print(f"   File size: {format_size(file_size)}")
    print(f"   Status   : {rating}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()