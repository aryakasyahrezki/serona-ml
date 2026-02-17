"""
test_memory_usage.py

Objective: Measure RAM memory usage during model loading and inference.
           Helps understand memory footprint for deployment planning.

Requirements:
    pip install memory-profiler psutil
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import psutil
import gc

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = '../models/model.pkl'

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_ram_usage():
    """
    Objective: Get current RAM usage of this process
    Parameter: None
    Return: Dictionary with memory usage in bytes, MB
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        'rss_bytes': mem_info.rss,
        'rss_mb': mem_info.rss / (1024 * 1024),
        'vms_bytes': mem_info.vms,
        'vms_mb': mem_info.vms / (1024 * 1024),
    }

def get_system_ram():
    """
    Objective: Get total and available system RAM
    Parameter: None
    Return: Dictionary with system RAM info
    """
    vm = psutil.virtual_memory()
    return {
        'total_mb': vm.total / (1024 * 1024),
        'available_mb': vm.available / (1024 * 1024),
        'used_mb': vm.used / (1024 * 1024),
        'percent': vm.percent
    }

def format_mb(mb):
    """
    Objective: Format MB value into readable string
    Parameter: mb - float value in megabytes
    Return: Formatted string
    """
    if mb >= 1024:
        return f"{mb/1024:.2f} GB"
    elif mb >= 1:
        return f"{mb:.2f} MB"
    else:
        return f"{mb*1024:.2f} KB"

def create_dummy_sample():
    """
    Objective: Create a dummy feature sample for inference testing
    Parameter: None
    Return: Pandas DataFrame with one sample of 8 features
    """
    feature_names = [
        'ratio_len_width', 'ratio_jaw_cheek', 'ratio_forehead_jaw',
        'avg_jaw_angle', 'ratio_chin_jaw', 'circularity', 'solidity', 'extent'
    ]
    sample_values = {
        'ratio_len_width': 1.18,
        'ratio_jaw_cheek': 0.88,
        'ratio_forehead_jaw': 0.85,
        'avg_jaw_angle': 145.0,
        'ratio_chin_jaw': 0.88,
        'circularity': 0.95,
        'solidity': 0.999,
        'extent': 0.79
    }
    return pd.DataFrame([sample_values], columns=feature_names)

# ==========================================
# MAIN ANALYSIS
# ==========================================

def main():
    print("=" * 60)
    print("MEMORY USAGE ANALYSIS")
    print("=" * 60)

    # Check model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at: {MODEL_PATH}")
        sys.exit(1)

    # ==========================================
    # 1. SYSTEM INFO
    # ==========================================
    system_ram = get_system_ram()
    print(f"\n💻 SYSTEM RAM:")
    print(f"   Total Available : {format_mb(system_ram['total_mb'])}")
    print(f"   Currently Used  : {format_mb(system_ram['used_mb'])} ({system_ram['percent']:.1f}%)")
    print(f"   Currently Free  : {format_mb(system_ram['available_mb'])}")

    # ==========================================
    # 2. BASELINE MEMORY (before loading model)
    # ==========================================
    gc.collect()  # Clean up garbage
    baseline_ram = get_ram_usage()

    print(f"\n📊 BASELINE MEMORY (process, before loading model):")
    print(f"   RSS (Physical RAM) : {format_mb(baseline_ram['rss_mb'])}")
    print(f"   VMS (Virtual Mem)  : {format_mb(baseline_ram['vms_mb'])}")

    # ==========================================
    # 3. MODEL LOADING MEMORY
    # ==========================================
    print(f"\n⏳ Loading model...")
    load_start = time.perf_counter()
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact['pipeline']
    label_encoder = artifact['label_encoder']
    feature_names = artifact['feature_names']
    load_end = time.perf_counter()

    after_load_ram = get_ram_usage()
    load_time_ms = (load_end - load_start) * 1000
    load_overhead_mb = after_load_ram['rss_mb'] - baseline_ram['rss_mb']

    print(f"\n📦 AFTER MODEL LOADING:")
    print(f"   RSS (Physical RAM)  : {format_mb(after_load_ram['rss_mb'])}")
    print(f"   RAM Increase        : +{format_mb(load_overhead_mb)} (model overhead)")
    print(f"   Load Time           : {load_time_ms:.2f} ms")

    # ==========================================
    # 4. SINGLE INFERENCE MEMORY
    # ==========================================
    print(f"\n🔮 SINGLE INFERENCE MEMORY:")

    # Warmup
    dummy_sample = create_dummy_sample()
    _ = pipeline.predict(dummy_sample)

    # Measure
    before_inference = get_ram_usage()
    t_start = time.perf_counter()

    _ = pipeline.predict(dummy_sample)
    pred_proba = pipeline.predict_proba(dummy_sample)

    t_end = time.perf_counter()
    after_inference = get_ram_usage()

    inference_time_ms = (t_end - t_start) * 1000
    inference_overhead_mb = after_inference['rss_mb'] - before_inference['rss_mb']

    print(f"   RAM Before Inference : {format_mb(before_inference['rss_mb'])}")
    print(f"   RAM After Inference  : {format_mb(after_inference['rss_mb'])}")
    print(f"   Inference Overhead   : +{format_mb(abs(inference_overhead_mb))}")
    print(f"   Inference Time       : {inference_time_ms:.2f} ms")

    # ==========================================
    # 5. BATCH INFERENCE MEMORY
    # ==========================================
    print(f"\n📊 BATCH INFERENCE MEMORY (simulating real usage):")

    batch_sizes = [1, 10, 50, 100]
    print(f"   {'Batch Size':<12} {'RAM Used':>10} {'Time (ms)':>10}")
    print(f"   {'-'*35}")

    for batch_size in batch_sizes:
        # Create batch
        batch_data = pd.concat([dummy_sample] * batch_size, ignore_index=True)

        # Measure
        gc.collect()
        before = get_ram_usage()
        t_start = time.perf_counter()
        _ = pipeline.predict(batch_data)
        t_end = time.perf_counter()
        after = get_ram_usage()

        ram_diff = after['rss_mb'] - before['rss_mb']
        batch_time = (t_end - t_start) * 1000

        print(f"   {batch_size:<12} {format_mb(abs(ram_diff)):>10} {batch_time:>10.2f}")

    # ==========================================
    # 6. TOTAL MEMORY FOOTPRINT SUMMARY
    # ==========================================
    print(f"\n📋 TOTAL MEMORY FOOTPRINT SUMMARY:")
    print(f"   {'Component':<30} {'Memory':>10}")
    print(f"   {'-'*42}")
    print(f"   {'Python baseline':<30} {format_mb(baseline_ram['rss_mb']):>10}")
    print(f"   {'Model loading overhead':<30} {format_mb(max(0, load_overhead_mb)):>10}")
    print(f"   {'Inference overhead (single)':<30} {format_mb(abs(inference_overhead_mb)):>10}")
    print(f"   {'-'*42}")
    total_needed = after_load_ram['rss_mb']
    print(f"   {'TOTAL (model in memory)':<30} {format_mb(total_needed):>10}")

    # ==========================================
    # 7. DEPLOYMENT RECOMMENDATION
    # ==========================================
    print(f"\n🚀 DEPLOYMENT RECOMMENDATION:")

    if total_needed < 256:
        rating = "🟢 EXCELLENT"
        note = "Very low memory - suitable for any deployment"
        recommendation = "Can run on smallest cloud instances or edge devices"
    elif total_needed < 512:
        rating = "🟡 GOOD"
        note = "Low memory - suitable for most deployments"
        recommendation = "Suitable for standard cloud instances (512MB+ RAM)"
    elif total_needed < 1024:
        rating = "🟠 ACCEPTABLE"
        note = "Moderate memory - check deployment constraints"
        recommendation = "Requires at least 1GB RAM cloud instance"
    else:
        rating = "🔴 HIGH"
        note = "High memory - optimize before deployment"
        recommendation = "Consider model compression or larger instance"

    print(f"   Rating         : {rating}")
    print(f"   Note           : {note}")
    print(f"   Recommendation : {recommendation}")

    # ==========================================
    # 8. COMPARISON WITH DEPLOYMENT TIERS
    # ==========================================
    print(f"\n📊 DEPLOYMENT TIER COMPATIBILITY:")
    tiers = [
        ("Azure Container (0.5 GB RAM)", 512),
        ("Azure Container (1 GB RAM)", 1024),
        ("Azure Container (2 GB RAM)", 2048),
        ("Standard VM (4 GB RAM)", 4096),
    ]

    for tier_name, tier_ram_mb in tiers:
        fits = total_needed < tier_ram_mb * 0.7  # Use max 70% of RAM
        status = "✅ FITS" if fits else "❌ TOO LARGE"
        print(f"   {tier_name:<35} {status}")

    print(f"\n{'='*60}")
    print(f"✅ MEMORY ANALYSIS COMPLETE")
    print(f"   Total RAM needed: {format_mb(total_needed)}")
    print(f"   Status: {rating}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    try:
        import psutil
    except ImportError:
        print("❌ psutil not installed!")
        print("   Run: pip install psutil")
        sys.exit(1)

    main()