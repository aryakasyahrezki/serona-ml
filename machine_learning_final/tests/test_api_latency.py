"""
End-to-End API Latency Benchmark
Tests complete request cycle: upload → process → predict → respond
"""

import requests
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ==========================================
# SYSTEM & ENVIRONMENT CONFIGURATION
# ==========================================
API_URL = "http://localhost:8000/predict" # Local deployment endpoint
TEST_IMAGES_DIR = "../data/raw_data_30s_cropped"  # Root directory for validation dataset
print("="*60)
print("END-TO-END API LATENCY BENCHMARK")
print("="*60)
print(f"API URL: {API_URL}")
print(f"Test images: {TEST_IMAGES_DIR}")
print("="*60)

# ==========================================
# COLLECT TEST IMAGES
# ==========================================
test_images = []
labels = []

for shape in ['Heart', 'Oblong', 'Oval', 'Round', 'Square']:
    folder = Path(TEST_IMAGES_DIR) / shape
    
    if not folder.exists():
        print(f"⚠️  Warning: Folder not found: {folder}")
        continue
    
    images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
    # Take 5 images per class (25 total)
    images = images[:5]
    
    test_images.extend(images)
    labels.extend([shape] * len(images))
    print(f"  ✓ {shape}: {len(images)} images")

if len(test_images) == 0:
    print("\n❌ No test images found! Check TEST_IMAGES_DIR path.")
    sys.exit(1)

print(f"\n✅ Total test images: {len(test_images)}")

# ==========================================
# API COLD-START MITIGATION (WARM-UP)
# ==========================================
print("\n" + "="*60)
print("WARMING UP API...")
print("="*60)

try:
    with open(test_images[0], 'rb') as f:
        response = requests.post(API_URL, files={'file': ('warmup.jpg', f, 'image/jpeg')})
        if response.status_code == 200:
            print("✅ API is responding")
        else:
            print(f"❌ API returned status code: {response.status_code}")
            sys.exit(1)
except Exception as e:
    print(f"❌ Cannot connect to API: {e}")
    print("\nMake sure API is running:")
    print("  python api.py")
    sys.exit(1)

# ==========================================
# BENCHMARK EXECUTION PHASE
# ==========================================
print("\n" + "="*60)
print(f"RUNNING BENCHMARK ({len(test_images)} predictions)...")
print("="*60)

latencies_client = []
latencies_server = []
successful = 0
failed = 0
predictions = []

for idx, (img_path, true_label) in enumerate(zip(test_images, labels)):
    try:
        with open(img_path, 'rb') as f:
            # Measure client-side latency (total request time)
            t_start = time.perf_counter()
            response = requests.post(
                API_URL, 
                files={'file': (img_path.name, f, 'image/jpeg')},
                timeout=30
            )
            t_end = time.perf_counter()
            
            client_latency = (t_end - t_start) * 1000  # Convert to ms
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') == 'success':
                    latencies_client.append(client_latency)
                    
                    # Extract server-side ML inference time
                    server_time = float(data.get('server_inference_ms', 0))
                    latencies_server.append(server_time)
                    
                    # Extract prediction
                    shape_pred = data.get('shape', '')
                    predictions.append({
                        'true': true_label,
                        'predicted': shape_pred,
                        'client_latency': client_latency,
                        'server_latency': server_time
                    })
                    
                    successful += 1
                else:
                    failed += 1
                    print(f"  ⚠️  Failed: {data.get('message', 'Unknown error')}")
            else:
                failed += 1
                print(f"  ⚠️  HTTP {response.status_code}: {img_path.name}")
        
        # Progress indicator
        if (idx + 1) % 5 == 0:
            print(f"  Progress: {idx + 1}/{len(test_images)} ({successful} successful)")
            
    except requests.exceptions.Timeout:
        failed += 1
        print(f"  ⏱️  Timeout: {img_path.name}")
    except Exception as e:
        failed += 1
        print(f"  ❌ Error on {img_path.name}: {e}")

# ==========================================
# CALCULATE STATISTICS
# ==========================================
print("\n" + "="*60)
print("CALCULATING STATISTICS...")
print("="*60)

if len(latencies_client) == 0:
    print("❌ No successful predictions!")
    sys.exit(1)

latencies_client = np.array(latencies_client)
latencies_server = np.array(latencies_server)
overhead = latencies_client - latencies_server

# Calculate percentiles
p50_client = np.percentile(latencies_client, 50)
p90_client = np.percentile(latencies_client, 90)
p95_client = np.percentile(latencies_client, 95)
p99_client = np.percentile(latencies_client, 99)

p50_server = np.percentile(latencies_server, 50)
p90_server = np.percentile(latencies_server, 90)
p95_server = np.percentile(latencies_server, 95)
p99_server = np.percentile(latencies_server, 99)

p50_overhead = np.percentile(overhead, 50)
p90_overhead = np.percentile(overhead, 90)
p95_overhead = np.percentile(overhead, 95)

# ==========================================
# DISPLAY RESULTS
# ==========================================
print("\n" + "="*60)
print("RESULTS: End-to-End API Latency")
print("="*60)
print(f"Successful requests: {successful}/{len(test_images)} ({successful/len(test_images)*100:.1f}%)")
print(f"Failed requests:     {failed}")
print("")
print("CLIENT-SIDE (Total Request Time):")
print(f"  Mean     : {latencies_client.mean():.2f} ms ± {latencies_client.std():.2f} ms")
print(f"  Min/Max  : {latencies_client.min():.2f} / {latencies_client.max():.2f} ms")
print(f"  P50      : {p50_client:.2f} ms")
print(f"  P90      : {p90_client:.2f} ms")
print(f"  P95      : {p95_client:.2f} ms")
print(f"  P99      : {p99_client:.2f} ms")
print("")
print("SERVER-SIDE (Pure ML Inference):")
print(f"  Mean     : {latencies_server.mean():.2f} ms ± {latencies_server.std():.2f} ms")
print(f"  Min/Max  : {latencies_server.min():.2f} / {latencies_server.max():.2f} ms")
print(f"  P50      : {p50_server:.2f} ms")
print(f"  P90      : {p90_server:.2f} ms")
print(f"  P95      : {p95_server:.2f} ms")
print(f"  P99      : {p99_server:.2f} ms")
print("")
print("OVERHEAD (Network + Image Processing + Feature Extraction):")
print(f"  Mean     : {overhead.mean():.2f} ms")
print(f"  P50      : {p50_overhead:.2f} ms")
print(f"  P90      : {p90_overhead:.2f} ms")
print(f"  P95      : {p95_overhead:.2f} ms")
print("="*60)

# ==========================================
# BREAKDOWN
# ==========================================
print("\nLATENCY BREAKDOWN:")
print(f"  ML Inference:        ~{latencies_server.mean():.0f} ms ({latencies_server.mean()/latencies_client.mean()*100:.1f}%)")
print(f"  Other Processing:    ~{overhead.mean():.0f} ms ({overhead.mean()/latencies_client.mean()*100:.1f}%)")
print(f"  Total:               ~{latencies_client.mean():.0f} ms (100%)")

# ==========================================
# PERFORMANCE DATA VISUALIZATION
# =========================================
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS...")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Client-side histogram
ax = axes[0, 0]
ax.hist(latencies_client, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
ax.axvline(p50_client, color='green', linestyle='--', linewidth=2, label=f'P50: {p50_client:.0f}ms')
ax.axvline(p90_client, color='orange', linestyle='--', linewidth=2, label=f'P90: {p90_client:.0f}ms')
ax.axvline(p95_client, color='red', linestyle='--', linewidth=2, label=f'P95: {p95_client:.0f}ms')
ax.set_xlabel('Latency (ms)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Client-Side Latency Distribution', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Server-side histogram
ax = axes[0, 1]
ax.hist(latencies_server, bins=20, edgecolor='black', alpha=0.7, color='lightcoral')
ax.axvline(p50_server, color='green', linestyle='--', linewidth=2, label=f'P50: {p50_server:.0f}ms')
ax.axvline(p90_server, color='orange', linestyle='--', linewidth=2, label=f'P90: {p90_server:.0f}ms')
ax.axvline(p95_server, color='red', linestyle='--', linewidth=2, label=f'P95: {p95_server:.0f}ms')
ax.set_xlabel('Latency (ms)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Server-Side ML Inference Distribution', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Comparison bar chart
ax = axes[1, 0]
categories = ['P50', 'P90', 'P95']
client_vals = [p50_client, p90_client, p95_client]
server_vals = [p50_server, p90_server, p95_server]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, client_vals, width, label='Client (Total)', color='skyblue', edgecolor='black')
bars2 = ax.bar(x + width/2, server_vals, width, label='Server (ML Only)', color='lightcoral', edgecolor='black')

ax.set_xlabel('Percentile', fontsize=11)
ax.set_ylabel('Latency (ms)', fontsize=11)
ax.set_title('Client vs Server Latency Comparison', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}ms',
                ha='center', va='bottom', fontsize=9)

# 4. Latency breakdown pie chart
ax = axes[1, 1]
breakdown_labels = ['ML Inference', 'Other Processing']
breakdown_sizes = [latencies_server.mean(), overhead.mean()]
breakdown_colors = ['lightcoral', 'lightblue']

wedges, texts, autotexts = ax.pie(breakdown_sizes, labels=breakdown_labels, colors=breakdown_colors,
                                    autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
ax.set_title('Average Latency Breakdown', fontsize=12, fontweight='bold')

# Make percentage text bold
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(11)

plt.tight_layout()

# Save figure
output_path = '../latency/api_latency_benchmark.png'
Path('./output').mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ Chart saved to: {output_path}")

plt.show()

# ==========================================
# SUMMARY
# ==========================================
print("\n" + "="*60)
print("BENCHMARK COMPLETE!")
print("="*60)
print(f"✅ Results saved to: {output_path}")
print("="*60)