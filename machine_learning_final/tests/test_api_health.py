import requests
import time
import os

# Configuration
API_URL = "http://localhost:8000/predict"
TEST_IMAGE = "../data/raw_data_30s_cropped/Oval/1.jpg"  # Adjust path if needed

print("="*60)
print("TESTING API SPEED - FILE PRELOADING TEST")
print("="*60)

# Check if file exists
if not os.path.exists(TEST_IMAGE):
    print(f"❌ Image not found: {TEST_IMAGE}")
    print("Please update TEST_IMAGE path in the script")
    exit()

print(f"Test image: {TEST_IMAGE}")
print(f"API URL: {API_URL}")
print()

# ===== PRE-LOAD FILE INTO MEMORY =====
print("Loading image into memory...")
with open(TEST_IMAGE, 'rb') as f:
    file_content = f.read()  # Read entire file into RAM

file_size_kb = len(file_content) / 1024
print(f"✅ Image loaded: {file_size_kb:.1f} KB")
print()

# Test 5 times
print("Running 5 test requests (file pre-loaded in memory)...")
print("-"*60)

times = []

for i in range(5):
    # ✅ NEW: File is already in memory, just measure the HTTP request
    start = time.time()
    
    try:
        response = requests.post(
            API_URL, 
            files={'file': ('test.jpg', file_content, 'image/jpeg')}
        )
        
        end = time.time()
        duration_ms = (end - start) * 1000
        times.append(duration_ms)
        
        print(f"\nRequest {i+1}:")
        print(f"  Client time: {duration_ms:.2f} ms")
        print(f"  Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"  Result: {data.get('shape')}")
            print(f"  Server ML time: {data.get('server_inference_ms')} ms")
            print(f"  Server total: {data.get('total_request_ms')} ms")
            
            # Show breakdown if available
            if 'breakdown' in data:
                breakdown = data['breakdown']
                print(f"  Server breakdown:")
                print(f"    MediaPipe:  {breakdown.get('mediapipe_ms')} ms")
                print(f"    ML:         {breakdown.get('ml_inference_ms')} ms")
                print(f"    Skintone:   {breakdown.get('skintone_ms')} ms")
        else:
            print(f"  Error: {response.text}")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")

print()
print("="*60)
print("RESULTS SUMMARY")
print("="*60)

if times:
    import statistics
    
    avg = statistics.mean(times)
    median = statistics.median(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"Client-side latency (with file pre-loaded):")
    print(f"  Average: {avg:.2f} ms")
    print(f"  Median:  {median:.2f} ms")
    print(f"  Min:     {min_time:.2f} ms")
    print(f"  Max:     {max_time:.2f} ms")
    print()
    
    if avg < 50:
        print("✅ RESULT: API is FAST! (~10-50ms)")
        print("   The 2-second delay was Windows file I/O overhead!")
    elif avg < 200:
        print("✅ RESULT: API is good! (~50-200ms)")
        print("   Some overhead remains but acceptable.")
    else:
        print("⚠️  RESULT: Still slow (>200ms)")
        print("   May need further investigation.")

print()
print("="*60)
print("👉 Check the API console for detailed server-side timing")
print("="*60)