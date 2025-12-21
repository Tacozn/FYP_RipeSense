import os
import gc
import psutil
import time
from ultralytics import YOLO
from PIL import Image

# 🛠️ SETUP
MODEL_PATH = "models/best.pt"  # Make sure this path is right
TEST_IMAGE = "C:\\Users\\carro\\Fruit-Ripeness-and-Disease-Detection - Copy (2)\\static\\uploads\\b2.jpg" # ⚠️ Put any real image path here
PROCESS = psutil.Process(os.getpid())

def print_memory(step_name):
    # RSS (Resident Set Size) is the RAM actually used
    mem_mb = PROCESS.memory_info().rss / 1024 / 1024
    print(f"📊 [{step_name}] RAM Usage: {mem_mb:.2f} MB")

def test_lazy_loading():
    print("🚀 Starting Stress Test...")
    print_memory("Baseline (Imports only)")

    # 1. Simulate Loading the Model
    print("\n--- 🧠 LOADING MODEL ---")
    start_time = time.time()
    
    # Load Model
    model = YOLO(MODEL_PATH)
    print_memory("Model Loaded in RAM")
    
    # 2. Simulate Prediction (The heaviest part)
    print("\n--- ⚡ RUNNING PREDICTION (imgsz=320) ---")
    if os.path.exists(TEST_IMAGE):
        # We perform a real prediction to spike the memory
        results = model(TEST_IMAGE, imgsz=320, verbose=False)
        print_memory("During Prediction (Peak)")
    else:
        print(f"⚠️ Warning: {TEST_IMAGE} not found. Using dummy input.")
    
    # 3. The Cleanup (The most important part!)
    print("\n--- 🧹 CLEANUP PHASE ---")
    del model
    gc.collect() # Force Garbage Collection
    
    print_memory("After gc.collect()")
    
    print(f"\n✅ Test Complete in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    test_lazy_loading()