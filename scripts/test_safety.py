from PIL import Image
import numpy as np
import cv2
import psutil
import os

PROCESS = psutil.Process(os.getpid())
IMAGE_PATH = "static/monster_image.jpg"

def get_ram():
    return PROCESS.memory_info().rss / 1024 / 1024

print(f"[BASELINE] RAM: {get_ram():.2f} MB")

# --- 1. SIMULATE UPLOAD ---
print(f"\n[OPENING] {IMAGE_PATH}...")
try:
    # Load the monster
    img_pil = Image.open(IMAGE_PATH).convert("RGB")
    print(f"   Original Size: {img_pil.width}x{img_pil.height}")
    print(f"   RAM after load: {get_ram():.2f} MB")

    # --- 2. RUN YOUR NEW SAFETY CODE ---
    print("\n[SAFETY] Running Safety Check...")
    
    if img_pil.width > 1024 or img_pil.height > 1024:
        print("   [WARNING] Image is too big! Resizing...")
        img_pil.thumbnail((1024, 1024)) # <--- THE FIX
        print(f"   [OK] New Size: {img_pil.width}x{img_pil.height}")
    else:
        print("   [OK] Image size is okay.")

    # --- 3. CONVERT TO OPENCV (The Dangerous Part) ---
    # If the resize worked, this RAM spike should be small.
    # If the resize failed, this will spike RAM by ~100MB+.
    
    print("\n[DANGER] Converting to OpenCV (The Danger Zone)...")
    img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    current_ram = get_ram()
    print(f"[FINAL] RAM Usage: {current_ram:.2f} MB")

    if current_ram < 500:
        print("\n[SUCCESS] RAM stayed low. Safe to deploy!")
    else:
        print("\n[ERROR] RAM spiked too high. Do not deploy.")

except Exception as e:
    print(f"[EXCEPTION] Error: {e}")