import os
import cv2
import numpy as np

TEST_ROOT = 'test_dataset'

print(f"{'FOLDER':<25} | {'COUNT':<5} | {'AVG HUE':<10} | {'SUGGESTED RANGE'}")
print("-" * 75)

for folder_name in sorted(os.listdir(TEST_ROOT)):
    folder_path = os.path.join(TEST_ROOT, folder_name)
    if not os.path.isdir(folder_path): continue
    
    hues = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        img = cv2.imread(filepath)
        if img is None: continue
        
        # Convert to HSV and get Median Hue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Crop center 100px
        h, w, _ = img.shape
        cy, cx = h//2, w//2
        crop = hsv[max(0, cy-50):min(h, cy+50), max(0, cx-50):min(w, cx+50)]
        
        if crop.size > 0:
            hues.append(np.median(crop[:,:,0]))

    if hues:
        avg_hue = np.mean(hues)
        min_hue = np.min(hues)
        max_hue = np.max(hues)
        print(f"{folder_name:<25} | {len(hues):<5} | {avg_hue:.1f}       | {min_hue:.0f} - {max_hue:.0f}")