import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# --- CONFIGURATION ---
TEST_ROOT = 'test_dataset'  # Folder containing your 9 sub-folders

# --- RIPENESS ALGORITHM (The "Color Logic" from your App) ---
def analyze_ripeness_algorithm(image_cv2, fruit_type):
    hsv = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2HSV)
    height, width, _ = image_cv2.shape
    
    # Crop center to focus on fruit
    center_y, center_x = height // 2, width // 2
    y1, y2 = max(0, center_y - 50), min(height, center_y + 50)
    x1, x2 = max(0, center_x - 50), min(width, center_x + 50)
    crop = hsv[y1:y2, x1:x2]
    
    if crop.size == 0: return "unknown"
    avg_hue = np.median(crop[:,:,0])
    
    # Logic for RED fruits (Tomato)
    RED_FRUITS = ['tomato', 'apple', 'strawberry']
    is_red = any(f in fruit_type.lower() for f in RED_FRUITS)

    if is_red:
        if avg_hue < 25 or avg_hue > 140: return "ripe"
        elif 35 < avg_hue < 90: return "unripe"
        else: return "overripe"
    # Logic for YELLOW/GREEN fruits (Banana, Mango)
    else:
        if 35 < avg_hue < 90: return "unripe"
        elif 15 <= avg_hue <= 35: return "ripe"
        else: return "overripe"

# --- MAIN EVALUATION LOOP ---
y_true = []
y_pred = []

print(f"🚀 Starting Ripeness-Only Evaluation on '{TEST_ROOT}'...\n")

if not os.path.exists(TEST_ROOT):
    print("❌ Error: test_dataset folder not found.")
    exit()

for folder_name in os.listdir(TEST_ROOT):
    folder_path = os.path.join(TEST_ROOT, folder_name)
    
    if os.path.isdir(folder_path):
        # 1. Get the TRUTH from the folder name
        # Folder: "ripe_mango" -> Label: "ripe mango"
        actual_label = folder_name.replace('_', ' ').lower()
        
        # Extract the fruit type (e.g., "mango")
        true_fruit_type = actual_label.split()[-1] 

        print(f"📂 Testing {actual_label}...")

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(folder_path, filename)
                img = cv2.imread(filepath)
                if img is None: continue

                # 2. THE BYPASS (Unit Test)
                # We skip YOLO. We force the algorithm to run as if it detected the fruit correctly.
                # This measures the performance of your COLOR LOGIC only.
                pred_ripeness = analyze_ripeness_algorithm(img, true_fruit_type)
                pred_label = f"{pred_ripeness} {true_fruit_type}"

                y_true.append(actual_label)
                y_pred.append(pred_label)

# --- RESULTS ---
if len(y_true) > 0:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    print("\n" + "="*40)
    print(f"📊 RESULTS (Ripeness Logic Only)")
    print("="*40)
    print(f"✅ Accuracy:  {accuracy*100:.2f}%")
    print(f"⚖️ F1-Score:  {f1*100:.2f}%")
    
    # Save Graph
    labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Color Algorithm Performance)')
    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    print(f"📈 Graph saved as 'evaluation_results.png'")
else:
    print("❌ No images found.")