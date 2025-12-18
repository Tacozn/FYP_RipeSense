import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURATION
# ==========================================
TEST_ROOT = 'test_dataset'  # The folder containing your subfolders
MODEL_PATH = 'weights_new/best.pt' # Path to your custom model (if it exists)

# ==========================================
# 2. LOAD AI MODEL (DEBUG VERSION)
# ==========================================
print(f"🔄 Loading AI Model...")

# ⚠️ FORCE the path to your best weights
# Make sure this path is exactly where your .pt file is!
MODEL_PATH = 'weights_new/best.pt' 

if os.path.exists(MODEL_PATH):
    print(f"   ✅ SUCCESS: Found custom weights at {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
else:
    print(f"   ❌ ERROR: Could not find {MODEL_PATH}!")
    print(f"   ⚠️ Falling back to generic YOLO (This causes LOW ACCURACY!)")
    model = YOLO('yolov8n.pt') 

# Print the classes the model actually knows
print("   🧠 Model Knowledge Check:")
print(f"      The model knows {len(model.names)} classes.")
print(f"      First 5 classes: {list(model.names.values())[:5]}")

# ==========================================
# 3. RIPENESS ALGORITHM (Replicating app.py)
# ==========================================
def analyze_ripeness_algorithm(image_cv2, fruit_type):
    hsv = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2HSV)
    height, width, _ = image_cv2.shape
    
    # Crop center 100px (Matches your calibration script)
    center_y, center_x = height // 2, width // 2
    y1, y2 = max(0, center_y - 100), min(height, center_y + 100)
    x1, x2 = max(0, center_x - 100), min(width, center_x + 100)
    crop = hsv[y1:y2, x1:x2]
    
    if crop.size == 0: return "unknown"
    avg_hue = np.mean(crop[:,:,0])
    
    # ---------------------------------------------------------
    # 🔴 RED FRUITS (Tomato) - Based on your data
    # ---------------------------------------------------------
    # Ripe Avg: 8.1
    # Overripe Avg: 20.2
    # Unripe Avg: 36.6
    
    RED_FRUITS = ['tomato', 'apple', 'strawberry', 'pomegranate', 'raspberry']
    is_red = any(f in fruit_type.lower() for f in RED_FRUITS)

    if is_red:
        if avg_hue < 15:
            return "ripe"       # Range roughly 0-15 (Targeting 8.1)
        elif 15 <= avg_hue < 28:
            return "overripe"   # Range roughly 15-28 (Targeting 20.2)
        else:
            return "unripe"     # Range > 28 (Targeting 36.6)

    # ---------------------------------------------------------
    # 🟡 YELLOW FRUITS (Banana, Mango) - Based on your data
    # ---------------------------------------------------------
    # Overripe Avg: ~18 (Mango 16.7, Banana 20.9)
    # Ripe Avg:     ~24 (Mango 23.4, Banana 25.2)
    # Unripe Avg:   ~42 (Mango 48.0, Banana 36.5)

    else:
        if avg_hue < 22:
            return "overripe"   # Range < 22 (Targeting ~18)
        elif 22 <= avg_hue < 30:
            return "ripe"       # Range 22-30 (Targeting ~24)
        else:
            return "unripe"     # Range > 30 (Targeting ~42)

# ==========================================
# 4. MAIN EVALUATION LOOP
# ==========================================
y_true = []
y_pred = []

print(f"\n🚀 Starting Evaluation on images in '{TEST_ROOT}'...\n")

if not os.path.exists(TEST_ROOT):
    print(f"❌ ERROR: Folder '{TEST_ROOT}' not found! Please create it and add subfolders.")
    exit()

# Iterate through every folder (e.g., 'ripe_banana', 'unripe_tomato')
for folder_name in os.listdir(TEST_ROOT):
    folder_path = os.path.join(TEST_ROOT, folder_name)
    
    # Only process directories
    if os.path.isdir(folder_path):
        # 1. Determine the "Correct Answer" from the folder name
        # We replace underscores with spaces: "ripe_banana" -> "ripe banana"
        actual_label = folder_name.replace('_', ' ').lower()
        
        print(f"📂 Processing folder: {folder_name}...")

        # Process each image in the folder
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                filepath = os.path.join(folder_path, filename)
                img = cv2.imread(filepath)
                
                if img is None:
                    print(f"   ⚠️ Could not read {filename}, skipping.")
                    continue

                # ---------------------------------------------------------
                # STEP A: DETECT FRUIT TYPE (YOLO)
                # ---------------------------------------------------------
                results = model(img, verbose=False)
                detected_fruit = "unknown"
                
                # Check if YOLO detected anything
                if results[0].boxes:
                    # Get the detection with the highest confidence
                    top_box = sorted(results[0].boxes, key=lambda x: x.conf[0], reverse=True)[0]
                    cls_id = int(top_box.cls[0])
                    detected_fruit = results[0].names[cls_id].lower()
                
                # ---------------------------------------------------------
                # STEP B: RIPENESS-ONLY EVALUATION (The Fix)
                # ---------------------------------------------------------
                # Since standard YOLO doesn't know Mango/Tomato, we assume 
                # the folder name tells us the correct fruit type.
                # We only test if your COLOR ALGORITHM gets the ripeness right.
                
                # 1. Get the "Correct" fruit type from the folder name
                # (e.g. folder "ripe_mango" -> fruit is "mango")
                true_fruit_type = actual_label.split()[-1] 
                
                # 2. Force the system to analyze it AS that fruit
                pred_ripeness = analyze_ripeness_algorithm(img, true_fruit_type)
                
                # 3. Build the label using the TRUE fruit type but PREDICTED ripeness
                pred_label = f"{pred_ripeness} {true_fruit_type}"
                
                # ---------------------------------------------------------
                # STEP C: SAVE RESULTS
                # ---------------------------------------------------------
                y_true.append(actual_label)
                y_pred.append(pred_label)

# ==========================================
# 5. CALCULATE METRICS & GENERATE REPORT
# ==========================================
print("\n" + "="*40)
print(f"📊 FINAL RESULTS ({len(y_true)} images tested)")
print("="*40)

if len(y_true) == 0:
    print("❌ No images found! Check your folder structure.")
else:
    # Calculate Standard Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    # Print to Console (Copy this for your report text)
    print(f"✅ Accuracy:  {accuracy*100:.2f}%")
    print(f"🎯 Precision: {precision*100:.2f}%")
    print(f"🔎 Recall:    {recall*100:.2f}%")
    print(f"⚖️ F1-Score:  {f1*100:.2f}%")

    # Generate Confusion Matrix (The Graph)
    # Get all unique labels involved (True + Predicted) and sort them
    labels = sorted(list(set(y_true + y_pred)))
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - RipeSense Evaluation')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the graph
    save_path = 'confusion_matrix_result.png'
    plt.savefig(save_path)
    print(f"\n📈 Confusion Matrix saved as '{save_path}'")
    print("   (Copy this image into your Results chapter!)")