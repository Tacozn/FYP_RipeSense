import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURATION
# ==========================================
TEST_ROOT = 'test_dataset'
MODEL_PATH = 'weights_new/best.pt' 

# ==========================================
# 2. LOAD MODEL
# ==========================================
print(f"🚀 Loading Custom Model from {MODEL_PATH}...")
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

y_true = []
y_pred = []

print(f"\n📊 Starting PURE AI Evaluation (Trusting the Neural Network)...")

# ==========================================
# 3. EVALUATION LOOP
# ==========================================
if not os.path.exists(TEST_ROOT):
    print(f"❌ Error: Folder '{TEST_ROOT}' not found.")
    exit()

for folder_name in os.listdir(TEST_ROOT):
    folder_path = os.path.join(TEST_ROOT, folder_name)
    
    if os.path.isdir(folder_path):
        # The folder name is the TRUTH (e.g., "ripe_banana")
        # We normalize it: "ripe_banana" -> "ripe banana"
        actual_label = folder_name.replace('_', ' ').lower().strip()
        
        print(f"   📂 Testing folder: {folder_name}...")

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                filepath = os.path.join(folder_path, filename)
                
                # Run YOLO Inference
                # verbose=False keeps the console clean
                results = model(filepath, verbose=False)
                
                # Get the prediction
                if results[0].probs:
                    # If it's a Classification model (yolov8-cls)
                    top1 = results[0].probs.top1
                    predicted_label = results[0].names[top1]
                elif results[0].boxes:
                    # If it's a Detection model (yolov8n)
                    # We take the box with highest confidence
                    best_box = max(results[0].boxes, key=lambda x: x.conf[0])
                    cls_id = int(best_box.cls[0])
                    predicted_label = results[0].names[cls_id]
                else:
                    # AI didn't see anything
                    predicted_label = "unknown"

                # Normalize prediction to match folder style
                predicted_label = predicted_label.replace('_', ' ').lower().strip()

                # Store results
                y_true.append(actual_label)
                y_pred.append(predicted_label)

# ==========================================
# 4. CALCULATE & PRINT RESULTS
# ==========================================
if len(y_true) > 0:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    print("\n" + "="*40)
    print(f"🏆 FINAL AI METRICS")
    print("="*40)
    print(f"✅ Accuracy:  {accuracy*100:.2f}%")
    print(f"🎯 Precision: {precision*100:.2f}%")
    print(f"🔎 Recall:    {recall*100:.2f}%")
    print(f"⚖️ F1-Score:  {f1*100:.2f}%")
    print("="*40)
    
    # Generate Confusion Matrix Graph
    labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted by AI')
    plt.ylabel('Actual Folder')
    plt.title('YOLOv8 Confusion Matrix')
    plt.tight_layout()
    plt.savefig('static/images/confusion_matrix_result.png') # Saving directly to static!
    print(f"📈 Graph saved to static/images/confusion_matrix_result.png")

else:
    print("❌ No images found to test!")