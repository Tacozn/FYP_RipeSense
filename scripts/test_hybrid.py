import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================
# 1. SETUP
# ==========================================
MODEL_PATH = 'weights_new/best.pt'   # Path to your custom model
TEST_IMAGE_PATH = 'test_dataset/unripe_tomato/ut1.jpg' # <--- CHANGE THIS to your image file

# Load the model
try:
    print(f"🧠 Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# ==========================================
# 2. HELPER: CALCULATE HUE
# ==========================================
def get_average_hue(cropped_img):
    """
    Takes a cropped image of a fruit, converts to HSV, 
    and returns the average Hue value (0-179).
    """
    # Convert to HSV color space (Hue, Saturation, Value)
    hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    
    # Extract just the Hue channel (Channel 0)
    hue_channel = hsv_img[:, :, 0]
    
    # Calculate the average (ignoring black pixels if using a mask, but simple mean is fine here)
    avg_hue = np.mean(hue_channel)
    return int(avg_hue)

# ==========================================
# 3. RUN DETECTION
# ==========================================
# Load image
image = cv2.imread(TEST_IMAGE_PATH)
if image is None:
    print(f"❌ Error: Could not find image at {TEST_IMAGE_PATH}")
    exit()

# Run YOLO inference
results = model(image)

# ==========================================
# 4. PROCESS RESULTS
# ==========================================
print(f"\n🔍 Analyzing {TEST_IMAGE_PATH}...")

for result in results:
    boxes = result.boxes
    for box in boxes:
        # --- A. THE FINDER (YOLO) ---
        # Get coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Get what YOLO thinks it is
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        yolo_label = model.names[cls_id]

        # --- B. THE JUDGE (Hue Calculation) ---
        # 1. Crop the fruit from the image
        fruit_crop = image[y1:y2, x1:x2]
        
        # 2. Calculate average color
        if fruit_crop.size > 0:
            avg_hue = get_average_hue(fruit_crop)
        else:
            avg_hue = 0

        # --- C. DRAW RESULTS ---
        # Draw the box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Text to display
        text = f"{yolo_label} ({conf:.2f}) | Hue: {avg_hue}"
        
        # Draw text background (for readability)
        cv2.rectangle(image, (x1, y1 - 30), (x1 + 400, y1), (0, 255, 0), -1)
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        print(f" > Found: {yolo_label} | Confidence: {conf:.2f} | Average Hue: {avg_hue}")

# ==========================================
# 5. SHOW OUTPUT
# ==========================================
cv2.imshow("Hybrid Detection Test", image)
print("\nPress any key to close the window...")
cv2.waitKey(0)
cv2.destroyAllWindows()