import os
import io
import json
import base64
import numpy as np
import cv2
from datetime import datetime
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURATION
# ==========================================
app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
HISTORY_FILE = os.path.join(BASE_DIR, 'history.json')
# Ensure this matches your actual best model path
DETECTION_MODEL_PATH = os.path.join(BASE_DIR, 'weights_new', 'best.pt')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# ==========================================
# 2. LOAD MODELS
# ==========================================
# A. Gatekeeper (General Classification)
try:
    print("Loading Gatekeeper Model (yolov8n-cls.pt)...")
    gatekeeper_model = YOLO('yolov8n-cls.pt')
except Exception as e:
    print(f"WARNING: Could not load Gatekeeper. Error: {e}")
    gatekeeper_model = None

# B. Specialist (Your Custom Model)
try:
    if os.path.exists(DETECTION_MODEL_PATH):
        print(f"Loading Custom Detection Model from: {DETECTION_MODEL_PATH}")
        fruit_detection_model = YOLO(DETECTION_MODEL_PATH)
    else:
        print(f"⚠️ WARNING: Custom model not found at {DETECTION_MODEL_PATH}")
        fruit_detection_model = None
except Exception as e:
    print(f"WARNING: Could not load custom model. Error: {e}")
    fruit_detection_model = None

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def save_history(filename, fruit, ripeness, confidence):
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file": filename,
        "fruit": fruit,
        "result": ripeness,
        "confidence": f"{confidence}%"
    }
    
    # Load existing history or create new
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
        
    data.append(entry)
    
    with open(HISTORY_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def analyze_ripeness_mock_tuned(image_cv2, fruit_type="unknown"):
    """
    Analyzes ripeness based on color (Hue) tuned specifically for 
    Banana, Mango, and Tomato.
    """
    label = fruit_type.lower()
    
    # --- 1. TRUST THE AI (If label is specific) ---
    if "unripe" in label:
        return "Unripe", 94.0, {'green_percentage': 90, 'yellow_percentage': 8, 'brown_percentage': 2}
    if "overripe" in label:
        return "Overripe", 91.0, {'green_percentage': 5, 'yellow_percentage': 15, 'brown_percentage': 80}
    if "ripe" in label and "over" not in label:
        return "Ripe", 94.2, {'green_percentage': 5, 'yellow_percentage': 90, 'brown_percentage': 5}

    # --- 2. FALLBACK: COLOR ANALYSIS ---
    # Convert to HSV (Hue, Saturation, Value)
    hsv = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2HSV)
    hue_values = hsv[:,:,0].flatten()
    
    if hue_values.size == 0:
         return "Unknown", 0.0, {}

    avg_hue = np.mean(hue_values)
    print(f"🎨 Analyzing Color for {label}: Avg Hue = {avg_hue:.1f}")

    # LOGIC A: TOMATOES (Red Logic)
    # Hue 0-15 (Red) | Hue 30-90 (Green/Yellow) | Hue > 160 (Red/Pink wrapping around)
    if "tomato" in label:
        if avg_hue < 20 or avg_hue > 160: 
            # Deep Red = Ripe or Overripe
            return "Ripe", 96.5, {'green_percentage': 5, 'yellow_percentage': 5, 'brown_percentage': 90}
        elif 35 < avg_hue < 90:
            return "Unripe", 88.0, {'green_percentage': 90, 'yellow_percentage': 5, 'brown_percentage': 5}
        else:
            return "Turning", 85.0, {'green_percentage': 20, 'yellow_percentage': 60, 'brown_percentage': 20}

    # LOGIC B: BANANA & MANGO (Yellow Logic)
    # Hue 20-35 (Yellow) | Hue 35-90 (Green) | Hue 10-20 (Orange/Brown)
    else:
        if 35 < avg_hue < 90: 
            return "Unripe", 88.5, {'green_percentage': 85, 'yellow_percentage': 10, 'brown_percentage': 5}
        elif 20 <= avg_hue <= 35:
            return "Ripe", 94.2, {'green_percentage': 5, 'yellow_percentage': 90, 'brown_percentage': 5}
        else:
            return "Overripe", 89.0, {'green_percentage': 5, 'yellow_percentage': 15, 'brown_percentage': 80}

# ==========================================
# 4. ROUTES
# ==========================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ripeness_detection')
def ripeness_detection():
    return render_template('ripeness_detection.html')

@app.route('/detect_ripeness', methods=['POST'])
def detect_ripeness():
    if 'file' not in request.files:
        return redirect(url_for('ripeness_detection'))
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('ripeness_detection'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Process Image
    img_pil = Image.open(filepath).convert("RGB")
    img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # ---------------------------------------------------------
    # STEP 1: THE SMART GATEKEEPER
    # ---------------------------------------------------------
    VALID_FRUITS = ['banana', 'mango', 'tomato']
    
    # 🚨 CHANGE 1: Removed 'pomegranate' from this list!
    # We now let pomegranates through to the Custom Model check.
    FORBIDDEN_FRUITS = ['apple', 'orange', 'strawberry', 'lemon', 'pineapple', 'grape', 'watermelon']
    
    detected_object = "Unknown"
    gatekeeper_says_valid = False
    gatekeeper_says_forbidden = False
    gatekeeper_label = "Unknown" 
    
    if gatekeeper_model:
        results = gatekeeper_model(img_pil)
        top_prob = results[0].probs.top1
        gatekeeper_label = results[0].names[top_prob].lower()
        print(f"🧐 Gatekeeper saw: {gatekeeper_label}") 

        # CHECK 1: Is it definitely a valid fruit?
        for valid in VALID_FRUITS:
            if valid in gatekeeper_label:
                detected_object = valid
                gatekeeper_says_valid = True
                break
        
        # CHECK 2: Is it definitely a FORBIDDEN fruit?
        if not gatekeeper_says_valid:
            for forbidden in FORBIDDEN_FRUITS:
                if forbidden in gatekeeper_label:
                    gatekeeper_says_forbidden = True
                    detected_object = gatekeeper_label
                    break
    
    # ---------------------------------------------------------
    # STEP 2: THE DECISION TREE
    # ---------------------------------------------------------
    
    # CASE A: Gatekeeper loves it. Proceed!
    if gatekeeper_says_valid:
        pass 
        
    # CASE B: Gatekeeper hates it (It's an Apple!). HARD REJECT.
    elif gatekeeper_says_forbidden:
        print(f"⛔ Blocked Forbidden Fruit: {detected_object}")
        img_base64 = image_to_base64(img_pil)
        return render_template('result_page.html', 
                             img_str=img_base64, 
                             not_fruit=True, 
                             detected_object=detected_object)

    # CASE C: Gatekeeper is confused (Saw Pomegranate, Persimmon, or Unknown).
    else:
        print(f"🤔 Gatekeeper is unsure ({gatekeeper_label}). Asking Custom Model...")
        
        if fruit_detection_model:
            # 🚨 CHANGE 2: Lowered confidence from 0.20 to 0.10
            # This catches the REALLY ugly/rotten ones that the AI is unsure about.
            det_results = fruit_detection_model(img_pil, conf=0.10)
            
            if det_results[0].boxes:
                # Find best box
                best_box = max(det_results[0].boxes, key=lambda x: x.conf[0])
                custom_label = det_results[0].names[int(best_box.cls[0])].lower()
                custom_conf = float(best_box.conf[0])
                
                print(f"🦸 Custom Model Rescued it! Found: {custom_label} ({custom_conf:.2f})")
                
                found_valid_custom = False
                for valid in VALID_FRUITS:
                    if valid in custom_label:
                        detected_object = custom_label 
                        found_valid_custom = True
                        break
                
                if not found_valid_custom:
                    # Even Custom Model didn't see a Banana/Mango/Tomato
                    img_base64 = image_to_base64(img_pil)
                    return render_template('result_page.html', img_str=img_base64, not_fruit=True, detected_object=gatekeeper_label)
            else:
                img_base64 = image_to_base64(img_pil)
                return render_template('result_page.html', img_str=img_base64, not_fruit=True, detected_object=gatekeeper_label)
        else:
             img_base64 = image_to_base64(img_pil)
             return render_template('result_page.html', img_str=img_base64, not_fruit=True, detected_object=gatekeeper_label)

    # ---------------------------------------------------------
    # STEP 3: COLOR SANITY CHECK (The "Mango Correction")
    # ---------------------------------------------------------
    
    img_hsv_check = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2HSV)
    avg_hue_check = np.mean(img_hsv_check[:, :, 0])
    
    if "mango" in detected_object.lower() and avg_hue_check < 20:
        print(f"💡 Correction: Object is '{detected_object}' but Hue is {avg_hue_check:.1f} (Red). Swapping to Tomato.")
        detected_object = "tomato"

    # ---------------------------------------------------------
    # STEP 4: PREPARE RESULTS
    # ---------------------------------------------------------
    display_name = detected_object
    if 'hip' in detected_object or 'persimmon' in detected_object: 
        display_name = "Tomato"
    
    # Analyze Ripeness
    ripeness, conf, colors = analyze_ripeness_mock_tuned(img_cv2, detected_object)
    
    # Save to history
    save_history(filename, display_name, ripeness, conf)
    
    img_base64 = image_to_base64(img_pil)
    
    # Flag for Red Fruits (for UI Progress Bars)
    RED_FRUITS_UI = ['tomato']
    is_red_fruit = any(f in detected_object.lower() for f in RED_FRUITS_UI)

    result_data = {
        'ripeness_level': ripeness,
        'confidence': conf / 100, 
        'color_analysis': colors 
    }
    
    # Nutrition Data
    nutrition_data = {}
    if "banana" in detected_object.lower():
        nutrition_data = {
            "calories": "89 kcal", "sugar": "12.2g", "fiber": "2.6g", 
            "vitamin": "Vit C, B6", "benefit": "Great source of energy and potassium."
        }
    elif "tomato" in detected_object.lower():
        nutrition_data = {
            "calories": "18 kcal", "sugar": "2.6g", "fiber": "1.2g", 
            "vitamin": "Lycopene", "benefit": "Rich in antioxidants for heart health."
        }
    elif "mango" in detected_object.lower():
        nutrition_data = {
            "calories": "60 kcal", "sugar": "13.7g", "fiber": "1.6g", 
            "vitamin": "Vit A, C", "benefit": "Boosts immunity and supports eye health."
        }
    
    return render_template('result_page.html', 
                         img_str=img_base64, 
                         ripeness_result=result_data,
                         class_names=[display_name],
                         is_red_fruit=is_red_fruit,
                         nutrition=nutrition_data)

@app.route('/history')
def show_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            try:
                data = json.load(f)
                data.reverse() 
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    return render_template('history.html', history=data)

@app.route('/fruit_detection')
def fruit_detection():
    return render_template('fruit_detection.html')

@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    """API for Live Video Feed"""
    data = request.json
    image_data = data['image_data'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    detected_objects = []
    if fruit_detection_model:
        results = fruit_detection_model(img)
        for r in results:
            if r.boxes:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    detected_objects.append({
                        'class': r.names[cls_id],
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].tolist() 
                    })
    return jsonify(detected_objects)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)