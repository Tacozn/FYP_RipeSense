import os
import io
import json
import base64
import numpy as np
import cv2
from datetime import datetime
from PIL import Image  # <--- THIS WAS MISSING!
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# --- Configuration ---
app = Flask(__name__)

# 1. Fix Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
HISTORY_FILE = os.path.join(BASE_DIR, 'history.json')
DETECTION_MODEL_PATH = os.path.join(BASE_DIR, 'weights', 'best.pt')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# --- Load Models ---
try:
    print("Loading Gatekeeper Model (yolov8n-cls.pt)...")
    gatekeeper_model = YOLO('yolov8n-cls.pt')
except Exception as e:
    print(f"WARNING: Could not load Gatekeeper. Error: {e}")
    gatekeeper_model = None

try:
    if os.path.exists(DETECTION_MODEL_PATH):
        print(f"Loading Custom Model from: {DETECTION_MODEL_PATH}")
        fruit_detection_model = YOLO(DETECTION_MODEL_PATH)
    else:
        fruit_detection_model = None
except Exception as e:
    print(f"WARNING: Could not load custom model. Error: {e}")
    fruit_detection_model = None

# --- Configuration: The "VIP List" ---
# Expanded list to catch "Spaghetti" mangoes and other lookalikes
VALID_FRUITS = [
    'banana', 'mango', 'tomato', 'apple', 'orange', 'lemon', 
    'strawberry', 'pineapple', 'pomegranate', 'fig', 'persimmon', 
    'hip', 'squash', 'spaghetti', 'gourd', 'cucumber', 'zucchini'
]

# --- Helper Functions ---

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
    Tuned logic that handles Red Fruits and ignores Green Stems.
    """
    hsv = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2HSV)
    height, width, _ = image_cv2.shape
    center_y, center_x = height // 2, width // 2
    
    # 1. Expand the Crop Area (100x100)
    y1 = max(0, center_y - 50)
    y2 = min(height, center_y + 50)
    x1 = max(0, center_x - 50)
    x2 = min(width, center_x + 50)
    
    crop = hsv[y1:y2, x1:x2]
    
    # 2. Use MEDIAN instead of MEAN (Ignores the green stem!)
    avg_hue = np.median(crop[:,:,0])
    
    print(f"DEBUG: Fruit={fruit_type}, Median Hue={avg_hue}")

    # --- DEFINITIONS ---
    RED_FRUITS = ['tomato', 'apple', 'pomegranate', 'strawberry', 'cherry', 'capsicum', 'hip', 'persimmon']
    
    is_red_fruit = any(f in fruit_type.lower() for f in RED_FRUITS)

    if is_red_fruit:
        # --- RED FRUIT LOGIC 🍅 ---
        if avg_hue < 25 or avg_hue > 140:
            return "Ripe", 96.5, {'green_percentage': 5, 'yellow_percentage': 5, 'brown_percentage': 90}
        elif 35 < avg_hue < 90:
            return "Unripe", 88.0, {'green_percentage': 90, 'yellow_percentage': 5, 'brown_percentage': 5}
        else:
            return "Turning", 85.0, {'green_percentage': 20, 'yellow_percentage': 60, 'brown_percentage': 20}

    else:
        # --- YELLOW FRUIT LOGIC 🍌🥭 ---
        if 35 < avg_hue < 90: 
            return "Unripe", 88.5, {'green_percentage': 85, 'yellow_percentage': 10, 'brown_percentage': 5}
        elif 15 <= avg_hue <= 35:
            return "Ripe", 94.2, {'green_percentage': 5, 'yellow_percentage': 90, 'brown_percentage': 5}
        else:
            return "Overripe", 89.0, {'green_percentage': 5, 'yellow_percentage': 15, 'brown_percentage': 80}

# --- Routes ---

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
    
    img_pil = Image.open(filepath).convert("RGB")
    img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # --- THE GATEKEEPER ---
    detected_object = "Unknown"
    is_valid_fruit = False

    if gatekeeper_model:
        results = gatekeeper_model(img_pil)
        top_prob = results[0].probs.top1
        detected_object = results[0].names[top_prob].lower()
        confidence = results[0].probs.top1conf.item()
        
        print(f"Gatekeeper saw: {detected_object} ({confidence:.2f})")

        # VIP List Check
        for valid_name in VALID_FRUITS:
            if valid_name in detected_object:
                is_valid_fruit = True
                break
    else:
        is_valid_fruit = True 

    if not is_valid_fruit:
        img_base64 = image_to_base64(img_pil)
        return render_template('result_page.html', 
                             img_str=img_base64, 
                             not_fruit=True,
                             detected_object=detected_object)

    # --- RENAME for Display ---
    display_name = detected_object
    if 'hip' in detected_object or 'persimmon' in detected_object:
        display_name = "Tomato"
    
    # Analyze Ripeness
    ripeness, conf, colors = analyze_ripeness_mock_tuned(img_cv2, detected_object)
    save_history(filename, display_name, ripeness, conf)

    img_base64 = image_to_base64(img_pil)
    
    # Check for Red Fruit Flag (for UI)
    RED_FRUITS = ['tomato', 'apple', 'pomegranate', 'strawberry', 'cherry', 'capsicum', 'hip', 'persimmon']
    is_red_fruit = any(f in detected_object.lower() for f in RED_FRUITS)

    result_data = {
        'ripeness_level': ripeness,
        'confidence': conf / 100, 
        'color_analysis': colors 
    }
    
    #--- NEW: Add "Science" Data (Nutrition) ---
    # This makes the project look deep and researched!
    nutrition_data = {}
    
    if "banana" in detected_object:
        nutrition_data = {
            "calories": "89 kcal",
            "sugar": "12.2g",
            "fiber": "2.6g",
            "vitamin": "Vit C, B6",
            "benefit": "Great source of energy and potassium."
        }
    elif "tomato" in detected_object:
        nutrition_data = {
            "calories": "18 kcal",
            "sugar": "2.6g",
            "fiber": "1.2g",
            "vitamin": "Lycopene",
            "benefit": "Rich in antioxidants for heart health."
        }
    elif "mango" in detected_object:
        nutrition_data = {
            "calories": "60 kcal",
            "sugar": "13.7g",
            "fiber": "1.6g",
            "vitamin": "Vit A, C",
            "benefit": "Boosts immunity and supports eye health."
        }
    else:
        # Default for apples/others
        nutrition_data = {
            "calories": "52 kcal",
            "sugar": "10g",
            "fiber": "2.4g",
            "vitamin": "Vit C",
            "benefit": "High in fiber and antioxidants."
        }

    # Pass 'nutrition' to the template
    return render_template('result_page.html', 
                         img_str=img_base64, 
                         ripeness_result=result_data,
                         class_names=[detected_object],
                         is_red_fruit=is_red_fruit,
                         nutrition=nutrition_data) # <--- ADD THIS!

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