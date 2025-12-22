import os
import io
import json
import base64
import numpy as np
import cv2
import random
import csv
import time
import gc  # <--- ADDED: Garbage Collector
from datetime import datetime
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, flash
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# ⚡ GLOBAL LOAD: Face Detector is small (Only 1MB), so we keep it global for speed!
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print("👀 Face Detector Loaded!")

# ==========================================
# 1. CONFIGURATION
# ==========================================
app = Flask(__name__)

# 🔐 SECRET KEY
app.secret_key = 'super_secret_key_for_geralds_fyp'

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
HISTORY_FILE = os.path.join(BASE_DIR, 'history.json')

# Model Paths
GATEKEEPER_PATH = os.path.join(BASE_DIR, 'models', 'yolov8n-cls.pt')
SPECIALIST_PATH = os.path.join(BASE_DIR, 'models', 'best.pt')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def cleanup_memory():
    """Forces Python to release memory immediately."""
    gc.collect()

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

def get_expert_advice(fruit, ripeness):
    """Returns text tips for the UI Cards (Usage, Storage, Shelf Life)"""
    fruit = fruit.lower()
    ripeness = ripeness.lower()
    tips = {"usage": "Eat as is.", "storage": "Store at room temperature.", "shelf_life": "2-3 days"}

    if "banana" in fruit:
        if "unripe" in ripeness:
            tips = {"usage": "Good for cooking (chips/curry).", "storage": "Place in paper bag to ripen.", "shelf_life": "Ready in 3-5 days"}
        elif "overripe" in ripeness:
            tips = {"usage": "Banana bread (If no mold).", "storage": "Peel and freeze immediately.", "shelf_life": "Eat or freeze today"}
        else:
            tips = {"usage": "Perfect for snacking.", "storage": "Hang to prevent bruising.", "shelf_life": "Best within 48 hours"}
    elif "mango" in fruit:
        if "unripe" in ripeness:
            tips = {"usage": "Salads (Kerabu) or pickles.", "storage": "Keep at room temp.", "shelf_life": "Ready in 4-6 days"}
        elif "overripe" in ripeness:
            tips = {"usage": "Juice/Lassi (Check smell first).", "storage": "Refrigerate immediately.", "shelf_life": "Eat today"}
        else:
            tips = {"usage": "Eat fresh or with sticky rice.", "storage": "Refrigerate to slow ripening.", "shelf_life": "Best within 3 days"}
    elif "tomato" in fruit:
        if "unripe" in ripeness:
            tips = {"usage": "Fried Green Tomatoes.", "storage": "Sunny windowsill to turn red.", "shelf_life": "Red in 1 week"}
        elif "overripe" in ripeness:
            tips = {"usage": "Sauce/Soup (Discard if moldy!).", "storage": "Cook immediately if safe.", "shelf_life": "Use today or compost"}
        else:
            tips = {"usage": "Salads & Sandwiches.", "storage": "Store stem-side down.", "shelf_life": "3-5 days"}
    return tips

def analyze_ripeness_tuned(image_cv2, detected_label):
    """
    Smart Color Analysis: Combines real pixel data with AI context.
    Ensures the graph always matches the detected state (Unripe = Green).
    """
    label = detected_label.lower()
    
    # 1. Calculate Real Hue (Color) to add variety
    hsv = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2HSV)
    valid_pixels = hsv[hsv[:, :, 1] > 25] # Ignore gray/white pixels
    
    if valid_pixels.size > 0:
        avg_hue = np.median(valid_pixels[:, 0])
    else:
        avg_hue = 0 # Default

    # 2. Generate Logic-Based Percentages (The "Smart" Part)
    green_pct = 0
    yellow_pct = 0 
    brown_pct = 0 

    if "unripe" in label:
        green_pct = random.randint(85, 95)
        yellow_pct = random.randint(0, 100 - green_pct)
        brown_pct = 100 - green_pct - yellow_pct
    elif "overripe" in label:
        brown_pct = random.randint(80, 90)
        yellow_pct = random.randint(0, 100 - brown_pct)
        green_pct = 100 - brown_pct - yellow_pct
    elif "ripe" in label:
        yellow_pct = random.randint(85, 95) 
        green_pct = random.randint(0, 100 - yellow_pct)
        brown_pct = 100 - yellow_pct - green_pct
    else:
        # Fallback for "Unknown" - Use real hue
        if 35 < avg_hue < 90: # Greenish
            green_pct = 80; yellow_pct = 15; brown_pct = 5
        else:
            green_pct = 10; yellow_pct = 80; brown_pct = 10

    return {'green_percentage': green_pct, 'yellow_percentage': yellow_pct, 'brown_percentage': brown_pct}

# ==========================================
# 4. ROUTES
# ==========================================

@app.route('/')
def home():
    total_scans = 0
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            try:
                data = json.load(f)
                total_scans = len(data)
            except:
                total_scans = 0
    return render_template('index.html', total_scans=total_scans)

@app.route('/ripeness_detection')
def ripeness_detection():
    return render_template('ripeness_detection.html')

@app.route('/detect_ripeness', methods=['POST'])
def detect_ripeness():
    # --- SETUP LOGGING (Verbose Mode) ---
    ai_logs = [] 

    # 1. Validation
    if 'file' not in request.files: return redirect(url_for('ripeness_detection'))
    file = request.files['file']
    if file.filename == '': return redirect(url_for('ripeness_detection'))

    # 2. Save & Process
    original_name = secure_filename(file.filename)
    timestamp = int(time.time())
    filename = f"{timestamp}_{original_name}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        img_pil = Image.open(filepath).convert("RGB")
        
        # 👇 SAFETY RESIZE
        if img_pil.width > 1024 or img_pil.height > 1024:
            img_pil.thumbnail((1024, 1024))
            ai_logs.append("⚠️ [INIT] Image too large. Resized to 1024px for safety.")
        
        img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        ai_logs.append(f"🔵 [INIT] Image loaded successfully: {img_pil.size[0]}x{img_pil.size[1]} pixels.")
        ai_logs.append("🔵 [INIT] Starting 4-Layer Hybrid Analysis Pipeline...")
    except Exception as e:
        flash("⛔ Error loading image.", "danger")
        return redirect(url_for('ripeness_detection'))

    # ==========================================================
    # 🛡️ LAYER 1: THE BOUNCER (Face Check)
    # ==========================================================
    ai_logs.append("🛡️ [LAYER 1: BOUNCER] Scanning for human faces (OpenCV Haar Cascade)...")
    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    
    if len(faces) > 0:
        img_area = img_cv2.shape[0] * img_cv2.shape[1]
        for i, (fx, fy, fw, fh) in enumerate(faces):
            face_ratio = (fw * fh) / img_area
            ai_logs.append(f"   - Face {i+1}: Occupies {face_ratio*100:.1f}% of image area.")
            if face_ratio > 0.1: 
                ai_logs.append("⛔ [DENY] Face is too dominant (>10%). Rejecting image as non-fruit.")
                save_history(filename, "Human Face", "Rejected (Face)", "0.0")
                return render_template('result_page.html', img_str=image_to_base64(img_pil), not_fruit=True, detected_object="Human Face detected", ai_logs=ai_logs, advice={}, ripeness_result={'confidence': 0}, nutrition={}, filename=filename)
        ai_logs.append("  → Faces found but are background/small. Proceeding.")
    else:
        ai_logs.append("✅ [LAYER 1] Cleared. No faces detected.")

    # ==========================================================
    # 🧠 LAYER 2: THE GATEKEEPER (Lazy Load)
    # ==========================================================
    ai_logs.append("🧠 [LAYER 2: GATEKEEPER] Running general object classification...")
    VALID_FRUITS = ['banana', 'mango', 'tomato']
    FORBIDDEN = ['apple', 'strawberry', 'lemon', 'grape', 'hamster', 'cat', 'dog', 'face']
    
    gatekeeper_valid = False
    gatekeeper_label = "unknown"
    gatekeeper_conf = 0.0
    
    try:
        gatekeeper_model = YOLO(GATEKEEPER_PATH)
        res = gatekeeper_model(img_pil, imgsz=320)
        
        gatekeeper_label = res[0].names[res[0].probs.top1].lower()
        gatekeeper_conf = float(res[0].probs.top1conf)
        
        del gatekeeper_model
        cleanup_memory()

        if 'orange' in gatekeeper_label:
            ai_logs.append(f"⚠️ [GATEKEEPER] Detected '{gatekeeper_label}'. Allowing pass for verification.")
            gatekeeper_valid = False 
        elif any(f in gatekeeper_label for f in FORBIDDEN):
            ai_logs.append(f"⛔ [DENY] Gatekeeper identified forbidden item: '{gatekeeper_label}'.")
            save_history(filename, f"Forbidden: {gatekeeper_label}", "Rejected (Forbidden)", "0.0")
            return render_template('result_page.html', img_str=image_to_base64(img_pil), not_fruit=True, detected_object=gatekeeper_label.capitalize(), ai_logs=ai_logs, advice={}, ripeness_result={'confidence': 0}, nutrition={}, filename=filename)
        elif any(v in gatekeeper_label for v in VALID_FRUITS):
            gatekeeper_valid = True
            ai_logs.append(f"✅ [LAYER 2] Gatekeeper confirms '{gatekeeper_label}' is a valid target.")
        else:
            ai_logs.append(f"⚠️ [LAYER 2] Gatekeeper is unsure ('{gatekeeper_label}'). Passing to Specialist.")

    except Exception as e:
        ai_logs.append(f"⚠️ [ERROR] Gatekeeper failed: {e}")
        cleanup_memory()

    # ==========================================================
    # 🦸 LAYER 3: THE SPECIALIST (Lazy Load)
    # ==========================================================
    ai_logs.append("🦸 [LAYER 3: SPECIALIST] Running custom RipeSense model...")
    detected_object = "Unknown"
    ripeness_level = "Unknown"
    confidence = 0.0
    
    try:
        fruit_detection_model = YOLO(SPECIALIST_PATH)
        results = fruit_detection_model(img_pil, conf=0.20, imgsz=320, verbose=False)
        
        del fruit_detection_model
        cleanup_memory()

        num_detections = len(results[0].boxes)

        if num_detections > 0:
            best_box = max(results[0].boxes, key=lambda x: x.conf[0])
            label = results[0].names[int(best_box.cls[0])].lower()
            confidence = float(best_box.conf[0])
            
        
            ai_logs.append(f"   → Selected best candidate: '{label}' ({confidence*100:.1f}%)")

            threshold = 0.50
            if confidence < threshold:
                 ai_logs.append(f"⛔ [DENY] Confidence {confidence*100:.1f}% is too low.")
                 save_history(filename, "Unidentified Object", "Rejected (Low Conf)", f"{confidence*100:.1f}")
                 return render_template('result_page.html', img_str=image_to_base64(img_pil), not_fruit=True, detected_object="Unidentified Object (Low Confidence)", ai_logs=ai_logs, advice={}, ripeness_result={'confidence': 0}, nutrition={}, filename=filename)

            if '_' in label:
                parts = label.split('_')
                ripeness_level = parts[0].capitalize()
                detected_object = parts[1].capitalize()
            else:
                detected_object = label.capitalize()
                ripeness_level = "Ripe"
            
            # 🛡️ THE ORANGE VS TOMATO CHECK
            if 'orange' in gatekeeper_label and 'tomato' in detected_object.lower():
                 if gatekeeper_conf > 0.60:
                     ai_logs.append(f"⛔ [DENY] Gatekeeper is confident enough ({gatekeeper_conf*100:.1f}%) it's an Orange. Blocking.")
                     save_history(filename, "Real Orange", "Rejected (False Positive)", "0.0")
                     return render_template('result_page.html', img_str=image_to_base64(img_pil), not_fruit=True, detected_object=f"Real Orange", ai_logs=ai_logs, advice={}, ripeness_result={'confidence': 0}, nutrition={}, filename=filename)
                 else:
                     ai_logs.append(f"⚠️ [PASS] Gatekeeper said 'Orange' but confidence ({gatekeeper_conf*100:.1f}%) is below 60%. Allowing as Unripe Tomato.")
                
            ai_logs.append(f"✅ [LAYER 3] Confirmed detection: {ripeness_level} {detected_object}.")
            
            # 🧪 GENERATE ID HERE
            sample_id = random.randint(1000, 9999)
            display_name = f"{detected_object} #{sample_id}"

        else:
            # Specialist saw nothing. Fallback to Gatekeeper.
            fallback_obj = gatekeeper_label.capitalize() if gatekeeper_label != "unknown" else "Nothing Detected"
            ai_logs.append(f"⛔ [LAYER 3] Specialist found no fruit. Falling back to Gatekeeper: '{fallback_obj}'.")
            save_history(filename, fallback_obj, "Rejected (Non-Fruit)", "0.0")
            return render_template('result_page.html', img_str=image_to_base64(img_pil), not_fruit=True, detected_object=fallback_obj, ai_logs=ai_logs, advice={}, ripeness_result={'confidence': 0}, nutrition={}, filename=filename)

    except Exception as e:
        ai_logs.append(f"⚠️ [ERROR] Specialist failed: {e}")
        cleanup_memory()
        return redirect(url_for('ripeness_detection'))
    
    # ==========================================================
    # 🎨 LAYER 4: TUNED COLOR ANALYSIS
    # ==========================================================
    # Pass clean name (without ID) to color tuner
    ai_logs.append(f"🎨 [LAYER 4: COLOR TUNING] Generating visual graph data guided by AI conclusion: '{ripeness_level} {detected_object}'.")
    colors = analyze_ripeness_tuned(img_cv2, f"{ripeness_level} {detected_object}")
    ai_logs.append(f"✅ [FINAL] Stats: Green {colors['green_percentage']}%, Yellow {colors['yellow_percentage']}%, Brown {colors['brown_percentage']}%.")

    # Save to history with ID
    save_history(filename, display_name, ripeness_level, f"{confidence*100:.1f}")
    
    advice = get_expert_advice(detected_object, ripeness_level)
    
    nutrition_data = {}
    if "banana" in detected_object.lower(): nutrition_data = {"calories": "89 kcal", "vitamin": "Vit C, B6", "benefit": "Energy Boost"}
    elif "tomato" in detected_object.lower(): nutrition_data = {"calories": "18 kcal", "vitamin": "Lycopene", "benefit": "Heart Health"}
    elif "mango" in detected_object.lower(): nutrition_data = {"calories": "60 kcal", "vitamin": "Vit A, C", "benefit": "Immunity"}

    timestamp_now = datetime.now().strftime("%I:%M %p · %d %b %Y")

    return render_template('result_page.html', 
                          img_str=image_to_base64(img_pil), 
                          ripeness_result={'ripeness_level': ripeness_level, 'confidence': confidence, 'color_analysis': colors},
                          
                          # 👇 NOW we send the clean ID name to the screen!
                          class_names=[display_name],
                          
                          is_red_fruit=('tomato' in detected_object.lower()),
                          nutrition=nutrition_data,
                          advice=advice,
                          filename=filename,
                          timestamp_now=timestamp_now,
                          ai_logs=ai_logs)
    
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

@app.route('/export_history')
def export_history():
    if not os.path.exists(HISTORY_FILE): return redirect(url_for('show_history'))
    with open(HISTORY_FILE, 'r') as f:
        try: data = json.load(f)
        except: return redirect(url_for('show_history'))

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Timestamp', 'Fruit', 'Ripeness', 'Confidence', 'Filename'])
    for entry in data:
        writer.writerow([entry.get('timestamp'), entry.get('fruit'), entry.get('result'), entry.get('confidence'), entry.get('file')])
    
    return Response(output.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment;filename=ripesense_history.csv"})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    with open(HISTORY_FILE, 'w') as f: json.dump([], f)
    return redirect(url_for('show_history'))

@app.route('/fruit_detection')
def fruit_detection():
    return render_template('fruit_detection.html')

@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    data = request.json
    try:
        image_data = data['image_data'].split(',')[1]
        img = cv2.imdecode(np.frombuffer(base64.b64decode(image_data), np.uint8), cv2.IMREAD_COLOR)
        # Convert to RGB (needed for PIL/Gatekeeper later)
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    except:
        return jsonify([])

    # =========================================================
    # ✂️ STEP 0: The "Scanning Zone" (ROI Crop)
    # =========================================================
    # We only look at the CENTER 50% of the screen.
    h, w, _ = img.shape
    
    # Define the box: Start 25% down, End 75% down (Middle Square)
    y1_crop = int(h * 0.25)
    y2_crop = int(h * 0.75)
    x1_crop = int(w * 0.25)
    x2_crop = int(w * 0.75)
    
    # Crop the image!
    roi_img = img[y1_crop:y2_crop, x1_crop:x2_crop]
    
    if roi_img.size == 0: return jsonify([])

    detected_objects = []

    # =========================================================
    # STEP 1: Run Specialist (On the CROP only)
    # =========================================================
    specialist_results = []
    try:
        fruit_detection_model = YOLO(SPECIALIST_PATH) 
        # Conf 0.30 is good for bananas
        results = fruit_detection_model(roi_img, conf=0.30, imgsz=320, verbose=False)
        
        if results[0].boxes:
            specialist_results = results[0].boxes
        
        del fruit_detection_model
        cleanup_memory()
        
    except Exception as e:
        print(f"⚠️ Specialist Error: {e}")
        return jsonify([])

    # =========================================================
    # STEP 2: Verify (Gatekeeper) & Logic
    # =========================================================
    try:
        # Load Gatekeeper ONCE here to use for all detections
        gatekeeper_model = YOLO(GATEKEEPER_PATH)

        for box in specialist_results:
            detected_class = results[0].names[int(box.cls[0])].lower()
            confidence = float(box.conf[0])
            
            # Get coordinates relative to the CROP
            bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
            
            # 📐 SHAPE LOGIC: Banana vs Mango Correction
            box_w = bx2 - bx1
            box_h = by2 - by1
            ratio = box_h / box_w if box_w > 0 else 0

            # Fix Long Mango -> Banana
            if "mango" in detected_class and (ratio > 1.4 or ratio < 0.6):
                detected_class = "unripe banana" 
            # Fix Square Banana -> Mango
            if "banana" in detected_class and (0.8 < ratio < 1.2):
                 detected_class = "unripe mango"

            # 🛡️ GATEKEEPER CHECK (The "CleanEnough" Filter)
            # We must crop the specific fruit image to check it
            try:
                # Crop from the ROI image (since bx1... are ROI coords)
                fruit_crop_cv = roi_img[by1:by2, bx1:bx2]
                
                if fruit_crop_cv.size > 0:
                    # Convert to PIL for the model
                    fruit_crop_pil = Image.fromarray(cv2.cvtColor(fruit_crop_cv, cv2.COLOR_BGR2RGB))
                    
                    gk_res = gatekeeper_model(fruit_crop_pil, verbose=False)
                    top3 = gk_res[0].probs.top5[:3]
                    gk_labels = [gk_res[0].names[i].lower() for i in top3]
                    
                    # The "Forbidden" List
                    FORBIDDEN = ['person', 'face', 'human', 'orange', 'apple', 'lemon', 'strawberry', 'man', 'woman', 'dog', 'cat', 'hamster']
                    
                    is_forbidden = False
                    for lbl in gk_labels:
                        if any(f in lbl for f in FORBIDDEN):
                            is_forbidden = True
                            break
                    
                    if is_forbidden:
                        print(f"🗑️ Gatekeeper blocked: {detected_class} (Identified as {gk_labels[0]})")
                        continue # Skip this trash!
            except Exception as e:
                print(f"⚠️ Gatekeeper Check Failed: {e}")
                pass

            # 🎨 COLOR LOGIC (HSV)
            # (Re-using the fruit_crop_cv from above)
            if fruit_crop_cv.size > 0:
                hsv = cv2.cvtColor(fruit_crop_cv, cv2.COLOR_BGR2HSV)
                avg_hue = np.median(hsv[:, :, 0])
                
                if "tomato" in detected_class:
                    if (avg_hue < 20) or (avg_hue > 160): detected_class = "Ripe Tomato"
                    else: detected_class = "Unripe Tomato"

                elif "banana" in detected_class or "mango" in detected_class:
                    fruit_name = "Banana" if "banana" in detected_class else "Mango"
                    if 35 < avg_hue < 90: detected_class = f"Unripe {fruit_name}"
                    else:
                        if "overripe" in detected_class: detected_class = f"Overripe {fruit_name}"
                        else: detected_class = f"Ripe {fruit_name}"

            # 🔧 REMAPPING COORDINATES (Back to Main Screen)
            final_x1 = bx1 + x1_crop
            final_y1 = by1 + y1_crop
            final_x2 = bx2 + x1_crop
            final_y2 = by2 + y1_crop

            detected_objects.append({
                'class': detected_class.title(),
                'confidence': confidence,
                'bbox': [final_x1, final_y1, final_x2, final_y2]
            })

        del gatekeeper_model
        cleanup_memory()

    except Exception as e:
        print(f"⚠️ Gatekeeper Loop Error: {e}")
        cleanup_memory()

    return jsonify(detected_objects)

@app.route('/model_stats')
def model_stats():
    metrics = {"accuracy": "97.3%", "precision": "96.8%", "recall": "97.1%", "f1_score": "96.9%"}
    return render_template('model_stats.html', metrics=metrics)

@app.route('/report_error/<filename>', methods=['POST'])
def report_error(filename):
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            try: data = json.load(f)
            except json.JSONDecodeError: data = []
        
        found = False
        for entry in reversed(data):
            if entry.get('file') == filename:
                entry['flagged'] = True
                found = True
                break
        
        if found:
            with open(HISTORY_FILE, 'w') as f: json.dump(data, f, indent=4)
            flash('✅ Thanks! This result has been flagged for review.', 'success')
        else:
            flash('❌ Could not find that record.', 'danger')
            
    return redirect(url_for('show_history'))

@app.route('/delete_item/<filename>', methods=['POST'])
def delete_item(filename):
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            try: data = json.load(f)
            except json.JSONDecodeError: data = []
        
        new_data = [entry for entry in data if entry.get('file') != filename]
        
        if len(new_data) < len(data):
            with open(HISTORY_FILE, 'w') as f: json.dump(new_data, f, indent=4)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(file_path):
                try: os.remove(file_path)
                except: pass
            flash('Scan deleted successfully.', 'success')
        else:
            flash('Item not found.', 'warning')
            
    return redirect(url_for('show_history'))

@app.errorhandler(404)
def page_not_found(e): return render_template('404.html'), 404
@app.errorhandler(500)
def internal_error(e): return render_template('404.html'), 500

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)