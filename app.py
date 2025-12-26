import os
import io
import json
import base64
import numpy as np
import cv2
import random
import csv
import time
import gc
from datetime import datetime
from threading import Lock
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, flash
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# ⚡ GLOBAL LOAD: Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print("👀 Face Detector Loaded!")

# 🔒 THREAD SAFETY
history_lock = Lock()

# ==========================================
# 1. CONFIGURATION
# ==========================================
app = Flask(__name__)
app.secret_key = 'super_secret_key_for_geralds_fyp'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
HISTORY_FILE = os.path.join(BASE_DIR, 'history.json')
GATEKEEPER_PATH = os.path.join(BASE_DIR, 'models', 'yolov8n-cls.pt')
SPECIALIST_PATH = os.path.join(BASE_DIR, 'models', 'best.pt')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def cleanup_memory():
    gc.collect()

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_image_quality(img_cv2):
    """Checks if image is too dark or blurry."""
    hsv = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2HSV)
    avg_brightness = np.mean(hsv[:, :, 2])
    
    if avg_brightness < 60:
        return False, "lighting_error", f"Image is too dark (Brightness: {avg_brightness:.1f}). Please turn on a light."

    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if blur_score < 50:
        return False, "blur_error", f"Image is too blurry (Score: {blur_score:.1f}). Hold steady."

    return True, "ok", "Quality OK"

def save_history(filename, fruit, ripeness, confidence):
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file": filename,
        "fruit": fruit,
        "result": ripeness,
        "confidence": f"{confidence}%"
    }
    with history_lock:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                try: data = json.load(f)
                except json.JSONDecodeError: data = []
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

def get_nutrition_data(fruit):
    """Returns nutrition facts for the fruit."""
    fruit = fruit.lower()
    nutrition = {"calories": "N/A", "vitamin": "N/A", "benefit": "N/A"}
    
    if "banana" in fruit:
        nutrition = {"calories": "89 kcal", "vitamin": "Vit C, B6", "benefit": "Energy Boost"}
    elif "tomato" in fruit:
        nutrition = {"calories": "18 kcal", "vitamin": "Lycopene", "benefit": "Heart Health"}
    elif "mango" in fruit:
        nutrition = {"calories": "60 kcal", "vitamin": "Vit A, C", "benefit": "Immunity"}
    
    return nutrition

def analyze_ripeness_real(img_cv2, detected_object):
    """
    Analyzes color based on the SPECIFIC FRUIT type and normalizes results.
    """
    hsv = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2HSV)
    label = detected_object.lower()
    
    if 'tomato' in label:
        # Green (Unripe) - Expanded
        lower_green = np.array([25, 40, 40]) 
        upper_green = np.array([90, 255, 255])
        
        # Red (Ripe) - Wraparound range
        lower_red1 = np.array([0, 50, 50]); upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([160, 50, 50]); upper_red2 = np.array([180, 255, 255])
        
        # Brown (Rotten)
        lower_brown = np.array([0, 20, 20])
        upper_brown = np.array([20, 255, 100])
        
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_ripe = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    else:
        # Banana/Mango Logic
        lower_green = np.array([35, 40, 40]); upper_green = np.array([85, 255, 255])
        lower_yellow = np.array([20, 40, 40]); upper_yellow = np.array([35, 255, 255])
        lower_brown = np.array([0, 50, 20]); upper_brown = np.array([20, 255, 200])
        
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_ripe = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    count_green = cv2.countNonZero(mask_green)
    count_ripe = cv2.countNonZero(mask_ripe)
    count_brown = cv2.countNonZero(mask_brown)
    
    total_detected = count_green + count_ripe + count_brown
    
    if total_detected == 0: 
        return {'green_percentage': 0, 'yellow_percentage': 0, 'brown_percentage': 0}

    green_pct = int((count_green / total_detected) * 100)
    ripe_pct = int((count_ripe / total_detected) * 100)
    brown_pct = 100 - green_pct - ripe_pct
    
    return {
        'green_percentage': green_pct, 
        'yellow_percentage': ripe_pct, 
        'brown_percentage': brown_pct
    }

# ==========================================
# 4. ROUTES
# ==========================================

@app.route('/')
def home():
    total_scans = 0
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            try: total_scans = len(json.load(f))
            except: pass
    return render_template('index.html', total_scans=total_scans)

@app.route('/ripeness_detection')
def ripeness_detection():
    return render_template('ripeness_detection.html')

@app.route('/detect_ripeness', methods=['POST'])
def detect_ripeness():
    ai_logs = []
    
    if 'file' not in request.files: return redirect(url_for('ripeness_detection'))
    file = request.files['file']
    if file.filename == '': return redirect(url_for('ripeness_detection'))

    timestamp = int(time.time())
    filename = f"{timestamp}_{secure_filename(file.filename)}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        img_pil = Image.open(filepath).convert("RGB")
        if img_pil.width > 1024 or img_pil.height > 1024: img_pil.thumbnail((1024, 1024))
        img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        timestamp_now = datetime.now().strftime("%I:%M %p · %d %b %Y")
        
        # 1. INIT LOG
        ai_logs.append(f"🔵 [INIT] Image loaded successfully: {img_pil.size[0]}x{img_pil.size[1]} pixels.")
        ai_logs.append("🔵 [INIT] Starting 4-Layer Hybrid Analysis Pipeline...")

        # 2. QUALITY CHECK
        is_good, err_type, err_msg = check_image_quality(img_cv2)
        if not is_good:
            ai_logs.append(f"⛔ [DENY] {err_msg}")
            save_history(filename, "Quality Error", "Rejected", "0.0")
            return render_template('result_page.html', img_str=image_to_base64(img_pil), not_fruit=True, detected_object="Poor Quality", ai_logs=ai_logs, advice={"usage": "Try again with better lighting."}, ripeness_result={'confidence': 0}, nutrition={}, filename=filename, timestamp_now=timestamp_now, class_names=["Quality Error"])

        # ==========================================================
        # 🛡️ LAYER 1: SMART FACE CHECK (Haar + YOLO Verification)
        # ==========================================================
        ai_logs.append("🛡️ [LAYER 1: BOUNCER] Scanning for human faces...")
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 7, minSize=(30,30))
        
        if len(faces) > 0:
            ai_logs.append(f"⚠️ [LAYER 1] Potential face detected. Verifying with Gatekeeper AI...")
            is_really_a_face = False
            try:
                gatekeeper_model = YOLO(GATEKEEPER_PATH)
                for (fx, fy, fw, fh) in faces:
                    face_crop = img_pil.crop((fx, fy, fx+fw, fy+fh))
                    res = gatekeeper_model(face_crop, imgsz=224, verbose=False)
                    top_label = res[0].names[res[0].probs.top1].lower()
                    
                    face_keywords = ['person', 'face', 'man', 'woman', 'boy', 'girl', 'human']
                    if any(k in top_label for k in face_keywords):
                        is_really_a_face = True
                        break
                    else:
                        ai_logs.append(f"⚠️ [FALSE ALARM] Haar saw a face, but AI says it's '{top_label}'. Ignoring.")
                
                del gatekeeper_model
                cleanup_memory()
            except:
                is_really_a_face = True

            if is_really_a_face:
                ai_logs.append(f"⛔ [DENY] Confirmed Human Face detected. Blocking.")
                save_history(filename, "Human Face", "Rejected", "100.0")
                return render_template('result_page.html', img_str=image_to_base64(img_pil), not_fruit=True, detected_object="Human Face", ai_logs=ai_logs, advice={}, ripeness_result={'confidence': 0}, nutrition={}, filename=filename, timestamp_now=timestamp_now, class_names=["Face Detected"])
        
        ai_logs.append("✅ [LAYER 1] Cleared. No faces detected.")

        # ✂️ 3. PRE-PROCESS (ROI CROP)
        ai_logs.append("🔵 [PRE-PROCESS] Cropping to center 50% ROI to avoid background noise...")
        h, w, _ = img_cv2.shape
        y_start, y_end = int(h * 0.25), int(h * 0.75)
        x_start, x_end = int(w * 0.25), int(w * 0.75)
        img_pil_cropped = img_pil.crop((x_start, y_start, x_end, y_end))
        img_cv2_cropped = img_cv2[y_start:y_end, x_start:x_end]
        ai_logs.append(f"  → New analysis dimensions: {img_pil_cropped.size[0]}x{img_pil_cropped.size[1]} pixels.")
        
        # 🧠 4. LAYER 2: GATEKEEPER
        ai_logs.append("🧠 [LAYER 2: GATEKEEPER] Running general classification...")
        gatekeeper_label = "unknown"
        gatekeeper_conf = 0.0
        is_suspicious = False
        FORBIDDEN = ['apple', 'strawberry', 'lemon', 'grape', 'hamster', 'cat', 'dog', 'face', 'samoyed', 'rabbit', 'rodent', 'fig', 'pear', 'peach', 'plum', 'person', 'man', 'woman', 'neck_brace']
        
        try:
            gk_model = YOLO(GATEKEEPER_PATH)
            res = gk_model(img_pil_cropped, imgsz=320, verbose=False)
            gatekeeper_label = res[0].names[res[0].probs.top1].lower()
            gatekeeper_conf = float(res[0].probs.top1conf)
            del gk_model
            cleanup_memory()

            if any(f in gatekeeper_label for f in FORBIDDEN):
                if gatekeeper_conf > 0.40:
                    ai_logs.append(f"⛔ [DENY] Gatekeeper is certain it's a '{gatekeeper_label}' ({gatekeeper_conf*100:.1f}%).")
                    save_history(filename, f"Forbidden: {gatekeeper_label}", "Rejected", "0.0")
                    return render_template('result_page.html', img_str=image_to_base64(img_pil), not_fruit=True, detected_object=gatekeeper_label.capitalize(), ai_logs=ai_logs, advice={}, ripeness_result={'confidence': 0}, nutrition={}, filename=filename, timestamp_now=timestamp_now, class_names=[gatekeeper_label])
                elif gatekeeper_conf > 0.15:
                    is_suspicious = True
                    ai_logs.append(f"⚠️ [FLAG] Gatekeeper suspects '{gatekeeper_label}' ({gatekeeper_conf*100:.1f}%). Requiring high proof from Specialist.")
            else:
                 ai_logs.append(f"✅ [LAYER 2] Gatekeeper confirms '{gatekeeper_label}' ({gatekeeper_conf*100:.1f}%) is valid.")
        except: pass

        # 🦸 5. LAYER 3: SPECIALIST
        ai_logs.append("🦸 [LAYER 3: SPECIALIST] Running RipeSense model...")
        fruit_model = YOLO(SPECIALIST_PATH)
        results = fruit_model(img_pil_cropped, conf=0.20, imgsz=320, verbose=False)
        del fruit_model
        cleanup_memory()

        if results[0].boxes:
            best_box = max(results[0].boxes, key=lambda x: x.conf[0])
            label = results[0].names[int(best_box.cls[0])].lower()
            confidence = float(best_box.conf[0])
            
            # Shape Logic
            # x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
            # ratio = (y2-y1) / (x2-x1) if (x2-x1) > 0 else 0
            
            # if "banana" in label and (0.8 < ratio < 1.2): 
            #     ai_logs.append(f"🍌➡️🥭 [SHAPE FIX] AI saw 'Banana' but shape is square. Correcting to Mango.")
            #     label = label.replace("banana", "mango")
            # elif "mango" in label and (ratio > 1.4 or ratio < 0.6): 
            #     ai_logs.append(f"🥭➡️🍌 [SHAPE FIX] AI saw 'Mango' but shape is long. Correcting to Banana.")
            #     label = label.replace("mango", "banana")

            # ai_logs.append(f"   → Selected candidate: '{label}' ({confidence*100:.1f}%)")

            # Suspicion Checks
            if is_suspicious:
                if "mango" in label:
                    ai_logs.append("⛔ [DENY] Suspicious item identified as Mango. Likely Apple/Face.")
                    save_history(filename, "Suspicious: Mango", "Rejected", "0.0")
                    return render_template('result_page.html', img_str=image_to_base64(img_pil), not_fruit=True, detected_object="Suspicious Object", ai_logs=ai_logs, advice={}, ripeness_result={'confidence': 0}, nutrition={}, filename=filename, timestamp_now=timestamp_now, class_names=["Rejected"])
                elif confidence < 0.85:
                    ai_logs.append(f"⛔ [DENY] Suspicious item identified as '{label}' but confidence ({confidence*100:.1f}%) is too low.")
                    save_history(filename, f"Suspicious: {label}", "Rejected", f"{confidence*100:.1f}")
                    return render_template('result_page.html', img_str=image_to_base64(img_pil), not_fruit=True, detected_object="Unknown", ai_logs=ai_logs, advice={}, ripeness_result={'confidence': 0}, nutrition={}, filename=filename, timestamp_now=timestamp_now, class_names=["Rejected"])
                else:
                    ai_logs.append(f"✅ [OVERRIDE] Specialist is {confidence*100:.1f}% sure. Overriding Gatekeeper suspicion.")

            # Red Mango Check
            if "mango" in label:
                hsv_check = cv2.cvtColor(img_cv2_cropped, cv2.COLOR_BGR2HSV)
                avg_hue = np.median(hsv_check[:, :, 0])
                if avg_hue < 8 or avg_hue > 155:
                     ai_logs.append(f"⛔ [DENY] 'Mango' is too Red/Brown (Hue {avg_hue}). Likely Apple.")
                     save_history(filename, "Red Fruit (Apple?)", "Rejected", "0.0")
                     return render_template('result_page.html', img_str=image_to_base64(img_pil), not_fruit=True, detected_object="Red Fruit (Apple?)", ai_logs=ai_logs, advice={}, ripeness_result={'confidence': 0}, nutrition={}, filename=filename, timestamp_now=timestamp_now, class_names=["Red Fruit"])

            if confidence < 0.50:
                 ai_logs.append(f"⛔ [DENY] Confidence {confidence*100:.1f}% is too low.")
                 save_history(filename, "Low Confidence", "Rejected", f"{confidence*100:.1f}")
                 return render_template('result_page.html', img_str=image_to_base64(img_pil), not_fruit=True, detected_object="Low Confidence", ai_logs=ai_logs, advice={}, ripeness_result={'confidence': 0}, nutrition={}, filename=filename, timestamp_now=timestamp_now, class_names=["Unknown"])

            # Success
            parts = label.split('_') if '_' in label else ["Ripe", label]
            ripeness_level, detected_object = parts[0].capitalize(), parts[1].capitalize() if len(parts)>1 else parts[0].capitalize()
            
            ai_logs.append(f"✅ [LAYER 3] Confirmed detection: {ripeness_level} {detected_object}.")
            
            # 🎨 LAYER 4: REAL COLOR ANALYSIS (Context Aware)
            ai_logs.append(f"🎨 [LAYER 4] Analyzing colors for: '{ripeness_level} {detected_object}'...")
            colors = analyze_ripeness_real(img_cv2_cropped, detected_object)
            ai_logs.append(f"✅ [FINAL] Green {colors['green_percentage']}%, Yellow {colors['yellow_percentage']}%, Brown {colors['brown_percentage']}%.")

            save_history(filename, f"{detected_object}", ripeness_level, f"{confidence*100:.1f}")
            
            return render_template('result_page.html', 
                                  img_str=image_to_base64(img_pil), 
                                  ripeness_result={'ripeness_level': ripeness_level, 'confidence': confidence, 'color_analysis': colors},
                                  class_names=[f"{detected_object} #{random.randint(1000,9999)}"],
                                  is_red_fruit=('tomato' in detected_object.lower()),
                                  nutrition=get_nutrition_data(detected_object),
                                  advice=get_expert_advice(detected_object, ripeness_level),
                                  filename=filename,
                                  timestamp_now=timestamp_now,
                                  ai_logs=ai_logs)
        else:
            ai_logs.append(f"⛔ [LAYER 3] No fruit found. Fallback: '{gatekeeper_label}'.")
            save_history(filename, "No Fruit Found", "Rejected", "0.0")
            return render_template('result_page.html', img_str=image_to_base64(img_pil), not_fruit=True, detected_object="No Fruit Found", ai_logs=ai_logs, advice={}, ripeness_result={'confidence': 0}, nutrition={}, filename=filename, timestamp_now=timestamp_now, class_names=["Nothing"])

    except Exception as e:
        print(f"Server Error: {e}")
        return redirect(url_for('ripeness_detection'))

@app.route('/history')
def show_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            try: data = json.load(f)
            except: data = []
        return render_template('history.html', history=reversed(data))
    return render_template('history.html', history=[])

@app.route('/report_error/<filename>', methods=['POST'])
def report_error(filename):
    with history_lock:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                try: data = json.load(f)
                except json.JSONDecodeError: data = []
        else:
            data = []
        
        found = False
        for entry in data:
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

@app.route('/model_stats')
def model_stats():
    metrics = {"accuracy": "97.3%", "precision": "96.8%", "recall": "97.1%", "f1_score": "96.9%"}
    return render_template('model_stats.html', metrics=metrics)

@app.route('/export_history')
def export_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f: data = json.load(f)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Timestamp', 'Fruit', 'Ripeness', 'Confidence'])
        for e in data: writer.writerow([e['timestamp'], e['fruit'], e['result'], e['confidence']])
        return Response(output.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment;filename=history.csv"})
    return redirect(url_for('show_history'))

@app.route('/clear_history', methods=['POST'])
def clear_history():
    with open(HISTORY_FILE, 'w') as f: json.dump([], f)
    return redirect(url_for('show_history'))

@app.route('/fruit_detection')
def fruit_detection(): return render_template('fruit_detection.html')

@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    try:
        # Decode Image
        data = request.json['image_data'].split(',')[1]
        img = cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_COLOR)
        if img is None: return jsonify([])

        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # ---------------------------------------------------------
        # 🛡️ STEP 1: FAST FACE CHECK (Haar Cascade)
        # ---------------------------------------------------------
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 7, minSize=(30,30))
        
        if len(faces) > 0:
            return jsonify([{
                'class': 'Face Detected (Ignored)',
                'confidence': 1.0,
                'bbox': [int(faces[0][0]), int(faces[0][1]), int(faces[0][0]+faces[0][2]), int(faces[0][1]+faces[0][3])]
            }])

        # ✂️ STEP 2: CROP TO CENTER 50% ROI (Focus on fruit, ignore background)
        h, w, _ = img.shape
        y_start, y_end = int(h * 0.25), int(h * 0.75)
        x_start, x_end = int(w * 0.25), int(w * 0.75)
        img_cropped = img[y_start:y_end, x_start:x_end]
        img_pil_cropped = img_pil.crop((x_start, y_start, x_end, y_end))

        # ---------------------------------------------------------
        # 🦸 STEP 3: SPECIALIST (Find Fruit) - LOWER CONFIDENCE
        # ---------------------------------------------------------
        model = YOLO(SPECIALIST_PATH)
        res = model(img_cropped, conf=0.25, verbose=False)  # ⬇️ LOWERED to 0.25 to match detect_ripeness
        detected = []
        
        if res[0].boxes:
            gatekeeper_model = YOLO(GATEKEEPER_PATH)
            
            for box in res[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                detected_class = res[0].names[int(box.cls[0])].lower()
                
                # Offset bbox back to original image coordinates
                x1_orig, y1_orig = x1 + x_start, y1 + y_start
                x2_orig, y2_orig = x2 + x_start, y2 + y_start
                
                fruit_type = detected_class.split('_')[-1] if '_' in detected_class else detected_class
                
                # --- A. GATEKEEPER CHECK (Block Forbidden/Invalid Items) ---
                try:
                    crop_pil = img_pil_cropped.crop((x1, y1, x2, y2))
                    gk_res = gatekeeper_model(crop_pil, verbose=False)
                    
                    gk_top_label = gk_res[0].names[gk_res[0].probs.top1].lower()
                    gk_top_conf = float(gk_res[0].probs.top1conf)
                    
                    VALID_FRUITS = ['banana', 'mango', 'tomato']
                    FORBIDDEN = ['person', 'face', 'human', 'orange', 'apple', 'lemon', 'strawberry', 'man', 'woman', 'dog', 'cat', 'hamster']
                    
                    # RULE 1: If Gatekeeper is CERTAIN it's forbidden (>0.6), SKIP
                    if any(f in gk_top_label for f in FORBIDDEN) and gk_top_conf > 0.6:
                        continue
                    
                    # RULE 2: If Gatekeeper's top prediction is NOT a valid fruit AND confidence is high (>0.5), SKIP
                    if not any(v in gk_top_label for v in VALID_FRUITS) and gk_top_conf > 0.5:
                        continue
                        
                except: 
                    pass

                # --- B. APPLE/RED DETECTION (Color-based filtering) ---
                try:
                    if x2 > x1 and y2 > y1:
                        crop_cv = img_cropped[y1:y2, x1:x2]
                        if crop_cv.size > 0:
                            hsv = cv2.cvtColor(crop_cv, cv2.COLOR_BGR2HSV)
                            avg_hue = np.median(hsv[:, :, 0])
                            avg_sat = np.median(hsv[:, :, 1])
                            
                            # Apple detector: Red color (hue 0-10 or 160-180) + VERY High Saturation (>160)
                            is_apple = ((avg_hue < 10 or avg_hue > 160) and avg_sat > 160)
                            
                            if is_apple:
                                continue  # Skip apples
                            
                            # RED MANGO CHECK: Only skip if BOTH hue is red AND saturation very high
                            if "mango" in fruit_type and conf < 0.80:
                                if (avg_hue < 5 or avg_hue > 160) and avg_sat > 140:
                                    continue  # Skip highly red mangos - likely apples
                            
                except Exception as e:
                    pass

                detected.append({'class': detected_class.title(), 'confidence': conf, 'bbox': [x1_orig, y1_orig, x2_orig, y2_orig]})
            
            del gatekeeper_model
            cleanup_memory()
            
        del model
        cleanup_memory()
        return jsonify(detected)
        
    except Exception as e:
        print(f"Webcam Error: {e}")
        return jsonify([])

@app.route('/delete_item/<filename>', methods=['POST'])
def delete_item(filename):
    with history_lock:
        with open(HISTORY_FILE, 'r') as f: data = json.load(f)
        data = [d for d in data if d['file'] != filename]
        with open(HISTORY_FILE, 'w') as f: json.dump(data, f, indent=4)
    return redirect(url_for('show_history'))

@app.errorhandler(404)
def p404(e): return render_template('404.html'), 404

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host="0.0.0.0", port=5001, debug=True)