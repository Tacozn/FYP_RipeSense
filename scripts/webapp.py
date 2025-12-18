import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# ==========================================
# 1. CONFIG & SETUP
# ==========================================
st.set_page_config(page_title="RipeSense Tester", page_icon="🍌")

# --- CUSTOM CSS FOR "FYP" LOOK ---
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .stMetric {
            background-color: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. FUNCTIONS
# ==========================================

@st.cache_resource
def load_model():
    """Loads the model once and keeps it in memory (speeds up the app)"""
    # ⚠️ MAKE SURE THIS PATH IS CORRECT
    model_path = 'weights_new/best.pt' 
    return YOLO(model_path)

def get_average_hue(cropped_img):
    """Calculates the average hue of the fruit crop"""
    if cropped_img.size == 0:
        return 0
    # Convert to HSV
    hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    # Extract Hue channel (0)
    hue_channel = hsv_img[:, :, 0]
    return int(np.mean(hue_channel))

def process_image(uploaded_file, model, conf_threshold):
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Run YOLO
    results = model(image, conf=conf_threshold)
    
    # Prepare data storage
    detections = []
    
    # Draw boxes
    for result in results:
        for box in result.boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            
            # Get Hue
            crop = image[y1:y2, x1:x2]
            avg_hue = get_average_hue(crop)
            
            # Store data
            detections.append({
                "Label": label,
                "Confidence": f"{conf:.2%}",
                "Hue Value": avg_hue,
                "Status": "Ripe" if avg_hue < 40 else "Unripe/Green" # Simple logic example
            })

            # Draw on image (Green Box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Draw Label Background
            cv2.rectangle(image, (x1, y1-35), (x1+250, y1), (0, 255, 0), -1)
            cv2.putText(image, f"{label} {conf:.2f} | Hue: {avg_hue}", 
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Convert BGR (OpenCV) to RGB (Streamlit)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, detections

# ==========================================
# 3. APP LAYOUT
# ==========================================
st.title("🍌 RipeSense: Model & Logic Tester")
st.write("Upload an image to test the Hybrid Logic (AI + Color Algorithm).")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# Load Model
try:
    model = load_model()
    st.sidebar.success("✅ Model Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()

# File Uploader
uploaded_file = st.file_uploader("Choose a fruit image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Process
    with st.spinner('Analyzing...'):
        processed_image, data = process_image(uploaded_file, model, conf_threshold)
    
    # Show Results
    st.image(processed_image, caption='Processed Image', use_container_width=True)
    
    if data:
        st.subheader("📊 Detection Stats")
        
        # Display as a nice table
        st.table(data)
        
        # Display Metrics for the first fruit found
        col1, col2, col3 = st.columns(3)
        col1.metric("First Detection", data[0]['Label'])
        col2.metric("Confidence", data[0]['Confidence'])
        col3.metric("Hue Value", f"{data[0]['Hue Value']}°")
        
    else:
        st.warning("⚠️ No fruit detected! Try lowering the Confidence Threshold in the sidebar.")