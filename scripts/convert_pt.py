from ultralytics import YOLO

# Load your custom model
model = YOLO('models/best.pt')

# Export to ONNX format (much lighter for CPU)
model.export(format='onnx', imgsz=320, half=True)

# Do the same for your gatekeeper
gatekeeper = YOLO('models/yolov8n-cls.pt')
gatekeeper.export(format='onnx', imgsz=320, half=True)