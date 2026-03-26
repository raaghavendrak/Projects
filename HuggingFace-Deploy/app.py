import gradio as gr
from ultralytics import YOLO
from PIL import Image
import os

# Load the model - using a try-except to catch loading errors in logs
try:
    model = YOLO("best.pt")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

def predict(img):
    # Run inference on the PIL image
    results = model.predict(source=img, conf=0.25, imgsz=640)
    
    # Plot results
    for r in results:
        res_plotted = r.plot()
    
    # Convert BGR (OpenCV) to RGB (PIL)
    return Image.fromarray(res_plotted[:, :, ::-1])

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="YOLOv8 26-Class Detector"
)

# CRITICAL: server_name and server_port must be exactly this for HF Spaces
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", port=7860)