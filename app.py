"""
Gradio app for Indian Crop Disease Detector.
Upload a photo of a crop leaf → model identifies the disease → shows top-5 predictions.
Using gr.Blocks for maximum stability.
"""

import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import gradio as gr
from model import create_model


# ──────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────
MODEL_PATH = "checkpoints/best_model.pth"
NUM_CLASSES = 38
IMG_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]


# ──────────────────────────────────────────────────
# MODEL LOAD
# ──────────────────────────────────────────────────
def load_model():
    model = create_model(num_classes=NUM_CLASSES, pretrained=False)
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        print(f"✅ Loaded model from {MODEL_PATH}")
    else:
        print(f"⚠️  Model file not found at {MODEL_PATH}")
    model.eval()
    return model

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

model = load_model()


# ──────────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────────
def predict(image: Image.Image) -> dict:
    if image is None:
        return {}
    try:
        print("📸 Preprocessing...")
        img_tensor = preprocess(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs / 2.0, dim=1)[0]

        top5_probs, top5_indices = torch.topk(probabilities, k=5)
        results = {CLASS_NAMES[idx.item()]: float(prob.item()) for prob, idx in zip(top5_probs, top5_indices)}
        
        print(f"✅ Predicted: {list(results.keys())[0]}")
        return results
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {"Error": str(e)}


# ──────────────────────────────────────────────────
# UI DESIGN (Using gr.Blocks)
# ──────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌿 Indian Crop Disease Detector")
    gr.Markdown("Upload a photo of a crop leaf to identify potential diseases using Deep Learning.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="📸 Upload Leaf Image")
            submit_btn = gr.Button("🔍 Detect Disease", variant="primary")
        
        with gr.Column():
            output_label = gr.Label(num_top_classes=5, label="🔍 Prediction Results")
    
    gr.Markdown("### 📊 About the Model")
    gr.Markdown(
        "Built with **EfficientNet-B0** and trained on the **PlantVillage dataset**.\n"
        "- **Accuracy:** 98.5% on validation set\n"
        "- **Classes:** 38 types of diseases across 13 different crops."
    )
    
    # Event listeners
    submit_btn.click(fn=predict, inputs=input_img, outputs=output_label)
    input_img.change(fn=predict, inputs=input_img, outputs=output_label)

if __name__ == "__main__":
    demo.launch()