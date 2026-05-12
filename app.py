"""
Gradio app for Indian Crop Disease Detector.
Upload a photo of a crop leaf → model identifies the disease → shows top-5 predictions
with confidence scores.
Entry point for HuggingFace Spaces deployment.
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

# Exact folder names from dataset — order must match training
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
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
        print(f"⚠️  Model file not found at {MODEL_PATH} — using random weights")
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
        print("📸 Received image, starting preprocessing...")
        img_tensor = preprocess(image).unsqueeze(0)

        print("🤖 Running model inference...")
        with torch.no_grad():
            outputs = model(img_tensor)
            
            # Temperature Scaling
            temperature = 2.0 
            probabilities = F.softmax(outputs / temperature, dim=1)[0]

        print("📊 Extracting Top-5 results...")
        top5_probs, top5_indices = torch.topk(probabilities, k=5)

        results = {}
        for prob, idx in zip(top5_probs, top5_indices):
            class_name = CLASS_NAMES[idx.item()]
            results[class_name] = float(prob.item())
        
        print(f"✅ Prediction complete: {list(results.keys())[0]}")
        return results

    except Exception as e:
        print(f"❌ ERROR during prediction: {str(e)}")
        return {"Error": str(e)}


# ──────────────────────────────────────────────────
# GRADIO UI
# ──────────────────────────────────────────────────
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="📸 Upload a crop leaf image"),
    outputs=gr.Label(num_top_classes=5, label="🔍 Disease Prediction"),
    title="🌿 Indian Crop Disease Detector",
    description=(
        "Upload a photo of a crop leaf to identify potential diseases. "
        "This model uses **EfficientNet-B0** with transfer learning, trained on the "
        "**PlantVillage dataset** (54,000+ images, 38 classes, **98.5% val accuracy**).\n\n"
        "Supports: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, "
        "Potato, Raspberry, Soybean, Squash, Strawberry, Tomato"
    ),
    article=(
        "### About This Project\n"
        "- **Model:** EfficientNet-B0 with two-phase transfer learning\n"
        "- **Phase 1:** Classifier head trained (backbone frozen) — Val Acc: 96.1%\n"
        "- **Phase 2:** Last 2 blocks unfrozen + fine-tuned — Val Acc: **98.5%**\n"
        "- **Dataset:** [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) — 54K+ images, 38 classes\n"
        "- **Training:** Tracked with [W&B](https://wandb.ai/bharathmuthyala62-dayananda-sagar-college-of-engineering/crop-disease-detector)\n"
        "- **Built by:** Bharath M\n\n"
        "⚠️ *This is a learning project and NOT a substitute for professional agricultural advice.*"
    ),
    theme=gr.themes.Soft(),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(ssr=False)