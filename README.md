---
title: Indian Crop Disease Detector
emoji: 🌿
colorFrom: green
colorTo: green
sdk: gradio
sdk_version: 4.32.0
python_version: 3.12
app_file: app.py
pinned: false
---

# 🌿 Indian Crop Disease Detector

EfficientNet-B0 trained on PlantVillage dataset — 38 classes, 98.5% val accuracy.

## 🔗 Live Demo
👉 [HuggingFace Spaces](https://huggingface.co/spaces/bharath2kk5/crop-disease-detector)

## 📊 Training Dashboard
👉 [W&B Run](https://wandb.ai/bharathmuthyala62-dayananda-sagar-college-of-engineering/crop-disease-detector)

## Results
| Phase | What was trained | Val Accuracy |
|-------|-----------------|--------------|
| Phase 1 | Classifier head only (backbone frozen) | 96.1% |
| Phase 2 | Last 2 blocks unfrozen + fine-tuned | **98.5%** |

## Dataset
- PlantVillage — 54,000+ images
- 38 disease classes across 13 crops
- 70,295 train / 17,572 val images

## Stack
PyTorch · EfficientNet-B0 · W&B · Gradio · HuggingFace Spaces
