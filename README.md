# 🌿 Indian Crop Disease Detector

> A deep learning system that identifies **38 plant diseases** from leaf images — built to help Indian farmers detect crop infections early, before they lose their harvest.

[![HuggingFace Spaces](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Spaces-yellow)](https://huggingface.co/spaces/bharath2kk5/crop-disease-detector)
[![W&B](https://img.shields.io/badge/📈%20Training%20Logs-W%26B-orange)](https://wandb.ai/bharathmuthyala62-dayananda-sagar-college-of-engineering/crop-disease-detector/runs/tj9g977m)
[![GitHub](https://img.shields.io/badge/💻%20Code-GitHub-black)](https://github.com/BharathM-20/crop-disease-detector)

---

## 🎯 What This Does

Upload a photo of a crop leaf → the model tells you what disease it has (if any), with a calibrated confidence score.

Trained on the **PlantVillage dataset** — 54,000+ images across 13 crops and 38 disease classes.

---

## 📊 Results

| Phase | What Was Trained | Val Accuracy |
|-------|-----------------|--------------|
| Phase 1 | Classifier head only (backbone frozen) | 96.1% |
| Phase 2 | Last 2 EfficientNet blocks unfrozen + fine-tuned | **98.5%** |

---

## 🏗️ Architecture

**Model:** EfficientNet-B0 (pretrained on ImageNet)

**Two-phase transfer learning:**
- **Phase 1 —** Freeze the entire backbone. Train only the classification head. Lets the model learn crop-disease-specific features without destroying pretrained weights.
- **Phase 2 —** Unfreeze the last 2 blocks. Fine-tune at a lower learning rate. Pushes accuracy from 96.1% → 98.5%.

**Why EfficientNet-B0?**
It hits the best accuracy-to-parameter ratio in the EfficientNet family — strong enough for 38-class classification, light enough for free-tier deployment on HuggingFace Spaces.

---

## 🔴 The Hardest Problem — Overconfidence

After Phase 2, the model predicted with **99.8% confidence** on genuinely ambiguous images.

In a real farming scenario, that's dangerous. A farmer acting on a wrong high-confidence prediction could make the wrong treatment decision and lose their crop.

**Fix: Temperature Scaling (T=2.0)**

Applied post-training to calibrate the softmax outputs. Instead of the model always being loud, it learned to be honest — flagging uncertain predictions with lower confidence scores.

```python
# Temperature scaling applied at inference
logits = model(image) / temperature  # T = 2.0
probabilities = F.softmax(logits, dim=1)
```

This is a production ML concept most tutorials skip. It made a real difference in output reliability.

---

## 🛠️ Other Challenges

**Deployment — Torch version conflicts on HuggingFace Spaces**
Pinned Gradio to `4.32.0` to resolve runtime errors at deployment. Took longer to debug than any modeling issue.

**Class imbalance**
38 classes with unequal image counts across diseases. Handled through careful augmentation and monitoring per-class accuracy in W&B.

**Inference speed**
Free-tier HuggingFace Spaces has CPU-only hardware. Kept inference fast by using EfficientNet-B0 (the lightest variant) and optimizing the Gradio interface to avoid unnecessary reloads.

---

## 📦 Dataset

| Detail | Value |
|--------|-------|
| Source | PlantVillage |
| Total images | 87,867 |
| Train split | 70,295 images |
| Val split | 17,572 images |
| Classes | 38 disease categories |
| Crops covered | 13 |

---

## 🧪 Experiment Tracking

All training runs tracked with **Weights & Biases** — loss curves, accuracy per epoch, learning rate schedules, and hyperparameters logged across both phases.

👉 [View the training run on W&B](https://wandb.ai/bharathmuthyala62-dayananda-sagar-college-of-engineering/crop-disease-detector/runs/tj9g977m)

---

## 🚀 Run It Locally

```bash
git clone https://github.com/BharathM-20/crop-disease-detector
cd crop-disease-detector
pip install -r requirements.txt
python app.py
```

---

## 🧰 Stack

| Tool | Purpose |
|------|---------|
| PyTorch | Model training |
| EfficientNet-B0 | Backbone architecture |
| Weights & Biases | Experiment tracking |
| Gradio | Demo interface |
| HuggingFace Spaces | Deployment |

---

## 🌾 Why This Matters

India has **150M+ smallholder farmers** — most of whom have no access to agronomists or plant pathologists.

Early disease detection = fewer crop losses = better livelihood for farmers.

A phone camera + this model is a step toward making that accessible.

---

## 👨‍💻 Built By

**Bharath M**
B.E. Robotics & AI — Dayananda Sagar College of Engineering, Bengaluru

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/bharath-muthyala-a59b8632b/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/bharath2kk5)
