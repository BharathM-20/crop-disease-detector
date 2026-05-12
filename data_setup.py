"""
Data loading utilities for PlantVillage dataset.

Expected directory structure after setup:
    data/
    ├── train/
    │   ├── Apple___Apple_scab/
    │   ├── Apple___Black_rot/
    │   ├── ...
    │   └── Tomato___healthy/
    └── val/
        ├── Apple___Apple_scab/
        ├── ...
        └── Tomato___healthy/
"""

import os
import shutil
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# ImageNet normalization — MUST use these because EfficientNet was pretrained with them
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Image size for EfficientNet-B0
IMG_SIZE = 224


def get_transforms():
    """
    Returns train and val transforms.

    Train: augmentation + normalize (we want the model to generalize)
    Val: just resize + normalize (we want honest evaluation)
    """
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, val_transform


def split_dataset(source_dir: str, dest_dir: str, val_split: float = 0.2, seed: int = 42):
    """
    Split a folder of class-subfolders into train/val sets.

    Args:
        source_dir: Path to the raw PlantVillage folder (has 38 class folders)
        dest_dir: Path to create train/ and val/ folders
        val_split: Fraction of data to use for validation
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    source = Path(source_dir)
    dest = Path(dest_dir)

    train_dir = dest / "train"
    val_dir = dest / "val"

    if train_dir.exists():
        print(f"⚠️  {train_dir} already exists — skipping split. Delete it to re-split.")
        return

    class_folders = [f for f in source.iterdir() if f.is_dir()]
    print(f"Found {len(class_folders)} classes")

    total_train, total_val = 0, 0

    for class_folder in sorted(class_folders):
        class_name = class_folder.name
        images = list(class_folder.glob("*.[jJpP][pPnN][gG]"))  # jpg, JPG, png, PNG
        random.shuffle(images)

        split_idx = int(len(images) * (1 - val_split))
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # Create directories and copy files
        (train_dir / class_name).mkdir(parents=True, exist_ok=True)
        (val_dir / class_name).mkdir(parents=True, exist_ok=True)

        for img in train_images:
            shutil.copy2(img, train_dir / class_name / img.name)
        for img in val_images:
            shutil.copy2(img, val_dir / class_name / img.name)

        total_train += len(train_images)
        total_val += len(val_images)

    print(f"✅ Split complete: {total_train} train / {total_val} val images")


def create_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 2):
    """
    Create train and val DataLoaders from the split dataset.

    Args:
        data_dir: Path to directory containing train/ and val/ folders
        batch_size: Batch size for DataLoaders
        num_workers: Number of workers for parallel data loading

    Returns:
        train_loader, val_loader, class_names
    """
    train_transform, val_transform = get_transforms()

    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "val"),
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    class_names = train_dataset.classes
    print(f"✅ DataLoaders ready: {len(train_dataset)} train / {len(val_dataset)} val images")
    print(f"   Classes: {len(class_names)}")
    print(f"   Batches: {len(train_loader)} train / {len(val_loader)} val")

    return train_loader, val_loader, class_names


def visualize_samples(data_loader: DataLoader, class_names: list, num_images: int = 16):
    """
    Visualize a grid of sample images from a DataLoader.
    Useful for verifying your data pipeline is working correctly.
    """
    images, labels = next(iter(data_loader))

    # Denormalize for display
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle("Sample Training Images", fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i >= min(num_images, len(images)):
            ax.axis("off")
            continue

        img = images[i] * std + mean  # Denormalize
        img = img.clamp(0, 1)
        img = img.permute(1, 2, 0).numpy()  # CHW → HWC

        # Clean up label: "Tomato___Late_blight" → "Tomato - Late blight"
        label = class_names[labels[i]].replace("___", " - ").replace("_", " ")

        ax.imshow(img)
        ax.set_title(label, fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("sample_images.png", dpi=150)
    plt.show()
    print("✅ Saved sample grid to sample_images.png")


if __name__ == "__main__":
    # ──────────────────────────────────────────────
    # STEP 1: First, download PlantVillage from Kaggle:
    #   https://www.kaggle.com/datasets/emmarex/plantdisease
    #   Extract to: data/raw/PlantVillage/
    #
    # STEP 2: Run this script to split into train/val:
    #   python data_setup.py
    # ──────────────────────────────────────────────

    RAW_DIR = "data/raw/PlantVillage/PlantVillage"  # ← adjust this to where you extracted the dataset
    SPLIT_DIR = "data"

    if os.path.exists(RAW_DIR):
        print("Splitting dataset...")
        split_dataset(source_dir=RAW_DIR, dest_dir=SPLIT_DIR, val_split=0.2)
    else:
        print(f"⚠️  Raw data not found at '{RAW_DIR}'")
        print(f"   Download from: https://www.kaggle.com/datasets/emmarex/plantdisease")
        print(f"   Extract to: {RAW_DIR}")

    # Test DataLoaders (only if split exists)
    if os.path.exists(os.path.join(SPLIT_DIR, "train")):
        train_loader, val_loader, class_names = create_dataloaders(SPLIT_DIR, batch_size=32)
        visualize_samples(train_loader, class_names)
