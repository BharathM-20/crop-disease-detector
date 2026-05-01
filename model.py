"""
Model definition — EfficientNet-B0 with transfer learning for crop disease classification.

Architecture:
  - Pretrained EfficientNet-B0 backbone (trained on ImageNet)
  - Custom classifier head for 38 PlantVillage classes
  - Two-phase training: freeze backbone first, then unfreeze last blocks
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def create_model(num_classes: int = 38, pretrained: bool = True) -> nn.Module:
    """
    Create an EfficientNet-B0 model with a custom classifier head.

    Args:
        num_classes: Number of output classes (PlantVillage has 38)
        pretrained: Whether to load ImageNet pretrained weights

    Returns:
        model: EfficientNet-B0 with custom head
    """
    # Load pretrained EfficientNet-B0
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)

    # Freeze ALL backbone layers — we'll only train the classifier head first
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the classifier head
    # Original: Linear(1280, 1000) for ImageNet
    # Ours: Dropout → Linear(1280, num_classes) for PlantVillage
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1280, out_features=num_classes)
    )

    return model


def unfreeze_last_n_blocks(model: nn.Module, n: int = 2):
    """
    Unfreeze the last N blocks of the EfficientNet backbone for fine-tuning.
    Call this AFTER Phase 1 training to start Phase 2.

    Args:
        model: The EfficientNet model
        n: Number of blocks to unfreeze from the end (default: 2)
    """
    # EfficientNet-B0 has 9 blocks (model.features[0] through model.features[8])
    total_blocks = len(model.features)
    for i in range(total_blocks - n, total_blocks):
        for param in model.features[i].parameters():
            param.requires_grad = True

    print(f"✅ Unfroze last {n} blocks (blocks {total_blocks - n} to {total_blocks - 1})")
    # Print trainable vs total params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")


if __name__ == "__main__":
    # Quick sanity check — run this file to verify model loads correctly
    model = create_model(num_classes=38)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model created successfully!")
    print(f"Total params: {total:,}")
    print(f"Trainable params (Phase 1 — head only): {trainable:,}")

    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}  (expected: [1, 38])")

    # Test unfreeze
    unfreeze_last_n_blocks(model, n=2)
