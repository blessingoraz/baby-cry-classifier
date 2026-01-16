import os
import glob
import torch
import torch.nn as nn
import torchvision.models as models

class CryResNet(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone="resnet18",
        pretrained=True,
        freeze_backbone=True,
        droprate=0.8,
        size_inner=512):    # <- inner layer size
        super().__init__()

        # Load pre-trained Resnet18
        if backbone == "resnet18":
            self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        elif backbone == "resnet34":
            self.base_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        else:
            raise ValueError("backbone must be resnet18 or resnet34")

        # Freeze base model parameters
        if freeze_backbone:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Replace classifier head
        in_features = self.base_model.fc.in_features

        self.base_model.fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(droprate),
            nn.Linear(in_features, size_inner),
            nn.Linear(size_inner, num_classes)
        )

        # Ensure head is trainable (safe even if freeze_backbone=True)
        for p in self.base_model.fc.parameters():
            p.requires_grad = True

    def forward(self, x):
        # x: (B, 1, n_mels, time) -> convert to 3-channel
        x = x.repeat(1, 3, 1, 1)
        return self.base_model(x)



def find_latest_checkpoint(pattern="models/best_*.pt"):
    """Find latest checkpoint by file creation time."""
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoints found with pattern: {pattern}")
    latest = max(files, key=os.path.getctime)
    return latest

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(
    ckpt_path: str,
    num_classes: int = 8,
    backbone: str = "resnet18",
    pretrained: bool = False,
    freeze_backbone: bool = True,
    droprate: float = 0.8,
    size_inner: int = 512,
    device=None,
):
    """
    Import-safe: does NOT run unless you call it.
    Returns (model, device, ckpt_metadata_dict)
    """
    if device is None:
        device = get_device()

    model = CryResNet(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        droprate=droprate,
        size_inner=size_inner,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.eval()
    return model, device

def main():
    best_ckpt = "models/checkpoints/best_lr_0.01_inner_512_drop_0.8.pt"
    ckpt_path = best_ckpt if os.path.exists(best_ckpt) else find_latest_checkpoint("models/best_*.pt")
    print("Loading model from:", ckpt_path)

    model, device, meta = load_model(
        ckpt_path=ckpt_path,
        num_classes=8,
        backbone="resnet18",
        pretrained=False,
        freeze_backbone=True,
        droprate=0.8,
        size_inner=512,
    )
    print("Model loaded and ready on device:", device)
    return model, device, meta

if __name__ == "__main__":
    main()