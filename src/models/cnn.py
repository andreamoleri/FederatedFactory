import torch
import torch.nn as nn
from torchvision.models import resnet50

def replace_bn_with_gn(module, num_groups=32):
    """
    Recursively replace BatchNorm2d with GroupNorm.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            # Get the number of channels
            num_channels = child.num_features
            # Create GroupNorm (ensure num_groups divides num_channels)
            # ResNet50 channels are usually divisible by 32
            gn = nn.GroupNorm(num_groups, num_channels)
            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child, num_groups)

class SimpleCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, input_resolution: int = 32):
        super().__init__()
        self.num_classes = num_classes

        # 1. Load Model
        self.model = resnet50(weights=None)

        # 2. Architecture Adaptation (Keep this, it's why your acc is good!)
        if input_resolution <= 64:
            self.model.conv1 = nn.Conv2d(
                in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.model.maxpool = nn.Identity()
        else:
            if in_ch != 3:
                self.model.conv1 = nn.Conv2d(
                    in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False
                )

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.fc = self.model.fc

        # 3. CRITICAL FIX: Replace BN with GN
        replace_bn_with_gn(self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)