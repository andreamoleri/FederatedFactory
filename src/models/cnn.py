# src/models/cnn.py
"""
🤖 Convolutional Neural Network Module
-------------------------------------

Defines a compact convolutional neural network (CNN) for image classification
with 32×32 spatial inputs. The architecture couples two convolution–pooling
stages with a fully connected classifier head, suitable for didactic use,
benchmarks on small image datasets, and lightweight prototypes.

🧠 Purpose:
    Provide a minimal, readable CNN baseline that emphasizes structural clarity
    over architectural novelty.

🔧 Core Functionalities:
    • Feature extractor: stacked Conv–ReLU–MaxPool blocks
    • Classifier head: flattening and linear layers with dropout regularization
    • Forward pass: deterministic mapping from input tensor to class logits

🎯 Intended Use:
    • Academic demonstrations and teaching labs
    • Quick baselines on 32×32 datasets (e.g., CIFAR-like)
    • Reference implementation for unit tests or ablation studies

📁 Dependencies:
    • torch
    • torch.nn

📝 Notes:
    • Expects inputs shaped as (N, C, 32, 32), where N is batch size and C is
      the number of channels.
    • The pooling strategy reduces spatial dimensions from 32→16→8, which sets
      the input size of the first linear layer to 64×8×8.
    • Outputs are raw, unnormalized logits; apply an appropriate loss (e.g.,
      CrossEntropyLoss) that combines LogSoftmax and NLL internally.

Author: Andrea Moleri
File Location: src/models/cnn.py
Last Modified: 12/08/2025
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """A compact CNN classifier for 32×32 inputs.

    This network consists of two convolutional blocks followed by a small
    fully connected classifier head. Convolutional layers use 5×5 kernels with
    padding=2 to preserve spatial resolution prior to pooling, and MaxPool2d
    (kernel_size=2) halves spatial dimensions at each stage.

    Attributes:
        num_classes (int): Number of target classes; determines the size of the
            final linear layer's output.
        conv (nn.Sequential): Feature extractor composed of Conv–ReLU–Pool
            stages.
        fc (nn.Sequential): Classifier head that flattens features and maps to
            class logits.
    """

    def __init__(self, in_ch: int, num_classes: int):
        """Initialize the SimpleCNN module.

        Parameters:
            in_ch (int): Number of input channels (e.g., 3 for RGB, 1 for gray).
            num_classes (int): Number of classification categories; defines the
                dimensionality of the output logits.

        Raises:
            ValueError: If `in_ch` or `num_classes` is not a positive integer.
                (Note: this constructor does not actively validate inputs; this
                is an implicit contract for correct usage.)
        """
        super().__init__()
        self.num_classes = num_classes

        # Feature extractor:
        # - First conv block preserves 32×32 via padding=2 (5×5 kernel),
        #   then MaxPool2d reduces to 16×16.
        # - Second conv block again preserves resolution before pooling to 8×8.
        # - Channel progression: in_ch → 32 → 64.
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((8, 8)),
        )

        # Classifier head:
        # - Flatten converts (N, 64, 8, 8) to (N, 64*8*8).
        # - A hidden layer (256 units) with ReLU and dropout provides limited
        #   regularization; final linear emits unnormalized logits of size
        #   `num_classes`.
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(256, num_classes),
        )


    def forward(self, x):
        """Compute logits from input images.

        Parameters:
            x (torch.Tensor): Input batch with shape (N, in_ch, 32, 32) and
                dtype compatible with the underlying layers (typically
                torch.float32). The channel count must equal `in_ch` specified
                at construction.

        Returns:
            torch.Tensor: Raw class logits of shape (N, num_classes). These are
                not probabilities; apply a softmax at inference time if
                probabilities are required, or use a loss that expects logits
                during training (e.g., CrossEntropyLoss).

        Raises:
            RuntimeError: Propagated from PyTorch if input shapes are
                incompatible with the defined layers (e.g., incorrect spatial
                size or channel count).
        """
        # The composition conv → fc is intentionally simple to emphasize
        # dimensionality flow and maintain readability.
        return self.fc(self.conv(x))
