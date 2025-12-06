"""
🤖 Convolutional Neural Network Models
------------------------------------

This module defines flexible wrappers around standard convolutional neural 
network architectures, specifically adapting ResNet backbones for varying 
input resolutions and channel configurations.

🧠 Purpose:
    The primary class, SimpleCNN, serves as an adaptive interface for the 
    ResNet50 architecture. It dynamically modifies the initial convolutional 
    layers based on input resolution, making standard ImageNet models 
    suitable for low-resolution academic datasets (e.g., CIFAR-10, CIFAR-100) 
    without excessive spatial downsampling.

🔧 Core Functionalities:
    • Instantiation of ResNet50 architectures with random initialization
    • Dynamic architectural adaptation for low-resolution inputs (≤ 64px)
    • Automatic adjustment of input channels (e.g., for grayscale or multispectral data)
    • Reconfiguration of the final classification layer

🎯 Intended Use:
    • Academic research involving computer vision benchmarks
    • Deep learning curriculum and pedagogical demonstrations
    • Prototyping custom classifiers on non-standard image dimensions

📁 Dependencies:
    • torch
    • torchvision

📝 Notes:
    This implementation modifies the internal structure of the torchvision 
    ResNet model. Specifically, it replaces the initial 7x7 convolution 
    and MaxPool layers with a 3x3 convolution when handling small images 
    to preserve feature map dimensions.

Author: Andrea Moleri
File Location: src/models/cnn.py
Last Modified: 23/04/2025
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet50


class SimpleCNN(nn.Module):
    """
    A wrapper class for the ResNet50 architecture that adapts the model structure
    based on input resolution and channel depth.

    This class overrides specific layers of the standard ResNet model to prevent
    information loss when processing low-resolution images (e.g., 32x32), a common
    requirement in academic benchmarking.
    """

    def __init__(self, in_ch: int, num_classes: int, input_resolution: int = 32):
        """
        Initializes the SimpleCNN model.

        Parameters
        ----------
        in_ch : int
            The number of input channels (e.g., 3 for RGB, 1 for grayscale).
        num_classes : int
            The number of target classes for the final classification layer.
        input_resolution : int, optional
            The spatial resolution (height/width) of the input images.
            Defaults to 32.

        Returns
        -------
        None
        """
        super().__init__()
        self.num_classes = num_classes

        # Initialize the standard ResNet50 model without pre-trained weights.
        # This is suitable for training from scratch on custom datasets.
        self.model = resnet50(weights=None)

        # Check if the input resolution is small (e.g., CIFAR-10/100, Tiny ImageNet).
        # Standard ResNet uses a 7x7 conv (stride 2) followed by a MaxPool (stride 2),
        # which reduces spatial dimensions by a factor of 4 immediately.
        # For 32x32 images, this aggressive downsampling destroys spatial features too early.
        if input_resolution <= 64:

            # Replace the initial large kernel with a smaller 3x3 kernel and stride 1.
            # This preserves the spatial dimensions of the input tensor.
            self.model.conv1 = nn.Conv2d(
                in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            # Remove the max pooling layer entirely to avoid further downsampling
            # at the initial stage.
            self.model.maxpool = nn.Identity()
        else:

            # For standard resolutions, ensure the first layer matches the input channels.
            # ResNet defaults to 3 channels (RGB); this modification supports
            # grayscale or multi-spectral inputs while keeping the original
            # downsampling architecture (7x7 kernel, stride 2).
            if in_ch != 3:
                self.model.conv1 = nn.Conv2d(
                    in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False
                )

        # Replace the final fully connected (linear) layer.
        # ResNet50's default output dimension is 1000 (ImageNet classes).
        # We adjust this to match the specific `num_classes` of the target dataset.
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # Create a direct reference to the classification head.
        # This facilitates access for feature extraction or external monitoring hooks.
        self.fc = self.model.fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the neural network.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor containing a batch of images.
            Shape: (Batch_Size, Channels, Height, Width)

        Returns
        -------
        torch.Tensor
            The raw logits output by the network.
            Shape: (Batch_Size, num_classes)
        """
        # Delegate the forward pass to the modified ResNet backbone.
        return self.model(x)
