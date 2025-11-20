"""
🤖 Deep Learning Model Architectures
------------------------------------

This package initialisation module aggregates and exposes the core neural
network architectures and utility functions provided by the library.

🧠 Purpose:
    To serve as the central entry point for accessing Variational Autoencoders
    (VAE), Convolutional Neural Networks (CNN), and Diffusion Transformers
    (DiT), facilitating streamlined imports for research and production
    pipelines.

🔧 Core Functionalities:
    • Expose VAE components (Encoder, Decoder, ResidualBlock) for latent
      variable modelling.
    • Provide access to SimpleCNN for baseline image feature extraction.
    • Export Diffusion Transformer (DiT) architectures and associated flow
      matching samplers for generative tasks.

🎯 Intended Use:
    • high-level model instantiation in training scripts.
    • Modular component access for custom architecture design.
    • Academic experimentation with generative flow models.

📁 Dependencies:
    • Internal submodules: .vae, .cnn, .diffusion

📝 Notes:
    This module implements the Facade pattern, simplifying the API surface
    by hiding the internal directory structure from the end user.

Author: Andrea Moleri
File Location: TBD
Last Modified: 20/11/2025
"""

# Import Variational Autoencoder components to expose them at the package level.
# These classes constitute the building blocks for latent variable generative models.
from .vae import VAE, Encoder, Decoder, ResidualBlock

# Import baseline Convolutional Neural Network architecture.
# Useful for comparative analysis or simple feature extraction tasks.
from .cnn import SimpleCNN

# Import Diffusion Transformer (DiT) components and flow matching utilities.
# This includes the model configuration, the loss function for training,
# and the sampler for inference.
from .diffusion import DiT, DiffusionConfig, rectified_flow_loss, rectified_flow_sampler

# Define the public API of the package.
# This list restricts the symbols exported when a client performs
# `from package import *`, ensuring internal helpers remain encapsulated.
__all__ = [
    "VAE",
    "Encoder",
    "Decoder",
    "ResidualBlock",
    "SimpleCNN",
    "DiT",
    "DiffusionConfig",
    "rectified_flow_loss",
    "rectified_flow_sampler",
]