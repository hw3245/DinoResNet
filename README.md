# DinoResNet Image Classification Models

This repository contains two Jupyter Notebook files implementing the DinoResNet models for image classification. These models integrate the DINO (Self-Supervised Vision Transformer) architecture with ResNet, a powerful and widely-used convolutional neural network.

## Repository Contents

- `DinoResNet_Horizontal.ipynb`: Implements the horizontal model, which processes images in parallel using both DINOv2 (ViT-S variant) and ResNet-50. It effectively combines global and local features for enhanced image classification accuracy.
- `DinoResNet_Vertical.ipynb`: Implements the vertical model, which sequentially processes the image first through DINOv2 (ViT-S) to extract the class token and then feeds it into the ResNet-50 model for final classification.

## Model Overview

### Horizontal Model

Parallel processing with DINOv2 and ResNet-50.

### Vertical Model

Sequential processing with DINOv2 followed by ResNet-50.

## Dataset

Trained and evaluated on a subset of the ImageNet-1K dataset.
