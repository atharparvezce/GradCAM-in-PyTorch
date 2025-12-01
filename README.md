# Grad-CAM on ResNet18 (PyTorch)

This repository demonstrates how to apply **Grad-CAM** (Gradient-weighted Class Activation Mapping) on a pretrained **ResNet18** model from `torchvision` to visualize which regions of an image influence the model’s prediction.

---

## Demo

### Input Image

<img src="pizza.jpg" alt="Original pizza slice" width="350"/>

### Grad-CAM Overlay

<img src="gradcam_overlay.png" alt="Grad-CAM overlay on pizza slice" width="350"/>

---

## How It Works

1. A pretrained **ResNet18** model is loaded.
2. A target convolutional layer (`layer4[1].conv2`) is chosen for Grad-CAM.
3. **Forward hook** captures activations from the target layer.
4. **Backward hook** captures gradients of the predicted class w.r.t. those activations.
5. Grad-CAM map is computed as a weighted sum of activations, passed through ReLU, then normalized.
6. The heatmap is resized and overlaid on the original image using OpenCV.

---

## Requirements

- Python 3.x  
- PyTorch  
- Torchvision  
- NumPy  
- OpenCV  
- Pillow  
- Matplotlib  

Install dependencies (example):

```bash
pip install torch torchvision opencv-python pillow matplotlib numpy
