# Grad-CAM on ResNet18 (PyTorch)

This repository demonstrates how to apply **Grad-CAM**
(Gradient-weighted Class Activation Mapping) on a pretrained
**ResNet18** model from `torchvision` to visualize which regions of an
image influence the model's prediction.

------------------------------------------------------------------------

# Demo

| Input Image | Grad-CAM Output |
|-------------|-----------------|
| <img src="pizza.jpg" width="350"> | <img src="gradcam_overlay.png" width="350"> |


------------------------------------------------------------------------

## How It Works

1.  A pretrained **ResNet18** model is loaded.
2.  The target convolutional layer (`layer4[1].conv2`) is selected for
    Grad-CAM.
3.  **Forward hooks** capture feature activations.
4.  **Backward hooks** capture gradients of the predicted class.
5.  The Grad-CAM map is computed using a weighted sum of activations and
    gradients, followed by ReLU and normalization.
6.  The heatmap is resized and overlaid on the original image using
    OpenCV.

------------------------------------------------------------------------

## Requirements

-   Python 3.x\
-   PyTorch\
-   Torchvision\
-   NumPy\
-   OpenCV\
-   Pillow\
-   Matplotlib

Install dependencies:

``` bash
pip install torch torchvision opencv-python pillow matplotlib numpy
```

------------------------------------------------------------------------

## Usage

### 1. Clone the repository

``` bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Add your image

Place your image in the repo as `pizza.jpg` or update the script:

``` python
img_path = 'pizza.jpg'
```

### 3. Run the script

``` bash
python grad_cam_pizza.py
```

This will:

-   Print the model's predicted class index\
-   Display (or save) the Grad-CAM heatmap overlay

------------------------------------------------------------------------

## Key Code Snippets

### Registering Hooks

``` python
activations = []
gradients = []

def forward_hook(module, input, output):
    activations.append(output)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

target_layer = model.layer4[1].conv2
forward_handle = target_layer.register_forward_hook(forward_hook)
backward_handle = target_layer.register_full_backward_hook(backward_hook)
```

### Computing Grad-CAM

``` python
weights = grad.mean(dim=(1, 2))  # (C,)
cam = (weights.view(-1, 1, 1) * act).sum(dim=0)
cam = functional.relu(cam)
cam = (cam - cam.min()) / (cam.max() + 1e-8)
```

------------------------------------------------------------------------

## Notes

-   Uses **ResNet18 pretrained on ImageNet**.
-   You can modify layers or swap in different models.
-   Normalization (`transforms.Normalize`) is optional but recommended
    for true ImageNet preprocessing.

------------------------------------------------------------------------

## License

Feel free to use, modify, and build upon this project.
