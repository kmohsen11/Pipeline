# Pipeline App

A GUI application for cell segmentation and analysis using Cellpose.

## Overview

This application provides a user-friendly interface for:
- Image augmentation
- Segmentation using Cellpose
- Single cell extraction
- Z-stack recombination
- Support for custom Cellpose model weights

## Installation

### Environment Setup

This application requires Python 3.8+ and several dependencies. It's recommended to use a virtual environment:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r App/requirements.txt
```

## Running the Application

After activating your virtual environment, run:

```bash
python GUI_test.py
```

## Features

### Image Augmentation
Apply transformations to your images for data augmentation purposes.

### Segmentation Augmentation
Apply transformations to segmentation masks.

### Run Pipeline
Process images using Cellpose for cell segmentation:
1. Click "Run Pipeline"
2. Select your input image
3. Choose an output directory
4. Select the channel to use (1-4)
5. Choose whether to use default model or custom weights
6. If using custom weights, select your weights file

### Extract Single Cells
Extract individual cells from a segmented stack.

### Recombine Z-Stack
Recombine extracted single cells into a Z-stack.

## Using Custom Weights

The application supports two types of custom weights:

### Standard Cellpose Weights
These are typically `.npy` files that can be loaded directly by Cellpose.

### PyTorch/Lightning Checkpoints
The application can also load PyTorch/Lightning-style checkpoints:
- Files like `checkpoint_run4_checkpoint_30000` (no file extension)
- PyTorch checkpoint files saved with `torch.save`
- Lightning checkpoints containing model state dictionaries

When running the pipeline, you'll be prompted to choose between the default model and custom weights. If you select custom weights, you'll be asked to locate your weights file.

## Troubleshooting

### Common Issues

1. **Module not found errors**: Make sure you've activated your virtual environment and installed all requirements.

2. **CUDA/GPU errors**: If you encounter GPU-related errors, try running with CPU only by modifying the code.

3. **Memory issues**: For large images, you might need to process them in smaller batches or tiles.

### Getting Help

If you encounter issues not covered here, please check the Cellpose documentation or open an issue in the project repository.

## Credits

This application uses [Cellpose](https://github.com/MouseLand/cellpose) for cell segmentation. 