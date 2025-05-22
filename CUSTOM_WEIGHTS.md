# Using Custom Weights in Pipeline App

This guide explains how to use custom model weights with the Pipeline App, particularly focusing on PyTorch/Lightning-style checkpoints.

## Supported Weight Formats

The application supports two types of custom weights:

### 1. Standard Cellpose Weights
- `.npy` files that are directly compatible with Cellpose
- Typically created using Cellpose's training interface

### 2. PyTorch/Lightning Checkpoints
- Files saved with `torch.save()`
- Often without file extensions (e.g., `checkpoint_run4_checkpoint_30000`)
- Contain model state dictionaries

## Using Custom Weights in the GUI

1. Launch the application:
   ```bash
   python GUI_test.py
   ```

2. Click on "Run Pipeline"

3. Select your input image when prompted

4. Choose an output directory

5. Select the channel to use (1-4)

6. When asked "Do you want to use custom weights instead of the default model?", click "Yes"

7. Browse to and select your custom weights file

8. The application will automatically detect the format and load the weights appropriately

## Understanding PyTorch/Lightning Checkpoints

PyTorch/Lightning checkpoints like `checkpoint_run4_checkpoint_30000` typically contain:
- A state dictionary with model weights
- Optimizer states
- Training metadata

Our application handles these files by:
1. Loading the checkpoint using `torch.load()`
2. Extracting the model's state dictionary
3. Loading the weights into a Cellpose model
4. Handling common prefix patterns (e.g., removing 'model.' prefix if needed)

## Troubleshooting Custom Weights

If you encounter issues with custom weights:

### Common Problems:

1. **Architecture Mismatch**: If the checkpoint was created with a different architecture than what Cellpose expects, loading may fail. The app will attempt to load with `strict=False` to handle partial matches.

2. **Key Naming Differences**: The app attempts to handle common key naming patterns, but if your checkpoint uses an unusual naming convention, it might not load correctly.

3. **Unsupported Format**: If your checkpoint is in an unusual format, the app may not recognize it correctly.

### Solutions:

1. **Check Console Output**: The application prints detailed information about the loading process, which can help diagnose issues.

2. **Convert Your Checkpoint**: You may need to convert your checkpoint to a more standard format:

   ```python
   import torch
   
   # Load your checkpoint
   checkpoint = torch.load("your_checkpoint")
   
   # Extract just the model state dict
   if "state_dict" in checkpoint:
       state_dict = checkpoint["state_dict"]
   else:
       state_dict = checkpoint
   
   # Save just the state dict
   torch.save(state_dict, "converted_checkpoint")
   ```

3. **Train a Standard Cellpose Model**: If all else fails, consider using Cellpose's training interface to create a compatible model.

## Converting to SafeTensors Format

For easier sharing and compatibility, you can convert your PyTorch checkpoint to the SafeTensors format:

```python
import torch
from safetensors.torch import save_file

# Load your checkpoint
checkpoint = torch.load("your_checkpoint")

# Extract the state dict
if "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

# Save as SafeTensors
save_file(state_dict, "model.safetensors")
```

The SafeTensors format provides better security and compatibility across platforms. 