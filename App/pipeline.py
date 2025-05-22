import os
import numpy as np
import tifffile as tiff
from cellpose import models
import torch

def merge_3d_masks(segmented_stack, overlap_threshold=0.1):
    """
    Merge 2D segmentation masks across slices into a single 3D segmentation mask
    that maintains cell identity across the z-stack.

    Parameters:
        segmented_stack (numpy.ndarray): 3D array (z, height, width) of segmentation labels.
        overlap_threshold (float): Minimum fraction of overlap required to link cells across slices.

    Returns:
        merged_mask (numpy.ndarray): 3D array with merged segmentation labels.
    """
    z, h, w = segmented_stack.shape
    merged_mask = np.zeros_like(segmented_stack, dtype=np.int32)
    next_global_id = 1

    # Process the first slice: assign new global labels for each cell.
    first_slice = segmented_stack[0]
    unique_labels = np.unique(first_slice)
    for label in unique_labels:
        if label == 0:
            continue
        merged_mask[0][first_slice == label] = next_global_id
        next_global_id += 1

    # Process subsequent slices.
    for z_idx in range(1, z):
        current_slice = segmented_stack[z_idx]
        prev_merged = merged_mask[z_idx - 1]
        unique_labels = np.unique(current_slice)
        for label in unique_labels:
            if label == 0:
                continue
            cell_mask = (current_slice == label)
            # Look at the previous slice in the same region.
            overlapping_prev = prev_merged[cell_mask]
            # Exclude background.
            overlapping_prev = overlapping_prev[overlapping_prev > 0]
            if overlapping_prev.size == 0:
                # No overlap; assign a new global label.
                global_label = next_global_id
                next_global_id += 1
            else:
                # Find the most common overlapping global label.
                unique_prev, counts = np.unique(overlapping_prev, return_counts=True)
                best_global = unique_prev[np.argmax(counts)]
                overlap_ratio = np.max(counts) / np.sum(cell_mask)
                # If the overlap is sufficient, use the previous global label.
                if overlap_ratio >= overlap_threshold:
                    global_label = best_global
                else:
                    global_label = next_global_id
                    next_global_id += 1
            # Assign the determined global label.
            merged_mask[z_idx][cell_mask] = global_label

    return merged_mask


class ImageProcessor3D:
    def __init__(self, model_type="cyto", multi_channel=False, selected_channel=1, pretrained_model=None):
        """
        Initialize the ImageProcessor3D with a specific Cellpose model type.

        :param model_type: str, type of Cellpose model ('cyto', 'nuclei', 'cyto2').
        :param multi_channel: bool, whether the input image is multi-channel.
        :param selected_channel: int, the channel (1-4) to use for segmentation.
        :param pretrained_model: str, path to custom model weights. If None, uses the default model.
        """
        self.multi_channel = multi_channel
        self.selected_channel = selected_channel
        self.using_custom_model = pretrained_model is not None
        
        if pretrained_model is not None:
            print(f"Loading custom model weights from: {pretrained_model}")
            # Check if the file is a PyTorch/Lightning checkpoint
            if self._is_pytorch_checkpoint(pretrained_model):
                self.model = self._load_from_pytorch_checkpoint(pretrained_model, model_type)
            else:
                # Use standard Cellpose loading for .npy files or other formats
                self.model = models.CellposeModel(pretrained_model=pretrained_model)
        else:
            self.model = models.CellposeModel(model_type=model_type)
        
        print(f"Cellpose model initialized. Multi-channel mode: {self.multi_channel}. Selected channel: {self.selected_channel}")
        if self.using_custom_model:
            print(f"Using custom model weights: {pretrained_model}")
        else:
            print(f"Using default model type: {model_type}")

    def _is_pytorch_checkpoint(self, file_path):
        """
        Check if the file is a PyTorch/Lightning checkpoint.
        
        :param file_path: str, path to the file
        :return: bool, True if it's a PyTorch checkpoint
        """
        try:
            # Try to load the file as a PyTorch checkpoint
            checkpoint = torch.load(file_path, map_location="cpu")
            # Check if it has the expected structure
            return isinstance(checkpoint, dict) and ("state_dict" in checkpoint or any(k.endswith(".weight") for k in checkpoint.keys()))
        except:
            return False

    def _load_from_pytorch_checkpoint(self, checkpoint_path, model_type):
        """
        Load a model from a PyTorch/Lightning checkpoint.
        
        :param checkpoint_path: str, path to the checkpoint file
        :param model_type: str, type of model to initialize if needed
        :return: CellposeModel instance with loaded weights
        """
        print("Loading PyTorch/Lightning checkpoint...")
        try:
            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            # Initialize a base CellposeModel
            model = models.CellposeModel(model_type=model_type)
            
            # Extract the state dict
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            
            # Load the state dict into the model
            # First, check if we need to remove 'model.' prefix from keys
            if all(k.startswith('model.') for k in state_dict.keys() if k.endswith('.weight')):
                state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            
            # Try to load the state dict
            try:
                model.net.load_state_dict(state_dict)
                print("Successfully loaded weights into model")
            except Exception as e:
                print(f"Warning: Could not load state dict directly: {e}")
                print("Attempting to load with strict=False...")
                model.net.load_state_dict(state_dict, strict=False)
                print("Successfully loaded weights with strict=False")
            
            return model
        except Exception as e:
            print(f"Error loading PyTorch checkpoint: {e}")
            print("Falling back to default CellposeModel")
            return models.CellposeModel(model_type=model_type)

    def process_image(self, input_path, output_dir):
        """
        Process the input image stack, segment cells, merge 3D masks, and save results.
        """
        input_filename = os.path.basename(input_path)
        image_stack = self.load_and_normalize_image(input_path)

        # Automatically detect multi-channel images.
        self.multi_channel = True if image_stack.ndim == 4 else False
        print(f"Multi-channel mode: {self.multi_channel}")

        if self.multi_channel and image_stack.ndim == 4:
            num_channels = image_stack.shape[1]
            if not (1 <= self.selected_channel <= num_channels):
                raise ValueError(f"Selected channel {self.selected_channel} is out of bounds for input image with {num_channels} channels.")
            print(f"Using channel {self.selected_channel} out of {num_channels} available.")

        segmented_stack, flows = self.segment_3d(image_stack)
        # Merge the per-slice segmentation masks to maintain cell identity across slices.
        merged_mask = merge_3d_masks(segmented_stack, overlap_threshold=0.1)
        self.save_results(image_stack, segmented_stack, merged_mask, flows, output_dir, input_filename)

    def load_and_normalize_image(self, input_path):
        """
        Load and normalize the 3D image stack to range 0-255.

        :param input_path: str, path to the input TIFF image.
        :return: Normalized 3D numpy array.
        """
        image_stack = tiff.imread(input_path)
        
        if self.multi_channel and image_stack.ndim == 4:
            print("Detected multi-channel image with shape:", image_stack.shape)
        
        normalized_stack = (image_stack - image_stack.min()) / (image_stack.max() - image_stack.min()) * 255.0
        print(f"Loaded and normalized image stack with shape: {image_stack.shape}")
        return normalized_stack

    def segment_3d(self, image_stack):
        """
        Perform 3D segmentation on the image stack using Cellpose.
        """
        if self.multi_channel and image_stack.ndim == 4:
            # Allocate segmentation mask for each slice (2D mask per slice).
            segmented_stack = np.zeros((image_stack.shape[0], image_stack.shape[2], image_stack.shape[3]), dtype=np.uint16)
        else:
            segmented_stack = np.zeros(image_stack.shape, dtype=np.uint16)

        all_flows = []
        print("Starting segmentation...")
        for z in range(image_stack.shape[0]):
            print(f"Processing slice {z + 1}/{image_stack.shape[0]}...")
            if self.multi_channel:
                # Use the selected channel (convert from 1-based to 0-based indexing).
                masks, flows, *_ = self.model.eval(
                    image_stack[z],
                    diameter=None,
                    channels=[self.selected_channel - 1, 0]
                )
            else:
                masks, flows, *_ = self.model.eval(image_stack[z], diameter=None, channels=[0, 0])
            segmented_stack[z] = masks
            all_flows.append(flows)
        print("Segmentation completed.")
        return segmented_stack, all_flows

    def save_results(self, image_stack, segmented_stack, merged_mask, flows, output_dir, input_filename):
        """
        Save segmentation results in NPY and TIFF formats.

        :param image_stack: Original normalized image stack.
        :param segmented_stack: Per-slice segmentation masks.
        :param merged_mask: Merged 3D segmentation mask with consistent cell IDs.
        :param flows: Flow fields from Cellpose.
        :param output_dir: Directory to save the output files.
        :param input_filename: Original input file's name.
        """
        base_name = os.path.splitext(os.path.basename(input_filename))[0]
        os.makedirs(output_dir, exist_ok=True)

        output_data = {
            "img": image_stack.astype(np.float32),
            "masks": segmented_stack.astype(np.uint16),
            "merged_masks": merged_mask.astype(np.uint16),
            "flows": flows,
        }
        npy_output_path = os.path.join(output_dir, f"{base_name}_segmentation_results.npy")
        np.save(npy_output_path, output_data)

        # If multi-channel, collapse extra channel dimension if needed.
        if self.multi_channel and segmented_stack.ndim == 4:
            print(f"Original segmented mask shape: {segmented_stack.shape}")
            segmented_stack = np.max(segmented_stack, axis=1)
            print(f"Fixed segmented mask shape: {segmented_stack.shape}")

        # Save merged masks in TIFF format.
        tiff_output_path = os.path.join(output_dir, f"{base_name}_merged_segmented.tiff")
        tiff.imwrite(tiff_output_path, merged_mask)
        print(f"Results saved to:\n- {npy_output_path}\n- {tiff_output_path}")

if __name__ == "__main__":
    input_image_path = "path_to_input_image.tif"  # Replace with actual path.
    output_directory = "output_dir"  # Replace with actual path.
    
    # For example, using channel 2.
    processor = ImageProcessor3D(model_type="cyto", selected_channel=2)
    processor.process_image(input_image_path, output_directory)
