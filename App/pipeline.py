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

    # Find the first slice with actual segmentations to start the merging process
    start_z = -1
    for z_idx in range(z):
        if np.any(segmented_stack[z_idx] > 0):
            start_z = z_idx
            break

    # If no cells are segmented in any slice, return the empty mask
    if start_z == -1:
        print("Warning: No cells detected in any slice.")
        return merged_mask

    # Process the first non-empty slice
    first_slice = segmented_stack[start_z]
    unique_labels = np.unique(first_slice)
    for label in unique_labels:
        if label == 0:
            continue
        merged_mask[start_z][first_slice == label] = next_global_id
        next_global_id += 1

    # Process subsequent slices from the one after the starting slice
    for z_idx in range(start_z + 1, z):
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
    def __init__(self, model_type="cyto", multi_channel=False, selected_channel=1, pretrained_model=None, diameter=None, cellprob_threshold=0.0, flow_threshold=0.4):
        """
        Initialize the ImageProcessor3D with a specific Cellpose model type.

        :param model_type: str, type of Cellpose model ('cyto', 'nuclei', 'cyto2').
        :param multi_channel: bool, whether the input image is multi-channel.
        :param selected_channel: int, the channel (1-4) to use for segmentation.
        :param pretrained_model: str, path to custom model weights. If None, uses the default model.
        :param diameter: float, cell diameter for segmentation. If None, Cellpose estimates it.
        :param cellprob_threshold: float, cell probability threshold.
        :param flow_threshold: float, flow error threshold.
        """
        self.multi_channel = multi_channel
        self.selected_channel = selected_channel
        self.using_custom_model = pretrained_model is not None
        self.diameter = diameter
        self.cellprob_threshold = cellprob_threshold
        self.flow_threshold = flow_threshold
        
        if self.using_custom_model:
            print(f"✅ Loading custom model weights from: {pretrained_model}")
            self.model = models.CellposeModel(pretrained_model=pretrained_model)
        else:
            print(f"✅ Loading default Cellpose model: {model_type}")
            self.model = models.CellposeModel(model_type=model_type)
        
        print(f"🔧 Cellpose model initialized successfully")
        print(f"📺 Multi-channel mode: {self.multi_channel}")
        print(f"📡 Selected channel: {self.selected_channel}")
        if self.using_custom_model:
            print(f"🎯 Using custom model weights: {pretrained_model}")
        else:
            print(f"🎯 Using default model type: {model_type}")

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
        Load and normalize the 3D image stack with per-slice normalization.
        Each slice is normalized individually to 0-255 range, matching Cellpose GUI behavior.

        :param input_path: str, path to the input TIFF image.
        :return: Normalized 3D numpy array.
        """
        image_stack = tiff.imread(input_path)
        
        print(f"Original image stack shape: {image_stack.shape}")
        print(f"Image data type: {image_stack.dtype}")
        print(f"Image intensity range: {image_stack.min()} - {image_stack.max()}")
        
        if self.multi_channel and image_stack.ndim == 4:
            print("Detected multi-channel image with shape:", image_stack.shape)

        if image_stack.ndim == 3:
            # Single-channel 3D stack
            normalized_stack = np.zeros_like(image_stack, dtype=np.uint8)
            for z in range(image_stack.shape[0]):
                slice_ = image_stack[z].astype(np.float32)
                slice_range = slice_.ptp()  # peak-to-peak (max - min)
                if slice_range > 0:
                    slice_ = (slice_ - slice_.min()) / (slice_range + 1e-8) * 255.0
                else:
                    slice_ = np.zeros_like(slice_)
                normalized_stack[z] = slice_.astype(np.uint8)  # Convert to uint8
                print(f"Slice {z}: range {slice_.min():.1f} - {slice_.max():.1f} (uint8)")
        elif image_stack.ndim == 4:
            # Multi-channel 3D stack
            normalized_stack = np.zeros_like(image_stack, dtype=np.uint8)
            for z in range(image_stack.shape[0]):
                for c in range(image_stack.shape[1]):
                    slice_ = image_stack[z, c].astype(np.float32)
                    slice_range = slice_.ptp()  # peak-to-peak (max - min)
                    if slice_range > 0:
                        slice_ = (slice_ - slice_.min()) / (slice_range + 1e-8) * 255.0
                    else:
                        slice_ = np.zeros_like(slice_)
                    normalized_stack[z, c] = slice_.astype(np.uint8)  # Convert to uint8
                    print(f"Slice {z}, Channel {c}: range {slice_.min():.1f} - {slice_.max():.1f} (uint8)")
        else:
            raise ValueError(f"Unsupported image shape: {image_stack.shape}")

        print(f"✅ Loaded and normalized image stack with shape: {normalized_stack.shape}")
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
        total_cells_detected = 0
        print("🔬 Starting segmentation...")
        for z in range(image_stack.shape[0]):
            print(f"Processing slice {z + 1}/{image_stack.shape[0]}...")
            
            # Get the slice for evaluation
            eval_slice = image_stack[z]
            print(f"  📊 Eval input - dtype: {eval_slice.dtype}, shape: {eval_slice.shape}, range: {eval_slice.min()} - {eval_slice.max()}")
            
            if self.multi_channel:
                # Use the selected channel (convert from 1-based to 0-based indexing).
                masks, flows, *_ = self.model.eval(
                    eval_slice,
                    diameter=self.diameter,
                    channels=[self.selected_channel - 1, 0],
                    cellprob_threshold=self.cellprob_threshold,
                    flow_threshold=self.flow_threshold
                )
            else:
                masks, flows, *_ = self.model.eval(
                    eval_slice,
                    diameter=self.diameter,
                    channels=[0, 0],
                    cellprob_threshold=self.cellprob_threshold,
                    flow_threshold=self.flow_threshold
                )
            
            # Count detected objects (excluding background = 0)
            unique_labels = np.unique(masks)
            n_cells_in_slice = len(unique_labels) - 1 if 0 in unique_labels else len(unique_labels)
            total_cells_detected += n_cells_in_slice
            
            print(f"  🎯 Slice {z + 1}: {n_cells_in_slice} cells detected")
            if n_cells_in_slice > 0:
                print(f"  📊 Mask labels: {unique_labels[:10]}...")  # Show first 10 labels
            
            segmented_stack[z] = masks
            all_flows.append(flows)
            
        print(f"✅ Segmentation completed. Total cells detected: {total_cells_detected}")
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
    # This is an example of how to run the processor directly.
    input_image_path = "path_to_input_image.tif"  # Replace with actual path
    output_directory = "output_dir"  # Replace with actual path
    
    # Example for using the default 'cyto' model
    print("Running with default 'cyto' model...")
    processor_default = ImageProcessor3D(model_type="cyto", selected_channel=1)
    # processor_default.process_image(input_image_path, output_directory)

    # Example for using a custom model
    # model_path = "path_to_your_model"  # Replace with actual path
    # if os.path.exists(model_path):
    #     print("\nRunning with custom model...")
    #     processor_custom = ImageProcessor3D(
    #         selected_channel=1,
    #         pretrained_model=model_path,
    #         diameter=30.0
    #     )
    #     processor_custom.process_image(input_image_path, output_directory)
    # else:
    #     print("\nCustom model path not found. Skipping custom model example.")
