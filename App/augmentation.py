import os
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import (
    Compose, ScaleIntensity, RandRotate90, RandAffine,
    Rand3DElastic, RandGaussianNoise, RandGaussianSmooth, RandZoom
)
from tifffile import TiffFile, TiffWriter

import torch

# Step 1: Define transformations
def define_transforms():
    image_transforms = Compose([
        ScaleIntensity(),
        RandRotate90(prob=0.5, spatial_axes=(0, 1)),
        RandAffine(prob=0.5, translate_range=(5, 5, 5), rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1)),
        RandGaussianNoise(prob=0.2, mean=0.0, std=0.1),
        RandGaussianSmooth(prob=0.2, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0))
    ])

    segmentation_transforms = Compose([
        RandRotate90(prob=0.5, spatial_axes=(0, 1)),
        RandAffine(prob=0.5, translate_range=(5, 5, 5), rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1)),
        RandGaussianNoise(prob=0.2, mean=0.0, std=0.1),
        RandGaussianSmooth(prob=0.2, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0))
    ])
    return image_transforms, segmentation_transforms
# Updated `load_data` function
def load_data(image_path=None, seg_path=None):
    data = {}
    try:
        # Load image if provided
        if image_path:
            if image_path.endswith('.npy'):
                data["image"] = np.load(image_path, allow_pickle=True)
            else:
                with TiffFile(image_path) as tif:
                    data["image"] = tif.asarray()
            print(f"Loaded image shape: {data['image'].shape}")

        # Load segmentation if provided
        if seg_path:
            if seg_path.endswith('.npy'):
                data["segmentation"] = np.load(seg_path, allow_pickle=True)
            else:
                with TiffFile(seg_path) as tif:
                    data["segmentation"] = tif.asarray()
            
            print(data['segmentation'].shape)
          
            # Validate segmentation shape
            if data["segmentation"].ndim == 0:  # Scalar check
                raise ValueError("Segmentation data is a scalar (empty).")
            if data["segmentation"].ndim < 3:  # Add channel dimension if missing
                data["segmentation"] = np.expand_dims(data["segmentation"], axis=0)

        # Ensure matching shapes if both are provided
        if "image" in data and "segmentation" in data:
            if data["segmentation"].shape != data["image"].shape:
                print("Segmentation shape mismatch; using a zero array fallback.")
                data["segmentation"] = np.zeros_like(data["image"], dtype=np.float32)
      
        return data

    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def apply_transformations_and_save(data, image_path=None, seg_path=None, output_image_dir=None, output_seg_dir=None, num_versions=20):
    """
    Apply transformations to the input data and save augmented versions with consistent naming.

    :param data: dict, containing image and/or segmentation data.
    :param image_path: str, input image path for naming.
    :param seg_path: str, input segmentation path for naming.
    :param output_image_dir: str, directory to save augmented images.
    :param output_seg_dir: str, directory to save augmented segmentations.
    :param num_versions: int, number of augmented versions to generate.
    """
    base_name = os.path.splitext(os.path.basename(image_path or seg_path))[0]

    # Ensure output directories exist
    if output_image_dir:
        os.makedirs(output_image_dir, exist_ok=True)
    if output_seg_dir:
        os.makedirs(output_seg_dir, exist_ok=True)

    image_transforms, segmentation_transforms = define_transforms()

    for i in range(num_versions):
        try:
            if image_path:
                augmented_image = image_transforms(data["image"])
                if isinstance(augmented_image, torch.Tensor):
                    augmented_image = augmented_image.numpy()

                output_image_path = os.path.join(output_image_dir, f"{base_name}_aug_{i + 1}.tiff")
                with TiffWriter(output_image_path) as tif:
                    tif.write(augmented_image.astype(np.float32))
                print(f"Saved version {i + 1} - Image: {output_image_path}")

            if seg_path:
                augmented_segmentation = segmentation_transforms(data["segmentation"])
                if isinstance(augmented_segmentation, torch.Tensor):
                    augmented_segmentation = augmented_segmentation.numpy()

                output_seg_path = os.path.join(output_seg_dir, f"{base_name}_aug_{i + 1}.npy")
                np.save(output_seg_path, augmented_segmentation)
                print(f"Saved version {i + 1} - Segmentation: {output_seg_path}")

        except Exception as e:
            print(f"Error during augmentation {i + 1}: {e}")

# Main execution
if __name__ == "__main__":
    # Define paths
    image_path = "/Users/nayeb/Downloads/DL_MBL_Data_DP/annotated_images/image_test/TC_17_Cortical2-1-2.tiff"
    seg_path = "/Users/nayeb/Downloads/DL_MBL_Data_DP/annotated_seg/seg_test/TC_17_Cortical2-1-2_seg.npy"
    output_image_dir = "/Users/nayeb/Desktop/UCSF_LAB_PRACTICE"
    output_seg_dir = "/Users/nayeb/Desktop/UCSF_LAB_PRACTICE"

    try:
        # Load data
        sample_data = load_data(image_path, seg_path)

        # Apply transformations and save augmented data
        apply_transformations_and_save(sample_data, output_image_dir, output_seg_dir)

    except Exception as e:
        print(f"Error in main execution: {e}")
