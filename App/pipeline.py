import os
import numpy as np
import tifffile as tiff
from cellpose import models

class ImageProcessor3D:
    def __init__(self, model_type="cyto"):
        """
        Initialize the ImageProcessor3D with a specific Cellpose model type.

        :param model_type: str, type of Cellpose model ('cyto', 'nuclei', 'cyto2').
        """
        self.model = models.Cellpose(model_type=model_type)
        print("Cellpose model initialized.")

    def process_image(self, input_path, output_dir):
        """
        Process the input image stack, segment cells, and save results.

        :param input_path: str, path to the input TIFF image.
        :param output_dir: str, directory to save the output files.
        """
        image_stack = self.load_and_normalize_image(input_path)
        segmented_stack, flows = self.segment_3d(image_stack)
        self.save_results(image_stack, segmented_stack, flows, output_dir)

    def load_and_normalize_image(self, input_path):
        """
        Load and normalize the 3D image stack to range 0-255.

        :param input_path: str, path to the input TIFF image.
        :return: Normalized 3D numpy array.
        """
        image_stack = tiff.imread(input_path)
        normalized_stack = (image_stack - image_stack.min()) / (image_stack.max() - image_stack.min()) * 255.0
        print(f"Loaded and normalized image stack with shape: {image_stack.shape}")
        return normalized_stack

    def segment_3d(self, image_stack):
        """
        Perform 3D segmentation on the image stack using Cellpose.

        :param image_stack: 3D numpy array of the input image stack.
        :return: Tuple of segmented stack and flow fields.
        """
        segmented_stack = np.zeros_like(image_stack, dtype=np.uint16)
        all_flows = []

        print("Starting segmentation...")
        for z in range(image_stack.shape[0]):
            print(f"Processing slice {z + 1}/{image_stack.shape[0]}...")
            masks, flows, *_ = self.model.eval(image_stack[z], diameter=None, channels=[0, 0])
            segmented_stack[z] = masks
            all_flows.append(flows)

        print("Segmentation completed.")
        return segmented_stack, all_flows

    def save_results(self, image_stack, segmented_stack, flows, output_dir):
        """
        Save segmentation results in NPY and TIFF formats.

        :param image_stack: Original normalized image stack.
        :param segmented_stack: Segmented masks.
        :param flows: Flow fields from Cellpose.
        :param output_dir: Directory to save the output files.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save NPY results
        output_data = {
            "img": image_stack.astype(np.float32),
            "masks": segmented_stack.astype(np.uint16),
            "flows": flows,
        }
        npy_output_path = os.path.join(output_dir, "segmentation_results.npy")
        np.save(npy_output_path, output_data)

        # Save segmented stack as TIFF
        tiff_output_path = os.path.join(output_dir, "segmented_stack.tif")
        tiff.imwrite(tiff_output_path, segmented_stack)

        print(f"Results saved to:\n- {npy_output_path}\n- {tiff_output_path}")


if __name__ == "__main__":
    # Example usage
    input_image_path = "path_to_input_image.tif"  # Replace with actual path
    output_directory = "output_dir"  # Replace with actual path

    processor = ImageProcessor3D(model_type="cyto")
    processor.process_image(input_image_path, output_directory)
