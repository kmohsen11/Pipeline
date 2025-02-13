import os
import numpy as np
import tifffile as tiff
from cellpose import models

class ImageProcessor3D:
    def __init__(self, model_type="cyto", multi_channel=False):
        """
        Initialize the ImageProcessor3D with a specific Cellpose model type.

        :param model_type: str, type of Cellpose model ('cyto', 'nuclei', 'cyto2').
        :param multi_channel: bool, whether the input image is multi-channel.
        """
        self.model = models.Cellpose(model_type=model_type)
        self.multi_channel = multi_channel
        print(f"Cellpose model initialized. Multi-channel mode: {self.multi_channel}")

    def process_image(self, input_path, output_dir):
        """
        Process the input image stack, segment cells, and save results.
        """
        input_filename = os.path.basename(input_path)
        image_stack = self.load_and_normalize_image(input_path)

        #Automatically detect multi-channel images
        self.multi_channel = True if image_stack.ndim == 4 else False
        print(f"Multi-channel mode: {self.multi_channel}")

        segmented_stack, flows = self.segment_3d(image_stack)
        self.save_results(image_stack, segmented_stack, flows, output_dir, input_filename)


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
        # Ensure correct segmentation stack shape
        if self.multi_channel and image_stack.ndim == 4:
            segmented_stack = np.zeros((image_stack.shape[0], image_stack.shape[2], image_stack.shape[3]), dtype=np.uint16)
        else:
            segmented_stack = np.zeros(image_stack.shape, dtype=np.uint16)

        all_flows = []

        print("Starting segmentation...")
        for z in range(image_stack.shape[0]):
            print(f"Processing slice {z + 1}/{image_stack.shape[0]}...")

            if self.multi_channel:
                masks, flows, *_ = self.model.eval(image_stack[z], diameter=None, channels=[1, 2])  # Adjust based on dataset
            else:
                masks, flows, *_ = self.model.eval(image_stack[z], diameter=None, channels=[0, 0])

            segmented_stack[z] = masks
            all_flows.append(flows)

        print("Segmentation completed.")
        return segmented_stack, all_flows


    def save_results(self, image_stack, segmented_stack, flows, output_dir, input_filename):
        """
        Save segmentation results in NPY and TIFF formats, ensuring correct formatting for Cellpose.

        :param image_stack: Original normalized image stack.
        :param segmented_stack: Segmented masks.
        :param flows: Flow fields from Cellpose.
        :param output_dir: Directory to save the output files.
        :param input_filename: Original input file's name.
        """
        base_name = os.path.splitext(os.path.basename(input_filename))[0]
        os.makedirs(output_dir, exist_ok=True)

        output_data = {
            "img": image_stack.astype(np.float32),
            "masks": segmented_stack.astype(np.uint16),
            "flows": flows,
        }
        npy_output_path = os.path.join(output_dir, f"{base_name}_segmentation_results.npy")
        np.save(npy_output_path, output_data)

        # **Fix the mask format issue**
        if self.multi_channel and segmented_stack.ndim == 4:
            print(f"Original segmented mask shape: {segmented_stack.shape}")
            segmented_stack = np.max(segmented_stack, axis=1)  # Collapse the extra channel dimension
            print(f"Fixed segmented mask shape: {segmented_stack.shape}")

        # Save masks in TIFF format
        tiff_output_path = os.path.join(output_dir, f"{base_name}_segmented.tiff")
        tiff.imwrite(tiff_output_path, segmented_stack)

        print(f"Results saved to:\n- {npy_output_path}\n- {tiff_output_path}")

if __name__ == "__main__":
    input_image_path = "path_to_input_image.tif"  # Replace with actual path
    output_directory = "output_dir"  # Replace with actual path
    
    processor = ImageProcessor3D(model_type="cyto")
    processor.process_image(input_image_path, output_directory)
