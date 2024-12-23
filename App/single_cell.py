import os
import numpy as np
import tifffile as tiff

def save_cells_with_volumes(segmented_stack_path, output_dir, voxel_size):
    """
    Save each segmented cell and calculate its volume based on voxel dimensions.

    :param segmented_stack_path: str, path to the segmented 3D stack (can be .npy or .tif).
    :param output_dir: str, directory to save the individual cell files.
    :param voxel_size: tuple, (x, y, z) dimensions of a voxel in microns.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load segmented stack
    if segmented_stack_path.endswith('.npy'):
        segmented_stack = np.load(segmented_stack_path)
    elif segmented_stack_path.endswith('.tif'):
        segmented_stack = tiff.imread(segmented_stack_path)
    else:
        raise ValueError("Unsupported file format. Use .npy or .tif.")

    # Get unique cell IDs (excluding background ID 0)
    unique_ids = np.unique(segmented_stack)
    unique_ids = unique_ids[unique_ids > 0]

    print(f"Found {len(unique_ids)} cells in the segmented stack.")

    # Calculate voxel volume in microns^3
    voxel_volume = voxel_size[0] * voxel_size[1] * voxel_size[2]

    # Process each cell
    for cell_id in unique_ids:
        print(f"Processing cell ID: {cell_id}")

        # Create a mask for the current cell
        cell_mask = segmented_stack == cell_id

        # Calculate the physical volume of the cell
        physical_volume = np.sum(cell_mask) * voxel_volume
        print(f"Cell ID {cell_id} volume: {physical_volume:.2f} microns^3")

        # Save cell data as .npy
        npy_path = os.path.join(output_dir, f'cell_{cell_id}.npy')
        np.save(npy_path, cell_mask)

        # Save cell data as .tif
        tif_path = os.path.join(output_dir, f'cell_{cell_id}.tif')
        tiff.imwrite(tif_path, cell_mask.astype(np.uint16))

        print(f"Saved Cell ID {cell_id}: {npy_path} and {tif_path}")

def recombine_cells_to_zstack(single_cell_dir, combined_tiff_path):
    """
    Combine single-cell data back into a z-stack.

    :param single_cell_dir: str, directory containing single-cell TIFF files.
    :param combined_tiff_path: str, path to save the combined z-stack TIFF.
    """
    # List all single-cell TIFF files
    cell_files = sorted(
        [os.path.join(single_cell_dir, f) for f in os.listdir(single_cell_dir) if f.endswith('.tif')]
    )

    if not cell_files:
        raise ValueError(f"No TIFF files found in directory: {single_cell_dir}")

    # Assume the first file gives the dimensions of the z-stack
    first_cell = tiff.imread(cell_files[0])
    z_dim, height, width = first_cell.shape

    # Initialize an empty array for the combined z-stack
    combined_stack = np.zeros((z_dim, height, width), dtype=np.uint16)

    # Process each single-cell TIFF file
    for cell_file in cell_files:
        print(f"Adding {cell_file} to z-stack")
        cell_data = tiff.imread(cell_file)

        # Check if the dimensions match
        if cell_data.shape != (z_dim, height, width):
            raise ValueError(f"Dimension mismatch: {cell_file} has shape {cell_data.shape}, expected {(z_dim, height, width)}")

        # Add the cell data to the combined stack
        combined_stack += cell_data.astype(np.uint16)

    # Save the combined z-stack as a multi-layer TIFF
    tiff.imwrite(combined_tiff_path, combined_stack)
    print(f"Combined z-stack saved to: {combined_tiff_path}")

if __name__ == "__main__":
    # Paths and parameters
    segmented_stack_path = "path_to_segmented_stack.tif"  # Replace with actual path
    single_cell_output_dir = "path_to_single_cell_output"  # Replace with actual path
    voxel_size = (0.5, 0.5, 1.0)  # Microns (x, y, z)
    
    # Recombine parameters
    original_image_path = "path_to_original_image.tif"  # Replace with actual path
    combined_tiff_path = "path_to_combined_zstack.tif"  # Replace with actual path

    # Step 1: Extract and save individual cells with volumes
    save_cells_with_volumes(segmented_stack_path, single_cell_output_dir, voxel_size)

    # Step 2: Recombine extracted cells into a z-stack
    recombine_cells_to_zstack(single_cell_output_dir, original_image_path, combined_tiff_path)
