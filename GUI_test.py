import os
import sys
import warnings
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QMessageBox, QWidget, QVBoxLayout
)
from PyQt5.QtGui import QIcon, QPalette, QColor, QFont
from PyQt5.QtCore import Qt

# Suppress resource tracker warnings
warnings.filterwarnings("ignore", category=UserWarning, module="resource_tracker")

# Set multiprocessing to use 'spawn' (important for macOS)
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

# Import functionalities from the `App` folder
try:
    from App.augmentation import apply_transformations_and_save, load_data
    from App.pipeline import ImageProcessor3D
    from App.single_cell import save_cells_with_volumes, recombine_cells_to_zstack
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pipeline App")
        self.setGeometry(100, 100, 900, 700)

        # Apply a custom theme
        self._apply_theme()

        # Main layout widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Buttons
        self._create_buttons()

    def _apply_theme(self):
        """Apply a custom color theme and fonts."""
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#2E3440"))  # Background color
        palette.setColor(QPalette.Button, QColor("#4C566A"))  # Button color
        palette.setColor(QPalette.ButtonText, QColor("#ECEFF4"))  # Button text
        palette.setColor(QPalette.WindowText, QColor("#ECEFF4"))  # Text color
        self.setPalette(palette)

        font = QFont("Arial", 12)
        self.setFont(font)

    def _create_buttons(self):
        """Creates buttons with enhanced styling and adds them to the layout."""
        self.image_augment_button = self._styled_button(
            "Image Augmentation", "icons/image.png", self.run_image_augmentation
        )
        self.segmentation_augment_button = self._styled_button(
            "Segmentation Augmentation", "icons/segmentation.png", self.run_segmentation_augmentation
        )
        self.pipeline_button = self._styled_button(
            "Run Pipeline", "icons/pipeline.png", self.run_pipeline
        )
        self.single_cell_button = self._styled_button(
            "Extract Single Cells", "icons/single_cell.png", self.run_single_cell
        )
        self.recombine_button = self._styled_button(
            "Recombine Z-Stack", "icons/recombine.png", self.run_recombine
        )

        # Add buttons to the layout
        self.layout.addWidget(self.image_augment_button)
        self.layout.addWidget(self.segmentation_augment_button)
        self.layout.addWidget(self.pipeline_button)
        self.layout.addWidget(self.single_cell_button)
        self.layout.addWidget(self.recombine_button)

    def _styled_button(self, text, icon_path, func):
        """Create a styled button with an icon."""
        button = QPushButton(text, self)
        button.setIcon(QIcon(icon_path))
        button.setMinimumHeight(50)
        button.setStyleSheet(
            """
            QPushButton {
                background-color: #4C566A;
                color: #ECEFF4;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #5E81AC;
            }
            QPushButton:pressed {
                background-color: #88C0D0;
            }
            """
        )
        button.clicked.connect(func)
        return button

    def _get_file(self, dialog_title, file_type="All Files (*)"):
        """Opens a file dialog to select a file."""
        file, _ = QFileDialog.getOpenFileName(self, dialog_title, filter=file_type)
        return file

    def _get_directory(self, dialog_title):
        """Opens a dialog to select a directory."""
        return QFileDialog.getExistingDirectory(self, dialog_title)

    def _show_message(self, title, message, is_error=False):
        """Displays a message box."""
        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setIcon(QMessageBox.Critical if is_error else QMessageBox.Information)
        msg.exec_()

    # GUI Updates
    def run_image_augmentation(self):
        """
        Handles the image augmentation process via the GUI.
        """
        input_image = self._get_file("Select Image for Augmentation", "Image Files (*.tiff *.tif *.npy)")
        if not input_image:
            return

        output_dir = self._get_directory("Select Output Directory")
        if not output_dir:
            return

        try:
            data = load_data(image_path=input_image)
            base_name = os.path.splitext(os.path.basename(input_image))[0]

            apply_transformations_and_save(
                data, image_path=input_image, output_image_dir=output_dir, num_versions=5
            )
            print(f"Image augmentation completed for {base_name}. Results saved to {output_dir}.")
            self._show_message("Success", f"Image augmentation completed for {base_name}.")
        except Exception as e:
            print(f"Error during image augmentation: {e}")
            self._show_message("Error", f"Image augmentation failed: {e}", is_error=True)


    def run_segmentation_augmentation(self):
        input_segmentation = self._get_file("Select Segmentation for Augmentation", "Segmentation Files (*.tiff *.tif *.npy)")
        if not input_segmentation:
            return

        output_dir = self._get_directory("Select Output Directory")
        if not output_dir:
            return

        try:
            data = load_data(seg_path=input_segmentation)
            apply_transformations_and_save(
                data, output_image_dir=None, output_seg_dir=output_dir, num_versions=5
            )
            self._show_message("Success", "Segmentation augmentation completed successfully.")
        except Exception as e:
            self._show_message("Error", f"Segmentation augmentation failed: {e}", is_error=True)

    def run_pipeline(self):
        input_image = self._get_file("Select Image for Pipeline", "Image Files (*.tiff *.tif *.npy)")
        if not input_image:
            return

        output_dir = self._get_directory("Select Output Directory")
        if not output_dir:
            return

        try:
            processor = ImageProcessor3D(model_type="cyto")
            processor.process_image(input_image, output_dir)
            self._show_message("Success", "Pipeline processing completed successfully.")
        except Exception as e:
            self._show_message("Error", f"Pipeline processing failed: {e}", is_error=True)

    def run_single_cell(self):
        segmented_stack = self._get_file("Select Segmented Stack", "Segmentation Files (*.tiff *.tif *.npy)")
        if not segmented_stack:
            return

        output_dir = self._get_directory("Select Output Directory for Single Cells")
        if not output_dir:
            return

        try:
            voxel_size = (0.5, 0.5, 1.0)  # Example voxel size
            save_cells_with_volumes(segmented_stack, output_dir, voxel_size)
            self._show_message("Success", "Single cell extraction completed successfully.")
        except Exception as e:
            self._show_message("Error", f"Single cell extraction failed: {e}", is_error=True)

    def run_recombine(self):
        single_cell_dir = self._get_directory("Select Single-Cell Directory")
        if not single_cell_dir:
            return

        combined_tiff_path, _ = QFileDialog.getSaveFileName(
            self, "Select Output File for Combined Z-Stack", filter="TIFF files (*.tiff *.tif)"
        )
        if not combined_tiff_path:
            return

        try:
            recombine_cells_to_zstack(single_cell_dir, combined_tiff_path)
            self._show_message("Success", "Recombined Z-Stack saved successfully.")
        except Exception as e:
            self._show_message("Error", f"Recombination failed: {e}", is_error=True)


if __name__ == "__main__":
    print("Starting the application...")
    app = QApplication(sys.argv)
    window = MainWindow()
    print("Showing the main window...")
    window.show()
    print("Entering the event loop...")
    sys.exit(app.exec_())
