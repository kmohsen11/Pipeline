import subprocess
import sys

def install(package):
    """Install a package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    print("Installing dependencies...")

    # List of required packages
    required_packages = [
        "PyQt5",
        "numpy",
        "tifffile",
        "monai",
        "cellpose"
    ]

    for package in required_packages:
        try:
            print(f"Installing {package}...")
            install(package)
        except Exception as e:
            print(f"Failed to install {package}: {e}")

    print("All dependencies installed successfully!")

if __name__ == "__main__":
    main()
