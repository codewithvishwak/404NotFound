import os

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Directory paths
IMAGES_DIR = os.path.join(PROJECT_ROOT, "Images")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Output_Reports")
OCR_SLICES_DIR = os.path.join(PROJECT_ROOT, "ocr_slices")
PROCESSED_IMAGES_DIR = os.path.join(PROJECT_ROOT, "processed_images")

# Create directories if they don't exist
for directory in [IMAGES_DIR, OUTPUT_DIR, OCR_SLICES_DIR, PROCESSED_IMAGES_DIR]:
    os.makedirs(directory, exist_ok=True)

# Tool configurations
poppler_locations = [
    r"C:\Program Files\poppler-23.11.0\Library\bin",
    r"C:\Program Files\poppler\bin",
    r"C:\poppler\bin"
]

POPPLER_PATH = None
for loc in poppler_locations:
    if os.path.exists(loc):
        POPPLER_PATH = loc
        break

# Supported file formats
SUPPORTED_FORMATS = {
    'image': ['.jpg', '.jpeg', '.png', '.bmp'],
    'pdf': ['.pdf']
}