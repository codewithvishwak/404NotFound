from fast_ocr import FastAttendanceOCR
import os

def main():
    # Initialize OCR
    ocr = FastAttendanceOCR()
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "Images", "page_1.jpg")
    
    # Process image
    print("Processing attendance sheet...")
    results = ocr.process_image(image_path)
    
    # Save results
    ocr.save_results(results, "attendance_results.csv")
    print("Done!")

if __name__ == "__main__":
    main()