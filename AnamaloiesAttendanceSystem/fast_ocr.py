import cv2
import numpy as np
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import pandas as pd
import os

class FastAttendanceOCR:
    def __init__(self):
        # Initialize OCR model
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
        if torch.cuda.is_available():
            self.model.to('cuda')
            
        # Create output directory
        self.output_dir = "Output_Reports"
        os.makedirs(self.output_dir, exist_ok=True)

    def process_image(self, image_path):
        """Process a single attendance sheet image"""
        # Read image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect table structure
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        
        # Find table cells
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cells = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > 100:  # Filter small noise
                cells.append((x, y, w, h))
        
        # Sort cells by position (top-to-bottom, left-to-right)
        cells.sort(key=lambda x: (x[1], x[0]))
        
        # Process each cell
        attendance_data = []
        for x, y, w, h in cells:
            # Extract cell
            cell = image[y:y+h, x:x+w]
            
            # Check if text is red
            hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
            lower_red = np.array([0,50,50])
            upper_red = np.array([10,255,255])
            red_mask = cv2.inRange(hsv, lower_red, upper_red)
            is_red = np.sum(red_mask) > (cell.shape[0] * cell.shape[1] * 20)
            
            # Convert to PIL for OCR
            cell_pil = Image.fromarray(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB))
            
            # Perform OCR
            pixel_values = self.processor(cell_pil, return_tensors="pt").pixel_values
            if torch.cuda.is_available():
                pixel_values = pixel_values.to('cuda')
            
            generated_ids = self.model.generate(pixel_values)
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Apply attendance rules
            text = text.strip().upper()
            if not text:
                status = 'Absent'  # Blank is absent
            elif is_red:
                status = 'Present' if text == 'P' else 'Absent'  # Red text is absent except 'P'
            else:
                status = 'Present'  # Non-red text is present
            
            attendance_data.append({
                'text': text,
                'status': status
            })
        
        return attendance_data

    def save_results(self, data, output_file):
        """Save attendance data to CSV"""
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.output_dir, output_file), index=False)
        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    # Initialize OCR
    ocr = FastAttendanceOCR()
    
    # Process image
    image_path = "Images/page_1.jpg"
    results = ocr.process_image(image_path)
    
    # Save results
    ocr.save_results(results, "attendance_results.csv")