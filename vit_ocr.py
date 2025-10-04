from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import cv2
import numpy as np

class VisionOCR:
    def __init__(self):
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
        if torch.cuda.is_available():
            self.model.to('cuda')
    
    def preprocess_image(self, image_path):
        # Read image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Threshold the image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find table structure
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
        
        # Detect lines
        horizontal_lines = cv2.erode(thresh, horizontal_kernel, iterations=1)
        vertical_lines = cv2.erode(thresh, vertical_kernel, iterations=1)
        
        # Get table cells
        table_mask = cv2.add(horizontal_lines, vertical_lines)
        contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by y-coordinate (top to bottom)
        bounding_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > 1000:  # Filter small boxes
                bounding_boxes.append((y, x, y+h, x+w))
        
        bounding_boxes.sort()  # Sort by y-coordinate
        return img, bounding_boxes

    def cell_to_text(self, image, box):
        # Extract cell from image
        cell_image = image[box[0]:box[2], box[1]:box[3]]
        
        # Convert to PIL Image
        cell_pil = Image.fromarray(cv2.cvtColor(cell_image, cv2.COLOR_BGR2RGB))
        
        # Process with OCR
        pixel_values = self.processor(cell_pil, return_tensors="pt").pixel_values
        if torch.cuda.is_available():
            pixel_values = pixel_values.to('cuda')
        
        # Generate text
        generated_ids = self.model.generate(pixel_values)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return text.strip()

    def process_attendance_sheet(self, image_path):
        """Process attendance sheet and extract relevant information"""
        # Preprocess image and get table cells
        image, boxes = self.preprocess_image(image_path)
        
        attendance_data = []
        header_processed = False
        
        # Process each row
        for i in range(0, len(boxes), 2):  # Assuming 2 cells per row (name and status)
            if i + 1 >= len(boxes):
                break
                
            name_box = boxes[i]
            status_box = boxes[i + 1]
            
            # Skip header row
            if not header_processed:
                header_processed = True
                continue
            
            # Extract text from cells
            name = self.cell_to_text(image, name_box)
            status = self.cell_to_text(image, status_box)
            
            if name and status:  # Only add if both cells have content
                attendance_data.append({
                    'name': name,
                    'status': status
                })
        
        return attendance_data