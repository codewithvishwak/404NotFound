import cv2
import numpy as np
import os
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime
from vit_ocr import VLLMOCRTool

@dataclass
class StudentRecord:
    student_id: str
    student_name: str
    status: str
    attendance_dates: Dict[str, str]  # date -> attendance status

class OcrToTableTool:
    def __init__(self, image, original_image):
        self.thresholded_image = image
        self.original_image = original_image
        self.vllm_ocr = VLLMOCRTool()
        
        # Set up directories
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.ocr_slices_dir = os.path.join(script_dir, "ocr_slices")
        os.makedirs(self.ocr_slices_dir, exist_ok=True)

    def execute(self):
        """Main execution flow"""
        # Pre-process the image
        self.detect_table_structure()
        self.find_cells()
        self.extract_text_from_cells()
        return self.table_data

    def detect_table_structure(self):
        """Detect and enhance table structure"""
        # Convert to grayscale if needed
        if len(self.original_image.shape) == 3:
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.original_image

        # Binary threshold
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))

        horizontal_lines = cv2.erode(binary, horizontal_kernel)
        horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel)

        vertical_lines = cv2.erode(binary, vertical_kernel)
        vertical_lines = cv2.dilate(vertical_lines, vertical_kernel)

        # Combine lines
        self.table_structure = cv2.add(horizontal_lines, vertical_lines)
        
        # Save intermediate result
        cv2.imwrite('processed_images/table_structure.jpg', self.table_structure)

    def find_cells(self):
        """Find table cells using contour detection"""
        # Find contours of table cells
        contours, _ = cv2.findContours(self.table_structure, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get image dimensions for filtering
        height, width = self.original_image.shape[:2]
        min_area = (width * height * 0.0001)  # Min cell size (0.01% of image)
        max_area = (width * height * 0.1)     # Max cell size (10% of image)
        
        # Filter and store valid cells
        self.cells = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if min_area < area < max_area:
                self.cells.append([x, y, w, h])

        # Sort cells by position (top to bottom, left to right)
        self.cells.sort(key=lambda box: (box[1], box[0]))
        
        # Visualize detected cells
        viz_img = self.original_image.copy()
        for x, y, w, h in self.cells:
            cv2.rectangle(viz_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imwrite('processed_images/detected_cells.jpg', viz_img)

    def extract_text_from_cells(self):
        """Extract and process text from each cell"""
        if not self.cells:
            print("No cells detected!")
            self.table_data = []
            return

        # Process cells into rows based on y-coordinate
        rows = []
        current_row = [self.cells[0]]
        current_y = self.cells[0][1]

        for cell in self.cells[1:]:
            if abs(cell[1] - current_y) < 20:  # Cells in same row
                current_row.append(cell)
            else:
                rows.append(current_row)
                current_row = [cell]
                current_y = cell[1]
        rows.append(current_row)

        # Process each row
        self.table_data = []
        for i, row in enumerate(rows):
            row_data = []
            for x, y, w, h in row:
                # Extract cell image
                cell_img = self.original_image[y:y+h, x:x+w]
                
                # Save cell image
                temp_path = os.path.join(self.ocr_slices_dir, f'cell_{i}_{x}.jpg')
                cv2.imwrite(temp_path, cell_img)
                
                # Process cell
                text, is_red = self.vllm_ocr.recognize_text(temp_path)
                
                # Apply attendance rules
                if not text or text.strip() == "":
                    attendance = "Absent"
                elif is_red:
                    attendance = "Present" if text.strip().upper() == "P" else "Absent"
                else:
                    attendance = "Present"
                
                row_data.append({
                    "text": text.strip() if text else "",
                    "status": attendance
                })
                
                # Cleanup
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            self.table_data.append(row_data)
