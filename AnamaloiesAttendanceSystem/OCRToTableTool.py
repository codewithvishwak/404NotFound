import cv2
import numpy as np
import subprocess
from PIL import Image
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
        self.dilate_image()
        self.store_process_image('0_dilated_image.jpg', self.dilated_image)
        self.find_contours()
        self.store_process_image('1_contours.jpg', self.image_with_contours_drawn)
        self.convert_contours_to_bounding_boxes()
        self.store_process_image('2_bounding_boxes.jpg', self.image_with_all_bounding_boxes)
        self.mean_height = self.get_mean_height_of_bounding_boxes()
        self.sort_bounding_boxes_by_y_coordinate()
        self.club_all_bounding_boxes_by_similar_y_coordinates_into_rows()
        self.sort_all_rows_by_x_coordinate()
        self.crop_each_bounding_box_and_ocr()
        self.generate_csv_file()
        # Return structured table (list of rows). callers can also call get_table_as_dataframe()
        return self.table

    def threshold_image(self):
        return cv2.threshold(self.grey_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def convert_image_to_grayscale(self):
        return cv2.cvtColor(self.image, self.dilated_image)

    def dilate_image(self):
        # Create a larger kernel for better table structure detection
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
        
        # Detect horizontal and vertical lines
        horizontal = cv2.erode(self.thresholded_image, horizontal_kernel, iterations=3)
        vertical = cv2.erode(self.thresholded_image, vertical_kernel, iterations=3)
        
        # Combine the lines
        self.dilated_image = cv2.add(horizontal, vertical)
        
        # Dilate to connect components
        kernel = np.ones((3,3), np.uint8)
        self.dilated_image = cv2.dilate(self.dilated_image, kernel, iterations=2)
        simple_kernel = np.ones((5,5), np.uint8)
        self.dilated_image = cv2.dilate(self.dilated_image, simple_kernel, iterations=2)
    
    def find_contours(self):
        result = cv2.findContours(self.dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = result[0]
        self.image_with_contours_drawn = self.original_image.copy()
        cv2.drawContours(self.image_with_contours_drawn, self.contours, -1, (0, 255, 0), 3)
    
    def approximate_contours(self):
        self.approximated_contours = []
        for contour in self.contours:
            approx = cv2.approxPolyDP(contour, 3, True)
            self.approximated_contours.append(approx)

    def draw_contours(self):
        self.image_with_contours = self.original_image.copy()
        cv2.drawContours(self.image_with_contours, self.approximated_contours, -1, (0, 255, 0), 5)

    def convert_contours_to_bounding_boxes(self):
        self.bounding_boxes = []
        self.image_with_all_bounding_boxes = self.original_image.copy()
        for contour in self.contours:
            x, y, w, h = cv2.boundingRect(contour)
            self.bounding_boxes.append((x, y, w, h))
            self.image_with_all_bounding_boxes = cv2.rectangle(self.image_with_all_bounding_boxes, (x, y), (x + w, y + h), (0, 255, 0), 5)

    def get_mean_height_of_bounding_boxes(self):
        heights = []
        for bounding_box in self.bounding_boxes:
            x, y, w, h = bounding_box
            heights.append(h)
        return np.mean(heights)

    def sort_bounding_boxes_by_y_coordinate(self):
        self.bounding_boxes = sorted(self.bounding_boxes, key=lambda x: x[1])

    def club_all_bounding_boxes_by_similar_y_coordinates_into_rows(self):
        self.rows = []
        half_of_mean_height = self.mean_height / 2
        current_row = [ self.bounding_boxes[0] ]
        for bounding_box in self.bounding_boxes[1:]:
            current_bounding_box_y = bounding_box[1]
            previous_bounding_box_y = current_row[-1][1]
            distance_between_bounding_boxes = abs(current_bounding_box_y - previous_bounding_box_y)
            if distance_between_bounding_boxes <= half_of_mean_height:
                current_row.append(bounding_box)
            else:
                self.rows.append(current_row)
                current_row = [ bounding_box ]
        self.rows.append(current_row)

    def sort_all_rows_by_x_coordinate(self):
        for row in self.rows:
            row.sort(key=lambda x: x[0])

    def crop_each_bounding_box_and_ocr(self):
        """Process bounding boxes, perform OCR, and build structured student records. Assumes first row is header (dates) and remaining rows are student data."""
        self.table = []  # Will store StudentRecord objects
        self.headers = []  # Will store header texts for columns
        image_number = 0
        
        # Process header row first to identify date columns
        if self.rows and len(self.rows) > 0:
            header_row = self.rows[0]
            self.headers = self._process_header_row(header_row)
            # Assuming first three columns are Student ID, Name, Status
            date_indices = list(range(3, len(self.headers)))
        else:
            print('No rows detected!')
            return
        
        # Process student rows, skipping header row
        for row in self.rows[1:]:
            cells = []
            for box in row:
                x, y, w, h = box
                pad = 3
                y0 = max(0, y - pad)
                x0 = max(0, x - pad)
                y1 = min(self.original_image.shape[0], y + h + pad)
                x1 = min(self.original_image.shape[1], x + w + pad)
                # Use the processed image with bounding boxes instead of original
                cropped_image = self.image_with_all_bounding_boxes[y0:y1, x0:x1]
                image_slice_path = os.path.join(self.ocr_slices_dir, f'img_{image_number}.jpg')
                cv2.imwrite(image_slice_path, cropped_image)
                
                # Use VLLM OCR with color detection
                ocr_text, is_red = self.vllm_ocr.recognize_text(image_slice_path)
                cleaned_text, attendance_status = self.clean_ocr_text(ocr_text, is_red)
                
                # Store both the text and its attendance status
                cells.append({
                    'text': cleaned_text,
                    'status': attendance_status
                })
                
                # Clean up temporary file
                os.remove(image_slice_path)
                image_number += 1
            # Expect at least 3 cells for ID, Name, Status
            if len(cells) >= 3:
                student_record = self._process_student_row(cells, date_indices)
                if student_record:
                    self.table.append(student_record)
        
        # Print a preview of the structured student records
        if self.table:
            print(f'\nProcessed {len(self.table)} student records')
            print('\nFirst few records:')
            for student in self.table[:3]:
                print(f'ID: {student.student_id}, Name: {student.student_name}, Status: {student.status}')
                print(f'Attendance: {student.attendance_dates}')

    def get_result_from_tersseract(self, image_path):
        """Legacy method kept for compatibility - now uses VLLM OCR"""
        return self.vllm_ocr.recognize_text(image_path)
        print(output)
        output = output.strip()

        os.chdir(r'C:\Users\Jash\OneDrive\Documents\AnamaloiesAttendanceSystem')
        return output

    def clean_ocr_text(self, text: str, is_red: bool = False) -> str:
        """
        Clean and validate OCR output, removing gibberish and normalizing valid text.
        Also handles the attendance marking logic based on text color.
        
        Returns: A tuple (cleaned_text, attendance_status)
        where attendance_status is 'Present' or 'Absent' based on the rules:
        - Any red text (except 'P') -> Absent
        - Everything else -> Present
        - Blank -> Absent
        """
        if text is None or text.strip() == "":
            return "", "Absent"
            
        # Remove line breaks and excess whitespace
        cleaned = " ".join(text.split())
        cleaned = cleaned.strip().upper()
        
        # Apply attendance rules
        if is_red:
            if cleaned == "P":
                return cleaned, "Present"
            else:
                return cleaned, "Absent"
        else:
            # Non-red text is always present unless blank
            if cleaned:
                return cleaned, "Present"
            else:
                return cleaned, "Absent"
        
        # Filter out gibberish/invalid text
        if len(cleaned) < 2:  # Too short
            return ""
        if not any(c.isalnum() for c in cleaned):  # No letters/numbers
            return ""
            
        # Normalize attendance marks
        if cleaned.upper() in ['P', 'PRESENT']:
            return 'P'
        if cleaned.upper() in ['A', 'ABSENT']:
            return 'A'
        if cleaned.upper() in ['AB']:
            return 'AB'
            
        return cleaned

    def generate_csv_file(self, output_path="Output_Reports/output.csv"):
        """Generate a structured CSV file with student attendance data.
        
        Args:
            output_path (str): Path where the CSV file should be saved
        """
        import csv
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if not self.table:
            return
            
        # Get all unique dates from all records
        all_dates = sorted(set(
            date for student in self.table 
            for date in student.attendance_dates.keys()
        ))
        
        # Prepare headers
        headers = ['Student ID', 'Student Name', 'Status'] + all_dates
        
        with open(output_path, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            
            # Write headers
            writer.writerow(headers)
            
            # Write student records
            for student in self.table:
                row = [
                    student.student_id,
                    student.student_name,
                    student.status
                ]
                # Add attendance for each date
                for date in all_dates:
                    row.append(student.attendance_dates.get(date, ''))
                writer.writerow(row)

    def get_table_as_dataframe(self):
        """Return the table as a pandas DataFrame if pandas is available.

        Columns will be numbered (0..N-1). If pandas is not installed, raises ImportError.
        """
        try:
            import pandas as pd
        except Exception as e:
            raise ImportError("pandas is required for DataFrame output") from e
        # normalize rows to same length by padding with empty strings
        max_cols = max((len(r) for r in self.table), default=0)
        normalized = [ r + [""]*(max_cols-len(r)) for r in self.table ]
        return pd.DataFrame(normalized)

    def _process_header_row(self, header_row):
        """Process the header row to identify date columns."""
        headers = []
        for box in header_row:
            x, y, w, h = box
            # expand box a little but clamp to image bounds
            pad = 3
            y0 = max(0, y - pad)
            x0 = max(0, x - pad)
            y1 = min(self.original_image.shape[0], y + h + pad)
            x1 = min(self.original_image.shape[1], x + w + pad)
            # Use the processed image with bounding boxes instead of original
            cropped = self.image_with_all_bounding_boxes[y0:y1, x0:x1]
            image_path = os.path.join(self.ocr_slices_dir, f"header_{len(headers)}.jpg")
            cv2.imwrite(image_path, cropped)
            text, _ = self.vllm_ocr.recognize_text(image_path)
            if os.path.exists(image_path):
                os.remove(image_path)
            headers.append(text.strip() if text else "")
        return headers

    def _process_student_row(self, cells, date_indices):
        """Convert a list of OCR extracted cells into a StudentRecord.
        Each cell is a dict with 'text' and 'status' keys.
        Expected:
            cells[0] -> Student ID
            cells[1] -> Student Name
            cells[2:] -> Attendance marks with their status
        """
        if len(cells) < 2:
            return None
        student = StudentRecord(
            student_id=cells[0],
            student_name=cells[1],
            status=cells[2],
            attendance_dates={}
        )
        
        # Map remaining cells to attendance dates based on header positions
        for idx in date_indices:
            if idx < len(cells):
                mark = cells[idx]
                if mark.upper() in ['P', 'PRESENT']:
                    attendance = 'P'
                elif mark.upper() in ['AB', 'ABSENT']:
                    attendance = 'AB'
                elif mark.upper() in ['A', 'ABSENT']:
                    attendance = 'A'
                else:
                    attendance = mark
                if idx < len(self.headers):
                    student.attendance_dates[self.headers[idx]] = attendance
        return student

    def store_process_image(self, file_name, image):
        path = "processed_images/ocr_table_tool/" + file_name
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, image)