import streamlit as st
import pandas as pd
import os
from attendance_trainer import AttendanceOCRTrainer
from pdf2image import convert_from_bytes
import cv2
import numpy as np

class AttendanceProcessor:
    def __init__(self):
        # Initialize the OCR model
        self.trainer = AttendanceOCRTrainer()
        
        # Try to load trained model if it exists
        model_dir = "trained_attendance_model"
        if os.path.exists(model_dir):
            self.trainer.load_model(model_dir)
    
    def process_pdf(self, pdf_bytes):
        """Process a PDF file and extract attendance."""
        try:
            # Convert PDF to images
            images = convert_from_bytes(pdf_bytes)
            
            all_results = []
            for i, image in enumerate(images):
                # Save image temporarily
                temp_path = f"temp_page_{i}.jpg"
                image.save(temp_path)
                
                # Process the image
                results = self.process_image(temp_path)
                all_results.extend(results)
                
                # Clean up
                os.remove(temp_path)
            
            return self.create_attendance_report(all_results)
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return None
    
    def process_image(self, image_path):
        """Process a single image and extract attendance."""
        try:
            # Read and preprocess image
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Find table cells
            cells = self.extract_table_cells(gray)
            
            # Process each cell
            results = []
            for x, y, w, h in cells:
                # Extract cell
                cell = image[y:y+h, x:x+w]
                temp_path = f"temp_cell.jpg"
                cv2.imwrite(temp_path, cell)
                
                # Get OCR results with attendance status
                result = self.trainer.evaluate(temp_path)
                results.append(result)
                
                # Clean up
                os.remove(temp_path)
            
            return results
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return []
    
    def extract_table_cells(self, gray_image):
        """Extract table cells from grayscale image."""
        # Threshold the image
        _, binary = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)
        
        # Find table structure
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        
        # Detect horizontal and vertical lines
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)
        
        # Combine lines
        table_structure = cv2.add(horizontal, vertical)
        
        # Find contours (cells)
        contours, _ = cv2.findContours(table_structure, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and sort cells
        cells = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > 100:  # Filter small noise
                cells.append((x, y, w, h))
        
        # Sort cells by position (top-to-bottom, left-to-right)
        cells.sort(key=lambda x: (x[1], x[0]))
        return cells
    
    def create_attendance_report(self, results):
        """Create attendance report from OCR results."""
        # Group results into rows (assuming every two cells form a record)
        data = []
        for i in range(0, len(results), 2):
            if i + 1 < len(results):
                name = results[i]['text']
                status = results[i+1]['attendance']
                data.append({
                    'Name': name,
                    'Status': status,
                    'Is Red': results[i+1]['is_red']
                })
        
        return pd.DataFrame(data)

# Streamlit interface
def main():
    st.title("Smart Attendance Processor")
    st.write("Upload an attendance sheet (PDF or Image)")
    
    # Initialize processor
    processor = AttendanceProcessor()
    
    # File upload
    uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        try:
            if uploaded_file.type == "application/pdf":
                results = processor.process_pdf(uploaded_file.read())
            else:
                # Save image temporarily
                temp_path = "temp_upload.jpg"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())
                results = processor.process_image(temp_path)
                os.remove(temp_path)
                
            if results is not None:
                st.success("Processing complete!")
                st.dataframe(results)
                
                # Save results
                output_path = os.path.join("Output_Reports", "attendance_report.csv")
                results.to_csv(output_path, index=False)
                st.write(f"Results saved to: {output_path}")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()