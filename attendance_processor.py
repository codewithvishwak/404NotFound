from vit_ocr import VisionOCR
import os
import csv

def process_attendance_folder(folder_path, output_csv):
    # Initialize the OCR model
    ocr = VisionOCR()
    
    # Process all images in the folder
    attendance_records = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            print(f"Processing {filename}...")
            
            # Extract attendance data from image
            attendance_data = ocr.process_attendance_sheet(image_path)
            attendance_records.extend(attendance_data)
    
    # Write results to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['name', 'status'])
        writer.writeheader()
        writer.writerows(attendance_records)
    
    print(f"Attendance data has been saved to {output_csv}")

if __name__ == "__main__":
    # Set paths
    image_path = "Images/page_1.jpg"
    output_file = "attendance_output.csv"
    
    # Initialize the OCR model
    ocr = VisionOCR()
    
    # Process single image
    print(f"Processing {image_path}...")
    attendance_data = ocr.process_attendance_sheet(image_path)
    
    # Write results to CSV
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['name', 'status'])
        writer.writeheader()
        writer.writerows(attendance_data)
    
    print(f"Attendance data has been saved to {output_file}")