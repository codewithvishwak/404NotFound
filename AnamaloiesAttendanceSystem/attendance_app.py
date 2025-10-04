import streamlit as st
import pandas as pd
import os
import re
from pdf2image import convert_from_bytes
from PIL import Image
import io
from config import *
from fast_ocr import FastAttendanceOCR
import openpyxl

# Initialize OCR
ocr_tool = FastAttendanceOCR()

# --- Symbol Mapping and Cleaning Function ---
# Define patterns for attendance marks
present_patterns = r'^[P\.\u2713]$'  # P, dot, or tick mark
absent_patterns = r'^[AX\u00d7]$|^AB$'  # A, X, cross mark, or AB

def clean_attendance_marks(mark, is_red=False):
    """
    Cleans and maps the OCR output based on specific rules:
    - Any red text (except 'P') -> Absent
    - P, dot(.), or tick(✓) -> Present
    - A, X, AB -> Absent
    - Blank -> Absent
    - Other non-red text -> Present
    """
    if pd.isna(mark) or str(mark).strip() == '':
        return 'Absent'

    mark_str = str(mark).strip().upper()

    # If text is red
    if is_red:
        return 'Present' if mark_str == 'P' else 'Absent'
    
    # For non-red text
    if re.match(present_patterns, mark_str):
        return 'Present'
    elif re.match(absent_patterns, mark_str):
        return 'Absent'
    else:
        return 'Present'  # Default to Present for non-red text

def convert_pdf_to_images(uploaded_file):
    """Converts an uploaded PDF (in bytes) to images and saves them."""
    st.info(f"Converting PDF: {uploaded_file.name} to images...")
    
    # Set up Poppler path
    poppler_path = r"C:\poppler\Release-23.11.0-0\Library\bin"
    
    try:
        # Convert PDF to images
        pdf_bytes = uploaded_file.read()
        images = convert_from_bytes(pdf_bytes, poppler_path=poppler_path, dpi=200)
        
        if not images:
            st.error("No images extracted from PDF")
            return []
        
        # Save Images to the Images folder
        base_name = os.path.splitext(uploaded_file.name)[0]
        image_paths = []
        
        for i, image in enumerate(images):
            page_name = f"{base_name}_page_{i+1}.jpg"
            save_path = os.path.join(IMAGES_DIR, page_name)
            image.save(save_path, 'JPEG')
            image_paths.append(save_path)
        
        st.success(f"Converted and saved {len(images)} page(s)")
        return image_paths
            
    except Exception as e:
        st.error(f"PDF Conversion Failed: {str(e)}")
        return []

def extract_attendance_data_from_image(image_path):
    """Extract attendance data from image using VLLM OCR"""
    try:
        # Process image using FastAttendanceOCR
        results = ocr_tool.process_image(image_path)
        
        # Convert results to DataFrame
        data = {
            'Roll No': [],
            'Student Name': [],
            'Status': [],
        }
        
        # Group results by pairs (assuming every two cells form a student record)
        for i in range(0, len(results), 2):
            if i + 1 < len(results):
                data['Roll No'].append(results[i]['text'])
                data['Student Name'].append(results[i + 1]['text'])
                data['Status'].append(results[i + 1]['status'])
        
        df = pd.DataFrame(data)
        return df
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return pd.DataFrame()

def generate_reports(raw_df: pd.DataFrame, subject_name: str):
    """Generates attendance reports."""
    if raw_df.empty:
        st.error("No data to generate reports.")
        return

    try:
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Process the data
        # Convert status to format needed for report
        status_map = {'Present': 1, 'Absent': 0}
        attendance_df = raw_df.copy()
        attendance_df['Attendance Value'] = attendance_df['Status'].map(status_map)

        # Calculate attendance statistics
        attendance_df['Total Classes'] = 1  # Each row represents one class
        attendance_stats = attendance_df.groupby(['Roll No', 'Student Name']).agg({
            'Attendance Value': 'sum',
            'Total Classes': 'sum'
        }).reset_index()
        
        attendance_stats['Attendance %'] = (attendance_stats['Attendance Value'] / attendance_stats['Total Classes'] * 100).round(2)
        attendance_stats['Status'] = attendance_stats['Attendance %'].apply(lambda x: 'Clear' if x >= 75 else 'Defaulter')

        # Generate Reports
        # 1. Main Attendance Report
        main_report_path = os.path.join(OUTPUT_DIR, f"{subject_name}_Attendance_Report.xlsx")
        attendance_stats.to_excel(main_report_path, index=False)

        # 2. Defaulters Report
        defaulters = attendance_stats[attendance_stats['Status'] == 'Defaulter']
        defaulters_path = os.path.join(OUTPUT_DIR, f"{subject_name}_Defaulters.xlsx")
        if not defaulters.empty:
            defaulters.to_excel(defaulters_path, index=False)

        # 3. Daily Attendance Record
        daily_record = attendance_df.pivot_table(
            index=['Roll No', 'Student Name'],
            columns=None,  # You might want to add a date column in your data
            values='Attendance Value',
            aggfunc='first'
        ).reset_index()
        daily_record_path = os.path.join(OUTPUT_DIR, f"{subject_name}_Daily_Record.xlsx")
        daily_record.to_excel(daily_record_path, index=False)

        return {
            "main_report": main_report_path,
            "defaulters_report": defaulters_path if not defaulters.empty else None,
            "daily_record": daily_record_path
        }

    except Exception as e:
        st.error(f"Error generating reports: {str(e)}")
        return None

# --- Streamlit Application Interface ---
st.set_page_config(layout="wide", page_title="Smart Attendance System")
st.title("Smart Attendance System")
st.caption("Upload attendance sheets and generate reports automatically")

# Setup sidebar
with st.sidebar:
    st.header("Configuration")
    subject_name = st.text_input("Subject Name", "AOA_SE_C1")
    st.markdown("---")
    st.markdown("### System Information")
    st.info(f"Images Directory: {IMAGES_DIR}")
    st.info(f"Output Directory: {OUTPUT_DIR}")

# Main interface
uploaded_file = st.file_uploader(
    "Upload attendance sheet (PDF format)",
    type=['pdf'],
    help="Upload the scanned attendance sheet in PDF format"
)

if uploaded_file:
    st.subheader("Processing Attendance Sheet")
    
    # Process the uploaded file
    with st.spinner("Converting PDF and extracting data..."):
        # Convert PDF to images
        image_paths = convert_pdf_to_images(uploaded_file)
        
        if image_paths:
            # Extract data from images
            data_df = extract_attendance_data_from_image(image_paths[0])
            
            if not data_df.empty:
                # Show extracted data
                st.subheader("Extracted Attendance Data")
                st.dataframe(data_df)
                
                # Generate reports automatically
                with st.spinner("Generating attendance reports..."):
                    report_paths = generate_reports(data_df, subject_name)
                    
                    if report_paths:
                        st.success("Reports generated successfully!")
                        
                        # Display download buttons
                        st.subheader("Download Reports")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            with open(report_paths["main_report"], "rb") as f:
                                st.download_button(
                                    "Download Main Report",
                                    f,
                                    file_name=f"{subject_name}_Attendance_Report.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                        
                        if report_paths.get("defaulters_report"):
                            with col2:
                                with open(report_paths["defaulters_report"], "rb") as f:
                                    st.download_button(
                                        "Download Defaulters List",
                                        f,
                                        file_name=f"{subject_name}_Defaulters.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                        
                        with col3:
                            with open(report_paths["daily_record"], "rb") as f:
                                st.download_button(
                                    "Download Daily Record",
                                    f,
                                    file_name=f"{subject_name}_Daily_Record.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                        
                        st.balloons()
            else:
                st.error("No attendance data could be extracted from the image. Please check the PDF quality.")
else:
    st.info("Please upload a PDF file to begin processing.")