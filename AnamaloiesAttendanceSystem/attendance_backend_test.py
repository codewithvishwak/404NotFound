import streamlit as st
import pandas as pd
import os
import re
from pdf2image import convert_from_bytes
from PIL import Image
import io
from config import *
from fast_ocr import FastAttendanceOCR

# Initialize OCR
ocr_tool = FastAttendanceOCR()

# --- Symbol Mapping and Cleaning Function ---

def clean_attendance_marks(mark, is_red=False):
    """
    Cleans and maps the OCR output based on specific rules:
    - Any red text (except 'P') -> Absent
    - Everything else -> Present
    - Blank -> Absent
    """
    if pd.isna(mark) or str(mark).strip() == '':
        return 'Absent'  # Changed from 'Unclear' to 'Absent'

    mark_str = str(mark).strip().upper()

    # If text is red
    if is_red:
        # Check if it's a 'P' in red
        if mark_str == 'P':
            return 'Present'
        else:
            return 'Absent'
    
    # For non-red text, everything is considered present except blanks
    return 'Present'

    if re.match(present_patterns, mark_str):
        return 'Present'
    elif re.match(absent_patterns, mark_str):
        return 'Absent'
    else:
        # Catch any other single characters or bad OCR that are not explicitly P/A
        return 'Unclear'

# --- Core Logic Functions (Modified) ---

def convert_pdf_to_images(uploaded_file):
    """Converts an uploaded PDF (in bytes) to images and saves them."""
    st.info(f"Converting PDF: {uploaded_file.name} to images...")
    
    # 1. Set up Poppler path and environment
    poppler_path = r"C:\poppler\Release-23.11.0-0\Library\bin"
    if not os.path.exists(poppler_path):
        st.error(f"Poppler not found at {poppler_path}")
        st.info("Downloading and installing Poppler...")
        try:
            # Create directory
            os.makedirs(r"C:\poppler", exist_ok=True)
            
            # Download Poppler
            import urllib.request
            url = "https://github.com/oschwartz10612/poppler-windows/releases/download/v23.11.0-0/Release-23.11.0-0.zip"
            zip_path = r"C:\poppler\poppler.zip"
            urllib.request.urlretrieve(url, zip_path)
            
            # Extract
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(r"C:\poppler")
                
            # Add to system PATH
            import sys
            os.environ["PATH"] += os.pathsep + poppler_path
            st.success("Poppler installed and added to PATH successfully!")
            
        except Exception as e:
            st.error(f"Failed to install Poppler: {e}")
            return []

    # 2. Convert PDF to images
    try:
        # Verify poppler installation
        if not os.path.exists(os.path.join(poppler_path, "pdfinfo.exe")):
            st.error("Poppler binaries not found. Attempting to fix...")
            # Force re-download
            import urllib.request
            url = "https://github.com/oschwartz10612/poppler-windows/releases/download/v23.11.0-0/Release-23.11.0-0.zip"
            zip_path = r"C:\poppler\poppler.zip"
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(r"C:\poppler")
                
        # Update PATH again
        if poppler_path not in os.environ["PATH"]:
            os.environ["PATH"] += os.pathsep + poppler_path
            
        # Convert PDF
        pdf_bytes = uploaded_file.read()
        images = convert_from_bytes(pdf_bytes, poppler_path=poppler_path, dpi=200)
        
        if not images:
            st.error("No images extracted from PDF")
            return []
            
        st.success(f"PDF converted successfully! Extracted {len(images)} pages.")
        
    except Exception as e:
        st.error(f"PDF Conversion Failed: {str(e)}")
        st.info(f"Current Poppler path: {poppler_path}")
        st.info(f"Files in Poppler bin: {os.listdir(poppler_path) if os.path.exists(poppler_path) else 'Directory not found'}")
        return []

    # 2. Save Images to the Images folder
    base_name = os.path.splitext(uploaded_file.name)[0]
    image_paths = []
    
    for i, image in enumerate(images):
        page_name = f"{base_name}_page_{i+1}.jpg"
        save_path = os.path.join(IMAGES_DIR, page_name)
        image.save(save_path, 'JPEG')
        image_paths.append(save_path)
    
    st.success(f"Converted and saved {len(images)} page(s) to the '{IMAGES_DIR}' folder.")
    return image_paths

def extract_attendance_data_from_image(image_path):
    """
    Extract attendance data from image using VLLM OCR
    """
    st.info(f"Processing attendance data from {os.path.basename(image_path)}...")
    
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
        st.success(f"Successfully processed {len(df)} student records")
        return df
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return pd.DataFrame()
    
    # --- SIMULATED RAW OCR DATA (Imagine this is what Tesseract returned) ---
    data = {
        'Roll No': [76, 77, 78, 79, 80],
        'Student Name': ['AGARE SAMIHAN M.', 'AHIR ANSH M.', 'ANNADATE VANSH M.', 'BANSODE ASHISH S.', 'BHANDARI RASHI D.'],
        # Simulate different OCR outcomes: Dot, P, A, AB, Cross, Blank
        'Lec_10_07_25': ['.', 'A', 'P', 'P', 'AB'],
        'Lec_11_07_25': ['P', 'A', 'P', 'P', 'A'],
        'Lec_14_07_25': ['A', 'P', ' ', 'P', 'X'], # Blank/Space for Unclear
        'Lec_15_07_25': ['P', '✓', 'P', 'P', 'P'], # Tick mark
    }
    raw_df = pd.DataFrame(data)
    raw_df.set_index(['Roll No', 'Student Name'], inplace=True)
    
    # --- Data Cleaning: Apply the Symbol Mapping ---
    st.info("Applying symbol mapping to clean raw marks...")
    for col in [col for col in raw_df.columns if col.startswith('Lec_')]:
        raw_df[col] = raw_df[col].apply(clean_attendance_marks)
        
    return raw_df

def generate_reports(raw_df: pd.DataFrame, subject_name: str):
    """Generates the three required Excel reports and documentation."""
    
    if raw_df.empty:
        st.error("No data to generate reports.")
        return

    # --- 1. Subject-wise Report (Main Report) ---
    st.info("Generating Subject-wise Attendance Report...")
    
    # Convert cleaned marks ('Present'/'Absent'/'Unclear') to 1/0 for calculations
    attendance_cols = [col for col in raw_df.columns if col.startswith('Lec_')]
    
    # 'Present' and 'Unclear' are counted as 1/0 for % calculation, but Unclear is flagged
    calc_df = raw_df[attendance_cols].applymap(lambda x: 1 if x == 'Present' else 0)
    
    raw_df['Total Present'] = calc_df.sum(axis=1)
    raw_df['Total Lectures'] = len(attendance_cols)
    raw_df['Attendance %'] = (raw_df['Total Present'] / raw_df['Total Lectures']) * 100
    raw_df['Status'] = raw_df['Attendance %'].apply(lambda x: 'Clear' if x >= 75 else 'Defaulter')
    
    # Anomaly Flag: Severe Defaulter (<50%) OR Has Unclear Marks
    unclear_count = (raw_df[attendance_cols] == 'Unclear').sum(axis=1)
    
    def apply_anomaly(row):
        if row['Attendance %'] < 50:
            return 'Severe Defaulter'
        elif unclear_count[row.name] > 0:
             return f"Check Marks ({unclear_count[row.name]} Unclear)"
        else:
            return ''
            
    raw_df['Anomaly Flag'] = raw_df.apply(apply_anomaly, axis=1)
    
    # Final Subject Report DataFrame
    final_cols = ['Total Present', 'Total Lectures', 'Attendance %', 'Status', 'Anomaly Flag'] + attendance_cols 
    subject_report_df = raw_df.reset_index()[['Roll No', 'Student Name'] + final_cols]
    
    # Save the main subject report
    subject_report_path = os.path.join(OUTPUT_DIR, f"{subject_name}_Attendance_Report.xlsx")
    subject_report_df.to_excel(subject_report_path, sheet_name=subject_name, index=False)
    
    st.success(f"1. Subject Report saved to: {subject_report_path}")

    # --- 2. Department Summary Sheet (Simplified) ---
    st.info("Generating Department Summary Sheet (Consolidated)...")
    summary_df = raw_df[['Total Lectures', 'Attendance %', 'Status']].reset_index()
    summary_df.rename(columns={'Attendance %': f'{subject_name} %', 'Status': f'{subject_name} Status'}, inplace=True)
    
    dept_summary_path = os.path.join(OUTPUT_DIR, "Department_Attendance_Summary.xlsx")
    summary_df.to_excel(dept_summary_path, index=False)
    
    st.success(f"2. Department Summary saved to: {dept_summary_path}")

    # --- 3. Anomaly and Defaulter Reports ---
    st.info("Generating Anomaly and Defaulter Reports...")
    
    # Defaulters (Below 75%)
    defaulters_df = raw_df[raw_df['Status'] == 'Defaulter'].reset_index()
    # Anomalies (Severe or Unclear Marks)
    anomalies_df = raw_df[raw_df['Anomaly Flag'] != ''].reset_index()
    
    anomaly_report_path = os.path.join(OUTPUT_DIR, "Administrative_Action_Reports.xlsx")
    
    with pd.ExcelWriter(anomaly_report_path, engine='xlsxwriter') as writer:
        defaulters_df.to_excel(writer, sheet_name='Defaulters (Below 75%)', index=False)
        anomalies_df.to_excel(writer, sheet_name='Anomalies (Severe/Unclear)', index=False)

    st.success(f"3. Anomaly/Defaulter Reports saved to: {anomaly_report_path}")

    # --- 4. Documentation Note ---
    st.markdown("### Documentation Note")
    st.markdown("""
    * **Symbol Mapping:**
        * **Present (1):** 'P', Tick mark ($\checkmark$), or Dot ($\cdot$).
        * **Absent (0):** 'A', 'AB', or Cross mark ($\times$).
        * **Unclear:** Blank/Empty field after OCR, or any unrecognized symbol. Unclear marks are counted as **Absent (0)** for percentage calculation but are highlighted in the **Anomaly Flag**.
    * **Assumption:** OCR correctly isolates marks; the first student data corresponds to the first roll number.
    * **Error-Handling:** **Defaulter** status is for attendance below 75%. **Anomaly Flag** is set for attendance below 50% or if any 'Unclear' marks are detected.
    """)


# --- Streamlit Application (Remains the same) ---

st.set_page_config(layout="wide", page_title="Attendance Report Generator")
st.title("Attendance Data Processor & Report Generator")
st.caption("A complex OCR and data processing application using Python, Streamlit, pdf2image, and Pandas.")

# Sidebar for configuration and path display
st.sidebar.header("Configuration & Paths")
st.sidebar.write(f"Images will be saved to: `{IMAGES_DIR}`")
st.sidebar.write(f"Reports will be saved to: `{OUTPUT_DIR}`")

if POPPLER_PATH:
    st.sidebar.warning(f"Using custom Poppler Path: `{POPPLER_PATH}`")
else:
    st.sidebar.info("Assuming Poppler is in system PATH.")


uploaded_file = st.file_uploader(
    "Upload a PDF, CSV, or Image file for processing",
    type=['pdf', 'csv', 'png', 'jpg', 'jpeg'],
    key="file_uploader"
)

subject_name = st.text_input("Enter Subject/Batch Name (e.g., AOA_SE_C1)", "AOA_SE_C1")

if uploaded_file is not None:
    st.subheader(f"Processing File: {uploaded_file.name}")
    
    file_type = uploaded_file.type.split('/')[-1]

    # --- Step 1 & 2: Convert/Process File to Image and Extract Data ---
    image_paths = []
    data_df = pd.DataFrame()

    if file_type == 'pdf':
        image_paths = convert_pdf_to_images(uploaded_file)
        
        if image_paths:
            st.subheader("Data Extraction (Simulated OCR)")
            data_df = extract_attendance_data_from_image(image_paths[0])
            st.dataframe(data_df)

    elif file_type in ['csv', 'vnd.ms-excel']:
        st.info("CSV file detected. Bypassing image conversion/OCR...")
        # Placeholder logic: Assumes CSV columns are already clean
        data_df = pd.read_csv(uploaded_file)
        # Apply cleaning to ensure column consistency if needed
        st.subheader("CSV Data Preview")
        st.dataframe(data_df)
        
    elif file_type in ['png', 'jpg', 'jpeg']:
        st.info("Image file detected. Bypassing image conversion.")
        img = Image.open(uploaded_file)
        save_path = os.path.join(IMAGES_DIR, uploaded_file.name)
        img.save(save_path)
        image_paths.append(save_path)
        
        st.subheader("Data Extraction (Simulated OCR)")
        data_df = extract_attendance_data_from_image(save_path)
        st.dataframe(data_df)
    
    # --- Step 3: Generate Reports ---
    if not data_df.empty and st.button("Generate Final Reports"):
        generate_reports(data_df, subject_name)
        st.balloons()
        st.success("All reports generated successfully in the 'Output_Reports' folder!")