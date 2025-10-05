# 404NotFound — Anomalies Attendance System

404NotFound is an attendance automation system that uses OCR (Optical Character Recognition) to extract and process attendance data from images or documents. It’s designed to simplify and streamline attendance record management by converting raw scans into structured CSVs.

---

## 📂 Repository Structure

```
404NotFound/
├── Images/                  ← Sample images (for demo/testing)
├── pycache/                 ← Python cache files
├── app.py                   ← Web interface / Flask app (if any)
├── attendance_processor.py   ← Main logic to orchestrate OCR & processing
├── vit_ocr.py               ← OCR utility for extracting text from images/docs
├── attendance_output.csv     ← Example output CSV with attendance records
├── poppler.zip              ← Poppler binaries for PDF-to-image conversion
├── requirements.txt         ← Python dependencies
└── README.md                ← This file
```

---

## 🚀 Features

- **OCR-based text extraction**  
  Converts scanned images, PDFs, or photos into text via Tesseract.

- **Attendance parsing & management**  
  Parses extracted text to derive attendance data like Moodle ID, student names, and attendance status.

- **CSV Output**  
  Saves structured attendance records in `.csv` format, ready for analysis or import.

- **PDF support (via Poppler)**  
  Uses Poppler tools to convert PDF pages into images for OCR processing.

- **Modular design**  
  Components like OCR logic and data parsing are separated for easier extension and maintenance.

---

## 🛠️ Setup & Installation

### Prerequisites

- **Python 3.7+**  
- **Tesseract OCR**  
  - **Ubuntu / Debian**:  
    ```bash
    sudo apt-get update
    sudo apt-get install tesseract-ocr
    ```
  - **Windows**:  
    Download and install from [Tesseract Windows builds](https://github.com/UB-Mannheim/tesseract/wiki) .

- **Poppler tools** (for PDF conversion)  
  - On Linux: `sudo apt-get install poppler-utils`  
  - On Windows: Use the `poppler.zip` included or download an appropriate build and reference the binaries.

### Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/codewithvishwak/404NotFound.git
   cd 404NotFound
   ```
   
2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS / Linux:
   source venv/bin/activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧰 Usage

### Command-line Usage

- **Run OCR only (for a single image)**:
   ```bash
   python vit_ocr.py path/to/image_or_pdf
   ```
   This outputs raw extracted text.

- **Process full attendance workflow**:
   ```bash
   python attendance_processor.py
   ```
   This script integrates OCR → parsing → output, and generates `attendance_output.csv`.

- **(Optional) Web Interface**:
   If `app.py` implements a web UI (Flask or similar), you can run:
   ```bash
   python app.py
---

## 📄 Output Format (attendance_output.csv)

The output CSV follows this structure:

| Moodle ID | Name                | 2025-10-05 | 2025-10-06 | 2025-10-07 | 2025-10-08 | 2025-10-09 |
|-----------|---------------------|------------|------------|------------|------------|------------|
| 23102092  | Vishwak Yellamalli   | Present    | Absent     | Present    | Absent     | Present    |
| 23102054  | Kush Vora           | Absent     | Present    | Present    | Absent     | Absent     |
| 23102041  | Maulik Zambad       | Present    | Present    | Absent     | Present    | Present    |
| 23102096  | Soham Walgude       | Absent     | Absent     | Present    | Present    | Absent     |
| 23102174  | Sairaj Walawalkar    | Present    | Present    | Present    | Absent     | Present    |

- **Moodle ID** → Unique identifier for each student
- **Name** → Student’s full name
- **lec_date** → Date of the lecture (column header)
- **Status** → "Present" or "Absent"

---

## 🧩 Extending & Customization

- **Improve OCR accuracy**:
  - Preprocessing: Image thresholding, denoising, skew correction.
  - Language models, custom character whitelists.

- **Support new formats**:
  - Handle Excel, DOCX, or scanned PDFs with multiple pages.

- **Integration pipelines**:
  - Automatically fetch images from email or Google Drive.
  - Sync output with Google Sheets or a database.

- **Error / anomaly detection**:
  - Flag suspicious attendance patterns (e.g., double check-ins, missing data).

---

## 🤝 Contributors

- [Soham Walgude](https://github.com/Soham8125)  
- [Maulik Zambad](https://github.com/Maulik-11)  
- [Kush Vora](https://github.com/Kushvora08)  

---

## 📜 License

This project is licensed under the MIT License.

---

## 📞 Contact

For inquiries, questions, or feedback, you can reach me at:

- Email: yellamallvishwak1@gmail.com
- LinkedIn: [Vishwak Yellamalli](https://www.linkedin.com/in/vishwakyellamalli)

