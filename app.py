from flask import Flask, render_template, request, redirect, url_for
import os
import sys
from werkzeug.utils import secure_filename

# Add the AnamaloiesAttendanceSystem directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
attendance_system_dir = os.path.join(current_dir, 'AnamaloiesAttendanceSystem')
sys.path.append(attendance_system_dir)

# Import the modules directly
import OCRToTableTool
import TableExtractor
import TableLinesRemover

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Images'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image
        try:
            # Initialize your tools
            table_lines_remover = TableLinesRemover.TableLinesRemover()
            table_extractor = TableExtractor.TableExtractor()
            ocr_tool = OCRToTableTool.OCRToTableTool()
            
            # Process the image using your existing tools
            processed_image = table_lines_remover.remove_lines(filepath)
            table_data = table_extractor.extract_table(processed_image)
            final_data = ocr_tool.process_image(table_data)
            
            return render_template('result.html', data=final_data)
        except Exception as e:
            return render_template('error.html', error=str(e))
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)