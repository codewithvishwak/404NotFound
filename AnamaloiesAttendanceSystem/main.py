import OCRToTableTool_new as ottt
import TableExtractor as te
import TableLinesRemover as tlr
import cv2
import os

# Get the absolute path to the image
script_dir = os.path.dirname(os.path.abspath(__file__))
path_to_image = os.path.join(script_dir, "Images", "page_1.jpg")
table_extractor = te.TableExtractor(path_to_image)
perspective_corrected_image = table_extractor.execute()

#cv2.imshow("perspective_corrected_image", perspective_corrected_image)


lines_remover = tlr.TableLinesRemover(perspective_corrected_image)
image_without_lines = lines_remover.execute()
# #cv2.imshow("image_without_lines", image_without_lines)

ocr_tool = ottt.OcrToTableTool(image_without_lines, perspective_corrected_image)
# Execute OCR and get results
table_data = ocr_tool.execute()

# Print results
if table_data:
    print("\nProcessed student records successfully!")
    print("\nFirst few records for verification:")
    for record in table_data[:3]:
        print(f"Row: {record}")