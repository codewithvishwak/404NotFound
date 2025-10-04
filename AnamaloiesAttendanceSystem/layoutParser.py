import layoutparser as lp
import cv2
import numpy as np
import os

image_path = "Images/page_1.jpg"
image = cv2.imread(image_path)

model = lp.models.Detectron2LayoutModel(
    config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5]
)

layout = model.detect(image)

for block in layout:
    if block.type == 'Table':
        x1, y1, x2, y2 = map(int, block.coordinates)
        table_image = image[y1:y2, x1:x2]
        cv2.imwrite("extracted_table.jpg", table_image)
        break

