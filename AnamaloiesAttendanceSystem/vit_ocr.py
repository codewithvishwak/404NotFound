from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import numpy as np

class VLLMOCRTool:
    def __init__(self):
        # Load pre-trained model and processors
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        self.image_processor = ViTImageProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/trocr-base-handwritten")
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def preprocess_image(self, image_path):
        """Load and preprocess the image."""
        # Open and convert to RGB (in case it's grayscale)
        image = Image.open(image_path).convert('RGB')
        
        # Process image for the model
        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values
        return pixel_values.to(self.device)

    def recognize_text(self, image_path):
        """Perform OCR on a single image and detect if text is red."""
        try:
            # Load image for color analysis
            img = Image.open(image_path)
            img_array = np.array(img)
            
            # Check if the image is predominantly red
            # This assumes the text color is the dominant color in the cell
            if len(img_array.shape) == 3:  # Color image
                red_channel = img_array[:,:,0]
                green_channel = img_array[:,:,1]
                blue_channel = img_array[:,:,2]
                
                # Calculate average color values
                avg_red = np.mean(red_channel)
                avg_green = np.mean(green_channel)
                avg_blue = np.mean(blue_channel)
                
                # Text is considered red if red channel is significantly higher
                is_red = avg_red > (1.5 * avg_green) and avg_red > (1.5 * avg_blue)
            else:
                is_red = False
            
            # Preprocess image for OCR
            pixel_values = self.preprocess_image(image_path)
            
            # Generate text
            generated_ids = self.model.generate(pixel_values, max_length=64)
            generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return generated_text, is_red
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return "", False

    def process_table_image(self, table_image_path):
        """
        Process a table image and extract text content.
        This method should be customized based on your specific table structure.
        """
        try:
            # The table processing logic would go here
            # For now, it just performs basic text recognition
            text = self.recognize_text(table_image_path)
            return text
        except Exception as e:
            print(f"Error processing table {table_image_path}: {str(e)}")
            return ""

    def process_batch(self, image_paths):
        """Process multiple images in batch."""
        results = []
        for path in image_paths:
            result = self.recognize_text(path)
            results.append(result)
        return results