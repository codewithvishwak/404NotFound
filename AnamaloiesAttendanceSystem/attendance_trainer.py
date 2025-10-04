import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import cv2
import numpy as np
from PIL import Image
import os

class AttendanceOCRTrainer:
    def __init__(self):
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        self.image_processor = ViTImageProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/trocr-base-handwritten")
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Set up training parameters
        self.model.train()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)

    def preprocess_image(self, image_path):
        """Load and preprocess the image."""
        # Open and convert to RGB
        image = Image.open(image_path).convert('RGB')
        
        # Get color information
        img_array = np.array(image)
        is_red = self.detect_red_text(img_array)
        
        # Process image for the model
        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values
        return pixel_values.to(self.device), is_red

    def detect_red_text(self, img_array):
        """Detect if text is in red."""
        if len(img_array.shape) == 3:
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            # Red color range in HSV
            lower_red = np.array([0,50,50])
            upper_red = np.array([10,255,255])
            red_mask = cv2.inRange(hsv, lower_red, upper_red)
            return np.sum(red_mask) > (img_array.shape[0] * img_array.shape[1] * 20)
        return False

    def process_attendance_mark(self, text, is_red):
        """Apply attendance rules:
        - P means present
        - Blank means unclear
        - Red text (except P) means absent
        """
        text = text.strip().upper() if text else ""
        
        if not text:
            return "UNCLEAR"
        elif is_red:
            return "PRESENT" if text == "P" else "ABSENT"
        return "PRESENT" if text == "P" else "UNCLEAR"

    def train_on_example(self, image_path, target_text):
        """Train model on a single example."""
        try:
            # Preprocess image
            pixel_values, is_red = self.preprocess_image(image_path)
            
            # Encode target text
            target_encoding = self.tokenizer(target_text, 
                                           padding="max_length", 
                                           max_length=64, 
                                           return_tensors="pt")
            labels = target_encoding.input_ids.to(self.device)
            
            # Forward pass
            outputs = self.model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            return float(loss.item())
        except Exception as e:
            print(f"Error training on {image_path}: {str(e)}")
            return None

    def save_model(self, output_dir):
        """Save the fine-tuned model."""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.image_processor.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

    def load_model(self, model_dir):
        """Load a fine-tuned model."""
        self.model = VisionEncoderDecoderModel.from_pretrained(model_dir)
        self.image_processor = ViTImageProcessor.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model.to(self.device)
        print(f"Model loaded from {model_dir}")

    def evaluate(self, image_path):
        """Evaluate an image and return processed text with attendance status."""
        pixel_values, is_red = self.preprocess_image(image_path)
        
        # Generate text
        generated_ids = self.model.generate(pixel_values, max_length=64)
        text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Process attendance
        attendance = self.process_attendance_mark(text, is_red)
        
        return {
            'text': text.strip(),
            'is_red': is_red,
            'attendance': attendance
        }