import os
from attendance_trainer import AttendanceOCRTrainer

def train_model():
    # Initialize trainer
    trainer = AttendanceOCRTrainer()
    
    # Training examples with expected results
    training_data = [
        {"text": "P", "is_red": False, "expected": "PRESENT"},
        {"text": "P", "is_red": True, "expected": "PRESENT"},
        {"text": "A", "is_red": True, "expected": "ABSENT"},
        {"text": "", "is_red": False, "expected": "UNCLEAR"},
        {"text": "X", "is_red": True, "expected": "ABSENT"},
        {"text": "✓", "is_red": False, "expected": "PRESENT"},
    ]
    
    # Training settings
    num_epochs = 10
    
    print("Starting training...")
    for epoch in range(num_epochs):
        total_loss = 0
        for item in training_data:
            loss = trainer.train_on_example(
                f"training_data/{item['text']}_{item['is_red']}.jpg",
                item['expected']
            )
            if loss:
                total_loss += loss
        
        avg_loss = total_loss / len(training_data)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    # Save the trained model
    trainer.save_model("trained_attendance_model")
    print("Training complete! Model saved to 'trained_attendance_model' directory")

if __name__ == "__main__":
    train_model()