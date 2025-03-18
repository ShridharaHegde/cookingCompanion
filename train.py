from ultralytics import YOLO
import os

output_path = '/Users/shridharahegde/Desktop/ML Projects/Image to recepie/'

model = YOLO('yolov8n.pt')

# Train the model
results = model.train(
    data='/Users/shridharahegde/Desktop/ML Projects/Image to recepie/Food Ingredients Dataset v4 Oct 6/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='custom_model.pt'
)

final_model_path = os.path.join(output_path, 'custom_model.pt')
model.save(final_model_path)

print(f"Model saved to {final_model_path}")

# Optionally, you can validate the model
model.val()


