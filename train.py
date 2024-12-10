import os
from ultralytics import YOLO

# Path to the configuration file
config_path = './config.yaml'

# Load a pre-trained YOLO model
model = YOLO('yolov8n.pt')  # Replace 'yolov8n.pt' with the correct model file or path

# Train the model
model.train(data=config_path, epochs=200, batch=32)
