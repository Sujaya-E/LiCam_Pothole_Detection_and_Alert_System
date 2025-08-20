from ultralytics import YOLO

# Load the YOLOv8n model (nano version, smallest and fastest)
model = YOLO('yolov10n.pt')

# Train the model
model.train(
    data=r"/content/drive/MyDrive/Pothole-codes/combined-dataset/data1.yaml",  # Path to your dataset YAML file
    epochs=120,                         # Number of training epochs
    imgsz=640,                         # Input image size (default is 640)
    batch=16,                          # Batch size
    workers=2,                         # Number of workers for data loading
    optimizer='Adam',                  # Optimizer (Adam or SGD)
    project='pothole_detection',       # Project name where results are saved
    name='yolov8n_pothole',            # Name of the specific run
    # device='cuda',                    # Use GPU (set to 'cpu' for CPU)
    augment=True                       # Enable image augmentation during training
)

# Save the final model weights after training completes
model.save('/content/drive/MyDrive/Pothole-codes/yolov10n_train.pt')

# Optional: Evaluate model performance on validation/test set
metrics = model.val()
print(f"Validation metrics: {metrics}")