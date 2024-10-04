from ultralytics import YOLO

def create_model(version, num_classes):
    if version not in ['yolov3', 'yolov5', 'yolov8']:
        raise ValueError(f"Unsupported YOLO version: {version}")
    
    # Create a new YOLO model
    model = YOLO(f"{version}.yaml")
    
    # Modify the model for the specific number of classes
    model.model.nc = num_classes
    
    return model

# Note: The actual model creation and modification will be handled by Ultralytics
# This function now serves as a wrapper to ensure compatibility with our existing code