from ultralytics import YOLO

__all__ = ["create_model"]

def create_model(version, model_size, model_function, num_classes):
    if version not in ['yolov3', 'yolov5', 'yolov8','yolo10','yolo11']:
        raise ValueError(f"Unsupported YOLO version: {version}")
    
    model_size = model_size.lower()

    if model_function not in ['detection', 'segmentation', 'pose', 'obb', 'classifcation']:
        raise ValueError(f"Unsupported model function: {model_function}")
    if model_function == 'classifcation':
        model_function = 'cls'
    if model_function == 'segmentation':
        model_function = 'seg'
    if model_function == 'pose':
        model_function = 'pose'
    if model_function == 'obb':
        model_function = 'obb'
        
    if model_function == 'detection':
        model_name = f"{version}{model_size}.pt"
    else:
        model_name = f"{version}{model_size}-{model_function}"
    
    # Create a new YOLO model
    model = YOLO(f"{model_name}.yaml")
    
    # Modify the model for the specific number of classes
    model.model.nc = num_classes
    
    return model

# Note: The actual model creation and modification will be handled by Ultralytics
# This function now serves as a wrapper to ensure compatibility with our existing code