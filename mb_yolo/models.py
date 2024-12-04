from ultralytics import YOLO

__all__ = ["create_model"]

def create_model(version, model_size, model_function, num_classes):
    """
    Create and initialize a YOLO model with specified parameters.
    
    This function creates a YOLO model with the specified version, size, and function.
    It supports multiple YOLO versions and various model functions like detection,
    segmentation, pose estimation, etc.
    
    Args:
        version (str): YOLO version to use. Options: 'yolov3', 'yolov5', 'yolov8', 'yolo10', 'yolo11'
        model_size (str): Size/complexity of the model. Options: 'n', 's', 'm', 'l', 'x'
        model_function (str): Function of the model. Options:
                            - 'detection' (default object detection)
                            - 'segmentation' (instance segmentation)
                            - 'pose' (pose estimation)
                            - 'obb' (oriented bounding box)
                            - 'classification' (image classification)
        num_classes (int): Number of classes the model should detect/classify
    
    Returns:
        YOLO: Initialized YOLO model with specified parameters
    
    Raises:
        ValueError: If unsupported YOLO version or model function is specified
    
    Example:
        >>> model = create_model('yolov8', 'n', 'detection', 3)
    """
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
        model_name = f"{version}{model_size}-{model_function}.pt"
    
    # Create a new YOLO model
    model = YOLO(f"{model_name}")
    
    # Modify the model for the specific number of classes
    model.model.nc = num_classes
    
    return model
