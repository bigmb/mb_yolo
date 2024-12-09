import yaml
from mb_yolo.models import create_model

__all__ = ["train"]

def train(config_path):
    """
    Train a YOLO model using the specified configuration.
    
    This function handles the complete training pipeline including:
    1. Loading configuration from YAML file
    2. Initializing the appropriate YOLO model
    3. Training using Ultralytics framework
    4. Saving results and model weights
    
    Args:
        config_path (str): Path to the YAML configuration file containing training parameters
                          including model type, size, batch size, epochs, etc.
    
    Returns:
        results: Training results from Ultralytics trainer
    
    Example:
        >>> train("config.yaml")
    """
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Initialize model
    model = create_model(config['model'],config['model_size'],config['model_function'] ,config['num_classes'])
    
    # Train the model using Ultralytics
    results = model.train(
        data=config['data_yaml'],
        epochs=config['epochs'],
        imgsz=config['img_size'],
        batch=config['batch_size'],
        device=config['device'],
        workers=config['n_cpu'],
        project=config['project'],
        name=config['name']
    )
    
    # The model is automatically saved by Ultralytics after training
    print(f"Training completed. Results: {results}")

