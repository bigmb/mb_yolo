import yaml
from mb_yolo.models import create_model

__all__ = ["train"]

def train(config_path):
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Initialize model
    model = create_model(config['model'], config['num_classes'])
    
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

if __name__ == "__main__":
    train("config.yaml")