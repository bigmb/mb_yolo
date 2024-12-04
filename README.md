# MB-YOLO

MB-YOLO is a Python package that implements YOLO (You Only Look Once) object detection with a focus on weld defect detection. It supports multiple YOLO versions and includes integration with SAM2 (Segment Anything Model 2) for advanced segmentation capabilities.

## Features

- Support for multiple YOLO versions (YOLOv3, YOLOv5, YOLOv8, YOLO10, YOLO11)
- Configurable model parameters (size, function, classes)
- Integration with SAM2 for detailed segmentation
- Easy-to-use training and inference pipeline
- Flexible configuration system
- Specialized for weld defect detection

## Package Structure

The `mb_yolo` package consists of several key modules:

### train.py
Main training module that handles:
- Configuration loading from YAML files
- Model initialization
- Training pipeline setup using Ultralytics
- Automated result saving and logging

```python
from mb_yolo.train import train
train("config.yaml")  # Starts training with specified configuration
```

### models.py
Model management module that provides:
- Support for multiple YOLO versions (v3, v5, v8, v10, v11)
- Model size configurations (n, s, m, l, x)
- Multiple model functions:
  - Detection (default)
  - Segmentation
  - Pose estimation
  - Oriented bounding box (OBB)
  - Classification
- Automatic model creation and class number modification

```python
from mb_yolo.models import create_model
model = create_model('yolov8', 'n', 'detection', num_classes=3)
```

### utils.py
Utility functions for object detection operations:
- IoU (Intersection over Union) calculation between bounding boxes
- Non-Maximum Suppression (NMS) implementation for filtering overlapping detections
- Bounding box format conversion (xywh to xyxy)

```python
from mb_yolo.utils import Utils
# Calculate IoU between two boxes
iou = Utils.iou(box1, box2)
# Convert box format
xyxy = Utils.xywh2xyxy(boxes)
# Apply NMS
kept_indices = Utils.non_max_suppression(boxes, scores, iou_threshold=0.5)
```

## Installation

```
pip install mb_yolo -U
```

```bash
# Clone the repository
git clone https://github.com/bigmb/mb_yolo.git

# Install the package
pip install -e .
```

Requirements:
- Python >= 3.8
- mb_base
- ultralytics

## Usage

### Training

1. Prepare your data configuration in `model_configs/data.yaml`:
```yaml
path: /path/to/dataset
train: train/images
val: valid/images
test: test/images

names:
  0: 'Bad Weld'
  1: 'Good Weld'
  2: 'Defect'
```

2. Configure training parameters in `config.yaml`:
```yaml
model: 'yolov8'
model_size: 'n'
model_function: 'detection'
num_classes: 3
img_size: 640
batch_size: 16
epochs: 100
data_yaml: './model_configs/data.yaml'
project: 'yolo_project'
name: 'run1'
device: 'cpu'
n_cpu: 4
```

3. Start training:
```python
from mb_yolo.train import train
train("./config.yaml")
```

### Inference

```python
from ultralytics import YOLO

# Load model
model = YOLO('./yolo_project/run1/weights/best.onnx')

# Run inference
results = model('path/to/your/image.jpg')

# Process results
for result in results:
    print(result.boxes)  # Bounding boxes
    print(result.names)  # Class names
    result.show()  # Display results
```

### SAM2 Integration

The package includes integration with SAM2 for advanced segmentation:

```python
from mb_annotation.segsam2 import image_predictor

# Initialize predictor
predictor = image_predictor('./sam2_hiera_s.yaml',
                          'path/to/sam2_checkpoint.pt')

# Set image and predict
predictor.set_image('path/to/image.jpg')
predictor.predict_item(bbox=detection_box, gemini_bbox=False)
```

## Configuration

### Model Configuration
- Supported models: YOLOv3, YOLOv5, YOLOv8, YOLO10, YOLO11
- Model sizes: n, s, m, l, x
- Functions: detection, segmentation, pose, obb, classification (YOLOv8 and YOLO11 only)

### Training Configuration
- Batch size
- Number of epochs
- Image size
- Device (CPU/GPU)
- Number of workers
- Project and run names for organizing results
