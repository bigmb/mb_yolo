import numpy as np
import torch

__all__ = ['Utils']

class Utils:
    @staticmethod
    def iou(box1, box2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        IoU measures the overlap between two bounding boxes and is commonly used
        in object detection for evaluating detection accuracy and filtering
        overlapping predictions.
        
        Args:
            box1 (tuple/list): First bounding box in format (x, y, width, height)
            box2 (tuple/list): Second bounding box in format (x, y, width, height)
        
        Returns:
            float: IoU score between 0 and 1, where:
                  - 0 means no overlap
                  - 1 means complete overlap
        
        Example:
            >>> box1 = [100, 100, 50, 50]  # x, y, width, height
            >>> box2 = [125, 125, 50, 50]  # x, y, width, height
            >>> iou_score = Utils.iou(box1, box2)
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        w_intersection = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        h_intersection = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        
        area_intersection = w_intersection * h_intersection
        area_union = w1 * h1 + w2 * h2 - area_intersection
        
        return area_intersection / area_union if area_union > 0 else 0

    @staticmethod
    def non_max_suppression(boxes, scores, iou_threshold):
        """
        Perform Non-Maximum Suppression (NMS) on bounding boxes.
        
        NMS is used to eliminate redundant overlapping bounding boxes in object detection.
        It keeps the most confident detection and removes overlapping boxes that exceed
        the IoU threshold.
        
        Args:
            boxes (torch.Tensor): Tensor of bounding boxes (N x 4)
            scores (torch.Tensor): Confidence scores for each box (N)
            iou_threshold (float): IoU threshold for considering boxes as overlapping
        
        Returns:
            torch.Tensor: Indices of kept boxes after NMS
        
        Example:
            >>> boxes = torch.tensor([[100, 100, 50, 50], [120, 120, 50, 50]])
            >>> scores = torch.tensor([0.9, 0.8])
            >>> kept_indices = Utils.non_max_suppression(boxes, scores, 0.5)
        """
        indices = torch.argsort(scores, descending=True)
        keep = []
        while indices.size(0) > 0:
            keep.append(indices[0].item())
            if indices.size(0) == 1:
                break
            iou = Utils.iou(boxes[indices[0]], boxes[indices[1:]])
            indices = indices[1:][iou <= iou_threshold]
        return torch.tensor(keep)

    @staticmethod
    def xywh2xyxy(x):
        """
        Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2].
        
        This function converts bounding box coordinates from center-width format
        (YOLO format) to corner format (commonly used in computer vision).
        
        Args:
            x (torch.Tensor or numpy.ndarray): Bounding boxes in [x, y, w, h] format
                where x,y is the center point and w,h are width and height
        
        Returns:
            torch.Tensor or numpy.ndarray: Bounding boxes in [x1, y1, x2, y2] format
                where (x1,y1) is top-left corner and (x2,y2) is bottom-right corner
        
        Example:
            >>> boxes_xywh = torch.tensor([[100, 100, 50, 50]])
            >>> boxes_xyxy = Utils.xywh2xyxy(boxes_xywh)
        """
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1 = x - w/2
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1 = y - h/2
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2 = x + w/2
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2 = y + h/2
        return y

    @staticmethod
    def xyxy2xywh(x):
        """
        Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h].
        
        This function converts bounding box coordinates from corner format
        (commonly used in computer vision) to center-width format (YOLO format).
        
        Args:
            x (torch.Tensor or numpy.ndarray): Bounding boxes in [x1, y1, x2, y2] format
                where (x1,y1) is top-left corner and (x2,y2) is bottom-right corner
        
        Returns:
            torch.Tensor or numpy.ndarray: Bounding boxes in [x, y, w, h] format
                where x,y is the center point and w,h are width and height
        
        Example:
            >>> boxes_xyxy = torch.tensor([[75, 75, 125, 125]])
            >>> boxes_xywh = Utils.xyxy2xywh(boxes_xyxy)
        """
        if isinstance(x, torch.Tensor):
            y = torch.zeros_like(x)
        elif isinstance(x, np.ndarray):
            y = np.zeros_like(x)
        else:
            raise TypeError("Input must be a torch.Tensor or numpy.ndarray")

        # Compute center x, center y
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # Center x
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # Center y

        # Compute width and height
        y[:, 2] = x[:, 2] - x[:, 0]  # Width
        y[:, 3] = x[:, 3] - x[:, 1]  # Height
        
        return y
