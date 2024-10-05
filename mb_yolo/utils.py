import numpy as np
import torch

__all__ = ['Utils']

class Utils:
    @staticmethod
    def iou(box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        w_intersection = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        h_intersection = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        
        area_intersection = w_intersection * h_intersection
        area_union = w1 * h1 + w2 * h2 - area_intersection
        
        return area_intersection / area_union if area_union > 0 else 0

    @staticmethod
    def non_max_suppression(boxes, scores, iou_threshold):
        """Perform Non-Maximum Suppression"""
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
        """Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]"""
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y