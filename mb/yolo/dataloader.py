import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os

__all__ = ['YOLODataset', 'create_dataloader']

class YOLODataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()
        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):
        # Load image
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = cv2.imread(img_path)
        
        # Load labels
        label_path = self.label_files[index % len(self.img_files)].rstrip()
        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
        
        # Apply augmentations
        if self.augment:
            # Add augmentation logic here (e.g., random flip, color jitter)
            pass
        
        # Resize and pad image
        h, w, _ = img.shape
        img, pad = self.pad_to_square(img, 127.5)
        _, padded_h, padded_w = img.shape
        
        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Normalize
        img = img[:,:,::-1].transpose((2, 0, 1)).copy()
        img = torch.from_numpy(img).float().div(255.0)
        
        # Handle labels
        if labels is not None:
            # Extract coordinates for unpadded + unscaled image
            x1 = w * (labels[:, 1] - labels[:, 3]/2)
            y1 = h * (labels[:, 2] - labels[:, 4]/2)
            x2 = w * (labels[:, 1] + labels[:, 3]/2)
            y2 = h * (labels[:, 2] + labels[:, 4]/2)
            
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            
            # Calculate ratios from coordinates
            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= w / padded_w
            labels[:, 4] *= h / padded_h
            
            # Fill matrix
            filled_labels = np.zeros((self.max_objects, 5))
            if labels is not None:
                filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
            filled_labels = torch.from_numpy(filled_labels)
            
        return img_path, img, filled_labels

    def __len__(self):
        return len(self.img_files)

    @staticmethod
    def pad_to_square(img, pad_value):
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # (upper / left) padding and (lower / right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        img = np.pad(img, pad, 'constant', constant_values=pad_value)

        return img, pad

def create_dataloader(path, batch_size, img_size, n_cpu):
    dataset = YOLODataset(path, img_size=img_size, augment=True, multiscale=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )
    return dataloader