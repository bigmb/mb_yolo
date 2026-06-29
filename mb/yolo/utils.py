import os

import cv2
import numpy as np
import torch

# YuNet ONNX face-detection model shipped alongside this module.
_YUNET_MODEL = os.path.join(os.path.dirname(__file__),
                            "face_detection_yunet_2023mar.onnx")

__all__ = ['Utils', 'compute_saliency', 'saliency_point', 'saliency_crop',
           'detect_faces']

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


def compute_saliency(img_bgr, method="spectral"):
    """Compute a saliency map for an image using OpenCV's saliency module.

    Finds where the visually important content is, so a crop can follow it
    instead of always cropping from the centre. Requires opencv-contrib-python
    (provides ``cv2.saliency``).

    Args:
        img_bgr (numpy.ndarray): BGR image as read by ``cv2.imread``.
        method (str): "spectral" -> StaticSaliencySpectralResidual (fast, blobby);
            "fine" -> StaticSaliencyFineGrained (sharper, slower).

    Returns:
        numpy.ndarray: float32 saliency map in [0, 1] with the same H,W as the
            input (brighter == more salient).

    Example:
        >>> img = cv2.imread("photo.jpg")
        >>> sal = compute_saliency(img)
    """
    if method == "fine":
        detector = cv2.saliency.StaticSaliencyFineGrained_create()
    else:
        detector = cv2.saliency.StaticSaliencySpectralResidual_create()

    success, sal = detector.computeSaliency(img_bgr)
    if not success:
        raise RuntimeError("OpenCV saliency computation failed")

    sal = sal.astype(np.float32)
    sal -= sal.min()
    if sal.max() > 0:
        sal /= sal.max()
    return sal


def detect_faces(img_bgr, backend="yunet", score_threshold=0.6,
                 model_path=None, min_size_frac=0.02):
    """Detect faces with OpenCV.

    Two backends:
      - "yunet" (default): the YuNet DNN detector — accurate, handles glasses,
        angled and partially-occluded faces. Uses the ONNX model shipped with
        the package. Falls back to Haar automatically if the model is missing.
      - "haar": the classic Haar cascade — no model file needed, but misses
        glasses/angled faces and produces more false positives.

    Args:
        img_bgr (numpy.ndarray): BGR image.
        backend (str): "yunet" or "haar".
        score_threshold (float): Min confidence for YuNet detections.
        model_path (str, optional): Path to the YuNet ONNX model; defaults to the
            one bundled with this package.
        min_size_frac (float): Ignore faces smaller than this fraction of the
            image width (filters tiny detections).

    Returns:
        list[tuple]: Faces as ``(x, y, w, h)`` boxes, sorted largest first.

    Example:
        >>> img = cv2.imread("photo.jpg")
        >>> faces = detect_faces(img)              # YuNet
        >>> faces = detect_faces(img, "haar")      # cascade fallback
    """
    H, W = img_bgr.shape[:2]
    min_side = int(W * min_size_frac)
    path = model_path or _YUNET_MODEL

    if backend == "yunet" and os.path.exists(path):
        detector = cv2.FaceDetectorYN_create(
            path, "", (W, H), score_threshold=score_threshold)
        detector.setInputSize((W, H))
        _, dets = detector.detect(img_bgr)
        faces = []
        if dets is not None:
            for d in dets:
                x, y, w, h = (int(v) for v in d[:4])
                if w >= min_side and h >= min_side:
                    faces.append((x, y, w, h))
    else:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        dets = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side))
        faces = [tuple(int(v) for v in f) for f in dets]

    faces.sort(key=lambda f: f[2] * f[3], reverse=True)   # largest first
    return faces


def boost_saliency_with_faces(sal, faces, face_weight=3.0):
    """Add Gaussian bumps at face locations to a saliency map.

    Makes faces dominate the map so saliency-guided crops keep them in frame.

    Args:
        sal (numpy.ndarray): float32 saliency map in [0, 1].
        faces (list[tuple]): ``(x, y, w, h)`` face boxes.
        face_weight (float): Peak height of each face bump relative to the map
            (which is normalised to 1). Larger areas get proportionally taller
            bumps so big/close faces win over small/distant ones.

    Returns:
        numpy.ndarray: float32 map renormalised to [0, 1].
    """
    if not faces:
        return sal
    boosted = sal.copy()
    H, W = sal.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    max_area = max(w * h for (_, _, w, h) in faces)
    for (x, y, w, h) in faces:
        cx, cy = x + w / 2.0, y + h / 2.0
        sigma = max(w, h) * 0.6
        bump = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))
        boosted += bump.astype(np.float32) * face_weight * (w * h / max_area)
    boosted -= boosted.min()
    if boosted.max() > 0:
        boosted /= boosted.max()
    return boosted


def saliency_point(img_bgr, sal=None, top_pct=95.0, method="spectral"):
    """Return the ``(x, y)`` point of strongest saliency in an image.

    Uses the centroid of the top ``top_pct`` percentile of the saliency map so
    the result is a stable region centre, not a single noisy peak pixel.

    Args:
        img_bgr (numpy.ndarray): BGR image.
        sal (numpy.ndarray, optional): Precomputed saliency map; computed if None.
        top_pct (float): Percentile threshold for the salient region (default 95).
        method (str): Saliency method passed to ``compute_saliency``.

    Returns:
        tuple: ``(x, y)`` integer pixel coordinates of the point of interest.

    Example:
        >>> img = cv2.imread("photo.jpg")
        >>> x, y = saliency_point(img)
    """
    if sal is None:
        sal = compute_saliency(img_bgr, method=method)
    thresh = np.percentile(sal, top_pct)
    ys, xs = np.nonzero(sal >= thresh)
    weights = sal[ys, xs]
    cx = int(round(np.average(xs, weights=weights)))
    cy = int(round(np.average(ys, weights=weights)))
    return cx, cy


def saliency_crop(img_bgr, tile_size, sal=None, mode="energy", method="spectral",
                  use_faces=False, face_weight=3.0):
    """Crop a tile of a target aspect ratio guided by saliency (and faces).

    Instead of cropping from the middle, this places the crop window over the
    most salient region of the image — useful for generating thumbnails/tiles
    that keep the subject in frame.

    Args:
        img_bgr (numpy.ndarray): BGR image.
        tile_size (tuple): ``(tile_w, tile_h)`` target aspect ratio (only the
            ratio matters). The crop is the LARGEST window of this ratio that
            fits inside the image.
        sal (numpy.ndarray, optional): Precomputed saliency map; computed if None.
        mode (str): "energy" slides the window to maximise total saliency inside
            it (most robust); "point" centres the window on ``saliency_point``
            and clamps to the image bounds.
        method (str): Saliency method passed to ``compute_saliency``.
        use_faces (bool): If True, detect faces and boost the saliency map at
            their locations so the crop keeps faces in frame. The returned point
            becomes the largest face's centre when any face is found.
        face_weight (float): Strength of the face boost (see
            ``boost_saliency_with_faces``).

    Returns:
        tuple: ``(crop, box, point)`` where ``crop`` is the cropped image,
            ``box`` is ``(x0, y0, x1, y1)`` in original coords, and ``point`` is
            the ``(x, y)`` point of interest used (largest face centre if
            ``use_faces`` found one, else the saliency peak).

    Example:
        >>> img = cv2.imread("photo.jpg")
        >>> crop, box, point = saliency_crop(img, tile_size=(16, 9), use_faces=True)
        >>> cv2.imwrite("smart_crop.jpg", crop)
    """
    if sal is None:
        sal = compute_saliency(img_bgr, method=method)

    faces = []
    if use_faces:
        faces = detect_faces(img_bgr)
        if faces:
            sal = boost_saliency_with_faces(sal, faces, face_weight=face_weight)

    H, W = img_bgr.shape[:2]
    tw, th = tile_size
    ar = tw / th

    # Largest window of aspect ratio `ar` that fits inside the image.
    if W / H > ar:                       # image wider than tile -> tile height = H
        cw, ch = int(round(H * ar)), H
    else:                                # image taller than tile -> tile width = W
        cw, ch = W, int(round(W / ar))
    cw, ch = min(cw, W), min(ch, H)

    if faces:
        fx, fy, fw, fh = faces[0]              # largest face
        pt = (fx + fw // 2, fy + fh // 2)
    else:
        pt = saliency_point(img_bgr, sal)

    if mode == "energy" and (cw < W or ch < H):
        # Integral image -> O(1) window-sum at every offset; pick the max.
        integral = cv2.integral(sal)     # (H+1, W+1)
        best, box = -1.0, (0, 0)
        x_range = range(0, W - cw + 1)
        y_range = range(0, H - ch + 1)
        step = max(1, min(cw, ch) // 64)  # coarse step keeps it fast; set 1 for exact
        for y0 in list(y_range)[::step] or [0]:
            for x0 in list(x_range)[::step] or [0]:
                s = (integral[y0 + ch, x0 + cw] - integral[y0, x0 + cw]
                     - integral[y0 + ch, x0] + integral[y0, x0])
                if s > best:
                    best, box = s, (x0, y0)
        x0, y0 = box
    else:
        # Centre the window on the salient point, clamp to bounds.
        x0 = int(np.clip(pt[0] - cw // 2, 0, W - cw))
        y0 = int(np.clip(pt[1] - ch // 2, 0, H - ch))

    x1, y1 = x0 + cw, y0 + ch
    return img_bgr[y0:y1, x0:x1].copy(), (x0, y0, x1, y1), pt
