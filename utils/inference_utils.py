from typing import Tuple
import cv2
import numpy as np
from utils.image_utils import create_square_crop_by_detection



def preprocess_frame(image: np.ndarray, size: int = 1024) -> Tuple[np.ndarray, Tuple[int, int]]:
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img, (sx, sy) = create_square_crop_by_detection(
        img,
        [0, 0, *img.shape[:2][::-1]],
        pad_value=114,
        return_shifts=True
    )

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)

    img = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)

    img = img - np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    img = img / np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

    return img[None], (sx, sy)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def postprocess_segmentation(prediction: np.ndarray, original_shape: Tuple[int, int], shifts: Tuple[int, int], threshold: float = 0.5) -> np.ndarray:
    max_image_side = max(original_shape[:2])

    pred = prediction[0][1]
    pred = cv2.resize(pred, (max_image_side, max_image_side), interpolation=cv2.INTER_CUBIC)

    pred = sigmoid(pred) > threshold

    sx, sy = shifts
    sx = abs(sx)
    sy = abs(sy)

    pred = pred[sy:sy + original_shape[0], sx:sx + original_shape[1]]

    num_labels, labels_im = cv2.connectedComponents(pred.astype(np.uint8) * 255)

    if num_labels > 2:
        max_lab = max(range(1, num_labels), key=lambda x: (labels_im == x).sum())
        pred = labels_im == max_lab

    res_mask = np.zeros((original_shape[0], original_shape[1]), dtype=np.uint8)
    res_mask[pred] = 1

    return res_mask
