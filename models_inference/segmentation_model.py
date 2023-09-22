from typing import List, Tuple
import cv2
import numpy as np
import onnxruntime
from utils.image_utils import create_square_crop_by_detection


def preprocess_frame(image: np.ndarray, size: int = 1024) -> Tuple[np.ndarray, Tuple[int, int]]:
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img, (sx, sy) = create_square_crop_by_detection(
        img,
        [0, 0, *img.shape[:2][::-1]],
        pad_value=114,
        return_shifts=True
    )
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)

    img = img - np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    img = img / np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

    return img[None], (sx, sy)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class SegmentationModelWrapper(object):
    def __init__(self, onnx_weights: str):
        self.onnx_seg_model = onnxruntime.InferenceSession(
            onnx_weights,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

        self.input_name = self.onnx_seg_model.get_inputs()[0].name
        self.output_name = self.onnx_seg_model.get_outputs()[0].name

    def run_model(self, image: np.ndarray):
        t_image, (sx, sy) = preprocess_frame(image)

        pred = self.onnx_seg_model.run(
            [self.output_name],
            {self.input_name: t_image}
        )

        max_image_side = max(image.shape[:2])

        pred = pred[0][0].transpose(1, 2, 0)
        pred = cv2.resize(pred, (max_image_side, max_image_side), interpolation=cv2.INTER_CUBIC)
        pred = pred.transpose(2, 0, 1)

        pred = sigmoid(pred) > 0.5

        sx = abs(sx)
        sy = abs(sy)

        pred = pred[:, sy:sy + image.shape[0], sx:sx + image.shape[1]]

        res_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)

        for i in range(3):
            res_mask[pred[i + 1]] = i + 1

        return res_mask

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Call method of class
        Args:
            image: BGR OpenCV image

        Returns:
            Mask with (H, W) shape and which contains 4 classes:
                0: background
                1: second railway
                2: main railway
                3: train
        """
        return self.run_model(image)
