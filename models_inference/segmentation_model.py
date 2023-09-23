from typing import List, Tuple
import cv2
import numpy as np
import onnxruntime
from timeit import default_timer as timer

from utils.inference_utils import preprocess_frame, postprocess_segmentation


class SegmentationModelWrapper(object):
    def __init__(self, onnx_weights: str):
        self.onnx_seg_model = onnxruntime.InferenceSession(
            onnx_weights,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

        self.input_name = self.onnx_seg_model.get_inputs()[0].name
        self.output_name = self.onnx_seg_model.get_outputs()[0].name

    def run_model(self, image: np.ndarray):
        start_preprocess_time = timer()
        t_image, (sx, sy) = preprocess_frame(image)
        finish_preprocess_time = timer()

        start_inference_time = timer()
        pred = self.onnx_seg_model.run(
            [self.output_name],
            {self.input_name: t_image}
        )
        finish_inference_time = timer()

        start_postprocess_time = timer()
        res_mask = postprocess_segmentation(pred[0], (image.shape[0], image.shape[1]), (sx, sy))
        finish_postprocess_time = timer()

        # print('Raw preprocess time: {:.3f}'.format(finish_preprocess_time - start_preprocess_time))
        # print('Raw segmentation time: {:.3f}'.format(finish_inference_time - start_inference_time))
        # print('Raw postprocess time: {:.3f}'.format(finish_postprocess_time - start_postprocess_time))

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
