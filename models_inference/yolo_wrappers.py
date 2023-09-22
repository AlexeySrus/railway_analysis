from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from utils.yolo_utils import scale_boxes, non_max_suppression_v6, non_max_suppression_v7, multiclass_nms_yolox
from models_inference.yolo_main_wrapper import YOLOONNXInference
from models_inference.yolo_trt_baseline import YoloTRTCaller


class YOLOTRTInference(YOLOONNXInference):
    def __init__(self, weights: str, image_size: int, window_batch_size: int = 4, window_size: int = 1280,
                 enable_sahi_postprocess: bool = False):

        super().__init__(weights, image_size, window_batch_size, window_size, enable_sahi_postprocess)
        self.device = 'cuda'

    def load_model(self):
        self.main_model = YoloTRTCaller(self.weights)

    def run_model(self, x: np.ndarray) -> np.ndarray:
        res = []
        for batch_sample in x:
            _out = self.main_model(batch_sample.transpose(1, 2, 0))
            res.append(_out.reshape((1, 84, -1)))
        return np.concatenate(res, axis=0)


if __name__ == '__main__':
    import cv2
    from models_inference.yolo_main_wrapper import convert_yoloonnx_detections_to_united_list
    from models_inference.yolo_config import WINDOW_SIZE, INPUT_SIZE

    w_path = 'weights/detection.pkl'
    model = YOLOTRTInference(
        w_path,
        image_size=INPUT_SIZE,
        window_size=WINDOW_SIZE,
        enable_sahi_postprocess=True
    )
    imagep = '/media/alexey/HDDData/datasets/railway/RZD_Alarm/frames/00_58_06/77.601.png'
    img = cv2.imread(imagep, cv2.IMREAD_COLOR)
    out = model(img, window_predict=True, tta_predict=False)
    out = convert_yoloonnx_detections_to_united_list(out)
    print(out)

    for det in out:
        box, prob, cls = det
        if cls != 0:
            continue
        img = cv2.rectangle(img, box[:2], box[2:], color=(50, 200, 20), thickness=5)

    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    cv2.imshow('Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
