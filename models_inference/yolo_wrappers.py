from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from utils.yolo_utils import scale_boxes, non_max_suppression_v6, non_max_suppression_v7, multiclass_nms_yolox

from models_inference.yolo_main_wrapper import YOLOONNXInference


class YOLOv7ONNXInference(YOLOONNXInference):
    def __init__(self, weights: str, image_size: int, window_batch_size: int = 4, window_size: int = 1280,
                 enable_sahi_postprocess: bool = False):

        super().__init__(weights, image_size, window_batch_size, window_size, enable_sahi_postprocess)

    def postprocess(self, preds: np.ndarray, post_img_shape: Optional[Tuple[int, int]], orig_img_shape: Optional[Tuple[int, int]], device: str = 'cpu') -> Tuple[List[np.ndarray], List[float]]:
        preds = non_max_suppression_v7(
            torch.from_numpy(preds).to(torch.float32).to(device),
            self.conf,
            self.iou
        )

        _results = []
        _confidences = []
        for i, pred in enumerate(preds):
            pred = pred.to('cpu')

            if post_img_shape is not None and orig_img_shape is not None:
                pred[:, :4] = scale_boxes(post_img_shape, pred[:, :4], orig_img_shape)

            for j in range(pred.size(0)):
                _results.append(
                    pred[j, :4].numpy()
                )
                _confidences.append(float(pred[j, 4]))

        return _results, _confidences

    def _postprocess_predictions(self, x: np.ndarray, shifts: np.ndarray) -> np.ndarray:
        x[:, :, :4] *= self.gain_k
        x[:, :, :2] += np.expand_dims(shifts.squeeze(2), axis=1)
        return x

    def _aggregate_tiled_predictions(self, x: List[np.ndarray]) -> np.ndarray:
        return np.concatenate(x, axis=1)


class YOLOv6ONNXInference(YOLOv7ONNXInference):
    def __init__(self, weights: str, support_weights: str, image_size: int, support_model_image_size: int,
                 support_window_size: int, window_batch_size: int = 4, window_size: int = 1280,
                 enable_sahi_postprocess: bool = False):

        super().__init__(weights, image_size, window_batch_size, window_size, enable_sahi_postprocess)

        _mean = [127.42253395, 129.56565639, 127.37871873]
        _std = [42.55010428, 43.93450545, 45.98804161]

        self._mean = np.array(_mean, dtype=np.float32)[:, None, None]
        self._std = np.array(_std, dtype=np.float32)[:, None, None]

    def preprocess(self, im: np.ndarray) -> np.ndarray:
        """Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.
        """
        im = np.expand_dims(self.pre_transform(im), axis=0)
        im = im.transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        img = im.astype(np.float32)
        img = (img - self._mean) / self._std
        return img

    def preprocess_for_tiling(self, im: np.ndarray) -> np.ndarray:
        """Prepares input image before tiling (window) inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): (1, 3, h, w) for tensor, (h, w, 3) for list.
        """
        im = im.transpose((2, 0, 1))  # HWC to CHW, (3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        img = im
        # img = im.astype(np.float32)
        # img = (img - self._mean) / self._std
        return img

    def postprocess(self, preds: np.ndarray, post_img_shape: Optional[Tuple[int, int]], orig_img_shape: Optional[Tuple[int, int]], device: str = 'cpu') -> Tuple[List[np.ndarray], List[float]]:
        preds = non_max_suppression_v6(
            torch.from_numpy(preds).to(torch.float32).to(device),
            self.conf,
            self.iou
        )

        _results = []
        _confidences = []
        for i, pred in enumerate(preds):
            pred = pred.to('cpu')

            if post_img_shape is not None and orig_img_shape is not None:
                pred[:, :4] = scale_boxes(post_img_shape, pred[:, :4], orig_img_shape)

            for j in range(pred.size(0)):
                _results.append(
                    pred[j, :4].numpy()
                )
                _confidences.append(float(pred[j, 4]))

        return _results, _confidences


class YOLOXONNXInference(YOLOONNXInference):
    def __init__(self, weights: str, image_size: int,
                 window_batch_size: int = 4, window_size: int = 1280,
                 enable_sahi_postprocess: bool = False):

        super().__init__(weights, image_size, window_batch_size, window_size, enable_sahi_postprocess)

    def preprocess_for_tiling(self, im: np.ndarray) -> np.ndarray:
        """Prepares input image before tiling (window) inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): (1, 3, h, w) for tensor, (h, w, 3) for list.
        """
        im = im.transpose((2, 0, 1))  # HWC to CHW, (3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        img = im.astype(np.float32)
        return img

    def postprocess(self, preds: np.ndarray, post_img_shape: Optional[Tuple[int, int]], orig_img_shape: Optional[Tuple[int, int]], device: str = 'cpu') -> Tuple[List[np.ndarray], List[float]]:
        _results = []
        _confidences = []

        for i in range(preds.shape[0]):
            _results.append(preds[i, :4])
            _confidences.append(preds[i, 4])

        return _results, _confidences

    def _postprocess_predictions(self, x: np.ndarray, shifts: np.ndarray) -> np.ndarray:
        p6 = False
        outputs = x
        grids = []
        expanded_strides = []
        strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]

        hsizes = [self.imgsz // stride for stride in strides]
        wsizes = [self.imgsz // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        result_boxes = []

        for oi, out in enumerate(outputs):
            boxes = out[:, :4]
            scores = out[:, 4:5] * out[:, 5:]

            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
            boxes_xyxy *= self.gain_k

            dets = multiclass_nms_yolox(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.5)
            if dets is not None and dets.shape[0] > 0:
                final_boxes, final_scores, _ = dets[:, :4], dets[:, 4], dets[:, 5]
                final_boxes[:, :2] += shifts[oi].squeeze(1)
                final_boxes[:, 2:4] += shifts[oi].squeeze(1)

                detections = np.concatenate((final_boxes, np.expand_dims(final_scores, axis=1)), axis=1)
                result_boxes.append(detections)

        if len(result_boxes) == 0:
            return np.zeros((0, 5), dtype=np.float32)

        return np.concatenate(result_boxes, axis=0)

    def _aggregate_tiled_predictions(self, x: List[np.ndarray]) -> np.ndarray:
        if len(x) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        return np.concatenate(x, axis=0)


if __name__ == '__main__':
    import cv2
    from models_inference.yolo_main_wrapper import convert_yoloonnx_detections_to_united_list

    w_path = 'weights/yolov8s.onnx'
    model = YOLOONNXInference(
        w_path,
        image_size=640,
        window_size=-1,
        enable_sahi_postprocess=True
    )
    imagep = '/home/alexey/Downloads/test_age.jpg'
    img = cv2.imread(imagep, cv2.IMREAD_COLOR)
    out = model(img, window_predict=False, tta_predict=False)
    out = convert_yoloonnx_detections_to_united_list(out)
    print(out)
