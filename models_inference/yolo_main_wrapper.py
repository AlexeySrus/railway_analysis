from typing import List, Optional, Tuple

import numpy
import onnxruntime
import numpy as np
import cv2
import torch
from sahi.postprocess.combine import GreedyNMMPostprocess
from sahi.prediction import ObjectPrediction

from utils.yolo_utils import scale_boxes, non_max_suppression_v8, get_tta_transforms, rotate_image, reverse_boxed_from_rotation
from models_inference.yolo_config import DETECTION_CONF, NMS_TH, SAHI_MATCHING_METRIC, GREEDY_NMM_TH, WINDOW_STRIDE


class LetterBox:
    """Resize image and padding for detection, instance segmentation, pose."""

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, stride=32):
        """Initialize LetterBox object with specific parameters."""
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride

    def __call__(self, labels=None, image=None):
        """Return updated labels and image with added border."""
        if labels is None:
            labels = {}
        img = labels.get('img') if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop('rect_shape', self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if labels.get('ratio_pad'):
            labels['ratio_pad'] = (labels['ratio_pad'], (dw, dh))  # for evaluation

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=(114, 114, 114))  # add border

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels['img'] = img
            labels['resized_shape'] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels."""
        labels['instances'].convert_bbox(format='xyxy')
        labels['instances'].denormalize(*labels['img'].shape[:2][::-1])
        labels['instances'].scale(*ratio)
        labels['instances'].add_padding(padw, padh)
        return labels


class YOLOONNXInference(object):
    def __init__(self,
                 weights: str,
                 image_size: int,
                 window_batch_size: int = 1,
                 window_size: int = 1280,
                 enable_sahi_postprocess: bool = False):
        self.imgsz = image_size
        self.w_bs = window_batch_size
        self.w_imgsz = window_size
        self.gain_k = self.w_imgsz / self.imgsz
        self.enable_sahi_postprocess = enable_sahi_postprocess

        self.weights = weights
        self.model_is_load: bool = False
        self.stride = 32

        self.pre_transformer = LetterBox((self.imgsz, self.imgsz), auto=False, stride=self.stride)
        self.shi_postprocessor = GreedyNMMPostprocess(match_threshold=GREEDY_NMM_TH, match_metric=SAHI_MATCHING_METRIC, class_agnostic=False)

        self._mean: Optional[numpy.ndarray] = None
        self._std: Optional[numpy.ndarray] = None
        self._set_mean_stde()

        self.conf = DETECTION_CONF
        self.iou = NMS_TH

    def load_model(self):
        if torch.cuda.is_available():
            providers = ['CUDAExecutionProvider']
            self.device = 'cuda'
        else:
            print('#' * 15 + ' WARNING: CPU RUN ' + '#' * 15)
            providers = ['CPUExecutionProvider']
            self.device = 'cpu'
        self.main_model = onnxruntime.InferenceSession(self.weights, providers=providers)
        meta = self.main_model.get_modelmeta().custom_metadata_map
        if 'stride' in meta:
            self.stride = int(meta['stride'])
        else:
            self.stride = 32

        self.model_input_name = self.main_model.get_inputs()[0].name
        self.model_output_name = self.main_model.get_outputs()[0].name

        self.model_is_load = True

    def pre_transform(self, x: np.ndarray) -> np.ndarray:
        return self.pre_transformer(image=x)

    def _set_mean_stde(self):
        _mean = [0, 0, 0]
        _std = [1, 1, 1]

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
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def preprocess_for_tiling(self, im: np.ndarray) -> np.ndarray:
        """Prepares input image before tiling (window) inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): (1, 3, h, w) for tensor, (h, w, 3) for list.
        """
        im = im.transpose((2, 0, 1))  # HWC to CHW, (3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        img = im.astype(np.float32)
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def run_model(self, x: np.ndarray) -> np.ndarray:
        out = self.main_model.run(
            [self.model_output_name],
            {self.model_input_name: x.astype(np.float32)}
        )
        return out[0]

    def postprocess(self, preds: np.ndarray, post_img_shape: Optional[Tuple[int, int]], orig_img_shape: Optional[Tuple[int, int]], device: str = 'cpu') -> Tuple[List[np.ndarray], List[float], List[int]]:
        preds = non_max_suppression_v8(
            torch.from_numpy(preds).to(torch.float32).to(device),
            self.conf,
            self.iou
        )

        _results = []
        _confidences = []
        _classes = []
        for i, pred in enumerate(preds):
            pred = pred.to('cpu')

            if post_img_shape is not None and orig_img_shape is not None:
                pred[:, :4] = scale_boxes(post_img_shape, pred[:, :4], orig_img_shape)

            for j in range(pred.size(0)):
                _results.append(
                    pred[j, :4].numpy()
                )
                _confidences.append(float(pred[j, 4]))
                _classes.append(int(pred[j, 5]))

        return _results, _confidences, _classes

    def _sahi_postprocess(self,
            boxes_result: List[np.ndarray],
            confidences: List[float],
            classes: List[int],
            image_shape: Tuple[int, int]) -> Tuple[List[np.ndarray], List[float], List[int], Tuple[int, int]]:
        sahi_boxes: List[ObjectPrediction] = []

        for i in range(len(confidences)):
            lst_box: List[int] = boxes_result[i].tolist()
            sahi_boxes.append(
                ObjectPrediction(
                    bbox=[
                        lst_box[0],
                        lst_box[1],
                        lst_box[0] + lst_box[2],
                        lst_box[1] + lst_box[3]
                    ],
                    category_id=classes[i],
                    category_name=str(classes[i]),
                    score=confidences[i],
                    full_shape=list(image_shape)
                )
            )

        postprocessed_sahi_boxes = self.shi_postprocessor(sahi_boxes)
        res_boxes = [np.array(pp.bbox.to_xywh()) for pp in postprocessed_sahi_boxes]
        res_confs = [pp.score.value for pp in postprocessed_sahi_boxes]
        res_cls = [int(pp.category.id) for pp in postprocessed_sahi_boxes]
        return res_boxes, res_confs, res_cls, image_shape

    def _generate_tiles_coordinates(self, img_size :Tuple[int, int, int]) -> Tuple[List[int], List[int]]:
        tile_size = self.w_imgsz
        stride = WINDOW_STRIDE
        x0_vec = []
        y0_vec = []

        target_x = 0
        while target_x + tile_size < img_size[2]:
            x0_vec.append(target_x)
            target_x += stride
        x0_vec.append(img_size[2] - tile_size - 1)

        target_y = 0
        while target_y + tile_size < img_size[1]:
            y0_vec.append(target_y)
            target_y += stride
        y0_vec.append(img_size[1] - tile_size - 1)

        return x0_vec, y0_vec

    def _postprocess_predictions(self, x: np.ndarray, shifts: np.ndarray) -> np.ndarray:
        x[:, :4] *= self.gain_k
        x[:, :2] += shifts
        return x

    def _aggregate_tiled_predictions(self, x: List[np.ndarray]) -> np.ndarray:
        return np.concatenate(x, axis=2)

    def _process_tiled_batch(self, inference_batch):
        images_inference_batch = np.stack([b[0] for b in inference_batch], axis=0)
        # print(images_inference_batch.shape, images_inference_batch.max())
        shifts = np.array([b[1] for b in inference_batch], dtype=np.float32)
        shifts = np.expand_dims(shifts, axis=2)
        batch_tiled_predictions = []

        if self.w_imgsz != self.imgsz:
            images_inference_batch = np.array(
                [cv2.resize(img.transpose((1, 2, 0)), (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA).transpose((2, 0, 1)) for img in images_inference_batch],
                dtype=np.float32
            )
            images_inference_batch = (images_inference_batch - self._mean) / self._std

        yolo_prediction = self.run_model(images_inference_batch)
        yolo_prediction = self._postprocess_predictions(yolo_prediction, shifts)
        for i in range(len(yolo_prediction)):
            batch_tiled_predictions.append(np.expand_dims(yolo_prediction[i], axis=0))

        return batch_tiled_predictions

    def window_predict(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[float], List[int]]:
        img = self.preprocess_for_tiling(image)
        tile_size = self.w_imgsz

        x0_vec, y0_vec = self._generate_tiles_coordinates((img.shape[0], 2 * img.shape[1] // 3, img.shape[2]))

        inference_batch = []
        tiled_predictions = []
        for y0 in y0_vec:
            for x0 in x0_vec:
                img_crop = img[:, y0:y0 + tile_size, x0:x0 + tile_size]
                if img_crop.shape[1] * img_crop.shape[2] == 0:
                    continue

                if len(inference_batch) < self.w_bs:
                    inference_batch.append((img_crop, (x0, y0)))
                else:
                    tiled_predictions += self._process_tiled_batch(inference_batch)
                    inference_batch = []

        if len(inference_batch) > 0:
            tiled_predictions += self._process_tiled_batch(inference_batch)

        tiled_predictions = self._aggregate_tiled_predictions(tiled_predictions)
        batch_predictions = self.postprocess(tiled_predictions, None, None, self.device)
        return batch_predictions

    def tta_inference(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[float], List[int]]:
        forward_transforms, inv_transforms = get_tta_transforms()

        transformed_image = self.preprocess(image.copy()).squeeze(0)

        tta_batch = np.stack(
            [rotate_image(transformed_image, tr) for tr in forward_transforms],
            axis=0
        )
        if self.w_bs == 1:
            yolo_prediction = np.concatenate([self.run_model(np.expand_dims(ta, axis=0)) for ta in tta_batch], axis=0)
        else:
            yolo_prediction = self.run_model(tta_batch)

        _boxes, _confidences, _classes = [], [], []

        for _bi in range(len(forward_transforms)):
            sample_preds = np.expand_dims(yolo_prediction[_bi], axis=0)
            changed_shape = _bi in [1, 3]
            image_shape = image.shape[:2]
            if changed_shape:
                image_shape = image_shape[::-1]

            sample_boxes, sample_confidences, sample_classes = self.postprocess(sample_preds, (self.imgsz, self.imgsz), image_shape)

            if len(sample_boxes) == 0:
                continue

            rev_boxes = reverse_boxed_from_rotation(np.array(sample_boxes), inv_transforms[_bi], image_shape)
            rev_boxes = [rb for rb in rev_boxes]
            _boxes += rev_boxes
            _confidences += sample_confidences
            _classes += sample_classes

        return _boxes, _confidences, _classes

    def tta_window_inference(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[float], List[int]]:
        forward_transforms, inv_transforms = get_tta_transforms(restricted=False)

        chw_image = image.transpose((2, 0, 1))

        tta_hwc_images = [rotate_image(chw_image, tr).transpose((1, 2, 0)) for tr in forward_transforms]
        _boxes, _confidences, _classes = [], [], []

        for _bi in range(len(forward_transforms)):
            _input_image = tta_hwc_images[_bi]
            changed_shape = _bi in [1, 3]
            image_shape = (image.shape[0], image.shape[1])
            if changed_shape:
                image_shape = image_shape[::-1]

            sample_boxes, sample_confidences, sample_classes = self.window_predict(_input_image)

            if len(sample_boxes) == 0:
                continue

            rev_boxes = reverse_boxed_from_rotation(np.array(sample_boxes), inv_transforms[_bi], image_shape)

            rev_boxes = [rb for rb in rev_boxes]
            _boxes += rev_boxes
            _confidences += sample_confidences
            _classes += sample_classes

        return _boxes, _confidences, _classes

    def __call__(self, image: str, window_predict: bool = False, tta_predict: bool = False) -> Tuple[List[np.ndarray], List[float], List[int], Tuple[int, int]]:
        """
        Call YOLO ONNX model
        Args:
            image: BGR OpenCV image
            window_predict: use window predict
            tta_predict: use TTA under inference

        Returns:
            Tuple of the following values:
                - List of boxes in XYWH format, where XY - is left upper point
                - List of confidences
                - List of categories indexes
                - Original image shape
        """
        if not self.model_is_load:
            self.load_model()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if tta_predict:
            assert self.enable_sahi_postprocess, 'To use TTA SAHI postprocess is enabled'

        boxes_result = []
        confidences = []
        classes = []

        if window_predict:
            if tta_predict:
                boxes_result, confidences, classes = self.tta_window_inference(image)
            else:
                boxes_result, confidences, classes = self.window_predict(image)

        if tta_predict:
            ff_boxes_result, ff_confidences, ff_classes = self.tta_inference(image)
        else:
            transformed_image = self.preprocess(image.copy())
            yolo_prediction = self.run_model(transformed_image)
            ff_boxes_result, ff_confidences, ff_classes = self.postprocess(yolo_prediction, (self.imgsz, self.imgsz), image.shape[:2])

        boxes_result += ff_boxes_result
        confidences += ff_confidences
        classes += ff_classes

        if len(boxes_result) == 0:
            return [], [], [], image.shape[:2]

        # XYXY -> XYWH
        boxes_result = np.array(boxes_result)
        boxes_result[:, 2:] = boxes_result[:, 2:] - boxes_result[:, :2]
        boxes_result = [boxes_result[_i] for _i in range(len(boxes_result))]
        if self.enable_sahi_postprocess:
            return self._sahi_postprocess(boxes_result, confidences, classes, image.shape[:2])
        return boxes_result, confidences, classes, image.shape[:2]


def xyhw2xyxy(_bbox: List[float]) -> List[int]:
    x1, y1, w, h = _bbox
    return [int(v) for v in [x1, y1, x1 + w, y1 + h]]


def convert_yoloonnx_detections_to_united_list(
        detections: Tuple[List[np.ndarray], List[float], List[int], Tuple[int, int]]) -> List[Tuple[List[int], float, int]]:
    if len(detections) == 0:
        return []
    return [
        (
            xyhw2xyxy(list(detections[0][i])),
            detections[1][i],
            detections[2][i]
        )
        for i in range(len(detections[0]))
    ]
