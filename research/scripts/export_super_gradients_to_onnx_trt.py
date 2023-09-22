from typing import Tuple
import cv2
import numpy as np
import os
from super_gradients.training import models
from PIL import Image
import torch
from super_gradients.common.object_names import Models

from super_gradients.training.utils.quantization.selective_quantization_utils import SelectiveQuantizer
from super_gradients.training.utils.quantization.export import export_quantized_module_to_onnx
from torch.utils.data import DataLoader
from super_gradients.conversion import ExportQuantizationMode, DetectionOutputFormatMode
from super_gradients.conversion import ExportTargetBackend


EXPORT_SEGMENTATION: bool = True
EXPORT_DETECTION: bool = True


if __name__ == '__main__':
    if EXPORT_SEGMENTATION:
        print('Exporting segmentation model')
        seg_model = models.get(model_name=Models.PP_LITE_T_SEG75,
                           arch_params={"use_aux_heads": False},
                           num_classes=4,
                           checkpoint_path=os.path.join('/media/alexey/SSDData/experiments/railway_segmentation/super_gradients', 'PP_LITE_T_SEG75_ms/', 'ckpt_best.pth'))
        seg_model.eval()

        q_util = SelectiveQuantizer(
            default_quant_modules_calibrator_weights="max",
            default_quant_modules_calibrator_inputs="histogram",
            default_per_channel_quant_weights=True,
            default_learn_amax=False,
            verbose=True,
        )
        q_util.quantize_module(seg_model)

        dummy_input = torch.randn([1, 3, 1024, 1024], device="cpu")
        export_quantized_module_to_onnx(
            model=seg_model.cpu(),
            onnx_filename='weights/seg.onnx',
            input_shape=(1, 3, 1024, 1024),
            input_size=(1, 3, 1024, 1024),
            input_names=["input"],
            output_names=["output"],
            train=False,
        )

    if EXPORT_DETECTION:
        print('Exporting detectionm model')
        detection_model = models.get(model_name=Models.PP_YOLOE_L,
                               pretrained_weights='coco')

        # dummy_calibration_dataset = [torch.randn((3, 640, 640), dtype=torch.float32) for _ in range(32)]
        # dummy_calibration_loader = DataLoader(dummy_calibration_dataset, batch_size=8, num_workers=0)
        # # THIS IS ONLY AN EXAMPLE. YOU SHOULD USE YOUR OWN DATA-LOADER HERE
        #
        # export_result = detection_model.export(
        #     "weights/detection.onnx",
        #     # output_predictions_format=DetectionOutputFormatMode.FLAT_FORMAT,
        #     quantization_mode=ExportQuantizationMode.INT8,
        #     calibration_loader=dummy_calibration_loader,
        #     engine=ExportTargetBackend.ONNXRUNTIME
        #
        # )

        q_util = SelectiveQuantizer(
            default_quant_modules_calibrator_weights="max",
            default_quant_modules_calibrator_inputs="histogram",
            default_per_channel_quant_weights=True,
            default_learn_amax=False,
            verbose=True,
        )
        q_util.quantize_module(detection_model)

        export_quantized_module_to_onnx(
            model=detection_model.cpu(),
            onnx_filename='weights/detection.onnx',
            input_shape=(1, 3, 640, 640),
            input_size=(1, 3, 640, 640),
            input_names=["input"],
            output_names=["output"],
            train=False,
        )