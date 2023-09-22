from typing import Tuple
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from tqdm.notebook import tqdm
from ultralytics import YOLO
from super_gradients.training import models
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, ToPILImage
from PIL import Image
import torch
from utils.image_utils import create_square_crop_by_detection
from super_gradients.common.object_names import Models
from timeit import default_timer

from super_gradients.training.utils.quantization.selective_quantization_utils import SelectiveQuantizer
from super_gradients.training.utils.quantization.export import export_quantized_module_to_onnx

from models_inference.segmentation_model import SegmentationModelWrapper


def plot_railway_masks(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    class_colors = np.array(
        [
            [200, 50, 20],
            [10, 200, 20],
            [15, 200, 200]
        ]
    ).astype(np.float16)

    viz_img = image.copy()

    for i in range(3):
        cls_num = i + 1

        viz_img[mask == cls_num] = (
                image[mask == cls_num].astype(np.float16) * 0.4 +
                class_colors[i] * 0.6
        ).astype(np.uint8)

    return viz_img


if __name__ == '__main__':
    yolo_model = YOLO("yolov8l.pt")

    # seg_model = models.get(model_name=Models.PP_LITE_T_SEG75,
    #                    arch_params={"use_aux_heads": False},
    #                    num_classes=4,
    #                    checkpoint_path=os.path.join('/media/alexey/SSDData/experiments/railway_segmentation/super_gradients', 'PP_LITE_T_SEG75_ms/', 'ckpt_best.pth'))
    # seg_model.eval()
    #
    # q_util = SelectiveQuantizer(
    #     default_quant_modules_calibrator_weights="max",
    #     default_quant_modules_calibrator_inputs="histogram",
    #     default_per_channel_quant_weights=True,
    #     default_learn_amax=False,
    #     verbose=True,
    # )
    # q_util.quantize_module(seg_model)
    #
    # dummy_input = torch.randn([1, 3, 1024, 1024], device="cpu")
    # export_quantized_module_to_onnx(
    #     model=seg_model.cpu(),
    #     onnx_filename='weights/seg.onnx',
    #     input_shape=(1, 3, 1024, 1024),
    #     input_size=(1, 3, 1024, 1024),
    #     input_names=["input"],
    #     output_names=["output"],
    #     train=False,
    # )
    # torch.onnx.export(
    #     seg_model,
    #     dummy_input,
    #     'weights/seg.onnx',
    #     input_names=["input"],
    #     output_names=["output"]
    # )
    #
    # onnx_seg_model = onnxruntime.InferenceSession(
    #     'weights/seg.onnx',
    #     providers=['CPUExecutionProvider']
    # )
    seg_model = SegmentationModelWrapper('weights/seg.onnx')

    vpath = '/media/alexey/HDDData/datasets/railway/RZD_Alarm/side_cam/10_37_17.mp4'

    stream = cv2.VideoCapture(vpath, cv2.CAP_FFMPEG)

    stream_width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    stream_height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(stream.get(cv2.CAP_PROP_FPS))
    print('FPS: {}, W: {}, H: {}'.format(fps, stream_width, stream_height))

    out_win = 'Video'
    cv2.namedWindow(out_win, cv2.WINDOW_NORMAL)

    while True:
        grabbed, frame = stream.read()

        if not grabbed:
            break

        timestamp = stream.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # model_out = yolo_model([frame])
        # viz = model_out[0].plot()

        viz = plot_railway_masks(frame, seg_model(frame))

        # title='Time: {:.3f} sec'.format(timestamp))
        cv2.imshow(out_win, viz)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break


    cv2.destroyWindow(out_win)
    stream.release()