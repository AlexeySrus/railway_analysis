from typing import Tuple
import cv2
import numpy as np
import os
from tqdm.notebook import tqdm
from ultralytics import YOLO
from PIL import Image
from models_inference.segmentation_model import SegmentationModelWrapper
from models_inference.segmentation_model_trt import SegmentationTRTInference


def plot_railway_masks(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    class_colors = np.array(
        [
            [200, 50, 20],
            [10, 200, 20],
            [15, 200, 200]
        ]
    ).astype(np.float16)

    viz_img = image.copy()

    for i in range(2):
        cls_num = i + 1

        viz_img[mask == cls_num] = (
                image[mask == cls_num].astype(np.float16) * 0.4 +
                class_colors[i] * 0.6
        ).astype(np.uint8)

    return viz_img


if __name__ == '__main__':
    yolo_model = YOLO("yolov8l.pt")

    # seg_model = SegmentationModelWrapper('weights/seg.onnx')
    seg_model = SegmentationTRTInference('weights/seg.pkl')

    vpath = '/media/alexey/HDDData/datasets/railway/RZD_Alarm/central_cam/13_12_10.mp4'

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