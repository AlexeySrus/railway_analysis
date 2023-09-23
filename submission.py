import os
import argparse
import numpy as np
import cv2
import datetime
import pandas as pd
import glob
import tqdm

from models_inference.segmentation_model import SegmentationModelWrapper
from models_inference.yolo_main_wrapper import (
    convert_yoloonnx_detections_to_united_list,
    YOLOONNXInference,
)


class IncidentCounter:
    def __init__(self, timeout: float = 1.5):
        self.state_change_timeout = timeout

        self.last_recorded_incident_state = 1
        self.last_recorded_incident_time = np.NINF

        self.new_incident_timeout = 0
        self.new_incident = 1
        self.new_incident_start = 0

        self.incidents = []

    @property
    def total_incidents(self):
        return len(self.incidents)

    def update(self, max_state: int, timestamp: float):
        if max_state != self.last_recorded_incident_state:
            if max_state != self.new_incident:
                self.new_incident = max_state
                self.new_incident_start = timestamp

            if max_state > self.last_recorded_incident_state:
                self.incidents.append(timestamp)

            self.new_incident_timeout = timestamp - self.new_incident_start

            if (
                self.last_recorded_incident_state < self.new_incident
            ) or self.new_incident_timeout > self.state_change_timeout:
                self.last_recorded_incident_state = self.new_incident
        else:
            self.new_incident_start = timestamp

        return self.last_recorded_incident_state


def plot_railway_masks(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    class_colors = np.array([[200, 50, 20], [10, 200, 20], [15, 200, 200]]).astype(
        np.float16
    )

    viz_img = image.copy()

    for i in range(3):
        cls_num = i + 1

        viz_img[mask == cls_num] = (
            image[mask == cls_num].astype(np.float16) * 0.4 + class_colors[i] * 0.6
        ).astype(np.uint8)

    return viz_img


def plot_detections(image: np.ndarray, detections, states: list):
    vis = image.copy()

    for i, d in enumerate(detections):
        coords, _, _ = d
        x1, y1, x2, y2 = coords

        if states[i] == 2:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)

        vis = cv2.rectangle(vis, (x1, y1), (x2, y2), thickness=2, color=color)

    return vis


def plot_status(image: np.ndarray, status: str) -> np.ndarray:
    h, w, _ = image.shape

    fontScale = 1.4
    thickness = 3
    text_size = cv2.getTextSize(status, 0, fontScale=fontScale, thickness=thickness)[0]

    return cv2.putText(
        image,
        status,
        (0, text_size[1]),
        fontFace=0,
        fontScale=fontScale,
        color=[200, 200, 0],
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def get_intersection_states(detections: list[tuple], mask: np.ndarray) -> list:
    res = []

    for d in detections:
        if mask[d[0][3], d[0][0]] == 1 or mask[d[0][3], d[0][2]] == 1:
            res.append(2)
        else:
            res.append(1)

    return res


def upscale_bbox(bbox: np.ndarray, shape: tuple, percent: float = 0.4) -> list:
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    padd = int(h * percent)

    return [int(max(x1 - padd, 0)), y1, min(x2 + padd, shape[1] - 1), y2]


def process_video(
    sm: SegmentationModelWrapper,
    detector: YOLOONNXInference,
    vpath: str,
    vis: bool = False,
) -> pd.DataFrame:
    tracker = IncidentCounter(2)

    stream = cv2.VideoCapture(vpath, cv2.CAP_FFMPEG)

    grabbed = True
    while grabbed:
        grabbed, frame = stream.read()
        timestamp = stream.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if not grabbed:
            break

        mask = sm(frame)
        detections = detector(frame, window_predict=False, tta_predict=False)
        detections = convert_yoloonnx_detections_to_united_list(detections)
        detections = [
            (upscale_bbox(d[0], frame.shape), d[1], d[2]) for d in detections
        ]

        states = get_intersection_states(detections, mask)
        tracker.update(max(states) if len(states) > 0 else 1, timestamp)

        vis_frame = plot_railway_masks(frame, mask)
        vis_frame = plot_detections(vis_frame, detections, states)

        status = (
            "INCIDENT" if tracker.last_recorded_incident_state == 2 else "NORMAL"
        )
        vis_frame = plot_status(
            vis_frame, f"STATUS: {status}, COUNT {tracker.total_incidents}"
        )

        if vis:
            cv2.imshow(
                "Ocular Hazard Avoidance Module",
                vis_frame,
            )
            cv2.waitKey(1)

    stream.release()
    
    return pd.DataFrame(
        {
            "filename": [1],
            "cases_count": [tracker.total_incidents],
            "timestamps": [[datetime.datetime.fromtimestamp(int(t)).strftime("%M:%S") for t in tracker.incidents]],
        }
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to the video")
    parser.add_argument("--output", type=str, default="output.csv", help="Output CSV file")
    parser.add_argument("--vis", default=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    sm = SegmentationModelWrapper("./weights/seg.onnx")
    detector = YOLOONNXInference(
        "./weights/yolov8s.onnx",
        image_size=640,
        window_size=-1,
        enable_sahi_postprocess=False,
    )

    vpaths = glob.glob(args.video)
    
    csv = pd.DataFrame(columns = ["filename", "cases_count", "timestamps"])
    for vpath in tqdm.tqdm(vpaths):
        new_csv = process_video(sm, detector, vpath, args.vis)
        csv = pd.concat([csv, new_csv])
        csv.to_csv(args.output, index=False)
