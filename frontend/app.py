import time
from typing import Optional
import datetime
import streamlit as st
import numpy as np
import cv2
from PIL import Image

from models_inference.segmentation_model import SegmentationModelWrapper
from models_inference.yolo_main_wrapper import (
    convert_yoloonnx_detections_to_united_list,
    YOLOONNXInference,
)
    

class IncidentCounter():
    def __init__(self, timeout: float = 1.5):
        self.state_change_timeout = timeout

        self.last_recorded_incident_state = 1
        self.last_recorded_incident_time = np.NINF

        self.new_incident_timeout = 0
        self.new_incident = 1
        self.new_incident_start = time.time()
        
        self.incidents = []

    @property
    def total_incidents(self):
        return len(self.incidents)
    
    def update(self, max_state: int, timestamp: Optional[float] = None):
        if max_state != self.last_recorded_incident_state:
            if max_state != self.new_incident:
                self.new_incident = max_state
                self.new_incident_start = time.time()

            if max_state > self.last_recorded_incident_state:
                self.incidents.append(timestamp)
                
            self.new_incident_timeout = time.time() - self.new_incident_start

            if (self.last_recorded_incident_state < self.new_incident) \
            or self.new_incident_timeout > self.state_change_timeout:
                self.last_recorded_incident_state = self.new_incident
        else:
            self.new_incident_start = time.time()

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
    text_size = cv2.getTextSize(
        status, 0, fontScale=fontScale, thickness=thickness
    )[0]

    return cv2.putText(
        image,
        status,
        # (w - text_size[0], h - text_size[1]),
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


def get_segmentation_model():
    return SegmentationModelWrapper("./data/seg.onnx")


def get_detection_model():
    return YOLOONNXInference(
        "./data/yolov8s.onnx",
        image_size=640,
        window_size=-1,
        enable_sahi_postprocess=False,
    )


if __name__ == "__main__":
    st.set_page_config(
        layout="wide",
        page_title="Цифровой прорыв/Безопасный маршрут"
        # initial_sidebar_state="expanded",
    )

    header = """
                <h1 style="color:#2a93b9;">
                Безопасный маршрут
                </h1>
                """

    st.markdown(header, unsafe_allow_html=True)
    st.caption("Разработано командой **RabotyagiTeam**👷")

    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                .css-z5fcl4 {padding: 1rem 6rem 1rem 6rem;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.markdown(
        """
        <style>
            span[data-baseweb="tag"][aria-label="Трещина, close by backspace"]{
                background-color: orange;
            }
            span[data-baseweb="tag"][aria-label="Заплатка, close by backspace"]{
                background-color: purple;
            }
            span[data-baseweb="tag"][aria-label="Дыра, close by backspace"]{
                background-color: red;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    sm = get_segmentation_model()
    detector = get_detection_model()

    with st.form("my_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.caption("Загрузка данных для обработки")
            uploaded_file = st.file_uploader("Выберите .mp4 файл")

            checkbox_val = st.checkbox("Сохранить видео проишествий")
            checkbox_val = st.checkbox("Сохранить csv отчет")

        with col2:
            st.caption("Параметры камеры")

        submitted = st.form_submit_button("Старт")

    if submitted and uploaded_file is not None:
        # with st.form("inference_form"):
        stop_inference = st.button("Остановить")

        if not stop_inference:
            st.warning(
                "Внимание! Обработка занимает некоторое время. Пожалуйста, не перезагружайте страницу!",
                icon="🚨",
            )

        with open(uploaded_file.name, mode="wb") as f:
            f.write(uploaded_file.read())

        stream = cv2.VideoCapture(uploaded_file.name, cv2.CAP_FFMPEG)

        st.session_state["stream"] = stream
        st.session_state["active"] = True
        # stream_length = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
        state_status = st.empty()
        frame_display = st.empty()
        # stop_button = st.button(label="Остановить")

        tracker = IncidentCounter(2)

        grabbed = True
        while grabbed and not stop_inference:
            f_start = time.time()

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
            tracker.update(max(states) if len(states) > 0 else 1)
            
            v_time = time.time()
            vis_frame = plot_railway_masks(frame, mask)
            vis_frame = plot_detections(vis_frame, detections, states)

            status = "INCIDENT" if tracker.last_recorded_incident_state == 2 else "NORMAL"
            if status == "INCIDENT":
                state_status.error("Опасная ситуация!", icon="🚨")
            else:
                state_status.info("Путь безопасен", icon="✅")
            # else:
            #     state_status.info("Опасная ситуация!")
            vis_frame = plot_status(vis_frame, f"STATUS: {status}, COUNT {tracker.total_incidents}")

            frame_display.image(
                Image.fromarray(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB))
            )
                # if stop_inference:
            # print("STOP")
            # stream.release()
            # os.remove(uploaded_file.name)
            # grabbed = False
