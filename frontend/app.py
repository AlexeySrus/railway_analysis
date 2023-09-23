import streamlit as st
import numpy as np
import cv2
from PIL import Image
from timeit import default_timer as timer

from models_inference.segmentation_model import SegmentationModelWrapper
from models_inference.yolo_main_wrapper import convert_yoloonnx_detections_to_united_list
from models_inference.yolo_wrappers import YOLOONNXInference


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


def plot_detections(image: np.ndarray, detections, mask):
    vis = image.copy()

    for d in detections:
        coords, conf, cls = d
        x1, y1, x2, y2 = coords

        if mask[d[0][3], d[0][0]] == 1 or mask[d[0][3], d[0][2]] == 1:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)

        vis = cv2.rectangle(vis, (x1, y1), (x2, y2), thickness=2, color=color)
    
    return vis


def upscale_bbox(bbox: np.ndarray, shape: tuple, percent: float = 0.4) -> list:
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    padd = int(h * percent)
    
    return [int(max(x1 - padd, 0)), y1, min(x2 + padd, shape[1] - 1), y2]


def get_segmentation_model():
    return SegmentationModelWrapper("./weights/seg.onnx")


def get_detection_model():
    return YOLOONNXInference(
        "./weights/yolov8l.onnx",
        image_size=640,
        window_size=-1,
        enable_sahi_postprocess=False
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
        unsafe_allow_html=True)

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
        with st.form("inference_form"):
            stop_inference = st.form_submit_button("Остановить")

            st.warning("Внимание! Обработка занимает некоторое время. Пожалуйста, не перезагружайте страницу!", icon="🚨")

            # progress_text = "Нейронки за работой.."
            # my_bar = st.progress(0, text=progress_text)

            with open(uploaded_file.name, mode='wb') as f:
                f.write(uploaded_file.read())

            stream = cv2.VideoCapture(uploaded_file.name, cv2.CAP_FFMPEG)

            st.session_state["stream"] = stream
            st.session_state["active"] = True
            # stream_length = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_display = st.empty()
            # stop_button = st.button(label="Остановить")

            grabbed = True
            while grabbed:
                grabbed, frame = stream.read()

                if not grabbed:
                    break
                
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                start_sm_time = timer()
                mask = sm(frame)
                finish_sm_time = timer()

                start_det_time = timer()
                detections = detector(frame, window_predict=False, tta_predict=False)
                detections = convert_yoloonnx_detections_to_united_list(detections)
                finish_det_tim = timer()
                detections = [(upscale_bbox(d[0], frame.shape), d[1], d[2]) for d in detections]

                print('Segm time: {:.3f}, det time: {:.3f}'.format(finish_sm_time - start_sm_time,
                                                                   finish_det_tim - start_det_time))

                vis_frame = plot_railway_masks(frame, mask)
                vis_frame = plot_detections(vis_frame, detections, mask)

                frame_display.image(Image.fromarray(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)))

                # if stop_inference:
            # print("STOP")
            # stream.release()
            # os.remove(uploaded_file.name)
            # grabbed = False