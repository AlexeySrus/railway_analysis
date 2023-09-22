import base64
import time
import os
import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import numpy as np
import cv2
from PIL import Image

from models_inference.segmentation_model import preprocess_frame, SegmentationModelWrapper


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



def get_segmentation_model():
    return SegmentationModelWrapper("./data/seg.onnx")


if __name__ == "__main__":
    st.set_page_config(
        layout="wide",
        page_title="Цифровой прорыв/Безопасный маршрут"
        # initial_sidebar_state="expanded",
    )

    header = """
                <h1>
                Безопасный маршрут
                </h1>
                """

    st.markdown(header, unsafe_allow_html=True)
    st.caption("Разработано командой **RabotyagiTeam**🍆")

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
        
        st.warning("Внимание! Обработка занимает некоторое время. Пожалуйста, не перезагружайте страницу!", icon="🚨")

        # progress_text = "Нейронки за работой.."
        # my_bar = st.progress(0, text=progress_text)

        with open(uploaded_file.name, mode='wb') as f:
            f.write(uploaded_file.read())

        stream = cv2.VideoCapture(uploaded_file.name, cv2.CAP_FFMPEG)

        # stream_length = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_display = st.empty()
        stop_button = st.button(label="Остановить")

        grabbed = True
        while grabbed:
            grabbed, frame = stream.read()

            if not grabbed:
                break
            
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mask = sm(frame)
            vis_frame = plot_railway_masks(frame, mask)
        
            frame_display.image(Image.fromarray(vis_frame), channels="BGR")
            # time.sleep(0.01)
            if stop_button:
                grabbed = False
                os.remove(uploaded_file.name)