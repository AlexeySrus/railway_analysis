import os


ROOT_DIR: str = os.path.join(os.path.dirname(__file__), '../')
USE_TRT: bool = False
SEGM_MODEL_PATH = os.path.join(ROOT_DIR, 'weights/seg.onnx')
DETECTION_MODEL_PATH = os.path.join(ROOT_DIR, 'weights/yolov8l.onnx')
