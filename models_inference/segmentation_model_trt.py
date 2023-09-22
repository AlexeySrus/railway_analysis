import cv2
import threading
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import torch
import ctypes
from typing import List, Dict

from utils.inference_utils import preprocess_frame, postprocess_segmentation


class super_gradients_segmentation(object):
    def __init__(self, engine_file_path: str, size: int = 1024):
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []
        self.size = size
        context.set_binding_shape(0, (1, 3, size, size))

        for binding in engine:
            print('bingding:', binding, engine.get_binding_shape(binding))
            binding_idx = engine.get_binding_index(binding)
            size = trt.volume(context.get_binding_shape(binding_idx)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = 1

    def inference(self, image_raw):
        threading.Thread.__init__(self)
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess
        blob = cv2.dnn.blobFromImage(image_raw, scalefactor=1, size=(self.size, self.size), swapRB=False)
        # Copy input image to host buffer
        np.copyto(host_inputs[0], blob.ravel())
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)

        # Synchronize the stream
        stream.synchronize()
        self.ctx.pop()

        result_masks = host_outputs[0]
        return result_masks

    def __del__(self):
        self.ctx.pop()


class SegmentationTRTInference:
    def __init__(self, trt_weights: str, size: int = 1024, number_classes: int = 4):
        self.model = super_gradients_segmentation(trt_weights)
        self.size = size
        self.number_classes = number_classes

    def run(self, image: np.ndarray) -> np.ndarray:
        return self.model.inference(image)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        t_image, (sx, sy) = preprocess_frame(image)
        pred = self.run(t_image.squeeze(0).transpose(1, 2, 0))
        pred = pred.reshape((1, self.number_classes, self.size, self.size))
        res_mask = postprocess_segmentation(pred, (image.shape[0], image.shape[1]), (sx, sy))
        return res_mask
