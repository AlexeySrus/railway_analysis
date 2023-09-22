import cv2
import threading
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import torch


class super_gradients_detection(object):
    def __init__(self, engine_file_path: str, image_size: int = 640):
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
        self.image_size = image_size
        context.set_binding_shape(0, (1, 3, image_size, image_size))

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
        blob = cv2.dnn.blobFromImage(image_raw, scalefactor=1, size=(self.image_size, self.image_size), swapRB=True).astype(np.float16)
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

        detections = host_outputs[0]
        return detections

    def __del__(self):
        self.ctx.pop()


class YoloTRTCaller:
    def __init__(self, trt_weights: str, size: int = 640, number_classes: int = 80):
        self.model = super_gradients_detection(trt_weights, size)
        self.size = size
        self.number_classes = number_classes

    def run(self, image: np.ndarray) -> np.ndarray:
        return self.model.inference(image)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        pred = self.run(image)
        return pred


if __name__ == '__main__':
    from utils.yolo_utils import non_max_suppression_v8

    w_path = 'weights/detection.pkl'
    model = YoloTRTCaller(w_path)
    imagep = '/home/alexey/Downloads/photo_2023-09-22_20-17-19.jpg'
    img = cv2.imread(imagep, cv2.IMREAD_COLOR)
    out = model(img)

    dets = non_max_suppression_v8(torch.from_numpy(out.reshape((1, 84, -1))))
    dets = dets[0]

    for det in dets:
        box = det[:4].numpy().astype(np.int32).tolist()
        print(box)
        img = cv2.rectangle(img, box[:2], box[2:], color=(50, 200, 20), thickness=5)

    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    cv2.imshow('Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()