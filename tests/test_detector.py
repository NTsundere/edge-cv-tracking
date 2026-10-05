import numpy as np
from app.detector import Detector


def test_detector_init():
    import os
    if not os.path.exists("models/yolov8n.onnx"):
        return
    det = Detector()
    assert det.session is not None
    assert det.input_name is not None


def test_preprocess_shape():
    import os
    if not os.path.exists("models/yolov8n.onnx"):
        return
    det = Detector()
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    tensor = det.preprocess(img)
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32