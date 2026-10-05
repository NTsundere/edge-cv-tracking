import time
import numpy as np
import onnxruntime as ort
import cv2
from typing import List, Dict, Any
from app.config import settings
from app.logger import setup_logger

logger = setup_logger(__name__)

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


class Detector:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or settings.model_path
        providers = ort.get_available_providers()
        logger.info(f"ONNX Runtime providers: {providers}")

        self.session = ort.InferenceSession(
            self.model_path,
            providers=providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [o.name for o in self.session.get_outputs()]
        logger.info(f"Model loaded: {self.model_path}, input: {self.input_shape}")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize + normalize + HWC -> CHW -> NCHW."""
        img = cv2.resize(image, (settings.img_size, settings.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return np.ascontiguousarray(img)

    def postprocess(self, output: np.ndarray, orig_shape: tuple) -> List[Dict[str, Any]]:
        """YOLOv8 output: [1, 84, 8400] -> detections."""
        output = output[0] 
        output = output.T   

        boxes = output[:, :4]      
        scores = output[:, 4:]    

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        mask = confidences > settings.conf_threshold
        boxes, scores, class_ids, confidences = boxes[mask], scores[mask], class_ids[mask], confidences[mask]

        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        h_orig, w_orig = orig_shape[:2]
        x1 = x1 * w_orig / settings.img_size
        y1 = y1 * h_orig / settings.img_size
        x2 = x2 * w_orig / settings.img_size
        y2 = y2 * h_orig / settings.img_size

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            confidences.tolist(),
            settings.conf_threshold,
            settings.iou_threshold,
        )

        if len(indices) == 0:
            return []

        indices = indices.flatten()
        detections = []
        for i in indices:
            detections.append({
                "bbox": [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
                "confidence": float(confidences[i]),
                "class_id": int(class_ids[i]),
                "class_name": COCO_CLASSES[int(class_ids[i])] if int(class_ids[i]) < len(COCO_CLASSES) else "unknown",
            })
        return detections

    def detect(self, image: np.ndarray) -> tuple:
        """Run inference. Returns (detections, latency_ms)."""
        start = time.perf_counter()
        input_tensor = self.preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        detections = self.postprocess(outputs[0], image.shape)
        latency = (time.perf_counter() - start) * 1000
        return detections, latency


_detector = None


def get_detector() -> Detector:
    global _detector
    if _detector is None:
        _detector = Detector()
    return _detector