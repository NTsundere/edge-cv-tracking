import numpy as np
from typing import List, Dict, Any
from app.config import settings
from app.logger import setup_logger

logger = setup_logger(__name__)

# Простой IoU-based трекер (замена ByteTrack без внешних зависимостей)
# Для production используй ultralytics YOLO.track() с bytetrack.yaml


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_age=30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = {}  # track_id -> {"bbox": ..., "age": ...}
        self.next_id = 1

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Match detections to existing tracks via IoU."""
        matched = set()
        for det in detections:
            best_iou = 0
            best_id = None
            for tid, track in self.tracks.items():
                if tid in matched:
                    continue
                score = iou(det["bbox"], track["bbox"])
                if score > best_iou and score > self.iou_threshold:
                    best_iou = score
                    best_id = tid

            if best_id is not None:
                det["track_id"] = best_id
                self.tracks[best_id]["bbox"] = det["bbox"]
                self.tracks[best_id]["age"] = 0
                matched.add(best_id)
            else:
                det["track_id"] = self.next_id
                self.tracks[self.next_id] = {"bbox": det["bbox"], "age": 0}
                self.next_id += 1

        # Increment age of unmatched tracks, remove old ones
        for tid in list(self.tracks.keys()):
            if tid not in matched:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]

        return detections


_tracker = None


def get_tracker() -> SimpleTracker:
    global _tracker
    if _tracker is None:
        _tracker = SimpleTracker()
    return _tracker