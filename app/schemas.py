from typing import List, Optional
from pydantic import BaseModel


class Detection(BaseModel):
    bbox: List[float]  
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None


class DetectionResponse(BaseModel):
    detections: List[Detection]
    latency_ms: float
    image_size: List[int]


class HealthResponse(BaseModel):
    status: str
    model_path: str
    onnx_providers: List[str]