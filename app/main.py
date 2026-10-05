import io
import time
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from app.detector import get_detector
from app.tracker import get_tracker
from app.schemas import DetectionResponse, Detection, HealthResponse
from app.config import settings
from app.logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI(
    title="Edge CV Tracking",
    description="YOLOv8n + ByteTrack on ONNX Runtime for Edge AI",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
async def health():
    import onnxruntime as ort
    return HealthResponse(
        status="healthy",
        model_path=settings.model_path,
        onnx_providers=ort.get_available_providers(),
    )


@app.post("/detect_image", response_model=DetectionResponse)
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    detector = get_detector()
    detections, latency = detector.detect(image)
    tracker = get_tracker()
    detections = tracker.update(detections)

    return DetectionResponse(
        detections=[Detection(**d) for d in detections],
        latency_ms=round(latency, 2),
        image_size=list(image.shape[:2]),
    )


@app.post("/detect_video")
async def detect_video(file: UploadFile = File(...)):
    """Принимает видео, возвращает видео с аннотациями (MJPEG stream)."""
    contents = await file.read()
    temp_path = "data/samples/temp_input.mp4"
    with open(temp_path, "wb") as f:
        f.write(contents)

    detector = get_detector()
    tracker = get_tracker()

    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Cannot open video")

    def frame_generator():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detections, _ = detector.detect(frame)
            detections = tracker.update(detections)

            for det in detections:
                x1, y1, x2, y2 = map(int, det["bbox"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det['class_name']}#{det.get('track_id', '?')} {det['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            _, jpeg = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

        cap.release()

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)