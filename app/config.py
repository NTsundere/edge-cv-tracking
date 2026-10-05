from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    model_path: str = "models/yolov8n.onnx"
    conf_threshold: float = 0.4
    iou_threshold: float = 0.5
    img_size: int = 640

    tracker_config: str = "bytetrack.yaml"
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    track_buffer: int = 30


settings = Settings()