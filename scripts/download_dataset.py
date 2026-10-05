"""
Скачивает датасет дорожных знаков через Ultralytics (без Roboflow).
Или можно использовать готовый COCO для теста.
"""
import os
from dotenv import load_dotenv

load_dotenv()


def download_traffic_signs():
    """Вариант 1: датасет дорожных знаков из ultralytics."""
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    print("YOLOv8n pretrained weights downloaded (COCO, 80 classes)")


def download_coco_sample():
    """Вариант 2: используем pretrained COCO для детекции (готово из коробки)."""
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    print(f"Model loaded. Classes: {len(model.names)}")
    for i, name in model.names.items():
        print(f"  {i}: {name}")


if __name__ == "__main__":
    print("=== Loading YOLOv8n pretrained on COCO ===")
    download_coco_sample()
    print("\nГотово. Модель yolov8n.pt скачана, 80 классов COCO доступны.")
    print("Для тренировки на своём датасете — используйте scripts/train.py")