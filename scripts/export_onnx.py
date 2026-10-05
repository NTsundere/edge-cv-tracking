"""
Экспорт YOLOv8n в ONNX + INT8 квантизация.
"""
import os
import shutil
from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType


def export_and_quantize():
    model_path = "yolov8n.pt"
    if os.path.exists("runs/detect/traffic_signs_v1/weights/best.pt"):
        model_path = "runs/detect/traffic_signs_v1/weights/best.pt"
        print(f"Using trained model: {model_path}")
    else:
        print(f"Using pretrained model: {model_path}")

    model = YOLO(model_path)

    onnx_path = model.export(format="onnx", imgsz=640, opset=17)
    print(f"ONNX exported: {onnx_path}")

    quantized_path = onnx_path.replace(".onnx", "_int8.onnx")
    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_path,
        weight_type=QuantType.QUInt8,
    )
    print(f"INT8 quantized: {quantized_path}")

    os.makedirs("models", exist_ok=True)
    shutil.copy(onnx_path, "models/yolov8n.onnx")
    shutil.copy(quantized_path, "models/yolov8n_int8.onnx")
    print("Models saved to models/")


if __name__ == "__main__":
    export_and_quantize()