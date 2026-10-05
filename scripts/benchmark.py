"""
Замер latency и FPS для ONNX FP32 и INT8-моделей.
"""
import os
import time
import numpy as np
import onnxruntime as ort


def benchmark(model_path: str, num_runs: int = 50, warmup: int = 5):
    if not os.path.exists(model_path):
        print(f"SKIP: {model_path} not found")
        return None

    providers = ort.get_available_providers()
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape

    h = input_shape[2] if isinstance(input_shape[2], int) else 640
    w = input_shape[3] if isinstance(input_shape[3], int) else 640
    dummy = np.random.rand(1, 3, h, w).astype(np.float32)

    for _ in range(warmup):
        session.run(None, {input_name: dummy})

    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000)

    avg = float(np.mean(latencies))
    p95 = float(np.percentile(latencies, 95))
    fps = 1000.0 / avg
    size_mb = os.path.getsize(model_path) / (1024 * 1024)

    print(f"\n=== {model_path} ===")
    print(f"Size:        {size_mb:.2f} MB")
    print(f"Avg latency: {avg:.2f} ms")
    print(f"P95 latency: {p95:.2f} ms")
    print(f"FPS:         {fps:.1f}")
    print(f"Providers:   {providers}")
    return {"model": model_path, "size_mb": size_mb, "avg_ms": avg, "p95_ms": p95, "fps": fps}


if __name__ == "__main__":
    results = []
    results.append(benchmark("models/yolov8n.onnx"))
    results.append(benchmark("models/yolov8n_int8.onnx"))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<30} {'Size (MB)':<12} {'Avg (ms)':<12} {'FPS':<8}")
    print("-" * 60)
    for r in results:
        if r:
            name = os.path.basename(r["model"])
            print(f"{name:<30} {r['size_mb']:<12.2f} {r['avg_ms']:<12.2f} {r['fps']:<8.1f}")