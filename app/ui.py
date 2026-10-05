import io
import os
import tempfile
import time
import urllib.request

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from app.detector import Detector, COCO_CLASSES
from app.tracker import SimpleTracker
from app.config import settings

TEST_VIDEO_URL = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4"
TEST_VIDEO_PATH = "data/samples/test_urban_highway.mp4"

st.set_page_config(
    page_title="Edge CV Tracking",
    layout="wide",
)

st.title("Edge CV Tracking")
st.markdown(
    "**YOLOv8n + ONNX Runtime** — детекция и трекинг объектов. Работает на CPU, оптимизировано для Edge-устройств."
)

with st.sidebar:
    st.header("Настройки")

    model_choice = st.selectbox(
        "Модель",
        ["models/yolov8n.onnx", "models/yolov8n_int8.onnx"],
        index=0,
        help="FP32 точнее, INT8 меньше по размеру",
    )

    conf_threshold = st.slider(
        "Порог уверенности",
        min_value=0.1,
        max_value=0.9,
        value=0.4,
        step=0.05,
    )

    iou_threshold = st.slider(
        "Порог IoU для NMS",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
    )

    enable_tracking = st.checkbox("Включить трекинг", value=True)

    st.markdown("---")
    st.markdown("### О проекте")
    st.markdown(
        "- Детекция: YOLOv8n, 80 классов COCO\n"
        "- Трекинг: IoU-matching\n"
        "- Инференс: ONNX Runtime (CPU)\n"
        "- Квантизация: INT8 dynamic"
    )


@st.cache_resource
def load_detector(model_path: str, conf: float, iou: float):
    settings.conf_threshold = conf
    settings.iou_threshold = iou
    return Detector(model_path=model_path)


@st.cache_resource
def load_tracker():
    return SimpleTracker()


def draw_detections(image: np.ndarray, detections: list) -> np.ndarray:
    img = image.copy()
    colors = {}
    for det in detections:
        cls_id = det["class_id"]
        if cls_id not in colors:
            np.random.seed(cls_id)
            colors[cls_id] = tuple(int(c) for c in np.random.randint(50, 255, 3))

        x1, y1, x2, y2 = map(int, det["bbox"])
        color = colors[cls_id]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"{det['class_name']}"
        if det.get("track_id") is not None:
            label += f" #{det['track_id']}"
        label += f" {det['confidence']:.2f}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            img, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
    return img


def download_test_video() -> str:
    os.makedirs("data/samples", exist_ok=True)
    if not os.path.exists(TEST_VIDEO_PATH):
        urllib.request.urlretrieve(TEST_VIDEO_URL, TEST_VIDEO_PATH)
    return TEST_VIDEO_PATH


tab1, tab2, tab3, tab4 = st.tabs(["Изображение", "Видео", "О модели", "О проекте"])

with tab1:
    st.header("Детекция объектов на изображении")

    uploaded = st.file_uploader(
        "Загрузите изображение (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        key="img_upload",
    )

    col1, col2 = st.columns(2)

    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with col1:
            st.subheader("Оригинал")
            st.image(image_rgb, use_container_width=True)

        with st.spinner("Detecting..."):
            detector = load_detector(model_choice, conf_threshold, iou_threshold)
            t0 = time.perf_counter()
            detections, latency = detector.detect(image)
            if enable_tracking:
                tracker = load_tracker()
                detections = tracker.update(detections)
            total_ms = (time.perf_counter() - t0) * 1000

        annotated = draw_detections(image, detections)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        with col2:
            st.subheader("Результат")
            st.image(annotated_rgb, use_container_width=True)

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Объектов найдено", len(detections))
        m2.metric("Latency (detect)", f"{latency:.1f} ms")
        m3.metric("Latency (total)", f"{total_ms:.1f} ms")
        m4.metric("FPS (approx)", f"{1000 / total_ms:.1f}")

        if detections:
            st.markdown("### Детали детекций")
            df = pd.DataFrame([
                {
                    "Класс": d["class_name"],
                    "Confidence": f"{d['confidence']:.3f}",
                    "Track ID": d.get("track_id", "—"),
                    "BBox (x1, y1, x2, y2)": ", ".join(f"{v:.0f}" for v in d["bbox"]),
                }
                for d in detections
            ])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Объектов не найдено. Попробуйте снизить порог confidence.")

        result_img = Image.fromarray(annotated_rgb)
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        st.download_button(
            "Скачать результат (PNG)",
            buf.getvalue(),
            file_name="detection_result.png",
            mime="image/png",
        )

    else:
        st.info("Загрузите изображение, чтобы увидеть детекцию.")
        st.markdown(
            "**Не знаете, где взять картинку?** Попробуйте классический тест YOLO: "
            "[bus.jpg](https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg)"
        )

with tab2:
    st.header("Детекция и трекинг на видео")
    st.caption("Обрабатывается покадрово на CPU. Для длинных видео это займёт время.")

    source = st.radio(
        "Источник видео",
        ["Загрузить своё", "Использовать тестовое видео"],
        horizontal=True,
    )

    video_path = None

    if source == "Загрузить своё":
        uploaded_video = st.file_uploader(
            "Загрузите видео (MP4, AVI, MOV)",
            type=["mp4", "avi", "mov"],
            key="video_upload",
        )
        if uploaded_video is not None:
            temp_path = os.path.join(tempfile.gettempdir(), "uploaded_video.mp4")
            with open(temp_path, "wb") as f:
                f.write(uploaded_video.read())
            video_path = temp_path
    else:
        if st.button("Загрузить тестовое видео"):
            with st.spinner("Скачивание тестового видео..."):
                video_path = download_test_video()
            st.success("Тестовое видео загружено")
        if os.path.exists(TEST_VIDEO_PATH):
            video_path = TEST_VIDEO_PATH

    if video_path is not None:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_src = cap.get(cv2.CAP_PROP_FPS)
        st.write(f"**Видео:** {total_frames} кадров, {fps_src:.1f} FPS источник")

        max_frames = st.slider(
            "Сколько кадров обработать",
            min_value=10,
            max_value=min(total_frames, 300),
            value=min(total_frames, 100),
            step=10,
        )

        if st.button("Запустить обработку"):
            detector = load_detector(model_choice, conf_threshold, iou_threshold)
            tracker = SimpleTracker()

            progress = st.progress(0)
            status = st.empty()
            preview = st.empty()

            out_frames = []
            total_ms = 0

            for i in range(max_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                t0 = time.perf_counter()
                detections, _ = detector.detect(frame)
                if enable_tracking:
                    detections = tracker.update(detections)
                total_ms += (time.perf_counter() - t0) * 1000

                annotated = draw_detections(frame, detections)
                out_frames.append(annotated)

                if i % 5 == 0:
                    preview.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        caption=f"Кадр {i + 1} / {max_frames}",
                        use_container_width=True,
                    )

                progress.progress((i + 1) / max_frames)
                status.text(f"Обработано {i + 1} / {max_frames} кадров")

            cap.release()

            avg_ms = total_ms / max(len(out_frames), 1)
            st.success(
                f"Готово! {len(out_frames)} кадров за {total_ms:.0f} ms "
                f"(~{avg_ms:.0f} ms/кадр, ~{1000 / avg_ms:.1f} FPS)"
            )

            if out_frames:
                h, w = out_frames[0].shape[:2]
                out_path = os.path.join(tempfile.gettempdir(), "output_video.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, min(fps_src, 15), (w, h))
                for f in out_frames:
                    writer.write(f)
                writer.release()

                with open(out_path, "rb") as f:
                    st.download_button(
                        "Скачать обработанное видео",
                        f.read(),
                        file_name="tracking_result.mp4",
                        mime="video/mp4",
                    )

with tab3:
    st.header("О модели и метриках")

    st.markdown("### Архитектура пайплайна")
    st.code(
        """
Изображение -> Препроцессинг (resize 640x640, normalize)
            -> ONNX Runtime Inference (YOLOv8n)
            -> Постпроцессинг (NMS, scale to original)
            -> Трекинг (IoU-matching)
            -> Аннотированное изображение + метрики
        """,
        language="text",
    )

    st.markdown("### Бенчмарки (AMD Ryzen 7 4800H, ONNX Runtime CPU)")
    bench_df = pd.DataFrame([
        {"Модель": "YOLOv8n FP32", "Размер (MB)": 12.23, "Latency (ms)": 62.07, "FPS": 16.1},
        {"Модель": "YOLOv8n INT8", "Размер (MB)": 3.33, "Latency (ms)": 100.53, "FPS": 9.9},
    ])
    st.dataframe(bench_df, use_container_width=True)

    st.info(
        "**Почему INT8 медленнее?** Dynamic quantization квантует только веса, "
        "активации остаются FP32. На CPU без VNNI-инструкций это даёт оверхед. "
        "Реальное ускорение INT8 требует TensorRT (Jetson) или Intel VNNI."
    )

    st.markdown("### 80 классов COCO")
    classes_df = pd.DataFrame({
        "ID": list(range(len(COCO_CLASSES))),
        "Класс": COCO_CLASSES,
    })
    st.dataframe(classes_df, use_container_width=True, height=300)

with tab4:
    st.header("О проекте")
    st.caption("Обоснование архитектурных решений и ответы на возможные вопросы.")

    with st.expander("Выбор модели детекции: YOLOv8n"):
        st.markdown("""
**Аналоги:** YOLOv5, YOLOv7, YOLOv9, YOLOv10, SSD, Faster R-CNN, EfficientDet.

**Почему YOLOv8n:**
- Ultralytics v8 — самая зрелая версия с поддержкой TensorRT 11 и INT8-квантизации.
- YOLOv8n (nano) — лучший баланс скорости и точности на edge: 64.08% mAP@0.5:0.95 при 12 ms latency на Raspberry Pi 4.
- Размер модели 6.2 MB (PyTorch), 12.2 MB (ONNX) — идеально для встраиваемых устройств.
- Встроенный трекинг (ByteTrack) и экспорт в ONNX из коробки.

**Почему не SSD или Faster R-CNN:** SSD менее точен на мелких объектах, Faster R-CNN слишком тяжёлый для edge. YOLOv9/10/11/12 новее, но с меньшим количеством production-кейсов.
""")

    with st.expander("Выбор среды исполнения: ONNX Runtime"):
        st.markdown("""
**Аналоги:** TensorFlow Lite, OpenVINO, TensorRT, PyTorch JIT.

**Почему ONNX Runtime:**
- Кроссплатформенный: работает на CPU, GPU, edge-устройствах без изменения кода.
- Поддерживает INT8-квантизацию через `onnxruntime.quantization`.
- Не требует привязки к конкретному фреймворку (PyTorch, TensorFlow).
- Доступен на Windows, Linux, macOS, ARM.

**Почему не TensorRT:** TensorRT требует NVIDIA GPU, недоступен на CPU. ONNX Runtime даёт переносимый код, который можно позже переключить на TensorRT через провайдеры.
""")

    with st.expander("Выбор квантизации: INT8 Dynamic"):
        st.markdown("""
**Аналоги:** FP16, INT8 Static, INT8 QAT.

**Почему INT8 Dynamic:**
- Не требует калибровочного датасета (в отличие от Static и QAT).
- Уменьшает размер модели в 3.7 раза (12.23 MB -> 3.33 MB).
- Прост в реализации: одна функция `quantize_dynamic`.

**Почему INT8 медленнее на CPU:** Dynamic quantization квантует только веса, активации остаются FP32. На каждом слое происходит dequant/quant. На CPU без VNNI (Intel Ice Lake+) это даёт оверхед. На ARM (Raspberry Pi) и NVIDIA (TensorRT) INT8 даёт реальное ускорение.
""")

    with st.expander("Выбор трекера: IoU-matching"):
        st.markdown("""
**Аналоги:** ByteTrack, DeepSORT, SORT, OC-SORT.

**Почему IoU-matching:**
- Не требует дополнительных зависимостей и моделей.
- Прост в реализации: 30 строк кода.
- Достаточен для коротких видео и демонстрации трекинга.

**Почему не ByteTrack:** ByteTrack требует интеграции с ultralytics и дополнительных настроек. Для production рекомендуется ByteTrack, но для демонстрации IoU-matching достаточен.
""")

    with st.expander("Выбор UI: Streamlit"):
        st.markdown("""
**Аналоги:** Gradio, FastAPI + HTML, Flask, Dash.

**Почему Streamlit:**
- Быстрый старт: 50 строк кода дают полноценный веб-интерфейс.
- Поддержка загрузки файлов, прогресс-баров, метрик, таблиц из коробки.
- Не требует знания HTML/CSS/JS.

**Почему не Gradio:** Gradio лучше для демонстрации ML-моделей, но менее гибок для многостраничных интерфейсов.
""")

    with st.expander("Ограничения и компромиссы"):
        st.markdown("""
**INT8 не ускоряет на CPU без VNNI.** Это не баг, а особенность архитектуры. Для реального edge-ускорения нужны TensorRT (Jetson), OpenVINO (Intel) или ARM Neon (Raspberry Pi).

**Трекер не использует нейросети.** IoU-matching не восстанавливает треки после перекрытий. Для production рекомендуется ByteTrack или DeepSORT.

**Нет GPU-инференса.** ONNX Runtime может использовать CUDA, но в текущей конфигурации провайдеры только CPU и Azure. Для GPU нужно установить `onnxruntime-gpu`.

**Нет мониторинга.** Метрики latency/FPS выводятся в UI, но не сохраняются. Для production нужен Prometheus или Grafana.
""")

    with st.expander("Что можно улучшить"):
        st.markdown("""
- Добавить ByteTrack из ultralytics для production-трекинга.
- Добавить TensorRT-экспорт для NVIDIA Jetson.
- Добавить OpenVINO для Intel CPU.
- Добавить мониторинг latency через Prometheus.
- Добавить поддержку RTSP-потоков для реального видео.
- Добавить CI/CD с автоматическим бенчмарком.
""")

st.markdown("---")
st.markdown(
    "Сделано на YOLOv8n + ONNX Runtime + Streamlit. "
    "GitHub: [NTsundere/edge-cv-tracking](https://github.com/NTsundere/edge-cv-tracking)"
)