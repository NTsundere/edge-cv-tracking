"""
Streamlit UI для Edge CV Tracking.
Запуск: streamlit run app/ui.py
"""
import io
import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from app.detector import Detector, COCO_CLASSES
from app.tracker import SimpleTracker
from app.config import settings


st.set_page_config(
    page_title="Edge CV Tracking",
    page_icon="🎯",
    layout="wide",
)

st.title("Edge CV Tracking")
st.markdown(
    "**YOLOv8n + ByteTrack на ONNX Runtime** — детекция и трекинг объектов в реальном времени. "
    "Работает на CPU, оптимизировано для Edge-устройств."
)

with st.sidebar:
    st.header("Настройки")

    model_choice = st.selectbox(
        "Модель",
        ["models/yolov8n.onnx", "models/yolov8n_int8.onnx"],
        index=0,
        help="FP32 — точнее, INT8 — меньше размер",
    )

    conf_threshold = st.slider(
        "Порог уверенности (confidence)",
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
        "- **Детекция**: YOLOv8n, 80 классов COCO\n"
        "- **Трекинг**: IoU-matching\n"
        "- **Инференс**: ONNX Runtime (CPU)\n"
        "- **Квантизация**: INT8 dynamic"
    )


@st.cache_resource
def load_detector(model_path: str, conf: float, iou: float):
    # Переопределяем пороги в settings
    settings.conf_threshold = conf
    settings.iou_threshold = iou
    detector = Detector(model_path=model_path)
    return detector


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


tab1, tab2, tab3 = st.tabs(["📷 Изображение", "🎬 Видео", "📊 О модели"])

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
            import pandas as pd
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
    st.caption("Обрабатывается покадрово на CPU — для длинных видео это займёт время.")

    uploaded_video = st.file_uploader(
        "Загрузите видео (MP4, AVI, MOV)",
        type=["mp4", "avi", "mov"],
        key="video_upload",
    )

    if uploaded_video is not None:
        import tempfile
        import os
        temp_path = os.path.join(tempfile.gettempdir(), "uploaded_video.mp4")
        with open(temp_path, "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_path)
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
                        "💾 Скачать обработанное видео",
                        f.read(),
                        file_name="tracking_result.mp4",
                        mime="video/mp4",
                    )

with tab3:
    st.header("О модели и метриках")

    st.markdown("### Архитектура пайплайна")
    st.code(
        """
Изображение → Препроцессинг (resize 640x640, normalize)
            → ONNX Runtime Inference (YOLOv8n)
            → Постпроцессинг (NMS, scale to original)
            → Трекинг (IoU-matching)
            → Аннотированное изображение + метрики
        """,
        language="text",
    )

    st.markdown("### Бенчмарки (AMD Ryzen 7 4800H, ONNX Runtime CPU)")
    import pandas as pd
    bench_df = pd.DataFrame([
        {"Модель": "YOLOv8n FP32", "Размер (MB)": 12.23, "Latency (ms)": 62.07, "FPS": 16.1},
        {"Модель": "YOLOv8n INT8", "Размер (MB)": 3.33, "Latency (ms)": 100.53, "FPS": 9.9},
    ])
    st.dataframe(bench_df, use_container_width=True)

    st.info(
        "**Почему INT8 медленнее?** Dynamic quantization квантует только веса, "
        "активации остаются FP32 — на каждом слое происходит dequant/quant. "
        "На CPU без VNNI-инструкций это даёт оверхед. Реальное ускорение INT8 "
        "требует TensorRT (Jetson) или Intel VNNI (Ice Lake+)."
    )

    st.markdown("### 80 классов COCO")
    classes_df = pd.DataFrame({
        "ID": list(range(len(COCO_CLASSES))),
        "Класс": COCO_CLASSES,
    })
    st.dataframe(classes_df, use_container_width=True, height=300)


st.markdown("---")
st.markdown(
    "Сделано на **YOLOv8n** + **ONNX Runtime** + **Streamlit**. "
    "GitHub: [NTsundere/edge-cv-tracking](https://github.com/NTsundere/edge-cv-tracking)"
)