Edge CV Tracking

Production Edge AI pipeline: детекция объектов (YOLOv8n) и трекинг (IoU-matching) на ONNX Runtime, с интерактивным Streamlit UI.
Возможности

    Детекция объектов на 80 классах COCO (person, car, bus, traffic light и т.д.)

    Мульти-объектный трекинг с присвоением track_id

    ONNX Runtime inference (CPU)

    INT8 dynamic квантизация (3.7x уменьшение размера)

    FastAPI REST API: /detect_image, /detect_video, /health

    Streamlit UI для интерактивного тестирования

    Docker и docker-compose

    CI/CD через GitHub Actions

    Метрики: latency, FPS, размер модели

Бенчмарки

Тестирование на AMD Ryzen 7 4800H, ONNX Runtime CPU:
Модель	Размер	Avg latency	FPS
YOLOv8n FP32	12.23 MB	62 ms	16.1
YOLOv8n INT8 (dynamic)	3.33 MB	100 ms	9.9

INT8 dynamic quant даёт 3.7x меньше размер, но на CPU без VNNI-инструкций работает медленнее FP32 из-за dequant-накладных расходов. Реальное ускорение INT8 требует TensorRT (Jetson) или Intel VNNI (Ice Lake+).
Архитектура

Изображение или видеопоток поступает в FastAPI. Detector (ONNX Runtime) выполняет препроцессинг (resize 640x640, normalize), инференс YOLOv8n, постпроцессинг (NMS, scale to original). Tracker (IoU-matching) сопоставляет детекции между кадрами и присваивает track_id. Результат — аннотированное изображение и JSON с метриками.
Быстрый старт

Требования: Python 3.9+, зависимости из requirements.txt.
Установи зависимости
pip install -r requirements.txt

Скачай pretrained YOLOv8n
python scripts/download_dataset.py

Экспортируй в ONNX + INT8
python scripts/export_onnx.py

Запусти бенчмарк (опционально)
python scripts/benchmark.py

Запусти Streamlit UI
python -m streamlit run app/ui.py

Открой в браузере http://localhost:8501
Запусти FastAPI (опционально)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

Streamlit UI
Четыре вкладки:
Изображение — загрузка картинки, детекция с bounding boxes, таблица детекций, скачивание результата
Видео — загрузка своего видео или тестового (Intel IoT DevKit: люди, велосипеды, машины), покадровая обработка с трекингом, скачивание результата
О модели — схема пайплайна, бенчмарки FP32 vs INT8, 80 классов COCO
О проекте — обоснование архитектурных решений

В сайдбаре: выбор модели (FP32/INT8), слайдеры confidence и IoU, включение трекинга.
FastAPI
Health check
text

curl.exe http://localhost:8000/health

Ответ: {"status": "healthy", "model_path": "models/yolov8n.onnx", "onnx_providers": ["CPUExecutionProvider"]}
Детекция изображения
curl.exe -X POST http://localhost:8000/detect_image -F "file=@data/samples/bus.jpg"

Ответ для bus.jpg — 4 person и 1 bus с confidence 0.87-0.90 и присвоенными track_id.
Детекция видео
curl.exe -X POST http://localhost:8000/detect_video -F "file=@data/samples/test_video.mp4" -o output.mp4

Возвращает MJPEG-стрим с аннотированными кадрами.
Обоснование архитектурных решений
Выбор модели детекции: YOLOv8n

Аналоги: YOLOv5, YOLOv7, YOLOv9, YOLOv10, SSD, Faster R-CNN, EfficientDet.

Почему YOLOv8n:
Ultralytics v8 — самая зрелая версия с поддержкой TensorRT 11 и INT8-квантизации
YOLOv8n (nano) — лучший баланс скорости и точности на edge: 64.08% mAP@0.5:0.95 при 12 ms latency на Raspberry Pi 4
Размер модели 6.2 MB (PyTorch), 12.2 MB (ONNX) — идеально для встраиваемых устройств
Встроенный трекинг (ByteTrack) и экспорт в ONNX из коробки

Почему не SSD или Faster R-CNN: SSD менее точен на мелких объектах, Faster R-CNN слишком тяжёлый для edge. YOLOv9/10/11/12 новее, но с меньшим количеством production-кейсов.
Выбор среды исполнения: ONNX Runtime

Аналоги: TensorFlow Lite, OpenVINO, TensorRT, PyTorch JIT.

Почему ONNX Runtime:
Кроссплатформенный: работает на CPU, GPU, edge-устройствах без изменения кода
Поддерживает INT8-квантизацию через onnxruntime.quantization
Не требует привязки к конкретному фреймворку (PyTorch, TensorFlow)
Доступен на Windows, Linux, macOS, ARM

Почему не TensorRT: TensorRT требует NVIDIA GPU, недоступен на CPU. ONNX Runtime даёт переносимый код, который можно позже переключить на TensorRT через провайдеры.
Выбор квантизации: INT8 Dynamic

Аналоги: FP16, INT8 Static, INT8 QAT.

Почему INT8 Dynamic:
Не требует калибровочного датасета (в отличие от Static и QAT)
Уменьшает размер модели в 3.7 раза (12.23 MB -> 3.33 MB)
Прост в реализации: одна функция quantize_dynamic

Почему INT8 медленнее на CPU: Dynamic quantization квантует только веса, активации остаются FP32. На каждом слое происходит dequant/quant. На CPU без VNNI (Intel Ice Lake+) это даёт оверхед. На ARM (Raspberry Pi) и NVIDIA (TensorRT) INT8 даёт реальное ускорение.
Выбор трекера: IoU-matching

Аналоги: ByteTrack, DeepSORT, SORT, OC-SORT.
Почему IoU-matching:
Не требует дополнительных зависимостей и моделей
Прост в реализации: 30 строк кода
Достаточен для коротких видео и демонстрации трекинга

Почему не ByteTrack: ByteTrack требует интеграции с ultralytics и дополнительных настроек. Для production рекомендуется ByteTrack, но для демонстрации IoU-matching достаточен.
Выбор UI: Streamlit

Аналоги: Gradio, FastAPI + HTML, Flask, Dash.
Почему Streamlit:
Быстрый старт: 50 строк кода дают полноценный веб-интерфейс
Поддержка загрузки файлов, прогресс-баров, метрик, таблиц из коробки
Не требует знания HTML/CSS/JS

Почему не Gradio: Gradio лучше для демонстрации ML-моделей, но менее гибок для многостраничных интерфейсов.

Технологический стек
Слой	Технология
Детекция	YOLOv8n (Ultralytics)
Трекинг	IoU-matching
Инференс	ONNX Runtime
Квантизация	INT8 dynamic
API	FastAPI, Uvicorn
UI	Streamlit
Обработка	OpenCV, NumPy
Контейнеризация	Docker, docker-compose
CI/CD	GitHub Actions
Известные ограничения

INT8 не ускоряет на CPU без VNNI. Это не баг, а особенность архитектуры. Для реального edge-ускорения нужны TensorRT (Jetson), OpenVINO (Intel) или ARM Neon (Raspberry Pi).

Трекер не использует нейросети. IoU-matching не восстанавливает треки после перекрытий. Для production рекомендуется ByteTrack или DeepSORT.

Нет GPU-инференса. ONNX Runtime может использовать CUDA, но в текущей конфигурации провайдеры только CPU и Azure. Для GPU нужно установить onnxruntime-gpu.

Нет мониторинга. Метрики latency и FPS выводятся в UI, но не сохраняются. Для production нужен Prometheus или Grafana.
Что можно улучшить
Добавить ByteTrack из ultralytics для production-трекинга
Добавить TensorRT-экспорт для NVIDIA Jetson
Добавить OpenVINO для Intel CPU
Добавить мониторинг latency через Prometheus
Добавить поддержку RTSP-потоков для реального видео
Добавить CI/CD с автоматическим бенчмарком