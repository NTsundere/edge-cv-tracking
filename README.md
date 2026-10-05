Edge CV Tracking

Production Edge AI pipeline: детекция объектов (YOLOv8n) + трекинг (IoU/ByteTrack) на ONNX Runtime, с интерактивным Streamlit UI.
Возможности

    Детекция объектов на 80 классах COCO (person, car, bus, traffic light и т.д.)

    Мульти-объектный трекинг с присвоением track_id

    ONNX Runtime inference (CPU)

    INT8 dynamic квантизация (3.7x уменьшение размера)

    FastAPI REST API: /detect_image, /detect_video, /health

    Streamlit UI для интерактивного тестирования

    Docker + docker-compose

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

Установи зависимости:
pip install -r requirements.txt

Скачай pretrained YOLOv8n:
python scripts/download_dataset.py

Экспортируй в ONNX + INT8:
python scripts/export_onnx.py

Запусти бенчмарк:
python scripts/benchmark.py

Запусти интерактивный веб-интерфейс:
python -m streamlit run app/ui.py

Открой в браузере http://localhost:8501

Что можно делать в UI:

    Загружать изображения (JPG, PNG) и видеть детекции с bounding boxes в реальном времени

    Переключаться между FP32 и INT8 моделями

    Менять порог confidence и IoU через слайдеры

    Обрабатывать видео с трекингом объектов (превью + скачивание результата)

    Смотреть таблицу детекций: класс, confidence, track_id, координаты

    Скачивать аннотированные изображения

    Изучать бенчмарки и все 80 классов COCO

На Windows удобнее запускать через run_ui.bat.
FastAPI
Запуск API:
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

Проверка health:
curl.exe http://localhost:8000/health

Детекция изображения:
curl.exe -X POST http://localhost:8000/detect_image -F "file=@data/samples/bus.jpg"

Пример ответа для bus.jpg — 4 person + 1 bus с confidence 0.87-0.90 и присвоенными track_id.
Структура проекта
edge-cv-tracking/
├── app/
│   ├── main.py           # FastAPI эндпоинты
│   ├── ui.py             # Streamlit UI
│   ├── detector.py       # YOLOv8 ONNX инференс
│   ├── tracker.py        # IoU-трекер
│   ├── config.py         # Настройки
│   ├── schemas.py        # Pydantic-схемы
│   └── logger.py         # JSON-логирование
├── scripts/
│   ├── download_dataset.py
│   ├── train.py
│   ├── export_onnx.py
│   └── benchmark.py
├── models/               # ONNX-модели (не в git)
├── data/samples/         # Тестовые данные
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── run_ui.bat
└── README.md

Технологический стек

Детекция: YOLOv8n (Ultralytics). Трекинг: IoU-matching. Инференс: ONNX Runtime. Квантизация: INT8 dynamic. API: FastAPI, Uvicorn. UI: Streamlit. Обработка: OpenCV, NumPy. Контейнеризация: Docker. CI/CD: GitHub Actions.
Известные ограничения

INT8 dynamic quantization не даёт ускорения на CPU без VNNI. Для реального edge-ускорения нужны TensorRT (NVIDIA Jetson), OpenVINO (Intel) или ARM Neon (Raspberry Pi). Текущий трекер — упрощённый IoU-matching, для production рекомендуется ByteTrack из ultralytics.