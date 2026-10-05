@echo off
echo Starting Edge CV Tracking UI...
python -m streamlit run app/ui.py --server.port 8501
pause