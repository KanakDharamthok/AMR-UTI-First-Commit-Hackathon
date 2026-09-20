@echo off
echo === Starting AMR-UTI Quantum Pipeline ===

:: 1. Start FastAPI Backend in a separate background window
cd backend
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate
echo Installing/Verifying requirements and Streamlit...
pip install -r requirements.txt --quiet
pip install streamlit --quiet

echo Starting FastAPI Server on Port 8000...
start cmd /k "venv\Scripts\activate && uvicorn api:app --reload --port 8000"

:: 2. Start Streamlit UI (running directly from backend/ or pointing to it)
echo Starting Streamlit UI on Port 8501...
python -m streamlit run app.py --server.port 8501
pause