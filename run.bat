@echo off
taskkill /F /IM streamlit.exe >nul 2>&1
call venv\Scripts\activate
streamlit run app.py
pause
