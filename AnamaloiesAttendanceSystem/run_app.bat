@echo off
cd /d "%~dp0"
SET PATH=%PATH%;C:\poppler\Release-23.11.0-0\Library\bin
python -m streamlit run attendance_backend_test.py
pause