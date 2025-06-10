@echo off
cd /d "%~dp0"
py -m uvicorn mythforge.main:app --host 0.0.0.0 --port 8000 --reload
