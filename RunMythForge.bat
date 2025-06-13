@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set MYTHFORGE_STDIN=1
py -m uvicorn mythforge.main:app --host 0.0.0.0 --port 8000 --reload
