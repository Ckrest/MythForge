@echo off
cd /d C:\Users\Ckrest\ZZZ-LLM-Server
py -m uvicorn MythForgeServer:app --host 0.0.0.0 --port 8000 --reload
