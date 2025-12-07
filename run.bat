@echo off
echo Starting RAG Chat API locally (Port 8000)...
uvicorn api.index:app --reload --port 8000
pause
