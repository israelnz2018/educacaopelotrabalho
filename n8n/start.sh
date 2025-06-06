#!/bin/sh
echo "🔁 Iniciando servidor FastAPI..."
uvicorn main:app --host=0.0.0.0 --port=8000

