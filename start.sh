#!/bin/bash
# Download models before starting the API
python download_models.py
exec uvicorn api:app --host 0.0.0.0 --port 7860