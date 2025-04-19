#!/bin/bash
# Download models with retry mechanism
MAX_RETRIES=3
RETRY_DELAY=5

for ((i=1; i<=$MAX_RETRIES; i++)); do
    echo "Downloading models (attempt $i/$MAX_RETRIES)..."
    if python download_models.py; then
        echo "Model download completed successfully"
        break
    else
        echo "Model download failed, retrying in $RETRY_DELAY seconds..."
        sleep $RETRY_DELAY
    fi
done

# Check final download status
if [ $? -ne 0 ]; then
    echo "Error: Failed to download models after $MAX_RETRIES attempts"
    exit 1
fi

# Start API server
exec uvicorn api:app --host 0.0.0.0 --port 7860