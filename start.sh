#!/bin/bash

# 设置模型目录
MODEL_DIR=${MODEL_DIR:-"/app/models"}
PRETRAINED_DIR="$MODEL_DIR/pretrained_weights"

# 确保目录存在
mkdir -p "$PRETRAINED_DIR"

# 检查并下载模型文件
if [ ! -d "$PRETRAINED_DIR/TripoSG" ] || [ ! -d "$PRETRAINED_DIR/RMBG-1.4" ]; then
    echo "Downloading model files to $PRETRAINED_DIR..."
    conda run -n triposg_env python download_models.py --output_dir "$MODEL_DIR" || {
        echo "[ERROR] Model download failed"
        exit 1
    }
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to download models"
        exit 1
    fi
    echo "Model download completed at $(date)"
    echo "TripoSG model: $PRETRAINED_DIR/TripoSG"
    echo "RMBG model: $PRETRAINED_DIR/RMBG-1.4"
else
    echo "Models already exist:"
    echo "TripoSG: $PRETRAINED_DIR/TripoSG"
    echo "RMBG: $PRETRAINED_DIR/RMBG-1.4"
fi

# 启动应用
echo "Starting TripoSG API server..."
conda run --no-capture-output -n triposg_env uvicorn api:app --host 0.0.0.0 --port 7860