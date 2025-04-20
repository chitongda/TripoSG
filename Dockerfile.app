# 继承基础镜像
FROM cuda_miniconda_base:12.1.1-cudnn8-devel-ubuntu22.04

# 设置工作目录
WORKDIR /app

# 创建并激活 Conda 环境
RUN . ~/.bashrc && \
    conda create -y --name triposg_env python=3.10 && \
    conda clean -afy

# 设置默认 shell
SHELL ["conda", "run", "-n", "triposg_env", "/bin/bash", "-c"]

# 复制 wheel 文件
COPY torch-2.1.0+cu121-cp310-cp310-linux_x86_64.whl .
COPY torchvision-0.16.0+cu121-cp310-cp310-linux_x86_64.whl .
COPY torchaudio-2.1.0+cu121-cp310-cp310-linux_x86_64.whl .

# 安装 Python 核心依赖(PyTorch必须最先安装以提供CUDA环境)
ENV CUDA_HOME=/usr/local/cuda
RUN pip install torch-2.1.0+cu121-cp310-cp310-linux_x86_64.whl \
    torchvision-0.16.0+cu121-cp310-cp310-linux_x86_64.whl \
    torchaudio-2.1.0+cu121-cp310-cp310-linux_x86_64.whl && \
    rm *.whl

# 编译diso_src本地源码(需要在PyTorch之后，其他依赖之前)
WORKDIR /app/diso_src
RUN python setup.py build_ext --inplace && \
    pip install . && \
    python -c "import diso; print(f'diso模块已正确安装，版本: {diso.__version__}')"

# 返回工作目录并复制应用代码
WORKDIR /app
COPY . .

# 安装其他依赖
RUN CPLUS_INCLUDE_PATH=/usr/local/cuda/include pip install -r requirements.txt \
    fastapi uvicorn python-multipart onnxruntime

# 整体验证
RUN python -c "\
import torch; \
import diso; \
import fastapi; \
print(f'环境验证通过: PyTorch {torch.__version__}, diso {diso.__version__}, FastAPI {fastapi.__version__}')"

# 设置模型存储卷
VOLUME /app/models

# 暴露端口
EXPOSE 7860

# 设置启动脚本权限
RUN chmod +x start.sh

# 运行 API
CMD ["/bin/bash", "start.sh"]