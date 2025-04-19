# 定义构建参数，指定基础镜像名称，默认值为 nvidia/cuda
ARG IMAGE_NAME=nvidia/cuda
# 基于指定的基础镜像创建构建阶段，使用 nvidia/cuda:12.8.1-runtime-ubuntu24.04 作为基础镜像
FROM ${IMAGE_NAME}:12.8.1-runtime-ubuntu24.04

# 设置环境变量，指定 NV_CUDA_LIB_VERSION，用于后续安装相关 CUDA 库
ENV NV_CUDA_LIB_VERSION="12.8.1-1"

# 设置一些必要的 CUDA 相关环境变量，根据实际需求精简
ENV NV_CUDA_CUDART_DEV_VERSION=12.8.90-1
ENV NV_NVML_DEV_VERSION=12.8.90-1
ENV NV_NVTX_VERSION=12.8.90-1
ENV NV_LIBNPP_VERSION=12.3.3.100-1
ENV NV_LIBNPP_PACKAGE=libnpp-12-8=${NV_LIBNPP_VERSION}
ENV NV_LIBNPP_DEV_VERSION=12.3.3.100-1
ENV NV_LIBNPP_DEV_PACKAGE=libnpp-dev-12-8=${NV_LIBNPP_DEV_VERSION}
ENV NV_LIBCUSPARSE_VERSION=12.5.8.93-1
ENV NV_LIBCUSPARSE_DEV_VERSION=12.5.8.93-1
ENV NV_LIBCUBLAS_VERSION=12.8.4.1-1
ENV NV_LIBCUBLAS_PACKAGE_NAME=libcublas-12-8
ENV NV_LIBCUBLAS_PACKAGE=${NV_LIBCUBLAS_PACKAGE_NAME}=${NV_LIBCUBLAS_VERSION}
ENV NV_LIBCUBLAS_DEV_VERSION=12.8.4.1-1
ENV NV_LIBCUBLAS_DEV_PACKAGE_NAME=libcublas-dev-12-8
ENV NV_LIBCUBLAS_DEV_PACKAGE=${NV_LIBCUBLAS_DEV_PACKAGE_NAME}=${NV_LIBCUBLAS_DEV_VERSION}
ENV NV_CUDA_NSIGHT_COMPUTE_VERSION=12.8.1-1
ENV NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE=cuda-nsight-compute-12-8=${NV_CUDA_NSIGHT_COMPUTE_VERSION}
ENV NV_NVPROF_VERSION=12.8.90-1
ENV NV_NVPROF_DEV_PACKAGE=cuda-nvprof-12-8=${NV_NVPROF_VERSION}
ENV NV_LIBNCCL_PACKAGE_NAME=libnccl2
ENV NV_LIBNCCL_PACKAGE_VERSION=2.25.1-1
ENV NCCL_VERSION=2.25.1-1
ENV NV_LIBNCCL_PACKAGE=${NV_LIBNCCL_PACKAGE_NAME}=${NV_LIBNCCL_PACKAGE_VERSION}+cuda12.8
ENV NV_LIBNCCL_DEV_PACKAGE_NAME=libnccl-dev
ENV NV_LIBNCCL_DEV_PACKAGE_VERSION=2.25.1-1
ENV NV_LIBNCCL_DEV_PACKAGE=${NV_LIBNCCL_DEV_PACKAGE_NAME}=${NV_LIBNCCL_DEV_PACKAGE_VERSION}+cuda12.8

# 为镜像添加元数据，标明维护者信息
LABEL maintainer="NVIDIA CORPORATION <cudatools@nvidia.com>"

# 更新 APT 包列表，并安装一系列与 CUDA 开发相关的软件包，版本由之前设置的环境变量指定
# 安装完成后删除 APT 缓存以减小镜像体积
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-dev-12-8=${NV_CUDA_CUDART_DEV_VERSION} \
    cuda-command-line-tools-12-8=${NV_CUDA_LIB_VERSION} \
    cuda-libraries-dev-12-8=${NV_CUDA_LIB_VERSION} \
    cuda-nvml-dev-12-8=${NV_NVML_DEV_VERSION} \
    ${NV_NVPROF_DEV_PACKAGE} \
    ${NV_LIBNPP_DEV_PACKAGE} \
    ${NV_LIBNPP_PACKAGE} \
    libcusparse-dev-12-8=${NV_LIBCUSPARSE_DEV_VERSION} \
    libcusparse-12-8=${NV_LIBCUSPARSE_VERSION} \
    ${NV_LIBCUBLAS_DEV_PACKAGE} \
    ${NV_LIBCUBLAS_PACKAGE} \
    ${NV_LIBNCCL_DEV_PACKAGE} \
    ${NV_LIBNCCL_PACKAGE} \
    ${NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE} \
    cuda-nvtx-12-8=${NV_NVTX_VERSION} \
    wget \
    coreutils \
    && rm -rf /var/lib/apt/lists/*

# 安装必要的构建工具，根据实际需求精简
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# 锁定 cublas 和 nccl 相关软件包的版本，防止 APT 自动升级这些包
RUN apt-mark hold ${NV_LIBCUBLAS_DEV_PACKAGE_NAME} ${NV_LIBNCCL_DEV_PACKAGE_NAME} ${NV_LIBCUBLAS_PACKAGE_NAME} ${NV_LIBNCCL_PACKAGE_NAME}

# 设置环境变量 LIBRARY_PATH，指定库文件的搜索路径
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

# 使用包含 Miniconda 的基础镜像，简化 Miniconda 安装步骤
FROM continuumio/miniconda3:latest

# 设置工作目录
WORKDIR /app

# 将主机上当前目录（构建上下文）的所有文件和目录复制到容器的 /app 目录中
COPY . .

# 设置环境变量 CUDA_HOME，其值为 /usr/local/cuda
ENV CUDA_HOME=/usr/local/cuda

# 安装 PyTorch 及其相关库，从指定的索引 URL 下载 CUDA 12.1 版本的包
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 设置环境变量 CPLUS_INCLUDE_PATH，其值为 /usr/local/cuda/include
# 安装 requirements.txt 文件中列出的 Python 依赖
RUN CPLUS_INCLUDE_PATH=/usr/local/cuda/include pip install -r requirements.txt

# 安装额外的 Python 依赖，如 FastAPI、Uvicorn、python-multipart 和 onnxruntime
RUN pip install fastapi uvicorn python-multipart onnxruntime

# 执行 download_models.py 脚本，下载应用所需的模型文件
RUN python download_models.py

# 声明容器将监听 7860 端口，用于外部访问容器内运行的应用
EXPOSE 7860

# 设置容器启动时默认执行的命令，使用 Uvicorn 运行名为 api:app 的 FastAPI 应用，监听在 0.0.0.0 地址的 7860 端口
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
