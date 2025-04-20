# 定义构建参数
ARG IMAGE_NAME=nvidia/cuda

# 基础镜像阶段，使用新的基础镜像
FROM ${IMAGE_NAME}:12.1.1-cudnn8-devel-ubuntu22.04 AS base
ENV NV_CUDA_LIB_VERSION="12.1.1-1"
# 通用的 CUDA 相关环境变量，根据新基础镜像调整
ENV NV_CUDA_CUDART_DEV_VERSION=12.1.105-1
ENV NV_NVML_DEV_VERSION=12.1.105-1
ENV NV_NVTX_VERSION=12.1.105-1
ENV NV_LIBNPP_VERSION=12.1.0.40-1
ENV NV_LIBNPP_PACKAGE=libnpp-12-1=${NV_LIBNPP_VERSION}
ENV NV_LIBNPP_DEV_VERSION=12.1.0.40-1
ENV NV_LIBNPP_DEV_PACKAGE=libnpp-dev-12-1=${NV_LIBNPP_DEV_VERSION}
ENV NV_LIBCUSPARSE_VERSION=12.1.0.106-1
ENV NV_LIBCUSPARSE_DEV_VERSION=12.1.0.106-1
ENV NV_LIBCUBLAS_VERSION=12.1.3.1-1
ENV NV_LIBCUBLAS_PACKAGE_NAME=libcublas-12-1
ENV NV_LIBCUBLAS_PACKAGE=${NV_LIBCUBLAS_PACKAGE_NAME}=${NV_LIBCUBLAS_VERSION}
ENV NV_LIBCUBLAS_DEV_VERSION=12.1.3.1-1
ENV NV_LIBCUBLAS_DEV_PACKAGE_NAME=libcublas-dev-12-1
ENV NV_LIBCUBLAS_DEV_PACKAGE=${NV_LIBCUBLAS_DEV_PACKAGE_NAME}=${NV_LIBCUBLAS_DEV_VERSION}
ENV NV_CUDA_NSIGHT_COMPUTE_VERSION=12.1.1-1
ENV NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE=cuda-nsight-compute-12-1=${NV_CUDA_NSIGHT_COMPUTE_VERSION}
ENV NV_NVPROF_VERSION=12.1.105-1
ENV NV_NVPROF_DEV_PACKAGE=cuda-nvprof-12-1=${NV_NVPROF_VERSION}
ENV NV_LIBNCCL_PACKAGE_NAME=libnccl2
ENV NV_LIBNCCL_PACKAGE_VERSION=2.17.1-1
ENV NCCL_VERSION=2.17.1-1
ENV NV_LIBNCCL_PACKAGE=${NV_LIBNCCL_PACKAGE_NAME}=${NV_LIBNCCL_PACKAGE_VERSION}+cuda12.1
ENV NV_LIBNCCL_DEV_PACKAGE_NAME=libnccl-dev
ENV NV_LIBNCCL_DEV_PACKAGE_VERSION=2.17.1-1
ENV NV_LIBNCCL_DEV_PACKAGE=${NV_LIBNCCL_DEV_PACKAGE_NAME}=${NV_LIBNCCL_DEV_PACKAGE_VERSION}+cuda12.1

# 添加镜像标签
LABEL maintainer="NVIDIA CORPORATION <cudatools@nvidia.com>"

# 合并 APT 更新和安装操作，添加 git 到安装列表
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-dev-12-1=${NV_CUDA_CUDART_DEV_VERSION} \
    cuda-command-line-tools-12-1=${NV_CUDA_LIB_VERSION} \
    cuda-minimal-build-12-1=${NV_CUDA_LIB_VERSION} \
    cuda-libraries-dev-12-1=${NV_CUDA_LIB_VERSION} \
    cuda-nvml-dev-12-1=${NV_NVML_DEV_VERSION} \
    ${NV_NVPROF_DEV_PACKAGE} \
    ${NV_LIBNPP_DEV_PACKAGE} \
    ${NV_LIBNPP_PACKAGE} \
    libcusparse-dev-12-1=${NV_LIBCUSPARSE_DEV_VERSION} \
    libcusparse-12-1=${NV_LIBCUSPARSE_VERSION} \
    ${NV_LIBCUBLAS_DEV_PACKAGE} \
    ${NV_LIBCUBLAS_PACKAGE} \
    ${NV_LIBNCCL_DEV_PACKAGE} \
    ${NV_LIBNCCL_PACKAGE} \
    ${NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE} \
    cuda-nvtx-12-1=${NV_NVTX_VERSION} \
    wget \
    coreutils \
    build-essential \
    gfortran \
    cmake \
    pkg-config \
    ninja-build \
    git \
    && rm -rf /var/lib/apt/lists/*

# 锁定特定包的版本
RUN apt-mark hold ${NV_LIBCUBLAS_DEV_PACKAGE_NAME} ${NV_LIBNCCL_DEV_PACKAGE_NAME} ${NV_LIBCUBLAS_PACKAGE_NAME} ${NV_LIBNCCL_PACKAGE_NAME}

# 设置库路径
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

# 添加入口脚本
COPY nvidia_entrypoint.sh /opt/nvidia/
ENV NVIDIA_PRODUCT_NAME="CUDA"
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]

# 设置路径环境变量
ENV PATH="/root/miniconda3/bin:${PATH}"

# 安装 Miniconda
RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    else \
        echo "Unsupported architecture: $arch"; \
        exit 1; \
    fi && \
    wget $MINICONDA_URL -O miniconda.sh && \
    mkdir -p /root/.conda && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm -f miniconda.sh

# 验证 Conda 安装
RUN conda --version    

# 设置工作目录
WORKDIR /app

# 创建并激活 Conda 环境
RUN . ~/.bashrc && \
    conda create -y --name triposg_env python=3.10 && \
    conda clean -afy

# 设置默认 shell
SHELL ["conda", "run", "-n", "triposg_env", "/bin/bash", "-c"]

# 复制应用代码
COPY . .

# 安装 Python 依赖
ENV CUDA_HOME=/usr/local/cuda
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN CPLUS_INCLUDE_PATH=/usr/local/cuda/include pip install -r requirements.txt

# 安装额外依赖
RUN pip install fastapi uvicorn python-multipart onnxruntime

# 设置模型存储卷
VOLUME /app/models

# 暴露端口
EXPOSE 7860

# 运行 API
CMD ["/bin/bash", "start.sh"]
