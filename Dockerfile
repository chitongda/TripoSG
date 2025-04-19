# 定义一个构建参数 IMAGE_NAME，默认值为 nvidia/cuda，用于指定基础镜像的名称
ARG IMAGE_NAME=nvidia/cuda
# 基于 IMAGE_NAME 变量指定的基础镜像（nvidia/cuda:12.8.1-runtime-ubuntu24.04）创建一个名为 base 的构建阶段
FROM ${IMAGE_NAME}:12.8.1-runtime-ubuntu24.04 AS base

# 设置环境变量 NV_CUDA_LIB_VERSION，其值为 "12.8.1-1"，可能用于后续安装与 CUDA 库相关的软件包
ENV NV_CUDA_LIB_VERSION="12.8.1-1"

# 从 base 阶段的镜像创建一个新的构建阶段 base-amd64，用于处理 amd64 架构相关的配置
FROM base AS base-amd64

# 设置与 CUDA 开发相关的一系列环境变量，这些变量用于指定 amd64 架构下各组件的版本信息
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

# 从 base 阶段的镜像创建一个新的构建阶段 base-arm64，用于处理 arm64 架构相关的配置
FROM base AS base-arm64

# 设置与 CUDA 开发相关的一系列环境变量，这些变量用于指定 arm64 架构下各组件的版本信息
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

ENV NV_LIBNCCL_PACKAGE_NAME=libnccl2
ENV NV_LIBNCCL_PACKAGE_VERSION=2.25.1-1
ENV NCCL_VERSION=2.25.1-1
ENV NV_LIBNCCL_PACKAGE=${NV_LIBNCCL_PACKAGE_NAME}=${NV_LIBNCCL_PACKAGE_VERSION}+cuda12.8
ENV NV_LIBNCCL_DEV_PACKAGE_NAME=libnccl-dev
ENV NV_LIBNCCL_DEV_PACKAGE_VERSION=2.25.1-1
ENV NV_LIBNCCL_DEV_PACKAGE=${NV_LIBNCCL_DEV_PACKAGE_NAME}=${NV_LIBNCCL_DEV_PACKAGE_VERSION}+cuda12.8

# 根据构建参数 TARGETARCH 的值，选择对应的架构特定的基础镜像（base-amd64 或 base-arm64）
FROM base-${TARGETARCH}

# 定义一个构建参数 TARGETARCH，用于指定目标架构（如 amd64 或 arm64）
ARG TARGETARCH

# 为镜像添加元数据，标明镜像的维护者为 NVIDIA CORPORATION 及其联系邮箱
LABEL maintainer="NVIDIA CORPORATION <cudatools@nvidia.com>"

# 更新 APT 包列表，并安装一系列与 CUDA 开发相关的软件包，版本由之前设置的环境变量指定
# 安装完成后删除 APT 缓存以减小镜像体积
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-dev-12-8=${NV_CUDA_CUDART_DEV_VERSION} \
    cuda-command-line-tools-12-8=${NV_CUDA_LIB_VERSION} \
    cuda-minimal-build-12-8=${NV_CUDA_LIB_VERSION} \
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

# 安装构建工具，如编译器、构建系统等，更新 APT 包列表后进行安装
# 安装完成后删除 APT 缓存以减小镜像体积
# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    cmake \
    pkg-config \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# 锁定 cublas 和 nccl 相关软件包的版本，防止 APT 自动升级这些包
# Keep apt from auto upgrading the cublas and nccl packages. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
RUN apt-mark hold ${NV_LIBCUBLAS_DEV_PACKAGE_NAME} ${NV_LIBNCCL_DEV_PACKAGE_NAME} ${NV_LIBCUBLAS_PACKAGE_NAME} ${NV_LIBNCCL_PACKAGE_NAME}

# 设置环境变量 LIBRARY_PATH，指定库文件的搜索路径
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

# 将本地的 nvidia_entrypoint.sh 脚本复制到容器的 /opt/nvidia/ 目录下，用于后续作为容器的入口脚本
# Add entrypoint items
COPY nvidia_entrypoint.sh /opt/nvidia/
# 设置环境变量 NVIDIA_PRODUCT_NAME，其值为 "CUDA"
ENV NVIDIA_PRODUCT_NAME="CUDA"
# 指定容器启动时执行的入口脚本为 /opt/nvidia/nvidia_entrypoint.sh
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]

# 设置环境变量 PATH，将 Miniconda 的可执行文件路径添加到系统路径中
ENV PATH="/root/miniconda3/bin:${PATH}"
# 定义一个构建参数 PATH，将 Miniconda 的可执行文件路径添加到系统路径中（与上面的环境变量设置类似）
ARG PATH="/root/miniconda3/bin:${PATH}"

# 安装 Miniconda，根据当前架构（x86_64 或 aarch64）下载对应的 Miniconda 安装脚本
# Install Miniconda
# Install Miniconda on x86 or ARM platforms
RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$arch" = "aarch64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
    echo "Unsupported architecture: $arch"; \
    exit 1; \
    fi && \
    wget $MINICONDA_URL -O miniconda.sh && \
    mkdir -p /root/.conda && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm -f miniconda.sh

# 验证 Miniconda 是否安装成功，执行 conda --version 命令查看版本信息
RUN conda --version

# 设置最终容器环境的 PATH 环境变量，将 /opt/conda/bin 添加到系统路径中
# Set PATH environment variable for the final container environment
ENV PATH=/opt/conda/bin:$PATH

# 创建并激活名为 triposg_env 的 Conda 环境，指定 Python 版本为 3.11
# 激活环境前需要加载 ~/.bashrc 文件，以确保 conda 命令在初始化后可用
# Create and activate Conda environment
# Need to source bashrc here to make conda command available after init
RUN . ~/.bashrc && \
    conda create -y --name triposg_env python=3.11 && \
    conda clean -afy

# 设置默认的 shell 为在 triposg_env Conda 环境中执行 /bin/bash -c 命令，用于后续的 RUN 命令
# Set the default shell to bash and activate the conda environment for subsequent RUN commands
SHELL ["conda", "run", "-n", "triposg_env", "/bin/bash", "-c"]

# 更新 Python 的包管理工具 pip 和 setuptools 到最新版本
# Update pip and setuptools first
RUN python -m ensurepip --upgrade
RUN python -m pip install --upgrade setuptools

# 验证 Conda 环境是否正确激活，输出 Conda 环境信息和 Python 版本信息
# Verify conda environment activation
RUN echo "Conda environment:" && \
    conda info --envs && \
    echo "Python version:" && \
    python --version

# 设置容器的工作目录为 /app，后续的操作将在该目录下进行
# Set working directory
WORKDIR /app

# 将主机上当前目录（构建上下文）的所有文件和目录复制到容器的 /app 目录中
# Copy the rest of the application code
COPY . .

# 设置环境变量 CUDA_HOME，其值为 /usr/local/cuda，可能用于后续与 CUDA 相关的操作
# 安装 PyTorch 及其相关库，从指定的索引 URL 下载 CUDA 12.1 版本的包
# Install Python dependencies
ENV CUDA_HOME=/usr/local/cuda
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# 设置环境变量 CPLUS_INCLUDE_PATH，其值为 /usr/local/cuda/include
# 安装 requirements.txt 文件中列出的 Python 依赖
RUN CPLUS_INCLUDE_PATH=/usr/local/cuda/include pip install -r requirements.txt

# 安装额外的 Python 依赖，如 FastAPI、Uvicorn、python-multipart 和 onnxruntime
# Install additional dependencies
RUN pip install fastapi uvicorn python-multipart onnxruntime

# 执行 download_models.py 脚本，下载应用所需的模型文件
# Copy and setup startup script
COPY start.sh /app/
RUN chmod +x /app/start.sh

# Model will be downloaded at container startup (see start.sh)

# 声明容器将监听 7860 端口，用于外部访问容器内运行的应用
# Expose the port the app runs on
EXPOSE 7860

# 设置容器启动时默认执行的命令，使用 Uvicorn 运行名为 api:app 的 FastAPI 应用，监听在 0.0.0.0 地址的 7860 端口
# Run the startup script
CMD ["/app/start.sh"]