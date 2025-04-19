# 定义构建参数，指定基础镜像名称，默认值为nvidia/cuda
ARG IMAGE_NAME=nvidia/cuda
# 基于指定的基础镜像创建构建阶段，使用nvidia/cuda:12.8.1-runtime-ubuntu24.04作为基础镜像，并命名为base
FROM ${IMAGE_NAME}:12.8.1-runtime-ubuntu24.04 AS base

# 设置环境变量，指定NV_CUDA_LIB_VERSION，用于后续安装相关CUDA库
ENV NV_CUDA_LIB_VERSION="12.8.1-1"

# 从base阶段的镜像创建一个新的构建阶段base-amd64，专门用于amd64架构的构建
FROM base AS base-amd64

# 设置一系列与CUDA开发相关的环境变量，用于指定amd64架构下各组件的版本信息
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

# 选择base-amd64作为最终的基础镜像
FROM base-amd64

# 为镜像添加元数据，标明维护者信息
LABEL maintainer="NVIDIA CORPORATION <cudatools@nvidia.com>"

# 更新APT包列表，并安装一系列与CUDA开发相关的软件包，版本由之前设置的环境变量指定
# 安装完成后删除APT缓存以减小镜像体积
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

# 安装构建工具，更新APT包列表后进行安装
# 安装完成后删除APT缓存以减小镜像体积
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    cmake \
    pkg-config \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# 锁定cublas和nccl相关软件包的版本，防止APT自动升级这些包
RUN apt-mark hold ${NV_LIBCUBLAS_DEV_PACKAGE_NAME} ${NV_LIBNCCL_DEV_PACKAGE_NAME} ${NV_LIBCUBLAS_PACKAGE_NAME} ${NV_LIBNCCL_PACKAGE_NAME}

# 设置环境变量LIBRARY_PATH，指定库文件的搜索路径
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

# 将本地的nvidia_entrypoint.sh脚本复制到容器的/opt/nvidia/目录下
COPY nvidia_entrypoint.sh /opt/nvidia/
# 设置环境变量NVIDIA_PRODUCT_NAME，其值为"CUDA"
ENV NVIDIA_PRODUCT_NAME="CUDA"
# 指定容器启动时执行的入口脚本为/opt/nvidia/nvidia_entrypoint.sh
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]

# 设置环境变量PATH，将Miniconda的可执行文件路径添加到系统路径中
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# 安装Miniconda，指定下载链接为适用于x86_64架构的Miniconda安装脚本
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    mkdir -p /root/.conda && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm -f miniconda.sh

# 验证Miniconda是否安装成功
RUN conda --version

# 设置最终容器环境的PATH环境变量，将/opt/conda/bin添加到系统路径中
ENV PATH=/opt/conda/bin:$PATH

# 创建并激活名为triposg_env的Conda环境，指定Python版本为3.11
RUN . ~/.bashrc && \
    conda create -y --name triposg_env python=3.11 && \
    conda clean -afy

# 设置默认的shell为在triposg_env Conda环境中执行/bin/bash -c命令，用于后续的RUN命令
SHELL ["conda", "run", "-n", "triposg_env", "/bin/bash", "-c"]

# 更新Python的包管理工具pip和setuptools到最新版本
RUN python -m ensurepip --upgrade
RUN python -m pip install --upgrade setuptools

# 验证Conda环境是否正确激活，输出Conda环境信息和Python版本信息
RUN echo "Conda environment:" && \
    conda info --envs && \
    echo "Python version:" && \
    python --version

# 设置容器的工作目录为/app
WORKDIR /app

# 将主机上当前目录（构建上下文）的所有文件和目录复制到容器的/app目录中
COPY . .

# 设置环境变量CUDA_HOME，其值为/usr/local/cuda
# 安装PyTorch及其相关库，从指定的索引URL下载CUDA 12.1版本的包
ENV CUDA_HOME=/usr/local/cuda
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# 设置环境变量CPLUS_INCLUDE_PATH，其值为/usr/local/cuda/include
# 安装requirements.txt文件中列出的Python依赖
RUN CPLUS_INCLUDE_PATH=/usr/local/cuda/include pip install -r requirements.txt

# 安装额外的Python依赖，如FastAPI、Uvicorn、python - multipart和onnxruntime
RUN pip install fastapi uvicorn python-multipart onnxruntime

# 执行download_models.py脚本，下载应用所需的模型文件
RUN python download_models.py

# 声明容器将监听7860端口，用于外部访问容器内运行的应用
EXPOSE 7860

# 设置容器启动时默认执行的命令，使用Uvicorn运行名为api:app的FastAPI应用，监听在0.0.0.0地址的7860端口
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
