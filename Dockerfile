ARG IMAGE_NAME=nvidia/cuda
FROM ${IMAGE_NAME}:12.8.1-runtime-ubuntu24.04 AS base

ENV NV_CUDA_LIB_VERSION="12.8.1-1"

FROM base AS base-amd64

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

FROM base AS base-arm64

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

FROM base-${TARGETARCH}

ARG TARGETARCH

LABEL maintainer="NVIDIA CORPORATION <cudatools@nvidia.com>"

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

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    cmake \
    pkg-config \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Keep apt from auto upgrading the cublas and nccl packages. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
RUN apt-mark hold ${NV_LIBCUBLAS_DEV_PACKAGE_NAME} ${NV_LIBNCCL_DEV_PACKAGE_NAME} ${NV_LIBCUBLAS_PACKAGE_NAME} ${NV_LIBNCCL_PACKAGE_NAME}

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

# Add entrypoint items
COPY nvidia_entrypoint.sh /opt/nvidia/
ENV NVIDIA_PRODUCT_NAME="CUDA"
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

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

RUN conda --version

# Set PATH environment variable for the final container environment
ENV PATH=/opt/conda/bin:$PATH

# Create and activate Conda environment
# Need to source bashrc here to make conda command available after init
RUN . ~/.bashrc && \
    conda create -y --name triposg_env python=3.11 && \
    conda clean -afy

# Set the default shell to bash and activate the conda environment for subsequent RUN commands
SHELL ["conda", "run", "-n", "triposg_env", "/bin/bash", "-c"]

# Update pip and setuptools first
RUN python -m ensurepip --upgrade
RUN python -m pip install --upgrade setuptools

# Verify conda environment activation
RUN echo "Conda environment:" && \
    conda info --envs && \
    echo "Python version:" && \
    python --version

# Set working directory
WORKDIR /app

# Copy the rest of the application code
COPY . .

# Install Python dependencies
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
RUN pip install -r requirements.txt

# Install additional dependencies
RUN pip install fastapi uvicorn python-multipart onnxruntime

# Download models
RUN python download_models.py

# Expose the port the app runs on
EXPOSE 7860

# Run the API - The CMD will be passed as arguments to the ENTRYPOINT script
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]




