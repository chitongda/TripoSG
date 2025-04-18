ARG IMAGE_NAME=nvidia/cuda
FROM ${IMAGE_NAME}:12.8.1-base-ubuntu24.04 AS base

ENV NV_CUDA_LIB_VERSION=12.8.1-1

FROM base AS base-amd64

ENV NV_NVTX_VERSION=12.8.90-1
ENV NV_LIBNPP_VERSION=12.3.3.100-1
ENV NV_LIBNPP_PACKAGE=libnpp-12-8=${NV_LIBNPP_VERSION}
ENV NV_LIBCUSPARSE_VERSION=12.5.8.93-1

ENV NV_LIBCUBLAS_PACKAGE_NAME=libcublas-12-8
ENV NV_LIBCUBLAS_VERSION=12.8.4.1-1
ENV NV_LIBCUBLAS_PACKAGE=${NV_LIBCUBLAS_PACKAGE_NAME}=${NV_LIBCUBLAS_VERSION}

ENV NV_LIBNCCL_PACKAGE_NAME=libnccl2
ENV NV_LIBNCCL_PACKAGE_VERSION=2.25.1-1
ENV NCCL_VERSION=2.25.1-1
ENV NV_LIBNCCL_PACKAGE=${NV_LIBNCCL_PACKAGE_NAME}=${NV_LIBNCCL_PACKAGE_VERSION}+cuda12.8

FROM base AS base-arm64

ENV NV_NVTX_VERSION=12.8.90-1
ENV NV_LIBNPP_VERSION=12.3.3.100-1
ENV NV_LIBNPP_PACKAGE=libnpp-12-8=${NV_LIBNPP_VERSION}
ENV NV_LIBCUSPARSE_VERSION=12.5.8.93-1

ENV NV_LIBCUBLAS_PACKAGE_NAME=libcublas-12-8
ENV NV_LIBCUBLAS_VERSION=12.8.4.1-1
ENV NV_LIBCUBLAS_PACKAGE=${NV_LIBCUBLAS_PACKAGE_NAME}=${NV_LIBCUBLAS_VERSION}

ENV NV_LIBNCCL_PACKAGE_NAME=libnccl2
ENV NV_LIBNCCL_PACKAGE_VERSION=2.25.1-1
ENV NCCL_VERSION=2.25.1-1
ENV NV_LIBNCCL_PACKAGE=${NV_LIBNCCL_PACKAGE_NAME}=${NV_LIBNCCL_PACKAGE_VERSION}+cuda12.8

FROM base-${TARGETARCH}

ARG TARGETARCH

LABEL maintainer="NVIDIA CORPORATION <cudatools@nvidia.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-libraries-12-8=${NV_CUDA_LIB_VERSION} \
    ${NV_LIBNPP_PACKAGE} \
    cuda-nvtx-12-8=${NV_NVTX_VERSION} \
    libcusparse-12-8=${NV_LIBCUSPARSE_VERSION} \
    ${NV_LIBCUBLAS_PACKAGE} \
    ${NV_LIBNCCL_PACKAGE} \
    wget \
    coreutils \
    && rm -rf /var/lib/apt/lists/*

# Keep apt from auto upgrading the cublas and nccl packages. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
RUN apt-mark hold ${NV_LIBCUBLAS_PACKAGE_NAME} ${NV_LIBNCCL_PACKAGE_NAME}

# Add entrypoint items
COPY nvidia_entrypoint.sh /opt/nvidia/
ENV NVIDIA_PRODUCT_NAME="CUDA"
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]

# Install Miniconda
ARG MINICONDA_VERSION=latest
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -O miniconda.sh && \\
    /bin/bash miniconda.sh -b -p /opt/conda && \\
    rm miniconda.sh && \\
    # Add conda to PATH for subsequent RUN instructions in this build stage
    echo 'export PATH=/opt/conda/bin:$PATH' >> /etc/profile.d/conda.sh && \\
    # Initialize conda for bash shells (modifies ~/.bashrc)
    /opt/conda/bin/conda init bash && \\
    # Clean up installation artifacts
    /opt/conda/bin/conda clean -tipsy

# Set PATH environment variable for the final container environment
ENV PATH=/opt/conda/bin:$PATH

# Create and activate Conda environment
# Need to source bashrc here to make conda command available after init
RUN . ~/.bashrc && \\
    conda create -y --name triposg_env python=3.12 && \\
    conda clean -afy

# Set the default shell to bash and activate the conda environment for subsequent RUN commands
SHELL ["conda", "run", "-n", "triposg_env", "/bin/bash", "-c"]

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
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
RUN pip install -r requirements.txt

# Install additional dependencies
RUN pip install fastapi uvicorn python-multipart onnxruntime

# Download models
RUN python download_models.py

# Expose the port the app runs on
EXPOSE 7860

# Run the API - The CMD will be passed as arguments to the ENTRYPOINT script
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]




