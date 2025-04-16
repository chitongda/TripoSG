# Use the Python version mentioned in the environment setup
FROM python:3.10-slim
ARG CACHEBUST=$(date +%s)

WORKDIR /app

# Install system dependencies that might be needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    # Add any other system dependencies if required by your libraries
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install PyTorch (CPU version for now - see notes for GPU)
# Pin versions for reproducibility if needed
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
RUN pip install --no-cache-dir -r requirements.txt

# Install API dependencies
RUN pip install --no-cache-dir fastapi "uvicorn[standard]"

# Copy the model download script and run it
COPY download_models.py .
RUN python download_models.py

# Copy the rest of the application code
# Ensure necessary directories/files are copied (adjust as needed)
COPY triposg ./triposg
COPY scripts ./scripts
COPY assets ./assets # If default assets are needed by the API
# Add api.py once created
COPY api.py .

# Create the output directory referenced in the script
# Using a non-root directory for output is often better practice
# but matching the script's default for now.
# RUN mkdir ./output_dir # Not needed for the API which uses temp files

# Expose the default port for uvicorn
EXPOSE 7861

# Command to run the API (assuming api.py defines an 'app' instance)
# Replace 'api:app' with the correct module and app variable name
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7861"]

# Temporarily set a placeholder command until api.py exists
# CMD ["tail", "-f", "/dev/null"] 