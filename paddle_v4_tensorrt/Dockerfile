FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install TensorRT 8.6 (compatible with CUDA 11.8)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnvinfer8 \
    libnvinfer-plugin8 \
    libnvparsers8 \
    libnvonnxparsers8 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages — direct ONNX Runtime, no RapidOCR or PaddlePaddle
RUN pip3 install --no-cache-dir \
    "numpy<2" \
    onnxruntime-gpu==1.17.0 \
    pypdfium2>=4.30.0 \
    fastapi>=0.115.0 \
    uvicorn[standard]>=0.30.0 \
    opencv-python-headless \
    pyclipper \
    shapely \
    tensorrt==8.6.1 \
    psutil \
    gputil \
    loguru \
    pydantic \
    aiofiles

# CUDA library symlinks
RUN ln -sf /usr/lib/x86_64-linux-gnu/libcudnn.so.8 /usr/local/cuda/lib64/libcudnn.so && \
    ln -sf /usr/lib/x86_64-linux-gnu/libcudnn*.so* /usr/local/cuda/lib64/ || true && \
    ln -sf /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcublas.so.11 /usr/local/cuda/lib64/libcublas.so && \
    ln -sf /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcublasLt.so.11 /usr/local/cuda/lib64/libcublasLt.so && \
    ln -sf /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcudart.so.11.0 /usr/local/cuda/lib64/libcudart.so

# Copy models and server code
COPY models/ ./models/
COPY server.py .

# Create TRT engine cache directory
RUN mkdir -p /app/trt_cache

# Environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
ENV CUDA_MODULE_LOADING=LAZY

# TensorRT engine cache (persisted via Docker volume)
ENV TRT_ENGINE_CACHE_PATH=/app/trt_cache

# Default configuration
ENV NUM_WORKERS=4
ENV DPI=100
ENV BUFFER_POOL_SIZE=50
ENV SERVER_PORT=8000
ENV INSTANCE_ID=v4_tensorrt

EXPOSE 8000

CMD ["python3", "server.py"]
