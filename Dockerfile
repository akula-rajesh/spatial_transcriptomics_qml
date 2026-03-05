# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3-pip \
    build-essential \
    curl \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python package manager
RUN curl -s https://bootstrap.pypa.io/get-pip.py | python3.9

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install quantum computing dependencies (optional)
RUN pip install pennylane || echo "Quantum computing dependencies installation failed (optional)"

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p data logs results configs

# Expose port for potential web interface (optional)
EXPOSE 8888

# Set up entry point
ENTRYPOINT ["python", "src/main.py"]

# Default command
CMD ["--help"]
