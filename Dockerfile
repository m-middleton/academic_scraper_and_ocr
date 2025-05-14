FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3.10 \
    python3-pip \
    python3.10-venv \
    ninja-build \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone the repository
RUN git clone https://github.com/Ucas-HaoranWei/GOT-OCR2.0.git

# The repository structure has a GOT-OCR-2.0-master subdirectory with the actual package
WORKDIR /app/GOT-OCR2.0/GOT-OCR-2.0-master

# Set up Python environment
RUN python3 -m pip install --upgrade pip
# Upgrade setuptools to a compatible version
RUN python3 -m pip install setuptools>=61.0.0 wheel

# Install build dependencies first
RUN pip install build ninja setuptools_scm toml packaging

# install latest version of transformers
RUN pip install transformers

# Install the package
RUN python3 -m pip install -e .

# Install Flash-Attention
RUN pip install ninja
RUN pip install flash-attn --no-build-isolation

# Install other dependencies
RUN pip install pdf2image

# Set environment variables
ENV PYTHONPATH=/app/GOT-OCR2.0/GOT-OCR-2.0-master:${PYTHONPATH}

# Create directory for weights
RUN mkdir -p /GOT_weights

# Default command
CMD ["python3", "GOT/demo/run_ocr_2.0.py", "--help"]
