# AutoDock-GPU Service
# GPU-accelerated molecular docking using AutoDock-GPU
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Set working directory
WORKDIR /opt

# Install system dependencies
# python3-openbabel provides the Python bindings PLIP needs at import time.
# Without it, plip's setup.py tries to build openbabel==3.1.1.1 from source
# via SWIG, which fails in the Docker build (no swig/pkg-config + legacy
# StrictVersion bug). Installing the prebuilt apt package is ~5 seconds and
# lets PLIP's `try: import openbabel` succeed, skipping the source build.
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    cmake \
    python3-pip \
    python3-dev \
    libboost-all-dev \
    openbabel \
    python3-openbabel \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Build AutoDock-GPU from source (128 work items, optimal for A100)
RUN git clone https://github.com/ccsb-scripps/AutoDock-GPU.git && \
    cd AutoDock-GPU && \
    make DEVICE=GPU NUMWI=128 GPU_INCLUDE_PATH=/usr/local/cuda/include GPU_LIBRARY_PATH=/usr/local/cuda/lib64 && \
    cp bin/autodock_gpu_128wi /usr/local/bin/autodock_gpu && \
    chmod +x /usr/local/bin/autodock_gpu && \
    cd / && rm -rf /opt/AutoDock-GPU

# Install AutoGrid4 (required for grid map generation)
# Build from ccsb-scripps/AutoGrid repo (separate from AutoDock4)
RUN apt-get update && apt-get install -y autoconf automake libtool csh && rm -rf /var/lib/apt/lists/* && \
    git clone https://github.com/ccsb-scripps/AutoGrid.git && \
    cd AutoGrid && \
    autoreconf -i && \
    mkdir build && cd build && \
    ../configure && \
    make -j$(nproc) && \
    cp autogrid4 /usr/local/bin/ && \
    chmod +x /usr/local/bin/autogrid4 && \
    cd / && rm -rf /opt/AutoGrid

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install meeko for proper PDBQT conversion
RUN pip3 install meeko

# Install PLIP (Protein-Ligand Interaction Profiler) for binding pose analysis (Theo P1)
# Extracts H-bonds, hydrophobic contacts, salt bridges, pi-stacking, halogen bonds,
# water bridges, and metal coordination from protein-ligand complexes. Populates the
# `contacts` field on every docked pose so scientists can see WHY a pose scored well
# (not just that it did). Pure Python, MIT-licensed, no license issues.
RUN pip3 install plip

# Copy application code
COPY main.py .

# Environment variables
ENV PORT=8022
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8022/health || exit 1

# Expose port
EXPOSE 8022

# Run the application
CMD ["python3", "main.py"]
