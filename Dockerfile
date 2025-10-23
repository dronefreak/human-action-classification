FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml .

# Install package
RUN pip install --no-cache-dir -e ".[demo,train]"

# Copy source code
COPY src/ src/
COPY configs/ configs/
COPY examples/ examples/

# Expose ports for Gradio
EXPOSE 7860

# Default command
CMD ["hac-demo", "--port", "7860", "--share"]
