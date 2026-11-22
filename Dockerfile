# CPU/GPU image for the Gradio image demo and CLI.
# Build: docker build -t hac .
# Run:   docker run --gpus all -p 7860:7860 -v $PWD/weights:/app/weights hac \
#            hac-demo --model weights/best.pth
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app

# OpenCV / MediaPipe runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy package metadata and sources before installing (README is referenced
# by pyproject.toml, src/ is needed for the package itself)
COPY pyproject.toml ./
COPY .github/README.md .github/README.md
COPY src/ src/
COPY scripts/ scripts/
COPY examples/ examples/
COPY assets/ assets/

RUN pip install --no-cache-dir ".[demo]"

EXPOSE 7860

# Listens on 0.0.0.0:7860; add --share only if you want a public Gradio link
CMD ["hac-demo", "--port", "7860"]
