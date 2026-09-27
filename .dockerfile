FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# requirements.txt pins Python 3.11 explicitly, but Ubuntu 24.04 ships 3.12 by
# default -> pull 3.11 from deadsnakes and bootstrap pip for it directly
# (there's no python3.11-pip apt package for a non-default interpreter).
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    gnupg \
    curl \
    ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-distutils \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# torch/torchvision wheels must match this image's CUDA build (see README.md).
# Override TORCH_INDEX_URL at build time (--build-arg) if your README specifies
# a different CUDA channel (e.g. cu121, cu124) or a CPU-only index.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124

RUN python3.11 -m pip install --break-system-packages \
    torch==2.5.1 torchvision==0.20.1 --index-url ${TORCH_INDEX_URL} \
    && python3.11 -m pip install --break-system-packages -r requirements.txt

# models/ and slugs_map.json are expected to sit next to the app in the build
# context (ocr_inference/), so this picks them up along with the rest of the code.
COPY . .

RUN mkdir -p /app/logs

CMD ["python3.11", "main.py"]
