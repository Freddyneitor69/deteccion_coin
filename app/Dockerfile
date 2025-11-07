ARG L4T_TAG=r32.7.1
FROM nvcr.io/nvidia/l4t-ml:${L4T_TAG}-py3

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev python3-setuptools \
    libgl1-mesa-glx libglib2.0-0 \
    gstreamer1.0-tools gstreamer1.0-libav \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# deps
COPY requirements.txt requirements.txt
RUN python3 -m pip install --upgrade pip && pip3 install -r requirements.txt

# código y modelos
COPY entrypoint.sh export_engine.py run_infer.py ./
COPY models/ models/

RUN chmod +x /workspace/entrypoint.sh

ENV INFER_SOURCE=0 IMG_SIZE=640 CONF_THRES=0.25 HALF=1 USE_TRT=0

ENTRYPOINT ["bash", "/workspace/entrypoint.sh"]
