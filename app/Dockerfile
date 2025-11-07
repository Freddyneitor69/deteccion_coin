# Jetson Nano (JetPack 4.6.x / L4T r32.7.1)
ARG L4T_TAG=r32.7.1
FROM nvcr.io/nvidia/l4t-ml:${L4T_TAG}-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# Dependencias para OpenCV con GUI (X11/GTK) + GStreamer/FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev python3-setuptools \
    libgl1-mesa-glx libglib2.0-0 \
    libgtk2.0-0 libsm6 libxext6 libxrender1 \
    gstreamer1.0-tools gstreamer1.0-libav \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Requisitos Python
COPY requirements.txt requirements.txt
RUN python3 -m pip install --upgrade pip && pip3 install -r requirements.txt

# Código (los modelos se montan por volumen)
COPY entrypoint.sh export_engine.py run_infer.py ./
RUN chmod +x /workspace/entrypoint.sh

# Variables por defecto (se pueden sobreescribir en compose)
ENV INFER_SOURCE=0 \
    IMG_SIZE=640 \
    CONF_THRES=0.25 \
    HALF=1 \
    USE_TRT=0 \
    SHOW=1

ENTRYPOINT ["bash", "/workspace/entrypoint.sh"]
