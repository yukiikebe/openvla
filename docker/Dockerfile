FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=US/Pacific

# Install dependencies and command-line tools.
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    g++ \
    wget \
    bzip2 \
    git \
    vim \
    tmux \
    htop \
    git \
    zip \
    unzip \
    ca-certificates \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    patchelf \
    libglu1-mesa \
    libxext6 \
    libxtst6 \
    libxrender1 \
    libxi6 \
    libjpeg-dev \
    libpng-dev \
    libopenblas-dev \
    libopencv-dev \
    libyaml-dev \
    libavformat-dev \
    libavcodec-dev \
    libswscale-dev \
    libavutil-dev \
    libavfilter-dev \
    libavdevice-dev \
    libswresample-dev \
    less \
    groff \
    mpich \
    ninja-build

# Set timezone
RUN ln -sf /usr/share/zoneinfo/US/Pacific /etc/localtime

# Set CUDA_ROOT
RUN export CUDA_HOME="/usr/local/cuda"


RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN pip install -U pip
RUN pip install accelerate

RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev patchelf \
    xvfb libglfw3-dev libosmesa-dev python3-opengl

RUN pip install \
    hydra-core==1.2.0 \
    numpy==1.22.4 \
    wandb==0.13.1 \
    easydict==1.9 \
    transformers==4.21.1 \
    opencv-python==4.6.0.66 \
    robomimic==0.2.0 \
    einops==0.4.1 \
    thop==0.1.1-2209072238 \
    bddl==1.0.1 \
    future==0.18.2 \
    matplotlib==3.5.3 \
    cloudpickle==2.1.0 \
    gym==0.25.2 \
    numba>=0.49.1 \
    scipy>=1.2.3 \
    Pillow \
    pynput \
    termcolor

RUN mkdir -p /opt/mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /opt/mujoco \
    && rm mujoco.tar.gz

ENV MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/opt/mujoco/mujoco210/bin:${LD_LIBRARY_PATH}

ENV MUJOCO_GL=osmesa

# VERSION SENSITIVE!!
RUN pip install mujoco==2.3.2 Cython==3.0.0a10 tensorflow==2.15.0 torch==2.2.0
RUN pip install accelerate>=0.25.0 draccus==0.8.0 einops huggingface_hub json-numpy jsonlines matplotlib peft==0.11.1 protobuf rich sentencepiece==0.1.99 timm==0.9.10 tokenizers==0.19.1 torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 transformers==4.40.1 wandb tensorflow==2.15.0 tensorflow_datasets==4.9.3 tensorflow_graphics==2021.12.3

RUN rm -r /workspace
RUN git clone https://github.com/openvla/openvla.git /workspace
RUN pip install -e /workspace
RUN pip install -r /workspace/experiments/robot/libero/libero_requirements.txt

RUN git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /workspace/third_party/LIBERO
RUN pip install -e /workspace/third_party/LIBERO

RUN pip install torch==2.2.0 transformers==4.40.1

RUN pip install packaging ninja
RUN ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
RUN pip install "flash-attn==2.5.5" --no-build-isolation

RUN apt-get remove libcudnn8 -y --allow-change-held-packages
RUN ln -s /workspace/third_party/LIBERO /LIBERO

RUN python -m pip install uvicorn fastapi
WORKDIR /workspace
