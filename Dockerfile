FROM ubuntu:22.04
LABEL authors="vicy chu"

SHELL ["/bin/bash", "-c"]

# Install the required packages
RUN apt-get update \
    && apt-get install -y build-essential wget ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install conda
RUN mkdir -p ~/miniconda3 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh \
    && bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 \
    && rm ~/miniconda3/miniconda.sh \
    && source ~/miniconda3/bin/activate

ENV PATH=~/miniconda3/bin:$PATH

# Install track_tool environment
RUN conda init bash \
    && source ~/miniconda3/etc/profile.d/conda.sh \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \ 
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \ 
    && conda create --name track_tool python=3.8 -y \
    && conda activate track_tool \
    && conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch \
    && pip install -U openmim \
    && mim install mmengine \
    && pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.13/index.html \
    && conda clean --all \
    && git clone https://github.com/open-mmlab/mmdetection.git /code/mmdetection \
    && cd /code/mmdetection \
    && pip install --no-cache-dir -v -e . -r requirements/tracking.txt \
    && pip install git+https://github.com/JonathonLuiten/TrackEval.git \
    && pip install lap

COPY ./object_detection/ /code/object_detection/

# Install object_detection environment
RUN conda init bash \
    && source ~/miniconda3/etc/profile.d/conda.sh \
    && conda create --name object_detection python=3.8 -y \
    && conda activate object_detection \
    && cd /code/object_detection \
    && pip install -v -e . \
    && pip uninstall mmcv-full mmcv -y \
    && pip install opencv-python-headless \
    && pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html \
    && pip install mmcv==1.5.3 \
    && pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html \
    && pip install timm flask flask_cors


# Install track_association environment
RUN conda init bash \
    && source ~/miniconda3/etc/profile.d/conda.sh \
    && conda create --name track_association python=3.10 -y \
    && conda activate track_association \
    && conda install numpy pandas tqdm