FROM zauberzeug/nicegui:1.2.13

RUN apt-get update && \
    apt-get install -y jpeginfo \
    python3-pip \
    libjpeg-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app/

RUN python3 -m pip install --upgrade pip

# Install system packages required by Pillow
# TODO upgrade to pillow >=10 in pyproject.toml
RUN apt-get update && \
    apt-get install -y \
    libjpeg-dev\
    && rm -rf /var/lib/apt/lists/*
    
RUN python3 -m pip install --no-cache-dir "uvicorn[standard]" tqdm numpy async_generator aiofiles retry debugpy pytest-asyncio psutil icecream psutil pytest autopep8 pynvml
RUN python3 -m pip install --no-cache-dir "learning-loop-node==0.7.53rc2"

RUN apt-get update && apt-get -y install libgl1
RUN python3 -m pip install --no-cache-dir opencv-python

ADD ./mock_annotation_node /app
ENV PYTHONPATH "${PYTHONPATH}:/app:/usr/local/lib/python3.11/site-packages"
ENV TZ=Europe/Amsterdam

EXPOSE 80
