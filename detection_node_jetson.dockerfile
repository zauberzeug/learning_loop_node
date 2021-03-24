FROM nvcr.io/nvidia/l4t-base:r32.4.4

# Source: https://github.com/dusty-nv/jetson-containers/blob/master/Dockerfile.ml 
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV LLVM_CONFIG="/usr/bin/llvm-config-9"
ARG MAKEFLAGS=-j6

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

#
# apt packages
#
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip \
		python3-distutils \
		python3-dev \
        python3-setuptools \
        python3-matplotlib \
        build-essential \
        gfortran \
        git \
        cmake \
        curl \
        vim \
        gnupg \
        libopencv-dev \
        libopenblas-dev \
        liblapack-dev \
        libblas-dev \
        libhdf5-serial-dev \
        hdf5-tools \
        libhdf5-dev \
        zlib1g-dev \
        zip \
        libjpeg8-dev \
        libopenmpi2 \
        openmpi-bin \
        openmpi-common \
        protobuf-compiler \
        libprotoc-dev \
		llvm-9 \
        llvm-9-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN ln -sf /usr/bin/python3.6 /usr/bin/python3 && ln -sf /usr/bin/python3.6 /usr/bin/python

#
# OpenCV
#

RUN apt-key adv --fetch-key https://repo.download.nvidia.com/jetson/jetson-ota-public.asc
# COPY /etc/apt/trusted.gpg.d/jetson-ota-public.asc /etc/apt/trusted.gpg.d/jetson-ota-public.asc

RUN echo "deb https://repo.download.nvidia.com/jetson/common r32.4 main" > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list && \
    cat /etc/apt/sources.list.d/nvidia-l4t-apt-source.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
            libopencv-dev \
		  libopencv-python \
    && rm /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
    && rm -rf /var/lib/apt/lists/*

# copied (not 1:1) from https://github.com/tiangolo/uvicorn-gunicorn-docker/blob/master/docker-images/python3.8.dockerfile

RUN python3 -m pip install --no-cache-dir "uvicorn[standard]" gunicorn

COPY ./start.sh /start.sh
RUN chmod +x /start.sh

COPY ./gunicorn_conf.py /gunicorn_conf.py

COPY ./start-reload.sh /start-reload.sh
RUN chmod +x /start-reload.sh

# Run the start script, it will check for an /app/prestart.sh script (e.g. for migrations)
# And then will start Gunicorn with Uvicorn
CMD ["/start.sh"]


# We use Poetry for dependency management
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | POETRY_HOME=/opt/poetry python && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

# needed for opencv
RUN apt-get update && apt-get -y install libgl1-mesa-dev

WORKDIR /learning_loop_node/

COPY ./learning_loop_node/ ./

WORKDIR /app/

RUN ln -s /learning_loop_node /app/learning_loop_node && ls -lha learning_loop_node/

COPY ./detection_node/pyproject.toml ./detection_node/poetry.lock* ./

ENV LANG C.UTF-8
RUN python -m pip install --upgrade pip
RUN poetry update
#RUN poetry export -f requirements.txt --output requirements.txt 
#RUN python -m pip install -r requirements.txt
# Allow installing dev dependencies to run tests
ARG INSTALL_DEV=false
RUN bash -c "if [ $INSTALL_DEV == 'true' ] ; then poetry install --no-root ; else poetry install --no-root --no-dev ; fi"

COPY ./detection_node/ /app
ENV PYTHONPATH=/app

EXPOSE 80
