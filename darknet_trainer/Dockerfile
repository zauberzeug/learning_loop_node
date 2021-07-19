FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update
RUN apt-get install -y curl python3.8 python3-distutils python3-pip git-all vim build-essential libopencv-dev 
RUN ln -sf /usr/bin/python3.8 /usr/bin/python3 && ln -sf /usr/bin/python3.8 /usr/bin/python

# copied (not 1:1) from https://github.com/tiangolo/uvicorn-gunicorn-docker/blob/master/docker-images/python3.8.dockerfile

RUN python -m pip install --no-cache-dir "uvicorn[standard]" gunicorn

COPY ./start.sh /start.sh
RUN chmod +x /start.sh

COPY ./gunicorn_conf.py /gunicorn_conf.py

COPY ./start-reload.sh /start-reload.sh
RUN chmod +x /start-reload.sh

# Run the start script, it will check for an /app/prestart.sh script (e.g. for migrations)
# And then will start Gunicorn with Uvicorn
CMD ["/start.sh"]

# <--- end

# darknet
COPY conf.sh /tmp/
ARG CONFIG

WORKDIR /
RUN git clone https://github.com/zauberzeug/darknet_alexeyAB.git darknet && cd darknet && git checkout 211bb29e9988f6204a32cd38d0720d171135873d 
RUN cd darknet && chmod +x /tmp/conf.sh && /tmp/conf.sh $CONFIG && make clean && make


# We use Poetry for dependency management
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | POETRY_HOME=/opt/poetry python && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

ADD ./learning_loop_node /learning_loop_node
COPY pyproject.toml poetry.lock* README.md /

WORKDIR /app/
COPY ./darknet_trainer/pyproject.toml ./darknet_trainer/poetry.lock* ./

RUN python3 -m pip install --upgrade pip

RUN poetry update

RUN poetry config experimental.new-installer false

ENV PIP_USE_FEATURE=in-tree-build 

RUN poetry install --no-root


# Allow installing dev dependencies to run tests
ARG INSTALL_DEV=false
RUN bash -c "if [ $INSTALL_DEV == 'true' ] ; then poetry install --no-root ; else poetry install --no-root --no-dev ; fi"

COPY ./darknet_trainer/ /app
ENV PYTHONPATH=/app

EXPOSE 80




CMD mkdir -p /data

