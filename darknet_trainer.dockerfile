FROM nvidia/cuda:11.0.3-devel-ubuntu20.04

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update
RUN apt-get install -y curl python3.8 python3-distutils python3-pip git-all  
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
WORKDIR /
RUN git clone https://github.com/AlexeyAB/darknet.git darknet && cd darknet && git checkout 64efa721ede91cd8ccc18257f98eeba43b73a6af 
RUN cd darknet && make clean && make


# We use Poetry for dependency management
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | POETRY_HOME=/opt/poetry python && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

WORKDIR /learning_loop_node/

COPY ./learning_loop_node/ ./

WORKDIR /app/

RUN ln -s /learning_loop_node /app/learning_loop_node && ls -lha learning_loop_node/

COPY ./darknet_trainer/pyproject.toml ./darknet_trainer/poetry.lock* ./

RUN poetry update

# Allow installing dev dependencies to run tests
ARG INSTALL_DEV=false
RUN bash -c "if [ $INSTALL_DEV == 'true' ] ; then poetry install --no-root ; else poetry install --no-root --no-dev ; fi"

COPY ./darknet_trainer/ /app
ENV PYTHONPATH=/app

EXPOSE 80




CMD mkdir -p /data

