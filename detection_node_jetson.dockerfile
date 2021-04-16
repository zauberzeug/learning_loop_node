FROM zauberzeug/l4t-tkdnn-darknet:nano-r32.4.4

# adapted from https://github.com/tiangolo/uvicorn-gunicorn-docker/blob/master/docker-images/python3.8.dockerfile

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
