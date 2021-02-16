FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7


# We use Poetry for dependency management
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | POETRY_HOME=/opt/poetry python && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

WORKDIR /learning_loop_node/

COPY ./learning_loop_node/ ./

RUN poetry update

WORKDIR /app/

RUN ln -s /learning_loop_node /app/learning_loop_node && ls -lha learning_loop_node/

COPY ./mock_trainer/pyproject.toml ./mock_trainer/poetry.lock* ./

RUN poetry update

# Allow installing dev dependencies to run tests
ARG INSTALL_DEV=false
RUN bash -c "if [ $INSTALL_DEV == 'true' ] ; then poetry install --no-root ; else poetry install --no-root --no-dev ; fi"

COPY ./mock_trainer/ /app
ENV PYTHONPATH=/app

EXPOSE 80