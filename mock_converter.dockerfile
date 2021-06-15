FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

# We use Poetry for dependency management
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | POETRY_HOME=/opt/poetry python && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

ADD ./learning_loop_node /learning_loop_node
COPY pyproject.toml poetry.lock* README.md /

WORKDIR /app/

COPY ./mock_converter/pyproject.toml ./mock_converter/poetry.lock* ./

RUN python3 -m pip install --upgrade pip

RUN poetry config experimental.new-installer false

ENV PIP_USE_FEATURE=in-tree-build

# Allow installing dev dependencies to run tests
ARG INSTALL_DEV=false
RUN bash -c "if [ $INSTALL_DEV == 'true' ] ; then poetry install --no-root ; else poetry install --no-root --no-dev ; fi"

COPY ./mock_converter/ /app
ENV PYTHONPATH=/app

EXPOSE 80
