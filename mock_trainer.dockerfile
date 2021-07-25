FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

RUN apt-get update && apt-get -y install python3-pip && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://gist.githubusercontent.com/b01/0a16b6645ab7921b0910603dfb85e4fb/raw/5186ea07a06eac28937fd914a9c8f9ce077a978e/download-vs-code-server.sh | bash

ENV VSCODE_SERVER=/root/.vscode-server/bin/*/server.sh

RUN $VSCODE_SERVER --install-extension ms-python.vscode-pylance \
    $VSCODE_SERVER --install-extension ms-python.python \
    $VSCODE_SERVER --install-extension himanoa.python-autopep8 \
    $VSCODE_SERVER --install-extension esbenp.prettier-vscode \
    $VSCODE_SERVER --install-extension littlefoxteam.vscode-python-test-adapter

# We use Poetry for dependency management
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | POETRY_HOME=/opt/poetry python && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

WORKDIR /app/

COPY ./mock_trainer/pyproject.toml ./mock_trainer/poetry.lock* ./

RUN python3 -m pip install --upgrade pip

RUN poetry config experimental.new-installer false

ENV PIP_USE_FEATURE=in-tree-build 

RUN poetry install --no-root

# while development this will be mounted but in deployment we need the latest code baked into the image
ADD ./learning_loop_node /usr/local/lib/python3.7/site-packages/learning_loop_node

COPY ./mock_trainer/ /app
ENV PYTHONPATH=/app

EXPOSE 80
