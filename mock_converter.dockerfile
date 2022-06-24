FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

RUN apt-get update && apt-get -y install python3-pip && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://gist.githubusercontent.com/b01/0a16b6645ab7921b0910603dfb85e4fb/raw/5186ea07a06eac28937fd914a9c8f9ce077a978e/download-vs-code-server.sh | bash

ENV VSCODE_SERVER=/root/.vscode-server/bin/*/server.sh

RUN $VSCODE_SERVER --install-extension ms-python.vscode-pylance \
    $VSCODE_SERVER --install-extension ms-python.python \
    $VSCODE_SERVER --install-extension himanoa.python-autopep8 \
    $VSCODE_SERVER --install-extension esbenp.prettier-vscode \
    $VSCODE_SERVER --install-extension littlefoxteam.vscode-python-test-adapter

WORKDIR /app/

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install --no-cache-dir "uvicorn[standard]"  tqdm debugpy icecream pytest autopep8
RUN python3 -m pip install --no-cache-dir "learning-loop-node==0.7.25"

# while development this will be mounted but in deployment we need the latest code baked into the image
ADD ./learning_loop_node /usr/local/lib/python3.7/site-packages/learning_loop_node

COPY ./mock_converter/ /app
ENV PYTHONPATH "${PYTHONPATH}:/app:/usr/local/lib/python3.7/site-packages"

EXPOSE 80
