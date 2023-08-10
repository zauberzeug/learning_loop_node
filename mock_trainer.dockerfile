FROM zauberzeug/nicegui:1.2.13


RUN apt-get update --allow-unauthenticated --allow-insecure-repositories && apt-get install -y jpeginfo && apt-get -y install python3-pip && apt-get clean && rm -rf /var/lib/apt/lists/*

# RUN curl -sSL https://gist.githubusercontent.com/b01/0a16b6645ab7921b0910603dfb85e4fb/raw/5186ea07a06eac28937fd914a9c8f9ce077a978e/download-vs-code-server.sh | bash

# ENV VSCODE_SERVER=/root/.vscode-server/bin/*/server.sh

# RUN $VSCODE_SERVER --install-extension ms-python.vscode-pylance \
#     $VSCODE_SERVER --install-extension ms-python.python \
#     $VSCODE_SERVER --install-extension himanoa.python-autopep8 \
#     $VSCODE_SERVER --install-extension esbenp.prettier-vscode \
#     $VSCODE_SERVER --install-extension littlefoxteam.vscode-python-test-adapter

WORKDIR /app/

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install --no-cache-dir "uvicorn[standard]"  tqdm async_generator aiofiles retry debugpy pytest-asyncio psutil icecream pytest "pytest-mock==3.6.1" autopep8 pynvml 
RUN python3 -m pip install --no-cache-dir "learning-loop-node==0.7.53rc2"

# while development this will be mounted but in deployment we need the latest code baked into the image
ADD ./learning_loop_node /usr/local/lib/python3.11/site-packages/learning_loop_node

ADD ./mock_trainer /app
ENV PYTHONPATH "${PYTHONPATH}:/app:/usr/local/lib/python3.11/site-packages"
ENV TZ=Europe/Amsterdam

EXPOSE 80
