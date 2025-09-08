FROM zauberzeug/nicegui:2.23.3

RUN apt-get update && \
    apt-get install -y \
    jpeginfo \
    python3-pip \
    libjpeg-dev \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app/

# delete everything in /app
RUN rm -rf /app/*

RUN python3 -m pip install --upgrade pip

# We use Poetry for dependency management (recommended way to install it)
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    cd /usr/local/bin && \
    ln -s ~/.local/bin/poetry && \
    poetry config virtualenvs.create false

COPY pyproject.toml ./

# Allow installing dev dependencies to run tests, can be disbaled by setting --build-arg INSTALL_DEV=false
ARG INSTALL_DEV=true
RUN echo "INSTALL_DEV is set to $INSTALL_DEV"
#RUN bash -c "if [ $INSTALL_DEV == 'true' ] ; then poetry install -vvv --no-root ; else poetry install -vvv --no-root --no-dev ; fi"
RUN if [ "$INSTALL_DEV" = 'true' ]; then poetry install -vvv --no-root; else poetry install -vvv --no-root --no-dev; fi


# while development this will be mounted but in deployment we need the latest code baked into the image
ADD ./learning_loop_node /usr/local/lib/python3.11/site-packages/learning_loop_node

# Overwrite the entrypoint to bash
ENTRYPOINT ["/bin/bash"]
