FROM base_node:latest

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip 
RUN python3 -m pip install opencv-python

COPY ./demo_segmentation_tool /app
ENV PYTHONPATH "${PYTHONPATH}:/app:/usr/local/lib/python3.11/site-packages"
ENV TZ=Europe/Amsterdam

EXPOSE 80
