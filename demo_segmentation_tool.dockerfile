FROM base_node:latest

ADD ./demo_annotation_tool /app
ENV PYTHONPATH "${PYTHONPATH}:/app:/usr/local/lib/python3.11/site-packages"
ENV TZ=Europe/Amsterdam

EXPOSE 80
