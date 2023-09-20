FROM base_node:latest

COPY ./mock_annotator /app
ENV PYTHONPATH "${PYTHONPATH}:/app:/usr/local/lib/python3.11/site-packages:/learning_loop_node/learning_loop_node"
ENV TZ=Europe/Amsterdam

EXPOSE 80
