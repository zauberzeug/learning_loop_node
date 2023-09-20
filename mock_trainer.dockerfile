FROM base_node:latest

COPY ./mock_trainer /app
ENV PYTHONPATH "${PYTHONPATH}:/app:/usr/local/lib/python3.11/site-packages"
ENV TZ=Europe/Amsterdam

EXPOSE 80
