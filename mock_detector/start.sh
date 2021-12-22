#!/usr/bin/env bash




if [[ $1 = "debug" ]]; then
    python3 -m debugpy --listen 5678 /app/main.py
elif [[ $1 = "profile" ]]; then
    kernprof -l /app/main.py
else
    uvicorn main:detector_node --host 0.0.0.0 --port 80 --reload --lifespan on --reload-dir /app/restart
fi