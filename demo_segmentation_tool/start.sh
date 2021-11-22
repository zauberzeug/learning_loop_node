#!/usr/bin/env bash

if [[ $1 = "debug" ]]; then
   python3 -m debugpy --listen 5678 /app/main.py
elif [[ $1 = "profile" ]]; then
    kernprof -l /app/main.py
else
   python3 /app/main.py
fi