#!/usr/bin/env bash

uvicorn main:trainer_node --host 0.0.0.0 --port 80 --reload --lifespan on --reload-dir /app