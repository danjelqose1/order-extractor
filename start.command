#!/bin/bash
cd "$(dirname "$0")"            # go to backend folder
source .venv/bin/activate       # activate venv
# load environment variables from .env
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi
uvicorn app:app --host 127.0.0.1 --port 5055 --reload
