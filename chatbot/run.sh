#!/bin/bash
cd /work/kibwa_project/chatbot

# Activate virtual environment
source /work/venv/bin/activate

# Get the local IP address
LOCAL_IP=$(hostname -I | awk '{print $1}')
echo "========================================"
echo "Starting server on:"
echo "- http://localhost:8000"
echo "- http://${LOCAL_IP}:8000"
echo "========================================"

# Run uvicorn using the full path from virtual environment
exec /work/venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 --reload
