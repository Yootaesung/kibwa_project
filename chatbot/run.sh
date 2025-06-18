#!/bin/bash

# 앱 디렉토리로 이동
cd /app/chatbot

# 로그 디렉토리 생성
mkdir -p /app/logs

# 환경 변수 설정
export PYTHONPATH=/app

# 로컬 IP 주소 가져오기 (컨테이너 내부 IP)
LOCAL_IP=$(hostname -i)

echo "========================================"
echo "Starting FastAPI Chatbot on:"
echo "- http://localhost:8000"
echo "- http://${LOCAL_IP}:8000"
echo "========================================"

# Uvicorn으로 앱 실행 (프로덕션 환경에서는 --reload 제거)
exec uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
