#!/bin/bash

# 앱 디렉토리로 이동
cd ~/deploy/chatbot

# 가상 환경 활성화
source venv/bin/activate

# 로그 디렉토리 생성
mkdir -p logs

# 환경 변수 설정
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 로컬 IP 주소 가져오기
LOCAL_IP=$(hostname -I | awk '{print $1}')

echo "========================================"
echo "Starting FastAPI Chatbot on:"
echo "- http://localhost:8000"
echo "- http://${LOCAL_IP}:8000"
echo "========================================"

# Uvicorn으로 앱 실행 (프로덕션 환경에서는 --reload 제거)
exec uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
