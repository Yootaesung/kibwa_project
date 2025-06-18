#!/bin/bash

# 환경 변수 확인
if [ -z "$DOCKER_HUB_USERNAME" ] || [ -z "$DOCKER_HUB_TOKEN" ]; then
  echo "Error: DOCKER_HUB_USERNAME and DOCKER_HUB_TOKEN must be set"
  exit 1
fi

# Docker Hub 로그인
echo $DOCKER_HUB_TOKEN | docker login -u $DOCKER_HUB_USERNAME --password-stdin

# 최신 이미지 가져오기
echo "--- Pulling latest image ---"
docker pull $DOCKER_HUB_USERNAME/kibwa-chatbot:latest || true

# 기존 컨테이너 정지 및 제거
echo "--- Stopping and removing old container ---"
docker stop kibwa-chatbot || true
docker rm kibwa-chatbot || true

# 필요한 디렉토리 생성
echo "--- Creating necessary directories ---"
mkdir -p /work/kibwa_project/chatbot/member_information
mkdir -p /work/kibwa_project/chatbot/emotion_data
mkdir -p /work/kibwa_project/chatbot/profanity_data
mkdir -p /work/kibwa_project/chatbot/chat_logs

# 환경 변수 설정
echo "--- Setting up environment variables ---"
cat > /work/kibwa_project/.env <<EOL
OPENAI_API_KEY=${OPENAI_API_KEY}
AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}
EOL

# 새 컨테이너 실행
echo "--- Starting new container ---"
cd /work/kibwa_project
docker-compose -f docker-compose.prod.yml up -d

# 불필요한 이미지 정리
echo "--- Cleaning up ---"
docker system prune -f

echo "--- Deployment completed successfully ---"
