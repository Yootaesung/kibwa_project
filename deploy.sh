#!/bin/bash
set -e  # 오류 발생 시 스크립트 중단

# Docker 디렉토리 확인 및 생성
if [ ! -d "/work/docker" ]; then
  echo "--- Creating Docker working directory ---"
  sudo mkdir -p /work/docker
  sudo chmod 777 /work/docker
fi

# 환경 변수 확인
if [ -z "$DOCKER_HUB_USERNAME" ] || [ -z "$DOCKER_HUB_TOKEN" ]; then
  echo "Error: DOCKER_HUB_USERNAME and DOCKER_HUB_TOKEN must be set"
  exit 1
fi

# Docker Hub 로그인
echo "--- Logging in to Docker Hub ---"
echo $DOCKER_HUB_TOKEN | docker login -u $DOCKER_HUB_USERNAME --password-stdin

# Docker 디스크 공간 확보
echo "--- Freeing up disk space ---"
sudo docker system prune -af

# 최신 이미지 가져오기
echo "--- Pulling latest image ---"
docker pull $DOCKER_HUB_USERNAME/kibwa-chatbot:latest

# 기존 컨테이너 정지 및 제거
echo "--- Stopping and removing old container ---"
docker stop kibwa-chatbot 2>/dev/null || true
docker rm kibwa-chatbot 2>/dev/null || true

# 필요한 디렉토리 생성 및 권한 설정
echo "--- Creating necessary directories ---"
sudo mkdir -p /work/kibwa_project/chatbot/member_information
sudo mkdir -p /work/kibwa_project/chatbot/emotion_data
sudo mkdir -p /work/kibwa_project/chatbot/profanity_data
sudo mkdir -p /work/kibwa_project/chatbot/chat_logs

# 디렉토리 소유권 변경
sudo chown -R $USER:$USER /work/kibwa_project

# 환경 변수 설정
echo "--- Setting up environment variables ---"
cat > /work/kibwa_project/.env <<EOL
OPENAI_API_KEY=${OPENAI_API_KEY}
AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}
EOL

# Docker Compose로 새 컨테이너 실행
echo "--- Starting new container with Docker Compose ---"
cd /work/kibwa_project
docker-compose -f docker-compose.prod.yml up -d

# 컨테이너 상태 확인
echo "--- Container status ---"
docker ps | grep kibwa-chatbot || echo "Container is not running"

# 불필요한 이미지 정리
echo "--- Cleaning up ---"
docker system prune -f

echo "✅ Deployment completed successfully"
