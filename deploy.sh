#!/bin/bash
set -e

# Docker 데몬이 사용할 디렉토리 설정
sudo mkdir -p /var/lib/docker
sudo chmod 711 /var/lib/docker

# 작업 디렉토리 생성
sudo mkdir -p /work/kibwa_project/chatbot/member_information
sudo mkdir -p /work/kibwa_project/chatbot/chat_logs
sudo chown -R $USER:$USER /work/kibwa_project

# 환경 변수 설정
echo "OPENAI_API_KEY=$OPENAI_API_KEY" > /work/kibwa_project/.env
echo "AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID" >> /work/kibwa_project/.env
echo "AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY" >> /work/kibwa_project/.env
echo "AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION" >> /work/kibwa_project/.env

# Docker Hub 로그인
echo "$DOCKER_HUB_TOKEN" | docker login -u "$DOCKER_HUB_USERNAME" --password-stdin

# 이전 컨테이너 정리
if [ -f "/work/kibwa_project/docker-compose.prod.yml" ]; then
    docker-compose -f /work/kibwa_project/docker-compose.prod.yml down || true
fi

# 디스크 공간 확보
echo "--- Freeing up disk space ---"
docker system prune -af

# Docker 이미지 가져오기
echo "--- Pulling latest image ---"
docker pull $DOCKER_HUB_USERNAME/kibwa-chatbot:latest

# 새 컨테이너 시작
echo "--- Starting new container ---"
cd /work/kibwa_project
docker-compose -f docker-compose.prod.yml up -d

echo "✅ 배포가 완료되었습니다"
docker ps | grep kibwa-chatbot || echo "Container is not running"
