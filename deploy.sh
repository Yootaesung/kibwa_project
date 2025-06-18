#!/bin/bash
set -e

# 필수 디렉토리 생성
mkdir -p /work/kibwa_project/chatbot/member_information
mkdir -p /work/kibwa_project/chatbot/chat_logs

# Docker Hub 로그인
echo "$DOCKER_HUB_TOKEN" | docker login -u "$DOCKER_HUB_USERNAME" --password-stdin

# 이전 컨테이너 정리
# 최신 이미지 가져오기
docker pull $DOCKER_HUB_USERNAME/kibwa-chatbot:latest

# 새 컨테이너 시작
docker-compose -f docker-compose.prod.yml up -d

echo "✅ 배포가 완료되었습니다"

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
