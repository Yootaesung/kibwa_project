#!/bin/bash

# 실행 중인 컨테이너 중지 및 제거
docker stop kibwa-chatbot || true
docker rm kibwa-chatbot || true

# 사용하지 않는 이미지, 네트워크, 빌드 캐시 정리
echo "Cleaning up Docker resources..."
docker system prune -f

echo "Cleanup complete!"
