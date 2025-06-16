#!/bin/bash

# 필요한 디렉토리 생성
mkdir -p ~/deploy/chatbot
cd ~/deploy/chatbot

# 소스 코드 복사 (GitHub Actions에서 수행할 것이므로 실제로는 필요 없을 수 있음)
# cp -r /work/kibwa_project/chatbot/* .

# Python 가상 환경 설정
python3 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install --upgrade pip
pip install -r requirements.txt

# 필요한 경우 추가 설치
# pip install gunicorn uvicorn

# 로그 디렉토리 생성
mkdir -p logs

# PM2로 앱 재시작
pm2 delete fastapi-chatbot || true
pm2 start ecosystem.config.js

# PM2 부팅 시 자동 시작 설정
pm2 save
sudo env PATH=$PATH:/home/ec2-user/.nvm/versions/node/$(nvm version)/bin /home/ec2-user/.nvm/versions/node/$(nvm version)/lib/node_modules/pm2/bin/pm2 startup systemd -u ec2-user --hp /home/ec2-user

# Nginx 설정 (필요한 경우)
# sudo cp nginx/chatbot.conf /etc/nginx/sites-available/
# sudo ln -sf /etc/nginx/sites-available/chatbot.conf /etc/nginx/sites-enabled/
# sudo nginx -t && sudo systemctl restart nginx

echo "Deployment completed successfully!"
