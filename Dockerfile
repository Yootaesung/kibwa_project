# Python FastAPI 애플리케이션을 위한 Dockerfile
FROM python:3.9-slim

# 1. 시스템 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# 2. PM2 전역 설치
RUN npm install -g pm2

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. Python 의존성 설치
COPY requirements.txt .
COPY chatbot/requirements.txt ./chatbot_requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r chatbot_requirements.txt

# 5. 애플리케이션 코드 복사
COPY . .

# 6. run.sh 실행 권한 부여
RUN chmod +x /app/chatbot/run.sh

# 7. 로그 디렉토리 생성
RUN mkdir -p /app/logs

# 8. 포트 노출
EXPOSE 8000

# 9. PM2를 사용하여 Python 애플리케이션 실행
CMD ["pm2-runtime", "/app/chatbot/ecosystem.config.js", "--env", "production"]