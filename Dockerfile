FROM node:18-alpine
WORKDIR /app

# 앱 의존성 설치
COPY package*.json ./
RUN npm install

# 소스 코드 복사
COPY . .

# 8000 포트 노출
EXPOSE 8000

# PM2로 앱 실행
RUN npm install -g pm2
CMD ["pm2-runtime", "chatbot/app.js"]
