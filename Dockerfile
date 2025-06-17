# Python 및 Node.js 기반 이미지 사용
FROM node:18-slim

# Python 설치를 위한 의존성 추가
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /work

# Python 가상 환경 설정
RUN python3 -m venv /work/venv
ENV VIRTUAL_ENV=/work/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# PM2 전역 설치
RUN npm install -g pm2

# 애플리케이션 코드 복사
COPY . .

# Python 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 필요한 포트 노출
EXPOSE 8000

# PM2를 사용하여 애플리케이션 실행
CMD ["pm2-runtime", "app.js"]
