FROM python:3.9-slim

WORKDIR /app

# 필요한 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 의존성 설치 (의존성 파일만 먼저 복사하여 캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사 (필요한 파일만 복사)
COPY chatbot/ /app/chatbot/
COPY config/ /app/config/

# 필요한 디렉토리 생성
RUN mkdir -p /app/logs

# 환경 변수 설정
ENV PYTHONPATH=/app

# 작업 디렉토리 설정
WORKDIR /app/chatbot

# 포트 노출
EXPOSE 8000

# 실행 명령어
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]