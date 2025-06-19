FROM python:3.9-slim

WORKDIR /app

# 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 의존성 파일 복사 및 설치
COPY chatbot/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY ./chatbot/ /app/chatbot/

# 로그 디렉토리 생성
RUN mkdir -p /app/chatbot/logs && \
    chmod -R 777 /app/chatbot/logs

# 포트 노출
EXPOSE 8000

# 애플리케이션 실행
CMD ["uvicorn", "chatbot.app:app", "--host", "0.0.0.0", "--port", "8000"]
