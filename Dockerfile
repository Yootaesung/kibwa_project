# Python 3.9 베이스 이미지 사용
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 필요한 디렉토리 생성 (정적 파일과 템플릿은 여전히 필요할 수 있으므로 유지)
RUN mkdir -p /app/chatbot/templates /app/chatbot/static

# 필요한 파일 및 디렉토리 복사
COPY . /app/

# 작업 디렉토리 설정
WORKDIR /app

# Python 경로 설정
ENV PYTHONPATH="${PYTHONPATH}:/app"

# 8000 포트 노출
EXPOSE 8000

# 애플리케이션 실행 (chatbot 디렉토리의 app 모듈 실행)
CMD ["uvicorn", "chatbot.app:app", "--host", "0.0.0.0", "--port", "8000"]
