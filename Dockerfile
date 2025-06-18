# 경량화된 베이스 이미지 사용
FROM python:3.9-slim

# 필요한 시스템 패키지만 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사
COPY requirements.txt .
COPY chatbot/requirements.txt ./chatbot_requirements.txt

# 의존성 설치 (캐시 최적화)
RUN pip install --no-cache-dir -r requirements.txt -r chatbot_requirements.txt

# 필요한 파일만 복사
COPY chatbot ./chatbot
COPY config ./config
COPY utils ./utils

# 불필요한 파일 제거
RUN find /usr/local -type f -name '*.pyc' -delete && \
    find /usr/local -type d -name '__pycache__' -delete

# 포트 노출
EXPOSE 8000

# 실행 명령
CMD ["uvicorn", "chatbot.app:app", "--host", "0.0.0.0", "--port", "8000"]