# 빌드 스테이지
FROM python:3.9-slim as builder

WORKDIR /app
COPY requirements.txt .
COPY chatbot/requirements.txt ./chatbot_requirements.txt

# 의존성 설치 (캐시 최적화)
RUN pip install --user -r requirements.txt -r chatbot_requirements.txt

# 런타임 스테이지
FROM python:3.9-slim

WORKDIR /app

# 필요한 시스템 패키지만 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# 빌드된 패키지 복사
COPY --from=builder /root/.local /root/.local
COPY . .

# 불필요한 파일 정리
RUN find /usr/local -type f -name '*.pyc' -delete && \
    find /usr/local -type d -name '__pycache__' -delete

# 경로 설정
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "chatbot.app:app", "--host", "0.0.0.0", "--port", "8000"]