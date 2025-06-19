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

# 필요한 디렉토리 생성
RUN mkdir -p /app/chatbot/templates /app/chatbot/static

# 1. 기본 파일들 복사
COPY . .

# 2. 템플릿과 정적 파일이 있는 경우에만 복사
COPY ./chatbot/templates/ /app/chatbot/templates/
COPY ./chatbot/static/ /app/chatbot/static/

# 디버깅을 위한 파일 목록 출력
RUN echo "=== 파일 목록 확인 ===" && \
    find /app -type f -name "*.html" && \
    ls -la /app/chatbot/templates/ && \
    echo "==================="

# Python이 모듈을 찾을 수 있도록 경로 추가
ENV PYTHONPATH="${PYTHONPATH}:/app"

# 8000 포트 노출
EXPOSE 8000

# 애플리케이션 실행 (chatbot 디렉토리의 app 모듈 실행)
CMD ["uvicorn", "chatbot.app:app", "--host", "0.0.0.0", "--port", "8000"]
