# Python 3.9 베이스 이미지 사용
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 환경 변수 선언 (런타임에 주입받을 변수들)
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="${PYTHONPATH}:/app" \
    
    # 챗봇 데이터용 S3 (kibwa-05)
    KIBWA05_ACCESS_KEY_ID="" \
    KIBWA05_SECRET_ACCESS_KEY="" \
    KIBWA05_DEFAULT_REGION="ap-northeast-3" \
    KIBWA05_BUCKET="kibwa-05" \
    KIBWA05_PREFIX="project/" \
    
    # 테스트 시나리오용 S3 (kibwa-12)
    AWS_ACCESS_KEY_ID="" \
    AWS_SECRET_ACCESS_KEY="" \
    AWS_DEFAULT_REGION="ap-southeast-2" \
    TEST_BUCKET="kibwa-12" \
    TEST_PREFIX="project/"

# 의존성 파일 복사 및 설치
COPY chatbot/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY ./chatbot/ /app/chatbot/

# 로그 디렉토리 생성
RUN mkdir -p /app/chatbot/logs && \
    chmod -R 777 /app/chatbot/logs

# 8000 포트 노출
EXPOSE 8000

# 애플리케이션 실행
CMD ["uvicorn", "chatbot.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
