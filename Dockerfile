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
ENV KIBWA05_DEFAULT_REGION="ap-northeast-3" \
    AWS_DEFAULT_REGION="ap-southeast-2" \
    KIBWA05_ACCESS_KEY_ID="" \
    KIBWA05_SECRET_ACCESS_KEY=""

# S3 버킷 환경 변수
ENV S3_BUCKET="kibwa-05" \
    S3_PREFIX="project/"

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 필요한 디렉토리 생성 (S3를 사용하므로 로컬 디렉토리는 임시용으로만 사용)
RUN mkdir -p /app/chatbot/temp /app/chatbot/logs

# 로그 디렉토리 권한 설정
RUN chmod -R 777 /app/chatbot/logs

# 필요한 파일들만 복사 (불필요한 데이터 디렉토리는 제외)
COPY ./chatbot/ /app/chatbot/
# config 디렉토리는 chatbot/ 하위에 있으므로 별도로 복사할 필요 없음

# Python 경로 설정
ENV PYTHONPATH="${PYTHONPATH}:/app"

# 8000 포트 노출
EXPOSE 8000

# 애플리케이션 실행 (chatbot 디렉토리의 app 모듈 실행)
CMD ["uvicorn", "chatbot.app:app", "--host", "0.0.0.0", "--port", "8000"]
