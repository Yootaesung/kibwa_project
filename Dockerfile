FROM python:3.9-slim

WORKDIR /app/chatbot

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY app.py .
COPY templates/ .
COPY static/ .

# 포트 노출
EXPOSE 8000

# 실행 명령어
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]