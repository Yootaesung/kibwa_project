FROM python:3.9-slim

WORKDIR /app/chatbot

# requirements.txt만 먼저 복사하여 의존성 캐싱
COPY chatbot/requirements.txt .

# 필요한 패키지만 설치
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip && \
    rm -rf /usr/local/lib/python3.9/site-packages/*.dist-info && \
    rm -rf /usr/local/lib/python3.9/site-packages/*.egg-info

# 필요한 파일들만 복사
COPY chatbot/app.py .
COPY chatbot/config/ .
COPY chatbot/templates/ .
COPY chatbot/static/ .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]