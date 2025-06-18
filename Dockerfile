# Stage 1: Builder stage for installing dependencies
FROM python:3.9-slim

WORKDIR /app

# Install minimal dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g pm2 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY chatbot/requirements.txt ./chatbot/requirements.txt

# Install Python deps (no cache)
RUN pip install --no-cache-dir -r requirements.txt -r chatbot/requirements.txt

# Copy app code
COPY . .

# Create log dir
RUN mkdir -p /app/logs

    && mkdir -p /app/logs

# Expose port
EXPOSE 8000

# Run PM2 to manage the application
CMD ["pm2-runtime", "start", "/app/chatbot/ecosystem.config.js", "--env", "production"]