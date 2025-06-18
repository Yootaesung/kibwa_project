# Base image
FROM python:3.9-slim

WORKDIR /app

# Install Node.js and PM2
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g pm2 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY chatbot/requirements.txt ./chatbot/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt -r chatbot/requirements.txt

# Copy app code
COPY chatbot/ ./chatbot/

# Create log directory
RUN mkdir -p /app/logs

# Expose port
EXPOSE 8000

# Run PM2 to manage the application
CMD ["pm2-runtime", "start", "/app/chatbot/ecosystem.config.js", "--env", "production"]