# Base image
FROM python:3.9-slim as builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g pm2 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only necessary files for installing dependencies
COPY requirements.txt .
COPY --chown=nobody:nogroup chatbot/requirements.txt ./chatbot/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt -r chatbot/requirements.txt

# Final image
FROM python:3.9-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install Node.js and PM2
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get update && \
    apt-get install -y nodejs && \
    npm install -g pm2 && \
    rm -rf /var/lib/apt/lists/*

# Copy only necessary files from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/bin/pm2 /usr/bin/pm2

# Copy only necessary application code
COPY chatbot/ ./chatbot/
COPY config/ ./config/
COPY utils/ ./utils/
COPY requirements.txt .

# Copy emotion data
COPY --chown=nobody:nogroup chatbot/emotion_data/ ./chatbot/emotion_data/

# Create necessary directories
RUN mkdir -p /app/logs /app/chat_logs && \
    chmod -R 777 /app/logs /app/chat_logs

# Clean up
RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Expose port
EXPOSE 8000

# Run PM2 to manage the application
CMD ["pm2-runtime", "start", "/app/chatbot/ecosystem.config.js", "--env", "production"]