# Stage 1: Builder stage for installing dependencies
FROM python:3.9-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install PM2 globally
RUN npm install -g pm2

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
COPY chatbot/requirements.txt ./chatbot_requirements.txt
RUN pip install --user -r requirements.txt -r chatbot_requirements.txt

# Stage 2: Final image
FROM python:3.9-slim

# Copy only necessary files from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /usr/local/bin/pm2 /usr/local/bin/pm2
COPY --from=builder /usr/local/lib/node_modules /usr/local/lib/node_modules

WORKDIR /app

# Copy application code
COPY . .

# Set environment variables
ENV PATH=/root/.local/bin:$PATH
ENV NODE_PATH=/usr/local/lib/node_modules

# Set permissions and create directories
RUN chmod +x /app/chatbot/run.sh \
    && mkdir -p /app/logs

# Expose port
EXPOSE 8000

# Run PM2 to manage the application
CMD ["pm2-runtime", "start", "/app/chatbot/ecosystem.config.js"]
CMD ["pm2-runtime", "/app/chatbot/ecosystem.config.js", "--env", "production"]