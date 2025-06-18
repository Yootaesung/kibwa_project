# Stage 1: Builder stage for installing dependencies
FROM python:3.9-slim AS builder

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

# Install Node.js in final image
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install PM2 globally in final image
RUN npm install -g pm2

WORKDIR /app

# Copy application code
COPY . .

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Set environment variables
ENV PATH="/root/.local/bin:${PATH}"

# Set permissions and create directories
RUN chmod +x /app/chatbot/run.sh \
    && mkdir -p /app/logs

# Expose port
EXPOSE 8000

# Run PM2 to manage the application
CMD ["pm2-runtime", "start", "/app/chatbot/ecosystem.config.js", "--env", "production"]