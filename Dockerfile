# Stage 1: Builder stage for Python dependencies
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
COPY chatbot/requirements.txt ./chatbot/requirements.txt
RUN pip install --user -r requirements.txt -r chatbot/requirements.txt

# Stage 2: Node.js and PM2 installation
FROM node:18-slim as node
RUN npm install -g pm2

# Stage 3: Final image
FROM python:3.9-slim
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy PM2 from node stage
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=node /usr/local/bin/pm2 /usr/local/bin/pm2

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only necessary files
COPY --from=builder /app/requirements.txt .
COPY chatbot/ ./chatbot/

# Create log directory
RUN mkdir -p /app/logs

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    NODE_ENV=production

# Run PM2 to manage the application
CMD ["pm2-runtime", "start", "/app/chatbot/ecosystem.config.js", "--env", "production"]