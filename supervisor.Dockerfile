# Supervisor-based Dockerfile for non-Kubernetes deployments
# Use: docker build -f supervisor.Dockerfile -t amul-oan-api:supervisor .

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including supervisor
RUN apt-get update && apt-get install -y \
    supervisor \
    gcc \
    python3-dev \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create logs directory for supervisord
RUN mkdir -p /app/logs

# Copy supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 8000

# Start supervisor to manage processes
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
