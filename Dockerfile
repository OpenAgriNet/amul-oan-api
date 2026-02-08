# Use Python as the base image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies (ffmpeg for pydub WebM/Opus -> WAV conversion in transcribe)
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI (Kubernetes manages process lifecycle)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 --workers ${AMUL_CHAT_BE_UVICORN_WORKERS:-1} --timeout-keep-alive 65 --timeout-graceful-shutdown 30 --backlog 2048"]