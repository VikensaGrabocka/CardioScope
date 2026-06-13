FROM python:3.10-slim

# Install system dependencies needed by librosa and scipy
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and models
COPY app.py .
COPY models/ models/

# Expose port
EXPOSE 8080

# Run with gunicorn (production server, better than flask dev server)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "120", "--workers", "1", "app:app"]