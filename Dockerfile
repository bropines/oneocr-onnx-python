# Use a lightweight official Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install basic system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to utilize Docker build cache
COPY requirements.txt .

# Install Python package dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the core library and tests
COPY oneocr/ oneocr/
COPY tests/ tests/

# Copy the decrypted models folder
# IMPORTANT: You must run prepare_files.py on Windows first to populate this folder!
COPY models/ models/

# Default entrypoint to run OCR on an image
ENTRYPOINT ["python", "tests/test_real_image.py"]
