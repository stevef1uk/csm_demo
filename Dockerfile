FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy application file
COPY app_simple_scaleway.py ./app.py

# Install packages in separate steps for better reliability
RUN pip install --no-cache-dir \
    gradio==5.23.1 \
    transformers==4.35.2 \
    numpy==1.25.2 \
    requests==2.31.0 \
    gtts==2.3.2 \
    pydub==0.25.1

# Install PyTorch separately with its custom index
RUN pip install --no-cache-dir \
    torch==2.1.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cpu

# Create directories
RUN mkdir -p audio_outputs user_sessions

# Expose port
EXPOSE 7860

# Start application
CMD ["python", "app.py"]