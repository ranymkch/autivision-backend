FROM python:3.11-slim-bullseye

WORKDIR /app

# System libraries required by OpenCV and MediaPipe on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from requirements.txt unchanged
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Override opencv with headless (no display needed on server)
RUN pip install --no-cache-dir opencv-python-headless==4.13.0.92

# Packages missing from requirements.txt (only needed in container)
RUN pip install --no-cache-dir python-dotenv==1.0.1 "packaging>=21.0"

# Copy application files
COPY main.py .
COPY face_landmarker.task .
COPY best_mobilenetv2.h5 .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
