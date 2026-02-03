FROM python:3.11-slim

# 1. Install library sistem yang dibutuhkan OpenCV & Mediapipe
# libgl1 adalah penyedia file libGL.so.1 yang hilang tersebut
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Copy & install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy seluruh kode
COPY . .

# 4. Set PYTHONPATH
ENV PYTHONPATH=/app

EXPOSE 8000

# 5. Jalankan Gunicorn
# CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "machine_learning_final.scripts.api:app"]
CMD ["sh", "-c", "gunicorn -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:${PORT:-8000} machine_learning_final.scripts.api:app"]