FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY web_app/requirements-local.txt /tmp/requirements-local.txt

# The browser extracts MediaPipe landmarks. The server only needs the CPU
# PyTorch runtime, NumPy, and the FastAPI/WebSocket dependencies.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
    && pip install --no-cache-dir numpy==1.26.4 -r /tmp/requirements-local.txt

COPY web_app/__init__.py /app/web_app/__init__.py
COPY web_app/backend /app/web_app/backend
COPY web_app/frontend /app/web_app/frontend
COPY web_app/best_model.pth /app/web_app/best_model.pth
COPY web_app/sign_videos /app/web_app/sign_videos
COPY show-50-signs/signs /app/show-50-signs/signs

EXPOSE 7860

CMD ["sh", "-c", "uvicorn web_app.backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
