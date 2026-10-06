FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY web_app/requirements-local.txt /tmp/requirements-local.txt
COPY web_app/requirements-slovo.txt /tmp/requirements-slovo.txt

# Linux runs Slovo ONNX inference on CPU.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements-local.txt

COPY web_app/__init__.py /app/web_app/__init__.py
COPY web_app/backend /app/web_app/backend
COPY web_app/frontend /app/web_app/frontend
COPY web_app/models/slovo /app/web_app/models/slovo
COPY web_app/sign_videos /app/web_app/sign_videos
COPY show-50-signs/signs /app/show-50-signs/signs

EXPOSE 7860

CMD ["sh", "-c", "uvicorn web_app.backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
