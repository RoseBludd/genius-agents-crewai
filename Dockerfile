FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CREWAI_STORAGE_DIR=/tmp/crewai \
    OTEL_SDK_DISABLED=true \
    CREWAI_TELEMETRY=false

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agents ./agents
COPY main.py .

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
