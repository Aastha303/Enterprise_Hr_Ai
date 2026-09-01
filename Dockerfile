# Enterprise HR AI - API service
# NOTE: this is a scaffold, not a hardened production image. It runs the app
# and its already-trained model/data as-is. No multi-stage build, no
# non-root user hardening, no secrets management - add those before
# actually deploying anywhere real.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
