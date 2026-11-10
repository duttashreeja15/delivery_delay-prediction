
FROM python:3.11-slim

WORKDIR /app

# Required for LightGBM in many Linux environments
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY deployment/model/ ./deployment/model/

ENV MODEL_URI=/app/deployment/model

EXPOSE 8001

CMD ["uvicorn", "backend.api:app", \
     "--host", "0.0.0.0", "--port", "8001"]
