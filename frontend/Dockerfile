
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/ ./frontend/
COPY data/processed/X_test.csv ./data/processed/X_test.csv
COPY data/processed/y_test.csv ./data/processed/y_test.csv

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
