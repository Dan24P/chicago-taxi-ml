FROM python:3.12-slim
WORKDIR /app
ENV PYTHONPATH=/app/src
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 

COPY src/ src/
COPY sql/ sql/
COPY run_pipeline.py .
CMD ["python","run_pipeline.py"]
