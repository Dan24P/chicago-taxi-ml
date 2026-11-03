FROM python:3.12-slim
WORKDIR /app
ENV PYTHONPATH=/app/src
COPY requirments.txt .
RUN pip install --no-cache-dir -r requirments.txt 

COPY src/ src/

COPY run_pipeline.py .
CMD ["python","run_pipeline.py"]
