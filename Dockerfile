FROM python:3.12-slim
WORKDIR /app
ENV PYTHONPATH=/app/src
COPY requirments.txt .
RUN pip install --no-cache-dir -r requirments.txt 

COPY src/ src/
COPY models/ models/

CMD ["python","-m","taxi_ml.forecasting.predict"]
