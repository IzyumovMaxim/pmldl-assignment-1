FROM python:3.12-slim
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY code ./code
COPY models/steam_success_model.joblib ./models/steam_success_model.joblib
ENV PYTHONPATH=/app MODEL_PATH=/app/models/steam_success_model.joblib
EXPOSE 8000
CMD ["uvicorn", "code.deployment.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

