FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /tmp/logs /tmp/nltk_data static/uploads
RUN chmod -R 777 /tmp static/uploads
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]