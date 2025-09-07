FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
ENV PORT=8080
CMD exec gunicorn -b 0.0.0.0:${PORT} -w 2 main:app
