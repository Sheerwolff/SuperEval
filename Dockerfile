FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=/app
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN chmod +x start.sh && useradd -m -u 1000 user && chown -R user /app
USER user

# Hugging Face Spaces routes traffic to 7860
EXPOSE 7860
CMD ["./start.sh"]
