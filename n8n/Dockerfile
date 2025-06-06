FROM python:3.11-slim

WORKDIR /app

COPY ./n8n /app/n8n

RUN chmod +x /app/n8n/start.sh \
 && pip install --upgrade pip \
 && pip install -r /app/n8n/requirements.txt

ENTRYPOINT ["sh", "/app/n8n/start.sh"]










