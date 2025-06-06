FROM python:3.11-slim

WORKDIR /app/n8n

COPY . /app/n8n

RUN chmod +x start.sh \
 && pip install --upgrade pip \
 && pip install -r /app/n8n/requirements.txt

CMD ["./start.sh"]







