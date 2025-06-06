FROM python:3.11-slim

WORKDIR /app/n8n

COPY . /app/n8n

RUN pip install --upgrade pip \
 && pip install -r /app/n8n/requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]






