FROM python:3.11-slim

WORKDIR /app

COPY . /app/n8n

RUN pip install --upgrade pip \
 && pip install -r requirements.txt

CMD ["uvicorn", "n8n.main:app", "--host", "0.0.0.0", "--port", "8000"]





