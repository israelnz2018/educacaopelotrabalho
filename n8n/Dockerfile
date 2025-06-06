FROM python:3.11-slim

WORKDIR /app

COPY ./ /app

RUN chmod +x start.sh \
 && pip install --upgrade pip \
 && pip install -r requirements.txt

ENTRYPOINT ["sh", "./start.sh"]









