FROM python:3.11-slim

# 1) Instala dependências Python
WORKDIR /app
COPY n8n/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /app/requirements.txt

# 2) Copia seu código (FastAPI + HTMLs + start.sh)
COPY n8n/ /app/n8n/
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# 3) Instala Node.js e a CLI do n8n
RUN apt-get update \
 && apt-get install -y curl gnupg \
 && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
 && apt-get install -y nodejs \
 && npm install -g n8n \
 && rm -rf /var/lib/apt/lists/*

# 4) Build‐arg PROJECT (fastapi ou n8n), default fastapi
ARG PROJECT=fastapi
ENV PROJECT=${PROJECT}

# 5) Expõe as portas
EXPOSE 8000 5678

# 6) Roda o script de entrada
ENTRYPOINT ["/app/start.sh"]



