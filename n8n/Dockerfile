FROM python:3.11-slim

# 1) Define o diretório de trabalho como /app
WORKDIR /app

# 2) Copia todo o conteúdo (contexto atual, que já é 'n8n/') para '/app/n8n'
COPY . /app/n8n

# 3) Torna o start.sh executável e instala dependências
RUN chmod +x /app/n8n/start.sh \
 && pip install --upgrade pip \
 && pip install -r /app/n8n/requirements.txt

# 4) Usa o start.sh que está em /app/n8n como ENTRYPOINT
ENTRYPOINT ["sh", "/app/n8n/start.sh"]













