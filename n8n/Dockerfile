FROM python:3.11-slim

# 1) Vamos criar o diretório /app/n8n dentro da imagem
WORKDIR /app

# 2) Copia todo o conteúdo de 'n8n/' (contexto) para '/app/n8n'
COPY . /app/n8n

# 3) Torna o start.sh executável e instala as dependências
RUN chmod +x /app/n8n/start.sh \
 && pip install --upgrade pip \
 && pip install -r /app/n8n/requirements.txt

# 4) Usa o start.sh que está em /app/n8n
ENTRYPOINT ["sh", "/app/n8n/start.sh"]











