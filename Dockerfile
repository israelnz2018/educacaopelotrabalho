# syntax=docker/dockerfile:1.4

###############################################################################
# Define build argument with default
###############################################################################
# PROJECT selects which stage to use: "fastapi" or "n8n"
ARG PROJECT=fastapi

###############################################################################
# STAGE 1: FastAPI (servindo seu formulário HTML)
###############################################################################
FROM python:3.11-slim AS fastapi

# configura /app como diretório de trabalho
WORKDIR /app

# copia seu código Python e HTML (pasta n8n/ contém main.py, start.sh, index.html, etc.)
COPY n8n/ /app/n8n/

# instala dependências Python e torna start.sh executável
RUN chmod +x /app/n8n/start.sh \
 && pip install --upgrade pip \
 && pip install -r /app/n8n/requirements.txt

###############################################################################
# STAGE 2: n8n UI (Workflow Designer)
###############################################################################
FROM n8nio/n8n:latest AS n8n

###############################################################################
# STAGE FINAL: escolhe qual imagem vai rodar
###############################################################################
# Usa o estágio cujo nome corresponde ao PROJECT
FROM ${PROJECT} AS final

# expõe as portas usadas por cada serviço
# - fastapi interna: 8000 (não mapeada externamente, se não precisar)
# - n8n padrão: 5678 (Railway mapeia $PORT para 5678)
EXPOSE 8000
EXPOSE 5678

# Define shell para comandos seguintes
SHELL ["/bin/sh", "-lc"]

# Se estivermos no estágio fastapi, gera um entrypoint que executa seu start.sh
RUN if [ "$PROJECT" = "fastapi" ] ; then \
      echo '#!/bin/sh\ncd /app/n8n\nsh start.sh' > /entrypoint.sh && chmod +x /entrypoint.sh ; \
    fi

# EntryPoint:
# - fastapi → /entrypoint.sh (inicia seu start.sh)
# - n8n     → entrypoint padrão da imagem n8n
ENTRYPOINT ["/entrypoint.sh"]



