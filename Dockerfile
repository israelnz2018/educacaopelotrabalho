# syntax=docker/dockerfile:1.4
###############################################################################
# STAGE 1: fastapi
###############################################################################
FROM python:3.11-slim AS fastapi

# configura /app como diretório de trabalho
WORKDIR /app

# copia seu código (incluindo pasta n8n/ com .py, .html, start.sh, requirements.txt, etc)
COPY n8n/ /app/n8n/

# instala dependências Python
RUN chmod +x /app/n8n/start.sh \
 && pip install --upgrade pip \
 && pip install -r /app/n8n/requirements.txt

###############################################################################
# STAGE 2: n8n
###############################################################################
FROM n8nio/n8n:latest AS n8n

###############################################################################
# STAGE FINAL: seleciona qual estágio usar
###############################################################################
# build‐arg PROJECT seleciona fastapi ou n8n (default=fastapi)
ARG PROJECT=fastapi

# usa o estágio cujo nome bate com o PROJECT
FROM ${PROJECT} AS final

# expõe a porta que cada um usa
# - fastapi interna: 8000
# - n8n padrão: 5678 (Railway mapeará $PORT para 5678)
EXPOSE 8000
EXPOSE 5678

# entrypoint: 
# - se fastapi, roda seu start.sh
# - se n8n, usa o comando padrão da imagem n8n
# note: a imagem n8n já inclui ENTRYPOINT, então só precisamos re-definir para fastapi
SHELL ["/bin/sh", "-lc"]
RUN if [ "$PROJECT" = "fastapi" ] ; then \
      echo '#!/bin/sh\ncd /app/n8n\nsh start.sh' > /entrypoint.sh && chmod +x /entrypoint.sh ; \
    fi

ENTRYPOINT ["/entrypoint.sh"]


