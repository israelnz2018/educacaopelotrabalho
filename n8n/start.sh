#!/bin/sh
echo "🟡 [start.sh] Script iniciado..."
echo "📁 Diretório atual: $(pwd)"
echo "📂 Conteúdo de /app e /app/n8n:"
ls -la /app
ls -la /app/n8n

echo "📄 Verificando se /app/n8n/main.py existe..."
if [ -f /app/n8n/main.py ]; then
  echo "✅ /app/n8n/main.py encontrado!"
else
  echo "❌ ERRO: /app/n8n/main.py NÃO encontrado!"
  exit 1
fi

# ——————————
# Aqui é a única mudança:
# ——————————

# 1) Navega para dentro de /app/n8n
cd /app/n8n

# 2) Inicia o servidor sem prefixar 'n8n.' nos imports
echo "🚀 Iniciando servidor com uvicorn na porta ${PORT}..."
uvicorn main:app --host 0.0.0.0 --port ${PORT}








