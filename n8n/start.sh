#!/bin/sh
echo "🟡 [start.sh] Script iniciado..."
echo "📁 Diretório atual: $(pwd)"
echo "📂 Conteúdo do diretório atual:"
ls -la /app
ls -la /app/n8n

echo "📄 Verificando se o arquivo /app/n8n/main.py existe..."
if [ -f /app/n8n/main.py ]; then
  echo "✅ /app/n8n/main.py encontrado!"
else
  echo "❌ ERRO: Arquivo /app/n8n/main.py NÃO encontrado!"
  exit 1
fi

echo "🚀 Iniciando servidor com uvicorn na porta ${PORT}..."
uvicorn n8n.main:app --host=0.0.0.0 --port=${PORT}






