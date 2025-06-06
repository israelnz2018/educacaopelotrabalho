#!/bin/sh
echo "🟡 [start.sh] Script iniciado..."
echo "📁 Diretório atual: $(pwd)"
echo "📂 Conteúdo do diretório atual:"
ls -la

echo "📄 Verificando se o arquivo n8n/main.py existe..."
if [ -f n8n/main.py ]; then
  echo "✅ n8n/main.py encontrado!"
else
  echo "❌ ERRO: Arquivo n8n/main.py NÃO encontrado!"
  exit 1
fi

echo "🚀 Iniciando servidor com uvicorn..."
uvicorn n8n.main:app --host=0.0.0.0 --port=8000




