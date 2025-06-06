#!/bin/sh
echo "🟡 [start.sh] Script iniciado..."
echo "📁 Diretório atual: $(pwd)"
echo "📂 Conteúdo do diretório atual:"
ls -la

echo "📄 Verificando se o arquivo main.py existe..."
if [ -f main.py ]; then
  echo "✅ main.py encontrado!"
else
  echo "❌ ERRO: Arquivo main.py NÃO encontrado!"
  exit 1
fi

echo "🚀 Iniciando servidor com uvicorn..."
uvicorn main:app --host=0.0.0.0 --port=8000



