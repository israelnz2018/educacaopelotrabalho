import json
import os
from flask import Flask, request, jsonify
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "✅ API está no ar!"

@app.route("/executar", methods=["POST"])
def executar():
    try:
        aluno_aba = request.args.get("aluno")
        if not aluno_aba:
            return jsonify({"erro": "Nome da aba do aluno não fornecido na URL"}), 400

        # Autenticação com a planilha usando a variável de ambiente
        credentials_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
        creds_dict = json.loads(credentials_json)
        creds = Credentials.from_service_account_info(creds_dict, scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
        client = gspread.authorize(creds)
        planilha = client.open("educacaopelotrabalho")
        aba = planilha.worksheet(aluno_aba)

        # Resto do seu código de leitura, análise e escrita na planilha...

        return jsonify({"status": "sucesso", "mensagem": f"Análise realizada para o aluno {aluno_aba}!"})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

        # Cria gráfico simples
        plt.figure()
        df.plot(kind='box')
        plt.savefig("grafico.png")

        # Atualiza a célula B3 com uma análise
        aba.update_acell("B3", "✅ Análise realizada com sucesso!")

        return jsonify({"status": "sucesso", "mensagem": "Processamento finalizado!"})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

