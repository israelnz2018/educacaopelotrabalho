from flask import Flask, request, jsonify
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "✅ API está no ar!"

@app.route("/executar", methods=["POST"])
def executar():
    try:
        # Dados recebidos do webhook
        data = request.get_json()
        aluno = data.get("aluno")
        prompt = data.get("prompt")

        # Acesso à planilha
        scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("credenciais.json", scope)
        client = gspread.authorize(creds)
        planilha = client.open("educacaopelotrabalho")
        aba = planilha.worksheet(aluno)

        # Pega os dados da planilha
        dados = aba.get_all_values()[2:]
        df = pd.DataFrame(dados, columns=aba.row_values(2))

        # Cria gráfico simples
        plt.figure()
        df.plot(kind='box')
        plt.savefig("grafico.png")

        # Atualiza a célula B3 com uma análise
        aba.update_acell("B3", "✅ Análise realizada com sucesso!")

        return jsonify({"status": "sucesso", "mensagem": "Processamento finalizado!"})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500
