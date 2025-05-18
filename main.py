from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import os

app = FastAPI()

@app.post("/analise")
async def analise_dados(
    file: UploadFile = File(...),
    ferramenta_estatistica: str = Form(...),
    ferramenta_grafica: str = Form(...),
    colunas: str = Form(...),  # Ex: "X,Y"
    prompt: str = Form(...)
):
    try:
        # Ler arquivo
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(await file.read()))
        elif file.filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(await file.read()))
        else:
            return JSONResponse(content={"erro": "Formato de arquivo não suportado"}, status_code=400)

        colunas_usadas = [c.strip() for c in colunas.split(",")]
        if len(colunas_usadas) != 2:
            return JSONResponse(content={"erro": "Selecione exatamente 2 colunas para regressão simples."}, status_code=400)

        x = df[colunas_usadas[0]]
        y = df[colunas_usadas[1]]

        # Regressão Linear Simples
        import statsmodels.api as sm
        X_const = sm.add_constant(x)
        modelo = sm.OLS(y, X_const).fit()

        resumo_texto = modelo.summary().as_text()

        # Gráfico
        plt.figure(figsize=(8, 6))
        sns.regplot(x=x, y=y, ci=None, line_kws={"color": "red"})
        plt.xlabel(colunas_usadas[0])
        plt.ylabel(colunas_usadas[1])
        plt.title("Regressão Linear")

        # Salvar imagem em base64
        img_path = "grafico.png"
        plt.savefig(img_path)
        plt.close()

        with open(img_path, "rb") as image_file:
            img_base64 = base64.b64encode(image_file.read()).decode("utf-8")

        os.remove(img_path)  # Limpar arquivo temporário

        return {
            "analise": resumo_texto,
            "grafico_base64": img_base64,
            "mensagem": f"Análise estatística '{ferramenta_estatistica}' e gráfico '{ferramenta_grafica}' concluídos.",
            "colunas_utilizadas": colunas_usadas,
            "amostra_dados": df[colunas_usadas].head(5).to_dict()
        }

    except Exception as e:
        return JSONResponse(content={"erro": str(e)}, status_code=500)
