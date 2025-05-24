from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import os
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc
import numpy as np

app = FastAPI()

def salvar_grafico():
    caminho = "grafico.png"
    plt.tight_layout()
    plt.savefig(caminho)
    plt.close()
    with open(caminho, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")
    os.remove(caminho)
    return img_base64

def grafico_dispersao(df, colunas):
    if len(colunas) < 2:
        raise ValueError("Gráfico de dispersão requer exatamente duas colunas.")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df[colunas[0]], y=df[colunas[1]])
    plt.title("Gráfico de Dispersão")
    return salvar_grafico()

def grafico_boxplot(df, colunas, coluna_y=None):
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    if coluna_y:
        col_y = coluna_y.strip()
        if col_y not in df.columns:
            raise ValueError(f"Coluna Y '{col_y}' não encontrada.")

        if colunas:
            grupo = df[colunas].astype(str).agg("-".join, axis=1)
            data = pd.DataFrame({col_y: df[col_y], "Grupo": grupo})
            ax = sns.boxplot(x="Grupo", y=col_y, data=data, color="#87CEFA", linewidth=1.5)

            medias = data.groupby("Grupo")[col_y].mean()
            for i, (grupo, media) in enumerate(medias.items()):
                plt.scatter(i, media, color="blue", marker="D", s=50, label="Média" if i == 0 else "")
            ax.set_xlabel("Grupo")
        else:
            ax = sns.boxplot(y=df[col_y], color="#87CEFA", linewidth=1.5)
            media = df[col_y].mean()
            plt.scatter(0, media, color="blue", marker="D", s=50, label="Média")
            ax.set_xlabel("")

        ax.set_ylabel(col_y)
        ax.set_title(f"Boxplot de {col_y}", fontsize=14, weight="bold")

    else:
        if len(colunas) < 2:
            raise ValueError("Para comparação entre variáveis, selecione ao menos duas colunas.")
        df_sel = df[colunas].copy()
        for col in colunas:
            df_sel[col] = pd.to_numeric(df_sel[col], errors="coerce")

        df_long = df_sel.melt(var_name="Variável", value_name="Valor")
        ax = sns.boxplot(x="Variável", y="Valor", data=df_long, color="#87CEFA", linewidth=1.5)

        medias = df_long.groupby("Variável")["Valor"].mean()
        for i, (variavel, media) in enumerate(medias.items()):
            plt.scatter(i, media, color="blue", marker="D", s=50, label="Média" if i == 0 else "")

        ax.set_xlabel("Variável")
        ax.set_ylabel("Valor")
        ax.set_title("Boxplot Comparativo", fontsize=14, weight="bold")

    sns.despine()
    return salvar_grafico()

ANALISES = {
    "regressao_simples": analise_regressao_linear_simples,
    "regressao_multipla": analise_regressao_linear_multipla,
    "regressao_logistica_binaria": analise_regressao_logistica_binaria
}

GRAFICOS = {
    "scatter": grafico_dispersao,
    "boxplot": grafico_boxplot
}

@app.post("/analise")
async def analisar(
    arquivo: UploadFile = File(None),
    ferramenta: str = Form(None),
    grafico: str = Form(None),
    coluna_y: str = Form(None),
    colunas_x: str | list[str] = Form(None)
):
    try:
        def interpretar_coluna(df, valor):
            valor = valor.strip()
            if len(valor) == 1 and valor.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                idx = ord(valor.upper()) - ord("A")
                if idx < len(df.columns):
                    return df.columns[idx]
                else:
                    raise ValueError(f"Coluna na posição '{valor}' não existe no arquivo.")
            return valor

        if arquivo and arquivo.filename.endswith(".xlsx"):
            file_bytes = await arquivo.read()
            df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
        else:
            return JSONResponse(content={"erro": "Envie um arquivo Excel (.xlsx) válido."}, status_code=400)

        df.columns = df.columns.str.strip()
        colunas_usadas = []

        if coluna_y:
            colunas_usadas.append(interpretar_coluna(df, coluna_y))

        colunas_x_lista = []
        if colunas_x:
            if isinstance(colunas_x, str):
                colunas_x_lista = [x.strip() for x in colunas_x.split(",") if x.strip()]
            elif isinstance(colunas_x, list):
                colunas_x_lista = [x.strip() for x in colunas_x if isinstance(x, str) and x.strip()]

        for c in colunas_x_lista:
            colunas_usadas.append(interpretar_coluna(df, c))

        if not colunas_usadas:
            return JSONResponse(content={"erro": "Informe ao menos coluna_y ou colunas_x."}, status_code=422)

        for col in colunas_usadas:
            if col not in df.columns:
                return JSONResponse(content={"erro": f"Coluna '{col}' não encontrada no arquivo."}, status_code=400)

        resultado_texto = None
        imagem_base64 = None

        if ferramenta and ferramenta.strip():
            funcao = ANALISES.get(ferramenta.strip())
            if not funcao:
                return JSONResponse(content={"erro": "Análise estatística desconhecida."}, status_code=400)
            resultado_texto, imagem_base64 = funcao(df, colunas_usadas)

        elif grafico and grafico.strip():
            funcao = GRAFICOS.get(grafico.strip())
            if not funcao:
                return JSONResponse(content={"erro": "Gráfico desconhecido."}, status_code=400)
            imagem_base64 = funcao(df, colunas_usadas, coluna_y=coluna_y)

        else:
            return JSONResponse(content={"erro": "Nenhuma ferramenta selecionada."}, status_code=400)

        return {
            "analise": resultado_texto or "",
            "grafico_base64": imagem_base64,
            "colunas_utilizadas": colunas_usadas
        }

    except ValueError as e:
        return JSONResponse(content={"erro": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"erro": "Erro interno ao processar a análise.", "detalhe": str(e)}, status_code=500)

