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

# 📊 Funções de análises de regressão com gráfico

def analise_regressao_linear_simples(df, colunas):
    X = df[colunas[0]]
    Y = df[colunas[1]]
    X_const = sm.add_constant(X)
    modelo = sm.OLS(Y, X_const).fit()
    resumo = modelo.summary().as_text()

    plt.figure(figsize=(8, 6))
    sns.regplot(x=X, y=Y, ci=None, line_kws={"color": "red"})
    plt.xlabel(colunas[0])
    plt.ylabel(colunas[1])
    plt.title("Regressão Linear Simples")
    return resumo, salvar_grafico()

def analise_regressao_linear_multipla(df, colunas):
    X = df[colunas[:-1]]
    Y = df[colunas[-1]]
    X_const = sm.add_constant(X)
    modelo = sm.OLS(Y, X_const).fit()
    resumo = modelo.summary().as_text()
    return resumo, None

def analise_regressao_logistica_binaria(df, colunas):
    X = df[colunas[:-1]]
    y = df[colunas[-1]]
    X_const = sm.add_constant(X)
    modelo = sm.Logit(y, X_const).fit(disp=False)
    resumo = modelo.summary().as_text()

    y_prob = modelo.predict(X_const)
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Curva ROC - Regressão Logística')
    plt.legend()
    return resumo, salvar_grafico()

# 📈 Gráficos independentes

def grafico_boxplot(df, colunas):
    plt.figure(figsize=(8, 6))
    if len(colunas) == 1:
        sns.boxplot(y=df[colunas[0]])
    elif len(colunas) == 2:
        sns.boxplot(x=df[colunas[1]], y=df[colunas[0]])
    plt.title("Boxplot")
    return salvar_grafico()

def grafico_histograma(df, colunas):
    plt.figure(figsize=(8, 6))
    for col in colunas:
        sns.histplot(df[col], kde=True, label=col)
    plt.legend()
    plt.title("Histograma")
    return salvar_grafico()

def grafico_dispersao(df, colunas):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df[colunas[0]], y=df[colunas[1]])
    plt.title("Gráfico de Dispersão")
    return salvar_grafico()

def grafico_pizza(df, colunas):
    plt.figure(figsize=(8, 6))
    counts = df[colunas[0]].value_counts()
    plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%')
    plt.title("Gráfico de Pizza")
    return salvar_grafico()

def grafico_pareto(df, colunas):
    plt.figure(figsize=(8, 6))
    counts = df[colunas[0]].value_counts().sort_values(ascending=False)
    cum_perc = counts.cumsum() / counts.sum() * 100
    ax = counts.plot(kind='bar')
    ax2 = ax.twinx()
    ax2.plot(cum_perc.values, color='red', marker='o', linestyle='-')
    ax2.axhline(80, color='gray', linestyle='dashed')
    ax.set_ylabel("Frequência")
    ax2.set_ylabel("Percentual acumulado (%)")
    plt.title("Gráfico de Pareto")
    return salvar_grafico()

def grafico_serie_temporal(df, colunas):
    plt.figure(figsize=(10, 6))
    for col in colunas[1:]:
        plt.plot(df[colunas[0]], df[col], label=col)
    plt.xlabel(colunas[0])
    plt.title("Série Temporal")
    plt.legend()
    return salvar_grafico()

# Utilitário para salvar gráfico em base64
def salvar_grafico():
    caminho = "grafico.png"
    plt.tight_layout()
    plt.savefig(caminho)
    plt.close()
    with open(caminho, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")
    os.remove(caminho)
    return img_base64

# Dicionários

ANALISES = {
    "regressao_linear_simples": analise_regressao_linear_simples,
    "regressao_linear_multipla": analise_regressao_linear_multipla,
    "regressao_logistica_binaria": analise_regressao_logistica_binaria
}

GRAFICOS = {
    "boxplot": grafico_boxplot,
    "histograma": grafico_histograma,
    "dispersao": grafico_dispersao,
    "pizza": grafico_pizza,
    "pareto": grafico_pareto,
    "serie_temporal": grafico_serie_temporal
}

@app.post("/analise")
async def analisar(
    file: UploadFile = File(...),
    ferramenta_estatistica: str = Form(None),
    ferramenta_grafica: str = Form(None),
    colunas: str = Form(...)
):
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(await file.read()))
        elif file.filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(await file.read()))
        else:
            return JSONResponse(content={"erro": "Formato de arquivo inválido."}, status_code=400)

        colunas_usadas = [col.strip() for col in colunas.split(",")]
        resultado_texto = None
        imagem_base64 = None

        if ferramenta_estatistica:
            funcao = ANALISES.get(ferramenta_estatistica)
            if not funcao:
                return JSONResponse(content={"erro": "Análise estatística desconhecida."}, status_code=400)
            resultado_texto, imagem_base64 = funcao(df, colunas_usadas)

        elif ferramenta_grafica:
            funcao = GRAFICOS.get(ferramenta_grafica)
            if not funcao:
                return JSONResponse(content={"erro": "Gráfico desconhecido."}, status_code=400)
            imagem_base64 = funcao(df, colunas_usadas)

        else:
            return JSONResponse(content={"erro": "Nenhuma ferramenta selecionada."}, status_code=400)

        return {
            "analise": resultado_texto,
            "grafico_base64": imagem_base64,
            "colunas_utilizadas": colunas_usadas
        }

    except Exception as e:
        return JSONResponse(content={"erro": str(e)}, status_code=500)

