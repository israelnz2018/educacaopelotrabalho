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

def formatar_analise(nome_teste, premissas, resultado, conclusao):
    return (
        "1. Nome do teste estatístico aplicado:\n"
        f"{nome_teste}\n\n"
        "2. Premissas:\n"
        f"{premissas}\n\n"
        "3. Resultado do teste:\n"
        f"{resultado}\n\n"
        "4. Conclusão:\n"
        f"{conclusao}"
    )

def analise_regressao_linear_simples(df, colunas):
    X = df[colunas[0]].astype(str).str.strip().str.replace(",", ".").str.replace(r"[^\d\.\-]", "", regex=True)
    Y = df[colunas[1]].astype(str).str.strip().str.replace(",", ".").str.replace(r"[^\d\.\-]", "", regex=True)
    X = pd.to_numeric(X, errors="coerce")
    Y = pd.to_numeric(Y, errors="coerce")
    validos = ~(X.isna() | Y.isna())
    X = X[validos]
    Y = Y[validos]
    if len(X) < 2 or len(Y) < 2:
        raise ValueError("Não há dados numéricos suficientes para a regressão.")
    X_const = sm.add_constant(X)
    modelo = sm.OLS(Y, X_const).fit()
    p_valor = modelo.pvalues[1]

    nome_teste = "Regressão Linear Simples"
    premissas = "Os resíduos da regressão são normalmente distribuídos, independentes e têm variância constante (homocedasticidade)."
    resultado = f"Valor-p da variável {colunas[0]}: {p_valor:.3f}"
    conclusao = f"{'Houve' if p_valor < 0.05 else 'Não houve'} uma diferença significativa, com 95% de confiança, indicando que a variável {colunas[0]} {'tem' if p_valor < 0.05 else 'não tem'} um impacto significativo sobre {colunas[1]}."

    resumo = formatar_analise(nome_teste, premissas, resultado, conclusao)

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
    r2 = modelo.rsquared_adj
    p_valor = modelo.f_pvalue

    nome_teste = "Regressão Linear Múltipla"
    premissas = "Linearidade, independência dos erros, homocedasticidade e normalidade dos resíduos."
    resultado = f"R² ajustado = {r2:.3f}, Valor-p global do modelo = {p_valor:.3f}"
    conclusao = f"{'O modelo explica' if p_valor < 0.05 else 'O modelo não explica'} significativamente a variável {colunas[-1]} com 95% de confiança."

    resumo = formatar_analise(nome_teste, premissas, resultado, conclusao)
    return resumo, None

def analise_regressao_logistica_binaria(df, colunas):
    X = df[colunas[:-1]]
    y = df[colunas[-1]]
    X_const = sm.add_constant(X)
    modelo = sm.Logit(y, X_const).fit(disp=False)
    y_prob = modelo.predict(X_const)
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    p_valor = modelo.llr_pvalue

    nome_teste = "Regressão Logística Binária"
    premissas = "Observações independentes, ausência de multicolinearidade, e relação linear entre preditores contínuos e o logit da variável dependente."
    resultado = f"AUC da curva ROC = {roc_auc:.2f}, Valor-p global do modelo = {p_valor:.3f}"
    conclusao = f"{'O modelo é estatisticamente significativo' if p_valor < 0.05 else 'O modelo não é estatisticamente significativo'}, com 95% de confiança."

    resumo = formatar_analise(nome_teste, premissas, resultado, conclusao)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Curva ROC - Regressão Logística')
    plt.legend()
    return resumo, salvar_grafico()

def grafico_dispersao(df, colunas):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df[colunas[0]], y=df[colunas[1]])
    plt.title("Gráfico de Dispersão")
    return salvar_grafico()

def salvar_grafico():
    caminho = "grafico.png"
    plt.tight_layout()
    plt.savefig(caminho)
    plt.close()
    with open(caminho, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")
    os.remove(caminho)
    return img_base64

ANALISES = {
    "regressao_simples": analise_regressao_linear_simples,
    "regressao_multipla": analise_regressao_linear_multipla,
    "regressao_logistica_binaria": analise_regressao_logistica_binaria
}

GRAFICOS = {
    "scatter": grafico_dispersao
}

@app.post("/analise")
async def analisar(
    arquivo: UploadFile = File(None),
    ferramenta: str = Form(None),
    grafico: str = Form(None),
    coluna_y: str = Form(None),
    colunas_x: str = Form(None)
):
    try:
        def interpretar_coluna(df, valor):
            valor = valor.strip()
            if len(valor) == 1 and valor.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                idx = ord(valor.upper()) - ord("A")
                if idx < len(df.columns):
                    return df.columns[idx]
                else:
                    raise ValueError(f"Coluna na posição '{valor}' não existe no arquivo. Arquivo tem apenas {len(df.columns)} colunas.")
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
        if colunas_x:
            for c in colunas_x.split(","):
                if c.strip():
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
            imagem_base64 = funcao(df, colunas_usadas)
        else:
            return JSONResponse(content={"erro": "Nenhuma ferramenta selecionada."}, status_code=400)

        return {
            "analise": resultado_texto,
            "grafico_base64": imagem_base64,
            "colunas_utilizadas": colunas_usadas
        }

    except ValueError as e:
        return JSONResponse(content={"erro": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"erro": "Erro interno ao processar a análise.", "detalhe": str(e)}, status_code=500)
