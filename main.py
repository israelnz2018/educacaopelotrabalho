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

# 🔍 Regressão Linear Simples
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

    a = modelo.params[0]
    b = modelo.params[1]
    p_valor = modelo.pvalues[1]
    r2 = modelo.rsquared
    r2_ajustado = modelo.rsquared_adj
    erro_padrao = np.sqrt(modelo.mse_resid)

    resumo = f"""
**Equação da reta:**  y = {a:.3f} + {b:.3f}·x  
**Valor-p da inclinação:**  {p_valor:.4f}  
**Coeficiente de determinação (R²):**  {r2:.4f}  
**R² ajustado:**  {r2_ajustado:.4f}  
**Erro padrão da estimativa:**  {erro_padrao:.4f}
""".strip()

    plt.figure(figsize=(8, 6))
    sns.regplot(x=X, y=Y, ci=None, line_kws={"color": "red"})
    plt.xlabel(colunas[0])
    plt.ylabel(colunas[1])
    plt.title("Regressão Linear Simples")

    return resumo, salvar_grafico()

# 🔍 Regressão Linear Múltipla
def analise_regressao_linear_multipla(df, colunas):
    X = df[colunas[:-1]]
    Y = df[colunas[-1]]
    X_const = sm.add_constant(X)
    modelo = sm.OLS(Y, X_const).fit()
    r2 = modelo.rsquared_adj
    p_valor = modelo.f_pvalue
    resumo = f"R² ajustado = {r2:.3f}, Valor-p global do modelo = {p_valor:.3f}"
    return resumo, None

# 🔍 Regressão Logística Binária
def analise_regressao_logistica_binaria(df, colunas):
    X = df[colunas[:-1]]
    y = df[colunas[-1]]
    X_const = sm.add_constant(X)
    modelo = sm.Logit(y, X_const).fit(disp=False)
    y_prob = modelo.predict(X_const)
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    p_valor = modelo.llr_pvalue
    resumo = f"AUC da curva ROC = {roc_auc:.2f}, Valor-p global do modelo = {p_valor:.3f}"
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Curva ROC - Regressão Logística')
    plt.legend()
    return resumo, salvar_grafico()

# 📊 Gráfico de Dispersão
def grafico_dispersao(df, colunas):
    if len(colunas) < 2:
        raise ValueError("Gráfico de dispersão requer exatamente duas colunas.")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df[colunas[0]], y=df[colunas[1]])
    plt.title("Gráfico de Dispersão")
    return salvar_grafico()

# 📊 Gráfico Boxplot estilo Minitab
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

# 💾 Salvar gráfico como imagem base64
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
                    raise ValueError(f"Coluna na posição '{valor}' não existe no arquivo. O arquivo tem apenas {len(df.columns)} colunas.")
            return valor

        # 📥 Leitura do arquivo Excel
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
        imagem_analise_base64 = None
        imagem_grafico_isolado_base64 = None

        # 🧠 Análise estatística (com gráfico embutido)
        if ferramenta and ferramenta.strip():
            funcao = ANALISES.get(ferramenta.strip())
            if not funcao:
                return JSONResponse(content={"erro": f"Análise estatística desconhecida: {ferramenta}"}, status_code=400)
            resultado_texto, imagem_analise_base64 = funcao(df, colunas_usadas)

        # 📊 Gráfico isolado
        if grafico and grafico.strip():
            funcao = GRAFICOS.get(grafico.strip())
            if not funcao:
                return JSONResponse(content={"erro": f"Gráfico desconhecido: {grafico}"}, status_code=400)
            imagem_grafico_isolado_base64 = funcao(df, colunas_x_lista, coluna_y if coluna_y else None)

        if not ferramenta and not grafico:
            return JSONResponse(content={"erro": "Nenhuma ferramenta selecionada."}, status_code=400)

        return {
            "analise": resultado_texto or "",
            "grafico_base64": imagem_analise_base64,
            "grafico_isolado_base64": imagem_grafico_isolado_base64,
            "colunas_utilizadas": colunas_usadas
        }

    except ValueError as e:
        return JSONResponse(content={"erro": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"erro": "Erro interno ao processar a análise.", "detalhe": str(e)}, status_code=500)
