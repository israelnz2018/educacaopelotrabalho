from fastapi import FastAPI, File, UploadFile, Form, Request
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
from scipy import stats


app = FastAPI()

# 🎨 Estilo Global - Inspirado no Minitab
def aplicar_estilo_minitab():
    plt.style.use("default")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#333333",
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "font.family": "Arial",
        "grid.linestyle": "--",
        "grid.color": "#CCCCCC",
        "grid.alpha": 0.7,
        "legend.frameon": False
    })
    plt.grid(True)

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
    aplicar_estilo_minitab()
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
    aplicar_estilo_minitab()
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
    aplicar_estilo_minitab()
    sns.scatterplot(x=df[colunas[0]], y=df[colunas[1]])
    plt.title("Gráfico de Dispersão")
    return salvar_grafico()

# 📊 Gráfico de Boxplot Simples (Y numérica)
def grafico_boxplot_simples(df, colunas, coluna_y=None):
    if not coluna_y:
        raise ValueError("Para o boxplot simples, a coluna Y (numérica) é obrigatória.")

    y = df[coluna_y].astype(str).str.replace(",", ".").str.replace(r"[^\d\.\-]", "", regex=True)
    y = pd.to_numeric(y, errors="coerce").dropna()
    if len(y) < 2:
        raise ValueError("Coluna Y deve conter ao menos dois valores numéricos.")

    df_box = pd.DataFrame({coluna_y: y, "grupo": "A"})

    plt.figure(figsize=(6, 6))
    aplicar_estilo_minitab()
    sns.boxplot(data=df_box, x="grupo", y=coluna_y, color="#89CFF0", width=0.3)
    sns.pointplot(data=df_box, x="grupo", y=coluna_y, estimator=np.mean,
                  markers="D", color="red", scale=1.2, errwidth=0)
    plt.xlabel("")
    plt.ylabel(coluna_y)
    plt.title("Boxplot Simples com Média (losango)")
    return salvar_grafico()

def grafico_pareto(df, colunas):
    if len(colunas) != 1:
        raise ValueError("O gráfico de Pareto requer exatamente uma coluna categórica.")

    coluna = colunas[0]
    contagem = df[coluna].value_counts().sort_values(ascending=False)
    porcentagem = contagem / contagem.sum() * 100
    acumulado = porcentagem.cumsum()

    plt.figure(figsize=(10, 6))
    aplicar_estilo_minitab()

    # Barras de frequência
    ax = sns.barplot(x=contagem.index, y=contagem.values, color="#89CFF0")
    ax.set_ylabel("Frequência")
    ax.set_xlabel(coluna)
    ax.set_title("Gráfico de Pareto")

    # Linha acumulada (%)
    ax2 = ax.twinx()
    ax2.plot(contagem.index, acumulado.values, color="red", marker="o", linewidth=2)
    ax2.set_ylabel("Acumulado (%)")
    ax2.set_ylim(0, 110)

    # Rótulos no eixo x girados para melhor leitura
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    return salvar_grafico()

# 📊 Histograma Múltiplo com Sobreposição + Curvas de Densidade por Categoria
def grafico_histograma_multiplo(df, colunas_x, coluna_y=None):
    if not coluna_y or coluna_y.strip() == "":
        raise ValueError("Você deve selecionar uma coluna Y com dados numéricos.")
    if not colunas_x or len(colunas_x) < 1:
        raise ValueError("Você deve selecionar ao menos uma coluna X com categorias para o histograma múltiplo.")

    coluna_categoria = colunas_x[0]

    # Convertendo Y para numérico
    y = df[coluna_y].astype(str).str.replace(",", ".").str.replace(r"[^\d\.\-]", "", regex=True)
    y = pd.to_numeric(y, errors="coerce")

    # Convertendo categorias para string
    categorias = df[coluna_categoria].astype(str)
    df_filtrado = pd.DataFrame({coluna_categoria: categorias, coluna_y: y}).dropna()

    if df_filtrado.empty:
        raise ValueError("Não há dados válidos suficientes para gerar o gráfico.")

    plt.figure(figsize=(10, 6))

    cores_fortes = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
    cores_claras = ['#aec7e8', '#ff9896', '#98df8a', '#ffbb78', '#c5b0d5', '#c49c94']

    for i, categoria in enumerate(df_filtrado[coluna_categoria].unique()):
        subset = df_filtrado[df_filtrado[coluna_categoria] == categoria][coluna_y]

        if len(subset) < 2:
            continue

        cor_hist = cores_fortes[i % len(cores_fortes)]
        cor_kde = cores_claras[i % len(cores_claras)]

        sns.histplot(
            subset,
            kde=False,
            color=cor_hist,
            label=f'{categoria} (Hist)',
            stat="density",
            element="step",
            edgecolor="black",
            alpha=0.5
        )
        sns.kdeplot(
            subset,
            color=cor_kde,
            label=f'{categoria} (Dens)',
            linewidth=2
        )

    plt.xlabel(coluna_y)
    plt.ylabel("Densidade")
    plt.title("Histograma Múltiplo com Curvas de Densidade por Categoria")
    plt.legend()
    plt.tight_layout()

    return salvar_grafico()


def grafico_boxplot_multiplo(df, colunas, coluna_y=None):
    if not coluna_y or not coluna_y.strip():
        raise ValueError("Você deve selecionar uma coluna Y com valores numéricos para o boxplot múltiplo.")

    coluna_y = coluna_y.strip()

    # Validação do título da Y
    if coluna_y.startswith("Unnamed") or coluna_y not in df.columns:
        raise ValueError(f"A coluna Y '{coluna_y}' não tem título válido ou não foi encontrada.")

    # Trata Y
    y = df[coluna_y].astype(str).str.replace(",", ".").str.replace(r"[^\d\.\-]", "", regex=True)
    y = pd.to_numeric(y, errors="coerce")
    if y.dropna().shape[0] < 2:
        raise ValueError("A coluna Y deve conter ao menos dois valores numéricos válidos.")

    # Trata colunas X (remover Y e Unnamed)
    colunas = [c.strip() for c in colunas if c.strip() and c.strip() != coluna_y and not c.strip().startswith("Unnamed")]

    if not colunas:
        raise ValueError("Nenhuma coluna X válida foi selecionada para o agrupamento.")

    x_col = colunas[0]
    if x_col not in df.columns:
        raise ValueError(f"A coluna X '{x_col}' não foi encontrada no arquivo.")

    grupo = df[x_col].astype(str)
    df_plot = pd.DataFrame({coluna_y: y, x_col: grupo}).dropna()

    if df_plot.empty:
        raise ValueError("Os dados da coluna Y e do grupo X selecionado não têm valores válidos simultaneamente.")

    plt.figure(figsize=(10, 6))
    aplicar_estilo_minitab()

    sns.boxplot(data=df_plot, x=x_col, y=coluna_y, color="#89CFF0")
    sns.pointplot(data=df_plot, x=x_col, y=coluna_y, estimator=np.mean,
                  markers="D", color="red", scale=1.1, errwidth=0)

    plt.title(f"Boxplot Múltiplo por '{x_col}'")
    plt.xlabel(x_col)
    plt.ylabel(coluna_y)
    plt.xticks(rotation=45)

    return salvar_grafico()

# 📊 Histograma Simples com Curva de Densidade (tipo Gauss)
def grafico_histograma_simples(df, colunas, coluna_y=None):
    if not colunas or len(colunas) == 0:
        raise ValueError("Você deve selecionar uma coluna Y com dados numéricos para o histograma simples.")

    coluna_y = colunas[0]

    y = df[coluna_y].astype(str).str.replace(",", ".").str.replace(r"[^\d\.\-]", "", regex=True)
    y = pd.to_numeric(y, errors="coerce").dropna()

    if len(y) < 2:
        raise ValueError("Coluna Y deve conter ao menos dois valores numéricos.")

    plt.figure(figsize=(8, 6))
    sns.histplot(y, kde=True, color="#89CFF0", edgecolor="black")
    plt.xlabel(coluna_y)
    plt.ylabel("Frequência")
    plt.title("Histograma Simples com Curva de Densidade")

    return salvar_grafico()

from scipy import stats

def analise_descritiva(df, colunas):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    import base64
    import io

    if len(colunas) != 1:
        raise ValueError("Selecione apenas uma coluna para a Análise Descritiva.")

    coluna = colunas[0]
    y = df[coluna].dropna().astype(float)
    n = len(y)

    if n < 3:
        raise ValueError("A amostra precisa ter pelo menos 3 valores numéricos.")

    media = np.mean(y)
    mediana = np.median(y)
    moda = stats.mode(y, keepdims=False).mode
    desvio = np.std(y, ddof=1)
    variancia = np.var(y, ddof=1)
    minimo = np.min(y)
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    maximo = np.max(y)
    assimetria = stats.skew(y)
    curtose = stats.kurtosis(y)

    # Intervalos de confiança 95%
    z = 1.96
    erro_media = z * (desvio / np.sqrt(n))
    erro_mediana = 1.57 * (desvio / np.sqrt(n))  # aproximação
    erro_variancia = z * (np.std([np.var(y, ddof=1) for _ in range(1000)], ddof=1))  # bootstrap rudimentar

    # Texto
    texto = f"""
**Análise Descritiva da coluna '{coluna}'**
- Média: {media:.4f}
- Mediana: {mediana:.4f}
- Moda: {moda:.4f}
- Desvio Padrão: {desvio:.4f}
- Variância: {variancia:.4f}
- Mínimo: {minimo:.4f}
- 1º Quartil: {q1:.4f}
- 3º Quartil: {q3:.4f}
- Máximo: {maximo:.4f}
- Assimetria: {assimetria:.4f}
- Curtose: {curtose:.4f}
- N: {n}
- IC 95% da Média: [{media - erro_media:.4f}, {media + erro_media:.4f}]
- IC 95% da Mediana (aprox): [{mediana - erro_mediana:.4f}, {mediana + erro_mediana:.4f}]
- IC 95% da Variância (estimado): ±{erro_variancia:.4f}
"""

    # Boxplot horizontal
    fig, ax = plt.subplots(figsize=(8, 2))
    box = ax.boxplot(y, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))

    # Média como losango
    ax.plot(media, 1, marker='D', color='darkred', label='Média')
    ax.set_title(f"Boxplot Horizontal de {coluna}")
    ax.set_yticklabels([''])

    ax.legend(loc="upper right")
    fig.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    imagem_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()

    return texto.strip(), imagem_base64


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

# 🔗 Dicionário de análises disponíveis
ANALISES = {
    "regressao_simples": analise_regressao_linear_simples,
    "regressao_multipla": analise_regressao_linear_multipla,
    "regressao_logistica_binaria": analise_regressao_logistica_binaria,
    "analise_descritiva": analise_descritiva
}

# 🔗 Dicionário de gráficos disponíveis
GRAFICOS = {
    "scatter": grafico_dispersao,
    "boxplot_simples": grafico_boxplot_simples,
    "grafico_pareto": grafico_pareto,
    "boxplot_multiplo": grafico_boxplot_multiplo,
    "histograma_simples": grafico_histograma_simples,
    "histograma_multiplo": grafico_histograma_multiplo
}

@app.post("/analise")
async def processar_analise(
    request: Request,
    arquivo: UploadFile = File(None),
    ferramenta: str = Form(None),
    grafico: str = Form(None),
    coluna_y: str = Form(None),
    colunas_x: str | list[str] = Form(None)
):
    try:
        # 🔍 LOG DE DEBUG: visualizar campos recebidos no Railway
        form = await request.form()
        print("🔎 FORM RECEBIDO:")
        for chave, valor in form.items():
            print(f"{chave}: {valor}")

        # Função auxiliar para tratar colunas A-Z
        def interpretar_coluna(df, valor):
            valor = valor.strip()
            if len(valor) == 1 and valor.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                idx = ord(valor.upper()) - ord("A")
                if idx < len(df.columns):
                    return df.columns[idx]
                else:
                    raise ValueError(f"Coluna na posição '{valor}' não existe no arquivo. Arquivo tem apenas {len(df.columns)} colunas.")
            return valor

        # Lê o DataFrame
        df = await ler_arquivo(arquivo)

        # Trata coluna Y
        coluna_y_final = interpretar_coluna(df, coluna_y) if coluna_y else None

        # Trata colunas X
        if isinstance(colunas_x, str):
            colunas_x_list = [interpretar_coluna(df, col) for col in colunas_x.split(",") if col.strip()]
        elif isinstance(colunas_x, list):
            colunas_x_list = [interpretar_coluna(df, col) for col in colunas_x if col.strip()]
        else:
            colunas_x_list = []

        analise_texto = None
        grafico_base64 = None

        if ferramenta:
            analise_texto = realizar_analise_estatistica(df, ferramenta, "", coluna_y_final, colunas_x_list)

        if grafico:
            grafico_base64 = gerar_grafico(df, grafico, colunas_x_list, coluna_y_final)

        return {
            "analise": analise_texto,
            "grafico_isolado_base64": grafico_base64
        }

    except ValueError as e:
        return JSONResponse(content={"erro": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"erro": "Erro interno ao processar a análise.", "detalhe": str(e)}, status_code=500)
