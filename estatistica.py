import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
import io
import base64
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro
from sklearn.metrics import roc_curve, auc

from estilo import aplicar_estilo_minitab
from suporte import interpretar_coluna

def salvar_grafico():
    caminho = "grafico.png"
    plt.tight_layout()
    plt.savefig(caminho)
    plt.close()
    with open(caminho, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")
    os.remove(caminho)
    return img_base64

def analise_regressao_linear_simples(df, colunas):
    colunas = [interpretar_coluna(df, c) for c in colunas]
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

    a = modelo.params.iloc[0]
    b = modelo.params.iloc[1]
    p_valor = modelo.pvalues.iloc[1]
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

from scipy.stats import anderson
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from suporte import interpretar_coluna
from estilo import aplicar_estilo_minitab

def analise_regressao_linear_multipla(df, colunas):
    colunas = [interpretar_coluna(df, c) for c in colunas]
    aplicar_estilo_minitab()

    y_col = colunas[-1]
    x_cols = colunas[:-1]

    X = df[x_cols].apply(pd.to_numeric, errors='coerce')
    Y = pd.to_numeric(df[y_col], errors='coerce')

    # Remover linhas com NaN
    dados = pd.concat([X, Y], axis=1).dropna()
    X = dados[x_cols]
    Y = dados[y_col]

    if len(dados) < 3:
        raise ValueError("Não há dados suficientes após remoção de NaNs para análise.")

    X_const = sm.add_constant(X)
    modelo = sm.OLS(Y, X_const).fit()

    # Equação
    eq_terms = [f"{coef:.2f}*{var}" for var, coef in modelo.params.items() if var != 'const']
    equacao = f"{modelo.params['const']:.2f} + " + " + ".join(eq_terms)

    r2 = modelo.rsquared
    r2_adj = modelo.rsquared_adj
    erro_padrao = modelo.mse_resid**0.5
    p_valor_modelo = modelo.f_pvalue

    # VIF
    vif_data = pd.DataFrame()
    vif_data["Variável"] = X_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]

    # Resíduos
    residuos = modelo.resid
    residuos_padronizados = (residuos - residuos.mean()) / residuos.std()
    outliers = sum(abs(residuos_padronizados) > 3)

    # Anderson-Darling
    stat_ad, crit_vals, sig_levels = stats.anderson(residuos, dist='norm')
    limiar_5 = crit_vals[sig_levels.tolist().index(5.0)]
    passou_normalidade = stat_ad < limiar_5

    # Durbin-Watson
    dw = durbin_watson(residuos)

    texto = f"""📊 Regressão Linear Múltipla

🔹 Equação:
Y = {equacao}

🔹 Qualidade do modelo:
- R² = {r2:.3f}
- R² ajustado = {r2_adj:.3f}
- Erro padrão da estimativa = {erro_padrao:.3f}
- Valor-p do modelo = {p_valor_modelo:.4f}

🔹 VIF (fator de inflação da variância):\n""" + \
        "\n".join([f"  - {row['Variável']}: {row['VIF']:.2f}" for _, row in vif_data.iterrows() if row['Variável'] != 'const']) + f"""

🔹 Resíduos:
- Teste de Anderson-Darling (normalidade, 5%): {'✅' if passou_normalidade else '❌'} (estatística = {stat_ad:.4f}, limite crítico = {limiar_5:.4f})
- Estatística de Durbin-Watson: {dw:.2f}
- Outliers (resíduos padronizados > 3): {outliers}
"""

    # Apenas 1 gráfico, conforme combinado: Histograma dos Resíduos
    plt.figure(figsize=(6, 4))
    aplicar_estilo_minitab()
    sns.histplot(residuos, kde=True, color="steelblue", edgecolor="black")
    plt.title("Histograma dos Resíduos")
    imagem = salvar_grafico()

    return texto.strip(), imagem

def analise_descritiva(df, coluna_y, colunas_x=None):
    if coluna_y not in df.columns:
        return {
            "analise": "❌ A coluna selecionada para análise descritiva não foi encontrada.",
            "graficos": [],
            "colunas_utilizadas": []
        }

    serie = df[coluna_y].dropna()

    if serie.empty:
        return {
            "analise": "❌ A coluna selecionada não contém dados numéricos válidos.",
            "graficos": [],
            "colunas_utilizadas": []
        }

    media = serie.mean()
    mediana = serie.median()
    desvio = serie.std()
    variancia = serie.var()
    minimo = serie.min()
    maximo = serie.max()
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    assimetria = serie.skew()
    curtose = serie.kurtosis()
    n = serie.count()

    resumo = f"""📊 **Análise Descritiva da coluna '{coluna_y}'**  
- Média: {media:.2f}  
- Mediana: {mediana:.2f}  
- Desvio Padrão: {desvio:.2f}  
- Variância: {variancia:.2f}  
- Mínimo: {minimo:.2f}  
- 1º Quartil (Q1): {q1:.2f}  
- 3º Quartil (Q3): {q3:.2f}  
- Máximo: {maximo:.2f}  
- Assimetria: {assimetria:.2f}  
- Curtose: {curtose:.2f}  
- N: {n}"""

    aplicar_estilo_minitab()
    fig, ax = plt.subplots(figsize=(8, 1.5))
    sns.boxplot(x=serie, orient='h', ax=ax)
    ax.set_title(f"Boxplot - {coluna_y}")
    ax.set_xlabel(coluna_y)

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    plt.close(fig)
    buffer.seek(0)
    imagem_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    return {
        "analise": resumo,
        "graficos": [imagem_base64],
        "colunas_utilizadas": [coluna_y]
    }


ANALISES = {
    "regressao_simples": analise_regressao_linear_simples,
    "regressao_multipla": analise_regressao_linear_multipla,
    "analise_descritiva": analise_descritiva
}
