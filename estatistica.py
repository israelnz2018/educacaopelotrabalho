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

def analise_regressao_linear_multipla(df, colunas):
    colunas = [interpretar_coluna(df, c) for c in colunas]
    aplicar_estilo_minitab()

    y_col = colunas[-1]
    x_cols = colunas[:-1]

    # 🔍 Conversão segura para numérico
    X = df[x_cols].apply(pd.to_numeric, errors='coerce')
    Y = pd.to_numeric(df[y_col], errors='coerce')

    # 🧹 Remove inf, -inf e NaN antes de montar o modelo
    dados = pd.concat([X, Y], axis=1)
    dados.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
    dados.dropna(inplace=True)

    X = dados[x_cols]
    Y = dados[y_col]
    X = sm.add_constant(X)

    modelo = sm.OLS(Y, X).fit()

    # 🧮 Equação do modelo
    eq_terms = [f"{coef:.2f}*{var}" for var, coef in modelo.params.items() if var != 'const']
    equacao = f"{modelo.params['const']:.2f} + " + " + ".join(eq_terms)

    r2 = modelo.rsquared
    r2_adj = modelo.rsquared_adj
    erro_padrao = modelo.mse_resid**0.5
    p_valor_modelo = modelo.f_pvalue

    # 🔁 VIF
    vif_data = pd.DataFrame()
    vif_data["Variável"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    # 📉 Resíduos
    residuos = modelo.resid
    residuos_padronizados = (residuos - residuos.mean()) / residuos.std()
    outliers = sum(abs(residuos_padronizados) > 3)

    # 📏 Teste de normalidade
    stat, p_shapiro = shapiro(residuos)

    # 🧪 Durbin-Watson
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
- Teste de Shapiro-Wilk (normalidade): p = {p_shapiro:.4f} {'✅' if p_shapiro > 0.05 else '❌'}
- Estatística de Durbin-Watson: {dw:.2f}
- Outliers (resíduos padronizados > 3): {outliers}
"""

    imagens = []

    # 📊 Histograma dos resíduos
    plt.figure(figsize=(6, 4))
    aplicar_estilo_minitab()
    sns.histplot(residuos, kde=True, color="steelblue", edgecolor="black")
    plt.title("Histograma dos Resíduos")
    imagens.append(salvar_grafico())

    # 📈 QQ-Plot
    sm.qqplot(residuos, line='45', fit=True)
    plt.title("QQ-Plot dos Resíduos")
    imagens.append(salvar_grafico())

    # 📉 Resíduos vs Ajustados
    plt.figure(figsize=(6, 4))
    aplicar_estilo_minitab()
    plt.scatter(modelo.fittedvalues, residuos, edgecolor="black", color="darkorange", alpha=0.6)
    plt.axhline(0, linestyle="--", color="gray")
    plt.xlabel("Valores Ajustados")
    plt.ylabel("Resíduos")
    plt.title("Resíduos vs Valores Ajustados")
    imagens.append(salvar_grafico())

    return texto.strip(), imagens


ANALISES = {
    "regressao_simples": analise_regressao_linear_simples,
    "regressao_multipla": analise_regressao_linear_multipla
}
