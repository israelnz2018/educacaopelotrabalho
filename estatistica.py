import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
import io
import base64



from estilo import aplicar_estilo_minitab

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

def analise_regressao_linear_multipla(df, colunas):
    X = df[colunas[:-1]]
    Y = df[colunas[-1]]
    X_const = sm.add_constant(X)
    modelo = sm.OLS(Y, X_const).fit()
    r2 = modelo.rsquared_adj
    p_valor = modelo.f_pvalue
    resumo = f"R² ajustado = {r2:.3f}, Valor-p global do modelo = {p_valor:.3f}"
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

def analise_descritiva(df, colunas):
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

    z = 1.96
    erro_media = z * (desvio / np.sqrt(n))
    erro_mediana = 1.57 * (desvio / np.sqrt(n))  # aproximação
    erro_variancia = z * (np.std([np.var(y, ddof=1) for _ in range(1000)], ddof=1))  # bootstrap rudimentar

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
""".strip()

    fig, ax = plt.subplots(figsize=(8, 2))
    aplicar_estilo_minitab()
    box = ax.boxplot(y, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
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

ANALISES = {
    "regressao_simples": analise_regressao_linear_simples,
    "regressao_multipla": analise_regressao_linear_multipla,
    "regressao_logistica_binaria": analise_regressao_logistica_binaria,
    "analise_descritiva": analise_descritiva
}
