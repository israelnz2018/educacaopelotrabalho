# 🔢 Bibliotecas de análise de dados
import pandas as pd
import numpy as np

# 📊 Visualização de dados
import matplotlib.pyplot as plt
import seaborn as sns

# 📦 Modelos estatísticos
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.miscmodels.ordinal_model import OrderedModel

# 🧪 Testes estatísticos
from scipy import stats
from scipy.stats import shapiro, anderson, kstest, norm

# 📈 Métricas de modelos
from sklearn.metrics import roc_curve, auc

# 💾 Manipulação de arquivos/imagens
import base64
from io import BytesIO
import os

# 🧠 Funções locais do projeto
from suporte import interpretar_coluna
from estilo import aplicar_estilo_minitab

# ✅ Todas as análises começam abaixo, dentro das funções (nunca aqui fora)



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

def analise_descritiva(df, colunas_usadas):
    coluna_y = colunas_usadas[0]  # ✅ pegando diretamente a coluna
    if coluna_y not in df.columns:
        return (
            "❌ A coluna selecionada para análise descritiva não foi encontrada.",
            None
        )

    serie = df[coluna_y].dropna()

    if serie.empty:
        return (
            "❌ A coluna selecionada não contém dados numéricos válidos.",
            None
        )

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

    return resumo, imagem_base64

def teste_normalidade(df, colunas_usadas):
    if not colunas_usadas:
        return "❌ Nenhuma coluna foi selecionada.", None

    coluna = colunas_usadas[0]
    serie = df[coluna].dropna()

    if serie.empty:
        return "❌ A coluna selecionada não contém dados válidos.", None

    resultados = []
    dicas = []

    # Shapiro-Wilk
    stat_sw, p_sw = shapiro(serie)
    conclusao_sw = "✅ Dados normais (p > 0.05)" if p_sw > 0.05 else "❌ Dados não normais (p ≤ 0.05)"
    resultados.append(f"🔹 Shapiro-Wilk: Estatística = {stat_sw:.4f}, p = {p_sw:.4f} → {conclusao_sw}")

    # Anderson-Darling
    ad = anderson(serie)
    lim_ad = ad.critical_values[2]  # nível de significância de 5%
    conclusao_ad = "✅ Dados normais" if ad.statistic < lim_ad else "❌ Dados não normais"
    resultados.append(f"🔹 Anderson-Darling: Estatística = {ad.statistic:.4f}, Limite Crítico (5%) = {lim_ad:.4f} → {conclusao_ad}")

    # Kolmogorov-Smirnov com comparação à normal padrão
    serie_padronizada = (serie - serie.mean()) / serie.std()
    stat_ks, p_ks = kstest(serie_padronizada, 'norm')
    conclusao_ks = "✅ Dados normais (p > 0.05)" if p_ks > 0.05 else "❌ Dados não normais (p ≤ 0.05)"
    resultados.append(f"🔹 Kolmogorov-Smirnov: Estatística = {stat_ks:.4f}, p = {p_ks:.4f} → {conclusao_ks}")

    # Se os três testes forem negativos, mostrar recomendações
    if all("❌" in linha for linha in resultados):
        # Outliers
        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        iqr = q3 - q1
        limites = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        outliers = serie[(serie < limites[0]) | (serie > limites[1])]
        if not outliers.empty:
            dicas.append("🔎 Foram identificados possíveis outliers. Recomendamos investigá-los e, se apropriado, removê-los antes de repetir o teste.")

        # Tamanho da amostra
        if len(serie) <= 30:
            dicas.append("📉 A amostra contém 30 dados ou menos. Sempre que possível, colete pelo menos 50 dados para garantir maior confiabilidade.")

        # Estabilidade do processo
        dicas.append("⚙️ Verifique se o processo estava estável no momento da coleta. Mudanças no ambiente, operador ou equipamento podem afetar a distribuição.")

    texto = f"""📊 **Teste de Normalidade - Coluna '{coluna}'**  
{chr(10).join(resultados)}

{chr(10).join(dicas)}""" if dicas else f"""📊 **Teste de Normalidade - Coluna '{coluna}'**  
{chr(10).join(resultados)}"""

    # 🎯 Gráfico de probabilidade normal (estilo Minitab)
    aplicar_estilo_minitab()

    fig, ax = plt.subplots(figsize=(6, 4))
    res = stats.probplot(serie, dist="norm", plot=ax)

    ax.get_lines()[1].set_color("red")  # linha de tendência em vermelho
    ax.set_title(f"Gráfico de Probabilidade de {coluna}", fontsize=14)
    ax.set_xlabel(coluna, fontsize=12)
    ax.set_ylabel("Percentual", fontsize=12)

    from matplotlib.ticker import FuncFormatter
    def formatar_percentual(x, _): return f"{100 * x:.0f}%"
    ax.yaxis.set_major_formatter(FuncFormatter(formatar_percentual))
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    imagem_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return texto, imagem_base64

from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from estilo import aplicar_estilo_minitab

def analise_regressao_logistica_binaria(df, colunas_usadas):
    if len(colunas_usadas) < 2:
        return "❌ É necessário selecionar uma coluna Y (resposta binária) e pelo menos uma coluna X (numérica).", None

    nome_coluna_y = colunas_usadas[0]
    nomes_colunas_x = colunas_usadas[1:]

    y_raw = df[nome_coluna_y].dropna()
    X_raw = df[nomes_colunas_x]

    df_modelo = pd.concat([y_raw, X_raw], axis=1).dropna()
    y = df_modelo[nome_coluna_y]
    X = df_modelo[nomes_colunas_x]

    # Conversão para binário (0 e 1) caso a variável Y seja categórica
    if y.dtype == object or str(y.dtype).startswith('category'):
        y = pd.factorize(y)[0]

    X = sm.add_constant(X)
    modelo = sm.Logit(y, X)
    resultado = modelo.fit(disp=0)

    pseudo_r2 = resultado.prsquared
    resumo = resultado.summary2().as_text()

    interpretacao = f"""📊 **Regressão Logística Binária**  
🔹 Variável de resposta (Y): {nome_coluna_y}  
🔹 Variáveis preditoras (X): {", ".join(nomes_colunas_x)}  
🔹 Pseudo R²: {pseudo_r2:.4f}  

📌 Este modelo estima a probabilidade de um resultado binário com base nas variáveis preditoras.  
- Coeficientes positivos indicam aumento na chance de ocorrência do evento à medida que a variável aumenta.  
- P-valores menores que 0.05 indicam significância estatística.  
- O Pseudo R² mede o quanto o modelo se ajusta aos dados (quanto mais próximo de 1, melhor)."""

    imagem_base64 = None

    # Gráfico de linha ajustada (apenas se houver uma variável preditora)
        # Gráfico de linha ajustada (apenas se houver uma variável preditora)
    if len(nomes_colunas_x) == 1:
        nome_x = nomes_colunas_x[0]
        x_plot = df_modelo[nome_x]
        y_plot = y

        x_ord = np.linspace(x_plot.min(), x_plot.max(), 100)
        X_pred = sm.add_constant(pd.DataFrame({nome_x: x_ord}))
        y_pred = resultado.predict(X_pred)

        aplicar_estilo_minitab()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(x_plot, y_plot, alpha=0.7, color="black", label="Dados")
        ax.plot(x_ord, y_pred, color="red", linewidth=2, label="Curva Ajustada")
        ax.set_xlabel(nome_x)
        ax.set_ylabel(f"Probabilidade de {nome_coluna_y}")
        ax.set_title("Gráfico de Linha Ajustada - Regressão Logística")
        ax.legend()
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        plt.close(fig)
        buffer.seek(0)
        imagem_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return interpretacao + "\n\n```\n" + resumo + "\n```", imagem_base64

def analise_regressao_logistica_nominal(df, colunas_usadas):
    if len(colunas_usadas) < 2:
        return "❌ É necessário selecionar uma coluna Y (nominal com mais de duas categorias) e pelo menos uma coluna X (numérica).", None

    nome_coluna_y = colunas_usadas[0]
    nomes_colunas_x = colunas_usadas[1:]

    y_raw = df[nome_coluna_y].dropna()
    X_raw = df[nomes_colunas_x]

    df_modelo = pd.concat([y_raw, X_raw], axis=1).dropna()
    y = df_modelo[nome_coluna_y].squeeze()
    X = df_modelo[nomes_colunas_x]

    # Converte Y para códigos numéricos se for categórica nominal
    if y.dtype == object or str(y.dtype).startswith("category"):
        y, categorias = pd.factorize(y)

    X = sm.add_constant(X)
    modelo = sm.MNLogit(y, X)
    resultado = modelo.fit(disp=0)

    pseudo_r2 = 1 - resultado.llf / resultado.llnull
    resumo = resultado.summary().as_text()

    interpretacao = f"""📊 **Regressão Logística Nominal**  
🔹 Variável de resposta (Y): {nome_coluna_y} (com múltiplas categorias)  
🔹 Variáveis preditoras (X): {", ".join(nomes_colunas_x)}  
🔹 Pseudo R² (McFadden): {pseudo_r2:.4f}  

📌 Este modelo estima a probabilidade de ocorrência de cada categoria de Y em função das variáveis X.  
- Coeficientes positivos indicam maior chance de uma categoria específica ocorrer.  
- P-valores < 0.05 indicam variáveis significativas.  
- O Pseudo R² mede a qualidade do ajuste do modelo."""

    imagem_base64 = None

# Gráfico de barras com a distribuição das categorias da variável resposta
try:
    aplicar_estilo_minitab()
    fig, ax = plt.subplots(figsize=(6, 4))
    df_modelo[nome_coluna_y].value_counts().sort_index().plot(kind="bar", ax=ax, color="skyblue")
    ax.set_title("Distribuição da variável resposta")
    ax.set_xlabel(nome_coluna_y)
    ax.set_ylabel("Frequência")
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    imagem_base64 = base64.b64encode(buffer.read()).decode("utf-8")
except Exception as e:
    print("Erro ao gerar gráfico:", str(e))
    imagem_base64 = None

    return interpretacao + "\n\n```\n" + resumo + "\n```", imagem_base64

def analise_regressao_logistica_ordinal(df, colunas_usadas):
    if len(colunas_usadas) < 2:
        return "❌ É necessário selecionar uma coluna Y (ordinal) e pelo menos uma coluna X.", None

    nome_coluna_y = colunas_usadas[0]
    nomes_colunas_x = colunas_usadas[1:]

    y_raw = df[nome_coluna_y].dropna()
    X_raw = df[nomes_colunas_x]

    df_modelo = pd.concat([y_raw, X_raw], axis=1).dropna()
    y = df_modelo[nome_coluna_y].squeeze()
    X_temp = df_modelo[nomes_colunas_x].apply(pd.to_numeric, errors="coerce")
    df_modelo = pd.concat([y, X_temp], axis=1).dropna()
    y = df_modelo[nome_coluna_y].squeeze()
    X = df_modelo[nomes_colunas_x]

    from statsmodels.miscmodels.ordinal_model import OrderedModel

    if not pd.api.types.is_categorical_dtype(y) or not y.cat.ordered:
        categorias_ordenadas = sorted(y.unique())
        y = pd.Categorical(y, categories=categorias_ordenadas, ordered=True)

    try:
        modelo = OrderedModel(y, X, distr="logit")
        resultado = modelo.fit(method="bfgs", disp=0)

        pseudo_r2 = 1 - resultado.llf / resultado.llnull
        resumo = resultado.summary().as_text()

        interpretacao = f"""📊 **Regressão Logística Ordinal**  
🔹 Variável de resposta (Y): {nome_coluna_y} (categorias com ordem definida)  
🔹 Variáveis preditoras (X): {", ".join(nomes_colunas_x)}  
🔹 Pseudo R² (McFadden): {pseudo_r2:.4f}  

📌 Este modelo estima a probabilidade acumulada de estar em uma determinada categoria ordinal ou inferior.  
- Coeficientes positivos indicam maior chance de estar em categorias mais altas.  
- P-valores < 0.05 indicam variáveis preditoras estatisticamente significativas."""

        # Geração do gráfico
        imagem_base64 = None
        try:
            aplicar_estilo_minitab()
            fig, ax = plt.subplots(figsize=(6, 4))
            df_modelo[nome_coluna_y].value_counts().sort_index().plot(kind="bar", ax=ax, color="skyblue")
            ax.set_title("Distribuição da variável resposta")
            ax.set_xlabel(nome_coluna_y)
            ax.set_ylabel("Frequência")
            plt.tight_layout()

            buffer = BytesIO()
            plt.savefig(buffer, format="png")
            plt.close(fig)
            buffer.seek(0)
            imagem_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        except Exception:
            imagem_base64 = None

        return interpretacao + "\n\n```\n" + resumo + "\n```", imagem_base64

    except Exception as e:
        return f"Erro ao ajustar modelo: {str(e)}", None


# Dicionário de análises estatísticas
ANALISES = {
    "regressao_simples": analise_regressao_linear_simples,
    "regressao_multipla": analise_regressao_linear_multipla,
    "analise_descritiva": analise_descritiva,
    "teste_normalidade": teste_normalidade,
    "regressao_logistica_binaria": analise_regressao_logistica_binaria,
    "regressao_logistica_nominal": analise_regressao_logistica_nominal,
    "regressao_logistica_ordinal": analise_regressao_logistica_ordinal,
}

