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
from scipy.stats import chi2_contingency
from scipy.stats import anderson, shapiro, kstest, norm
from sklearn.preprocessing import PowerTransformer
from statsmodels.stats.diagnostic import normal_ad



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

def analise_capabilidade_normal(df, colunas_usadas):
    from scipy.stats import norm, shapiro, anderson, kstest
    from io import BytesIO
    import base64

    nome_coluna_y = colunas_usadas[0]
    nome_coluna_x = colunas_usadas[1]

    dados = df[nome_coluna_y].dropna().astype(float)
    limites = df[nome_coluna_x].dropna().astype(float).unique()

    if len(limites) < 1 or len(limites) > 2:
        raise ValueError("A coluna de limites deve conter um ou dois valores numéricos.")

    # Detecta LSL e USL
    LSL, USL = None, None
    if len(limites) == 1:
        # Detecta com base na posição na planilha
        if not pd.isna(df[nome_coluna_x].iloc[1]):
            LSL = limites[0]
        else:
            USL = limites[0]
    else:
        LSL, USL = sorted(limites[:2])

    media = np.mean(dados)
    desvio_padrao = np.std(dados, ddof=1)
    desvio_padrao_pop = np.std(dados, ddof=0)

    # 🧪 Testes de normalidade
    sw_stat, sw_p = shapiro(dados)
    ad_result = anderson(dados)
    ad_stat = ad_result.statistic
    ad_critico = ad_result.critical_values[2]  # 5%
    ks_stat, ks_p = kstest(dados, 'norm', args=(media, desvio_padrao))

    normal = False
    if sw_p > 0.05 or ad_stat < ad_critico or ks_p > 0.05:
        normal = True

    if not normal:
        return {
            "analise": f"""❌ Os dados **não são normais** segundo os testes de normalidade.
- Shapiro-Wilk: p = {sw_p:.4f}
- Anderson-Darling: estatística = {ad_stat:.4f} | critério = {ad_critico:.4f}
- Kolmogorov-Smirnov: p = {ks_p:.4f}

Recomenda-se utilizar a **Análise de Capabilidade para Dados Não Normais**.""",
            "graficos": [],
            "colunas_utilizadas": colunas_usadas
        }

    # Cp e Cpk (quando possível)
    cp = ((USL - LSL) / (6 * desvio_padrao)) if (USL and LSL) else None
    cpu = ((USL - media) / (3 * desvio_padrao)) if USL else None
    cpl = ((media - LSL) / (3 * desvio_padrao)) if LSL else None
    cpk = min(cpu or float('inf'), cpl or float('inf'))

    # Pp e Ppk (quando possível)
    pp = ((USL - LSL) / (6 * desvio_padrao_pop)) if (USL and LSL) else None
    ppu = ((USL - media) / (3 * desvio_padrao_pop)) if USL else None
    ppl = ((media - LSL) / (3 * desvio_padrao_pop)) if LSL else None
    ppk = min(ppu or float('inf'), ppl or float('inf'))

    # Nível sigma estimado
    sigma_nivel = 3 * cpk

    # 📈 Gráfico de Capabilidade Estilo Minitab
    aplicar_estilo_minitab()
    fig, ax = plt.subplots(figsize=(8, 4))

    counts, bins, patches = ax.hist(dados, bins=15, color="#A6CEE3", edgecolor='black', alpha=0.9, density=True)

    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 500)
    p = norm.pdf(x, media, desvio_padrao)
    ax.plot(x, p, 'darkred', linewidth=2)

    if LSL: ax.axvline(LSL, color='maroon', linestyle='--', linewidth=1.5, label='LSL')
    if USL: ax.axvline(USL, color='maroon', linestyle='--', linewidth=1.5, label='USL')
    ax.axvline(media, color='darkgreen', linestyle='-', linewidth=2, label='Média')
    ax.text(media, max(p) * 1.05, "Alvo", ha='center', va='bottom', fontsize=10, color='darkgreen')

    ax.set_title('Capabilidade do Processo (Normal)', fontsize=14, weight='bold')
    ax.set_xlabel(nome_coluna_y)
    ax.set_ylabel('Densidade')
    ax.set_xlim(xmin, xmax)
    ax.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    imagem_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    # Texto explicativo
    texto = f"""📊 **Análise de Capabilidade (Distribuição Normal)**

- Média: {media:.4f}
- Desvio Padrão: {desvio_padrao:.4f}
"""

    if LSL: texto += f"- LSL: {LSL:.4f}\n"
    if USL: texto += f"- USL: {USL:.4f}\n"
    if cp: texto += f"- Cp: {cp:.4f}\n"
    if cpk: texto += f"- Cpk: {cpk:.4f}\n"
    if pp: texto += f"- Pp: {pp:.4f}\n"
    if ppk: texto += f"- Ppk: {ppk:.4f}\n"

    texto += f"- Nível Sigma estimado: {sigma_nivel:.4f}"

    return {
        "analise": texto,
        "graficos": [imagem_base64],
        "colunas_utilizadas": colunas_usadas
    }


def analise_capabilidade_nao_normal(df, colunas_usadas):
    
    nome_coluna_y = colunas_usadas[0]
    nome_coluna_x = colunas_usadas[1]

    dados = df[nome_coluna_y].dropna().astype(float)
    limites = df[nome_coluna_x].dropna().unique()

    if len(limites) < 1 or len(limites) > 2:
        raise ValueError("A coluna de limites deve conter um ou dois valores numéricos (LSL e/ou USL).")

    lsl = usl = None
    if len(limites) == 2:
        lsl, usl = sorted(limites)
    elif len(limites) == 1:
        if df[nome_coluna_x].iloc[1] != '':
            lsl = limites[0]
        else:
            usl = limites[0]

    # 🔍 Teste de normalidade (sem mostrar para o aluno)
    media = np.mean(dados)
    desvio = np.std(dados, ddof=1)
    normal1 = shapiro(dados)[1] > 0.05
    normal2 = kstest(dados, 'norm', args=(media, desvio))[1] > 0.05
    normal3 = anderson(dados).statistic < 0.6810

    if normal1 or normal2 or normal3:
        texto = "📊 **Análise de Capabilidade**\n\n✅ Os dados parecem seguir uma distribuição normal. Recomenda-se utilizar a análise de capabilidade normal."
        return texto, None

    # 🧪 Tentar ajuste com distribuições alternativas
    distribuicoes = ['lognorm', 'weibull_min', 'gamma', 'expon', 'beta']
    resultados = []

    for dist_name in distribuicoes:
        dist = getattr(stats, dist_name)
        try:
            params = dist.fit(dados)
            stat, p = kstest(dados, dist_name, args=params)
            resultados.append((dist_name, p, params))
        except Exception:
            continue

    resultados.sort(key=lambda x: x[1], reverse=True)

    texto = "📊 **Teste de Aderência às Distribuições**\n\n"
    for nome, pval, _ in resultados:
        texto += f"- {nome}: p = {pval:.4f}\n"

    if len(resultados) == 0:
        texto += "\n❌ Nenhuma distribuição pôde ser ajustada."
        texto += "\n\n🔁 Recomenda-se aplicar transformação matemática (ex: Yeo-Johnson)."
        return texto, None

    melhor_nome, melhor_p, melhor_params = resultados[0]

    if melhor_p > 0.05:
        texto += f"\n✅ **Melhor distribuição encontrada: {melhor_nome} (p = {melhor_p:.4f})**"

        # Geração do gráfico com curva ajustada
        x = np.linspace(min(dados), max(dados), 500)
        dist = getattr(stats, melhor_nome)

        try:
            y = dist.pdf(x, *melhor_params)
        except Exception:
            return texto + "\n\n❌ Erro ao gerar gráfico da distribuição.", None

        # Gráfico com estilo Minitab
        def aplicar_estilo_minitab():
            plt.style.use('default')
            plt.grid(True, linestyle=':', linewidth=0.5)
            plt.rcParams['axes.facecolor'] = 'white'
            plt.rcParams['axes.edgecolor'] = 'black'
            plt.rcParams['axes.grid'] = True
            plt.rcParams['grid.alpha'] = 0.4

        aplicar_estilo_minitab()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(dados, bins=15, density=True, alpha=0.7, color="#A6CEE3", edgecolor='black')
        ax.plot(x, y, 'darkred', linewidth=2)

        media = np.mean(dados)
        if lsl is not None:
            ax.axvline(lsl, color='maroon', linestyle='--')
        if usl is not None:
            ax.axvline(usl, color='maroon', linestyle='--')
        ax.axvline(media, color='darkgreen', linestyle='-', linewidth=2)
        ax.text(media, max(y) * 1.05, "Média", ha='center', fontsize=10, color='darkgreen')

        ax.set_title(f'Capabilidade - {melhor_nome}')
        ax.set_xlabel(nome_coluna_y)
        ax.set_ylabel("Densidade")
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        imagem_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()

        return texto, imagem_base64

    else:
        texto += "\n\n❌ Nenhuma distribuição apresentou p > 0.05."
        texto += "\n\n🔁 Recomenda-se aplicar uma transformação matemática (ex: Yeo-Johnson) para tornar os dados aproximadamente normais e então calcular a capabilidade."
        return texto, None

def aplicar_transformacao_johnson(df, colunas_usadas):
    nome_coluna_y = colunas_usadas[0]
    nome_coluna_x = colunas_usadas[1]

    dados = df[nome_coluna_y].dropna().astype(float)
    limites = df[nome_coluna_x].dropna().unique()

    if len(limites) < 1 or len(limites) > 2:
        raise ValueError("A coluna de limites deve conter um ou dois valores numéricos (LSL e/ou USL).")

    # Definir LSL e USL
    lsl = usl = None
    if len(limites) == 2:
        lsl, usl = sorted(limites)
    elif len(limites) == 1:
        if df[nome_coluna_x].iloc[1] != '':
            lsl = limites[0]
        else:
            usl = limites[0]

    # Aplicar transformação Yeo-Johnson
    pt = PowerTransformer(method='yeo-johnson')
    dados_transformados = pt.fit_transform(dados.values.reshape(-1, 1)).flatten()

    # Teste de normalidade
    stat_ad = anderson(dados_transformados).statistic
    normal = stat_ad < 0.6810

    if normal:
        media = np.mean(dados_transformados)
        desvio = np.std(dados_transformados, ddof=1)

        ppl = ppu = pp = ppk = None
        if lsl is not None:
            ppl = (media - lsl) / (3 * desvio)
        if usl is not None:
            ppu = (usl - media) / (3 * desvio)

        if ppl is not None and ppu is not None:
            pp = (usl - lsl) / (6 * desvio)
            ppk = min(ppl, ppu)
        elif ppl is not None:
            ppk = ppl
        elif ppu is not None:
            ppk = ppu

        sigma_nivel = 3 * ppk if ppk is not None else None

        texto = "🔁 **Transformação Yeo-Johnson aplicada com sucesso**\n\n"
        texto += f"📌 Média transformada: {media:.4f}\n"
        texto += f"📌 Desvio padrão: {desvio:.4f}\n"
        if lsl is not None:
            texto += f"- LSL: {lsl:.4f}\n"
        if usl is not None:
            texto += f"- USL: {usl:.4f}\n"
        if pp is not None:
            texto += f"- Pp: {pp:.4f}\n"
        if ppk is not None:
            texto += f"- Ppk: {ppk:.4f}\n"
        if sigma_nivel is not None:
            texto += f"- Nível Sigma estimado: {sigma_nivel:.4f}"

        aplicar_estilo_minitab()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(dados_transformados, bins=15, density=True, alpha=0.7, color="#A6CEE3", edgecolor='black')

        x = np.linspace(min(dados_transformados), max(dados_transformados), 500)
        y = norm.pdf(x, media, desvio)
        ax.plot(x, y, 'darkred', linewidth=2)

        if lsl is not None:
            ax.axvline(lsl, color='maroon', linestyle='--')
        if usl is not None:
            ax.axvline(usl, color='maroon', linestyle='--')
        ax.axvline(media, color='darkgreen', linestyle='-', linewidth=2)
        ax.text(media, max(y) * 1.05, "Média", ha='center', fontsize=10, color='darkgreen')

        ax.set_title("Capabilidade (dados transformados)")
        ax.set_xlabel(nome_coluna_y + " (transformado)")
        ax.set_ylabel("Densidade")
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        imagem_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()

        return texto, imagem_base64, True

    else:
        # Discretização: % fora dos limites
        total = len(dados)
        abaixo_lsl = len(dados[dados < lsl]) if lsl is not None else 0
        acima_usl = len(dados[dados > usl]) if usl is not None else 0
        fora = abaixo_lsl + acima_usl
        percentual_fora = fora / total

        # Estimativa sigma via tabela Z inversa (1-tailed)
        try:
            sigma_estimado = norm.ppf(1 - percentual_fora / 2)
        except:
            sigma_estimado = None

        texto = "🔁 **Transformação Yeo-Johnson falhou em normalizar os dados**\n"
        texto += f"\n❌ Dados continuam não normais após transformação."
        texto += f"\n📊 Estimando capabilidade com base em dados discretos:"
        texto += f"\n- Total de amostras: {total}"
        if lsl is not None:
            texto += f"\n- Abaixo do LSL: {abaixo_lsl}"
        if usl is not None:
            texto += f"\n- Acima do USL: {acima_usl}"
        texto += f"\n- % Fora dos limites: {percentual_fora:.2%}"
        if sigma_estimado:
            texto += f"\n- Nível Sigma estimado (aproximado): {sigma_estimado:.2f}"

        aplicar_estilo_minitab()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(dados, bins=15, color="#A6CEE3", edgecolor='black', alpha=0.8)

        if lsl is not None:
            ax.axvline(lsl, color='maroon', linestyle='--', label='LSL')
        if usl is not None:
            ax.axvline(usl, color='maroon', linestyle='--', label='USL')
        ax.axvline(np.mean(dados), color='darkgreen', linestyle='-', linewidth=2, label='Média')
        ax.set_title("Capabilidade (dados discretos)")
        ax.set_xlabel(nome_coluna_y)
        ax.set_ylabel("Frequência")
        ax.legend()
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        imagem_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()

        return texto, imagem_base64, False


def analise_chi_quadrado(df, colunas_usadas):
    if len(colunas_usadas) < 2:
        raise ValueError("O teste qui-quadrado requer pelo menos duas colunas: uma Y e uma X.")

    col_y = colunas_usadas[0]
    col_x = colunas_usadas[1]
    col_freq = colunas_usadas[2] if len(colunas_usadas) >= 3 else None

    # Se o aluno forneceu a coluna de frequência explicitamente
    if col_freq and col_freq in df.columns:
        tabela = df.pivot_table(
            index=col_x,
            columns=col_y,
            values=col_freq,
            aggfunc="sum",
            fill_value=0
        )
    else:
        # Dados linha a linha
        tabela = pd.crosstab(df[col_x], df[col_y])

    # Aplica o teste
    chi2, p, dof, expected = chi2_contingency(tabela)

    # Monta a interpretação
    resumo = f"""🔎 **Teste do Qui-Quadrado de Independência**

Tabela de Contingência:
{tabela.to_string()}

Estatística Qui-Quadrado: {chi2:.4f}
Graus de Liberdade: {dof}
Valor-p: {p:.4f}

"""

    if p < 0.05:
        conclusao = "❗Existe associação estatística significativa entre as variáveis (p < 0.05)."
    else:
        conclusao = "✅ Não há evidência estatística de associação entre as variáveis (p ≥ 0.05)."

    # Gráfico de barras agrupadas
    aplicar_estilo_minitab()
    tabela.plot(kind='bar')
    plt.title("Distribuição das Categorias")
    plt.xlabel(col_x)
    plt.ylabel("Frequência")

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)
    imagem_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return resumo + conclusao, imagem_base64

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

    try:
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

    except Exception as e:
        return f"Erro ao ajustar modelo: {str(e)}", None

def analise_regressao_logistica_ordinal(df, colunas_usadas):
    if len(colunas_usadas) < 2:
        return "❌ É necessário selecionar uma coluna Y (ordinal) e pelo menos uma coluna X.", None

    nome_coluna_y = colunas_usadas[0]
    nomes_colunas_x = colunas_usadas[1:]

    # Prepara Y e converte para ordinal, se necessário
    y_raw = df[nome_coluna_y].dropna()
    if not pd.api.types.is_categorical_dtype(y_raw) or not y_raw.cat.ordered:
        categorias_ordenadas = sorted(y_raw.dropna().unique())
        y = pd.Categorical(y_raw, categories=categorias_ordenadas, ordered=True)
    else:
        y = y_raw

    # Converte X para numérico
    X_raw = df[nomes_colunas_x].copy()
    for col in nomes_colunas_x:
        X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")

    # Junta e remove linhas com dados ausentes
    df_modelo = pd.concat([pd.Series(y, name=nome_coluna_y), X_raw], axis=1).dropna()
    y = df_modelo[nome_coluna_y]
    X = df_modelo[nomes_colunas_x]

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
        except Exception as e:
            print("Erro ao gerar gráfico:", str(e))
            imagem_base64 = None

        return interpretacao + "\n\n```\n" + resumo + "\n```", imagem_base64

    except Exception as e:
        return f"Erro ao ajustar modelo: {str(e)}", None

def teste_2sample_t(df, colunas_usadas):
    if len(colunas_usadas) != 2:
        return "❌ É necessário selecionar exatamente duas colunas numéricas para o Teste 2 Sample T.", None

    col1, col2 = colunas_usadas
    serie1 = pd.to_numeric(df[col1], errors="coerce").dropna()
    serie2 = pd.to_numeric(df[col2], errors="coerce").dropna()

    if len(serie1) < 2 or len(serie2) < 2:
        return "❌ As colunas selecionadas não possuem dados suficientes para o teste.", None

    # Teste de normalidade para cada grupo (Anderson-Darling)
    ad1 = anderson(serie1)
    ad2 = anderson(serie2)
    lim1 = ad1.critical_values[2]
    lim2 = ad2.critical_values[2]
    normal1 = ad1.statistic < lim1
    normal2 = ad2.statistic < lim2

    # Teste F para igualdade de variâncias
    stat_f = np.var(serie1, ddof=1) / np.var(serie2, ddof=1)
    df1, df2 = len(serie1)-1, len(serie2)-1
    p_f = 2 * min(stats.f.cdf(stat_f, df1, df2), 1 - stats.f.cdf(stat_f, df1, df2))
    equal_var = p_f > 0.05

    # Teste t
    t_stat, p_valor = stats.ttest_ind(serie1, serie2, equal_var=equal_var)

    texto = f"""📊 **Teste T para 2 Amostras Independentes**

🔹 Coluna 1: {col1}  
🔹 Coluna 2: {col2}  

🔹 Teste de normalidade (Anderson-Darling, 5%):  
- {col1}: {"✅ Normal" if normal1 else "❌ Não normal"} (estatística = {ad1.statistic:.4f}, limite crítico = {lim1:.4f})  
- {col2}: {"✅ Normal" if normal2 else "❌ Não normal"} (estatística = {ad2.statistic:.4f}, limite crítico = {lim2:.4f})  

🔹 Teste F para igualdade de variâncias:  
- Estatística F = {stat_f:.4f}, p = {p_f:.4f} → {"✅ Variâncias iguais" if equal_var else "❌ Variâncias diferentes"}

🔹 Resultado do Teste T:  
- Estatística t = {t_stat:.4f}, p = {p_valor:.4f}  
- {"✅ Não há diferença significativa" if p_valor > 0.05 else "❌ Diferença estatisticamente significativa entre as médias"}"""

    # Gráfico estilo Minitab
    try:
        aplicar_estilo_minitab()
        fig, ax = plt.subplots(figsize=(6, 4))

        dados_plot = pd.DataFrame({
            'Valor': pd.concat([serie1, serie2]),
            'Grupo': [col1] * len(serie1) + [col2] * len(serie2)
        })

        sns.boxplot(x="Grupo", y="Valor", data=dados_plot, ax=ax, width=0.6, palette="pastel")
        medias = dados_plot.groupby("Grupo")["Valor"].mean()
        ax.plot(range(len(medias)), medias, marker="o", linestyle="-", color="black", linewidth=2, label="Média")
        ax.set_title(f"Boxplot de {col1} e {col2}")
        ax.set_ylabel("Valores")
        ax.legend()
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        plt.close(fig)
        buffer.seek(0)
        imagem_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    except:
        imagem_base64 = None

    return texto, imagem_base64

def analise_teste_paired_t(df, colunas_usadas):
    if len(colunas_usadas) != 2:
        return "❌ O teste pareado requer exatamente duas colunas.", None

    col1, col2 = colunas_usadas
    serie1 = pd.to_numeric(df[col1], errors="coerce")
    serie2 = pd.to_numeric(df[col2], errors="coerce")

    diferencas = (serie1 - serie2).dropna()
    if len(diferencas) < 2:
        return "❌ Dados insuficientes para o teste pareado.", None

    stat, p_valor = stats.ttest_rel(serie1, serie2, nan_policy='omit')
    media = diferencas.mean()
    desvio = diferencas.std(ddof=1)
    n = len(diferencas)

    t_crit = stats.t.ppf(1 - 0.025, df=n - 1)
    erro = desvio / np.sqrt(n)
    ic = (media - t_crit * erro, media + t_crit * erro)

    interpretacao = f"""📊 **Teste T Pareado**  
🔹 Comparação entre: {col1} e {col2}  
🔹 Número de pares: {n}  
🔹 Média das diferenças: {media:.4f}  
🔹 Desvio padrão das diferenças: {desvio:.4f}  
🔹 Intervalo de confiança (95%): ({ic[0]:.4f}, {ic[1]:.4f})  
🔹 Valor-p: {p_valor:.4f}  

📌 **Conclusão**: {"❌ As médias são estatisticamente diferentes." if p_valor < 0.05 else "✅ Não há diferença estatística entre as médias."}"""

    aplicar_estilo_minitab()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(diferencas, vert=False, patch_artist=True, boxprops=dict(facecolor='skyblue'))
    ax.set_title("Boxplot das Diferenças")
    ax.set_xlabel("Diferenças")
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax.plot(media, 1, marker="x", color="red", markersize=10, label="Média")
    ax.hlines(1, ic[0], ic[1], color="black", linewidth=2, label="IC 95%")
    ax.legend()
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    imagem_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return interpretacao, imagem_base64

def teste_variancias(df, colunas_usadas):
    if len(colunas_usadas) != 2:
        return "❌ Selecione exatamente duas colunas para comparar as variâncias.", None

    col1, col2 = colunas_usadas
    serie1 = pd.to_numeric(df[col1], errors="coerce").dropna()
    serie2 = pd.to_numeric(df[col2], errors="coerce").dropna()

    if len(serie1) < 3 or len(serie2) < 3:
        return "❌ É necessário pelo menos 3 dados em cada grupo para realizar o teste de variâncias.", None

    # 🧪 Teste de normalidade (Anderson-Darling)
    p_norm1 = anderson(serie1).critical_values[2]
    p_norm2 = anderson(serie2).critical_values[2]
    normal1 = stats.normaltest(serie1).pvalue > 0.05
    normal2 = stats.normaltest(serie2).pvalue > 0.05

    aviso = ""
    if not (normal1 and normal2):
        aviso = "⚠️ A premissa de normalidade foi violada em pelo menos um dos grupos.\n\n"

    # 🧪 Teste F
    stat_f = np.var(serie1, ddof=1) / np.var(serie2, ddof=1)
    df1, df2 = len(serie1)-1, len(serie2)-1
    p_f = 2 * min(stats.f.cdf(stat_f, df1, df2), 1 - stats.f.cdf(stat_f, df1, df2))

    interpretacao = f"""📊 **Teste de Igualdade de Variâncias (F-Teste)**  
🔹 Grupos comparados: {col1} e {col2}  
🔹 Estatística F: {stat_f:.4f}  
🔹 Valor-p (bilateral): {p_f:.4f}  

{"✅ As variâncias são significativamente diferentes." if p_f < 0.05 else "➖ Não há evidência de diferença entre as variâncias."}
"""

    # 🎨 Gráfico de boxplot com estilo Minitab
    try:
        aplicar_estilo_minitab()
        df_plot = pd.DataFrame({
            "Valor": pd.concat([serie1, serie2]),
            "Grupo": [col1] * len(serie1) + [col2] * len(serie2)
        })
        plt.figure(figsize=(8, 5))
        sns.boxplot(x="Valor", y="Grupo", data=df_plot, orient="h", palette="pastel", showmeans=True,
                    meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black"})
        plt.title("Comparação das Variâncias (Boxplot)")
        imagem_base64 = salvar_grafico()
    except Exception as e:
        print("Erro ao gerar gráfico:", str(e))
        imagem_base64 = None

    return aviso + "```\n" + interpretacao + "\n```", imagem_base64

def teste_anova(df, colunas_usadas):
    if len(colunas_usadas) < 2:
        return "❌ O Teste ANOVA exige no mínimo duas colunas com dados numéricos (grupos).", None

    dados_grupos = [df[coluna].dropna() for coluna in colunas_usadas]
    normalidade = []
    for i, grupo in enumerate(dados_grupos):
        stat, critico, _ = stats.anderson(grupo)
        if stat < critico[2]:  # 5%
            normalidade.append(f"✅ Grupo {colunas_usadas[i]}: distribuição normal (Anderson-Darling)")
        else:
            normalidade.append(f"⚠️ Grupo {colunas_usadas[i]}: não segue distribuição normal")

    # Teste ANOVA
    try:
        f_stat, p_valor = stats.f_oneway(*dados_grupos)
    except Exception as e:
        return f"❌ Erro ao executar o teste ANOVA: {str(e)}", None

    # Interpretação
    interpretacao = f"""📊 **Teste ANOVA (Análise de Variância)**  
🔹 Grupos comparados: {", ".join(colunas_usadas)}  
🔹 Estatística F: {f_stat:.4f}  
🔹 Valor-p: {p_valor:.4f}  

📌 Este teste verifica se há diferença significativa entre as médias dos grupos.  
- Se **valor-p < 0.05**, rejeitamos H₀ e concluímos que **pelo menos um grupo tem média diferente**.
- Se **valor-p ≥ 0.05**, **não há evidências suficientes** para afirmar que as médias diferem.

🔍 **Verificação de normalidade (Anderson-Darling, 5%)**:
""" + "\n".join(normalidade)

    # Gráfico
    try:
        aplicar_estilo_minitab()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(dados_grupos, vert=False, patch_artist=True,
                   labels=colunas_usadas, boxprops=dict(facecolor="skyblue"))
        medias = [grupo.mean() for grupo in dados_grupos]
        for i, media in enumerate(medias, start=1):
            ax.plot(media, i, marker="o", color="red")
        ax.set_title("Boxplot por Grupo (ANOVA)")
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        plt.close(fig)
        buffer.seek(0)
        imagem_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    except Exception as e:
        print("Erro ao gerar o gráfico:", str(e))
        imagem_base64 = None

    return interpretacao, imagem_base64



# Dicionário de análises estatísticas
ANALISES = {
    "regressao_simples": analise_regressao_linear_simples,
    "regressao_multipla": analise_regressao_linear_multipla,
    "analise_descritiva": analise_descritiva,
    "teste_normalidade": teste_normalidade,
    "regressao_logistica_binaria": analise_regressao_logistica_binaria,
    "regressao_logistica_nominal": analise_regressao_logistica_nominal,
    "regressao_logistica_ordinal": analise_regressao_logistica_ordinal,
    "teste_2sample_t": teste_2sample_t,
    "teste_paired_t": analise_teste_paired_t,
    "teste_variancias": teste_variancias,
    "teste_anova": teste_anova,
    "analise_chi_quadrado": analise_chi_quadrado,
    "capabilidade_normal": analise_capabilidade_normal,
    "capabilidade_nao_normal": analise_capabilidade_nao_normal,
    "Transformação Johnson": aplicar_transformacao_johnson,


}

